"""Run three camera → OpenVLA → bounded robot movement cycles.

Run in VS Code. Keeps the model loaded; saves all images and timings under
outputs/openvla_loop_*. No viewer is opened. Supports Mac MPS and Linux CUDA.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    # Replace atomically so a reader never sees half-written status JSON.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def update_status(folder, phase, **details):
    old = json.loads((folder / "status.json").read_text()) if (folder / "status.json").exists() else {}
    save_json(folder / "status.json", {
        **old, "phase": phase,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), **details,
    })
    print(phase, details.get("step", ""), flush=True)


def simulation(stage, folder):
    subprocess.run([
        str(ROOT / ".venv/bin/python"), str(ROOT / "openvla_single_action.py"),
        "--stage", stage, "--run-dir", str(folder), "--no-view",
    ], cwd=ROOT, check=True)


def run_loaded_model(folder, steps, predict):
    """Use one loaded-model callback, feeding each new observation to it."""
    start = time.monotonic()
    reports = []
    for index in range(1, steps + 1):
        step_dir = folder / f"step_{index:02d}"
        if index > 1:
            previous = folder / f"step_{index-1:02d}"
            step_dir.mkdir()
            # Both files describe the same resulting scene. Do not reset home.
            shutil.copyfile(previous / "after.png", step_dir / "before.png")
            shutil.copyfile(previous / "after_state.npz", step_dir / "scene_state.npz")
        update_status(folder, "Predicting one action", step=index, completed_steps=index-1)
        prediction = predict(step_dir / "before.png")
        save_json(step_dir / "prediction.json", prediction)
        update_status(folder, "Executing one action", step=index,
                      last_inference_seconds=prediction["inference_seconds"])
        simulation("execute", step_dir)
        report = json.loads((step_dir / "execution.json").read_text())
        reports.append(report)
        save_json(folder / "summary.json", {
            "completed_steps": index, "requested_steps": steps,
            "inference_seconds_per_step": [r["inference_seconds"] for r in reports],
            "total_inference_seconds": sum(r["inference_seconds"] for r in reports),
            "loop_seconds_excluding_model_loading": time.monotonic()-start,
            "steps": reports,
        })
        update_status(folder, "Action completed", completed_steps=index)
    update_status(folder, "Completed", completed_steps=steps,
                  total_inference_seconds=sum(r["inference_seconds"] for r in reports),
                  loop_seconds_excluding_model_loading=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--instruction", default="pick up the red cube")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    args = parser.parse_args()
    if not 1 <= args.steps <= 3:
        parser.error("This initial test supports between 1 and 3 actions")
    for name in (".venv", ".venv-openvla"):
        if not (ROOT / name / "bin/python").exists():
            raise RuntimeError(f"Missing {name}; see README.md")
    if not (ROOT / ".cache-openvla/model_revision.json").exists():
        raise RuntimeError("Download the checkpoint first; see README.md")
    folder = ROOT / "outputs" / datetime.now().strftime("openvla_loop_%Y%m%d_%H%M%S_%f")
    (folder / "step_01").mkdir(parents=True)
    started = time.monotonic()
    update_status(folder, "Capturing initial scene", requested_steps=args.steps,
                  completed_steps=0, run_dir=str(folder),
                  device=args.device,
                  started_at_utc=datetime.now(timezone.utc).isoformat())
    print(f"Results: {folder}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if args.device == "cuda":
        # Set before the simulation subprocess imports MuJoCo. No display needed.
        environment.setdefault("MUJOCO_GL", "egl")
    try:
        with (folder / "run.log").open("w") as log:
            update_status(folder, "Checking inference device")
            subprocess.run([
                str(ROOT / ".venv-openvla/bin/python"), str(ROOT / "openvla_mac.py"),
                "--preflight", "--device", args.device,
            ], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            update_status(folder, "Capturing initial scene")
            subprocess.run([
                str(ROOT / ".venv/bin/python"), str(ROOT / "openvla_single_action.py"),
                "--stage", "capture", "--run-dir", str(folder / "step_01"), "--no-view",
            ], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            update_status(folder, "Loading model")
            subprocess.run([
                str(ROOT / ".venv-openvla/bin/python"), str(ROOT / "openvla_mac.py"),
                "--loop-dir", str(folder), "--loop-steps", str(args.steps),
                "--instruction", args.instruction, "--unnorm-key", "bridge_orig",
                "--device", args.device,
            ], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    except (Exception, KeyboardInterrupt) as error:
        state = json.loads((folder / "status.json").read_text())
        update_status(folder, "Failed", failed_stage=state["phase"],
                      error=str(error), total_elapsed_seconds=time.monotonic()-started)
        raise
    update_status(folder, "Completed", total_elapsed_seconds=time.monotonic()-started)
    print(f"Completed {args.steps} actions. See {folder / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
