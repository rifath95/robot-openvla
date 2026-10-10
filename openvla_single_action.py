"""Capture → predict → execute one bounded action → capture again.

Run this file in VS Code. Each run writes its own status, log, and images under
outputs/single_action_*. Close the final simulator window to exit.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
from panda_actions import PHYSICS_STEPS, action_contract


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def adapt_bridge_action(values):
    """Provisional identity world-axis mapping; bound the *whole* delta vector.

    Bridge uses 0=closed, 1=open. PandaController uses -1=closed, +1=open.
    Translation/Euler deltas are interpreted in the simulation world axes for
    this plumbing experiment, not calibrated between WidowX and Panda robots.
    """
    from panda_actions import to_controller_action
    return to_controller_action(values)


def simulation_stage(args):
    import mujoco
    import numpy as np
    from capture_camera import save_rgb_png
    from panda_controller import PandaController
    from panda_env import PandaEnv

    folder = args.run_dir
    with PandaEnv() as env:
        if args.stage == "capture":
            env.controller.step(500)  # Let contacts and the arm settle first.
            save_rgb_png(folder / "before.png", env.observe())
            np.savez(folder / "scene_state.npz", qpos=env.data.qpos,
                     qvel=env.data.qvel, act=env.data.act, ctrl=env.data.ctrl,
                     time=env.data.time)
            return

        # Restore exactly the scene that produced the input image. No physics
        # runs while the separate inference process holds the GPU/model memory.
        with np.load(folder / "scene_state.npz") as state:
            for name in ("qpos", "qvel", "act", "ctrl"):
                getattr(env.data, name)[:] = state[name]
            env.data.time = float(state["time"])
        mujoco.mj_forward(env.model, env.data)
        from panda_actions import make_controller
        env.controller = make_controller(env.model, env.data)
        prediction = json.loads((folder / "prediction.json").read_text())
        if prediction.get("unnorm_key") != "bridge_orig":
            raise ValueError("Only bridge_orig action adaptation is implemented")
        action = adapt_bridge_action(prediction["action"])
        before_position, before_quaternion = env.controller.end_effector_pose()
        ik = env.controller.apply_action(action)
        if not ik.converged:
            raise RuntimeError("IK rejected the action; scene will not be advanced")
        env.controller.step(PHYSICS_STEPS)
        after_position, after_quaternion = env.controller.end_effector_pose()
        save_rgb_png(folder / "after.png", env.observe())
        np.savez(folder / "after_state.npz", qpos=env.data.qpos,
                 qvel=env.data.qvel, act=env.data.act, ctrl=env.data.ctrl,
                 time=env.data.time)
        tracking_error = float(np.linalg.norm(
            env.controller.target_position - after_position))
        if not np.all(np.isfinite(env.data.qpos)):
            raise RuntimeError("Non-finite simulation state after execution")
        report = {
            "raw_action": prediction["action"], "controller_action": action.tolist(),
            "mapping": "Provisional Bridge deltas using panda_grasp_v1; grasp point",
            "action_contract": action_contract(),
            "translation_limit_metres": 0.01, "rotation_limit_radians": 0.05,
            "gripper_mapping": "2 * model_gripper - 1; no sign inversion",
            "before_position": before_position.tolist(),
            "after_position": after_position.tolist(),
            "actual_translation": (after_position - before_position).tolist(),
            "before_quaternion_wxyz": before_quaternion.tolist(),
            "after_quaternion_wxyz": after_quaternion.tolist(),
            "target_position": env.controller.target_position.tolist(),
            "position_tracking_error_metres": tracking_error,
            "ik_converged": bool(ik.converged),
            "physics_seconds": PHYSICS_STEPS * float(env.model.opt.timestep),
            "inference_seconds": prediction.get("inference_seconds"),
        }
        write_json(folder / "execution.json", report)
        if tracking_error > 0.005:
            raise RuntimeError(f"Action executed but tracking error is too large: {tracking_error:.4f} m")
        print(json.dumps(report, indent=2), flush=True)
        write_json(folder / "status.json", {"phase": "Completed", "run_dir": str(folder)})
        if not args.no_view:
            import mujoco.viewer
            print("Showing the scene after one action. Close the window to exit.", flush=True)
            mujoco.viewer.launch(env.model, env.data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default="pick up the red cube")
    parser.add_argument("--no-view", action="store_true", help="Save results without opening a final viewer")
    parser.add_argument("--stage", choices=("capture", "execute"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage:
        simulation_stage(args)
        return

    sim_python = ROOT / ".venv/bin/python"
    model_python = ROOT / ".venv-openvla/bin/python"
    for interpreter in (sim_python, model_python):
        if not interpreter.exists():
            raise RuntimeError(f"Missing environment: {interpreter}. See README.md setup instructions.")
    folder = ROOT / "outputs" / datetime.now().strftime("single_action_%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    print(f"Results and status: {folder}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    commands = (
        ("Capturing scene", [str(sim_python), __file__, "--stage", "capture", "--run-dir", str(folder)]),
        ("Predicting one action", [str(model_python), str(ROOT / "openvla_mac.py"),
         "--image", str(folder / "before.png"), "--instruction", args.instruction,
         "--unnorm-key", "bridge_orig", "--output", str(folder / "prediction.json")]),
        ("Executing one action", [str(sim_python), __file__, "--stage", "execute",
         "--run-dir", str(folder)] + (["--no-view"] if args.no_view else [])),
    )
    with (folder / "run.log").open("w") as log:
        for phase, command in commands:
            write_json(folder / "status.json", {"phase": phase, "run_dir": str(folder)})
            print(phase, flush=True)
            log.write(f"\n--- {phase} ---\n")
            log.flush()
            result = subprocess.run(command, cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(folder / "status.json", {"phase": "Failed", "failed_stage": phase,
                           "returncode": result.returncode, "run_dir": str(folder)})
                raise RuntimeError(f"{phase} failed. See {folder / 'run.log'}")
    print(f"Completed. Before/after images and execution report: {folder}", flush=True)


if __name__ == "__main__":
    main()
