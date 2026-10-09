"""Mac camera → cloud OpenVLA → one bounded Panda movement → new camera image."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from connection_client import request_prediction
from openvla_single_action import adapt_bridge_action

ROOT = Path(__file__).resolve().parent


def run_single_action(*, server_url="http://127.0.0.1:8000",
                      instruction="pick up the red cube", timeout_seconds=180,
                      output_dir=None, no_view=False):
    import mujoco
    import numpy as np
    from capture_camera import save_rgb_png
    from panda_env import PandaEnv

    request_id = uuid.uuid4().hex
    folder = Path(output_dir) if output_dir else ROOT / "outputs" / f"remote_single_action_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{request_id[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    phase = "Initializing simulation"
    motion_started = False
    def save(name, value):
        (folder / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    def status(new_phase):
        nonlocal phase
        phase = new_phase
        save("status.json", {"phase": phase, "request_id": request_id,
                             "elapsed_seconds": time.perf_counter() - started,
                             "motion_started": motion_started})
        print(phase, flush=True)
    def save_state(name, env):
        np.savez(folder / name, qpos=env.data.qpos, qvel=env.data.qvel,
                 act=env.data.act, ctrl=env.data.ctrl, time=env.data.time)

    print(f"Results and status: {folder}", flush=True)
    status(phase)
    try:
        with PandaEnv() as env:
            env.controller.step(500)
            status("Capturing scene")
            capture_started = time.perf_counter()
            save_rgb_png(folder / "before.png", env.observe())
            save_state("scene_state.npz", env)
            capture_seconds = time.perf_counter() - capture_started
            before_qpos = env.data.qpos.copy()
            simulation_time_before = float(env.data.time)
            before_position, before_quaternion = env.controller.end_effector_pose()

            # Keep this exact live environment. No physics advances during HTTP inference.
            prediction = request_prediction(folder / "before.png", instruction,
                                            server_url=server_url, expected_mode="openvla",
                                            timeout_seconds=timeout_seconds,
                                            request_id=request_id, progress=status)
            save("prediction.json", prediction)
            status("Validating returned action")
            if prediction.get("unnorm_key") != "bridge_orig":
                raise ValueError("Only bridge_orig action adaptation is implemented")
            if not np.array_equal(env.data.qpos, before_qpos) or float(env.data.time) != simulation_time_before:
                raise RuntimeError("Simulation changed while waiting for inference")
            action = adapt_bridge_action(prediction["action"])
            status("Solving bounded action")
            ik_started = time.perf_counter()
            ik = env.controller.apply_action(action)
            ik_seconds = time.perf_counter() - ik_started
            if not ik.converged:
                raise RuntimeError("IK rejected the action; simulation is not advanced")
            status("Executing one action")
            execution_started = time.perf_counter()
            motion_started = True
            env.controller.step(1000)
            movement_seconds = time.perf_counter() - execution_started
            after_position, after_quaternion = env.controller.end_effector_pose()
            status("Capturing resulting scene")
            next_capture_started = time.perf_counter()
            save_rgb_png(folder / "after.png", env.observe())
            save_state("after_state.npz", env)
            next_capture_seconds = time.perf_counter() - next_capture_started
            tracking_error = float(np.linalg.norm(env.controller.target_position - after_position))
            finite_state = bool(np.all(np.isfinite(env.data.qpos)) and np.all(np.isfinite(env.data.qvel)))
            tracking_passed = finite_state and tracking_error <= 0.005
            report = {
                "request_id": request_id, "raw_action": prediction["action"],
                "controller_action": action.tolist(), "unnorm_key": "bridge_orig",
                "mapping": "Provisional Bridge world XYZ/Euler to Panda world XYZ/Euler; hand-origin control point",
                "translation_limit_metres": 0.01, "rotation_limit_radians": 0.05,
                "gripper_mapping": "2 * model_gripper - 1",
                "before_position": before_position.tolist(), "after_position": after_position.tolist(),
                "actual_translation": (after_position - before_position).tolist(),
                "before_quaternion_wxyz": before_quaternion.tolist(),
                "after_quaternion_wxyz": after_quaternion.tolist(),
                "target_position": env.controller.target_position.tolist(),
                "position_tracking_error_metres": tracking_error,
                "ik_converged": bool(ik.converged), "finite_state": finite_state,
                "tracking_passed": tracking_passed,
                "physics_seconds": 1000 * float(env.model.opt.timestep),
                "simulation_frozen_during_request": True,
                "motion_started": motion_started,
                "timings_seconds": {
                    "initial_capture_and_state_save": capture_seconds,
                    "ik": ik_seconds, "robot_execution_wall": movement_seconds,
                    "next_capture_and_state_save": next_capture_seconds,
                    "action_to_next_image_ready": time.perf_counter() - ik_started,
                    "total_to_next_image_ready": time.perf_counter() - started,
                },
                "server_timings_seconds": prediction.get("timings_seconds"),
                "request_timings_seconds": prediction["client_timings_seconds"],
            }
            save("execution.json", report)
            if not tracking_passed:
                raise RuntimeError(f"Action executed but tracking failed: {tracking_error:.6f} m error, finite_state={finite_state}")
            status("Completed")
            print("Raw OpenVLA action:", prediction["action"], flush=True)
            print("Bounded controller action:", action.tolist(), flush=True)
            print(f"Position tracking error: {tracking_error * 1000:.3f} mm", flush=True)
            print(f"Before/after images and timings: {folder}", flush=True)
            if not no_view:
                import mujoco.viewer
                print("Showing the final scene. Close the viewer to exit; no further requests are made.", flush=True)
                mujoco.viewer.launch(env.model, env.data)
            return folder
    except Exception as exc:
        save("status.json", {"phase": "Failed", "failed_stage": phase,
                             "request_id": request_id, "error": str(exc),
                             "motion_started": motion_started,
                             "elapsed_seconds": time.perf_counter() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--instruction", default="pick up the red cube")
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-view", action="store_true")
    args = parser.parse_args()
    run_single_action(server_url=args.server_url, instruction=args.instruction,
                      timeout_seconds=args.timeout_seconds, output_dir=args.output_dir,
                      no_view=args.no_view)


if __name__ == "__main__":
    main()
