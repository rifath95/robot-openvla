"""Repeated cloud predictions driving a local Panda, with a live viewer."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import threading
import time
import uuid

from connection_client import request_prediction
from openvla_single_action import adapt_bridge_action

ROOT = Path(__file__).resolve().parent


def run_loop(*, server_url="http://127.0.0.1:8000", instruction="pick up the red cube",
             steps=10, timeout_seconds=180, max_runtime_seconds=120,
             output_dir=None, no_view=False, start_immediately=False,
             stop_event=None):
    import mujoco
    import numpy as np
    from capture_camera import save_rgb_png
    from panda_env import PandaEnv

    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if any(not math.isfinite(x) or x <= 0 for x in (timeout_seconds, max_runtime_seconds)):
        raise ValueError("Timeout and runtime must be finite positive seconds")
    stop_event = stop_event if stop_event is not None else threading.Event()
    paused = threading.Event()
    if not no_view and not start_immediately:
        paused.set()
    folder = Path(output_dir) if output_dir else ROOT / "outputs" / f"remote_loop_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    completed = []
    phase = "Initializing simulation"
    current_step = 0
    reason = None
    viewer = None
    last_refresh = 0.0

    def save(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def status(new_phase, **extra):
        nonlocal phase
        phase = new_phase
        save(folder / "status.json", {
            "phase": phase, "current_step": current_step, "completed_steps": len(completed),
            "requested_steps": steps, "elapsed_seconds": time.perf_counter() - started,
            **extra,
        })
        print(f"Step {current_step}/{steps}: {phase}", flush=True)
        with (folder / "run.log").open("a") as log:
            log.write(f"{time.perf_counter() - started:.3f}s step {current_step}: {phase}\n")

    def state(path, env):
        np.savez(path, qpos=env.data.qpos, qvel=env.data.qvel,
                 act=env.data.act, ctrl=env.data.ctrl, time=env.data.time)

    def stopping():
        nonlocal reason
        if stop_event.is_set():
            reason = "user_stop"
        elif viewer is not None and not viewer.is_running():
            reason = "viewer_closed"
        elif time.perf_counter() - started >= max_runtime_seconds:
            reason = "runtime_limit"
        return reason is not None

    def refresh(force=False):
        nonlocal last_refresh
        if viewer is not None and viewer.is_running() and (force or time.perf_counter() - last_refresh >= 1 / 30):
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                             mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             f"Cloud OpenVLA | {len(completed)}/{steps} actions",
                             f"{'PAUSED' if paused.is_set() else phase}\nSpace: start/pause/resume | Esc: stop\nClose window: stop | Local limit: {max_runtime_seconds:g}s"))
            viewer.sync()
            last_refresh = time.perf_counter()

    def wait_ready():
        while paused.is_set() and not stopping():
            refresh()
            time.sleep(0.01)
        return not stopping()

    def on_key(key):
        import glfw
        if key == glfw.KEY_ESCAPE:
            stop_event.set()
        elif key == glfw.KEY_SPACE:
            if paused.is_set():
                paused.clear()
            else:
                paused.set()

    print(f"Results and status: {folder}", flush=True)
    status(phase)
    try:
        with PandaEnv() as env:
            env.controller.step(500)
            if no_view:
                context = nullcontext(None)
            else:
                import mujoco.viewer
                context = mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key,
                                                       show_left_ui=False, show_right_ui=False)
            with context as viewer:
                if viewer is not None:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, env.camera)
                refresh(force=True)
                if paused.is_set():
                    status("Paused; press Space to start")
                for current_step in range(1, steps + 1):
                    if not wait_ready():
                        break
                    step_dir = folder / f"step_{current_step:03d}"
                    step_dir.mkdir()
                    cycle_started = time.perf_counter()
                    status("Capturing scene")
                    capture_started = time.perf_counter()
                    with viewer.lock() if viewer is not None else nullcontext():
                        save_rgb_png(step_dir / "before.png", env.observe())
                        state(step_dir / "scene_state.npz", env)
                        snapshot = {name: getattr(env.data, name).copy() for name in ("qpos", "qvel", "ctrl")}
                        sim_time = float(env.data.time)
                        before_position, before_quaternion = env.controller.end_effector_pose()
                    capture_seconds = time.perf_counter() - capture_started

                    status("Waiting for cloud prediction")
                    replies = queue.Queue(maxsize=1)
                    # The daemon only performs HTTP and queues a response. It never accesses
                    # simulation/viewer state, so closing the viewer need not wait for HTTP.
                    def predict(path=step_dir / "before.png", destination=replies):
                        try:
                            response = request_prediction(path, instruction, server_url=server_url,
                                                          timeout_seconds=min(timeout_seconds, max(0.01, max_runtime_seconds - (time.perf_counter() - started))),
                                                          expected_mode="openvla")
                            destination.put((response, None))
                        except Exception as exc:
                            destination.put((None, exc))
                    threading.Thread(target=predict, daemon=True).start()
                    reply = None
                    while not stopping():
                        refresh()
                        try:
                            reply = replies.get_nowait()
                            break
                        except queue.Empty:
                            time.sleep(0.01)
                    if reply is None or not wait_ready():
                        break  # Late/in-flight predictions are never executed after a stop.
                    prediction, error = reply
                    if error is not None:
                        raise error
                    save(step_dir / "prediction.json", prediction)
                    if prediction.get("unnorm_key") != "bridge_orig":
                        raise ValueError("Only bridge_orig adaptation is implemented")
                    with viewer.lock() if viewer is not None else nullcontext():
                        if float(env.data.time) != sim_time or any(
                            not np.array_equal(getattr(env.data, name), value) for name, value in snapshot.items()
                        ):
                            raise RuntimeError("Scene changed during prediction; returned action is discarded")
                        action = adapt_bridge_action(prediction["action"])
                        status("Solving bounded action")
                        ik_started = time.perf_counter()
                        ik = env.controller.apply_action(action)
                        ik_seconds = time.perf_counter() - ik_started
                    if not ik.converged:
                        raise RuntimeError("IK rejected the action; no physics is advanced for it")
                    if stopping():
                        break
                    status("Executing bounded movement")
                    execution_started = time.perf_counter()
                    executed_steps = 0
                    while executed_steps < 1000 and wait_ready():
                        chunk_started = time.perf_counter()
                        count = min(20, 1000 - executed_steps)
                        with viewer.lock() if viewer is not None else nullcontext():
                            env.controller.step(count)
                        executed_steps += count
                        refresh()
                        if viewer is not None:
                            # Pace motion to simulation time so it is visible, not instantaneous.
                            time.sleep(max(0, count * float(env.model.opt.timestep) - (time.perf_counter() - chunk_started)))
                    execution_seconds = time.perf_counter() - execution_started
                    capture_started = time.perf_counter()
                    with viewer.lock() if viewer is not None else nullcontext():
                        after_position, after_quaternion = env.controller.end_effector_pose()
                        save_rgb_png(step_dir / "after.png", env.observe())
                        state(step_dir / "after_state.npz", env)
                        tracking_error = float(np.linalg.norm(env.controller.target_position - after_position))
                        finite = bool(np.all(np.isfinite(env.data.qpos)) and np.all(np.isfinite(env.data.qvel)))
                    after_capture_seconds = time.perf_counter() - capture_started
                    report = {
                        "step": current_step, "request_id": prediction["request_id"],
                        "raw_action": prediction["action"], "controller_action": action.tolist(),
                        "translation_limit_metres": 0.01, "rotation_limit_radians": 0.05,
                        "mapping": "Provisional Bridge world XYZ/Euler to Panda world XYZ/Euler; hand origin",
                        "gripper_mapping": "2 * model_gripper - 1", "unnorm_key": "bridge_orig",
                        "before_position": before_position.tolist(), "after_position": after_position.tolist(),
                        "actual_translation": (after_position - before_position).tolist(),
                        "before_quaternion_wxyz": before_quaternion.tolist(), "after_quaternion_wxyz": after_quaternion.tolist(),
                        "position_tracking_error_metres": tracking_error, "finite_state": finite,
                        "ik_converged": bool(ik.converged), "simulation_frozen_during_request": True,
                        "executed_physics_steps": executed_steps,
                        "physics_seconds": executed_steps * float(env.model.opt.timestep),
                        "interrupted": executed_steps != 1000 or reason is not None,
                        "timings_seconds": {"capture_and_state_save": capture_seconds, "ik": ik_seconds,
                                            "robot_execution_wall": execution_seconds,
                                            "next_capture_and_state_save": after_capture_seconds,
                                            "action_to_next_image_ready": execution_seconds + after_capture_seconds,
                                            "cycle": time.perf_counter() - cycle_started},
                        "request_timings_seconds": prediction["client_timings_seconds"],
                        "server_timings_seconds": prediction.get("timings_seconds"),
                    }
                    save(step_dir / "execution.json", report)
                    if report["interrupted"]:
                        break
                    if not finite or tracking_error > 0.005:
                        raise RuntimeError(f"Action executed but tracking failed: {tracking_error:.6f} m")
                    completed.append(report)
                    save(folder / "summary.json", {"completed_steps": len(completed), "steps": completed})
                    print(f"Action {current_step}: tracking error {tracking_error * 1000:.3f} mm", flush=True)
                status("Stopped" if reason is not None else "Completed", stop_reason=reason)
                refresh(force=True)
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
        status("Stopped", stop_reason=reason)
    except Exception as exc:
        failed_phase = phase
        status("Failed", failed_stage=failed_phase, error=str(exc))
        raise
    finally:
        # Preserve successful steps even if a later request/movement fails or is stopped.
        save(folder / "summary.json", {"completed_steps": len(completed), "steps": completed,
                                      "phase": phase, "stop_reason": reason,
                                      "elapsed_seconds": time.perf_counter() - started})
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--instruction", default="pick up the red cube")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--max-runtime-seconds", type=float, default=120)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--start-immediately", action="store_true")
    args = parser.parse_args()
    if not args.no_view:
        from end_effector_demo import _restart_with_mjpython
        _restart_with_mjpython()
    run_loop(**vars(args))


if __name__ == "__main__":
    main()
