"""Repeated cloud predictions driving a local Panda, with a live viewer."""

from common.output_layout import output_path

from common.paths import PROJECT_ROOT

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

from simulation.ik_retries import apply_with_ik_retries
from cloud.connection_client import request_prediction
from inference.openvla_single_action import adapt_bridge_action
from simulation.panda_actions import PHYSICS_STEPS, action_contract, from_controller_action, to_controller_action
from simulation.control_diagnostics import pose_diagnostics
from simulation.viewer_text import show_instruction

ROOT = PROJECT_ROOT




class ActiveTimer:
    """Thread-safe elapsed time excluding periods when the viewer is paused."""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.started = clock()
        self.paused_at = None
        self.paused_seconds = 0.0
        self.lock = threading.Lock()

    def set_paused(self, value):
        with self.lock:
            now = self.clock()
            if value and self.paused_at is None:
                self.paused_at = now
            elif not value and self.paused_at is not None:
                self.paused_seconds += now - self.paused_at
                self.paused_at = None

    def elapsed(self):
        with self.lock:
            end = self.clock() if self.paused_at is None else self.paused_at
            return end - self.started - self.paused_seconds


def run_loop(*, server_url="http://127.0.0.1:8000", instruction=None,
             steps=100, timeout_seconds=180, max_runtime_seconds=900,
             max_consecutive_rejections=3,
             output_dir=None, no_view=False, start_immediately=False, startup_info=None,
             stop_event=None, task='cube', scene_seed=0, sample_actions=False, temperature=1., top_p=1.,
             target_color=None, cube_colors=None, scene_positions=None):
    from inference.action_sampling import generation_options
    generation=generation_options(sample_actions,temperature,top_p)
    import mujoco
    import numpy as np
    from simulation.capture_camera import save_rgb_png
    from simulation.panda_env import PandaEnv
    if task not in ('cube','bowl','touch','lift','color_touch','color_transfer','multi_color_touch'):
        raise ValueError('Unknown simulation task')
    if task=='color_touch':
        from inference.color_touch_task import select_instruction, ContactOutcome
        instruction,target_color=select_instruction(instruction,target_color)
    if task=='color_transfer':
        from simulation.color_palette import select_instruction as transfer_instruction, configure_colors
        from inference.color_touch_task import ContactOutcome
        instruction,target_slot,color_names=transfer_instruction(instruction,target_color,cube_colors or ['black','purple'])
        target_color=color_names[target_slot]
    elif task=='multi_color_touch':
        from simulation.multi_color_scene import validate_colors, ContactOutcome
        validate_colors(cube_colors or [])
        if target_color not in cube_colors:raise ValueError('Requested color must be visible')
        expected=f'touch the {target_color} cube.'
        if instruction is not None and instruction.strip().lower().rstrip('.')!=expected.rstrip('.'):
            raise ValueError('Instruction and target color disagree')
        instruction=instruction or expected
        color_names={f'cube_{i}':c for i,c in enumerate(cube_colors)}
    elif cube_colors is not None:
        raise ValueError('Cube color overrides require the separate color_transfer task')
    if instruction is None:
        instruction={'cube':'pick up the red cube','bowl':'pick up the red cube and drop it in the blue bowl.',
                     'touch':'touch the red cube.','lift':'pick up the red cube.'}[task]
    if scene_positions is not None and task not in ('lift','color_touch','color_transfer','multi_color_touch'):
        raise ValueError('Saved scene positions require a two-cube task')
    touch_result=None
    env_options={}
    camera='workspace_camera'
    if task=='bowl':
        from simulation.bowl_scene import SCENE_PATH, CAMERA_NAME, randomize_scene
        env_options=dict(scene_path=SCENE_PATH,camera=CAMERA_NAME)
        camera=CAMERA_NAME

    if task=='lift':
        from simulation.lift_scene import SCENE_PATH, CAMERA_NAME, initialize_scene, LiftSuccess
        from data_tools.panda_demonstrations import cube_position
        if type(scene_seed) is not int or scene_seed<0:raise ValueError('Lift scene seed must be nonnegative')
        env_options=dict(scene_path=SCENE_PATH,camera=CAMERA_NAME)
        camera=CAMERA_NAME

    if task=='touch':
        from simulation.touch_scene import SCENE_PATH, CAMERA_NAME, initialize_scene
        from data_tools.collect_panda_touch import scenario as touch_scenario
        from data_tools.touch_demonstrations import task_result as touch_task_result
        from data_tools.panda_demonstrations import cube_position
        if type(scene_seed) is not int or scene_seed<0:raise ValueError('Touch scene seed must be nonnegative')
        env_options=dict(scene_path=SCENE_PATH,camera=CAMERA_NAME)
        camera=CAMERA_NAME

    if task in ('color_touch','color_transfer'):
        from simulation.color_touch_scene import SCENE_PATH, CAMERA_NAME, initialize_scene
        from data_tools.collect_panda_color_touch import pair_spec
        if type(scene_seed) is not int or scene_seed<0:raise ValueError('Color scene seed must be nonnegative')
        env_options=dict(scene_path=SCENE_PATH,camera=CAMERA_NAME)
        camera=CAMERA_NAME

    if task=='multi_color_touch':
        from simulation.multi_color_scene import scene_path, CAMERA_NAME
        env_options=dict(scene_path=scene_path(len(cube_colors)),camera=CAMERA_NAME)
        camera=CAMERA_NAME

    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(max_consecutive_rejections) is not int or max_consecutive_rejections < 1:
        raise ValueError("max_consecutive_rejections must be a positive integer")
    if any(not math.isfinite(x) or x <= 0 for x in (timeout_seconds, max_runtime_seconds)):
        raise ValueError("Timeout and runtime must be finite positive seconds")
    stop_event = stop_event if stop_event is not None else threading.Event()
    paused = threading.Event()
    if not no_view and not start_immediately:
        paused.set()
    folder = Path(output_dir) if output_dir else output_path(f"remote_loop_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}")
    folder.mkdir(parents=True, exist_ok=False)
    from simulation.replay_run import write_launcher
    write_launcher(folder)
    if startup_info is not None:
        metadata = json.loads(Path(startup_info).read_text())
        (folder / "startup_info.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    started = time.perf_counter()
    active_timer = ActiveTimer()
    active_timer.set_paused(paused.is_set())
    completed = []
    rejections = []
    consecutive_rejections = 0
    prediction_attempts = 0
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
            "prediction_attempts": prediction_attempts, "rejected_steps": len(rejections),
            "consecutive_rejections": consecutive_rejections,
            "active_seconds": active_timer.elapsed(), "paused": paused.is_set(),
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
        elif active_timer.elapsed() >= max_runtime_seconds:
            reason = "runtime_limit"
        return reason is not None

    def refresh(force=False):
        nonlocal last_refresh
        if viewer is not None and viewer.is_running() and (force or time.perf_counter() - last_refresh >= 1 / 30):
            show_instruction(viewer, instruction, (mujoco.mjtFontScale.mjFONTSCALE_150,
                             mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             f"Cloud OpenVLA | {prediction_attempts}/{steps} cycles | {len(completed)} moves | {len(rejections)} held",
                             f"{'PAUSED' if paused.is_set() else phase}\nSpace: start/pause/resume | Esc: stop\nClose window: stop | Active limit: {max_runtime_seconds:g}s (pauses excluded)"))
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
                active_timer.set_paused(False)
                paused.clear()
            else:
                active_timer.set_paused(True)
                paused.set()

    print(f"Results and status: {folder}", flush=True)
    save(folder / "action_contract.json", action_contract(camera=camera))
    save(folder / "generation_options.json", generation)
    status(phase)
    try:
        with PandaEnv(**env_options) as env:
            if task=='bowl':
                env.controller,spec=randomize_scene(env.model,env.data,np.random.default_rng(scene_seed),
                    full_pose=True,random_cube_yaw=True)
                save(folder/'episode.json',dict(scene_profile='panda_bowl_v1',
                    scene_seed=scene_seed,instruction=instruction,scenario=dict(scene=spec),
                    training_eligible=False))
            elif task=='lift':
                from inference.lift_task import cube_position_for_seed,validate_positions
                xy=cube_position_for_seed(scene_seed)
                if scene_positions is not None:
                    saved=json.loads(Path(scene_positions).read_text())
                    validate_positions([saved],[{'red':xy}])
                    xy=saved['red']  # Execute the frozen plan's coordinates on every model.
                scene=initialize_scene(env,xy)
                initial_cube=cube_position(env)
                lift_tracker=LiftSuccess(initial_cube)
                touch_result=lift_tracker.observe(env)
                save(folder/'episode.json',dict(scene_profile='panda_lift_v1',
                    scene_seed=scene_seed,instruction=instruction,scenario=dict(scene=scene),
                    training_eligible=False))
            elif task=='touch':
                touch_spec=touch_scenario(scene_seed)
                scene=initialize_scene(env,touch_spec['cube_xy'])
                initial_cube=cube_position(env)
                touch_result=touch_task_result(env,initial_cube)
                save(folder/'episode.json',dict(scene_profile='panda_touch_v1',
                    scene_seed=scene_seed,instruction=instruction,scenario=dict(scene=scene),
                    demonstration_slot=scene_seed,demonstration_id=touch_spec['id'],
                    known_collected_position=scene_seed<50,training_eligible=False))
            elif task in ('color_touch','color_transfer'):
                spec=pair_spec(scene_seed)
                if scene_positions is not None:
                    saved_positions=json.loads(Path(scene_positions).read_text())
                    from cloud.sweep_plan import validate_scene_positions
                    validate_scene_positions([spec['cube_positions']],[saved_positions])
                    spec=dict(spec,cube_positions=saved_positions)
                scene=initialize_scene(env,spec['cube_positions'])
                if task=='color_transfer':configure_colors(env.model,color_names)
                color_outcome=ContactOutcome(env,target_slot if task=='color_transfer' else target_color,
                                             color_names if task=='color_transfer' else None)
                color_outcome.observe(env)
                touch_result=color_outcome.result(env)
                save(folder/'episode.json',dict(scene_profile='panda_color_transfer_v1' if task=='color_transfer' else 'panda_color_touch_v1',
                    scene_seed=scene_seed,instruction=instruction,target_color=target_color,
                    scenario=dict(scene=scene,target_color=target_color),
                    known_collected_position=scene_seed<25,training_eligible=False,
                    **({'cube_colors':color_names} if task=='color_transfer' else {})))
            elif task=='multi_color_touch':
                from simulation.multi_color_scene import positions, initialize_scene
                value=positions(scene_seed,len(cube_colors))
                if scene_positions is not None:
                    from cloud.sweep_plan import validate_multi_scene_positions
                    saved=json.loads(Path(scene_positions).read_text())
                    validate_multi_scene_positions(value,saved)
                    value=saved
                scene=initialize_scene(env,value,color_names)
                color_outcome=ContactOutcome(env,target_color,color_names)
                color_outcome.observe(env);touch_result=color_outcome.result(env)
                save(folder/'episode.json',dict(scene_profile='panda_multi_color_touch_v1',scene_seed=scene_seed,
                    instruction=instruction,target_color=target_color,cube_colors=color_names,
                    scenario=dict(scene=scene,target_color=target_color),training_eligible=False))
            else:
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
                        # Refresh derived transforms before rendering the observation.
                        before_position, before_quaternion = env.controller.end_effector_pose()
                        diagnostics = pose_diagnostics(env.controller)
                        save_rgb_png(step_dir / "before.png", env.observe())
                        state(step_dir / "scene_state.npz", env)
                        snapshot = {name: getattr(env.data, name).copy() for name in ("qpos", "qvel", "ctrl")}
                        sim_time = float(env.data.time)
                    capture_seconds = time.perf_counter() - capture_started

                    prediction_attempts += 1
                    status("Waiting for cloud prediction")
                    replies = queue.Queue(maxsize=1)
                    # The daemon only performs HTTP and queues a response. It never accesses
                    # simulation/viewer state, so closing the viewer need not wait for HTTP.
                    def predict(path=step_dir / "before.png", destination=replies):
                        try:
                            response = request_prediction(path, instruction, server_url=server_url,
                                                          timeout_seconds=min(timeout_seconds, max(0.01, max_runtime_seconds - active_timer.elapsed())),
                                                          expected_mode="openvla",
                                                          **({"generation": generation} if sample_actions else {}))
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
                    unnorm_key = prediction.get("unnorm_key")
                    if unnorm_key not in ("bridge_orig", "panda_grasp_v1"):
                        raise ValueError("Unsupported model action statistics")
                    with viewer.lock() if viewer is not None else nullcontext():
                        if float(env.data.time) != sim_time or any(
                            not np.array_equal(getattr(env.data, name), value) for name, value in snapshot.items()
                        ):
                            raise RuntimeError("Scene changed during prediction; returned action is discarded")
                        bounded_action = (to_controller_action(prediction["action"])
                                          if unnorm_key == "panda_grasp_v1"
                                          else adapt_bridge_action(prediction["action"]))
                        status("Solving bounded action")
                        ik_started = time.perf_counter()
                        action, ik, ik_attempts = apply_with_ik_retries(env.controller, bounded_action)
                        ik_seconds = time.perf_counter() - ik_started
                    save(step_dir / "ik_attempts.json", {
                        "bounded_action": bounded_action.tolist(), "attempts": ik_attempts,
                        "selected_scale": ik_attempts[-1]["scale"] if action is not None else None,
                        "pose_diagnostics_before": diagnostics,
                    })
                    if not ik.converged:
                        consecutive_rejections += 1
                        status("Action rejected; holding pose", reason="ik_exhausted")
                        capture_started = time.perf_counter()
                        with viewer.lock() if viewer is not None else nullcontext():
                            save_rgb_png(step_dir / "after.png", env.observe())
                            state(step_dir / "after_state.npz", env)
                            if float(env.data.time) != sim_time or any(
                                not np.array_equal(getattr(env.data, name), value) for name, value in snapshot.items()
                            ):
                                raise RuntimeError("Rejected action unexpectedly changed the simulation")
                        rejection = {
                            "step": current_step, "request_id": prediction["request_id"],
                            "outcome": "rejected", "reason": "ik_exhausted", "executed": False,
                            "raw_action": prediction["action"], "bounded_action": bounded_action.tolist(),
                            "controller_action": None, "action_scale": None, "ik_attempts": ik_attempts,
                            "pose_diagnostics_before": diagnostics,
                            "consecutive_rejections": consecutive_rejections,
                            "executed_physics_steps": 0, "physics_seconds": 0,
                            "timings_seconds": {"capture_and_state_save": capture_seconds, "ik": ik_seconds,
                                                "robot_execution_wall": 0,
                                                "next_capture_and_state_save": time.perf_counter() - capture_started,
                                                "cycle": time.perf_counter() - cycle_started},
                            "request_timings_seconds": prediction["client_timings_seconds"],
                            "server_timings_seconds": prediction.get("timings_seconds"),
                        }
                        save(step_dir / "execution.json", rejection)
                        rejections.append(rejection)
                        if consecutive_rejections >= max_consecutive_rejections:
                            reason = "consecutive_rejection_limit"
                            break
                        refresh(force=True)
                        continue  # New observation/prediction, not another execution of this action.
                    if len(ik_attempts) > 1:
                        print(f"Action {current_step}: IK accepted {ik_attempts[-1]['scale']:g} of the bounded movement/rotation", flush=True)
                    if stopping():
                        break
                    status("Executing bounded movement")
                    execution_started = time.perf_counter()
                    executed_steps = 0
                    motion_poses = [env.data.qpos.copy()]
                    motion_times = [float(env.data.time)]
                    while executed_steps < PHYSICS_STEPS and wait_ready():
                        chunk_started = time.perf_counter()
                        count = min(20, PHYSICS_STEPS - executed_steps)
                        with viewer.lock() if viewer is not None else nullcontext():
                            env.controller.step(count)
                            if task=='lift':touch_result=lift_tracker.observe(env)
                            if task in ('color_touch','color_transfer','multi_color_touch'):color_outcome.observe(env)
                            motion_poses.append(env.data.qpos.copy())
                            motion_times.append(float(env.data.time))
                        executed_steps += count
                        refresh()
                        if viewer is not None:
                            # Pace motion to simulation time so it is visible, not instantaneous.
                            time.sleep(max(0, count * float(env.model.opt.timestep) - (time.perf_counter() - chunk_started)))
                    np.savez_compressed(step_dir / "motion_frames.npz",
                                        qpos=np.asarray(motion_poses), time=np.asarray(motion_times))
                    execution_seconds = time.perf_counter() - execution_started
                    capture_started = time.perf_counter()
                    with viewer.lock() if viewer is not None else nullcontext():
                        after_position, after_quaternion = env.controller.end_effector_pose()
                        save_rgb_png(step_dir / "after.png", env.observe())
                        state(step_dir / "after_state.npz", env)
                        tracking_error = float(np.linalg.norm(env.controller.target_position - after_position))
                        finite = bool(np.all(np.isfinite(env.data.qpos)) and np.all(np.isfinite(env.data.qvel)))
                        if task=='touch':touch_result=touch_task_result(env,initial_cube)
                        if task in ('color_touch','color_transfer','multi_color_touch'):touch_result=color_outcome.result(env)
                    after_capture_seconds = time.perf_counter() - capture_started
                    report = {
                        "step": current_step, "request_id": prediction["request_id"],
                        "raw_action": prediction["action"], "controller_action": action.tolist(),
                        "bounded_action": bounded_action.tolist(),
                        "action_scale": ik_attempts[-1]["scale"], "ik_attempts": ik_attempts,
                        "outcome": "executed", "executed": True, "pose_diagnostics_before": diagnostics,
                        "translation_limit_metres": 0.01, "rotation_limit_radians": 0.05,
                        "mapping": ("Panda training action contract; grasp-point control"
                                    if unnorm_key == "panda_grasp_v1" else
                                    "Provisional Bridge deltas interpreted using panda_grasp_v1; grasp-point control"),
                        "action_contract": action_contract(camera=camera),
                        "executed_panda_action": from_controller_action(action).tolist(),
                        "training_eligible": False,
                        "gripper_mapping": "2 * model_gripper - 1", "unnorm_key": unnorm_key,
                        "before_position": before_position.tolist(), "after_position": after_position.tolist(),
                        "actual_translation": (after_position - before_position).tolist(),
                        "before_quaternion_wxyz": before_quaternion.tolist(), "after_quaternion_wxyz": after_quaternion.tolist(),
                        "position_tracking_error_metres": tracking_error, "finite_state": finite,
                        "ik_converged": bool(ik.converged), "simulation_frozen_during_request": True,
                        "executed_physics_steps": executed_steps,
                        "physics_seconds": executed_steps * float(env.model.opt.timestep),
                        "interrupted": executed_steps != PHYSICS_STEPS or reason is not None,
                        "timings_seconds": {"capture_and_state_save": capture_seconds, "ik": ik_seconds,
                                            "robot_execution_wall": execution_seconds,
                                            "next_capture_and_state_save": after_capture_seconds,
                                            "action_to_next_image_ready": execution_seconds + after_capture_seconds,
                                            "cycle": time.perf_counter() - cycle_started},
                        "request_timings_seconds": prediction["client_timings_seconds"],
                        "server_timings_seconds": prediction.get("timings_seconds"),
                    }
                    if task in ('touch','lift','color_touch','color_transfer','multi_color_touch'):report['task_result']=touch_result
                    save(step_dir / "execution.json", report)
                    if report["interrupted"]:
                        break
                    if not finite or tracking_error > 0.005:
                        raise RuntimeError(f"Action executed but tracking failed: {tracking_error:.6f} m")
                    completed.append(report)
                    consecutive_rejections = 0
                    save(folder / "summary.json", {"completed_steps": len(completed), "steps": completed,
                                                  "rejections": rejections, "prediction_attempts": prediction_attempts})
                    print(f"Action {current_step}: tracking error {tracking_error * 1000:.3f} mm", flush=True)
                    if task=='lift' and touch_result['success']:
                        print('Lift succeeded: held at least 5 cm above start for 1 simulator second.',flush=True)
                        break
                    if task=='touch' and touch_result['success']:
                        print('Touch task succeeded: verified finger-cube contact.',flush=True)
                        break
                    if task in ('color_touch','color_transfer','multi_color_touch') and (touch_result['correct_cube_contact'] or touch_result['wrong_cube_contact']):
                        print('Color touch task ended: '+touch_result['outcome'],flush=True)
                        break
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
                                      "requested_steps": steps, "prediction_attempts": prediction_attempts,
                                      "rejected_steps": len(rejections), "rejections": rejections,
                                      "active_seconds": active_timer.elapsed(),
                                      "elapsed_seconds": time.perf_counter() - started,
                                      **({"task_result":touch_result} if task in ('touch','lift','color_touch','color_transfer','multi_color_touch') else {})})
        from analysis.run_report import write_run_report
        write_run_report(folder)
        from analysis.summarize_gpu_inference import comparison
        try:
            comparison(ROOT / 'outputs', ROOT / 'docs/benchmarks/GPU_INFERENCE_COMPARISON.md')
        except (OSError, ValueError, KeyError) as exc:
            print(f'Inference comparison could not be refreshed: {exc}', flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--instruction", help="Instruction; defaults to the selected task’s demonstration instruction")
    parser.add_argument('--task', choices=['cube','bowl','touch','lift','color_touch','color_transfer','multi_color_touch'], default='cube')
    parser.add_argument('--target-color')
    parser.add_argument('--cube-colors',nargs='+')
    parser.add_argument('--scene-seed', type=int, default=0)
    parser.add_argument('--scene-positions',type=Path,help='Use verified saved two-cube coordinates for cross-platform resume')
    parser.add_argument("--steps", type=int, default=100,
                        help="Maximum observation/prediction cycles, including held/rejected actions")
    parser.add_argument("--max-consecutive-rejections", type=int, default=3,
                        help="Hold rejected actions; stop cleanly after this many consecutive IK rejections")
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--max-runtime-seconds", type=float, default=900,
                        help="Maximum active seconds; time paused in the viewer is excluded")
    parser.add_argument("--startup-info", type=Path, help="Cloud session startup metadata to copy into this run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--start-immediately", action="store_true")
    from inference.action_sampling import add_sampling_arguments
    add_sampling_arguments(parser)
    args = parser.parse_args()
    if not args.no_view:
        from simulation.viewer import _restart_with_mjpython
        _restart_with_mjpython()
    run_loop(**vars(args))


if __name__ == "__main__":
    main()
