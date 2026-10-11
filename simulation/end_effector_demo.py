"""Interactively test the Panda's complete 7D Cartesian action interface."""

from common.paths import PROJECT_ROOT

from pathlib import Path
import argparse
import json
import os
import queue
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import glfw

from simulation.viewer import _restart_with_mjpython
from simulation.bowl_scene import SCENE_PATH, randomize_scene
TRANSLATION_STEP = 0.025  # metres
ROTATION_STEP = 0.10  # radians




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, help='Repeat the same random scene and reset sequence')
    parser.add_argument('--headless', action='store_true', help='Save a scene preview without opening a window')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/bowl_scene_preview'))
    args = parser.parse_args()
    if not args.headless:
        _restart_with_mjpython()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)
    controller, specification = randomize_scene(model, data, rng)
    print(json.dumps(specification, indent=2), flush=True)
    if args.headless:
        from simulation.capture_camera import save_rgb_png
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(data, camera='workspace_camera')
            save_rgb_png(args.output_dir / 'scene.png', renderer.render())
        (args.output_dir / 'scene.json').write_text(json.dumps(specification, indent=2)+'\n')
        print(f'Preview saved: {args.output_dir / "scene.png"}')
        return
    # Keep at most one pending command so holding a key cannot flood the robot
    # with incremental targets faster than it can execute them.
    pending_actions: queue.Queue[str] = queue.Queue(maxsize=1)
    gripper_command = 1.0
    rotation_mode = False
    last_key_time = {}
    key_commands = {
        glfw.KEY_UP: "axis0+", glfw.KEY_DOWN: "axis0-",
        glfw.KEY_LEFT: "axis1+", glfw.KEY_RIGHT: "axis1-",
        glfw.KEY_HOME: "axis2+", glfw.KEY_END: "axis2-",
        glfw.KEY_SPACE: "mode", glfw.KEY_ENTER: "gripper",
        glfw.KEY_N: "new_scene",
    }

    def on_key(key):
        command = key_commands.get(key)
        now = time.monotonic()
        if command is None or now - last_key_time.get(key, 0) < 0.25:
            return
        last_key_time[key] = now
        try:
            pending_actions.put_nowait(command)
        except queue.Full:
            pass

    def show_controls(viewer):
        mode = "ROTATE" if rotation_mode else "MOVE"
        axes = "roll / pitch / yaw" if rotation_mode else "world X / Y / Z"
        viewer.set_texts((
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            f"Panda: {mode}",
            f"Up/Down, Left/Right, Home/End: {axes}\n"
            "Mac Home/End: Fn+Left / Fn+Right\n"
            "Space: switch Move/Rotate | Enter: open/close\n"
            "N: new random arm, cube and bowl positions\n"
            "Esc: free camera | Close window to quit",
        ))

    with mujoco.viewer.launch_passive(
        model, data, key_callback=on_key, show_right_ui=False
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'workspace_camera')
        show_controls(viewer)

        while viewer.is_running():
            try:
                command = pending_actions.get_nowait()
            except queue.Empty:
                command = None

            if command == 'new_scene':
                with viewer.lock():
                    controller, specification = randomize_scene(model, data, rng)
                gripper_command = 1.0
                print(json.dumps(specification, indent=2), flush=True)
                command = None
            elif command == "mode":
                rotation_mode = not rotation_mode
                show_controls(viewer)
                command = None
            elif command == "gripper":
                command = "close" if gripper_command > 0 else "open"
            elif command is not None and command.startswith("axis"):
                axes = ("roll", "pitch", "yaw") if rotation_mode else ("x", "y", "z")
                command = axes[int(command[4])] + command[5]

            if command is not None:
                action = np.zeros(7)
                action[6] = gripper_command
                index_and_sign = {
                    "x+": (0, 1), "x-": (0, -1),
                    "y+": (1, 1), "y-": (1, -1),
                    "z+": (2, 1), "z-": (2, -1),
                    "roll+": (3, 1), "roll-": (3, -1),
                    "pitch+": (4, 1), "pitch-": (4, -1),
                    "yaw+": (5, 1), "yaw-": (5, -1),
                }

                if command in index_and_sign:
                    index, sign = index_and_sign[command]
                    step = TRANSLATION_STEP if index < 3 else ROTATION_STEP
                    action[index] = sign * step
                elif command == "close":
                    gripper_command = -1.0
                    action[6] = gripper_command
                elif command == "open":
                    gripper_command = 1.0
                    action[6] = gripper_command

                with viewer.lock():
                    result = controller.apply_action(action)
                print(
                    f"{command:7s} | IK converged={result.converged} "
                    f"position_error={result.position_error:.6f} m "
                    f"orientation_error={result.orientation_error:.6f} rad"
                )

            with viewer.lock():
                controller.step()
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
