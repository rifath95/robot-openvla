"""Interactively test the Panda's complete 7D Cartesian action interface."""

from pathlib import Path
import os
import queue
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import glfw

from panda_controller import PandaController
from panda_actions import make_controller


SCENE_PATH = Path(__file__).with_name("scene.xml")
TRANSLATION_STEP = 0.025  # metres
ROTATION_STEP = 0.10  # radians


def _restart_with_mjpython() -> None:
    """Use MuJoCo's required macOS GUI launcher despite spaces in the path."""
    if sys.platform != "darwin" or mujoco.viewer._MJPYTHON is not None:
        return

    venv_bin = Path(__file__).parent / ".venv" / "bin"
    python = venv_bin / "python"
    mjpython = venv_bin / "mjpython"
    os.execv(
        str(python),
        [str(python), str(mjpython), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
    )


def main() -> None:
    _restart_with_mjpython()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "scene_home")
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    mujoco.mj_forward(model, data)

    controller = make_controller(model, data)
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
            "Esc: free camera | Close window to quit",
        ))

    with mujoco.viewer.launch_passive(
        model, data, key_callback=on_key, show_right_ui=False
    ) as viewer:
        show_controls(viewer)

        while viewer.is_running():
            try:
                command = pending_actions.get_nowait()
            except queue.Empty:
                command = None

            if command == "mode":
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
