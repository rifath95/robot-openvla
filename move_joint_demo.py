"""Demonstrate basic Panda actuator control with one joint movement."""

from pathlib import Path

import mujoco
import mujoco.viewer


SCENE_PATH = Path(__file__).with_name("scene.xml")
JOINT_1_OFFSET = 0.35  # radians (about 20 degrees)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "scene_home")
    mujoco.mj_resetDataKeyframe(model, data, home_key)

    # The Panda's first seven controls are joint-position targets.
    data.ctrl[0] += JOINT_1_OFFSET
    mujoco.mj_forward(model, data)

    print("Joint 1 target changed by 0.35 radians.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
