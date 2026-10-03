"""Launch the Phase 1 Panda scene in MuJoCo's interactive viewer."""

from pathlib import Path

import mujoco
import mujoco.viewer


SCENE_PATH = Path(__file__).with_name("scene.xml")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "scene_home")
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    mujoco.mj_forward(model, data)

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
