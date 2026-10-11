"""Launch the randomized cube-to-plate scene with MuJoCo actuator controls."""

import argparse
import json
import numpy as np

import mujoco
import mujoco.viewer


from plate_scene import SCENE_PATH, randomize_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, help='Reproduce the same starting scene')
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    _, specification = randomize_scene(model, data, np.random.default_rng(args.seed))
    print(json.dumps(specification, indent=2), flush=True)

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
