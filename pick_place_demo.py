"""Scripted pick/place using known cube coordinates and the 7D action API.

Run from VS Code for a viewer, or use --headless for a physics-only test.
No cube teleportation, welds, or external grasp forces are used.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import time
from datetime import datetime

import mujoco
import mujoco.viewer
import numpy as np

from panda_controller import PandaController
from panda_actions import TRANSLATION_LIMIT, make_controller
from end_effector_demo import _restart_with_mjpython
from capture_camera import save_rgb_png


def sequence(controller, snapshot=lambda name: None):
    """Yield once per physics step, allowing the caller to render or pace it."""
    model, data = controller.model, controller.data
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")

    def wait(seconds):
        for _ in range(round(seconds / model.opt.timestep)):
            controller.step()
            yield

    def move(target, gripper):
        for _ in range(80):
            position, _ = controller.end_effector_pose()
            delta = target - position
            if np.linalg.norm(delta) < 0.0015:
                return
            delta *= min(1.0, TRANSLATION_LIMIT / np.linalg.norm(delta))
            result = controller.apply_action([*delta, 0, 0, 0, gripper])
            if not result.converged:
                raise RuntimeError("Waypoint IK failed; stopping pick/place")
            yield from wait(0.4)
        raise RuntimeError("Waypoint did not settle")

    yield from wait(0.8)
    mujoco.mj_forward(model, data)
    initial = data.xpos[cube_id].copy()
    destination = initial + np.array([0, 0.12, 0])
    snapshot("01_start")
    print("Approach cube", flush=True)
    yield from move(initial + [0, 0, 0.14], 1)
    print("Lower between fingers", flush=True)
    yield from move(initial + [0, 0, 0.003], 1)
    controller.apply_action([0, 0, 0, 0, 0, 0, -1])
    print("Close gripper", flush=True)
    yield from wait(1)
    snapshot("02_grasp")
    print("Lift", flush=True)
    yield from move(initial + [0, 0, 0.14], -1)
    mujoco.mj_forward(model, data)
    lifted = data.xpos[cube_id].copy()
    if lifted[2] < initial[2] + 0.08:
        raise RuntimeError(f"Grasp failed: cube only rose {lifted[2]-initial[2]:.3f} m")
    print(f"Verified cube lift: {lifted[2]-initial[2]:.3f} m", flush=True)
    snapshot("03_lift")
    print("Transfer 12 cm sideways", flush=True)
    yield from move(destination + [0, 0, 0.14], -1)
    snapshot("04_transfer")
    print("Lower and release", flush=True)
    yield from move(destination + [0, 0, 0.006], -1)
    controller.apply_action([0, 0, 0, 0, 0, 0, 1])
    yield from wait(0.8)
    yield from move(destination + [0, 0, 0.14], 1)
    yield from wait(1)
    mujoco.mj_forward(model, data)
    final = data.xpos[cube_id].copy()
    error = np.linalg.norm(final[:2] - destination[:2])
    if error > 0.015 or abs(final[2] - initial[2]) > 0.01:
        raise RuntimeError(f"Placement failed: cube={final}, target={destination}")
    print(f"PASS: cube placed on table; horizontal error {error*1000:.1f} mm", flush=True)
    snapshot("05_placed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--capture", action="store_true", help="Save images even in headless mode")
    args = parser.parse_args()
    if not args.headless:
        _restart_with_mjpython()
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("scene.xml")))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "scene_home")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    # 0.0584 m finger mounting offset + approximately 0.044 m pad offset.
    controller = make_controller(model, data)
    context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)
    capture = not args.headless or args.capture
    renderer_context = mujoco.Renderer(model, height=480, width=640) if capture else nullcontext(None)
    with context as viewer, renderer_context as renderer:
        output = Path(__file__).parent / "outputs" / ("pick_place_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        if renderer:
            output.mkdir(parents=True)
            print(f"Camera snapshots: {output}", flush=True)

        def snapshot(name):
            if renderer:
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="workspace_camera")
                save_rgb_png(output / f"{name}.png", renderer.render())

        if viewer:
            print("Pick/place begins in 3 seconds", flush=True)
            time.sleep(3)
        steps = sequence(controller, snapshot)
        while viewer is None or viewer.is_running():
            with viewer.lock() if viewer else nullcontext():
                try:
                    next(steps)
                except StopIteration:
                    break
            if viewer:
                viewer.sync()
                time.sleep(model.opt.timestep)
        if viewer:
            print("Demo complete. Close the viewer to exit.", flush=True)
            while viewer.is_running():
                with viewer.lock():
                    controller.step()
                viewer.sync()
                time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
