"""Minimal camera-to-action environment for the Phase 1 Panda scene."""

from pathlib import Path

import mujoco
import numpy as np

from panda_controller import IKResult, PandaController
from panda_actions import PHYSICS_STEPS, make_controller


class PandaEnv:
    """Expose MuJoCo through reset, RGB observation, and 7D action methods."""

    def __init__(
        self,
        scene_path: str | Path | None = None,
        *,
        camera: str = "workspace_camera",
        width: int = 640,
        height: int = 480,
    ) -> None:
        if scene_path is None:
            scene_path = Path(__file__).with_name("scene.xml")

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.camera = camera
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.controller: PandaController
        self.reset()

    def reset(self) -> np.ndarray:
        """Restore the scene home keyframe and return its RGB observation."""
        home_key = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "scene_home"
        )
        if home_key < 0:
            raise ValueError("scene_home keyframe is missing")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
        mujoco.mj_forward(self.model, self.data)
        self.controller = make_controller(self.model, self.data)
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return a new HxWx3 uint8 RGB image from the workspace camera."""
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def step(
        self,
        action: np.ndarray | list[float],
        *,
        physics_steps: int = PHYSICS_STEPS,
    ) -> tuple[np.ndarray, IKResult]:
        """Execute one 7D action and return the resulting image and IK status."""
        if physics_steps < 1:
            raise ValueError("physics_steps must be at least 1")
        from remote_loop import apply_with_ik_retries
        _, ik_result, _ = apply_with_ik_retries(self.controller, np.asarray(action, dtype=float))
        if ik_result.converged:
            self.controller.step(physics_steps)
        return self.observe(), ik_result

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> "PandaEnv":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
