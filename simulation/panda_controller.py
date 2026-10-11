"""Cartesian 7D action controller for the MuJoCo Franka Panda.

Actions have the form:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]

Translation is expressed in world-frame metres, rotation in world-frame radians,
and gripper is normalized to [-1, 1], where -1 is closed and +1 is open.
"""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class IKResult:
    joint_targets: np.ndarray
    position_error: float
    orientation_error: float
    iterations: int
    converged: bool


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two quaternions stored in MuJoCo's [w, x, y, z] order."""
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _euler_xyz_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert XYZ Euler deltas to a quaternion."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    qx = np.array([cr, sr, 0.0, 0.0])
    qy = np.array([cp, 0.0, sp, 0.0])
    qz = np.array([cy, 0.0, 0.0, sy])
    return _quat_multiply(qz, _quat_multiply(qy, qx))


def _orientation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return the world-frame rotation vector from current to target."""
    conjugate = current.copy()
    conjugate[1:] *= -1
    difference = _quat_multiply(target, conjugate)
    if difference[0] < 0:
        difference *= -1

    vector_norm = np.linalg.norm(difference[1:])
    if vector_norm < 1e-12:
        return np.zeros(3)

    angle = 2 * np.arctan2(vector_norm, np.clip(difference[0], -1.0, 1.0))
    return difference[1:] * (angle / vector_norm)


class PandaController:
    """Convert incremental Cartesian actions into Panda position targets."""

    ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
    ARM_ACTUATOR_NAMES = tuple(f"actuator{i}" for i in range(1, 8))

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        max_translation: float = 0.05,
        max_rotation: float = 0.25,
        damping: float = 0.03,
        tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.model = model
        self.data = data
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.damping = damping
        self.tool_offset = np.asarray(tool_offset, dtype=float)
        if self.tool_offset.shape != (3,) or not np.all(np.isfinite(self.tool_offset)):
            raise ValueError("tool_offset must contain three finite coordinates")

        self.hand_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.joint_ids = np.array(
            [self._required_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.ARM_JOINT_NAMES]
        )
        self.actuator_ids = np.array(
            [
                self._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.ARM_ACTUATOR_NAMES
            ]
        )
        self.gripper_actuator_id = self._required_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids]
        self.dof_indices = self.model.jnt_dofadr[self.joint_ids]
        self.joint_limits = self.model.jnt_range[self.joint_ids].copy()
        self.target_position, self.target_quaternion = self.end_effector_pose()
        self._steps_since_ik = 0

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Required MuJoCo object not found: {name}")
        return object_id

    def end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return hand position and quaternion in world coordinates."""
        mujoco.mj_forward(self.model, self.data)
        return self._tool_position(self.data), self.data.xquat[self.hand_id].copy()

    def _tool_position(self, data):
        return data.xpos[self.hand_id] + data.xmat[self.hand_id].reshape(3, 3) @ self.tool_offset

    def apply_action(self, action: np.ndarray | list[float]) -> IKResult:
        """Apply one clipped 7D delta action and update actuator targets."""
        action = np.asarray(action, dtype=float)
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), received {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action values must all be finite")

        translation = np.clip(action[:3], -self.max_translation, self.max_translation)
        rotation = np.clip(action[3:6], -self.max_rotation, self.max_rotation)
        gripper = float(np.clip(action[6], -1.0, 1.0))

        current_position, current_quaternion = self.end_effector_pose()
        target_position = current_position + translation
        delta_quaternion = _euler_xyz_to_quat(*rotation)
        target_quaternion = _quat_multiply(delta_quaternion, current_quaternion)
        target_quaternion /= np.linalg.norm(target_quaternion)

        result = self._solve_ik(target_position, target_quaternion)
        if result.converged:
            self.target_position = target_position
            self.target_quaternion = target_quaternion
            self.data.ctrl[self.actuator_ids] = result.joint_targets

        # Panda actuator8 maps 0..255 to closed..open.
        self.data.ctrl[self.gripper_actuator_id] = (gripper + 1.0) * 127.5
        return result

    def _solve_ik(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        *,
        max_iterations: int = 150,
        position_tolerance: float = 1e-4,
        orientation_tolerance: float = 2e-3,
    ) -> IKResult:
        """Solve a pose target with damped least-squares inverse kinematics."""
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0

        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        position_norm = float("inf")
        orientation_norm = float("inf")

        for iteration in range(1, max_iterations + 1):
            mujoco.mj_forward(self.model, scratch)
            tool_position = self._tool_position(scratch)
            position_error = target_position - tool_position
            rotation_error = _orientation_error(
                target_quaternion, scratch.xquat[self.hand_id]
            )
            position_norm = float(np.linalg.norm(position_error))
            orientation_norm = float(np.linalg.norm(rotation_error))

            if (
                position_norm <= position_tolerance
                and orientation_norm <= orientation_tolerance
            ):
                break

            mujoco.mj_jac(
                self.model,
                scratch,
                jacobian_position,
                jacobian_rotation,
                tool_position,
                self.hand_id,
            )
            jacobian = np.vstack(
                (
                    jacobian_position[:, self.dof_indices],
                    jacobian_rotation[:, self.dof_indices],
                )
            )
            error = np.concatenate((position_error, rotation_error))
            regularized = jacobian @ jacobian.T + (self.damping**2) * np.eye(6)
            joint_delta = jacobian.T @ np.linalg.solve(regularized, error)

            # Keep each numerical IK iteration small and predictable.
            largest_change = np.max(np.abs(joint_delta))
            if largest_change > 0.12:
                joint_delta *= 0.12 / largest_change

            next_positions = scratch.qpos[self.qpos_indices] + joint_delta
            scratch.qpos[self.qpos_indices] = np.clip(
                next_positions,
                self.joint_limits[:, 0] + 1e-4,
                self.joint_limits[:, 1] - 1e-4,
            )

        converged = (
            position_norm <= position_tolerance
            and orientation_norm <= orientation_tolerance
        )
        return IKResult(
            joint_targets=scratch.qpos[self.qpos_indices].copy(),
            position_error=position_norm,
            orientation_error=orientation_norm,
            iterations=iteration,
            converged=converged,
        )

    def step(self, steps: int = 1, *, ik_interval: int = 10) -> None:
        """Advance physics and continually correct Cartesian tracking error."""
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if ik_interval < 1:
            raise ValueError("ik_interval must be at least 1")

        for _ in range(steps):
            if self._steps_since_ik >= ik_interval:
                correction = self._solve_ik(
                    self.target_position, self.target_quaternion
                )
                # Accumulate a small outer-loop correction. This compensates
                # for gravity and actuator tracking error instead of merely
                # reissuing the same ideal kinematic joint configuration.
                joint_error = (
                    correction.joint_targets - self.data.qpos[self.qpos_indices]
                )
                corrected_targets = self.data.ctrl[self.actuator_ids] + 0.15 * joint_error
                self.data.ctrl[self.actuator_ids] = np.clip(
                    corrected_targets,
                    self.joint_limits[:, 0],
                    self.joint_limits[:, 1],
                )
                self._steps_since_ik = 0
            mujoco.mj_step(self.model, self.data)
            self._steps_since_ik += 1
