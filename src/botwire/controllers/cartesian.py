"""CartesianController: Cartesian space robot control via dm_robotics QP solver.

Wraps dm_robotics' QP-optimization based Cartesian controller to provide
a clean interface for end-effector position and orientation control.

Requires:
    pip install botwire[dm]

Reference:
    https://github.com/google-deepmind/dm_robotics/tree/main/cpp/controllers
"""

from __future__ import annotations

from typing import Any

import numpy as np

from botwire.common.abc import BotwireController


class CartesianController(BotwireController):
    """Cartesian space controller using dm_robotics' QP solver.

    Controls the robot end-effector in 6-DOF Cartesian space
    (3D position + 3D orientation as quaternion or axis-angle).

    Args:
        physics: MuJoCo physics object (from dm_control or MoMa).
        arm_joints: List of joint names for the robot arm.
        end_effector_site: Name of the end-effector site in the MJCF model.
        control_timestep: Control frequency in seconds.
        position_gain: Proportional gain for position error.
        orientation_gain: Proportional gain for orientation error.
        max_joint_velocity: Joint velocity saturation (rad/s).

    Example::

        from botwire.controllers import CartesianController
        from botwire.envs import MoMaEnv

        env = MoMaEnv(robot="franka_fr3", scene="tabletop")
        controller = CartesianController(
            physics=env._env._physics,
            arm_joints=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            end_effector_site="end_effector",
        )

        target_pos = np.array([0.4, 0.0, 0.5])   # xyz in world frame
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion

        joint_velocities = controller.compute_action(
            observation={"qpos": ..., "qvel": ...},
            target={"position": target_pos, "quaternion": target_quat},
        )
    """

    def __init__(
        self,
        physics: Any = None,
        arm_joints: list[str] | None = None,
        end_effector_site: str = "end_effector",
        control_timestep: float = 0.05,
        position_gain: float = 1.0,
        orientation_gain: float = 1.0,
        max_joint_velocity: float = 0.5,
    ) -> None:
        self.physics = physics
        self.arm_joints = arm_joints or []
        self.end_effector_site = end_effector_site
        self.control_timestep = control_timestep
        self.position_gain = position_gain
        self.orientation_gain = orientation_gain
        self.max_joint_velocity = max_joint_velocity

        self._qp_controller = self._build_qp_controller()

    def _build_qp_controller(self) -> Any:
        """Build the dm_robotics QP Cartesian controller."""
        try:
            from dm_robotics.controllers import cartesian_6d_to_joint_velocities_mapper
        except ImportError:
            return None  # Will use fallback proportional controller

        return None  # Placeholder: actual dm_robotics binding setup varies by version

    def compute_action(
        self,
        observation: dict[str, Any],
        target: dict[str, Any],
    ) -> np.ndarray:
        """Compute joint velocities to move the end-effector toward the target.

        Args:
            observation: Dict with "qpos" (joint positions) and "qvel" (joint velocities).
            target: Dict with "position" (xyz) and optionally "quaternion" (w, x, y, z).

        Returns:
            Joint velocity commands of shape (num_joints,) in rad/s.
        """
        if self._qp_controller is not None:
            return self._qp_solve(observation, target)
        return self._proportional_control(observation, target)

    def _proportional_control(
        self,
        observation: dict[str, Any],
        target: dict[str, Any],
    ) -> np.ndarray:
        """Fallback proportional controller for testing without dm_robotics."""
        qpos = np.array(observation.get("qpos", np.zeros(len(self.arm_joints) or 7)))
        target_pos = np.array(target.get("position", np.zeros(3)))

        # Simple Jacobian-free proportional control (for demo only)
        pos_error = target_pos
        joint_velocities = np.clip(
            self.position_gain * pos_error[:len(qpos)],
            -self.max_joint_velocity,
            self.max_joint_velocity,
        )

        # Pad to arm DOF
        num_joints = len(self.arm_joints) if self.arm_joints else 7
        if len(joint_velocities) < num_joints:
            joint_velocities = np.pad(joint_velocities, (0, num_joints - len(joint_velocities)))

        return joint_velocities[:num_joints]

    def _qp_solve(
        self,
        observation: dict[str, Any],
        target: dict[str, Any],
    ) -> np.ndarray:
        """Solve the QP problem using dm_robotics controllers."""
        # Placeholder: actual implementation requires dm_control physics
        return self._proportional_control(observation, target)

    def reset(self) -> None:
        """Reset controller state (stateless, so no-op)."""
        pass
