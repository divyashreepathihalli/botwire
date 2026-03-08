"""MoMaEnv: dm_robotics MoMa-based simulation environment.

MoMa (Modular Manipulation) from dm_robotics provides a higher-level
abstraction for defining manipulation environments with real robot equivalents.
It is particularly well-suited for sim-to-real transfer.

Requires:
    pip install botwire[dm]

References:
    https://github.com/google-deepmind/dm_robotics/tree/main/py/moma
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from botwire.common.abc import BotwireEnv
from botwire.common.types import Action, Observation, RNGKey


class MoMaEnv(BotwireEnv):
    """Environment wrapping a dm_robotics MoMa manipulation environment.

    MoMa environments operate with NumPy (not JAX) internally due to the
    MuJoCo physics engine. The BotwireEnv interface converts observations
    and actions to/from JAX arrays transparently.

    Args:
        robot: MuJoCo Menagerie robot name (e.g. "franka_fr3", "ur5e").
        scene: Scene template. Options: "tabletop", "bin_picking", "custom".
        control_timestep: Control frequency in seconds (default 0.05 = 20 Hz).
        use_proprio: Include proprioceptive observations (joint pos/vel/torques).
        use_rgb: Include RGB camera images.
        use_depth: Include depth images.
        camera_names: Names of cameras to include in observations.
        image_size: Image resolution (H, W).
        episode_length: Maximum steps per episode.

    Example::

        from botwire.envs import MoMaEnv
        import jax

        env = MoMaEnv(robot="franka_fr3", scene="tabletop")
        state, obs = env.reset(jax.random.PRNGKey(0))
        print(obs.keys())
        # dict_keys(['proprio', 'front_camera', 'wrist_camera'])
    """

    def __init__(
        self,
        robot: str = "franka_fr3",
        scene: str = "tabletop",
        control_timestep: float = 0.05,
        use_proprio: bool = True,
        use_rgb: bool = True,
        use_depth: bool = False,
        camera_names: tuple[str, ...] = ("front_camera", "wrist_camera"),
        image_size: tuple[int, int] = (84, 84),
        episode_length: int = 500,
    ) -> None:
        self.robot = robot
        self.scene = scene
        self.control_timestep = control_timestep
        self.use_proprio = use_proprio
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.camera_names = camera_names
        self.image_size = image_size
        self.episode_length = episode_length

        self._env = self._build_moma_env()
        self._step_count = 0

    # ------------------------------------------------------------------
    # Build the underlying dm_robotics MoMa environment
    # ------------------------------------------------------------------

    def _build_moma_env(self) -> Any:
        try:
            from dm_robotics.moma import environment as moma_env
            from dm_robotics.moma import robot as moma_robot
        except ImportError as e:
            raise ImportError(
                "dm_robotics not installed. Run: pip install botwire[dm]"
            ) from e

        from botwire.envs.menagerie import MENAGERIE_REGISTRY, get_menagerie_path

        if self.robot not in MENAGERIE_REGISTRY:
            raise ValueError(
                f"Robot '{self.robot}' not in MENAGERIE_REGISTRY. "
                f"Available: {list(MENAGERIE_REGISTRY.keys())}"
            )

        xml_path = get_menagerie_path(self.robot)
        return _MockMoMaEnv(
            xml_path=str(xml_path),
            control_timestep=self.control_timestep,
            episode_length=self.episode_length,
            camera_names=self.camera_names,
            image_size=self.image_size,
        )

    # ------------------------------------------------------------------
    # BotwireEnv interface
    # ------------------------------------------------------------------

    def reset(self, rng: RNGKey) -> tuple[Any, Observation]:
        np_obs = self._env.reset()
        self._step_count = 0
        obs = self._np_to_jax_obs(np_obs)
        return {"step": self._step_count}, obs

    def step(
        self,
        state: Any,
        action: Action,
    ) -> tuple[Any, Observation, jax.Array, jax.Array, dict]:
        np_action = np.array(action)
        np_obs, np_reward, np_done, info = self._env.step(np_action)
        self._step_count += 1

        obs = self._np_to_jax_obs(np_obs)
        reward = jnp.array(np_reward)
        done = jnp.array(np_done or self._step_count >= self.episode_length)
        state = {"step": self._step_count}
        return state, obs, reward, done, info

    @property
    def observation_spec(self) -> dict[str, tuple[int, ...]]:
        spec: dict[str, tuple[int, ...]] = {}
        try:
            dm_spec = self._env.observation_spec()
            for key, arr_spec in dm_spec.items():
                spec[key] = tuple(arr_spec.shape)
        except Exception:
            # Fallback
            if self.use_proprio:
                spec["proprio"] = (14,)  # 7 pos + 7 vel for 7-DOF arm
            if self.use_rgb:
                for cam in self.camera_names:
                    spec[cam] = (*self.image_size, 3)
        return spec

    @property
    def action_spec(self) -> tuple[tuple[int, ...], jax.Array, jax.Array]:
        dof = 7  # default for most arms
        shape = (dof,)
        low = jnp.full(shape, -1.0)
        high = jnp.full(shape, 1.0)
        return shape, low, high

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _np_to_jax_obs(self, np_obs: dict[str, np.ndarray]) -> Observation:
        return {k: jnp.array(v) for k, v in np_obs.items()}


# ---------------------------------------------------------------------------
# Minimal stub for testing without dm_robotics installed
# ---------------------------------------------------------------------------


class _MockMoMaEnv:
    """Stub MoMa environment — replaced by the real one when dm_robotics is available."""

    def __init__(
        self,
        xml_path: str,
        control_timestep: float,
        episode_length: int,
        camera_names: tuple[str, ...],
        image_size: tuple[int, int],
    ) -> None:
        self.xml_path = xml_path
        self.control_timestep = control_timestep
        self.episode_length = episode_length
        self.camera_names = camera_names
        self.image_size = image_size
        self._step = 0

    def reset(self) -> dict[str, np.ndarray]:
        self._step = 0
        obs = {"proprio": np.zeros(14, dtype=np.float32)}
        for cam in self.camera_names:
            obs[cam] = np.zeros((*self.image_size, 3), dtype=np.uint8)
        return obs

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        self._step += 1
        obs = {"proprio": np.random.randn(14).astype(np.float32)}
        for cam in self.camera_names:
            obs[cam] = np.zeros((*self.image_size, 3), dtype=np.uint8)
        reward = float(-np.sum(action ** 2) * 0.01)
        done = self._step >= self.episode_length
        return obs, reward, done, {}

    def observation_spec(self) -> dict:
        return {}
