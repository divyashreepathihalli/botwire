"""MuJoCo Menagerie robot model registry.

MuJoCo Menagerie (https://github.com/google-deepmind/mujoco_menagerie)
provides 70+ well-calibrated MJCF robot models. This module maintains a
registry of known models and provides helpers to load them into Brax/MJX.

Installation:
    pip install mujoco-menagerie
    # or from source:
    # git clone https://github.com/google-deepmind/mujoco_menagerie

Usage:
    from botwire.envs.menagerie import MENAGERIE_REGISTRY, load_menagerie_model
    import mujoco
    model = load_menagerie_model("franka_fr3")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from absl import logging

# ---------------------------------------------------------------------------
# Registry of MuJoCo Menagerie robots
# ---------------------------------------------------------------------------

# Maps short name → (menagerie package subdir, scene XML relative path)
MENAGERIE_REGISTRY: dict[str, dict[str, str]] = {
    # Franka Robotics
    "franka_fr3": {
        "subdir": "franka_fr3",
        "scene": "scene.xml",
        "category": "arm",
        "dof": 7,
        "description": "Franka Research 3 arm (7-DOF)",
    },
    "franka_emika_panda": {
        "subdir": "franka_emika_panda",
        "scene": "scene.xml",
        "category": "arm",
        "dof": 7,
        "description": "Franka Emika Panda arm (7-DOF)",
    },
    # Universal Robots
    "ur5e": {
        "subdir": "universal_robots_ur5e",
        "scene": "scene.xml",
        "category": "arm",
        "dof": 6,
        "description": "Universal Robots UR5e arm (6-DOF)",
    },
    "ur10e": {
        "subdir": "universal_robots_ur10e",
        "scene": "scene.xml",
        "category": "arm",
        "dof": 6,
        "description": "Universal Robots UR10e arm (6-DOF)",
    },
    # KUKA
    "kuka_iiwa_14": {
        "subdir": "kuka_iiwa_14",
        "scene": "scene.xml",
        "category": "arm",
        "dof": 7,
        "description": "KUKA iiwa 14 arm (7-DOF)",
    },
    # Boston Dynamics
    "spot": {
        "subdir": "boston_dynamics_spot",
        "scene": "scene.xml",
        "category": "quadruped",
        "dof": 12,
        "description": "Boston Dynamics Spot quadruped",
    },
    # Unitree
    "unitree_go1": {
        "subdir": "unitree_go1",
        "scene": "scene.xml",
        "category": "quadruped",
        "dof": 12,
        "description": "Unitree Go1 quadruped",
    },
    "unitree_go2": {
        "subdir": "unitree_go2",
        "scene": "scene.xml",
        "category": "quadruped",
        "dof": 12,
        "description": "Unitree Go2 quadruped",
    },
    "unitree_h1": {
        "subdir": "unitree_h1",
        "scene": "scene.xml",
        "category": "humanoid",
        "dof": 19,
        "description": "Unitree H1 humanoid (19-DOF)",
    },
    "unitree_g1": {
        "subdir": "unitree_g1",
        "scene": "scene.xml",
        "category": "humanoid",
        "dof": 23,
        "description": "Unitree G1 humanoid (23-DOF)",
    },
    # ANYbotics
    "anymal_c": {
        "subdir": "anybotics_anymal_c",
        "scene": "scene.xml",
        "category": "quadruped",
        "dof": 12,
        "description": "ANYbotics ANYmal C quadruped",
    },
    # PAL Robotics
    "pal_talos": {
        "subdir": "pal_talos",
        "scene": "scene.xml",
        "category": "humanoid",
        "dof": 32,
        "description": "PAL Robotics Talos humanoid",
    },
    # Hello Robot
    "stretch3": {
        "subdir": "hello_robot_stretch",
        "scene": "scene.xml",
        "category": "mobile_manipulator",
        "dof": 9,
        "description": "Hello Robot Stretch 3 mobile manipulator",
    },
    # Robotiq grippers
    "robotiq_2f85": {
        "subdir": "robotiq_2f85",
        "scene": "scene.xml",
        "category": "gripper",
        "dof": 1,
        "description": "Robotiq 2F-85 parallel gripper",
    },
    # Google Robot
    "google_robot": {
        "subdir": "google_robot",
        "scene": "scene.xml",
        "category": "mobile_manipulator",
        "dof": 8,
        "description": "Google Robot mobile manipulator",
    },
    # Crazyflie drone
    "crazyflie2": {
        "subdir": "bitcraze_crazyflie_2",
        "scene": "scene.xml",
        "category": "drone",
        "dof": 4,
        "description": "Bitcraze Crazyflie 2 quadrotor",
    },
}


def load_menagerie_model(robot_name: str) -> Any:
    """Load a MuJoCo model from the Menagerie package.

    Requires the ``mujoco_menagerie`` package to be installed.

    Args:
        robot_name: Key from :data:`MENAGERIE_REGISTRY`.

    Returns:
        A ``mujoco.MjModel`` instance.

    Raises:
        ImportError: If mujoco_menagerie is not installed.
        KeyError: If robot_name is not in the registry.
        FileNotFoundError: If the model XML cannot be located.
    """
    try:
        import mujoco
        import mujoco_menagerie
    except ImportError as e:
        raise ImportError(
            "MuJoCo Menagerie not found. Install it with:\n"
            "  pip install mujoco mujoco-mjx\n"
            "  pip install git+https://github.com/google-deepmind/mujoco_menagerie\n"
            "or: pip install botwire[sim]"
        ) from e

    if robot_name not in MENAGERIE_REGISTRY:
        available = ", ".join(sorted(MENAGERIE_REGISTRY.keys()))
        raise KeyError(
            f"Robot '{robot_name}' not in MENAGERIE_REGISTRY. "
            f"Available robots: {available}"
        )

    info = MENAGERIE_REGISTRY[robot_name]
    menagerie_root = Path(mujoco_menagerie.__file__).parent
    model_path = menagerie_root / info["subdir"] / info["scene"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model XML not found at '{model_path}'. "
            "The mujoco_menagerie package may be corrupted."
        )

    logging.info("Loading Menagerie model '%s' from '%s'", robot_name, model_path)
    return mujoco.MjModel.from_xml_path(str(model_path))


def get_menagerie_path(robot_name: str) -> Path:
    """Return the path to the robot's scene XML without loading it.

    Useful for passing directly to Brax's ``mjx`` backend.
    """
    try:
        import mujoco_menagerie
    except ImportError as e:
        raise ImportError(
            "mujoco_menagerie not installed. Run: pip install botwire[sim]"
        ) from e

    if robot_name not in MENAGERIE_REGISTRY:
        raise KeyError(f"Robot '{robot_name}' not in MENAGERIE_REGISTRY.")

    info = MENAGERIE_REGISTRY[robot_name]
    menagerie_root = Path(mujoco_menagerie.__file__).parent
    return menagerie_root / info["subdir"] / info["scene"]


def list_robots(category: str | None = None) -> list[str]:
    """List available robot names, optionally filtered by category.

    Args:
        category: One of "arm", "quadruped", "humanoid",
                  "mobile_manipulator", "gripper", "drone". None = all.

    Returns:
        Sorted list of robot names.
    """
    if category is None:
        return sorted(MENAGERIE_REGISTRY.keys())
    return sorted(
        name
        for name, info in MENAGERIE_REGISTRY.items()
        if info.get("category") == category
    )
