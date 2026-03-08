"""Example: Deploy a VLA policy using Gemini Robotics SDK on a real or sim robot.

This script shows how to use the Gemini Robotics-ER model for open-vocabulary
robot control via natural language task descriptions.

Requirements:
    pip install botwire[vla]
    export GOOGLE_API_KEY=your_api_key
    # (or enroll in Gemini Robotics Trusted Tester Program for full access)

Usage:
    python examples/deploy_gemini_agent.py \\
        --task="Pick up the red cube and place it in the blue bowl" \\
        --sim   # use simulation instead of real robot
"""

from absl import app, flags, logging
import jax
import numpy as np

flags.DEFINE_string("task", "Pick up the red cube.", "Natural language task description.")
flags.DEFINE_boolean("sim", True, "Use simulation (Brax/MoMa) instead of real robot.")
flags.DEFINE_string("env", "franka_fr3", "Robot environment (sim mode only).")
flags.DEFINE_integer("steps", 100, "Number of control steps to run.")
flags.DEFINE_string("model_id", "gemini-robotics-er-2.0", "Gemini Robotics model ID.")
FLAGS = flags.FLAGS


def main(argv):
    del argv

    from botwire.configs import get_vla_config
    from botwire.policies.vla import VLAPolicy

    logging.info("Task: '%s'", FLAGS.task)
    logging.info("Model: %s", FLAGS.model_id)

    # Build VLA policy
    vla_config = get_vla_config()
    vla_config.model_id = FLAGS.model_id
    policy = VLAPolicy(vla_config, action_dim=7, task_description=FLAGS.task)
    params = {}  # VLA has no local params

    if FLAGS.sim:
        _run_in_sim(policy, params)
    else:
        _run_on_real_robot(policy, params)


def _run_in_sim(policy: "VLAPolicy", params: dict) -> None:
    """Run the VLA policy in a MoMa simulation."""
    from botwire.envs import MoMaEnv

    logging.info("Running VLA policy in simulation (%s)...", FLAGS.env)
    try:
        env = MoMaEnv(robot=FLAGS.env, use_rgb=True)
    except ImportError:
        logging.warning("dm_robotics not installed; using BraxEnv instead.")
        from botwire.envs import BraxEnv
        env = BraxEnv(env_name="ant")

    rng = jax.random.PRNGKey(0)
    state, obs = env.reset(rng)

    for step in range(FLAGS.steps):
        rng, act_rng = jax.random.split(rng)
        action = policy.select_action(params, obs, act_rng, task_description=FLAGS.task)
        state, obs, reward, done, info = env.step(state, action)

        if step % 10 == 0:
            logging.info("Step %d | reward=%.4f", step, float(reward))

        if bool(done):
            logging.info("Episode done at step %d.", step)
            break

    logging.info("Simulation run complete.")


def _run_on_real_robot(policy: "VLAPolicy", params: dict) -> None:
    """Run the VLA policy on a real robot via the hardware interface."""
    logging.warning(
        "Real robot deployment requires hardware setup. "
        "See botwire/hardware/ros_bridge.py for ROS2 integration."
    )
    logging.info("Task: '%s'", FLAGS.task)
    logging.info("To deploy on a real robot:")
    logging.info("  1. Connect your robot via ROS2 or Gemini Robotics SDK")
    logging.info("  2. Instantiate RobotInterface (e.g. ROSBridge)")
    logging.info("  3. Use policy.select_action(params, obs, rng) in your control loop")


if __name__ == "__main__":
    app.run(main)
