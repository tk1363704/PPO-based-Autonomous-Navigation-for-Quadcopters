from gymnasium.envs.registration import register

from .airsim_env import AirSimDroneEnv, TestEnv  # noqa: F401

# Register AirSim environment as a gym environment
# register: This is a function provided by the Gym library, which is a toolkit
# for developing and comparing reinforcement learning algorithms. It's used to
# register a new environment with Gym.
# id: This is the unique identifier for the environment. In this case, the
# environment is being registered with the ID "airsim-env-v0." This ID is used
# to reference and create instances of this environment in your RL experiments.
# entry_point: This specifies how to create an instance of the environment. It's
# set to "scripts:AirSimDroneEnv," which means that when you create an instance
# of the "airsim-env-v0" environment, it will be created using the
# "AirSimDroneEnv" class or function defined in a script/module named "scripts."
register(
    id="airsim-env-v0",
    entry_point="scripts:AirSimDroneEnv",
)

# Register AirSim environment as a gym environment
register(
    id="test-env-v0",
    entry_point="scripts:TestEnv",
)
