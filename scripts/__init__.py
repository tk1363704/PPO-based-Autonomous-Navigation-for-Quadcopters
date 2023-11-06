from .airsim_env import AirSimDroneEnv, TestEnv
from gymnasium.envs.registration import register

# Register AirSim environment as a gym environment
register(
    id="airsim-env-v0", entry_point="scripts:AirSimDroneEnv",
)

# Register AirSim environment as a gym environment
register(
    id="test-env-v0", entry_point="scripts:TestEnv",
)
