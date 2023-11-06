import gymnasium as gym
import time
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Get train environment configs
with open('scripts/config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

# Create a DummyVecEnv
# DummyVecEnv: This is a class provided by the stable-baselines library, which is often used in conjunction with Gym environments. It allows you to wrap multiple Gym environments into a single vectorized environment, making it easier to work with algorithms that require parallelism.
#
# [lambda: Monitor(...)]: This is a list containing a lambda function. The lambda function is used to create an instance of the environment, and the environment is wrapped in a Monitor. The Monitor is used to record statistics about the environment during training.
#
# gym.make("scripts:airsim-env-v0", ...): This line is creating an instance of the "airsim-env-v0" environment that you previously registered using register. It's passing some configuration parameters like the IP address, image shape, and env_config to customize the environment.
# Here using the class airsim_env.__init__() to create an instance.
# Image_shape=(50, 50, 3) describes an image or observation with a width of 50 pixels, a height of 50 pixels, and three color channels (red, green, blue). This shape is often encountered when working with image-based reinforcement learning tasks, where the agent receives images from the environment as observations.
    env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "scripts:airsim-env-v0", 
        ip_address="127.0.0.1", 
        image_shape=(50,50,3),
        env_config=env_config["TrainEnv"]
    )
)])

# Wrap env as VecTransposeImage (Channel last to channel first)
env = VecTransposeImage(env)

# Initialize PPO
model = PPO(
    'CnnPolicy', 
    env, 
    verbose=1, 
    seed=42,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

# Evaluation callback
# EvalCallback is a callback class commonly used in reinforcement learning libraries like Stable Baselines to facilitate the evaluation of trained reinforcement learning agents during training. This callback allows you to monitor and evaluate an agent's performance at specified intervals while it's learning from interactions with the environment.
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=4,
    best_model_save_path=".",
    log_path=".",
    eval_freq=500,
)

callbacks.append(eval_callback)
kwargs = {}
kwargs["callback"] = callbacks

log_name = "ppo_run_" + str(time.time())

model.learn(
    total_timesteps=150000,
    tb_log_name=log_name,
    **kwargs
)

# Save policy weights
model.save("ppo_navigation_policy")
