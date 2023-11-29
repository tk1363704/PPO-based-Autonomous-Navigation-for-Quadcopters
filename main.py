from pathlib import Path

import time

import yaml

import typer

from typing_extensions import Annotated

import gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback

from scripts.airsim_env import TrainConfig
import numpy as np
import wandb

app = typer.Typer()

current_file_path = Path(__file__).parent


@app.command()
def evaluate(
    config_file: Annotated[Path, typer.Option()] = Path(
        current_file_path / "scripts/config.yml"
    ),
    sim_ip: Annotated[str, typer.Option()] = "127.0.0.1",
    run_steps: Annotated[int, typer.Option()] = 100,
):
    # Get train environment configs
    with open(config_file, "r", encoding="utf8") as f:
        env_config = yaml.safe_load(f)
        config = TrainConfig(**env_config["TrainEnv"])

    # Create a DummyVecEnv
    env = DummyVecEnv(
        [
            lambda: Monitor(
                gymnasium.make(
                    "scripts:airsim-env-v0",
                    ip_address=sim_ip,
                    env_config=config
                )
            )
        ]
    )

    # Wrap env as VecTransposeImage (Channel last to channel first)
    env = VecTransposeImage(env)

    # Load an existing model
    model = PPO.load(env=env, path="/home/nick/Dev/AirSim/PPO-based-Autonomous-Navigation-for-Quadcopters/best_model")

    num_success = 0
    num_timeout = 0
    num_collision = 0
    # Run the trained policy
    for _ in range(run_steps):
        obs = env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done[0] == True:
                if info[0]["is_success"] == True:
                    num_success += 1 
                if info[0]["is_collision"] == True:
                    num_collision += 1 
                if info[0]["timeout"] == True:
                    num_timeout += 1

    print(f"number of success = {num_success}")
    print(f"number of collision = {num_collision}")
    print(f"number of timeout = {num_timeout}")


@app.command()
def train(
    config_file: Annotated[Path, typer.Option()] = Path(
        current_file_path / "scripts/config.yml"
    ),
    sim_ip: Annotated[str, typer.Option()] = "127.0.0.1",
    train_timesteps: Annotated[int, typer.Option()] = 400000,
    seed: Annotated[int, typer.Option()] = 5,
):
    # Get train environment configs
    with open(config_file, "r", encoding="utf8") as f:
        env_config = yaml.safe_load(f)
        config = TrainConfig(**env_config["TrainEnv"])

    # Create a DummyVecEnv
    # DummyVecEnv: This is a class provided by the stable-baselines library,
    # which is often used in conjunction with Gym environments. It allows you to
    # wrap multiple Gym environments into a single vectorized environment,
    # making it easier to work with algorithms that require parallelism.
    #
    # [lambda: Monitor(...)]: This is a list containing a lambda function. The
    # lambda function is used to create an instance of the environment, and the
    # environment is wrapped in a Monitor. The Monitor is used to record
    # statistics about the environment during training.
    #
    # gym.make("scripts:airsim-env-v0", ...): This line is creating an instance
    # of the "airsim-env-v0" environment that you previously registered using
    # register. It's passing some configuration parameters like the IP address,
    # image shape, and env_config to customize the environment.

    # Here using the class airsim_env.__init__() to create an instance.
    # Image_shape=(50, 50, 3) describes an image or observation with a width of
    # 50 pixels, a height of 50 pixels, and three color channels
    # (red, green, blue). This shape is often encountered when working with
    # image-based reinforcement learning tasks, where the agent receives images
    # from the environment as observations.
    # env = DummyVecEnv(
    #     [
    #         lambda: Monitor(
    #             gymnasium.make(
    #                 "scripts:airsim-env-v0",
    #                 ip_address=sim_ip,
    #                 env_config=config,
    #             )
    #         )
    #     ]
    # )

    # # Wrap env as VecTransposeImage (Channel last to channel first)
    # env = VecTransposeImage(env)

    env = gymnasium.make("scripts:airsim-env-v0",
                    ip_address=sim_ip,
                    env_config=config)
    # Initialize PPO
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=seed,
        device="cuda",
        tensorboard_log="./tb_logs/",
    )

    wandb.init(project="drone_nav",
                   entity="nahsen",
                   sync_tensorboard=True
                   )
        
    # Evaluation callback
    # EvalCallback is a callback class commonly used in reinforcement learning
    # libraries like Stable Baselines to facilitate the evaluation of trained
    # reinforcement learning agents during training. This callback allows you
    # to monitor and evaluate an agent's performance at specified intervals
    # while it's learning from interactions with the environment.
    callbacks = []
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=None,
        n_eval_episodes=20,
        best_model_save_path=".",
        log_path=".",
        eval_freq=5000,
    )

    callbacks.append(eval_callback)
    kwargs = {}
    kwargs["callback"] = callbacks

    log_name = "ppo_run_" + str(time.time())

    model.learn(total_timesteps=train_timesteps, tb_log_name=log_name, **kwargs)

    # Save policy weights
    model.save("ppo_navigation_policy")

@app.command()
def test(config_file: Annotated[Path, typer.Option()] = Path("/home/nick/Dev/AirSim/PPO-based-Autonomous-Navigation-for-Quadcopters/scripts/config.yml"),
    sim_ip: Annotated[str, typer.Option()] = "127.0.0.1"
    ):
    # Get train environment configs
    with open(config_file, "r", encoding="utf8") as f:
        env_config = yaml.safe_load(f)
        config = TrainConfig(**env_config["TrainEnv"])

    env = gymnasium.make("scripts:airsim-env-v0",
                    ip_address=sim_ip,
                    env_config=config)
    import time
    for i in range(30):
        env.reset()
        env.step([0, 0, 1])
        time.sleep(5)

if __name__ == "__main__":
    # app()
    evaluate()
    #train()
