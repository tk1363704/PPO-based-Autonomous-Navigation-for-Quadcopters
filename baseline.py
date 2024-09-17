from pathlib import Path
from enum import Enum

import yaml

import typer
from typing_extensions import Annotated

import gymnasium
import numpy as np

from scripts.airsim_env import TrainConfig
from scripts.airsim_env import ActionType


class BaselineState(int, Enum):
    MOVE_UP = 0
    MOVE_ACROSS = 1
    MOVE_DOWN = 2

    def succ(self):
        v = self.value + 1
        if v > 16:
            raise ValueError("Enumeration ended")
        return BaselineState(v)

    def pred(self):
        v = self.value - 1
        if v == 0:
            raise ValueError("Enumeration ended")
        return BaselineState(v)


current_file_path = Path(__file__).parent

app = typer.Typer()


@app.command()
def position(
    config_file: Annotated[Path, typer.Option()] = Path(
        current_file_path / "scripts/config.yml"
    ),
    sim_ip: Annotated[str, typer.Option()] = "127.0.0.1",
):
    # Get train environment configs
    with open(config_file, "r", encoding="utf8") as f:
        env_config = yaml.safe_load(f)
        config = TrainConfig(**env_config["TrainEnv"])

    env = gymnasium.make(
        "scripts:airsim-env-v0",
        ip_address=sim_ip,
        env_config=config,
    )
    for _ in range(100):
        env.reset()
        goal = env.target_pos
        env.move_to_pos(goal)


@app.command()
def avoid(
    config_file: Annotated[Path, typer.Option()] = Path(
        current_file_path / "scripts/config.yml"
    ),
    sim_ip: Annotated[str, typer.Option()] = "127.0.0.1",
    max_steps: Annotated[int, typer.Option()] = 100,
    num_trials: Annotated[int, typer.Option()] = 100,
):
    # Get train environment configs
    with open(config_file, "r", encoding="utf8") as f:
        env_config = yaml.safe_load(f)
        config = TrainConfig(**env_config["TrainEnv"])

    env = gymnasium.make(
        "scripts:airsim-env-v0",
        ip_address=sim_ip,
        env_config=config,
        action_type=ActionType.CONTINUOUS_VELOCITY,
        sim_dt=1.0,
    )

    state = BaselineState.MOVE_UP

    env.reset()
    gain = 1.0
    succ = 0
    steps_per_trial = []
    for nt in range(num_trials):
        env.reset()
        goal = env.target_pos
        state = BaselineState.MOVE_UP
        max_error = 5.0
        height = -30
        for curr_step in range(max_steps):
            current_pose = env.get_wrapper_attr("current_pose")
            if state == BaselineState.MOVE_UP:
                goal_ = [current_pose[0], current_pose[1], height]
                max_error = 5.0
                gain = 2.0
            elif state == BaselineState.MOVE_ACROSS:
                goal_ = [goal[0], goal[1], height]
                max_error = 1.0
                gain = 0.25
            elif state == BaselineState.MOVE_DOWN:
                goal_ = goal
                gain = 0.25
                max_error = 0.0

            velocity_desired = gain * (goal_ - current_pose)
            if ((goal_ - current_pose) ** 2).sum() < max_error:
                state = state.succ()

            _, reward, done, _, _ = env.step(velocity_desired)

            if done:
                if reward > 100:
                    print("succesful!!")
                    succ += 1
                env.reset()
                steps_per_trial.append(curr_step)

                print(
                    "succ: ",
                    succ,
                    "num trials: ",
                    nt,
                    "avereage steps:",
                    np.mean(steps_per_trial),
                )

                break


@app.command()
def simple(
    config_file: Annotated[Path, typer.Option()] = Path(
        current_file_path / "scripts/config.yml"
    ),
    sim_ip: Annotated[str, typer.Option()] = "127.0.0.1",
    max_steps: Annotated[int, typer.Option()] = 100,
    num_trials: Annotated[int, typer.Option()] = 100,
):
    # Get train environment configs
    with open(config_file, "r", encoding="utf8") as f:
        env_config = yaml.safe_load(f)
        config = TrainConfig(**env_config["TrainEnv"])

    env = gymnasium.make(
        "scripts:airsim-env-v0",
        ip_address=sim_ip,
        env_config=config,
        action_type=ActionType.CONTINUOUS_VELOCITY,
        sim_dt=1.0,
    )

    env.reset()
    gain = 0.5
    succ = 0
    steps_per_trial = []
    for nt in range(num_trials):
        env.reset()
        goal = env.target_pos
        for curr_step in range(max_steps):
            current_pose = env.get_wrapper_attr("current_pose")
            velocity_desired = gain * (goal - current_pose)
            _, reward, done, _, _ = env.step(velocity_desired)

            if done:
                if reward > 100:
                    print("succesful!!")
                    succ += 1
                env.reset()
                steps_per_trial.append(curr_step)

                print(
                    "succ: ",
                    succ,
                    "num trials: ",
                    nt,
                    "avereage steps:",
                    np.mean(steps_per_trial),
                )

                break


if __name__ == "__main__":
    app()
