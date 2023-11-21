from pathlib import Path
from enum import Enum

import yaml

import typer
from typing_extensions import Annotated

import gymnasium
import numpy as np
from scripts.airsim_env import TrainConfig

# import cv2


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
def velocity(
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

    state = BaselineState.MOVE_UP

    env.reset()
    # gain = 1
    succ = 0
    for nt in range(100):
        env.reset()
        goal = env.target_pos
        state = BaselineState.MOVE_UP
        for _ in range(200):
            # interface with the weird current actions
            if state == BaselineState.MOVE_UP:
                current = env.get_wrapper_attr("current_pose")
                goal_ = [current[0], current[1], 140]
            elif state == BaselineState.MOVE_ACROSS:
                goal_ = [goal[0], goal[1], 140]
            elif state == BaselineState.MOVE_DOWN:
                goal_ = goal

            # velocity_desired = gain * (goal_ - env.get_wrapper_attr("current_pose"))
            print("goal_", goal)
            # print("velocity_desired", velocity_desired)
            velocity_desired = np.array([0.0, 0.0, 100.0])
            if ((goal_ - env.get_wrapper_attr("current_pose")) ** 2).sum() < 5:
                state = state.succ()

            _, reward, done, _, _ = env.step(velocity_desired)
            # img = env.get_depth_image() / 2.0
            # cv2.imshow("depth_image", img)
            # cv2.waitKey(1)
            # input()
            if done:
                if reward > 100:
                    print("succesful!!")
                    succ += 1
                env.reset()
                break
            print("delta_vel: ", velocity_desired)
            print("Reward: ", reward)
            print("done: ", done)
            print("succ: ", succ, "num trials: ", nt)


if __name__ == "__main__":
    app()
