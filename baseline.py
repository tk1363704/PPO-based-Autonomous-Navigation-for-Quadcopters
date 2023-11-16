from pathlib import Path

import yaml

import typer
from typing_extensions import Annotated

import gymnasium
import numpy as np
from scripts.airsim_env import TrainConfig

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

    env.reset()
    gain = 1
    succ = 0
    for nt in range(100):
        env.reset()
        goal = env.target_pos
        for _ in range(200):
            # interface with the weird current actions
            velocity_desired = gain * (goal - env.get_wrapper_attr("current_pose"))
            velocity_vec = env.get_wrapper_attr("current_vel")
            velocuty_real = np.array(
                [velocity_vec.x_val, velocity_vec.y_val, velocity_vec.z_val]
            )
            delta_vel = velocity_desired - velocuty_real

            max_index = np.argmax(np.abs(delta_vel))
            if delta_vel[max_index] < 0:
                max_index += 3

            _, reward, done, _, _ = env.step(max_index)
            if done:
                if reward > 100:
                    print("succesful!!")
                    succ += 1
                env.reset()
                break
            print("max_index", max_index)
            print("delta_vel: ", delta_vel)
            print("Reward: ", reward)
            print("done: ", done)

            print("succ: ", succ, "num trials: ", nt)


if __name__ == "__main__":
    app()
