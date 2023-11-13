from pathlib import Path

import yaml

import typer
from typing_extensions import Annotated

import gymnasium
import numpy as np

current_file_path = Path(__file__).parent

app = typer.Typer()


@app.command()
def main(
    config_file: Annotated[Path, typer.Option()] = Path(
        current_file_path / "scripts/config.yml"
    ),
    sim_ip: Annotated[str, typer.Option()] = "127.0.0.1",
):
    # Get train environment configs
    with open(config_file, "r", encoding="utf8") as f:
        env_config = yaml.safe_load(f)

    env = gymnasium.make(
        "scripts:airsim-env-v0",
        ip_address=sim_ip,
        image_shape=(50, 50, 3),
        env_config=env_config["TrainEnv"],
    )

    # if select_action == 0:
    #     quad_offset = (speed, 0, 0)
    #     # print('Left!!!')
    # elif select_action == 1:
    #     quad_offset = (0, speed, 0)
    #     # print('Forward!!!')
    # elif select_action == 2:
    #     quad_offset = (0, 0, speed)
    #     # print('Down!!!')
    # elif select_action == 3:
    #     quad_offset = (-speed, 0, 0)
    #     # print('Right!!!')
    # elif select_action == 4:
    #     quad_offset = (0, -speed, 0)
    #     # print('Back!!!')
    # elif select_action == 5:
    #     quad_offset = (0, 0, -speed)
    #     # print('Up!!!')
    # else:
    #     quad_offset = (0, 0, 0)

    env.reset()
    goal = env.target_pos
    gain = 1
    succ = 0
    for _ in range(100):
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

            print("succ: ", succ)


if __name__ == "__main__":
    app()
