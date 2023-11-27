from typing import Any, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import gymnasium as gym
import numpy as np
import binvox

from . import airsim

from .util import create_voxel_grid


@dataclass
class TrainConfig:
    binvox: Path

    def __post_init__(self):
        self.binvox = Path(self.binvox)


class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS_VELOCITY = 1


class AirSimDroneEnv(gym.Env):
    def __init__(
        self,
        ip_address: str,
        env_config: TrainConfig,
        action_type: ActionType,
        sim_dt: float,
    ):
        self.drone = airsim.MultirotorClient(ip=ip_address)
        rgb_shape = self.get_rgb_image().shape

        self.drone.simPause(True)

        voxel_path = env_config.binvox

        if not voxel_path.exists():
            create_voxel_grid(voxel_path)

        self.voxel = binvox.Binvox.read(str(voxel_path.absolute()), "dense")

        self.observation_space = gym.spaces.Dict(
            {
                "rgb_image": gym.spaces.Box(
                    low=0, high=255, shape=rgb_shape, dtype=np.uint8
                ),
                "goal": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64),
                    high=np.array([np.inf, np.inf, np.inf], dtype=np.float64),
                ),
                "target": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64),
                    high=np.array([np.inf, np.inf, np.inf], dtype=np.float64),
                ),
                "relative_speed": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64),
                    high=np.array([np.inf, np.inf, np.inf], dtype=np.float64),
                ),
            }
        )

        self.action_type = action_type
        self.sim_dt = sim_dt
        if self.action_type == ActionType.DISCRETE:
            self.action_space = gym.spaces.Discrete(7)
        elif self.action_type == ActionType.CONTINUOUS_VELOCITY:
            self.action_space = gym.spaces.Box(
                np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )
        else:
            raise ValueError("Invalid Action Type!")

        self.random_start = True
        self.setup_flight()

        self.steps = 0
        self.target_dist_prev = 0.0
        self.collision_time = -1

    def sample_start_goal_pos(self):
        indx = np.where(self.voxel.numpy() == 0)

        a = np.random.randint(len(indx[0]))
        start_x, start_y, start_z = (arr[a] for arr in indx)

        print(f"x : {start_x}, y: {start_y}")

        start_pos = np.array(self.voxel.transform_coord((start_x, start_y, start_z)))

        relative_pos = np.random.uniform(low=2.0, high=4.0, size=3)
        goal_pos = start_pos + relative_pos

        return start_pos, goal_pos

    def step(self, action):
        self.drone.simPause(False)

        if self.action_type == ActionType.DISCRETE:
            self.do_action_moving_x(action)
        elif self.action_type == ActionType.CONTINUOUS_VELOCITY:
            self.set_velocity(action)
        else:
            raise ValueError("Invalid Action Type!")

        self.drone.simPause(True)

        obs, info = self.get_obs()
        reward, done = self.compute_reward()
        truncated = self.steps > 200
        return obs, reward, done, truncated, info

    def set_velocity(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.drone.moveByVelocityAsync(
            *action,
            duration=self.sim_dt,
        ).join()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        self.setup_flight()
        obs, _ = self.get_obs()
        return obs, {}

    def render(self):
        return self.get_obs()

    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)

        # Arming a drone means preparing it for flight.
        self.drone.armDisarm(True)

        # Prevent drone from falling after reset
        self.drone.moveToZAsync(-1, 1)

        # Get collision time stamp
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp

        self.agent_start_pos, self.target_pos = self.sample_start_goal_pos()

        print("-----------A new flight!------------")
        print(f"Start point is {self.agent_start_pos}")
        print(f"Target point is {self.target_pos}")
        print("-----------Start flying!------------")
        self.steps = 0

        pose = airsim.Pose(airsim.Vector3r(*self.agent_start_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        self.drone.simPause(False)
        self.drone.simPause(True)

        self.target_dist_prev = np.linalg.norm(self.agent_start_pos - self.target_pos)
        print(f"target_dist_prev: {self.target_dist_prev}")

    def do_action(self, select_action):
        speed = 0.4
        if select_action == 0:
            vy, vz = (-speed, -speed)
        elif select_action == 1:
            vy, vz = (0, -speed)
        elif select_action == 2:
            vy, vz = (speed, -speed)
        elif select_action == 3:
            vy, vz = (-speed, 0)
        elif select_action == 4:
            vy, vz = (0, 0)
        elif select_action == 5:
            vy, vz = (speed, 0)
        elif select_action == 6:
            vy, vz = (-speed, speed)
        elif select_action == 7:
            vy, vz = (0, speed)
        else:
            vy, vz = (speed, speed)

        self.drone.moveByVelocityBodyFrameAsync(
            speed, vy, vz, duration=self.sim_dt
        ).join()
        self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1)

    def do_action_moving_x(self, select_action):
        speed = 0.4
        if select_action == 0:
            quad_offset = (speed, 0, 0)
        elif select_action == 1:
            quad_offset = (0, speed, 0)
        elif select_action == 2:
            quad_offset = (0, 0, speed)
        elif select_action == 3:
            quad_offset = (-speed, 0, 0)
        elif select_action == 4:
            quad_offset = (0, -speed, 0)
        elif select_action == 5:
            quad_offset = (0, 0, -speed)
        else:
            quad_offset = (0, 0, 0)

        new_vel = self.current_vel + quad_offset

        self.drone.moveByVelocityBodyFrameAsync(
            *new_vel,
            duration=self.sim_dt,
        ).join()

        # self.drone.step()

    def move_to_pos(self, goal):
        self.drone.moveToPositionAsync(*goal, velocity=1.5).join()

    def get_obs(self):
        info = {"collision": self.is_collision()}
        obs = {
            "rgb_image": self.get_rgb_image(),
            "goal": self.target_pos,
            "target": self.current_pose,
            "relative_speed": self.current_vel,
        }
        return obs, info

    @property
    def current_pose(self):
        x, y, z = self.drone.simGetVehiclePose().position
        return np.array([x, y, z], dtype=np.float64)

    @property
    def current_vel(self):
        vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val])

    def compute_reward(self) -> Tuple[float, bool]:
        reward = 0.0
        done = False
        self.steps += 1

        # Target distance based reward
        target_dist_curr = float(np.linalg.norm(self.current_pose - self.target_pos))
        print("target_dist_curr: ", target_dist_curr)
        reward += (self.target_dist_prev - target_dist_curr) * 20

        self.target_dist_prev = target_dist_curr
        if self.steps % 10 == 0:
            print(f"Steps {self.steps} -> target_dist_prev: {self.target_dist_prev}")

        # Alignment reward
        if target_dist_curr < 0.30:
            reward += 12

        elif target_dist_curr < 0.45:
            reward += 7

        # Collision penalty
        if self.is_collision():
            print("The drone has collided with the obstacle!!!")
            reward = -100.0
            done = True

        elif target_dist_curr < 0.087:
            print("The drone has reached the target!!!")
            reward += 100
            done = True

        # Check if the drone's altitude is less than the landing threshold
        elif self.is_landing():
            print("Drone has touched the ground!!!")
            reward = -100.0
            done = True

        if done == 1 or self.steps % 10 == 0:
            print(f"Steps {self.steps} -> reward: {reward}, done: {done}")
        return reward, done

    def is_landing(self):
        # Set a threshold for how close the drone should be to the ground
        # to consider it landed
        landing_threshold = -0.1  # You may need to adjust this value
        state = self.drone.getMultirotorState()
        position = state.kinematics_estimated.position
        return position.z_val > landing_threshold

    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        if current_collision_time != self.collision_time:
            flag = True
            self.collision_time = self.drone.simGetCollisionInfo().time_stamp
        else:
            flag = False

        return flag

    def is_collided(self):
        flag = self.drone.simGetCollisionInfo().has_collided
        if flag:
            print("collided!!!!")
        return flag

    def get_rgb_image(self) -> np.ndarray:
        rgb_image_request = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.fromstring(  # type: ignore
            responses[0].image_data_uint8, dtype=np.uint8
        )
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))
        img_rgb = np.flipud(img2d)
        return img_rgb

    def get_depth_image(self, thresh=2.0) -> np.ndarray:
        depth_image_request = airsim.ImageRequest(
            1, airsim.ImageType.DepthPerspective, True, False
        )
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float64)
        depth_image = np.reshape(depth_image, (responses[0].height, responses[0].width))
        depth_image[depth_image > thresh] = thresh
        return depth_image
