from typing import Any, List, Tuple
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from collections import OrderedDict
import os
from . import airsim
from scripts import binvox_rw
import random
from scipy.spatial.transform import Rotation as R
import pprint

@dataclass
class Section:
    target: List[float]
    offset: List[float]


@dataclass
class TrainConfig:
    sections: List[Section]

    def __post_init__(self):
        if isinstance(self.sections[0], dict):
            self.sections = [Section(**sec) for sec in self.sections]


class AirSimDroneEnv(gym.Env):
    def __init__(
        self,
        ip_address: str,
        env_config: TrainConfig,
    ):
        self.sections = env_config.sections

        # Run this for the first time to create a voxel grid map. TODO: If map doesn't exist, run this
        # self.create_voxel_grid()

        with open("/home/nick/Dev/AirSim/PPO-based-Autonomous-Navigation-for-Quadcopters/map2.binvox", 'rb') as f:
            self.map = binvox_rw.read_as_3d_array(f)        

        self.drone = airsim.MultirotorClient(ip=ip_address)
        rgb_shape = self.get_rgb_image().shape
        # gym.spaces.Box is a class provided by the OpenAI Gym library that
        # represents a continuous space in a reinforcement learning environment.
        # In RL, a "space" defines the possible values that state and action
        # variables can take. The Box space is used when these variables can
        # take on any real-valued number within a specified range.

        # The space is image whose shape is (50, 50, 3) and value range is
        # [0, 255];
        observation_space = OrderedDict()
        observation_space["goal_obs"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        observation_space["image_obs"] = gym.spaces.Box(low=0, high=255, shape=rgb_shape, dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(observation_space)

        # For instance, if you were working with a grid-world environment, these
        # nine discrete actions might correspond to moving in different
        # directions (e.g., up, down, left, right, or diagonally) or taking
        # specific actions within the environment.
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.random_start = True
        #self.setup_flight()
        self.steps = 0
        self.target_dist_prev = 0.0
        self.collision_time = -1

    def create_voxel_grid(self):

        client = airsim.VehicleClient()
        center = airsim.Vector3r(0, 0, 0)
        voxel_size = 100
        res = 1
        output_path = os.path.join(os.getcwd(), "map.binvox")
        client.simCreateVoxelGrid(center, 100, 100, 50, res, output_path)
        print("voxel map generated!")

        with open("/home/nick/Dev/AirSim/PPO-based-Autonomous-Navigation-for-Quadcopters/map.binvox", 'rb') as f:
            map = binvox_rw.read_as_3d_array(f)   
        # Set every below ground level as "occupied". #TODO: add inflation to the map
        map.data[:,:,:26] = True
        with open("/home/nick/Dev/AirSim/PPO-based-Autonomous-Navigation-for-Quadcopters/map_edited.binvox", 'wb') as f:
            binvox_rw.write(map, f)

    def sample_start_goal_pos(self):

        indx = np.where(self.map.data == 0)

        a = random.randint(0 , len(indx[0]))
        start_x = indx[0][a]
        start_y = indx[1][a]
        start_z = indx[2][a]
        print(f"x : {start_x}, y: {start_y}")

        start_pos = [start_y + self.map.translate[0] , start_x + self.map.translate[1], abs(self.map.translate[2]) - start_z]
        
        # Set relative position and orientation wrt to the start, not 100% correct but can't be bothered
        x = [-1,1][random.randrange(2)]
        y = [-1,1][random.randrange(2)]
        z = [-1,1][random.randrange(2)]
        relative_pos = [x*random.uniform(2, 4),  y*random.uniform(2, 4), z*random.uniform(1, 2)] # TODO: need to sample a collision free pos from the map
        goal_pos = [x + y for x, y in zip(start_pos, relative_pos)]
        
        asset_name = 'Sphere'
        
        scale = airsim.Vector3r(0.1, 0.1, 0.1)
        desired_name = f"{asset_name}_spawn_{random.randint(0, 100)}"
        pose = airsim.Pose(position_val=airsim.Vector3r(goal_pos[0], goal_pos[1], goal_pos[2]))

        # obj_name = self.drone.simSpawnObject(desired_name, asset_name, pose, scale, False)

        # print(f"Created object {obj_name} from asset {asset_name} "
        #     f"at pose {pose}, scale {scale}")

        return start_pos, goal_pos

    def step(self, action):
        self.do_action(action)
        obs, info = self.get_obs()
        truncated = self.steps > 200
        reward, done, info = self.compute_reward(info)
        
        return obs, reward, done, truncated, info

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
        # Arming a drone means preparing it for flight. When you arm a drone,
        # you enable its propulsion system, allowing it to generate thrust and
        # lift off the ground.
        # Disarming a drone means shutting down its propulsion system and
        # disabling its ability to generate thrust. This is typically done when
        # you want to power down or land the drone safely.
        # True to arm, False to disarm the vehicle
        self.drone.armDisarm(True)

        # Prevent drone from falling after reset
        #self.drone.takeoffAsync().join()
        self.drone.moveToZAsync(-1, 1)

        # Get collision time stamp
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp
        
        start_pos, goal_pos = self.sample_start_goal_pos()

        self.agent_start_pos = np.array(start_pos, dtype=np.float64)
        self.target_pos = np.array(goal_pos, dtype=np.float64)

        print("-----------A new flight!------------")
        print(f"Start point is {self.agent_start_pos}")
        print(f"Target point is {self.target_pos}")
        print("-----------Start flying!------------")
        self.steps = 0
        start_x_pos, start_y_pos, start_z_pos = (
            float(self.agent_start_pos[0]),
            float(self.agent_start_pos[1]),
            float(self.agent_start_pos[2]),
        )

        
        # # Start the agent at random section at a random yz position
        # y_pos, z_pos = ((np.random.rand(1,2)-0.5)*2).squeeze()
        # airsim.Pose: This is a class provided by the AirSim library for
        # defining the pose of an object. A pose typically includes information
        # about its position and orientation.
        # airsim.Vector3r(self.agent_start_pos, y_pos, z_pos): This part creates
        # a Vector3r object, which represents a 3D vector. It's used to specify
        # the position of the object. self.agent_start_pos is likely a variable
        # or value representing the x-coordinate, y_pos is the y-coordinate,
        # and z_pos is the z-coordinate.
        pose = airsim.Pose(airsim.Vector3r(*self.agent_start_pos))
        # Set the pose of the vehicle
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        
        self.drone.moveToZAsync(start_z_pos, 10).join()

        #self.drone.moveToPositionAsync(start_x_pos, start_y_pos, start_z_pos, 5).join()
        # Get target distance for reward calculation
        # This line of code calculates the Euclidean distance between two
        # 2D points: [y_pos, z_pos] and self.target_pos
        # self.target_dist_prev: This variable is assigned the computed distance
        # value. It seems to be used to store the previous distance between the
        # two points, possibly for tracking changes in distance over time.
        self.target_dist_prev = np.linalg.norm(self.agent_start_pos - self.target_pos)
        print(f"target_dist_prev: {self.target_dist_prev}")

    def do_action(self, action):

        # Execute action
        print(action)
        self.drone.moveByVelocityBodyFrameAsync(float(action[0]), float(action[1]), float(action[2]), duration=0.1).join()

    def move_to_pos(self, goal):
        self.drone.moveToPositionAsync(*goal, velocity=1.5).join()

    def get_obs(self):
        info = {"collision": self.is_collision()}

        obs = OrderedDict()
        obs["image_obs"] = self.get_rgb_image()
        goal_obs = self.global_to_local(self.target_pos)
        obs["goal_obs"] = goal_obs

        return obs, info

    @property
    def current_vel(self):
        vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val])

    def compute_reward(self, info) -> Tuple[float, bool]:
        reward = 0.0
        done = False
        self.steps += 1
        info["is_success"] = False
        info["is_collision"] = False
        info["timeout"] = False


        drone_pose = self.drone.simGetVehiclePose()
        drone_pos = np.array([drone_pose.position.x_val, drone_pose.position.y_val, drone_pose.position.z_val])

        # Target distance based reward
        potential_reward_weight = 0.20 # TODO: add in config file
        target_dist_curr = float(np.linalg.norm(self.target_pos - drone_pos))
        #print("target_dist_curr: ", target_dist_curr)
        reward += (self.target_dist_prev - target_dist_curr) * potential_reward_weight

        self.target_dist_prev = target_dist_curr

        # Goal reward
        goal_threshold = 0.30
        if target_dist_curr < goal_threshold:
            reward += 1
            done = True
            info["is_success"] = True

        # Timestep reward
        reward += -0.005

        # Collision penalty
        if self.is_collision():
            print("The drone has collided with the obstacle!!!")
            reward += -1
            info["is_collision"] = True
            done = True
        elif self.is_landing():
            # Check if the drone's altitude is less than the landing threshold
            print("Drone has touched the ground!!!")
            reward += -1
            done = True
        elif target_dist_curr >= 50:
            print("The drone has flown out of the specified range!!!")
            reward += -1
            done = True
        elif self.steps > 100:
            info["is_timeout"] = True
            reward += -1
            done = True

        if done == True or self.steps % 10 == 0:
            print(f"Steps {self.steps} -> reward: {reward}, done: {done}")
        
        return reward, done, info

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
        # # camera control
        # # simGetImage returns compressed png in array of bytes
        # # image_type uses one of the ImageType members
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.fromstring(  # type: ignore
            responses[0].image_data_uint8, dtype=np.uint8
        )
        try:
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))
        except:
            np.zeros((144,256,3))

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
    
    def global_to_local(self, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        drone_pose = self.drone.simGetVehiclePose()
        drone_pos = np.array([drone_pose.position.x_val, drone_pose.position.y_val, drone_pose.position.z_val])
        v = np.array(pos) - drone_pos

        local_to_global = R.from_quat([drone_pose.orientation.x_val, drone_pose.orientation.y_val, 
            drone_pose.orientation.z_val, drone_pose.orientation.w_val]).as_matrix()
        global_to_local = local_to_global.T
        return np.dot(global_to_local, v)

class TestEnv(AirSimDroneEnv):
    def __init__(self, ip_address, env_config):
        self.eps_n = 0
        super().__init__(ip_address, env_config)
        self.agent_traveled = []
        self.random_start = False

    def setup_flight(self):
        super().setup_flight()
        self.eps_n += 1

        # Start the agent at a random yz position
        # y_pos, z_pos = (0, 0)
        pose = airsim.Pose(airsim.Vector3r(*self.agent_start_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)

    def compute_reward(self):
        reward = 0
        done = 0

        x, _, _ = self.current_pose

        if self.is_collision():
            done = 1
            self.agent_traveled.append(x)

        if done and self.eps_n % 5 == 0:
            print("---------------------------------")
            print("> Total episodes:", self.eps_n)
            print(f"> Flight distance (mean): {np.mean(self.agent_traveled):.2f}")
            print("> Holes reached (max):", int(np.max(self.agent_traveled) // 4))
            print("> Holes reached (mean):", int(np.mean(self.agent_traveled) // 4))
            print("---------------------------------\n")

        return reward, done
