import datetime

from . import airsim
import gym
import numpy as np

class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address, image_shape, env_config):
        self.image_shape = image_shape
        self.sections = env_config["sections"]

        self.drone = airsim.MultirotorClient(ip=ip_address)
        # gym.spaces.Box is a class provided by the OpenAI Gym library that represents a continuous space in a reinforcement learning environment. In RL, a "space" defines the possible values that state and action variables can take. The Box space is used when these variables can take on any real-valued number within a specified range.
        # The space is image whose shape is (50, 50, 3) and value range is [0, 255];
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        # For instance, if you were working with a grid-world environment, these nine discrete actions might correspond to moving in different directions (e.g., up, down, left, right, or diagonally) or taking specific actions within the environment.
        # self.action_space = gym.spaces.Discrete(9)
        self.action_space = gym.spaces.Discrete(7)

        self.info = {"collision": False}

        self.collision_time = 0
        self.random_start = True
        self.setup_flight()
        self.steps = 0

    def step(self, action):
        # self.do_action(action)
        self.do_action_moving_x(action)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()
        return obs, reward, done, info

    def reset(self):
        self.setup_flight()
        obs, _ = self.get_obs()
        return obs

    def render(self):
        return self.get_obs()

    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        # Arming a drone means preparing it for flight. When you arm a drone, you enable its propulsion system, allowing it to generate thrust and lift off the ground.
        # Disarming a drone means shutting down its propulsion system and disabling its ability to generate thrust. This is typically done when you want to power down or land the drone safely.
        # True to arm, False to disarm the vehicle
        self.drone.armDisarm(True)

        # Prevent drone from falling after reset
        self.drone.moveToZAsync(-1, 1)

        # Get collision time stamp
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp

        # Get a random section
        if self.random_start == True:
            self.target_pos_idx = np.random.randint(len(self.sections))
        else:
            self.target_pos_idx = 0

        section = self.sections[self.target_pos_idx]
        self.agent_start_pos = section["offset"]
        self.target_pos = section["target"]

        print('-----------A new flight!------------')
        print('Start point is {}'.format(self.agent_start_pos))
        print('Target point is {}'.format(self.target_pos))
        print('-----------Start flying!------------')
        self.steps = 0
        start_x_pos, start_y_pos, start_z_pos = self.agent_start_pos[0],self.agent_start_pos[1],self.agent_start_pos[2]

        # # Start the agent at random section at a random yz position
        # y_pos, z_pos = ((np.random.rand(1,2)-0.5)*2).squeeze()
        # airsim.Pose: This is a class provided by the AirSim library for defining the pose of an object. A pose typically includes information about its position and orientation.
        # airsim.Vector3r(self.agent_start_pos, y_pos, z_pos): This part creates a Vector3r object, which represents a 3D vector. It's used to specify the position of the object. self.agent_start_pos is likely a variable or value representing the x-coordinate, y_pos is the y-coordinate, and z_pos is the z-coordinate.
        pose = airsim.Pose(airsim.Vector3r(start_x_pos, start_y_pos, start_z_pos))
        # Set the pose of the vehicle
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=False)
        
        # Get target distance for reward calculation
        # This line of code calculates the Euclidean distance between two 2D points: [y_pos, z_pos] and self.target_pos
        # self.target_dist_prev: This variable is assigned the computed distance value. It seems to be used to store the previous distance between the two points, possibly for tracking changes in distance over time.
        self.target_dist_prev = np.linalg.norm(np.array([start_x_pos, start_y_pos, start_z_pos]) - self.target_pos)
        print('target_dist_prev: {}'.format(self.target_dist_prev))

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

        # Execute action
        # vx (float): desired velocity in the X axis of the vehicle's local NED frame.
        # vy (float): desired velocity in the Y axis of the vehicle's local NED frame.
        # vz (float): desired velocity in the Z axis of the vehicle's local NED frame.
        # duration (float): Desired amount of time (seconds), to send this command for
        # call .join() to wait for method to finish.
        self.drone.moveByVelocityBodyFrameAsync(speed, vy, vz, duration=1).join()
        # Prevent swaying
        # If you want to control the vehicle's velocity in its own body frame, use moveByVelocityBodyFrameAsync. If you prefer to control the velocity in a global reference frame (e.g., for navigation or waypoint following), use moveByVelocityAsync.
        self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1)

    def do_action_moving_x(self, select_action):
        speed = 0.4
        if select_action == 0:
            quad_offset = (speed, 0, 0)
            # print('Left!!!')
        elif select_action == 1:
            quad_offset = (0, speed, 0)
            # print('Forward!!!')
        elif select_action == 2:
            quad_offset = (0, 0, speed)
            # print('Down!!!')
        elif select_action == 3:
            quad_offset = (-speed, 0, 0)
            # print('Right!!!')
        elif select_action == 4:
            quad_offset = (0, -speed, 0)
            # print('Back!!!')
        elif select_action == 5:
            quad_offset = (0, 0, -speed)
            # print('Up!!!')
        else:
            quad_offset = (0, 0, 0)

        # # Execute action
        # # vx (float): desired velocity in the X axis of the vehicle's local NED frame.
        # # vy (float): desired velocity in the Y axis of the vehicle's local NED frame.
        # # vz (float): desired velocity in the Z axis of the vehicle's local NED frame.
        # # duration (float): Desired amount of time (seconds), to send this command for
        # # call .join() to wait for method to finish.
        # self.drone.moveByVelocityBodyFrameAsync(quad_offset[0], quad_offset[1], quad_offset[2], duration=1).join()
        # # Prevent swaying
        # # If you want to control the vehicle's velocity in its own body frame, use moveByVelocityBodyFrameAsync. If you prefer to control the velocity in a global reference frame (e.g., for navigation or waypoint following), use moveByVelocityAsync.
        # self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1)

        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityBodyFrameAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            duration=3
        ).join()

    def get_obs(self):
        self.info["collision"] = self.is_collision()
        obs = self.get_rgb_image()
        return obs, self.info

    def compute_reward(self):
        reward = 0
        done = 0
        self.steps += 1

        # Target distance based reward
        x,y,z = self.drone.simGetVehiclePose().position
        target_dist_curr = np.linalg.norm(np.array([x, y, z]) - self.target_pos)
        reward += (self.target_dist_prev - target_dist_curr)*20

        self.target_dist_prev = target_dist_curr
        if self.steps % 10 == 0:
            print('Steps {} -> target_dist_prev: {}'.format(self.steps, self.target_dist_prev))

        # # Get meters agent traveled
        # agent_traveled_x = np.abs(self.agent_start_pos - x)

        # Alignment reward
        if target_dist_curr < 0.30:
            reward += 12
            # # Alignment becomes more important when agent is close to the hole
            # if agent_traveled_x > 2.9:
            #     reward += 7

        elif target_dist_curr < 0.45:
            reward += 7

        # Collision penalty
        if self.is_collision():
            print("The drone has collided with the obstacle!!!")
            reward = -100
            done = 1

        # # Check if agent passed through the hole
        # elif agent_traveled_x > 3.7:
        #     reward += 10
        #     done = 1

        elif target_dist_curr < 0.87:
            print("The drone has reached the target!!!")
            reward += 100
            done = 1

        # Check if the drone's altitude is less than the landing threshold
        elif self.is_landing():
            print("Drone has touched the ground!!!")
            reward = -100
            done = 1

        elif target_dist_curr >= 50:
            print("The drone has flown out of the specified range!!!")
            reward += -50
            done = 1

        # Check if the hole disappeared from camera frame
        # (target_dist_curr-0.3) : distance between agent and hole's end point
        # (3.7-agent_traveled_x) : distance between agent and wall
        # (3.7-agent_traveled_x)*sin(60) : end points that camera can capture
        # FOV : 120 deg, sin(60) ~ 1.732
        # The condition being checked is (target_dist_curr-0.3) > (3.7-agent_traveled_x)*1.732. It's comparing the distance between the agent and the hole's end point with the distance between the agent and the wall scaled by the FOV. If the condition is true, it means that the hole has "disappeared" from the camera frame because it's farther away than the camera's FOV can capture.
        # elif (target_dist_curr-0.3) > (3.7-agent_traveled_x)*1.732:
        #     reward = -100
        #     done = 1
        if done == 1 or self.steps % 10 == 0:
            print('Steps {} -> reward: {}, done: {}'.format(self.steps, reward, done))
        return reward, done

    def is_landing(self):
        # Set a threshold for how close the drone should be to the ground to consider it landed
        landing_threshold = -0.1  # You may need to adjust this value
        state = self.drone.getMultirotorState()
        position = state.kinematics_estimated.position
        if position.z_val > landing_threshold:
            return True
        else:
            return False

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
            print('collided!!!!')
        return flag
    
    def get_rgb_image(self):
        rgb_image_request = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
        # # camera control
        # # simGetImage returns compressed png in array of bytes
        # # image_type uses one of the ImageType members
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3)) 

        # Sometimes no image returns from api
        try:
            return img2d.reshape(self.image_shape)
        except:
            return np.zeros((self.image_shape))

    def get_depth_image(self, thresh = 2.0):
        depth_image_request = airsim.ImageRequest(1, airsim.ImageType.DepthPerspective, True, False)
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_image = np.reshape(depth_image, (responses[0].height, responses[0].width))
        depth_image[depth_image>thresh]=thresh
        return depth_image


class TestEnv(AirSimDroneEnv):
    def __init__(self, ip_address, image_shape, env_config):
        self.eps_n = 0
        super(TestEnv, self).__init__(ip_address, image_shape, env_config)
        self.agent_traveled = []
        self.random_start = False

    def setup_flight(self):
        super(TestEnv, self).setup_flight()
        self.eps_n += 1

        # Start the agent at a random yz position
        y_pos, z_pos = (0,0)
        pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos,y_pos,z_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        
    def compute_reward(self):
        reward = 0
        done = 0

        x,_,_ = self.drone.simGetVehiclePose().position

        if self.is_collision():
            done = 1
            self.agent_traveled.append(x)
    
        if done and self.eps_n % 5 == 0:
            print("---------------------------------")
            print("> Total episodes:", self.eps_n)
            print("> Flight distance (mean): %.2f" % (np.mean(self.agent_traveled)))
            print("> Holes reached (max):", int(np.max(self.agent_traveled)//4))
            print("> Holes reached (mean):", int(np.mean(self.agent_traveled)//4))
            print("---------------------------------\n")

        return reward, done
