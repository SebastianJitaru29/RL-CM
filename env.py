import pybullet as pb
import pybullet_data
import time
import gymnasium as gym
import numpy as np
from ball import Ball

from panda_robot import PandaRobot

class Env(gym.Env):
    
    def __init__(
            self, 
            sampling_rate: float,
            reward_func: callable,
            real_time: bool=False
    ):
        """Initializes the Environment.
        
        @param sampling_rate (float): The sampling rate used for PyBullet.
        @param reward_func (callable): The used reward function, changable
            for curriculum learning. 
            Signature: func(joint_info, ball_info, goal_info) / func(state)
        @real_time (bool): Whether to run simulation in real time.
        """
        self.sampling_rate = sampling_rate
        self.physics_client = pb.connect(pb.GUI)
        pb.setGravity(0,0,-9.81)
        pb.setTimeStep(sampling_rate)
        self.panda_robot = PandaRobot(include_gripper=True)

        self.real_time = real_time
        
        # Add plane
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = pb.loadURDF("plane.urdf")

        # TODO create wrapper class as with pandarobot
        #      possible to make one robot baseclass
        position = [0.5,0.5,0.5]
        self.ball = Ball(position=position,scaling=0.2)
        
        # TODO get goal URDF
        # self.goal_id = pb.loadSDF("stadium.sdf")
        self.reward_func = reward_func
    

    def step(self, action):

        # TODO create controller
        torques = action
        # torques = self.controller(action)

        self.panda_robot.set_torques(torques)
        pb.stepSimulation()
        
        joint_pos, joint_vel = self.panda_robot.get_position_and_velocity()
        joint_array = np.array([joint_pos, joint_vel], dtype=float)

        # ee_pos = self.panda_robot.get_link_state(8)[0]

        ball_array = None   # TODO (pos(x, y, z), vel(x, y, z))
        goal_array = None   # TODO (pos(x, y, z), rot(z))
        
        reward = 0
        #reward = self.reward_func(joint_array, ball_array, goal_array)

        state = None         # TODO 

        # terminal = self.terminal_func()
        terminal = False

        if self.real_time:
            time.sleep(self.sampling_rate)

        return (
            state,
            reward, 
            terminal,   # terminated
            terminal,   # truncated
            {},         # empty info
        )
    
    def reset_random(self):
        #self.ball           # TODO
        #self.goal           # TODO
        self.panda_robot.reset_state()
        self.ball.reset_state([5,6,7])
        joint_pos, _ = self.panda_robot.get_position_and_velocity()

        return self.step(joint_pos)[0]     # state

    def reset(self, **kwargs):
        super().reset(**kwargs)
        return self.reset_random(), {}
    
    def set_realtime(self, real_time: bool):
        self.real_time = real_time

    def set_reward(self, func: callable):
        self.reward_func = func





    