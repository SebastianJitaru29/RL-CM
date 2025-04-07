import pybullet as pb
import pybullet_data
import time
import gymnasium as gym
import numpy as np
from ball import Ball

from panda_robot import PandaRobot
from goal_post import GoalPost
from pd_grav import PDGravController

from typing import List

class Env(gym.Env):
    
    def __init__(
            self, 
            sampling_rate: float,
            reward_func: callable,
            max_steps: int=5000,
            *,
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

        self.controller = PDGravController(
            self.panda_robot,
            [200, 200, 200, 200, 250, 50, 20, 0, 0],
            [30, 30, 30, 30, 10, 10, 5, 0, 0],
        )

        self.real_time = real_time
        
        # Add plane
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = pb.loadURDF("plane.urdf")

        # TODO create wrapper class as with pandarobot
        #      possible to make one robot baseclass
        position = [0.5,0.5,0.5]
        self.ball = Ball(position=position, scaling=0.2)
        self.goal = GoalPost([0, 3, 0], 1)
        
        # TODO get goal URDF
        # self.goal_id = pb.loadSDF("stadium.sdf")
        self.reward_func = reward_func if reward_func is not None else lambda x: 0
        self.cur_step = 0
        self.score_step = 0
        self.max_steps = max_steps


    def step(self, action: np.ndarray):
        """Taking one environment step at provided sampling rate.

        @param action (np.ndarry): The action, target joint positions.
        """
        # TODO create controller
        self.controller.set_joint_desired(action)
        self.controller.step()

        pb.stepSimulation()
        
        joint_pos, joint_vel = self.panda_robot.get_position_and_velocity()
        joint_array = np.array([joint_pos, joint_vel], dtype=float)

        ee_state = self.panda_robot.get_link_state(8)
        ee_pos = ee_state[0]
        ee_vel = ee_state[-2]
        ee_array = np.array([ee_pos, ee_vel], dtype=float)

        ball_pos, ball_vel = self.ball.get_position_and_velocity()
        ball_array = np.array([ball_pos, ball_vel], dtype=float)

        goal_array = np.zeros([2, 0, 0])   # TODO (pos(x, y, z), rot(z))

        # TODO incorperate time by appending previous states?
        state = (joint_array, ee_array, ball_array, goal_array)

        score = False
        # If we haven't scored, check scoring
        if self.score_step == -1:
            if self.goal.get_score(ball_pos):
                score = True
                self.score_step = self.cur_step
        
        # The ball was in the goal at some point
        terminal = self.score_step != -1 or self.cur_step >= self.max_steps

        if self.real_time:
            time.sleep(self.sampling_rate)
            if self.score_step != -1 and self.cur_step - self.score_step < 500:
                terminal = False

        state_dict = {
            'joint_info': joint_array,
            'ee_info': ee_array,
            'ball_info': ball_array,
            'goal_info': goal_array,
            'score': score,
        }

        reward = self.reward_func(state_dict)
        self.cur_step += 1

        return (
            state,
            reward, 
            terminal,   # terminated
            terminal,   # truncated
            {},         # empty info
        )
    
    def reset_random(self):
        self.cur_step = 0
        self.score_step = -1
        self.panda_robot.reset_state()
        self.ball.reset_state(self._random_ball_pos())
        #self.goal.reset_state([0, 2, 0], [0, 0, 0, 1])
        joint_pos, _ = self.panda_robot.get_position_and_velocity()

        return self.step(joint_pos)[0]     # state

    def reset(self, **kwargs):
        super().reset(**kwargs)
        return self.reset_random(), {}
    
    def set_realtime(self, real_time: bool):
        self.real_time = real_time

    def set_reward(self, func: callable):
        self.reward_func = func

    def _random_ball_pos(self):
        dist_min = 0.3
        dist_max = 0.7
        
        x_axis = 2 * (dist_max - 0.1) * np.random.random() - (dist_max - 0.1)
        
        y_min = np.sqrt(dist_min ** 2 - x_axis ** 2) if dist_min > np.abs(x_axis) else 0
        y_max = np.sqrt(dist_max ** 2 - x_axis ** 2)

        y_axis = (y_max - y_min) * np.random.random() + y_min
        
        return [x_axis, y_axis, 0.1]



    