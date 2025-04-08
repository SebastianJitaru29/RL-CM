import pybullet as pb
import pybullet_data
import time
import gymnasium as gym
import numpy as np
from ball import Ball

from panda_robot import PandaRobot
from goal_post import GoalPost
from pd_grav import PDGravController

from typing import List, Callable

class Env(gym.Env):
    
    def __init__(
            self, 
            sampling_rate: float,
            reward_func: Callable,
            max_steps: int=100,
            *,
            steps_per_step: int = 50,
            real_time: bool=False
    ):
        """Initializes the Environment.
        
        @param sampling_rate (float): The sampling rate used for PyBullet.
        @param reward_func (Callable): The used reward function
            Signature: func(state_dict, terminal)
        @param max_steps (int): maximum number of agent steps.
        @param steps_per_step (int): physics step per agent step.
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
        self.reward_func = reward_func if reward_func is not None else lambda x, y: 0
        self.cur_step = 0
        self.score_step = -1
        self.max_steps = max_steps
        self.physics_steps = range(steps_per_step)


    def step(self, action: np.ndarray):
        """Taking one environment step at provided sampling rate.

        @param action (np.ndarry): The action, target joint positions.
        """
        self.controller.set_joint_desired(action)

        # TODO make some extra steps, like 10 or 50 or smth
        for _ in self.physics_steps:
            self.controller.step()
            pb.stepSimulation()
            ball_pos, ball_vel = self.ball.get_position_and_velocity()

            if self.real_time:
                time.sleep(self.sampling_rate)
            
            # If we haven't scored, check scoring
            if self.score_step == -1:
                if self.goal.get_score(ball_pos):
                    self.score_step = self.cur_step
        
        joint_pos, joint_vel = self.panda_robot.get_position_and_velocity()
        joint_array = np.array([joint_pos, joint_vel], dtype=float)

        ee_state = self.panda_robot.get_link_state(8)
        ee_pos = ee_state[0]
        ee_vel = ee_state[-2]
        ee_array = np.array([ee_pos, ee_vel], dtype=float)

        ball_array = np.array([ball_pos, ball_vel], dtype=float)

        goal_pos = np.array([0, 3, 0])   # TODO (pos(x, y, z), rot(z))

        # TODO incorperate time by appending previous states?
        # state = (joint_array, ee_array, ball_array, goal_pos)
        
        # The ball was in the goal at some point
        terminal = self.score_step != -1 or self.cur_step >= self.max_steps

        state_dict = {
            'joint_info': joint_array,
            'ee_info': ee_array,
            'ball_info': ball_array,
            'goal_info': goal_pos,
            'score': self.score_step != -1,
        }

        reward = self.reward_func(state_dict, terminal)
        self.cur_step += 1

        return (
            state_dict,
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



    