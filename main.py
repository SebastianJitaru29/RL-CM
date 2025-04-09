from panda_robot import PandaRobot
from env import Env
import numpy as np

from curriculum import Curriculum
from simulator import Simulator

from randomagent import RandomAgent

if __name__ == '__main__':

    curri = Curriculum(
        weights=[[1., 0.5, 10., 0.05, 0.],
                 [0., 2., 10., 0., 5e-3],
                 [0., 0., 10., 0., 5e-3]],
        thresholds=[-0.1, 1.5, 9.],
        n_success=10,
    )

    agent = RandomAgent(
        action_space=[1],
        observation_size=20,
        n_trajectories=32,
        buffer_size=10,
        max_trajectory_length=100,
        learning_rate=0.01,
        discount_factor=0.9,
    )

    simulator = Simulator(
        agent=agent,
        n_simulations=1,
        n_episodes=5,
        curriculum=curri,
        real_time=True
    )

    simulator.run()
