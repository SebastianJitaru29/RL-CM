from panda_robot import PandaRobot
from env import Env
import numpy as np

from curriculum import Curriculum
from simulator import Simulator

from randomagent import RandomAgent
from cqnagent import CQNAgent
from randomizer import Randomizer

if __name__ == '__main__':

    #[[0.5, 1, 5., 0.1, 0.05]]
    curri = Curriculum(
        weights=[[1., 0.5, 1., 0.5, 0.1],
                 [0.1, 1., 2., 0.5, 0.1],
                 [0.1, 0.2, 2., 0.5, 0.1]],
        thresholds=[4.5, 3., 1.70],
        n_success=10,
    )

    randomizer = Randomizer(
        0.4, 0.3, 0.4,
        7.5e-5, 7.5e-5, 7.5e-5,
        0.016, 0.016, 0.016,
        0.2, 0.16, 0.2,
        max_steps = 4,
    )

    agent = CQNAgent(
        n_joints=7,
        n_levels=3,
        n_bins=3,
        observation_size=20,
        n_trajectories=1,
        batch_size=32,
        buffer_size=1024,
        max_trajectory_length=12,
        theta_bar_delay=64,
        randomizer=randomizer,
        learning_rate=0.001,
        discount_factor=0.95,
    )

    simulator = Simulator(
        agent=agent,
        n_simulations=1,
        n_episodes=50000,
        curriculum=curri,
        real_time=False,
        data_dir='/home/mattias/Documents/university/ms/y2/cmr/rl/RL-CM/data',
    )

    simulator.run()
