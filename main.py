from panda_robot import PandaRobot
from env import Env
import numpy as np

from curriculum import Curriculum
from simulator import Simulator

from randomagent import RandomAgent
from cqnagent import CQNAgent
from randomizer import Randomizer

if __name__ == '__main__':

    curri = Curriculum(
        weights=[[1., 0.8, 10., 0.25, 0.],
                 [0., 2., 10., 0.01, 5e-2],
                 [0., 0., 10., 0.01, 5e-2]],
        thresholds=[-2, 1.5, 9.],
        n_success=10,
    )

    randomizer = Randomizer(
        0.3, 0.3, 0.3,
        2.5e-3, 2.5e-3, 2.5e-3,
        0.016, 0.016, 0.016,
        0.1, 0.1, 0.1,
        max_steps = 3,
    )

    agent = CQNAgent(
        n_joints=7,
        n_levels=3,
        n_bins=3,
        observation_size=20,
        n_trajectories=64,
        batch_size=32,
        buffer_size=512,
        max_trajectory_length=10,
        randomizer=randomizer,
        learning_rate=0.01,
        discount_factor=0.9,
    )

    simulator = Simulator(
        agent=agent,
        n_simulations=1,
        n_episodes=10000,
        curriculum=curri,
        real_time=False,
    )

    simulator.run()
