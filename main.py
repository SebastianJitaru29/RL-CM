from panda_robot import PandaRobot
from env import Env
import numpy as np

if __name__ == '__main__':
    env = Env(1e-3, None, True)
    state, info = env.reset()

    for step in range(10000):
        _ = env.step(np.zeros(9))