from panda_robot import PandaRobot
from env import Env
import numpy as np

if __name__ == '__main__':
    env = Env(1e-3, None, real_time=True)
    state, info = env.reset()

    while True:
        for step in range(2000):
            _ = env.step(np.zeros(9))
        state, info = env.reset()
    

        