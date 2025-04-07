from panda_robot import PandaRobot
from env import Env
import numpy as np

if __name__ == '__main__':
    env = Env(1e-3, None, real_time=True)
    state, info = env.reset()

    while True:
        action = np.random.random(9)
        action[3] = -action[3]
        for step in range(2000):
            _ = env.step(action)
        state, info = env.reset()
    

        