from panda_robot import PandaRobot
from env import Env
import numpy as np

from curriculum import Curriculum

if __name__ == '__main__':

    curri = Curriculum(
        weights=[[1., 0.5, 10., 0.25, 0.],
                 [0., 2., 10., 0., 5e-3],
                 [0., 0., 10., 0., 5e-3]],
        thresholds=[-0.1, 1.5, 9.],
        n_success=10,
    )
    env = Env(1e-3, curri, real_time=True)
    state, info = env.reset()

    while True:
        action = np.random.random(9)
        action[3] = -action[3]
        for step in range(50):
            _ = env.step(action)
        state, info = env.reset()
    

        