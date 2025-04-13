
## RL Curriculum Learning with PyBullet


Most Reinforcement Learning code such as: `simulator.py`, `agent.py` and `env.py` are based on a previous assignment from Deep Reinforcement Learning.
The code used as reference can be found on https://github.com/MattiasvdK/rug-rl-pong.

The `PandaRobot` class in `panda_robot` was taken from 


### Running the code

The code can be run using the following command:

`$python main.py`


### Important files/classes

`cqnagent.py`: The implementation of our MACQN agent as described in the report.
`simulator.py`: Class that runs the whole training loop.
`env.py`: The training environment.
`curriculum.py`: The class to handle the curriculum.

The different reward functions can be found in `rewards.py`.

