from agent import Agent
from env import Env
from curriculum import Curriculum

import numpy as np
from tqdm import tqdm

from typing import Union, Iterable, Dict, Tuple
import os


class Simulator:

    def __init__(
            self,
            agent: Union[Agent, Iterable[Agent]],
            n_simulations: int, # TODO check if we want this
            n_episodes: int,
            curriculum: Curriculum,
            data_dir: os.PathLike=None,
            *,
            real_time: bool=False,
    ) -> None:
        
        self.agent = agent
        
        self.n_simulations = n_simulations
        self.n_episodes = n_episodes
        self.data_dir = data_dir
        self.curri = curriculum

        self.env = Env(
            sampling_rate=1e-3, 
            reward_func=self.curri,
            max_steps=10,
            steps_per_step=500,
            real_time=real_time
        )

        state, _ = self.env.reset()
        self.observation = self._state2obs(state)
    

    def run(self):
        if isinstance(self.agent, Agent):
            print('Running Single Agent')
            self._simulate(self.agent, self._identify(0))
        else:
            for idx, agent in enumerate(self.agent):
                self._simulate(agent, self._identify(idx))


    def _simulate(
            self, 
            agent: Agent,
            identifier: str    # For logging data
    ) -> None:
        """"""

        results = np.zeros((self.n_simulations, self.n_episodes))

        for sim in tqdm(range(self.n_simulations), 'Simulations'):
            agent.reset()
            
            for episode in tqdm(range(self.n_episodes), 'Episodes'):
                reward = self._episode(agent)
                results[sim, episode] = reward
                print(reward)
        # TODO write results, add current task to said results


    def _episode(self, agent: Agent) -> float:
        """Run one episode of the task.
        
        @param agent (Agent): The agent controlling the arm.
        
        @return (float): The cumulative reward OR scoring (TODO)
        """
        state, _ = self.env.reset()
        self.observation = self._state2obs(state)
        
        cumulative_reward = 0
        while True:
            reward, terminated = self._step(agent)
            cumulative_reward += reward

            if terminated:
                break
        
        return cumulative_reward


    def _step(self, agent: Agent) -> Tuple[float, bool]:
        action = agent.choose_action(self.observation)
        state, reward, terminated, _, _ = self.env.step(action)

        prev_observation = self.observation
        self.observation = self._state2obs(state)

        agent.observe_result(reward, terminated, prev_observation)

        return reward, terminated


    @staticmethod
    def _state2obs(state: Dict):
        """Turn state dict into observation for agent."""

        observation = np.zeros(20)
        observation[0:7] = state['joint_info'][0][0:7]
        observation[7:14] = state['joint_info'][1][0:7]
        observation[14:17] = state['ball_info'][0]
        observation[17:20] = state['ball_info'][1]
        
        return observation

    # This function is borrowed from myself:
    @staticmethod
    def _identify(index: int) -> str:
        """Creates identification string based on index.

        @param index (int): The index of the agent
        @return (str) The identification string
        """
        ident = str(index)
        if len(ident) < 3:
            ident = '0' * (3 - len(ident)) + ident
        return ident
    