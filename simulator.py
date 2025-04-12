from agent import Agent
from env import Env
from curriculum import Curriculum

import numpy as np
from tqdm import tqdm

from typing import Union, Iterable, Dict, Tuple
import os


class Simulator:
    """Class to handle running the training loop of the RL Agent.

    This class handles the entire training loop of the rl agent.
    It handles the needed interface between the agent and the environment
    and calls the appropriate functions on both.
    """
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
        """Initializes the Simulator.
        
        @param agent (Union[Agent, Iterable[Agent]]): The agent to train
            if an iterable is provided, the simulator iterates and trains
            each of the agents.
        @param n_simulations (int): The number of repeated trainings from
            scratch. Used to gather reliable results.
        @param n_episodes (int): The number of episodes to train the agent.
        @param curriculum (Curriculum): The reward curriculum.
        @param data_dir (os.PathLike): Directory to save results.
        *,
        @param real_time (bool): Bool to run the simulation in real time.
        """
        
        self.agent = agent
        
        self.n_simulations = n_simulations
        self.n_episodes = n_episodes
        self.data_dir = data_dir
        self.curri = curriculum

        self.env = Env(
            sampling_rate=1e-3, 
            reward_func=self.curri,
            max_steps=12,
            steps_per_step=500,
            real_time=real_time
        )

        state, _ = self.env.reset()
        self.observation = self._state2obs(state)
    

    def run(self) -> None:
        """Runs the entire training setup.

        This function runs each agent for the given number of simulations
        and episodes and logs the results.
        """
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
        """Helper function to run the setup for one agent.
        
        @param agent (Agent): The agent to run the training setup for.
        @param identifier (str): The identifier of the agent.
        """

        results = np.zeros((self.n_simulations, self.n_episodes))

        for sim in tqdm(range(self.n_simulations), 'Simulations'):
            agent.reset()
            
            log_path = os.path.join(self.data_dir, f'results_{identifier}.csv')
            with open(log_path, 'w') as file:
                file.write('episode,avg_reward,scoring\n')

            for episode in tqdm(range(self.n_episodes), 'Episodes'):
                reward, _ = self._episode(agent)
                results[sim, episode] = reward

                if episode % 100 == 0:
                    # TODO test -> no random, save results
                    result_test, scored = self._test_agent(agent)

                    with open(log_path, 'a') as file:
                        file.write(f'{episode},{result_test},{scored}\n')

                    print(f'TEST: {result_test} | {scored}')
                    next_tast = self.curri.examinate(result_test)
                    if next_tast:
                        agent.next_task()

                if episode % 1000 == 0 and episode != 0:
                    agent.save_networks('/home/mattias/Documents/university/ms/y2/cmr/rl/RL-CM/networks')
            
            agent.save_networks('/home/mattias/Documents/university/ms/y2/cmr/rl/RL-CM/networks')
            


    def _episode(self, agent: Agent) -> Tuple[float, float]:
        """Run one episode of the task.
        
        @param agent (Agent): The agent controlling the arm.
        
        @return (Tuple[float, float]): The cumulative reward and scoring.
        """
        state, _ = self.env.reset()
        self.observation = self._state2obs(state)
        
        cumulative_reward = 0
        while True:
            reward, terminated, scored = self._step(agent)
            cumulative_reward += reward

            if terminated:
                break
        
        return cumulative_reward, scored

    def _step(self, agent: Agent) -> Tuple[float, bool, bool]:
        """Helper function to perform one step in the cycle.
        
        Takes one step in which the agent acts and observes.

        @param agent (Agent): The agent.
        
        @return (Tuple[float, bool, bool]): The reward,
            whether the episode terminated and whether the agent scored.
        """
        action = agent.choose_action(self.observation)
        state, reward, terminated, _, _ = self.env.step(action)

        prev_observation = self.observation
        self.observation = self._state2obs(state)

        agent.observe_result(reward, terminated, prev_observation)

        return reward, terminated, state['score']
    
    def _test_agent(self, agent: Agent) -> Tuple[float, float]:
        """Helper function to test an agent.
        
        Runs the episod function without randomization to test
        the agents actual performance.

        @param agent (Agent): The agent to test.

        @return (Tuple[float, float]): The average reward and goals.
        """
        agent.set_eval(True)
        test_reward = 0
        num_scored = 0

        for _ in range(10):
            reward, scored = self._episode(agent)
            test_reward += reward
            num_scored += scored

        agent.set_eval(False)
        return test_reward / 10, num_scored / 10

    @staticmethod
    def _state2obs(state: Dict) -> np.ndarray:
        """Turn state dict into observation for agent.
        
        @param state (Dict): The state dictionary from the environment.

        @return (np.ndarray): The observation for the agent.
        """

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
    