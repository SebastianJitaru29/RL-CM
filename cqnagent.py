from agent import Agent

import torch
import torch.optim as optim
import numpy as np

from network import NeuralNetwork

class CQNAgent(Agent):
    """
    """

    def __init__(
            self,
            n_joints,
            n_levels,
            n_bins,
            *args,
            **kwargs,
    ) -> None:
        """"""
        super().__init__(
            action_space=(n_levels, n_joints, n_bins),
            *args,
            **kwargs
        )

        self.n_levels = n_levels
        self.n_joints = n_joints
        self.n_bins = n_bins

        # Set up all the networks.......
        self.networks = self._setup_networks()

        self.reset()


    def choose_action(
            self,
            observation: np.ndarray,
    ) -> np.ndarray:
        
        obs_list = observation.tolist()

        actions = np.array((self.n_levels, self.n_joints, self.n_bins))

        for l_idx, level in enumerate(self.networks):
            for n_idx, network in enumerate(level):
                local_obs = torch.tensor(obs_list, dtype=torch.float32)
                action = torch.argmax(network(local_obs))
                actions[l_idx, n_idx, action] = 1
                obs_list.append(action - self.n_bins // 2) # [-1, 0, 1]
                
        # TODO torch / numpy?
        self.chosen_actions[self.trajectory, self.timestep] = actions

        # TODO include randomness

        return self._unjumble_the_mumble(actions)
    
        
    def learn(self) -> None:
        # TODO TODO TODO
        pass


    def reset(self):
        super().reset()


    def _setup_networks(self):
        networks = []
        n_prev_joints = 0

        for _ in range(self.n_levels):
            joint_networks = []
            for _ in range(self.n_joints):
                joint_networks.append(
                    NeuralNetwork(
                        input_size=self.observation_size+n_prev_joints,
                        hidden=[48, 48],
                        out_size=3,
                        device=self.device,
                    )
                )
                n_prev_joints += 1
            networks.append(joint_networks)
        
        return networks
    
    def _unjumble_the_mumble(self, actions):

        joint_space = np.zeros(self.n_joints)

        for j_idx in range(self.n_joints):
            bin = 0
            for lvl_idx in range(self.n_levels):
                bin *= self.n_bins
                bin += np.argmax(actions[lvl_idx, j_idx])
            
            j_range = self.limits[j_idx, 1] - self.limits[j_idx, 0]
            j_step = j_range / (self.n_bins ** self.n_levels)
            joint_space[j_idx] = (j_step * bin + j_step / 2 
                                  + self.limits[j_idx, 0])
        
        return joint_space
