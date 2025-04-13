from agent import Agent

import torch
import numpy as np
import os

from network import NeuralNetwork
from copy import deepcopy

from typing import List, Tuple

class CQNAgent(Agent):
    """Implementation of the MACQN Agent as described in the report.
    """

    def __init__(
            self,
            n_joints: int,
            n_levels: int,
            n_bins: int,
            theta_bar_delay: int=32,
            batch_size: int=16,
            *args,
            **kwargs,
    ) -> None:
        """Initializes the MACQN Agent.
        
        @param n_joints (int): the number of joints of the arm.
        @param n_levels (int): the number of discretization levels.
        @param n_bins (int): the number of discretization bins.
        @param theta_bar_delay (int): default=32, the delay of updating
            the target networks theta^-.
        @param batch_size (int): default=16, number of trajectories to
            train on during a training step.
        """
        super().__init__(
            action_space=(n_levels, n_joints),
            *args,
            **kwargs
        )

        self.n_levels = n_levels
        self.n_joints = n_joints
        self.n_bins = n_bins
        self.theta_bar_delay = theta_bar_delay
        self.theta_counter = 0
        self.batch_size = batch_size

        self.mse = torch.nn.MSELoss()

        # Set up all the networks.......
        self.optims = None
        self.theta_bar = None
        self.networks = self._setup_networks()
        self.reset()


    def choose_action(
            self,
            observation: np.ndarray,
    ) -> np.ndarray:

        obs_list = observation.tolist()
        actions = np.zeros((self.n_levels, self.n_joints))

        # Loop over all the networks to select their bin and append it to
        # the observation list.
        for l_idx, level in enumerate(self.networks):
            for n_idx, network in enumerate(level):
                local_obs = torch.tensor(obs_list, dtype=torch.float32)
                with torch.no_grad():
                    action = torch.argmax(network(local_obs)).item()

                actions[l_idx, n_idx] = action
                obs_list.append(action - self.n_bins // 2) # [-1, 0, 1]
                
        if not self.eval:
            actions = self.randomizer.randomize(actions)
        
        self.chosen_actions[self.trajectory, self.timestep] = actions
        
        act_7 = self._unjumble_the_mumble(actions)

        # Append 0 position for the gripper joints.
        return act_7.tolist() + [0., 0.]

    def learn(self) -> None:
        for lst_optim in self.optims:
            for optim in lst_optim:
                optim.zero_grad()

        self._set_train_or_eval('train', self.networks)
        indices = self._random_trajectories(self.batch_size)

        batch, batch_next, rewards, actions, terms = self._create_batch(indices)
       
        # Obtain the contemporary predicted values of the selected actions
        values = self._batch_pass(self.networks, batch, actions)

        # Compute the predicted maximum value of the next state
        with torch.no_grad():
            values_next = self._batch_pass(self.theta_bar, batch_next)
        
        rewards = torch.tensor(rewards)
        target = self.discount_factor * values_next

        # Set terminated next states to have values of 0
        idx = 0
        for term in terms:
            idx += term
            target[idx] = torch.zeros((self.n_levels, self.n_joints))

        # Add the reward to the calculated next values
        for l_idx in range(self.n_levels):
            for j_idx in range(self.n_joints):
                target[:, l_idx, j_idx] += rewards

        loss = self.mse(target, values)
        loss.backward()
        
        # Take an optimization step for each network
        for lst_optim in self.optims:
            for optim in lst_optim:
                optim.step()

        self._set_train_or_eval('eval', self.networks)

        if self.theta_counter == self.theta_bar_delay:
            self.theta_bar = deepcopy(self.networks)
            self.theta_counter = 0
        else:
            self.theta_counter += 1
        
        self.randomizer.step()


    def reset(self) -> None:
        super().reset()
        self.theta_bar = deepcopy(self.networks)
        self._set_train_or_eval('eval', self.theta_bar)
        self._set_train_or_eval('eval', self.networks)
        self.theta_counter = 0


    def save_networks(self, directory: os.PathLike) -> None:
        for lvl in range(self.n_levels):
            for joint in range(self.n_joints):
                path = directory + f'/net_{lvl}_{joint}.pt'
                torch.save(self.networks[lvl][joint].state_dict(), path)


    def _setup_networks(self) -> None:
        """Helper function to create all the needed neural networks."""
        networks = []
        self.optims = []
        n_prev_joints = 0

        for _ in range(self.n_levels):
            joint_networks = []
            joint_optims = []
            for _ in range(self.n_joints):
                network = NeuralNetwork(
                        input_size=self.observation_size+n_prev_joints,
                        output_size=3,
                        hidden=[128, 128],
                        device=self.device,
                    )
                joint_networks.append(network)
                n_prev_joints += 1
                joint_optims.append(self.optimizer_class(
                    network.parameters(),
                    lr=self.learning_rate
                ))
            networks.append(joint_networks)
            self.optims.append(joint_optims)
        
        return networks
    
    def _unjumble_the_mumble(
            self, 
            actions: List[List[int]],
    ) -> List[List[int]]:
        """Helper function to map the selected bins to joint positions."""
        joint_space = np.zeros(self.n_joints)

        for j_idx in range(self.n_joints):
            bin = 0
            for lvl_idx in range(self.n_levels):
                bin *= self.n_bins
                bin += actions[lvl_idx, j_idx]
            
            j_range = self.limits[1, j_idx] - self.limits[0, j_idx]
            j_step = j_range / (self.n_bins ** self.n_levels)
            joint_space[j_idx] = (j_step * bin + j_step / 2 
                                  + self.limits[0, j_idx])
        
        return joint_space
    
    def _batch_pass(
            self, 
            networks,
            batch: np.ndarray,
            actions: np.ndarray = None,
    ) -> np.ndarray:
        """Helper function to obtain network outputs for an entire batch."""
        # Shape BATCH, STATE
        batch_size = batch.shape[0]
        batch_list = batch.tolist()

        if actions is not None:
            values = torch.zeros((batch_size, self.n_levels, self.n_joints))
        else:
            values = torch.zeros((batch_size, self.n_levels, self.n_joints))

        for l_idx, level in enumerate(networks):
            for n_idx, network in enumerate(level):
                local_obs = torch.tensor(batch_list, dtype=torch.float32)
                vals = network(local_obs) # (BATCH, 3)

                for b_idx in range(batch_size):
                    if actions is not None:
                        action = actions[b_idx, l_idx, n_idx]
                    else:
                        action = torch.argmax(vals[b_idx])
                    values[b_idx, l_idx, n_idx] = vals[b_idx, action]
                    batch_list[b_idx].append(
                        action - self.n_bins // 2
                    ) # [-1, 0, 1]

        return values

    def _set_train_or_eval(
            self,
            mode: str,
            networks: List[List[NeuralNetwork]],
    ) -> None:
        """Helper function to set the correct mode of the networks."""
        for row, lvl in enumerate(networks):
            for col, _ in enumerate(lvl):
                if mode == 'train':
                    networks[row][col].train()
                else:
                    networks[row][col].eval()

    def _random_trajectories(self, num):
        """Helper function to select random trajectories from replay buffer."""
        idx_max = self.buffer_size if self.buffered else self.trajectory
        indices = np.random.choice(idx_max, num, replace=False)
        return indices

    def _create_batch(
            self,
            trajectories: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[bool]]:
        """Helper function to create a batch from the given trajectories.
        
        @param trajectories (np.ndarray): the trajectories to put in batch.
        
        @return (
            Tuple[
                np.ndarray, (states)
                np.ndarray, (next states)
                np.ndarray, (rewards)
                np.ndarray, (selected actions)
                List[bool], (terminal index per trajectory)
            ]
        )
        """
        batch_all = None
        batch_next_all = None
        rewards_all = None
        actions_all = None
        terms = []

        for traj_idx in trajectories:
            term_idx = self.terminated[traj_idx]
            terms.append(term_idx)
            batch = self.states[traj_idx, 0:term_idx+1]
            batch_next = np.zeros_like(batch)
            batch_next[0:term_idx] = self.states[traj_idx, 1:term_idx+1]

            actions = self.chosen_actions[traj_idx, 0:term_idx+1]
            rewards = self.rewards[traj_idx, 0:term_idx+1]

            if batch_all is None:
                batch_all = batch
            else:
                batch_all = np.concatenate((batch_all, batch), axis=0)

            if batch_next_all is None:
                batch_next_all = batch_next
            else:
                batch_next_all = np.concatenate((batch_next_all, batch_next), axis=0)

            if rewards_all is None:
                rewards_all = rewards
            else:
                rewards_all = np.concatenate((rewards_all, rewards), axis=0)

            if actions_all is None:
                actions_all = actions
            else:
                actions_all = np.concatenate((actions_all, actions), axis=0)

        return (batch_all, batch_next_all, rewards_all, actions_all, terms)
