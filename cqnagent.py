from agent import Agent

import torch
import torch.optim as optim
import numpy as np

from network import NeuralNetwork

from copy import deepcopy

class CQNAgent(Agent):
    """
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
        """"""
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
        
        # TODO joint normalization
        obs_list = observation.tolist()
        actions = np.zeros((self.n_levels, self.n_joints))

        for l_idx, level in enumerate(self.networks):
            for n_idx, network in enumerate(level):
                local_obs = torch.tensor(obs_list, dtype=torch.float32)
                with torch.no_grad():
                    action = torch.argmax(network(local_obs)).item()
                
                # TODO change overtime and improve
                if np.random.random() <= 0.01:
                    action = np.random.randint(0, self.n_bins)

                actions[l_idx, n_idx] = action
                obs_list.append(action - self.n_bins // 2) # [-1, 0, 1]
                
        # TODO include randomness
        actions = self.randomizer.randomize(actions)
        
        # TODO torch / numpy?
        self.chosen_actions[self.trajectory, self.timestep] = actions
        
        act_7 = self._unjumble_the_mumble(actions)
        return act_7.tolist() + [0., 0.]

    def learn(self) -> None:
        # TODO TODO TODO
        # Retrieve random datapoints
        # Create batch
        # Put batch through networks to calculate current value
        # Calculate loss using theta_bar
        # Update networks
        print('--LEARNING--')
        for lst_optim in self.optims:
            for optim in lst_optim:
                optim.zero_grad()

        self._set_train_or_eval('train', self.networks)
        indices = self._random_trajectories(self.batch_size)

        for traj_idx in indices:
            term_idx = self.terminated[traj_idx]
            batch = self.states[traj_idx, 0:term_idx]
            batch_next = self.states[traj_idx, 1:term_idx]
            actions = self.chosen_actions[traj_idx, 0:term_idx]

            values = self._batch_pass(batch, actions)
            with torch.no_grad():
                values_next = self._batch_pass(batch_next)
            rewards = self.rewards[traj_idx, 0:term_idx]
            rewards = torch.tensor(rewards)
        
            # target = rewards + values_next
            target = self.discount_factor * values_next
            for l_idx in range(self.n_levels):
                for j_idx in range(self.n_joints):
                    target[:, l_idx, j_idx] += rewards

            loss = self.mse(target, values)
            loss.backward(retain_graph=True)
        
        # TODO for each network?
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


    def reset(self):
        super().reset()
        self.theta_bar = deepcopy(self.networks)
        self._set_train_or_eval('eval', self.theta_bar)
        self._set_train_or_eval('eval', self.networks)
        self.theta_counter = 0

    def _setup_networks(self):
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
                        hidden=[48, 48],
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
    
    def _unjumble_the_mumble(self, actions):

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
            batch: np.ndarray,
            actions: np.ndarray = None,
    ) -> np.ndarray:
        # Shape BATCH, STATE
        batch_size = batch.shape[0]
        batch_list = batch.tolist()

        if actions is not None:
            values = torch.zeros((batch_size, self.n_levels, self.n_joints))
        else:
            values = torch.zeros((batch_size+1, self.n_levels, self.n_joints))

        for l_idx, level in enumerate(self.networks):
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


    def _set_train_or_eval(self, mode, networks):
        for row, lvl in enumerate(networks):
            for col, _ in enumerate(lvl):
                if mode == 'train':
                    networks[row][col].train()
                else:
                    networks[row][col].eval()

    def _random_trajectories(self, num):
        idx_max = self.buffer_size if self.buffered else self.trajectory
        indices = np.random.choice(idx_max, num, replace=False)
        return indices

    def _retrieve_data(self):
        NUM_TRAJECTORIES = 2

        idx_max = self.buffer_size if self.buffered else self.trajectory
        indices = np.random.choice(idx_max, NUM_TRAJECTORIES)

        number_of_samples = np.sum(self.terminated(indices))
        batch = np.zeros((number_of_samples, self.observation_size))
        rewards = np.zeros((number_of_samples))
        # STATES, REWARDS, NEXT_STATES, TERMINATED

        
