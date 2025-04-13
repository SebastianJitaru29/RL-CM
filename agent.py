from abc import abstractmethod
import torch
import torch.optim as optim
import numpy as np
import os

limits = np.array([
    [-2.7437,  2.7437],
    [-1.7837,  1.7837],
    [-2.9007,  2.9007],
    [-3.0421, -0.1518],
    [-2.8065,  2.8065],
    [ 0.5445,  4.5169],
    [-3.0159,  3.0159],
    [ 0.0000,  0.0000],
    [ 0.0000,  0.0000],
]).transpose()

class Agent:
    def __init__(
            self,
            action_space: int,
            observation_size: int,
            n_trajectories: int,
            buffer_size: int,
            max_trajectory_length: int,
            learning_rate: float,
            discount_factor: float,
            randomizer: object,
            optimizer_class: optim.Optimizer = optim.RMSprop,
            device: torch.device = torch.device('cuda')
    ) -> None:
        """Initializes the agent.
        
        Agent is an abstract class that can be used by RL algorithm
        implementations. It is the class the simulator uses thus any
        agent should inherit from the Agent class.

        @param action_space (int): Depth of discritizing the actionspace.
        @param observation_size: (int): --
        @param n_trajectories (int): Number of trajectories between
        @param buffer_size (int): --
        @param max_trajectory_length (int): Maximum trajectory length.
        @param learning_rate (float): Learning rate.
        @param discount_factor (float): Discount factor of future rewards.
            learning steps.
        @param optimizer_class (torch.optim.Optimizer): Optimizer class
            to use for training the network.
        @param device (torch.device): Device to use for the agent.
        """

        self.n_trajectories = n_trajectories
        self.action_space = action_space
        self.device = device
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.observation_size = observation_size
        self.max_trajectory_length = max_trajectory_length
        self.buffer_size = buffer_size
        self.randomizer = randomizer

        discounts = [[1 * discount_factor ** (max_trajectory_length - ts - 1)
                     for ts in range(max_trajectory_length)]]
        
        self.discounts = torch.tensor(discounts, dtype=torch.float32,
                                      device=self.device)
        
        self.timestep = 0
        self.trajectory = 0
        self.traj_till_learn = self.n_trajectories
        self.states = None
        self.chosen_actions = None
        self.rewards = None
        self.terminated = None
        self.limits = limits
        self.buffered = False
        self.eval = False


    @abstractmethod
    def choose_action(
            self,
            observation: np.ndarray,
    ) -> np.ndarray:
        """Select an action based on the observation.
        
        @param observation (np.ndarray): the observed environment state.

        @return (np.ndarray): The selected action.
        """
        return np.zeros(7)


    @abstractmethod
    def learn(self) -> None:
        """Perform a training step for the agent."""
        pass


    def observe_result(
            self,
            reward: float,
            terminated: bool,
            observation: np.ndarray,
    ) -> None:
        """Observe the results after a taken action.

        This function stores the state and resulting information in the
        memory replay buffer.

        @param reward (float): the observerd reward.
        @param terminated (bool): whether the new state was terminal.
        @param observation (np.ndarray): the state in which the action
            was taken.
        
        """
        # Do not save state if evaluating
        if self.eval:
            return
        
        self.rewards[self.trajectory, self.timestep] = reward
        self.states[self.trajectory, self.timestep] = observation

        if terminated:
            self.terminated[self.trajectory] = self.timestep
            self.trajectory += 1
            self.timestep = 0

            # We reached the end of the buffers, override oldest
            # datapoints
            if self.trajectory >= self.buffer_size:
                self.trajectory = 0
                self.buffered = True

            self.traj_till_learn -= 1
            if (
                (self.buffered or self.trajectory >= 2*self.batch_size)
            ):
                self.learn()
                self.traj_till_learn = self.n_trajectories

        else:
            self.timestep += 1

    def reset(self) -> None:
        """Reset the agent."""
        self.timestep = 0
        self.trajectory = 0
        self.traj_till_learn = self.n_trajectories
        self.buffered = False

        self._reset_buffers()


    def set_eval(self, val: bool) -> None:
        """Set the evaluation mode of the agent.
        
        @param val (bool): whether to be in evaluation mode.
        """
        self.eval = val

    def next_task(self) -> None:
        """Prepare the agent for the next task.
        
        Resets the memory replay buffer and the randomizer to setup
        training for the next task.
        """
        self.randomizer.reset()
        Agent.reset(self)

    @abstractmethod
    def save_networks(self, directory: os.PathLike) -> None:
        """Save the networks of the agent.
        
        @param directory (os.PathLike): the directory to save the networks in.
        """

        pass

    def _reset_buffers(self) -> None:
        """Helper function to reset the buffers."""
        self.states = np.zeros((self.buffer_size,
                                  self.max_trajectory_length,
                                  self.observation_size))
        
        self.chosen_actions = np.zeros((self.buffer_size,
                                          self.max_trajectory_length,
                                          *self.action_space),
                                          dtype=int)
        
        self.rewards = np.zeros((self.buffer_size,
                                   self.max_trajectory_length))
        
        self.terminated = np.zeros(self.buffer_size,
                                      dtype=int)


    def _unnormalize(self, output):
        """[0, 1) -> joint limits [lim_min, lim_max)"""
        return output * (self.limits[1] - self.limits[0]) + self.limits[0]
