from abc import abstractmethod
import torch
import torch.optim as optim
import numpy as np
from torchvision.transforms import transforms, Resize


class Agent:
    def __init__(
        self,
        n_actions: int,
        learning_rate: float,
        discount_factor: float,
        n_trajectories: int,
        max_trajectory_length: int = 11,
        optimizer_class: optim.Optimizer = optim.RMSprop,
        device: torch.device = torch.device('cuda'),
        resize: transforms = Resize((21, 21)),
    ) -> None:
        """!
        Initializes the agent.
        Agent is an abstract class that should be inherited by any agent that
        wants to interact with the environment. The agent should be able to
        store transitions, choose actions based on observations, and learn from the
        transitions.

        @param n_actions (int): Number of actions the agent can take.
        @param learning_rate (float): Learning rate.
        @param discount_factor (float): Discount factor of future rewards.
        @param n_trajectories (int): Number of trajectories between
            learning steps.
        @param max_trajectory_length (int): Maximum trajectory length.
        @param optimizer_class (torch.optim.Optimizer): Optimizer class
            to use for training the network.
        @param device (torch.device): Device to use for the agent.
        @param resize (Transform): Transform to apply to observations.
        """

        self.n_trajectories = n_trajectories
        self.max_trajectory_length = max_trajectory_length
        self.n_actions = n_actions
        self.device = device
        self.resize = resize
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class

        discounts = [1 * discount_factor ** (max_trajectory_length - ts - 1)
                     for ts in range(max_trajectory_length)]

        self.discounts = torch.tensor(discounts, dtype=torch.float32,
                                      device=self.device)
        print(self.discounts)

    @abstractmethod
    def choose_action(
        self,
        observation: np.ndarray
    ) -> int:  # Is this always an int?
        """!
        Abstract method that should be implemented by the child class, e.g. DQN or DDQN agents.
        This method should contain the full logic needed to choose an action based on the current state.
        Maybe you can store the neural network in the agent class and use it here to decide which action to take?

        @param observation (np.ndarray): Vector describing current state

        @return (int): Action to take
        """

        return 0

    @abstractmethod
    def learn(self) -> None:
        """!
        Update the parameters of the internal networks.
        This method should be implemented by the child class.
        """

        pass

    def _preprocess_observation(self, observation: np.ndarray) -> torch.Tensor:
        """Preprocesses the observation.

        This function rescales the observation to the minimum size
        where no information is lost, in this case (21, 21).

        @param observation (np.ndarray): Observation to preprocess

        @return (torch.Tensor): Preprocessed observation
        """
        image = torch.from_numpy(observation).float() / 255.0
        image = torch.moveaxis(image, -1, 0)
        image = self.resize(image)
        return torch.where(image > 0.25, torch.tensor(1.0), torch.tensor(0.0))

    def reset(self) -> None:
        """Resets the agent for a new training run.

        This function prepares the agent for being trained. It should
        be implemented by the child class.
        """
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        """Resets the memory buffers of the agent."""
        self.timestep = 0
        self.trajectory = 0

        self.states = torch.zeros(self.n_trajectories,
                                  self.max_trajectory_length,
                                  4, 21, 21)

        self.chosen_actions = torch.zeros(self.n_trajectories,
                                          self.max_trajectory_length,
                                          dtype=torch.long,
                                          device=self.device)

        self.rewards = torch.zeros(self.n_trajectories, device=self.device)
        self.terminated = torch.zeros(self.n_trajectories, dtype=torch.int)

    def observe_result(self, reward, terminated, state) -> None:
        """Observe the environment after an action is taken.

        @param reward (float): Returned reward.
        @param terminated (bool): Whether the episode is terminated.
        @param state (torch.Tensor): The state after the action is taken.
        """
        self.states[self.trajectory][self.timestep] \
            = self._preprocess_observation(state)

        if terminated:
            self.terminated[self.trajectory] = self.timestep
            self.rewards[self.trajectory] = reward
            self.trajectory += 1
            if self.trajectory >= self.n_trajectories:
                self.learn()
                self._reset_buffers()
            else:
                self.timestep = 0
            return
        self.timestep += 1

    def select_probability(
        self,
        probabilities: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Select the action probabilities of the taken actions.

        @param probabilities (torch.Tensor): Predicted probabilities
            over the action space.
        @param actions (torch.Tensor): The taken actions of which to
            select the probabilities.
        """
        selected_actions = torch.zeros(actions.shape[0],
                                       device=self.device)
        for idx in range(actions.shape[0]):
            selected_actions[idx] = probabilities[idx, actions[idx]]
        return selected_actions.view((self.max_trajectory_length, 1))

    @staticmethod
    def name() -> str:
        """Returns the name of the agent.

        To be overwritten by child classes in order to have their name
        available to the writer classes.
        """
        return 'agent'
