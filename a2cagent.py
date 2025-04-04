from agent import Agent
import torch
from network import NeuralNetwork
import numpy as np
from typing import Tuple, List


class A2CAgent(Agent):
    """Implementation of the A2C algorithm as RL Agent.

    This class implements the Advantage Actor Critic (A2C) algorithm.
    It uses one network for both the actor and the critic.
    """

    # TODO use state_dimensions for Agent state buffer
    def __init__(
            self,
            n_actions: int,
            learning_rate: float,
            discount_factor: float,
            negative_scale: float = -1.0,
            positive_scale: float = 1.0,
            theta_delay: int = 10,
            cnn_layers: List[int] = [7],
            pooling: bool = True,
            fcl_layers: List[int] = [512],
            n_trajectories: int = 11,
    ) -> None:
        """Initialize the A2C agent.

        @param state_dimensions (Tuple[int, int, int]): Dimensions of
            the state of the environment.
        @param n_actions (int): Number of actions the agent can take.
        @param learning_rate (float): The learning rate of the network.
        @param discount_factor (float): The discount factor of
            future rewards.
        @param negative_scale (float): The negative value of the
            terminal reward.
        @param positive_scale (float): The positive value of the
            terminal reward.
        @param theta_delay (int): The delay between updating the
            theta-minus network
        @param cnn_layers (List[int]): A list of kernel sizes for the
            cnn layers. This also defines the number of cnn layers.
        @param pooling (bool): Whether to use a max-pooling after cnn
            layers.
        @param fcl_layers (List[int]): A list of neurons for dense
            layers. This also defines the number of dense layers.
        @param n_trajectories (int): The number of trajectories before
            performing a gradient descent step.
        """
        super().__init__(n_actions, learning_rate,
                         discount_factor, n_trajectories)

        self.theta_delay = theta_delay
        self.theta_counter = 0

        self.cnn_layers = cnn_layers
        self.pooling = pooling
        self.fcl_layers = fcl_layers
        self.negative_scale = negative_scale
        self.positive_scale = positive_scale
        self.network = None
        self.network_theta = None
        self.optimizer = None

        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

        self.reset()

    def choose_action(
            self,
            observation: np.ndarray
    ) -> int:  # Is this always an int?
        """Choose action based on the current policy.

        Selects an action based on the current policy by sampling
        from the predicted probability distribution.

        @param observation (np.ndarray): The state of the environment.

        @return int: The integer value of the action to take.
        """
        observation = self._preprocess_observation(observation)

        with torch.no_grad():
            action_probs, next_value = self.network(observation)
        np_action_probs = action_probs.cpu().detach().numpy()

        # TODO simulated annealing?
        action = np.random.choice(
            np.arange(len(np_action_probs)),
            p=np_action_probs
        )

        self.chosen_actions[self.trajectory][self.timestep] = action

        return action

    def learn(self) -> None:
        """Perform a gradient descent step.

        This function calculates the loss and performs gradient descent
        based on the episodes collected in the last n_trajectories.
        """

        # TODO clean up the code
        rewards = torch.where(self.rewards > 0.5, self.positive_scale, self.negative_scale)
        total_loss = torch.zeros(1, device=self.device)
        total_value = 0.0
        total_action = 0.0

        self.optimizer.zero_grad()

        self.network.train()
        for trajectory in range(self.n_trajectories):
            # TODO move this to a different function
            action_prob, value = self.network(self.states[trajectory])

            with torch.no_grad():
                _, value_prime = self.network_theta(self.states[trajectory])
            reward = rewards[trajectory].unsqueeze(0).unsqueeze(0)

            next_value = value_prime[1:] * self.discount_factor
            next_value = torch.concat((next_value, reward), 0)

            prime = value[1:] * self.discount_factor
            prime = torch.concat((prime, reward), 0)
            prime = prime.detach()

            advantage = prime - value
            advantage = advantage.detach()

            selected_probabilities = self.select_probability(
                action_prob,
                self.chosen_actions[trajectory]
            )

            selected_probabilities = torch.where(selected_probabilities < 1e-6, 1e-6, selected_probabilities)

            value_loss = self.mse(value, next_value)
            adv_loss = torch.mean(-torch.log(selected_probabilities) * advantage)

            total_value += value_loss.item()
            total_action += adv_loss.item()

            adv_loss.backward(retain_graph=True)
            value_loss.backward(retain_graph=True)

        self.optimizer.step()

        if self.theta_counter == self.theta_delay:
            self._reset_theta()
            self.theta_counter = 0
        else:
            self.theta_counter += 1

    def reset(self) -> None:
        super().reset()
        self.network = NeuralNetwork(
            cnn_layers=self.cnn_layers,
            pooling=self.pooling,
            fcl_layers=self.fcl_layers,
        )
        self.network.train()
        self.network_theta = NeuralNetwork(
            cnn_layers=self.cnn_layers,
            pooling=self.pooling,
            fcl_layers=self.fcl_layers,
        )
        self._reset_theta()
        self.network_theta.requires_grad = False

        self.optimizer = self.optimizer_class(self.network.parameters(),
                                              lr=self.learning_rate)

    def _reset_theta(self):
        """Resets the theta-minus network.

        This function sets the theta-minus parameters to be equal to
        the theta parameters.
        """
        self.network_theta.load_state_dict(self.network.state_dict())
        self.theta_counter = 0

    @staticmethod
    def name():
        return 'a2c'
