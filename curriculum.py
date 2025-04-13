from rewards import *

from typing import List, Dict


class Curriculum:

    def __init__(
            self,
            weights: List[List[float]],
            thresholds: List[float],
            n_success: int,
            lenience: int=1,
    ):
        """Initialize the curriculum.

        @param weights (List[List[float]]): List of weights for the different
            reward functions.
        @param thresholds (List[float]): Thresholds for final reward to be
            counted as successful.
        @param n_success (int): Number of successes before moving to the next
            step in the curriculum.
        @param lenience (int): The number of failures to be accepted in a
            sequence of successes before moving on.
        """
        self.weights = weights
        self.thresholds = thresholds
        self.f_idx = -1
        self.n_success = n_success
        self.lenience = lenience

        self.functions = [
            reward_EE,
            reward_kick,
            reward_score,
            reward_effort,
            reward_time,
        ]

        self.cur_weights = None
        self.cur_thresh = None

        self._next_function()

    
    def __call__(self, state: Dict) -> float:
        """Calculate the reward based on the given state.
        
        @param state (Dict): the state dict.

        @return (float): the obtained reward.
        """
        
        reward = 0
        for weight, func in zip(self.cur_weights, self.functions):
            reward += weight * func(state)

        return reward
    

    def examinate(self, reward: float) -> bool:
        """Function to check if the network passed the threshold.
        
        This function checks if the given reward is higher than the
        threshold needed to pass the current course.

        @param reward (float): The obtained testing reward.

        @return (bool): Whether it surpassed the threshold.
        """
        if reward >= self.cur_thresh:
            self._next_function()
            print('--NEXT CURRICULUM SECTION--')
            return True
        return False


    def _next_function(self):
        """Helper function to transition to next course."""
        if self.f_idx == len(self.weights) - 1:
            return
        self.f_idx += 1
        self.cur_weights = self.weights[self.f_idx]
        self.cur_thresh = self.thresholds[self.f_idx]
        
        