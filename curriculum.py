from rewards import *

from typing import List


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
        self.track_record = None
        self.cummulative = 0

        self._next_function()

    
    def __call__(self, state, terminal):
        
        reward = 0
        for weight, func in zip(self.cur_weights, self.functions):
            reward += weight * func(state)
        
        self.cummulative += reward
        
        if terminal:
            self.track_record[0:-1] = self.track_record[1:]
            self.track_record[-1] = self.cummulative >= self.cur_thresh
            self.cummulative = 0
            
            if np.sum(self.track_record) >= self.n_success:
                self._next_function()
                print('--NEXT CURRICULUM SECTION--')

        return reward


    def _next_function(self):
        if self.f_idx == len(self.weights) - 1:
            return
        self.f_idx += 1
        self.cur_weights = self.weights[self.f_idx]
        self.cur_thresh = self.thresholds[self.f_idx]
        self.track_record = np.zeros(self.n_success + self.lenience,
                                     dtype=bool)
        
        