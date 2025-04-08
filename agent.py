from abc import abstractmethod
import torch
import torch.optim as optim
import numpy as np

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
    ) -> None:
        pass


    @abstractmethod
    def choose_action(
            self,
            observation: np.ndarray
    ) -> np.ndarray:
        return np.zeros(7)


    @abstractmethod
    def learn(self) -> None:
        pass


    def observe_result(
            self,
            reward: float,
            terminated: bool,
            observation: np.ndarray,
    ):
        return
    

    def reset(self):
        pass


    @staticmethod
    def _unnormalize(output):
        global limits
        """[0, 1) -> joint limits [lim_min, lim_max)"""
        return output * limits[1] + limits[0]
