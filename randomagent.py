from agent import Agent
import numpy as np


class RandomAgent(Agent):
    def __init__(
            self,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)


    def choose_action(self, observation):
        rand = np.random.random(9)
        
        return self._unnormalize(rand)
    
    def learn(self):
        return
    
