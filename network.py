import torch
import torch.nn as nn

from typing import Union, List

class NeuralNetwork(nn.Module):

    def __init__(
            self,
            in_size: int,
            hidden: Union[int, List[int]],
            out_size: int,
            device: torch.device,
    ):
        pass
