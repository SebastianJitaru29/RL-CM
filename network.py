import torch
import torch.nn as nn
from typing import Union, List, Tuple

#TODO fix easier enumeration

"""
Initial idea of networks M and D:
 - predict diagonal -> non-zeroing activation
 - predict bottom left -> no activation
 : possible to use upsampling instead?

V:
 - Single output neuron

A:
 - N*W output, reshape to matrix
"""

class NeuralNetwork(nn.Module):
    """"""

    def __init__(
            self,
            input_size: int,
            output_size: Union[int, Tuple[int, int]],
            hidden: Union[int, List[int]],
            device: torch.device,
            *_,
            leaky_alpha=0.05,
        ):
        """"""
        super(NeuralNetwork, self).__init__()

        # assert(isinstance(output_size, int) 
        #        or isinstance(output_size, tuple[int, int]))
        
        self.act = nn.LeakyReLU(leaky_alpha)

        self.layers = []
        self.out = None
        self.device = device

        if isinstance(hidden, int):
            self._create_network_single(input_size, hidden, output_size)
        else:
            self._create_network_list(input_size, hidden, output_size)

        self.layers = nn.ModuleList(self.layers)
        self.layers = self.layers.to(self.device)
        self.out = self.out.to(self.device)

    def forward(self, x):
        """"""
        ret_dev = 'cpu' if x.get_device() == -1 else self.device

        x = x.to(self.device)
        for layer in self.layers:
            x = self.act(layer(x))

        x = self.out(x)
        x = x.to(ret_dev)

        return x
    
    def _create_network_single(self, n_in: int, n_hid: int, n_out: int):
        self.layers.append(nn.Linear(n_in, n_hid))
        if isinstance(n_out, int):
            self.out = nn.Linear(n_hid, n_out)
        else:
            self.out = nn.Sequential(
                nn.Linear(n_hid, n_out[0] * n_out[1]),
                nn.Unflatten(1, n_out)
            )

    def _create_network_list(self, n_in: int, hid: List[int], n_out: int):
        self.layers.append(nn.Linear(n_in, hid[0]))
        for idx in range(1, len(hid)):
            self.layers.append(nn.Linear(hid[idx-1], hid[idx]))
        if isinstance(n_out, int):
            self.out = nn.Linear(hid[-1], n_out)
        else:
            self.out = nn.Sequential(
                nn.Linear(hid[-1], n_out[0] * n_out[1]),
                nn.Unflatten(1, n_out)
            )
