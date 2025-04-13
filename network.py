import torch
import torch.nn as nn
from typing import Union, List, Tuple

#TODO fix easier enumeration

class NeuralNetwork(nn.Module):
    """A Neural Network class.
    
    This class is nearly identical to the basic Neural Network
    class used in our model free controller.
    """

    def __init__(
            self,
            input_size: int,
            output_size: Union[int, Tuple[int, int]],
            hidden: Union[int, List[int]],
            device: torch.device,
            *_,
            leaky_alpha: float=0.05,
        ):
        """Initialize the neural network.

        @param input_size (int): The number of input features.
        @param output_size (Union[int, Tuple[int, int]]): the number of
            output features, if a tuple is provided a matrix is produced.
        @param hidden (Union[int, List[int]]): The number of hidden neurons
            if a list is provided each element defines one hidden layer.
        @param device (torch.device): The device to host the weights.
        *,
        @param leaky_alpha (float): default=0.05, the value for
            LeakyReLu's alpha.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network.
        
        @param x (torch.Tensor): The input for the network.
        @return (torch.Tensor): The output of the network.
        """
        ret_dev = 'cpu' if x.get_device() == -1 else self.device

        x = x.to(self.device)
        for layer in self.layers:
            x = self.act(layer(x))

        x = self.out(x)
        x = x.to(ret_dev)

        return x
    
    def _create_network_single(self, n_in: int, n_hid: int, n_out: int):
        """Helper function to create the network with a single hidden layer."""
        self.layers.append(nn.Linear(n_in, n_hid))
        if isinstance(n_out, int):
            self.out = nn.Linear(n_hid, n_out)
        else:
            self.out = nn.Sequential(
                nn.Linear(n_hid, n_out[0] * n_out[1]),
                nn.Unflatten(1, n_out)
            )

    def _create_network_list(self, n_in: int, hid: List[int], n_out: int):
        """Helper function to create the network with multiple hidden layers."""
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
