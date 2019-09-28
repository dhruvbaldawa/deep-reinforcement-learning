import itertools

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=()):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers(tuple): Tuple of hidden layers in the neural network
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        layers = (
            [nn.Linear(state_size, hidden_layers[0])] +
            [(nn.Linear(start_layer, end_layer))
                for start_layer, end_layer in zip(hidden_layers, hidden_layers[1:])
            ] +
            [nn.Linear(hidden_layers[-1], action_size)]
        )

        layers_with_relu = []
        for idx, layer in enumerate(layers):
            layers_with_relu.append(layer)

            # add ReLU between all layers except the last one
            if idx != (len(layers) - 1):
                layers_with_relu.append(nn.ReLU())

        self.layers = nn.Sequential(*layers_with_relu)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.layers(state)
