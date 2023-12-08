import torch
from torch import nn


def get_activation(activation: str):
    """
    Convert a torch activation provided in a string to a pytorch module.
    """
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        raise ValueError("Activation not supported")