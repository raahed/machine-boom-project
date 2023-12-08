import torch

def maximum_squared_error(output, target):
    return torch.sum(torch.max((output - target)** 2, dim=0)[0])


def get_loss_function():
    return torch.nn.MSELoss()
