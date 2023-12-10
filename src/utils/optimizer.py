import torch
from torch import nn, optim
from typing import Optional

# Warmup rate from https://nlp.seas.harvard.edu/annotated-transformer/

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def get_optimizer_function(optimizer: str, model: nn.Module, learning_rate: float, momentum: Optional[float] = 0) -> torch.optim:
    """
    Converts a string into a pytorch optimizer object.
    """
    optimizer = optimizer.lower()

    if optimizer == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer == "adagrad":
        return optim.Adagrad(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer function")
    

def get_learning_rate_scheduler(optimizer: torch.optim, model_size: int, warmup_steps: int, factor: float = 1) -> torch.optim:
    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda step: rate(step, model_size, factor, warmup_steps))