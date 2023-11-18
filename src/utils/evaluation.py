import torch
from torch import Tensor
from torch import nn
from typing import List

from torch.utils.data import DataLoader


def compute_loss_on(dataloader: DataLoader, model: nn.Module, loss_function, device: torch.device = 'cpu'):
    """
    Compute the loss on a torch DataLoader using a torch model and torch loss function.
    """
    model.to(device)
    
    running_loss = 0
    with torch.no_grad():
        for i, (inputs, true_values) in enumerate(dataloader):

            inputs = inputs.to(device)
            true_values = true_values.to(device)

            outputs = model(inputs)

            running_loss += loss_function(outputs, true_values)
    return running_loss / (i + 1)


def compute_predictions(test_dataloader: DataLoader, model: nn.Module, device: torch.device = 'cpu') -> List[Tensor]:
    """
    Compute the predictions on a dataloader using a model.
    The model is switched to eval mode in this function.
    """
    model.to(device)

    prediction_batches = []
    ground_truth_batches = []
    model.eval()
    with torch.no_grad():
        for _, (inputs, true_values) in enumerate(test_dataloader):
            inputs, true_values = inputs.to(device), true_values.to(device)
            prediction_batches.append(model(inputs))
            ground_truth_batches.append(true_values)
    
    return torch.cat(prediction_batches, axis=0), torch.cat(ground_truth_batches, axis=0)


def compute_losses_from(predictions, ground_truths, loss_function):
    """
    Compute the losses from model predictions and the ground truth using a predefined torch loss function.
    """
    return loss_function(predictions, ground_truths)  
