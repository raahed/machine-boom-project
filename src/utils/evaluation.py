import torch
from torch import Tensor
from torch import nn
from typing import Tuple

from torch.utils.data import DataLoader


def compute_loss_on(dataloader: DataLoader, model: nn.Module, loss_function, reshape: bool = False, device: torch.device = 'cpu'):
    """
    Compute the loss on a torch DataLoader using a torch model and torch loss function.
    """
    model.to(device)
    
    running_loss = 0
    with torch.no_grad():
        for i, (inputs, true_values) in enumerate(dataloader):

            inputs = inputs.to(device)
            true_values = true_values.to(device)

            if reshape:
                inputs_shape, true_values_shape = inputs.size(), true_values.size()
                inputs = inputs.view(inputs_shape[1], inputs_shape[2], inputs_shape[0], inputs_shape[3])
                true_values = true_values.view(true_values_shape[1], true_values_shape[2], true_values_shape[0], true_values_shape[3])
        
            outputs = model(inputs)
            running_loss += loss_function(outputs, true_values)
    return running_loss / (i + 1)


def compute_predictions(test_dataloader: DataLoader, model: nn.Module, device: torch.device = 'cpu') -> Tuple[Tensor, Tensor]:
    """
    Compute the predictions on a dataloader using a model.
    The model is switched to eval mode in this function.
    """
    model.to(device)

    prediction_batches = []
    ground_truth_batches = []
    model.eval()
    with torch.no_grad():
        for inputs, true_values in test_dataloader:
            inputs, true_values = inputs.to(device), true_values.to(device)
            predictions = model(inputs)

            prediction_batches.append(predictions)
            ground_truth_batches.append(true_values)
    
    return torch.cat(prediction_batches, axis=0), torch.cat(ground_truth_batches, axis=0)


def compute_sliding_window_predictions(test_dataloader: DataLoader, model: nn.Module, device: torch.device = 'cpu') -> Tuple[Tensor, Tensor]:
    """
    Compute the sliding window predictions of the dataloader.
    The dataloader must return the following tuple: inputs, true_values, last_nonzero_index. 
    Inputs is the padded input sequence (shape: N x S x I), true_values (shape: N x S x O) is the padded sequence of ground truth values and last_nonzero_index is the last nonzero index of those sequences.
    N is the batch size, S the sequence length, I the input dimension and O the output dimension.
    The model is switched to eval mode in this function.
    """
    model.to(device)

    prediction_batches = []
    ground_truth_batches = []
    model.eval()
    with torch.no_grad():
        for inputs, true_values, last_indices in test_dataloader:
            inputs, true_values = inputs.to(device), true_values.to(device)
            predictions = model(inputs)
            
            prediction_batch = []
            for prediction, last_index in zip(predictions, last_indices):
                prediction_batch.append(prediction[last_index])

            prediction_batches.append(torch.stack(prediction_batch))
            ground_truth_batches.append(true_values)
    
    return torch.cat(prediction_batches, axis=0), torch.cat(ground_truth_batches, axis=0)


def compute_losses_from(predictions, ground_truths, loss_function):
    """
    Compute the losses from model predictions and the ground truth using a predefined torch loss function. 
    """
    return loss_function(predictions, ground_truths)  
