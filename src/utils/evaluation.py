import torch

from torch.utils.data import DataLoader


def compute_loss_on(dataloader: DataLoader, model, loss_function):
    running_loss = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, true_values = data
            outputs = model(inputs)
            running_loss += loss_function(outputs, true_values)
    return running_loss / (i + 1)


def compute_predictions(test_dataloader: DataLoader, model):
    prediction_batches = []
    ground_truth_batches = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, true_values) in enumerate(test_dataloader):
            prediction_batches.append(model(inputs))
            ground_truth_batches.append(true_values)
    
    return torch.cat(prediction_batches, axis=0), torch.cat(ground_truth_batches, axis=0)


def compute_losses_from(predictions, ground_truths, loss_function):
    return loss_function(predictions, ground_truths)  
