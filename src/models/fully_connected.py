import sys; sys.path.insert(0, '../')

from utils.evaluation import compute_loss_on
from utils.early_stopping import EarlyStopping

import torch
import numpy as np

from torch import nn, Tensor
from torch.utils.data import DataLoader
from pathlib import Path
from ray import train as ray_train
from typing import List, Optional

class FullyConnected(nn.Module):
    def __init__(self, flattened_input_dim: int, intermediate_dims: List[int],
                 output_dim: int, dropout: float = 0.25, hidden_activation: nn.Module = nn.ReLU) -> None:
        super().__init__()
        self.total_epochs = 0
        self.flatten = nn.Flatten()
        self.hidden = nn.Sequential()
        for i, dim in enumerate(intermediate_dims):
            if i == 0:
                self.hidden.add_module(f"linear_{i+1}", nn.Linear(flattened_input_dim, dim))
            else:
                self.hidden.add_module(f"linear_{i+1}", nn.Linear(intermediate_dims[i-1], dim))
            self.hidden.add_module(f"hidden_activation_{i+1}", hidden_activation())
            self.hidden.add_module(f"dropout_{i+2}", nn.Dropout(dropout))

        self.last = nn.Linear(intermediate_dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.hidden(x)
        return self.last(x)
    

def train_epoch(train_dataloader: DataLoader, model: nn.Module, loss_function, optimizer, 
                device: torch.device, report_interval: int = 1000) -> float:

    running_loss = 0
    last_loss = 0
    
    for i, (inputs, true_values) in enumerate(train_dataloader):

        inputs = inputs.to(device)
        true_values = true_values.to(device)
                
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, true_values)
        running_loss += loss
        loss.backward()
        optimizer.step() 
    
    if i % report_interval == report_interval - 1:
        last_loss = running_loss / report_interval

        print(f"batch {i + 1}, Mean Squared Error: {last_loss}")
            
        running_loss = 0
    
    return last_loss 


def train(epochs: int, train_dataloader: DataLoader, validation_dataloader: DataLoader, model: nn.Module, loss_function, optimizer, 
          checkpoint_path: Optional[Path], device: torch.device = 'cpu', report_interval: int = 1000, tune: bool = False, 
          early_stopping: Optional[EarlyStopping] = None) -> nn.Module:

    best_val_loss = float("inf")
    val_losses = []

    if checkpoint_path != None:
        checkpoint_file = checkpoint_path / "checkpoint.pt"

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.to(device)

    if checkpoint_path != None and checkpoint_file.exists():
        model_state = torch.load(checkpoint_file)
        model.load_state_dict(model_state)

    for epoch in range(model.total_epochs, epochs):
        if not tune:
            print(f"Epoch: {epoch + 1}")

        model.train(True)
        avg_loss = train_epoch(train_dataloader, model, loss_function, optimizer, device, report_interval)
        model.eval()

        with torch.no_grad():
            avg_val_loss = compute_loss_on(validation_dataloader, model, loss_function, device=device)

        if not tune:
            print(f"Loss on train: {avg_loss}, loss on validation: {avg_val_loss}")

        model.total_epochs += 1
    
        if checkpoint_path != None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss            
            
            torch.save(model.state_dict(), checkpoint_file)

        if tune:
            ray_train.report(metrics={ "loss": float(avg_val_loss) })
        
        val_losses.append(float(avg_val_loss))

        if early_stopping != None and early_stopping(avg_val_loss):
            return model, np.array(val_losses)
            
    return model, np.array(val_losses)
