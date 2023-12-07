import sys; sys.path.insert(0, '../')

from utils.evaluation import compute_loss_on

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from pathlib import Path
from ray import train as ray_train
from typing import List

feature_columns = [
    'left_boom_base_yaw_joint', 
    'left_boom_base_pitch_joint',
    'left_boom_main_prismatic_joint',
    'left_boom_second_roll_joint',
    'left_boom_second_yaw_joint',
    'left_boom_top_pitch_joint'
]

label_columns = [
    'cable1_lowest_point',
    'cable2_lowest_point',
    'cable3_lowest_point'
]

class FullyConnected(nn.Module):
    def __init__(self, flattened_input_dim: int, intermediate_dims: List[int],
                 output_dim: int, dropout: float = 0.25, hidden_activation: str = 'relu') -> None:
        super().__init__()
        self.total_epochs = 0

        if hidden_activation == 'relu':
            hidden_activation_class = nn.ReLU
        elif hidden_activation == 'tanh':
            hidden_activation_class = nn.Tanh
        else:
            raise ValueError("Can not interfer hidden activation")

        self.flatten = nn.Flatten()
        self.hidden = nn.Sequential()
        for i, dim in enumerate(intermediate_dims):
            if i == 0:
                self.hidden.add_module(f"linear_{i+1}", nn.Linear(flattened_input_dim, dim))
            else:
                self.hidden.add_module(f"linear_{i+1}", nn.Linear(intermediate_dims[i-1], dim))
            self.hidden.add_module(f"hidden_activation_{i+1}", hidden_activation_class())
            self.hidden.add_module(f"dropout_{i+2}", nn.Dropout(dropout))

        self.last = nn.Linear(intermediate_dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.hidden(x)
        return self.last(x)
    

def get_optimizer_function(model: nn.Module, learning_rate: float) -> torch.optim:
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def get_loss_function() -> nn.Module:
    return torch.nn.MSELoss()


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
          checkpoint_path: Path, device: torch.device = 'cpu', report_interval: int = 1000, tune: bool = False) -> nn.Module:

    best_val_loss = float("inf")

    checkpoint_file = checkpoint_path / "checkpoint.pt"

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.to(device)

    if checkpoint_file.exists():
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
    
        if avg_val_loss < best_val_loss or tune:
            best_val_loss = avg_val_loss            
            
            torch.save(model.state_dict(), checkpoint_file)

        if tune:
            ray_train.report(metrics={ "loss": float(avg_val_loss) })
            
    return model   