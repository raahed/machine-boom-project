import torch
from torch import nn, Tensor
from typing import List

class FullyConnected(nn.Module):
    def __init__(self, flattened_input_dim: int, intermediate_dims: List[int],
                 output_dim: int, dropout: float = 0.25, hidden_activation = nn.ReLU) -> None:
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
    
def get_optimizer_function(model: nn.Module, learning_rate: float) -> torch.optim:
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def get_loss_function() -> nn.Module:
    return torch.nn.MSELoss()

