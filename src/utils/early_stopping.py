from typing import Any
import torch

class EarlyStopping:
    def __init__(self, patience, delta: float = 1e-3) -> None:
        self.patience = patience
        self.counter = 0
        self.delta = delta
        self.last_loss = None

    def __call__(self, loss: torch.Tensor) -> bool:
        self.update_counter(loss)
        self.last_loss = loss
        return self.should_stop()    
    
    def update_counter(self, loss: torch.Tensor) -> None:
        if not self.change_smaller_than_delta(loss):
            self.counter += 1
        else:
            self.counter = 0
    
    def should_stop(self) -> bool:
        return self.counter == self.patience

    def change_smaller_than_delta(self, current_loss: torch.Tensor) -> bool:
        return torch.norm(self.last_loss - current_loss) < self.delta if self.has_last_loss() else True

    def has_last_loss(self) -> bool:
        return self.last_loss is not None
    