import sys; sys.path.insert(0, '../')

from typing import Optional, Tuple
from pathlib import Path

import torch
import numpy as np

from umap import UMAP
from torch import nn, Tensor
from torch.utils.data import DataLoader
from ray import tune, train as ray_train

from models.transformer import TransformerEncoderModel
from utils.evaluation import compute_loss_on
from utils.early_stopping import EarlyStopping


class ParallelEncoderModel(nn.Module):
    def __init__(self, num_decoders: int, num_heads: int, model_dim: int, feedforward_hidden_dim: int, output_dim: int,
                 num_encoder_layers: int = 6, transformer_dropout: float = 0.1, pos_encoder_dropout: float = 0.25, activation: nn.Module = nn.ReLU,
                 projection_function: UMAP = None) -> None:
        super().__init__()
        self.model_type = 'Transformer'
        self.total_epochs = 0
        self.encoder = TransformerEncoderModel(num_heads, model_dim, feedforward_hidden_dim, 
                                               num_encoder_layers, transformer_dropout, pos_encoder_dropout, activation=activation, projection_function=projection_function)
        self.decoders = nn.ModuleList([nn.Linear(model_dim, output_dim) for i in range(num_decoders)])
        self.activation = nn.ReLU()

    def forward(self, source: Tensor, source_mask: Tensor = None) -> Tensor:
        decoded = []
        for i, decoder in enumerate(self.decoders):
            trajectory_source = source[i, :, :, :]
            trajectory_source = self.encoder(trajectory_source, source_mask)
            decoded_trajectory = decoder(trajectory_source)
            decoded_trajectory = self.activation(decoded_trajectory)
            decoded.append(decoded_trajectory)
        return torch.stack(decoded, dim=0)
    
    @property
    def downprojection(self):
        return self.encoder.downprojection is not None
    
    @property
    def pos_encoder(self):
        return self.encoder.pos_encoder
    

def train_epoch(train_dataloader: DataLoader, model, loss_function, optimizer, lr_scheduler,
                device: torch.device, report_interval: int = 10):
    
    running_loss = 0
    last_loss = 0
    
    for i, (inputs, true_values) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        true_values = true_values.to(device)
    
        inputs_shape, true_values_shape = inputs.size(), true_values.size()
        inputs = inputs.view(inputs_shape[1], inputs_shape[2], inputs_shape[0], inputs_shape[3])
        true_values = true_values.view(true_values_shape[1], true_values_shape[2], true_values_shape[0], true_values_shape[3])
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, true_values)
        running_loss += loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
    
        if i % report_interval == report_interval - 1:
            last_loss = running_loss / report_interval
            print(f"batch {i + 1}, Mean Squared Error: {last_loss}")
            running_loss = 0

    return last_loss


def train(epochs: int, train_dataloader: DataLoader, validation_dataloader: DataLoader, model: TransformerEncoderModel,
           loss_function, optimizer, lr_scheduler, checkpoint_path: Optional[Path], device: torch.device = 'cpu', 
           report_interval: int = 1000, tune: bool = False, early_stopping: Optional[EarlyStopping] = None) -> Tuple[nn.Module, np.ndarray]:
    
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
        avg_loss = train_epoch(train_dataloader, model, loss_function, optimizer, lr_scheduler, device, report_interval)
        model.eval()

        with torch.no_grad():
            avg_val_loss = compute_loss_on(validation_dataloader, model, loss_function, reshape=True, device=device)

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