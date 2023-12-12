import sys; sys.path.insert(0, '../')

from utils.evaluation import compute_loss_on

import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
from pathlib import Path
from ray import tune, train as ray_train
from typing import List, Optional
from umap import UMAP
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x += self.pe[:x.size(0)]
        return self.dropout(x)
    
        
class TransformerEncoderModel(nn.Module):
    def __init__(self, num_heads: int, model_dim: int, feedforward_hidden_dim: int,
                 num_encoder_layers: int, output_dim: int, transformer_dropout: float = 0.1, pos_encoder_dropout: float = 0.25,
                 downprojection: bool = False, projection_num_neighbors: int = 5, activation: nn.Module = nn.ReLU) -> None:
        super().__init__()
        self.model_type = 'Transformer'
        self.total_epochs = 0
        self.model_dim = model_dim
        self.downprojection = downprojection
        if self.downprojection:
            self.create_projection(projection_num_neighbors)
            self.head = nn.Linear(model_dim, output_dim)
            self.head_activation = activation()
        encoder_layers = TransformerEncoderLayer(model_dim, num_heads, feedforward_hidden_dim, transformer_dropout, activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)   
        self.pos_encoder = PositionalEncoding(model_dim, pos_encoder_dropout)

    def create_projection(self, projection_num_neighbors: int):
        self.projection_function = UMAP(n_components=self.model_dim, n_neighbors=projection_num_neighbors)            

    def forward(self, source: Tensor, source_msk: Tensor = None) -> Tensor:
        # expect input shape to be (S, N, E) with S being the sequence length, N batch size and, E the input dimensionality
        source = self.project(source)
        source = self.pos_encoder(source)
        source = self.transformer_encoder(source, source_msk)
        if self.downprojection:
            source = self.head(source)
            return self.head_activation(source)
        return source
    
    def project(self, source: Tensor) -> Tensor:
        if self.downprojection:
            return self.projection_function.transform(source)
        return source
    

def train_epoch(train_dataloader: DataLoader, model, loss_function, optimizer, lr_scheduler,
                device: torch.device, report_interval: int = 1000):
    
    running_loss = 0
    last_loss = 0
    
    for i, (inputs, true_values) in enumerate(train_dataloader):
        
        inputs = inputs.to(device)
        true_values = true_values.to(device)
    
        inputs_shape, true_values_shape = inputs.size(), true_values.size()
        inputs = inputs.view(inputs_shape[1], inputs_shape[0], inputs_shape[2])
        true_values = true_values.view(true_values_shape[1], true_values_shape[0], true_values_shape[2])
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


def train(epochs: int, train_dataloader: DataLoader, validation_dataloader: DataLoader, model: nn.Module,
           loss_function, optimizer, lr_scheduler, checkpoint_path: Optional[Path], device: torch.device = 'cpu', 
           report_interval: int = 1000, tune: bool = False) -> nn.Module:
    
    best_val_loss = float("inf")

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
            avg_val_loss = compute_loss_on(validation_dataloader, model, loss_function, device=device)

        if not tune:
            print(f"Loss on train: {avg_loss}, loss on validation: {avg_val_loss}")

        model.total_epochs += 1

        if checkpoint_path != None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss            
            
            torch.save(model.state_dict(), checkpoint_file)

        if tune:
            ray_train.report(metrics={ "loss": float(avg_val_loss) })    
            
    return model