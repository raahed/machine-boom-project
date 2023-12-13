import sys; sys.path.insert(0, '../')

from utils.evaluation import compute_loss_on

import math
import pickle

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
        activation_func = activation()
        if self.downprojection:
            self.create_projection(projection_num_neighbors)
        else:
            self.head = nn.Linear(model_dim, output_dim)
            self.head_activation = activation_func
        encoder_layers = TransformerEncoderLayer(model_dim, num_heads, dim_feedforward=feedforward_hidden_dim, 
                                                 dropout=transformer_dropout, activation=activation_func)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)   
        self.pos_encoder = PositionalEncoding(model_dim, pos_encoder_dropout)

    def create_projection(self, projection_num_neighbors: int):
        self.projection_function = UMAP(n_components=self.model_dim, n_neighbors=projection_num_neighbors)            

    def forward(self, source: Tensor, source_msk: Tensor = None) -> Tensor:
        # expect input shape to be (S, N, E) with S being the sequence length, N batch size and, E the input dimensionality
        source = self.project(source)
        source = self.pos_encoder(source)
        source = self.transformer_encoder(source, source_msk)
        if not self.downprojection:
            source = self.head(source)
            return self.head_activation(source)
        return source
    
    def project(self, source: Tensor) -> Tensor:
        if self.downprojection:
            s = source.size(0)
            n = source.size(1)
            device = source.device
            source = source.flatten(start_dim=0, end_dim=1).cpu()
            source = self.projection_function.transform(source)
            source = torch.from_numpy(source)
            source = source.view(s, n, source.shape[1])
            return source.to(device)
        return source
    
    def save(self, path: Path) -> None:
        torch.save(self.state_dict, path)
        if self.downprojection:
            projection_path = self.infer_projection_filepath(path)
            pickle.dump(self.projection_function, projection_path.open("wb"))

    def load(self, path: Path) -> None:
        model_state_dict = torch.load(path)
        if self.downprojection:
            projection_path = self.infer_projection_filepath(path)
            self.projection_function = pickle.load(projection_path.open("rb"))
        self.load_state_dict(model_state_dict)

    def infer_projection_filepath(self, path_to_model_dict: Path) -> Path:
        projection_filename = path_to_model_dict.stem + ".projection.pkl"
        projection_path = path_to_model_dict.parent / projection_filename
        return projection_path

    

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


def train(epochs: int, train_dataloader: DataLoader, validation_dataloader: DataLoader, model: TransformerEncoderModel,
           loss_function, optimizer, lr_scheduler, checkpoint_path: Optional[Path], device: torch.device = 'cpu', 
           report_interval: int = 1000, tune: bool = False) -> TransformerEncoderModel:
    
    best_val_loss = float("inf")

    if model.downprojection:
        print("Fitting downprojection!")
        train_downprojection(model.projection_function, train_dataloader)

    if checkpoint_path != None:
        checkpoint_file = checkpoint_path / "checkpoint.pt"

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.to(device)

    if checkpoint_path != None and checkpoint_file.exists():
        model.load(checkpoint_file)

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
            model.save(checkpoint_file)

        if tune:
            ray_train.report(metrics={ "loss": float(avg_val_loss) })    
            
    return model


def train_downprojection(projection: UMAP, train_dataloader: DataLoader) -> None:
    feature_vectors = []
    for features, labels in train_dataloader:
        features = torch.flatten(features, start_dim=0, end_dim=-2)
        feature_vectors.append(features)
    features = torch.concat(feature_vectors, dim=0)
    projection = projection.fit(features)
