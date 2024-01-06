from typing import Optional, Tuple
from pathlib import Path

import torch
import numpy as np

from ray import tune, train as ray_train
from torch import nn, Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import DataLoader

from models.transformer import PositionalEncoding, TransformerEncoderModel
from utils.early_stopping import EarlyStopping


class TransformerDecoderModel(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, feedforward_dim: int, num_decoder_layers: int, 
                 pos_encoder: PositionalEncoding, transformer_dropout: float = 0.25, activation: nn.Module = nn.ReLU) -> None:
        super().__init__()
        self.model_type = 'Transformer'
        self.total_epochs = 0
        self.model_dim = model_dim
        decoder_layer = TransformerDecoderLayer(model_dim, num_heads, dim_feedforward=feedforward_dim, dropout=transformer_dropout, activation=activation())
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.pos_encoder = pos_encoder

    def forward(self, memory: Tensor, target: Tensor, target_mask: Tensor = None) -> Tensor:
        # expect input shape to be (S, N, E) with S being the sequence length, N batch size and, E the input dimensionality
        if target_mask is None:
            target_mask = nn.Transformer.generate_square_subsequent_mask(target.shape[0])
        target = self.pos_encoder(target)
        return self.decoder(memory, target, tgt_mask=target_mask)
    

class TransformerModel(nn.Module):
    def __init__(self, encoder: TransformerEncoderModel, decoder: TransformerDecoderModel) -> None:
        super().__init__()
        if encoder.model_dim != decoder.model_dim:
            raise ValueError("Both encoder and decoder must have the same model dimension!")
        self.model_type = 'Transformer'
        self.total_epochs = 0
        self.encoder = encoder
        self.decoder = decoder
        self.encoder.eval()

    def forward(self, source: Tensor, target: Tensor = None, source_mask: Tensor = None, 
                target_mask: Tensor = None) -> Tensor:
        # expect input shape to be (S, N, E) with S being the sequence length, N batch size and, E the input dimensionality
        # target_mask masks out all values right of the diagonal such that information from the target sequence cant bleed into the left hand side at training time
        if self.training:
            prediction = self.forward_train(source, target, source_mask, target_mask)
        else:
            prediction = self.generate(source)

        return prediction
    
    def forward_train(self, source: Tensor, target: Tensor, source_mask: Tensor = None, 
                      target_mask: Tensor = None) -> Tensor:
        if target is None:
            raise ValueError("In train mode a target sequence has to be provided!")
        
        memory = self.encoder(source, source_mask)
        return self.decoder(memory, target, target_mask)
    
    def generate(self, source: Tensor):
        memory = self.encoder(source)
        target = torch.zeros_like(memory)

        for i in range(memory.shape[0]):
            pred = self.decoder(memory, target)
            if i + 1 < target.shape[0]:
                target[i + 1] = pred[i]
            else:
                target = torch.cat([target, pred[i].unsqueeze(0)], dim=0)
        return target[1:]
    
    @property
    def downprojection(self) -> bool:
        return self.encoder.downprojection
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()


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

        model.train()
        avg_loss = train_epoch(train_dataloader, model, loss_function, optimizer, lr_scheduler, device, report_interval)
        model.eval()

        running_loss = 0
        i = 0
        with torch.no_grad():
            for (input_trajectories, true_value_trajectories) in validation_dataloader:
                input_trajectories = input_trajectories.view(input_trajectories.shape[1], input_trajectories.shape[2], input_trajectories.shape[0], input_trajectories.shape[3])
                true_value_trajectories = true_value_trajectories.view(true_value_trajectories.shape[1], true_value_trajectories.shape[2], true_value_trajectories.shape[0], true_value_trajectories.shape[3])
                for (inputs, true_values) in zip(input_trajectories, true_value_trajectories):
                    inputs = inputs.to(device)
                    true_values = true_values.to(device)
                    outputs = model(inputs)
                    running_loss += loss_function(outputs, true_values)
                    i += 1
            avg_val_loss = running_loss / (i + 1)

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


def train_epoch(train_dataloader: DataLoader, model, loss_function, optimizer, lr_scheduler, 
                device: torch.device, report_interval: int = 250):
    
    running_loss = 0
    last_loss = 0
    
    for i, (input_trajectories, true_value_trajectories) in enumerate(train_dataloader):
        input_trajectories = input_trajectories.view(input_trajectories.shape[1], input_trajectories.shape[2], input_trajectories.shape[0], input_trajectories.shape[3])
        true_value_trajectories = true_value_trajectories.view(true_value_trajectories.shape[1], true_value_trajectories.shape[2], true_value_trajectories.shape[0], true_value_trajectories.shape[3])
        for (inputs, true_values) in zip(input_trajectories, true_value_trajectories):
            inputs = inputs.to(device)
            true_values = true_values.to(device)
        
            inputs_shape, true_values_shape = inputs.size(), true_values.size()
            inputs = inputs.view(inputs_shape[1], inputs_shape[0], inputs_shape[2])
            true_values = true_values.view(true_values_shape[1], true_values_shape[0], true_values_shape[2])
            optimizer.zero_grad()
            outputs = model(inputs, target=true_values)
            loss = loss_function(outputs, true_values)
            running_loss += loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            i += 1
        
            if i % report_interval == report_interval - 1:
                last_loss = running_loss / report_interval
                print(f"batch {i + 1}, Mean Squared Error: {last_loss}")
                running_loss = 0

    return last_loss