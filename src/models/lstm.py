import sys; sys.path.insert(0, '../')

from utils.evaluation import compute_loss_on
from utils.early_stopping import EarlyStopping

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from pathlib import Path
from ray import tune, train as ray_train
from typing import List, Optional

class DecoderLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, dropout_lstm: float = 0.25, dropout_final: float = 0.25,
                 num_lstm_layers: int = 1, bidirectional: bool = False, proj_size: int = 0) -> None:
        super().__init__()
        self.total_epochs = 0
        self.d = 2 if bidirectional else 1
        self.proj_size = proj_size
        self.num_lstm_layers = num_lstm_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_lstm_layers, dropout=dropout_lstm, bidirectional=bidirectional, proj_size=proj_size)
        
        if proj_size == 0:
            self.final_dropout = nn.Dropout(dropout_final)
            self.out = nn.Linear(hidden_dim * self.d, out_dim)
        
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[1]
        # expect x to be of shape (sequence_length, batch_size, input_dim)
        h0 = torch.randn(self.d * self.num_lstm_layers, batch_size, self.proj_size if self.proj_size > 0 else self.hidden_dim).to(x.device)
        c0 = torch.randn(self.d * self.num_lstm_layers, batch_size, self.hidden_dim).to(x.device)
        # output shape is (sequence_length, batch_size, d * hidden_dim)
        output, (hn, cn) = self.lstm(x, (h0, c0))

        if self.proj_size != 0:
            return output

        output = self.final_dropout(output)
        return self.out(output)


def train_epoch(train_dataloader: DataLoader, model, loss_function, optimizer,
                device: torch.device, report_interval: int = 128):
    
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
    
        if i % report_interval == report_interval - 1:
            last_loss = running_loss / report_interval

            print(f"batch {i + 1}, Mean Squared Error: {last_loss}")
            
            running_loss = 0
    
    return last_loss


def train(epochs: int, train_dataloader: DataLoader, validation_dataloader: DataLoader, model: nn.Module,
           loss_function, optimizer, checkpoint_path: Optional[Path], device: torch.device = 'cpu', 
           report_interval: int = 1000, tune: bool = False, early_stopping: Optional[EarlyStopping] = None) -> nn.Module:
    
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

        if early_stopping != None and early_stopping(avg_val_loss):
            return model
 
    return model