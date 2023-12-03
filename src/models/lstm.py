import torch
from torch import nn, Tensor
from typing import List

class DecoderLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, dropout_lstm: float = 0.25, dropout_final: float = 0.25,
                 num_lstm_layers: int = 1, bidirectional: bool = False) -> None:
        super().__init__()
        self.total_epochs = 0
        self.hidden_dim = hidden_dim
        self.d = 2 if bidirectional else 1
        self.num_lstm_layers = num_lstm_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_lstm_layers, dropout=dropout_lstm, bidirectional=bidirectional)
        self.final_dropout = nn.Dropout(dropout_final)
        self.out = nn.Linear(hidden_dim * self.d, out_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[1]
        # expect x to be of shape (sequence_length, batch_size, input_dim)
        h0 = torch.randn(self.d * self.num_lstm_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.randn(self.d * self.num_lstm_layers, batch_size, self.hidden_dim).to(x.device)
        # output shape is (sequence_length, batch_size, d * hidden_dim)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        output = self.final_dropout(output)
        return self.out(output)

def get_optimizer_function(model: nn.Module, learning_rate: float) -> torch.optim:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)

def get_loss_function() -> nn.Module:
    return torch.nn.MSELoss()