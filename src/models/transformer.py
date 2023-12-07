import math
import torch
from torch import nn, Tensor
from typing import List
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x += self.pe[:x.size(0)]
        return self.dropout(x)
    
class Transformer(nn.Module):
    def __init__(self, num_heads: int, model_dim: int, feedforward_hidden_dim: int, output_dim: int,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, transformer_dropout: float = 0.1, pos_encoder_dropout: float = 0.25):
        super().__init__()
        self.model_type = 'Transformer'
        self.total_epochs = 0
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, dim_feedforward=feedforward_hidden_dim, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=transformer_dropout)
        self.pos_encoder = PositionalEncoding(model_dim, pos_encoder_dropout)
        self.head = nn.Linear(model_dim, output_dim)

    def forward(self, source: Tensor, target: Tensor = None):
        # expect input shape to be (S, N, E) with S being the sequence length, N batch size and, E the input dimensionality
        # target_mask masks out all values right of the diagonal such that information from the target sequence cant bleed into the left hand side at training time
        target_mask = nn.Transformer.generate_square_subsequent_mask(source.shape[0], device=device)
        source = self.pos_encoder(source)
        return self.forward_train(source, target, target_mask) if target is not None else self.forward_inference(source, target_mask)
        
    def forward_train(self, source: Tensor, target: Tensor, target_mask: Tensor) -> Tensor:
        self.train()
        target = self.pos_encoder(target)
        target = self.transformer(source, target, tgt_mask=target_mask)
        return self.head(target)

    def forward_inference(self, source: Tensor, target_mask: Tensor) -> Tensor:
        self.eval()
        with torch.no_grad():
            target_pos_encoding = self.pos_encoder(torch.zeros_like(source, device=device))
            target = torch.zeros(source.shape[0] + 1, source.shape[1], source.shape[2], device=device)
            target[0, :, :] = torch.ones(source.shape[1:])
            for i, (source_batch, target_batch) in enumerate(zip(source, target)):
                out_target = self.transformer(source_batch, target_batch) 
                target[i + 1, :, :] = out_target + target_pos_encoding[i, :, :]
            return self.head(target[1:, :, :])    
        
class TransformerEncoderOnly(nn.Module):
    def __init__(self, num_heads: int, model_dim: int, feedforward_hidden_dim: int, output_dim: int,
                 num_encoder_layers: int = 6, transformer_dropout: float = 0.1, pos_encoder_dropout: float = 0.25) -> None:
        super().__init__()
        self.model_type = 'Transformer'
        self.total_epochs = 0
        encoder_layers = TransformerEncoderLayer(model_dim, num_heads, feedforward_hidden_dim, transformer_dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)   
        self.pos_encoder = PositionalEncoding(model_dim, pos_encoder_dropout)
        self.head = nn.Linear(model_dim, output_dim)

    def forward(self, source: Tensor, source_msk: Tensor = None) -> Tensor:
        # expect input shape to be (S, N, E) with S being the sequence length, N batch size and, E the input dimensionality
        # target_mask masks out all values right of the diagonal such that information from the target sequence cant bleed into the left hand side at training time
        source = self.pos_encoder(source)
        source = self.transformer_encoder(source, source_msk)
        return self.head(source)
    
def get_optimizer_function(model: nn.Module, learning_rate: float) -> torch.optim:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)

def get_loss_function() -> nn.Module:
    return torch.nn.MSELoss()