import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

# dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        # create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # register buffer (not a parameter, but should be saved and loaded with the model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)
    
class TimeSeriesTransformer(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        num_classes: int,
        d_model: int = 64, 
        nhead: int = 4, 
        num_layers: int = 2, 
        dim_feedforward: int = 256, 
        dropout: float = 0.1
    ):
        super(TimeSeriesTransformer, self).__init__()  
        self.input_dim = input_dim
        self.seq_len = 9  
        self.feature_dim = input_dim // self.seq_len
        # input projection
        self.input_projection = nn.Linear(self.feature_dim, d_model)
        # positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        # transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        # final classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    def forward(self, x):
        batch_size = x.size(0)
        # reshape input to [batch_size, seq_len, feature_dim]
        x = x.view(batch_size, self.seq_len, self.feature_dim)
        # project input to d_model dimensions
        x = self.input_projection(x)
        # add positional encoding
        x = self.positional_encoding(x)
        # apply transformer encoder
        x = self.transformer_encoder(x)
        # global average pooling over time dimension
        x = torch.mean(x, dim=1)
        # classification
        return self.classifier(x)