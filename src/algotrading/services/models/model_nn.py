"""Neural network models for return prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np

from ...core.types import ModelFamily, TrainingConfig


class LSTMRegressor(nn.Module):
    """LSTM-based regressor for return prediction."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 embedding_dim: int = 12,
                 dropout: float = 0.2,
                 num_symbols: int = 6):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_symbols = num_symbols
        
        # Symbol embedding
        self.symbol_embedding = nn.Embedding(num_symbols, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim + embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                elif 'linear' in name:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, symbol_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch_size, seq_len, input_dim)
            symbol_ids: Symbol indices of shape (batch_size,)
            
        Returns:
            Predictions of shape (batch_size,)
        """
        batch_size, seq_len, _ = x.shape
        
        # Get symbol embeddings
        symbol_emb = self.symbol_embedding(symbol_ids)  # (batch_size, embedding_dim)
        
        # Expand symbol embeddings to sequence length
        symbol_emb_expanded = symbol_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate features with symbol embeddings
        x_with_emb = torch.cat([x, symbol_emb_expanded], dim=-1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x_with_emb)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Generate prediction
        prediction = self.head(last_output).squeeze(-1)
        
        return prediction
    
    def predict(self, x: torch.Tensor, symbol_ids: torch.Tensor) -> torch.Tensor:
        """Make predictions in eval mode."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, symbol_ids)
    
    def get_embeddings(self, symbol_ids: torch.Tensor) -> torch.Tensor:
        """Get symbol embeddings."""
        return self.symbol_embedding(symbol_ids)


class SimpleLSTM(nn.Module):
    """Simplified LSTM for Colab demo."""
    
    def __init__(self, input_dim: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.head(last_output).squeeze(-1)


class ModelFactory:
    """Factory for creating models."""
    
    @staticmethod
    def create_model(model_family: ModelFamily, 
                    input_dim: int, 
                    config: TrainingConfig,
                    num_symbols: int = 6) -> nn.Module:
        """Create a model instance."""
        
        if model_family == ModelFamily.LSTM:
            return LSTMRegressor(
                input_dim=input_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                embedding_dim=config.embedding_dim,
                dropout=config.dropout,
                num_symbols=num_symbols
            )
        else:
            raise ValueError(f"Unsupported model family: {model_family}")
    
    @staticmethod
    def create_simple_model(input_dim: int, hidden_size: int = 64, num_layers: int = 2) -> nn.Module:
        """Create a simple model for Colab demo."""
        return SimpleLSTM(input_dim, hidden_size, num_layers)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Get model summary information."""
    total_params = count_parameters(model)
    
    # Count parameters by layer
    layer_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_params[name] = param.numel()
    
    return {
        "total_parameters": total_params,
        "layer_parameters": layer_params,
        "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
    }
