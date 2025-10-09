"""
Transformer-based regressor for time series prediction.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer inputs."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        return x + self.pe[:x.size(0), :]


class TransformerRegressor(nn.Module):
    """
    Transformer-based regressor for time series prediction.
    
    Features:
    - Causal masking to prevent look-ahead bias
    - Symbol embeddings for cross-sectional learning
    - Positional encoding for temporal structure
    - Multi-head attention for feature interactions
    """
    
    def __init__(
        self,
        input_dim: int,
        num_symbols: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        use_symbol_embeddings: bool = True,
        use_sector_embeddings: bool = False,
        num_sectors: int = 0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_symbols = num_symbols
        self.d_model = d_model
        self.use_symbol_embeddings = use_symbol_embeddings
        self.use_sector_embeddings = use_sector_embeddings
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Symbol embeddings
        if use_symbol_embeddings:
            self.symbol_embedding = nn.Embedding(num_symbols, d_model)
        
        # Sector embeddings (optional)
        if use_sector_embeddings and num_sectors > 0:
            self.sector_embedding = nn.Embedding(num_sectors, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # (seq_len, batch, d_model)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask to prevent look-ahead bias."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def forward(
        self, 
        x: torch.Tensor, 
        symbols: torch.Tensor,
        sectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch_size, seq_len, input_dim)
            symbols: Symbol indices of shape (batch_size,)
            sectors: Sector indices of shape (batch_size,) - optional
            
        Returns:
            Predictions of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input features
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add symbol embeddings
        if self.use_symbol_embeddings:
            symbol_emb = self.symbol_embedding(symbols)  # (batch_size, d_model)
            # Broadcast to sequence length
            symbol_emb = symbol_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, d_model)
            x = x + symbol_emb
        
        # Add sector embeddings (optional)
        if self.use_sector_embeddings and sectors is not None:
            sector_emb = self.sector_embedding(sectors)  # (batch_size, d_model)
            sector_emb = sector_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, d_model)
            x = x + sector_emb
        
        # Transpose for transformer: (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len).to(x.device)
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=None, mask=causal_mask)
        
        # Take the last timestep for prediction
        x = x[-1]  # (batch_size, d_model)
        
        # Apply output head
        output = self.head(x)  # (batch_size, 1)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor, symbols: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input features of shape (batch_size, seq_len, input_dim)
            symbols: Symbol indices of shape (batch_size,)
            
        Returns:
            Attention weights of shape (batch_size, nhead, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input features
        x = self.input_projection(x)
        
        # Add symbol embeddings
        if self.use_symbol_embeddings:
            symbol_emb = self.symbol_embedding(symbols)
            symbol_emb = symbol_emb.unsqueeze(1).expand(-1, seq_len, -1)
            x = x + symbol_emb
        
        # Transpose for transformer
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        
        # Get attention weights from first layer
        first_layer = self.transformer.layers[0]
        x_norm = first_layer.norm1(x)
        
        # Multi-head attention
        attn_output, attn_weights = first_layer.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=self.create_causal_mask(seq_len).to(x.device),
            need_weights=True
        )
        
        return attn_weights


def create_transformer_model(
    input_dim: int,
    num_symbols: int,
    config: dict
) -> TransformerRegressor:
    """
    Create a transformer model with the given configuration.
    
    Args:
        input_dim: Number of input features
        num_symbols: Number of unique symbols
        config: Model configuration dictionary
        
    Returns:
        Configured TransformerRegressor
    """
    return TransformerRegressor(
        input_dim=input_dim,
        num_symbols=num_symbols,
        d_model=config.get('d_model', 64),
        nhead=config.get('nhead', 8),
        num_layers=config.get('num_layers', 3),
        dim_feedforward=config.get('dim_feedforward', 256),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 100),
        use_symbol_embeddings=config.get('use_symbol_embeddings', True),
        use_sector_embeddings=config.get('use_sector_embeddings', False),
        num_sectors=config.get('num_sectors', 0)
    )
