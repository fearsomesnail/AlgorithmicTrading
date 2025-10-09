"""
Model zoo for creating different types of models.
"""

from typing import Dict, Any, Union
from .model_nn import LSTMRegressor
from .transformer import TransformerRegressor, create_transformer_model


def build_model(
    model_type: str,
    input_dim: int,
    num_symbols: int,
    config: Dict[str, Any]
) -> Union[LSTMRegressor, TransformerRegressor]:
    """
    Build a model based on the specified type and configuration.
    
    Args:
        model_type: Type of model ('lstm' or 'transformer')
        input_dim: Number of input features
        num_symbols: Number of unique symbols
        config: Model configuration dictionary
        
    Returns:
        Configured model instance
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type.lower() == 'lstm':
        return LSTMRegressor(
            input_dim=input_dim,
            num_symbols=num_symbols,
            hidden_size=config.get('hidden_size', 32),
            num_layers=config.get('num_layers', 1),
            embedding_dim=config.get('embedding_dim', 12),
            dropout=config.get('dropout', 0.05)
        )
    elif model_type.lower() == 'transformer':
        return create_transformer_model(
            input_dim=input_dim,
            num_symbols=num_symbols,
            config=config
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'lstm', 'transformer'")


def get_model_config(model_type: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get model-specific configuration with defaults.
    
    Args:
        model_type: Type of model
        base_config: Base configuration dictionary
        
    Returns:
        Model-specific configuration
    """
    config = base_config.copy()
    
    if model_type.lower() == 'lstm':
        # LSTM-specific defaults
        config.setdefault('hidden_size', 32)
        config.setdefault('num_layers', 1)
        config.setdefault('embedding_dim', 12)
        config.setdefault('dropout', 0.05)
        
    elif model_type.lower() == 'transformer':
        # Transformer-specific defaults
        config.setdefault('d_model', 64)
        config.setdefault('nhead', 8)
        config.setdefault('num_layers', 3)
        config.setdefault('dim_feedforward', 256)
        config.setdefault('dropout', 0.1)
        config.setdefault('max_seq_len', 100)
        config.setdefault('use_symbol_embeddings', True)
        config.setdefault('use_sector_embeddings', False)
        config.setdefault('num_sectors', 0)
    
    return config


def get_supported_models() -> list:
    """Get list of supported model types."""
    return ['lstm', 'transformer']


def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    Get information about a model type.
    
    Args:
        model_type: Type of model
        
    Returns:
        Dictionary with model information
    """
    info = {
        'lstm': {
            'name': 'LSTM Regressor',
            'description': 'LSTM-based regressor with symbol embeddings',
            'features': ['Temporal modeling', 'Symbol embeddings', 'Cross-sectional learning'],
            'best_for': ['Time series with clear temporal patterns', 'Small to medium datasets']
        },
        'transformer': {
            'name': 'Transformer Regressor',
            'description': 'Transformer-based regressor with causal masking',
            'features': ['Multi-head attention', 'Causal masking', 'Symbol embeddings', 'Positional encoding'],
            'best_for': ['Complex feature interactions', 'Large datasets', 'Long sequences']
        }
    }
    
    return info.get(model_type.lower(), {})
