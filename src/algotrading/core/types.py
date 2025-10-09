"""Core types and data structures for ALGOTRADING."""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd


class ModelFamily(str, Enum):
    """Supported model families."""
    LSTM = "lstm"
    XGBOOST = "xgboost"
    RIDGE = "ridge"


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # Model architecture
    sequence_length: int = 30
    hidden_size: int = 32  # Further reduced for tiny cross-section
    num_layers: int = 1    # Keep single layer
    embedding_dim: int = 12
    dropout: float = 0.05  # Further reduced to avoid over-shrinking outputs
    
    # Training parameters
    learning_rate: float = 3e-4  # Lower LR for stability
    batch_size: int = 6    # Exactly one day per batch for cross-sectional training
    max_epochs: int = 20   # More epochs with cosine+warmup
    early_stopping_patience: int = 4  # Shorter patience
    weight_decay: float = 1e-5  # Reduced for tiny cross-section
    
    # Data parameters
    horizon_days: int = 5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    embargo_days: int = 21  # Days to skip between train/val/test splits (21 trading days = ~1 month to avoid leakage via slow-moving features)
    min_train_samples: int = 200  # Minimum samples required for training
    min_val_samples: int = 200   # Minimum samples required for validation
    min_test_samples: int = 200  # Minimum samples required for testing
    
    # Feature engineering
    features: List[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = [
                "ret1", "ret5", "ret21", "rsi14", "volz"
            ]


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""
    ic: float
    rank_ic: float
    mse: float
    rmse: float
    n_samples: int
    
    # Additional diagnostics
    prediction_std: float = 0.0
    target_std: float = 0.0
    constant_predictions: bool = False
    dead_features: List[str] = None
    
    def __post_init__(self):
        if self.dead_features is None:
            self.dead_features = []


@dataclass
class BacktestResults:
    """Backtesting results."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float
    win_rate: float
    n_trades: int
    
    # Additional metrics
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    hit_rate: float = 0.0


@dataclass
class TrainingData:
    """Training data container."""
    features: np.ndarray  # Shape: (n_samples, seq_len, n_features)
    targets: np.ndarray   # Shape: (n_samples,)
    symbols: np.ndarray   # Shape: (n_samples,)
    dates: np.ndarray     # Shape: (n_samples,)
    
    def __post_init__(self):
        assert len(self.features) == len(self.targets) == len(self.symbols) == len(self.dates)
        assert self.features.ndim == 3, "Features must be 3D: (samples, seq_len, features)"


@dataclass
class ModelOutput:
    """Model prediction output."""
    predictions: np.ndarray  # Raw predictions
    scores: np.ndarray       # Normalized scores
    ranks: np.ndarray        # Symbol rankings
    probabilities: np.ndarray  # Sigmoid-normalized probabilities
    
    # Trading signals
    long_signals: np.ndarray = None
    short_signals: np.ndarray = None
    
    def __post_init__(self):
        if self.long_signals is None:
            self.long_signals = np.zeros_like(self.predictions, dtype=bool)
        if self.short_signals is None:
            self.short_signals = np.zeros_like(self.predictions, dtype=bool)


class DataSplit:
    """Temporal data split for validation."""
    
    def __init__(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split(self, data: TrainingData) -> tuple:
        """Split data temporally."""
        n_samples = len(data.features)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))
        
        train_data = TrainingData(
            features=data.features[:train_end],
            targets=data.targets[:train_end],
            symbols=data.symbols[:train_end],
            dates=data.dates[:train_end]
        )
        
        val_data = TrainingData(
            features=data.features[train_end:val_end],
            targets=data.targets[train_end:val_end],
            symbols=data.symbols[train_end:val_end],
            dates=data.dates[train_end:val_end]
        )
        
        test_data = TrainingData(
            features=data.features[val_end:],
            targets=data.targets[val_end:],
            symbols=data.symbols[val_end:],
            dates=data.dates[val_end:]
        )
        
        return train_data, val_data, test_data


def calculate_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Information Coefficient (Pearson correlation)."""
    if len(predictions) < 2:
        return 0.0
    
    # Remove NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
    if np.sum(valid_mask) < 2:
        return 0.0
    
    pred_clean = predictions[valid_mask]
    target_clean = targets[valid_mask]
    
    # Check for constant values
    if np.std(pred_clean) < 1e-8 or np.std(target_clean) < 1e-8:
        return 0.0
    
    try:
        corr_matrix = np.corrcoef(pred_clean, target_clean)
        if np.isnan(corr_matrix[0, 1]):
            return 0.0
        return float(corr_matrix[0, 1])
    except (ValueError, np.linalg.LinAlgError):
        return 0.0


def calculate_rank_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Rank Information Coefficient (Spearman correlation)."""
    if len(predictions) < 2:
        return 0.0
    
    # Remove NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
    if np.sum(valid_mask) < 2:
        return 0.0
    
    pred_clean = predictions[valid_mask]
    target_clean = targets[valid_mask]
    
    try:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(pred_clean, target_clean)
        if np.isnan(corr):
            return 0.0
        return float(corr)
    except (ValueError, ImportError):
        # Fallback to manual rank correlation if scipy not available
        try:
            pred_ranks = np.argsort(np.argsort(pred_clean))
            target_ranks = np.argsort(np.argsort(target_clean))
            return calculate_ic(pred_ranks, target_ranks)
        except:
            return 0.0


def calculate_daily_ic(predictions: np.ndarray, targets: np.ndarray, 
                      dates: np.ndarray) -> Tuple[float, float, int]:
    """
    Calculate daily cross-sectional IC and return mean, std, and count.
    
    Args:
        predictions: Model predictions
        targets: Actual targets
        dates: Date array (same length as predictions/targets)
    
    Returns:
        ic_mean, ic_std, n_days
    """
    import pandas as pd
    
    # Create DataFrame for easy grouping
    df = pd.DataFrame({
        'prediction': predictions,
        'target': targets,
        'date': dates
    })
    
    # Remove NaN values
    df = df.dropna()
    
    if len(df) < 2:
        return 0.0, 0.0, 0
    
    daily_ics = []
    
    for date, group in df.groupby('date'):
        if len(group) < 2:
            continue
            
        # Check for sufficient variance
        if group['prediction'].std() < 1e-8 or group['target'].std() < 1e-8:
            continue
            
        # Calculate correlation for this date
        try:
            ic = group['prediction'].corr(group['target'])
            if not np.isnan(ic):
                daily_ics.append(ic)
        except:
            continue
    
    if len(daily_ics) == 0:
        return 0.0, 0.0, 0
    
    return float(np.mean(daily_ics)), float(np.std(daily_ics)), len(daily_ics)


def calculate_daily_rank_ic(predictions: np.ndarray, targets: np.ndarray, 
                           dates: np.ndarray) -> Tuple[float, float, int]:
    """
    Calculate daily cross-sectional Rank-IC and return mean, std, and count.
    """
    daily_rank_ics = _calculate_daily_rank_ic_values(predictions, targets, dates)
    
    if len(daily_rank_ics) == 0:
        return 0.0, 0.0, 0
    
    return float(np.mean(daily_rank_ics)), float(np.std(daily_rank_ics)), len(daily_rank_ics)

def _calculate_daily_rank_ic_values(predictions: np.ndarray, targets: np.ndarray, 
                                   dates: np.ndarray) -> List[float]:
    """
    Calculate daily cross-sectional Rank-IC and return individual daily values.
    """
    import pandas as pd
    
    # Create DataFrame for easy grouping
    df = pd.DataFrame({
        'prediction': predictions,
        'target': targets,
        'date': dates
    })
    
    # Remove NaN values
    df = df.dropna()
    
    if len(df) < 2:
        return []
    
    daily_rank_ics = []
    
    for date, group in df.groupby('date'):
        if len(group) < 2:
            continue
            
        # Calculate rank correlation for this date
        try:
            from scipy.stats import spearmanr
            rank_ic, _ = spearmanr(group['prediction'], group['target'])
            if not np.isnan(rank_ic):
                daily_rank_ics.append(rank_ic)
        except:
            # Fallback to manual rank correlation
            try:
                pred_ranks = group['prediction'].rank()
                target_ranks = group['target'].rank()
                rank_ic = pred_ranks.corr(target_ranks)
                if not np.isnan(rank_ic):
                    daily_rank_ics.append(rank_ic)
            except:
                continue
    
    return daily_rank_ics


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> ModelMetrics:
    """Calculate comprehensive model metrics."""
    ic = calculate_ic(predictions, targets)
    rank_ic = calculate_rank_ic(predictions, targets)
    mse = float(np.mean((predictions - targets) ** 2))
    rmse = float(np.sqrt(mse))
    prediction_std = float(np.std(predictions))
    constant_predictions = prediction_std < 1e-8
    
    return ModelMetrics(
        ic=ic,
        rank_ic=rank_ic,
        mse=mse,
        rmse=rmse,
        n_samples=len(predictions),
        prediction_std=prediction_std,
        constant_predictions=constant_predictions
    )


def standardize_targets(targets: np.ndarray) -> np.ndarray:
    """Standardize targets for training."""
    mean = np.mean(targets)
    std = np.std(targets)
    return (targets - mean) / (std + 1e-9)
