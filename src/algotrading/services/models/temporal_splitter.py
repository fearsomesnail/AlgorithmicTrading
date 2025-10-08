"""Temporal data splitting with proper embargo periods."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemporalSplitConfig:
    """Configuration for temporal splitting."""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    embargo_days: int = 35  # L + H = 30 + 5
    min_train_samples: int = 1000
    min_val_samples: int = 200
    min_test_samples: int = 200


class TemporalSplitter:
    """Temporal data splitter with embargo periods to prevent leakage."""
    
    def __init__(self, config: TemporalSplitConfig):
        self.config = config
        
    def split(self, data: 'TrainingData') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data temporally with embargo periods.
        
        Returns:
            train_indices, val_indices, test_indices
        """
        n_samples = len(data.features)
        dates = data.dates
        
        # Sort by date to ensure temporal order
        sort_indices = np.argsort(dates)
        sorted_dates = dates[sort_indices]
        
        # Debug: print date ranges
        logger.info(f"Date range: {pd.Timestamp(sorted_dates.min(), unit='s')} to {pd.Timestamp(sorted_dates.max(), unit='s')}")
        logger.info(f"Total samples: {n_samples}")
        
        # Calculate temporal split points based on actual dates
        min_date = sorted_dates.min()
        max_date = sorted_dates.max()
        total_days = (max_date - min_date) / (24 * 3600)
        
        # Calculate split dates
        train_end_date = min_date + (total_days * self.config.train_ratio) * (24 * 3600)
        val_end_date = min_date + (total_days * (self.config.train_ratio + self.config.val_ratio)) * (24 * 3600)
        
        # Apply embargo periods
        embargo_seconds = self.config.embargo_days * 24 * 3600
        
        # Find indices for each split
        train_mask = sorted_dates <= train_end_date
        val_mask = (sorted_dates > train_end_date + embargo_seconds) & (sorted_dates <= val_end_date)
        test_mask = sorted_dates > val_end_date + embargo_seconds
        
        train_indices = sort_indices[train_mask]
        val_indices = sort_indices[val_mask]
        test_indices = sort_indices[test_mask]
        
        # Validate splits
        self._validate_splits(train_indices, val_indices, test_indices, sorted_dates)
        
        # Check for leakage with embargo proof
        self._check_leakage_proof(train_indices, val_indices, test_indices, sorted_dates)
        
        # Log split information
        self._log_split_info(train_indices, val_indices, test_indices, sorted_dates, data)
        
        return train_indices, val_indices, test_indices
    
    def _validate_splits(self, train_indices: np.ndarray, val_indices: np.ndarray, 
                        test_indices: np.ndarray, sorted_dates: np.ndarray):
        """Validate that splits are non-overlapping and have sufficient samples."""
        
        # Check sample counts
        if len(train_indices) < self.config.min_train_samples:
            raise ValueError(f"Train split too small: {len(train_indices)} < {self.config.min_train_samples}")
        if len(val_indices) < self.config.min_val_samples:
            raise ValueError(f"Val split too small: {len(val_indices)} < {self.config.min_val_samples}")
        if len(test_indices) < self.config.min_test_samples:
            raise ValueError(f"Test split too small: {len(test_indices)} < {self.config.min_test_samples}")
    
    def _check_leakage_proof(self, train_indices: np.ndarray, val_indices: np.ndarray, 
                           test_indices: np.ndarray, sorted_dates: np.ndarray):
        """Check for data leakage with embargo proof."""
        import pandas as pd
        
        # Get date ranges for each split
        train_dates = sorted_dates[train_indices]
        val_dates = sorted_dates[val_indices]
        test_dates = sorted_dates[test_indices]
        
        train_end = pd.Timestamp(train_dates.max(), unit='s')
        val_start = pd.Timestamp(val_dates.min(), unit='s')
        val_end = pd.Timestamp(val_dates.max(), unit='s')
        test_start = pd.Timestamp(test_dates.min(), unit='s')
        
        # Check embargo requirements
        embargo_days = self.config.embargo_days
        train_val_gap = (val_start - train_end).days
        val_test_gap = (test_start - val_end).days
        
        # Assert no leakage
        assert train_end < val_start - pd.Timedelta(days=embargo_days), \
            f"Train-Val leakage: gap {train_val_gap} days < embargo {embargo_days} days"
        assert val_end < test_start - pd.Timedelta(days=embargo_days), \
            f"Val-Test leakage: gap {val_test_gap} days < embargo {embargo_days} days"
        
        logger.info(f"Leakage checks passed with embargo={embargo_days} days")
        logger.info(f"Train-Val gap: {train_val_gap} days, Val-Test gap: {val_test_gap} days")
        
        # Check for overlaps
        all_indices = np.concatenate([train_indices, val_indices, test_indices])
        if len(all_indices) != len(np.unique(all_indices)):
            raise ValueError("Overlapping indices found in splits")
        
        # Check temporal ordering
        if len(train_indices) > 0 and len(val_indices) > 0:
            train_max_date = sorted_dates[train_indices].max()
            val_min_date = sorted_dates[val_indices].min()
            if train_max_date >= val_min_date:
                raise ValueError(f"Train/Val overlap: train_max={train_max_date}, val_min={val_min_date}")
        
        if len(val_indices) > 0 and len(test_indices) > 0:
            val_max_date = sorted_dates[val_indices].max()
            test_min_date = sorted_dates[test_indices].min()
            if val_max_date >= test_min_date:
                raise ValueError(f"Val/Test overlap: val_max={val_max_date}, test_min={test_min_date}")
    
    def _log_split_info(self, train_indices: np.ndarray, val_indices: np.ndarray,
                       test_indices: np.ndarray, sorted_dates: np.ndarray, data):
        """Log detailed split information."""
        
        def format_timestamp(ts):
            return pd.Timestamp(ts, unit='s').strftime('%Y-%m-%d')
        
        logger.info("=" * 60)
        logger.info("TEMPORAL SPLIT INFORMATION")
        logger.info("=" * 60)
        
        if len(train_indices) > 0:
            train_dates = sorted_dates[train_indices]
            logger.info(f"TRAIN: {len(train_indices)} samples")
            logger.info(f"  Date range: {format_timestamp(train_dates.min())} to {format_timestamp(train_dates.max())}")
            logger.info(f"  Unique symbols: {len(np.unique(data.symbols[train_indices]))}")
        
        if len(val_indices) > 0:
            val_dates = sorted_dates[val_indices]
            logger.info(f"VAL: {len(val_indices)} samples")
            logger.info(f"  Date range: {format_timestamp(val_dates.min())} to {format_timestamp(val_dates.max())}")
            logger.info(f"  Unique symbols: {len(np.unique(data.symbols[val_indices]))}")
        
        if len(test_indices) > 0:
            test_dates = sorted_dates[test_indices]
            logger.info(f"TEST: {len(test_indices)} samples")
            logger.info(f"  Date range: {format_timestamp(test_dates.min())} to {format_timestamp(test_dates.max())}")
            logger.info(f"  Unique symbols: {len(np.unique(data.symbols[test_indices]))}")
        
        # Calculate gaps
        if len(train_indices) > 0 and len(val_indices) > 0:
            train_max_date = sorted_dates[train_indices].max()
            val_min_date = sorted_dates[val_indices].min()
            gap_days = (val_min_date - train_max_date) / (24 * 3600)
            logger.info(f"TRAIN-VAL GAP: {gap_days:.1f} days")
        
        if len(val_indices) > 0 and len(test_indices) > 0:
            val_max_date = sorted_dates[val_indices].max()
            test_min_date = sorted_dates[test_indices].min()
            gap_days = (test_min_date - val_max_date) / (24 * 3600)
            logger.info(f"VAL-TEST GAP: {gap_days:.1f} days")
        
        logger.info("=" * 60)


def create_temporal_splits(data, config: TemporalSplitConfig):
    """Create temporal train/validation/test splits with embargo."""
    
    splitter = TemporalSplitter(config)
    train_indices, val_indices, test_indices = splitter.split(data)
    
    # Import TrainingData here to avoid circular imports
    from ...core.types import TrainingData
    
    # Create split data
    train_data = TrainingData(
        features=data.features[train_indices],
        targets=data.targets[train_indices],
        symbols=data.symbols[train_indices],
        dates=data.dates[train_indices]
    )
    
    val_data = TrainingData(
        features=data.features[val_indices],
        targets=data.targets[val_indices],
        symbols=data.symbols[val_indices],
        dates=data.dates[val_indices]
    )
    
    test_data = TrainingData(
        features=data.features[test_indices],
        targets=data.targets[test_indices],
        symbols=data.symbols[test_indices],
        dates=data.dates[test_indices]
    )
    
    return train_data, val_data, test_data
