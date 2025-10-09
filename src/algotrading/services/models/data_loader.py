"""Data loading and preprocessing for model training."""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

from ...core.types import TrainingData, TrainingConfig
from .temporal_splitter import TemporalSplitConfig, create_temporal_splits
from .scaler_manager import ScalerManager, validate_feature_target_alignment, enforce_feature_config

logger = logging.getLogger(__name__)


class ASXDataLoader:
    """Data loader for ASX equity data."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.universe = [
            "BHP.AX", "CBA.AX", "CSL.AX", "WES.AX", "WBC.AX", "TLS.AX"
        ]
        self.benchmark = "^AXJO"
        
    def download_data(self, start_date: str, end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Download price and volume data for universe and benchmark."""
        logger.info(f"Downloading data from {start_date} to {end_date or 'present'}")
        
        # Download universe data
        universe_data = yf.download(
            self.universe,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )
        
        # Download benchmark data
        benchmark_data = yf.download(
            self.benchmark,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )
        
        prices = universe_data["Close"].dropna(how="all")
        volumes = universe_data["Volume"].dropna(how="all")
        benchmark_prices = benchmark_data["Close"].dropna()
        
        logger.info(f"Downloaded {len(prices)} days of data for {len(self.universe)} symbols")
        
        return prices, volumes, benchmark_prices
    
    def calculate_features(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate technical features for each symbol."""
        logger.info("Calculating technical features")
        
        features = {}
        returns = prices.pct_change()
        
        for symbol in self.universe:
            if symbol not in prices.columns:
                continue
                
            symbol_data = pd.DataFrame(index=prices.index)
            
            # Price returns
            symbol_data["ret1"] = returns[symbol]
            symbol_data["ret5"] = prices[symbol].pct_change(5)
            symbol_data["ret21"] = prices[symbol].pct_change(21)
            
            # RSI calculation
            rsi = self._calculate_rsi(returns[symbol], window=14)
            symbol_data["rsi14"] = rsi
            
            # Volume z-score
            if symbol in volumes.columns:
                vol_mean = volumes[symbol].rolling(60).mean()
                vol_std = volumes[symbol].rolling(60).std()
                symbol_data["volz"] = (volumes[symbol] - vol_mean) / (vol_std + 1e-9)
            else:
                symbol_data["volz"] = 0.0
            
            # Additional features
            symbol_data["volatility"] = returns[symbol].rolling(21).std()
            symbol_data["volume_ratio"] = volumes[symbol] / volumes[symbol].rolling(21).mean() if symbol in volumes.columns else 1.0
            
            # Enforce feature configuration - only keep configured features in exact order
            symbol_data = enforce_feature_config(symbol_data, self.config.features)
            
            # CRITICAL: Reorder columns to match config.features exactly
            symbol_data = symbol_data[self.config.features]
            
            # Log raw feature statistics BEFORE demeaning
            logger.info(f"Raw feature statistics for {symbol} (pre-demean):")
            raw_stats = {}
            for feature in self.config.features:
                if feature in symbol_data.columns:
                    mean_val = symbol_data[feature].mean()
                    std_val = symbol_data[feature].std()
                    raw_stats[feature] = {'mean': mean_val, 'std': std_val}
                    logger.info(f"  {feature}: mean={mean_val:.6f}, std={std_val:.6f}")
                    
                    # Assertions for raw features
                    if feature in ['ret1', 'ret5', 'ret21'] and abs(mean_val) > 0.02:
                        raise ValueError(f"FEATURE ORDER MISMATCH: {feature} mean {mean_val:.6f} too high - likely wrong column order")
                    elif feature == 'rsi14' and not (35 <= mean_val <= 65):
                        raise ValueError(f"FEATURE ORDER MISMATCH: {feature} mean {mean_val:.6f} outside RSI range [35, 65] - likely wrong column order")
                    elif feature == 'volz' and std_val < 0.1:
                        logger.warning(f"  {feature} std {std_val:.6f} seems low for volatility")
            
            # Cross-sectional demeaning will be applied after all symbols are combined
            logger.info(f"Raw feature statistics for {symbol} (will be cross-sectionally demeaned later):")
            for feature in self.config.features:
                if feature in symbol_data.columns:
                    mean_val = symbol_data[feature].mean()
                    std_val = symbol_data[feature].std()
                    logger.info(f"  {feature}: mean={mean_val:.6f}, std={std_val:.6f}")
            
            # Final validation: ensure column order matches config exactly
            if list(symbol_data.columns) != self.config.features:
                raise ValueError(f"FEATURE ORDER MISMATCH: columns {list(symbol_data.columns)} != config {self.config.features}")
            
            # Don't drop NaN values here - handle them during sequence building
            features[symbol] = symbol_data
            
        logger.info(f"Calculated features for {len(features)} symbols")
        return features
    
    def _calculate_rsi(self, returns: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        gains = returns.clip(lower=0)
        losses = (-returns).clip(lower=0)
        
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()
        
        rs = avg_gains / (avg_losses + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_targets(self, prices: pd.DataFrame, benchmark_prices: pd.Series, 
                         horizon: int = 5) -> Dict[str, pd.Series]:
        """Calculate forward excess returns as targets."""
        logger.info(f"Calculating {horizon}-day forward excess returns")
        
        targets = {}
        benchmark_returns = benchmark_prices.pct_change()
        
        for symbol in self.universe:
            if symbol not in prices.columns:
                continue
                
            # Calculate forward returns
            symbol_returns = prices[symbol].pct_change(horizon).shift(-horizon)
            
            # Calculate benchmark forward returns
            benchmark_forward = benchmark_returns.rolling(horizon).apply(
                lambda x: (1 + x).prod() - 1, raw=False
            ).shift(-horizon)
            
            # Ensure benchmark_forward is a Series
            if isinstance(benchmark_forward, pd.DataFrame):
                benchmark_forward = benchmark_forward.iloc[:, 0]
            
            # Calculate excess returns (ensure both are Series)
            if isinstance(symbol_returns, pd.Series) and isinstance(benchmark_forward, pd.Series):
                excess_returns = symbol_returns - benchmark_forward
                targets[symbol] = excess_returns.dropna()
            else:
                logger.warning(f"Unexpected data types for {symbol}: symbol_returns={type(symbol_returns)}, benchmark_forward={type(benchmark_forward)}")
                continue
            
        logger.info(f"Calculated targets for {len(targets)} symbols")
        return targets
    
    def build_sequences(self, features: Dict[str, pd.DataFrame], 
                       targets: Dict[str, pd.Series]) -> TrainingData:
        """Build training sequences from features and targets."""
        logger.info("Building training sequences")
        
        X_list, y_list, symbols_list, dates_list = [], [], [], []
        
        for symbol in self.universe:
            if symbol not in features or symbol not in targets:
                logger.warning(f"Symbol {symbol} not found in features or targets")
                continue
                
            # Align features and targets
            target_series = targets[symbol].copy()
            target_series.name = "target"
            df = features[symbol].join(target_series, how="inner")
            
            # Debug: check what columns we have
            logger.info(f"  Columns after join: {list(df.columns)}")
            logger.info(f"  Target series name: {target_series.name}")
            logger.info(f"  Target series length: {len(target_series)}")
            
            # Only drop rows where target is NaN, not where features are NaN
            if 'target' in df.columns:
                df = df.dropna(subset=['target'])
            else:
                logger.warning(f"Target column not found in dataframe for {symbol}")
                continue
            
            logger.info(f"Symbol {symbol}: {len(features[symbol])} features, {len(targets[symbol])} targets, {len(df)} aligned")
            logger.info(f"  Feature date range: {features[symbol].index[0]} to {features[symbol].index[-1]}")
            logger.info(f"  Target date range: {targets[symbol].index[0]} to {targets[symbol].index[-1]}")
            if len(df) > 0:
                logger.info(f"  Aligned date range: {df.index[0]} to {df.index[-1]}")
            
            if len(df) < self.config.sequence_length + 1:
                logger.warning(f"Symbol {symbol}: insufficient data ({len(df)} < {self.config.sequence_length + 1})")
                continue
                
            # Extract feature columns
            feature_cols = [col for col in df.columns if col != "target"]
            feature_data = df[feature_cols].values
            target_data = df["target"].values
            
            # Build sequences
            for i in range(self.config.sequence_length, len(df)):
                # Get the sequence window
                sequence = feature_data[i-self.config.sequence_length:i]
                
                # Check if the sequence has any NaN values
                if np.isnan(sequence).any():
                    continue  # Skip this sequence if it has NaN values
                
                X_list.append(sequence)
                y_list.append(target_data[i])
                symbols_list.append(symbol)
                dates_list.append(df.index[i])
        
        if not X_list:
            raise ValueError("No valid sequences found")
            
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        
        # Convert symbols to integer indices
        symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.universe)}
        symbols = np.array([symbol_to_idx[symbol] for symbol in symbols_list], dtype=np.int32)
        
        # Convert dates to timestamps (float64)
        dates = np.array([pd.Timestamp(date).timestamp() for date in dates_list], dtype=np.float64)
        
        # Sort all data by date to ensure temporal order
        sort_indices = np.argsort(dates)
        X = X[sort_indices]
        y = y[sort_indices]
        symbols = symbols[sort_indices]
        dates = dates[sort_indices]
        
        # Apply cross-sectional demeaning after data is combined
        logger.info("Applying cross-sectional demeaning...")
        features_to_demean = ['ret1', 'ret5', 'ret21', 'rsi14']
        feature_indices = {feature: i for i, feature in enumerate(self.config.features) if feature in features_to_demean}
        
        for date in np.unique(dates):
            date_mask = dates == date
            if np.sum(date_mask) > 1:  # Need at least 2 symbols for demeaning
                for feature, idx in feature_indices.items():
                    if idx < X.shape[2]:  # Make sure index is valid
                        # Demean across symbols for this date
                        X[date_mask, :, idx] = X[date_mask, :, idx] - np.mean(X[date_mask, :, idx])
        
        logger.info(f"Built {len(X)} sequences with shape {X.shape}")
        logger.info(f"Date range: {pd.Timestamp(dates.min(), unit='s')} to {pd.Timestamp(dates.max(), unit='s')}")
        
        # Final feature order validation
        logger.info(f"FINAL FEATURE ORDER: {self.config.features}")
        # Note: feature_cols is extracted from the first symbol's data, should match config.features
        if X_list:  # Only validate if we have data
            first_symbol_data = features[list(features.keys())[0]]
            first_symbol_data = first_symbol_data[self.config.features]  # Reorder to match config
            
            # CRITICAL: Assert exact feature order
            EXPECTED = ['ret1', 'ret5', 'ret21', 'rsi14', 'volz']
            actual_features = list(first_symbol_data.columns)
            assert actual_features == EXPECTED, f"Feature order mismatch: {actual_features} != {EXPECTED}"
            logger.info("Feature order validation passed")
        
        return TrainingData(
            features=X,
            targets=y,
            symbols=symbols,
            dates=dates
        )
    
    def load_training_data(self, start_date: str = "2018-01-01", 
                          end_date: Optional[str] = None) -> TrainingData:
        """Load complete training dataset."""
        logger.info("Loading complete training dataset")
        
        # Download data
        prices, volumes, benchmark_prices = self.download_data(start_date, end_date)
        
        # Calculate features
        features = self.calculate_features(prices, volumes)
        
        # Calculate targets
        targets = self.calculate_targets(prices, benchmark_prices, self.config.horizon_days)
        
        # Build sequences
        training_data = self.build_sequences(features, targets)
        
        logger.info(f"Loaded training data: {len(training_data.features)} samples")
        return training_data


def standardize_targets(targets: np.ndarray) -> np.ndarray:
    """Standardize targets for training."""
    mean = np.mean(targets)
    std = np.std(targets)
    return (targets - mean) / (std + 1e-9)


def create_data_splits(data: TrainingData, config: TrainingConfig) -> Tuple[TrainingData, TrainingData, TrainingData]:
    """Create temporal train/validation/test splits with embargo."""
    
    # Create temporal split configuration
    split_config = TemporalSplitConfig(
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        embargo_days=config.sequence_length + config.horizon_days,  # L + H
        min_train_samples=1000,
        min_val_samples=200,
        min_test_samples=200
    )
    
    return create_temporal_splits(data, split_config)
