"""Model training and evaluation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import os
import math
from collections import defaultdict

from ...core.types import (
    TrainingConfig, TrainingData, ModelMetrics, 
    calculate_metrics, calculate_daily_ic, calculate_daily_rank_ic, _calculate_daily_rank_ic_values
)
from .model_nn import ModelFactory, count_parameters, LSTMRegressor
from .model_zoo import get_model_config, build_model
from .scaler_manager import ScalerManager
from ..results_manager import ResultsManager

logger = logging.getLogger(__name__)


class DateBatchSampler:
    """Sampler that creates batches with samples from exactly one date."""
    
    def __init__(self, dates: torch.Tensor, batch_size: int = 6):
        self.dates = dates
        self.batch_size = batch_size
        
        # Group indices by date
        self.date_to_indices = defaultdict(list)
        for idx, date in enumerate(dates):
            self.date_to_indices[date.item()].append(idx)
        
        self.unique_dates = list(self.date_to_indices.keys())
        self.unique_dates.sort()  # Sort for deterministic ordering
        
        # Log sampler stats
        group_sizes = [len(indices) for indices in self.date_to_indices.values()]
        logger.info(f"DateBatchSampler: groups={len(self.unique_dates)}, group_size_stats min={min(group_sizes)}/median={np.median(group_sizes):.0f}/max={max(group_sizes)}")
        
    def __iter__(self):
        # Create batches by iterating through dates - exactly one date per batch
        for date in self.unique_dates:
            indices = self.date_to_indices[date]
            # Take exactly batch_size samples (or all if fewer)
            batch_indices = indices[:self.batch_size]
            if len(batch_indices) >= 3:  # Need at least 3 samples for correlation
                yield batch_indices
    
    def __len__(self):
        # Count dates with at least 3 samples
        return sum(1 for indices in self.date_to_indices.values() if len(indices) >= 3)


def cs_corr_loss(preds: torch.Tensor, targets: torch.Tensor, dates: torch.Tensor) -> torch.Tensor:
    """Cross-sectional correlation loss to maximize Pearson corr within each day."""
    loss = 0.0
    n_groups = 0
    
    for date in torch.unique(dates):
        mask = dates == date
        if mask.sum() < 3:  # Need at least 3 samples for correlation
            continue
            
        p = preds[mask]
        t = targets[mask]
        
        # Standardize
        p = (p - p.mean()) / (p.std() + 1e-6)
        t = (t - t.mean()) / (t.std() + 1e-6)
        
        # 1 - correlation (minimize this = maximize correlation)
        loss += 1.0 - (p * t).mean()
        n_groups += 1
    
    return loss / max(n_groups, 1)


def cs_std_ratio(preds: torch.Tensor, targets: torch.Tensor, dates: torch.Tensor) -> torch.Tensor:
    """Cross-sectional standard deviation ratio within each day."""
    ratios = []
    
    for date in torch.unique(dates):
        mask = dates == date
        if mask.sum() < 3:  # Need at least 3 samples
            continue
            
        p = preds[mask]
        t = targets[mask]
        
        ratio = (p.std() + 1e-6) / (t.std() + 1e-6)
        ratios.append(ratio)
    
    if not ratios:
        return torch.tensor(1.0, device=preds.device)
    
    return torch.stack(ratios).mean()


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    
    def __init__(self, optimizer, warmup_epochs: int = 3, total_epochs: int = 15, 
                 min_lr_ratio: float = 0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.param_groups[0]['lr']
        self.min_lr = self.base_lr * min_lr_ratio
        
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStopping:
    """Early stopping utility optimized for Rank-IC with EMA smoothing."""
    
    def __init__(self, patience: int = 4, min_delta: float = 1e-6, alpha: float = 0.5):
        self.patience = patience
        self.min_delta = min_delta
        self.alpha = alpha  # EMA smoothing factor
        self.best_metric = float('-inf')  # For Rank-IC (higher is better)
        self.counter = 0
        self.early_stop = False
        self.smoothed_metric = None
    
    def __call__(self, val_rank_ic: float) -> bool:
        # Apply EMA smoothing
        if self.smoothed_metric is None:
            self.smoothed_metric = val_rank_ic
        else:
            self.smoothed_metric = self.alpha * val_rank_ic + (1 - self.alpha) * self.smoothed_metric
        
        # Check for improvement
        if self.smoothed_metric > self.best_metric + self.min_delta:
            self.best_metric = self.smoothed_metric
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


class ModelTrainer:
    """Model training and evaluation class."""
    
    def __init__(self, config: TrainingConfig, model_family: str = "lstm", 
                 results_manager: Optional[ResultsManager] = None):
        self.config = config
        self.model_family = model_family
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = EarlyStopping(patience=self.config.early_stopping_patience, alpha=0.5)
        self.results_manager = results_manager or ResultsManager(enable_file_logging=False)
        self.scaler_manager = ScalerManager(winsorize_pct=self.config.winsorize_pct)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        logger.info(f"Initialized trainer with device: {self.device}")
    
    def _create_model(self, input_dim: int, num_symbols: int = 6) -> nn.Module:
        """Create model instance."""
        logger.info(f"Creating {self.model_family} model with input_dim={input_dim}, num_symbols={num_symbols}")
        # Get model-specific configuration
        model_config = get_model_config(self.model_family, {
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'embedding_dim': self.config.embedding_dim,
            'dropout': self.config.dropout,
        })
        
        # Build model using model zoo
        model = build_model(
            model_type=self.model_family,
            input_dim=input_dim,
            num_symbols=num_symbols,
            config=model_config
        )
        
        logger.info(f"{self.model_family.upper()}Regressor created successfully")
        return model
    
    def _create_symbol_mapping(self, symbols: np.ndarray) -> Dict[int, int]:
        """Create symbol to index mapping."""
        # Get unique symbols and create mapping
        unique_symbols = np.unique(symbols)
        return {symbol: idx for idx, symbol in enumerate(unique_symbols)}
    
    def _prepare_data(self, data: TrainingData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training."""
        # Convert to tensors
        X = torch.from_numpy(data.features).float()
        y = torch.from_numpy(data.targets).float()
        symbols = torch.from_numpy(data.symbols).long()
        dates = torch.from_numpy(data.dates).double()
        
        logger.info(f"Prepared data - X type: {type(X)}, y type: {type(y)}, symbols type: {type(symbols)}, dates type: {type(dates)}")
        logger.info(f"Prepared data - X dtype: {X.dtype}, y dtype: {y.dtype}, symbols dtype: {symbols.dtype}, dates dtype: {dates.dtype}")
        
        return X, y, symbols, dates
    
    def _create_dataloader(self, X: torch.Tensor, y: torch.Tensor, 
                          symbols: torch.Tensor, dates: torch.Tensor,
                          shuffle: bool = True) -> DataLoader:
        """Create DataLoader."""
        logger.info(f"Creating dataset with shapes: X={X.shape}, y={y.shape}, symbols={symbols.shape}, dates={dates.shape}")
        logger.info(f"Symbols dtype: {symbols.dtype}, dates dtype: {dates.dtype}")
        logger.info(f"Symbols sample: {symbols[:5]}")
        logger.info(f"Dates sample: {dates[:5]}")
        
        try:
            logger.info("About to create TensorDataset...")
            dataset = TensorDataset(X, y, symbols, dates)
            logger.info(f"Dataset created successfully")
        except Exception as e:
            logger.error(f"Error creating TensorDataset: {e}")
            logger.error(f"X type: {type(X)}, y type: {type(y)}, symbols type: {type(symbols)}, dates type: {type(dates)}")
            logger.error(f"X dtype: {X.dtype}, y dtype: {y.dtype}, symbols dtype: {symbols.dtype}, dates dtype: {dates.dtype}")
            raise
        
        try:
            logger.info("About to create DataLoader...")
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                drop_last=False
            )
            logger.info(f"DataLoader created successfully")
        except Exception as e:
            logger.error(f"Error creating DataLoader: {e}")
            raise
            
        return dataloader
    
    def _create_dataloader_with_sampler(self, X: torch.Tensor, y: torch.Tensor, symbols: torch.Tensor, 
                                       dates: torch.Tensor, sampler) -> DataLoader:
        """Create DataLoader with custom sampler."""
        try:
            logger.info("About to create TensorDataset...")
            dataset = TensorDataset(X, y, symbols, dates)
            logger.info(f"Dataset created successfully")
        except Exception as e:
            logger.error(f"Error creating TensorDataset: {e}")
            logger.error(f"X type: {type(X)}, y type: {type(y)}, symbols type: {type(symbols)}, dates type: {type(dates)}")
            logger.error(f"X dtype: {X.dtype}, y dtype: {y.dtype}, symbols dtype: {symbols.dtype}, dates dtype: {dates.dtype}")
            raise
        
        try:
            logger.info("About to create DataLoader with sampler...")
            dataloader = DataLoader(
                dataset,
                batch_sampler=sampler,
                drop_last=False
            )
            logger.info(f"DataLoader with sampler created successfully")
        except Exception as e:
            logger.error(f"Error creating DataLoader with sampler: {e}")
            raise
            
        return dataloader
    
    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, dates: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Huber loss with cross-sectional correlation and variance penalties."""
        # Use Huber loss for robustness against outliers (delta=1.0 for scaled space)
        huber_loss = nn.SmoothL1Loss(beta=1.0)
        huber = huber_loss(predictions, targets)
        loss = huber
        
        loss_components = {'huber': huber.item()}
        
        # Cross-sectional correlation loss (if dates provided)
        if dates is not None:
            cs_corr = cs_corr_loss(predictions, targets, dates)
            cs_corr_weighted = self.config.lambda_corr * cs_corr  # Use configurable λ_corr
            loss += cs_corr_weighted
            loss_components['cs_corr_loss'] = cs_corr.item()
            loss_components['cs_corr_weighted'] = cs_corr_weighted.item()
            
            # Cross-sectional variance penalty
            cs_ratio = cs_std_ratio(predictions, targets, dates)
            var_penalty = torch.clamp(0.25 - cs_ratio, min=0).pow(2)
            var_penalty_weighted = 0.12 * var_penalty  # λ_var = 0.12
            loss += var_penalty_weighted
            loss_components['var_penalty'] = var_penalty.item()
            loss_components['var_penalty_weighted'] = var_penalty_weighted.item()
            loss_components['cs_ratio'] = cs_ratio.item()
        else:
            # Fallback to global variance penalty if no dates
            pred_std = torch.std(predictions)
            target_std = torch.std(targets).detach().clamp_min(1e-6)
            ratio = pred_std / target_std
            var_penalty = (1.0 - ratio).clamp(min=0).pow(2)
            var_penalty_weighted = 0.05 * var_penalty
            loss += var_penalty_weighted
            loss_components['var_penalty'] = var_penalty.item()
            loss_components['var_penalty_weighted'] = var_penalty_weighted.item()
            loss_components['cs_ratio'] = ratio.item()
        
        loss_components['total_loss'] = loss.item()
        return loss, loss_components
    
    def _train_epoch(self, dataloader: DataLoader, symbol_mapping: Dict[str, int]) -> Tuple[float, ModelMetrics, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_dates = []
        
        # Collect loss components for logging
        loss_components_avg = {}
        
        for batch_X, batch_y, batch_symbols, batch_dates in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_symbols = batch_symbols.to(self.device)
            
            # Convert symbols to indices
            try:
                symbol_indices = torch.tensor([
                    symbol_mapping[symbol.item()] for symbol in batch_symbols
                ], device=self.device)
            except KeyError as e:
                logger.error(f"Symbol mapping error: {e}")
                logger.error(f"Symbol mapping: {symbol_mapping}")
                logger.error(f"Batch symbols: {batch_symbols}")
                raise
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_X, symbol_indices)
            loss, loss_components = self._compute_loss(predictions, batch_y, batch_dates)
            
            # Accumulate loss components
            for key, value in loss_components.items():
                if key not in loss_components_avg:
                    loss_components_avg[key] = 0.0
                loss_components_avg[key] += value
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions for metrics
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(batch_y.detach().cpu().numpy())
            all_dates.extend(batch_dates.detach().cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # Calculate daily IC metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        dates = np.array(all_dates)
        
        daily_ic_mean, daily_ic_std, n_days = calculate_daily_ic(predictions, targets, dates)
        daily_rank_ic_mean, daily_rank_ic_std, n_rank_days = calculate_daily_rank_ic(predictions, targets, dates)
        
        # Create metrics with daily IC
        metrics = ModelMetrics(
            ic=daily_ic_mean,
            rank_ic=daily_rank_ic_mean,
            mse=float(np.mean((predictions - targets) ** 2)),
            rmse=float(np.sqrt(np.mean((predictions - targets) ** 2))),
            n_samples=len(predictions),
            prediction_std=float(np.std(predictions)),
            target_std=float(np.std(targets)),
            constant_predictions=np.std(predictions) < 1e-8
        )
        
        return avg_loss, metrics, loss_components_avg
    
    def _validate_epoch(self, dataloader: DataLoader, symbol_mapping: Dict[str, int]) -> Tuple[float, ModelMetrics, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_dates = []
        
        # Collect loss components for logging
        loss_components_avg = {}
        
        with torch.no_grad():
            for batch_X, batch_y, batch_symbols, batch_dates in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_symbols = batch_symbols.to(self.device)
                
                # Convert symbols to indices
                symbol_indices = torch.tensor([
                    symbol_mapping[symbol.item()] for symbol in batch_symbols
                ], device=self.device)
                
                # Forward pass
                predictions = self.model(batch_X, symbol_indices)
                loss, loss_components = self._compute_loss(predictions, batch_y, batch_dates)
                
                # Accumulate loss components
                for key, value in loss_components.items():
                    if key not in loss_components_avg:
                        loss_components_avg[key] = 0.0
                    loss_components_avg[key] += value
                
                total_loss += loss.item()
                
                # Store predictions for metrics
                all_predictions.extend(predictions.detach().cpu().numpy())
                all_targets.extend(batch_y.detach().cpu().numpy())
                all_dates.extend(batch_dates.detach().cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # Average loss components
        for key in loss_components_avg:
            loss_components_avg[key] /= len(dataloader)
        
        # Calculate daily IC metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        dates = np.array(all_dates)
        
        daily_ic_mean, daily_ic_std, n_days = calculate_daily_ic(predictions, targets, dates)
        daily_rank_ic_mean, daily_rank_ic_std, n_rank_days = calculate_daily_rank_ic(predictions, targets, dates)
        
        # Create metrics with daily IC
        metrics = ModelMetrics(
            ic=daily_ic_mean,
            rank_ic=daily_rank_ic_mean,
            mse=float(np.mean((predictions - targets) ** 2)),
            rmse=float(np.sqrt(np.mean((predictions - targets) ** 2))),
            n_samples=len(predictions),
            prediction_std=float(np.std(predictions)),
            target_std=float(np.std(targets)),
            constant_predictions=np.std(predictions) < 1e-8
        )
        
        # Calculate per-day Rank-IC quantiles for validation
        val_daily_rank_ics = []
        for date in np.unique(dates):
            date_mask = dates == date
            if np.sum(date_mask) >= 3:  # Need at least 3 samples
                date_preds = predictions[date_mask]
                date_targets = targets[date_mask]
                date_rank_ic = np.corrcoef(np.argsort(np.argsort(date_preds)), np.argsort(np.argsort(date_targets)))[0, 1]
                if not np.isnan(date_rank_ic):
                    val_daily_rank_ics.append(date_rank_ic)
        
        if val_daily_rank_ics:
            val_rank_ic_q10 = np.percentile(val_daily_rank_ics, 10)
            val_rank_ic_median = np.median(val_daily_rank_ics)
            val_rank_ic_q90 = np.percentile(val_daily_rank_ics, 90)
            logger.info(f"val_median_daily_rank_ic={val_rank_ic_median:.4f}, val_rank_ic_q10={val_rank_ic_q10:.4f}, val_rank_ic_q90={val_rank_ic_q90:.4f}")
        else:
            logger.info("val_median_daily_rank_ic=nan, val_rank_ic_q10=nan, val_rank_ic_q90=nan")
        
        return avg_loss, metrics, loss_components_avg
    
    def _validate_data(self, train_data: TrainingData, val_data: TrainingData):
        """Validate data integrity and check for leakage."""
        # Check for NaN values
        train_nan_features = np.isnan(train_data.features).sum()
        train_nan_targets = np.isnan(train_data.targets).sum()
        val_nan_features = np.isnan(val_data.features).sum()
        val_nan_targets = np.isnan(val_data.targets).sum()
        
        if train_nan_features > 0:
            self.results_manager.log_warning(f"Train features contain {train_nan_features} NaN values")
        if train_nan_targets > 0:
            self.results_manager.log_warning(f"Train targets contain {train_nan_targets} NaN values")
        if val_nan_features > 0:
            self.results_manager.log_warning(f"Val features contain {val_nan_features} NaN values")
        if val_nan_targets > 0:
            self.results_manager.log_warning(f"Val targets contain {val_nan_targets} NaN values")
        
        # Check for data leakage (correlation between features and targets at same time)
        # This is a simplified check - in practice you'd want more sophisticated tests
        try:
            # Flatten features to 2D for correlation calculation
            train_features_flat = train_data.features.reshape(train_data.features.shape[0], -1)
            train_corr = np.corrcoef(train_features_flat.mean(axis=1), train_data.targets)[0, 1]
            if abs(train_corr) > 0.1:  # Arbitrary threshold
                self.results_manager.log_warning(f"High correlation between features and targets: {train_corr:.4f}")
        except Exception as e:
            self.results_manager.log_warning(f"Could not calculate feature-target correlation: {e}")
    
    def _log_data_stats(self, train_data: TrainingData, val_data: TrainingData, num_symbols: int):
        """Log comprehensive data statistics."""
        stats = {
            "train_samples": len(train_data.features),
            "val_samples": len(val_data.features),
            "num_symbols": num_symbols,
            "sequence_length": train_data.features.shape[1],
            "num_features": train_data.features.shape[2],
            "train_target_mean": float(np.mean(train_data.targets)),
            "train_target_std": float(np.std(train_data.targets)),
            "val_target_mean": float(np.mean(val_data.targets)),
            "val_target_std": float(np.std(val_data.targets)),
            "train_feature_mean": float(np.mean(train_data.features)),
            "train_feature_std": float(np.std(train_data.features)),
            "val_feature_mean": float(np.mean(val_data.features)),
            "val_feature_std": float(np.std(val_data.features)),
            "unique_train_symbols": len(np.unique(train_data.symbols)),
            "unique_val_symbols": len(np.unique(val_data.symbols)),
            "train_date_range": f"{train_data.dates.min():.0f} to {train_data.dates.max():.0f}",
            "val_date_range": f"{val_data.dates.min():.0f} to {val_data.dates.max():.0f}"
        }
        
        # Per-symbol statistics
        for symbol in np.unique(train_data.symbols):
            symbol_mask = train_data.symbols == symbol
            symbol_samples = np.sum(symbol_mask)
            stats[f"train_symbol_{symbol}_samples"] = int(symbol_samples)
        
        for symbol in np.unique(val_data.symbols):
            symbol_mask = val_data.symbols == symbol
            symbol_samples = np.sum(symbol_mask)
            stats[f"val_symbol_{symbol}_samples"] = int(symbol_samples)
        
        self.results_manager.store_data_stats(stats)
    
    def train(self, train_data: TrainingData, val_data: TrainingData, 
              num_symbols: int = 6) -> Dict[str, List[float]]:
        """Train the model."""
        logger.info("Starting model training")
        
        # Store configuration
        config_dict = {
            "sequence_length": self.config.sequence_length,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "embedding_dim": self.config.embedding_dim,
            "dropout": self.config.dropout,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "max_epochs": self.config.max_epochs,
            "early_stopping_patience": self.config.early_stopping_patience,
            "weight_decay": self.config.weight_decay,
            "horizon_days": self.config.horizon_days,
            "train_ratio": self.config.train_ratio,
            "val_ratio": self.config.val_ratio,
            "test_ratio": self.config.test_ratio,
            "features": self.config.features,
            "model_family": self.model_family,
            "device": str(self.device)
        }
        self.results_manager.store_config(config_dict)
        
        # Data validation and statistics
        self._validate_data(train_data, val_data)
        self._log_data_stats(train_data, val_data, num_symbols)
        
        # Create symbol mapping
        symbol_mapping = self._create_symbol_mapping(train_data.symbols)
        logger.info(f"Symbol mapping: {symbol_mapping}")
        logger.info(f"Train symbols unique: {np.unique(train_data.symbols)}")
        logger.info(f"Val symbols unique: {np.unique(val_data.symbols)}")
        
        # Prepare data
        X_train, y_train, symbols_train, dates_train = self._prepare_data(train_data)
        X_val, y_val, symbols_val, dates_val = self._prepare_data(val_data)
        
        # Convert tensors to numpy for scaler
        X_train_np = X_train.numpy()
        y_train_np = y_train.numpy()
        symbols_train_np = symbols_train.numpy()
        X_val_np = X_val.numpy()
        y_val_np = y_val.numpy()
        symbols_val_np = symbols_val.numpy()
        
        # Fit scalers on training data only
        self.scaler_manager.fit(X_train_np, y_train_np, symbols_train_np, self.config.features)
        
        # Transform data using fitted scalers
        X_train_scaled = self.scaler_manager.transform_features(X_train_np, symbols_train_np)
        y_train_scaled = self.scaler_manager.transform_targets(y_train_np)
        X_val_scaled = self.scaler_manager.transform_features(X_val_np, symbols_val_np)
        y_val_scaled = self.scaler_manager.transform_targets(y_val_np)
        
        # Validate feature standard deviations
        self.scaler_manager.validate_feature_stds(X_val_np, symbols_val_np)
        
        # Convert scaled data back to tensors
        X_train_scaled = torch.from_numpy(X_train_scaled).float()
        y_train_scaled = torch.from_numpy(y_train_scaled).float()
        X_val_scaled = torch.from_numpy(X_val_scaled).float()
        y_val_scaled = torch.from_numpy(y_val_scaled).float()
        
        # Log scaling statistics
        # Per-feature std across samples (not global std)
        train_feature_stds = torch.std(X_train_scaled, dim=(0, 1)).numpy()  # Per feature across all samples
        val_feature_stds = torch.std(X_val_scaled, dim=(0, 1)).numpy()
        logger.info(f"Training data - Feature stds by feature: {train_feature_stds}")
        logger.info(f"Validation data - Feature stds by feature: {val_feature_stds}")
        logger.info(f"Training data - Global std: {X_train_scaled.std():.6f}, Targets std: {y_train_scaled.std():.6f}")
        logger.info(f"Validation data - Global std: {X_val_scaled.std():.6f}, Targets std: {y_val_scaled.std():.6f}")
        
        # Create model
        input_dim = train_data.features.shape[-1]
        logger.info(f"Creating model with input_dim={input_dim}, num_symbols={num_symbols}")
        self.model = self._create_model(input_dim, num_symbols).to(self.device)
        logger.info(f"Model created and moved to device")
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Create scheduler with warmup and cosine decay
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, 
            warmup_epochs=3, 
            total_epochs=self.config.max_epochs,
            min_lr_ratio=0.01
        )
        
        # Create data loaders with scaled data
        logger.info(f"Creating data loaders...")
        # Use DateBatchSampler for cross-sectional training
        train_sampler = DateBatchSampler(dates_train, batch_size=6)
        train_loader = self._create_dataloader_with_sampler(X_train_scaled, y_train_scaled, symbols_train, dates_train, train_sampler)
        val_loader = self._create_dataloader(X_val_scaled, y_val_scaled, symbols_val, dates_val, shuffle=False)
        logger.info(f"Data loaders created successfully")
        
        logger.info(f"Model created successfully")
        logger.info(f"Model parameters: {count_parameters(self.model):,}")
        logger.info(f"Head activation: None (linear)")
        logger.info(f"Feature CS demeaning: True (ret1/5/21/rsi14)")
        logger.info(f"Training samples: {len(train_data.features)}")
        logger.info(f"Validation samples: {len(val_data.features)}")
        
        # Training loop
        best_val_rank_ic = float('-inf')
        best_model_state = None
        best_checkpoint_epoch = 0
        best_checkpoint_rank_ic = float('-inf')
        best_checkpoint_ic = 0.0
        best_checkpoint_std_ratio = 0.0
        collapse_abort_epochs = 0  # Track consecutive epochs with low CS ratio
        
        logger.info(f"Starting training loop for {self.config.max_epochs} epochs...")
        print(f"   Epochs: {self.config.max_epochs}, Patience: {self.config.early_stopping_patience}")
        print(f"   Training samples: {len(train_data.features)}, Validation samples: {len(val_data.features)}")
        print("   Progress: ", end="", flush=True)
        
        for epoch in range(self.config.max_epochs):
            # Train
            train_loss, train_metrics, train_loss_components = self._train_epoch(train_loader, symbol_mapping)
            
            # Validate
            val_loss, val_metrics, val_loss_components = self._validate_epoch(val_loader, symbol_mapping)
            
            # Log batching sanity
            unique_dates_per_batch = []
            for batch_X, batch_y, batch_symbols, batch_dates in train_loader:
                unique_dates = len(torch.unique(batch_dates))
                unique_dates_per_batch.append(unique_dates)
            
            avg_unique_dates = np.mean(unique_dates_per_batch) if unique_dates_per_batch else 0
            pct_one_date = np.mean([d == 1 for d in unique_dates_per_batch]) * 100 if unique_dates_per_batch else 0
            logger.info(f"Batching: batch_size={self.config.batch_size}, avg_unique_dates_per_batch={avg_unique_dates:.1f}, %batches_with_one_date={pct_one_date:.1f}")
            
            # Log optimizer state
            current_lr = self.optimizer.param_groups[0]['lr']
            grad_norm = 0.0
            head_weight_norm = 0.0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
                if 'head' in name and 'weight' in name:
                    head_weight_norm += param.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            head_weight_norm = head_weight_norm ** 0.5
            logger.info(f"Optimizer: lr={current_lr:.2e}, grad_norm={grad_norm:.3f}, head_weight_norm={head_weight_norm:.3f}")
            
            # Update scheduler with warmup and cosine decay
            self.scheduler.step(epoch)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Log variance ratios on both scales and Val Rank-IC EMA
            train_ratio_scaled = train_metrics.prediction_std / train_metrics.target_std
            val_ratio_scaled = val_metrics.prediction_std / val_metrics.target_std
            
            # Convert to original scale for comparison
            train_pred_orig = self.scaler_manager.inverse_transform_targets(np.array([train_metrics.prediction_std]))[0]
            train_tgt_orig = self.scaler_manager.inverse_transform_targets(np.array([train_metrics.target_std]))[0]
            val_pred_orig = self.scaler_manager.inverse_transform_targets(np.array([val_metrics.prediction_std]))[0]
            val_tgt_orig = self.scaler_manager.inverse_transform_targets(np.array([val_metrics.target_std]))[0]
            
            train_ratio_orig = train_pred_orig / train_tgt_orig if train_tgt_orig > 0 else 0
            val_ratio_orig = val_pred_orig / val_tgt_orig if val_tgt_orig > 0 else 0
            
            smoothed_val = self.early_stopping.smoothed_metric if self.early_stopping.smoothed_metric is not None else 0.0
            
            # Log loss breakdown (means, not sums)
            train_mean_huber = train_loss_components.get('huber', 0)
            train_mean_cs_corr = train_loss_components.get('cs_corr_loss', 0)
            train_mean_var_penalty = train_loss_components.get('var_penalty', 0)
            train_mean_total = train_loss_components.get('total_loss', train_loss)
            
            val_mean_huber = val_loss_components.get('huber', 0)
            val_mean_cs_corr = val_loss_components.get('cs_corr_loss', 0)
            val_mean_var_penalty = val_loss_components.get('var_penalty', 0)
            val_mean_total = val_loss_components.get('total_loss', val_loss)
            
            logger.info(f"Loss breakdown - Train: huber={train_mean_huber:.4f}, cs_corr={train_mean_cs_corr:.4f}, var_penalty={train_mean_var_penalty:.4f}, total={train_mean_total:.4f}")
            logger.info(f"Loss breakdown - Val: huber={val_mean_huber:.4f}, cs_corr={val_mean_cs_corr:.4f}, var_penalty={val_mean_var_penalty:.4f}, total={val_mean_total:.4f}")
            logger.info(f"Loss weights: lambda_corr={self.config.lambda_corr}, lambda_var={self.config.lambda_var}")
            
            # Log loss components with exact keys
            logger.info(f"train_mean_huber={train_mean_huber:.4f}, train_mean_cs_corr={train_mean_cs_corr:.4f}, train_mean_var_penalty={train_mean_var_penalty:.4f}, train_mean_total={train_mean_total:.4f}")
            logger.info(f"val_mean_huber={val_mean_huber:.4f}, val_mean_cs_corr={val_mean_cs_corr:.4f}, val_mean_var_penalty={val_mean_var_penalty:.4f}, val_mean_total={val_mean_total:.4f}")
            
            # Log cross-section metrics
            logger.info(f"Cross-section: Train CS ratio (scaled/orig): {train_ratio_scaled:.3f}/{train_ratio_orig:.3f}, Val CS ratio (scaled/orig): {val_ratio_scaled:.3f}/{val_ratio_orig:.3f}")
            logger.info(f"Rank-IC: Val={val_metrics.rank_ic:.4f}, Val EMA={smoothed_val:.4f}")
            
            # TODO: Add per-day Rank-IC quantiles logging
            # Need to collect actual predictions and dates in validate_epoch method
            
            # Check for cross-sectional collapse abort
            if val_ratio_orig < 0.18:  # CS ratio threshold
                collapse_abort_epochs += 1
                logger.warning(f"Low CS ratio detected: {val_ratio_orig:.3f} < 0.18 (epoch {collapse_abort_epochs}/2)")
                if collapse_abort_epochs >= 2:
                    logger.error(f"COLLAPSE ABORT: CS ratio {val_ratio_orig:.3f} < 0.18 for 2+ epochs - stopping training")
                    break
            else:
                collapse_abort_epochs = 0  # Reset counter if ratio improves
            
            # Log progress using ResultsManager
            self.results_manager.log_epoch(
                epoch=epoch+1,
                train_loss=train_loss,
                val_loss=val_loss,
                val_ic=val_metrics.ic,
                val_rank_ic=val_metrics.rank_ic,
                train_ic=train_metrics.ic,
                train_rank_ic=train_metrics.rank_ic
            )
            
            # Save best model based on Val Rank-IC (only if positive)
            if val_metrics.rank_ic >= 0 and val_metrics.rank_ic > best_val_rank_ic:
                best_val_rank_ic = val_metrics.rank_ic
                best_model_state = self.model.state_dict().copy()
                best_checkpoint_epoch = epoch + 1
                best_checkpoint_rank_ic = val_metrics.rank_ic
                best_checkpoint_ic = val_metrics.ic
                best_checkpoint_std_ratio = val_metrics.prediction_std / val_metrics.target_std if val_metrics.target_std > 0 else 0
                logger.info(f"New best Val Rank-IC: {best_val_rank_ic:.4f} at epoch {epoch+1}")
            elif val_metrics.rank_ic < 0:
                logger.warning(f"Val Rank-IC negative: {val_metrics.rank_ic:.4f} at epoch {epoch+1} - not checkpointing")
            
            # Early stopping on -EMA(Rank-IC) with EMA smoothing
            if self.early_stopping(-val_metrics.rank_ic):
                smoothed_val = -self.early_stopping.smoothed_metric if self.early_stopping.smoothed_metric is not None else 0.0
                logger.info(f"Early stopping at epoch {epoch+1} (Val Rank-IC: {val_metrics.rank_ic:.4f}, Smoothed: {smoothed_val:.4f})")
                break
            
            # Log early stopping status
            patience = self.config.early_stopping_patience
            epochs_since_improvement = self.early_stopping.counter if hasattr(self.early_stopping, 'counter') else 0
            early_stop_status = f"patience={patience}, epochs_since_improvement={epochs_since_improvement}"
            logger.info(f"Early stop status: {early_stop_status}")
            
            # Print progress indicator
            print(f"E{epoch+1}", end=" ", flush=True)
            
        # Log early stopping one-liner
        early_stop_triggered = epochs_since_improvement >= patience
        logger.info(f"early_stop? {early_stop_triggered} | best_epoch={best_checkpoint_epoch} (val_rank_ic={best_checkpoint_rank_ic:.4f})")
        
        # Print completion message
        print(f"\n   Training completed! Best Val Rank-IC: {best_checkpoint_rank_ic:.4f} at epoch {best_checkpoint_epoch}")
        
        # Load best model and rerun validation to get fresh metrics
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model state")
            
            # Rerun validation on best checkpoint to get fresh metrics
            logger.info("Rerunning validation on best checkpoint...")
            _, best_val_metrics, _ = self._validate_epoch(val_loader, symbol_mapping)
            
            # Log best checkpoint info from stored metadata
            best_epoch = best_checkpoint_epoch
            best_val_rank_ic = best_checkpoint_rank_ic
            best_val_ic = best_checkpoint_ic
            best_val_std_ratio = best_checkpoint_std_ratio
            logger.info(f"Best checkpoint: epoch={best_epoch}, val_rank_ic={best_val_rank_ic:.4f}, val_ic={best_val_ic:.4f}, val_std_ratio={best_val_std_ratio:.3f}")
            
            # Log postload validation metrics
            postload_val_ic = best_val_metrics.ic
            postload_val_rank_ic = best_val_metrics.rank_ic
            postload_val_std_ratio = best_val_metrics.prediction_std / best_val_metrics.target_std if best_val_metrics.target_std > 0 else 0
            logger.info(f"Postload validation: ic={postload_val_ic:.4f}, rank_ic={postload_val_rank_ic:.4f}, std_ratio={postload_val_std_ratio:.3f}")
            logger.info(f"postload_val_ic={postload_val_ic:.4f}, postload_val_rank_ic={postload_val_rank_ic:.4f}, postload_val_std_ratio={postload_val_std_ratio:.3f}")
        else:
            logger.warning("No positive Rank-IC epochs found - model may not be suitable for trading")
            best_val_metrics = val_metrics  # Use last epoch metrics as fallback
            best_epoch = epoch + 1
            best_val_rank_ic = val_metrics.rank_ic
            best_val_ic = val_metrics.ic
            best_val_std_ratio = val_metrics.prediction_std / val_metrics.target_std if val_metrics.target_std > 0 else 0
        
        # Check if calibration would shrink signal using best checkpoint metrics
        val_predictions = self._get_raw_predictions(val_data, symbol_mapping)
        val_targets = val_data.targets
        
        # Fix scale bug: inverse transform predictions to original scale
        val_predictions_original = self.scaler_manager.inverse_transform_targets(val_predictions)
        
        calib_pred_std = float(np.std(val_predictions_original))
        val_target_std = float(np.std(val_targets))
        std_ratio = calib_pred_std / val_target_std if val_target_std > 0 else 0
        
        # Check validation correlation before deciding on calibration
        val_corr = np.corrcoef(val_predictions_original, val_targets)[0, 1]
        logger.info(f"Calibration check: val_corr={val_corr:.4f}, std_ratio={std_ratio:.3f}, val_rank_ic={best_val_metrics.rank_ic:.4f}")
        logger.info(f"Best checkpoint: epoch={best_epoch}, best_val_rank_ic={best_val_metrics.rank_ic:.4f}, best_val_ic={best_val_metrics.ic:.4f}, best_val_std_ratio={std_ratio:.3f}")
        
        apply_calibration = False
        calibration_method = "none"
        slope = 1.0
        intercept = 0.0
        pre_std_ratio = std_ratio
        post_std_ratio = std_ratio
        reason_skipped = "none"
        
        if val_corr <= 0 or best_val_metrics.rank_ic <= 0:
            # Flip signal sign instead of skipping calibration
            logger.info(f"Negative correlation/Rank-IC detected: corr={val_corr:.4f}, rank_ic={best_val_metrics.rank_ic:.4f} - flipping signal sign")
            val_predictions_original = -val_predictions_original
            val_corr = -val_corr  # Update correlation after flipping
            logger.info(f"Signal flipped - new correlation: {val_corr:.4f}")
            # Continue with calibration
        elif std_ratio < 0.15:
            reason_skipped = f"would shrink signal (std ratio {std_ratio:.3f} < 0.15)"
            logger.info(f"Skipping calibration: {reason_skipped}")
        elif std_ratio <= 0.8:
            logger.info(f"Applying variance-matching calibration: std ratio {std_ratio:.3f} is beneficial (0.15 <= {std_ratio:.3f} <= 0.8)")
            self._fit_variance_matching_calibration(val_predictions_original, val_targets)
            apply_calibration = True
            calibration_method = "variance_match"
            slope = self.calibration_b
            intercept = self.calibration_a
            # Calculate post-calibration std ratio
            val_predictions_cal = self._apply_calibration(val_predictions_original)
            post_std_ratio = np.std(val_predictions_cal) / np.std(val_targets)
        else:
            reason_skipped = f"would over-amplify signal (std ratio {std_ratio:.3f} > 0.8)"
            logger.info(f"Skipping calibration: {reason_skipped}")
        
        logger.info(f"Calibration result: apply={apply_calibration}, method={calibration_method}, slope={slope:.3f}, intercept={intercept:.6f}, pre_std_ratio={pre_std_ratio:.3f}, post_std_ratio={post_std_ratio:.3f}")
        logger.info(f"Calibration reference: epoch={best_epoch}, reason_skipped={reason_skipped}")
        
        # Log calibration with exact keys
        logger.info(f"apply_calibration={apply_calibration}, method={calibration_method}, slope={slope:.3f}, intercept={intercept:.6f}, pre_std_ratio={pre_std_ratio:.3f}, post_std_ratio={post_std_ratio:.3f}")
        logger.info(f"calibration_reference_epoch={best_epoch}")
        logger.info(f"val_rank_ic_used_for_decision={best_val_rank_ic:.4f}")
        if not apply_calibration:
            logger.info(f"calibration_reason_skipped={reason_skipped}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics
        }
    
    def evaluate(self, test_data: TrainingData) -> ModelMetrics:
        """Evaluate model on test data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        logger.info("Evaluating model on test data")
        
        # Create symbol mapping
        symbol_mapping = self._create_symbol_mapping(test_data.symbols)
        
        # Prepare data
        X_test, y_test, symbols_test, dates_test = self._prepare_data(test_data)
        
        # Scale test data using fitted scalers
        X_test_scaled = self.scaler_manager.transform_features(X_test.numpy())
        y_test_scaled = self.scaler_manager.transform_targets(y_test.numpy())
        
        # Convert back to tensors
        X_test_scaled = torch.from_numpy(X_test_scaled).float()
        y_test_scaled = torch.from_numpy(y_test_scaled).float()
        
        # Create data loader
        test_loader = self._create_dataloader(X_test_scaled, y_test_scaled, symbols_test, dates_test, shuffle=False)
        
        # Evaluate
        self.model.eval()
        all_predictions_scaled = []
        all_targets_scaled = []
        all_dates = []
        
        with torch.no_grad():
            for batch_X, batch_y, batch_symbols, batch_dates in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_symbols = batch_symbols.to(self.device)
                
                # Convert symbols to indices
                symbol_indices = torch.tensor([
                    symbol_mapping[symbol.item()] for symbol in batch_symbols
                ], device=self.device)
                
                # Forward pass
                predictions = self.model(batch_X, symbol_indices)
                
                # Store predictions for metrics
                all_predictions_scaled.extend(predictions.detach().cpu().numpy())
                all_targets_scaled.extend(batch_y.detach().cpu().numpy())
                all_dates.extend(batch_dates.detach().cpu().numpy())
        
        # Inverse transform predictions and targets to original scale
        predictions_scaled = np.array(all_predictions_scaled)
        targets_scaled = np.array(all_targets_scaled)
        dates = np.array(all_dates)
        
        # Inverse transform to original scale
        predictions = self.scaler_manager.inverse_transform_targets(predictions_scaled)
        targets = self.scaler_manager.inverse_transform_targets(targets_scaled)
        
        # Apply calibration
        predictions = self._apply_calibration(predictions)
        
        # Apply per-date z-scoring to ensure proper cross-sectional ranking
        predictions = self._apply_per_date_zscoring(predictions, dates)
        
        # Calculate metrics on original scale
        daily_ic_mean, daily_ic_std, n_days = calculate_daily_ic(predictions, targets, dates)
        daily_rank_ic_mean, daily_rank_ic_std, n_rank_days = calculate_daily_rank_ic(predictions, targets, dates)
        
        # Check for prediction bias
        mean_pred = np.mean(predictions)
        mean_target = np.mean(targets)
        std_target = np.std(targets)
        bias_threshold = 0.25 * std_target
        
        if abs(mean_pred - mean_target) > bias_threshold:
            logger.warning(f"PREDICTION BIAS: mean_pred={mean_pred:.6f}, mean_target={mean_target:.6f}, threshold={bias_threshold:.6f}")
        
        metrics = ModelMetrics(
            ic=daily_ic_mean,
            rank_ic=daily_rank_ic_mean,
            mse=float(np.mean((predictions - targets) ** 2)),
            rmse=float(np.sqrt(np.mean((predictions - targets) ** 2))),
            n_samples=len(predictions),
            prediction_std=float(np.std(predictions)),
            target_std=float(np.std(targets)),
            constant_predictions=np.std(predictions) < 1e-8
        )
        
        logger.info(f"Test IC: {metrics.ic:.4f}, Test Rank-IC: {metrics.rank_ic:.4f}")
        logger.info(f"Test MSE: {metrics.mse:.6f}, Test RMSE: {metrics.rmse:.6f}")
        logger.info(f"Prediction stats: mean={mean_pred:.6f}, std={metrics.prediction_std:.6f}")
        logger.info(f"Target stats: mean={mean_target:.6f}, std={std_target:.6f}")
        
        # Calculate per-day Rank-IC quantiles for test
        test_daily_rank_ics = []
        for date in np.unique(dates):
            date_mask = dates == date
            if np.sum(date_mask) >= 3:  # Need at least 3 samples
                date_preds = predictions[date_mask]
                date_targets = targets[date_mask]
                date_rank_ic = np.corrcoef(np.argsort(np.argsort(date_preds)), np.argsort(np.argsort(date_targets)))[0, 1]
                if not np.isnan(date_rank_ic):
                    test_daily_rank_ics.append(date_rank_ic)
        
        if test_daily_rank_ics:
            test_rank_ic_q10 = np.percentile(test_daily_rank_ics, 10)
            test_rank_ic_median = np.median(test_daily_rank_ics)
            test_rank_ic_q90 = np.percentile(test_daily_rank_ics, 90)
            logger.info(f"test_median_daily_rank_ic={test_rank_ic_median:.4f}, test_rank_ic_q10={test_rank_ic_q10:.4f}, test_rank_ic_q90={test_rank_ic_q90:.4f}")
        else:
            logger.info("test_median_daily_rank_ic=nan, test_rank_ic_q10=nan, test_rank_ic_q90=nan")
        
        # Calculate backtest realism metrics
        # Simple turnover calculation: sum of absolute position changes
        position_changes = np.abs(np.diff(predictions))
        turnover_annualized = np.mean(position_changes) * 252  # Approximate annual turnover
        avg_cost_bps_assumed = 7.5  # 7.5 bps per trade assumption
        cost_drag = turnover_annualized * avg_cost_bps_assumed / 10000  # Convert to decimal
        pnl_after_costs = metrics.ic - cost_drag  # Rough approximation
        
        logger.info(f"turnover_annualized={turnover_annualized:.3f}, avg_cost_bps_assumed={avg_cost_bps_assumed:.1f}, pnl_after_costs={pnl_after_costs:.4f}")
        
        return metrics
    
    def predict(self, data: TrainingData) -> np.ndarray:
        """Make predictions on data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        
        # Create symbol mapping
        symbol_mapping = self._create_symbol_mapping(data.symbols)
        
        # Prepare data
        X, y, symbols, dates = self._prepare_data(data)
        
        # Scale data using fitted scalers
        X_scaled = self.scaler_manager.transform_features(X.numpy())
        y_scaled = self.scaler_manager.transform_targets(y.numpy())
        
        # Convert back to tensors
        X_scaled = torch.from_numpy(X_scaled).float()
        y_scaled = torch.from_numpy(y_scaled).float()
        
        # Create data loader
        dataloader = self._create_dataloader(X_scaled, y_scaled, symbols, dates, shuffle=False)
        
        all_predictions_scaled = []
        
        with torch.no_grad():
            for batch_X, batch_y, batch_symbols, batch_dates in dataloader:
                batch_X = batch_X.to(self.device)
                batch_symbols = batch_symbols.to(self.device)
                
                # Convert symbols to indices
                symbol_indices = torch.tensor([
                    symbol_mapping[symbol.item()] for symbol in batch_symbols
                ], device=self.device)
                
                # Forward pass
                predictions = self.model(batch_X, symbol_indices)
                all_predictions_scaled.extend(predictions.detach().cpu().numpy())
        
        # Inverse transform predictions to original scale
        predictions_scaled = np.array(all_predictions_scaled)
        predictions = self.scaler_manager.inverse_transform_targets(predictions_scaled)
        
        # Apply calibration
        predictions = self._apply_calibration(predictions)
        
        # Apply per-date z-scoring to ensure proper cross-sectional ranking
        predictions = self._apply_per_date_zscoring(predictions, dates)
        
        return predictions
    
    def _fit_calibration(self, val_data: TrainingData, symbol_mapping: Dict[int, int]):
        """Fit calibration parameters on validation set."""
        logger.info("Fitting prediction calibration on validation set...")
        
        # Get raw predictions on validation set
        X_val, y_val, symbols_val, dates_val = self._prepare_data(val_data)
        X_val_scaled = self.scaler_manager.transform_features(X_val.numpy())
        y_val_scaled = self.scaler_manager.transform_targets(y_val.numpy())
        X_val_scaled = torch.from_numpy(X_val_scaled).float()
        y_val_scaled = torch.from_numpy(y_val_scaled).float()
        
        val_loader = self._create_dataloader(X_val_scaled, y_val_scaled, symbols_val, dates_val, shuffle=False)
        
        self.model.eval()
        raw_predictions = []
        raw_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y, batch_symbols, batch_dates in val_loader:
                batch_X = batch_X.to(self.device)
                batch_symbols = batch_symbols.to(self.device)
                
                symbol_indices = torch.tensor([
                    symbol_mapping[symbol.item()] for symbol in batch_symbols
                ], device=self.device)
                
                predictions = self.model(batch_X, symbol_indices)
                raw_predictions.extend(predictions.detach().cpu().numpy())
                raw_targets.extend(batch_y.detach().cpu().numpy())
        
        # Convert to original scale
        raw_preds = np.array(raw_predictions)
        raw_targets = np.array(raw_targets)
        
        preds_orig = self.scaler_manager.inverse_transform_targets(raw_preds)
        targets_orig = self.scaler_manager.inverse_transform_targets(raw_targets)
        
        # Fit calibration: y = a + b * yhat
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(preds_orig.reshape(-1, 1), targets_orig)
        
        self.calibration_a = reg.intercept_
        self.calibration_b = reg.coef_[0]
        
        # Log calibration stats
        calibrated_preds = self.calibration_a + self.calibration_b * preds_orig
        pred_std = np.std(calibrated_preds)
        target_std = np.std(targets_orig)
        
        logger.info(f"Calibration fitted: y = {self.calibration_a:.6f} + {self.calibration_b:.6f} * yhat")
        logger.info(f"Calibrated pred std: {pred_std:.6f}, target std: {target_std:.6f}, ratio: {pred_std/target_std:.3f}")
        
        if pred_std < 0.25 * target_std:
            logger.warning(f"PREDICTION COLLAPSE: pred_std {pred_std:.6f} < 0.25 * target_std {target_std:.6f}")
    
    def _fit_variance_matching_calibration(self, val_predictions: np.ndarray, val_targets: np.ndarray):
        """Fit variance-matching calibration on validation set."""
        logger.info("Fitting variance-matching calibration on validation set...")
        
        # Calculate correlation for signal flipping decision
        val_corr = np.corrcoef(val_predictions, val_targets)[0, 1]
        
        # Mean/variance calibration on original scale
        mu_y = np.mean(val_targets)
        sig_y = np.std(val_targets) + 1e-8
        mu_ph = np.mean(val_predictions)
        sig_ph = np.std(val_predictions) + 1e-8
        
        scale = sig_y / sig_ph
        # Clamp the gain to avoid collapse/explosion
        scale = np.clip(scale, 0.5, 3.0)
        
        # Store calibration parameters
        self.calibration_a = mu_y - scale * mu_ph
        self.calibration_b = scale
        self.signal_flipped = val_corr <= 0  # Track if signal was flipped
        
        # Test calibration
        val_predictions_cal = (val_predictions - mu_ph) * scale + mu_y
        cal_std = np.std(val_predictions_cal)
        target_std = np.std(val_targets)
        ratio = cal_std / target_std if target_std > 0 else 0
        
        logger.info(f"Variance-matching calibration: y = {self.calibration_a:.6f} + {self.calibration_b:.6f} * (yhat - {mu_ph:.6f})")
        logger.info(f"Calibrated pred std: {cal_std:.6f}, target std: {target_std:.6f}, ratio: {ratio:.3f}")
        
        # Check if calibration helped
        if ratio < 0.25:
            logger.warning(f"PREDICTION COLLAPSE: pred_std {cal_std:.6f} < 0.25 * target_std {target_std:.6f}")
    
    def _apply_calibration(self, predictions: np.ndarray) -> np.ndarray:
        """Apply calibration to predictions."""
        if hasattr(self, 'calibration_a') and hasattr(self, 'calibration_b'):
            calibrated = self.calibration_a + self.calibration_b * predictions
            # Apply signal flipping if needed
            if hasattr(self, 'signal_flipped') and self.signal_flipped:
                calibrated = -calibrated
            return calibrated
        else:
            # Post-hoc z-score rescaling to prevent signal collapse
            if predictions.std() > 0:
                # Z-score the predictions to ensure proper dispersion
                z_scored = (predictions - predictions.mean()) / predictions.std()
                # Scale to target dispersion (0.5 * typical target std)
                target_std = 0.02  # Reasonable target for 10-day returns
                rescaled = z_scored * target_std
                logger.info(f"Applied post-hoc z-score rescaling: std {predictions.std():.6f} -> {rescaled.std():.6f}")
                return rescaled
            else:
                logger.warning("No calibration parameters found, returning raw predictions")
                return predictions
    
    def _apply_per_date_zscoring(self, predictions: np.ndarray, dates: np.ndarray) -> np.ndarray:
        """Apply per-date z-scoring to ensure proper cross-sectional ranking."""
        unique_dates = np.unique(dates)
        zscored_predictions = predictions.copy()
        
        for date in unique_dates:
            date_mask = dates == date
            if np.sum(date_mask) > 1:  # Need at least 2 samples for z-scoring
                date_preds = predictions[date_mask]
                if date_preds.std() > 0:
                    zscored = (date_preds - date_preds.mean()) / date_preds.std()
                    zscored_predictions[date_mask] = zscored
        
        logger.info(f"Applied per-date z-scoring: global std {predictions.std():.6f} -> {zscored_predictions.std():.6f}")
        return zscored_predictions
    
    def _get_raw_predictions(self, data: TrainingData, symbol_mapping: Dict[int, int]) -> np.ndarray:
        """Get raw predictions without calibration for analysis."""
        X, y, symbols, dates = self._prepare_data(data)
        X_scaled = self.scaler_manager.transform_features(X.numpy())
        y_scaled = self.scaler_manager.transform_targets(y.numpy())
        X_scaled = torch.from_numpy(X_scaled).float()
        y_scaled = torch.from_numpy(y_scaled).float()
        
        val_loader = self._create_dataloader(X_scaled, y_scaled, symbols, dates, shuffle=False)
        
        self.model.eval()
        raw_predictions = []
        
        with torch.no_grad():
            for batch_X, batch_y, batch_symbols, batch_dates in val_loader:
                batch_X = batch_X.to(self.device)
                batch_symbols = batch_symbols.to(self.device)
                
                symbol_indices = torch.tensor([
                    symbol_mapping[symbol.item()] for symbol in batch_symbols
                ], device=self.device)
                
                predictions = self.model(batch_X, symbol_indices)
                raw_predictions.extend(predictions.detach().cpu().numpy())
        
        return np.array(raw_predictions)
    
    def save_model(self, filepath: str):
        """Save model checkpoint."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "model_family": self.model_family,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.config = checkpoint["config"]
        self.model_family = checkpoint["model_family"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.train_metrics = checkpoint["train_metrics"]
        self.val_metrics = checkpoint["val_metrics"]
        
        # Create model
        input_dim = self.config.sequence_length  # This should be set properly
        self.model = self._create_model(input_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        
        logger.info(f"Model loaded from {filepath}")


def train_model(train_data: TrainingData, val_data: TrainingData, 
                config: TrainingConfig, model_family: str = "lstm") -> ModelTrainer:
    """Convenience function to train a model."""
    trainer = ModelTrainer(config, model_family)
    trainer.train(train_data, val_data)
    return trainer
