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

from ...core.types import (
    TrainingConfig, TrainingData, ModelMetrics, 
    calculate_metrics, calculate_daily_ic, calculate_daily_rank_ic
)
from .model_nn import ModelFactory, count_parameters
from .scaler_manager import ScalerManager
from ..results_manager import ResultsManager

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
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
        self.early_stopping = EarlyStopping(patience=10)  # Increased patience for Rank-IC with smoothing
        self.results_manager = results_manager or ResultsManager(enable_file_logging=False)
        self.scaler_manager = ScalerManager()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        logger.info(f"Initialized trainer with device: {self.device}")
    
    def _create_model(self, input_dim: int, num_symbols: int = 6) -> nn.Module:
        """Create model instance."""
        logger.info(f"Creating {self.model_family} model with input_dim={input_dim}, num_symbols={num_symbols}")
        if self.model_family == "lstm":
            from .model_nn import LSTMRegressor
            model = LSTMRegressor(
                input_dim=input_dim,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                embedding_dim=self.config.embedding_dim,
                dropout=self.config.dropout,
                num_symbols=num_symbols
            )
            logger.info(f"LSTMRegressor created successfully")
            return model
        elif self.model_family == "simple_lstm":
            from .model_nn import SimpleLSTM
            model = SimpleLSTM(
                input_dim=input_dim,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers
            )
            logger.info(f"SimpleLSTM created successfully")
            return model
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")
    
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
    
    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss with dispersion penalty."""
        mse_loss = nn.MSELoss()
        mse = mse_loss(predictions, targets)
        
        # Add dispersion penalty to encourage prediction variance
        pred_std = torch.std(predictions)
        target_std = torch.std(targets)
        
        # Penalize if prediction std is too low relative to target std
        dispersion_penalty = 0.0
        if target_std > 1e-6:  # Avoid division by zero
            dispersion_ratio = pred_std / (target_std + 1e-6)
            if dispersion_ratio < 0.1:  # If predictions are too flat
                dispersion_penalty = 0.1 * (0.1 - dispersion_ratio) ** 2
        
        return mse + dispersion_penalty
    
    def _train_epoch(self, dataloader: DataLoader, symbol_mapping: Dict[str, int]) -> Tuple[float, ModelMetrics]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_dates = []
        
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
            loss = self._compute_loss(predictions, batch_y)
            
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
            constant_predictions=np.std(predictions) < 1e-8
        )
        
        return avg_loss, metrics
    
    def _validate_epoch(self, dataloader: DataLoader, symbol_mapping: Dict[str, int]) -> Tuple[float, ModelMetrics]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_dates = []
        
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
                loss = self._compute_loss(predictions, batch_y)
                
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
            constant_predictions=np.std(predictions) < 1e-8
        )
        
        return avg_loss, metrics
    
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
        
        # Fit scalers on training data only
        self.scaler_manager.fit(X_train, y_train)
        
        # Transform data using fitted scalers
        X_train_scaled = self.scaler_manager.transform_features(X_train)
        y_train_scaled = self.scaler_manager.transform_targets(y_train)
        X_val_scaled = self.scaler_manager.transform_features(X_val)
        y_val_scaled = self.scaler_manager.transform_targets(y_val)
        
        # Convert scaled data back to tensors
        X_train_scaled = torch.from_numpy(X_train_scaled).float()
        y_train_scaled = torch.from_numpy(y_train_scaled).float()
        X_val_scaled = torch.from_numpy(X_val_scaled).float()
        y_val_scaled = torch.from_numpy(y_val_scaled).float()
        
        # Log scaling statistics
        logger.info(f"Training data - Features std: {X_train_scaled.std():.6f}, Targets std: {y_train_scaled.std():.6f}")
        logger.info(f"Validation data - Features std: {X_val_scaled.std():.6f}, Targets std: {y_val_scaled.std():.6f}")
        
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
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Create data loaders with scaled data
        logger.info(f"Creating data loaders...")
        train_loader = self._create_dataloader(X_train_scaled, y_train_scaled, symbols_train, dates_train, shuffle=True)
        val_loader = self._create_dataloader(X_val_scaled, y_val_scaled, symbols_val, dates_val, shuffle=False)
        logger.info(f"Data loaders created successfully")
        
        logger.info(f"Model created successfully")
        logger.info(f"Model parameters: {count_parameters(self.model):,}")
        logger.info(f"Training samples: {len(train_data.features)}")
        logger.info(f"Validation samples: {len(val_data.features)}")
        
        # Training loop
        best_val_loss = float('inf')
        best_val_rank_ic = float('-inf')
        best_model_state = None
        smoothed_rank_ic = None
        alpha = 0.3  # EMA smoothing factor
        
        logger.info(f"Starting training loop for {self.config.max_epochs} epochs...")
        for epoch in range(self.config.max_epochs):
            # Train
            train_loss, train_metrics = self._train_epoch(train_loader, symbol_mapping)
            
            # Validate
            val_loss, val_metrics = self._validate_epoch(val_loader, symbol_mapping)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
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
            
            # Save best model based on Val Rank-IC
            if val_metrics.rank_ic > best_val_rank_ic:
                best_val_rank_ic = val_metrics.rank_ic
                best_model_state = self.model.state_dict().copy()
                logger.info(f"New best Val Rank-IC: {best_val_rank_ic:.4f} at epoch {epoch+1}")
            
            # Early stopping on Val Rank-IC (smoothed with EMA)
            if smoothed_rank_ic is None:
                smoothed_rank_ic = val_metrics.rank_ic
            else:
                smoothed_rank_ic = alpha * val_metrics.rank_ic + (1 - alpha) * smoothed_rank_ic
            
            if self.early_stopping(-smoothed_rank_ic):  # Negative because we want to maximize Rank-IC
                logger.info(f"Early stopping at epoch {epoch+1} (Val Rank-IC: {val_metrics.rank_ic:.4f}, Smoothed: {smoothed_rank_ic:.4f})")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model state")
        
        # Check if calibration would shrink signal
        val_predictions = self._get_raw_predictions(val_data, symbol_mapping)
        val_targets = val_data.targets
        
        calib_pred_std = float(np.std(val_predictions))
        val_target_std = float(np.std(val_targets))
        std_ratio = calib_pred_std / val_target_std if val_target_std > 0 else 0
        
        if std_ratio < 0.15:
            logger.info(f"Skipping calibration: would shrink signal (std ratio {std_ratio:.3f} < 0.15)")
        else:
            logger.info(f"Calibration would be beneficial (std ratio {std_ratio:.3f} >= 0.15), but skipping for consistency")
        
        # Skip calibration to prevent signal shrinkage
        # self._fit_calibration(val_data, symbol_mapping)
        
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
            constant_predictions=np.std(predictions) < 1e-8
        )
        
        logger.info(f"Test IC: {metrics.ic:.4f}, Test Rank-IC: {metrics.rank_ic:.4f}")
        logger.info(f"Test MSE: {metrics.mse:.6f}, Test RMSE: {metrics.rmse:.6f}")
        logger.info(f"Prediction stats: mean={mean_pred:.6f}, std={metrics.prediction_std:.6f}")
        logger.info(f"Target stats: mean={mean_target:.6f}, std={std_target:.6f}")
        
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
    
    def _apply_calibration(self, predictions: np.ndarray) -> np.ndarray:
        """Apply calibration to predictions."""
        if hasattr(self, 'calibration_a') and hasattr(self, 'calibration_b'):
            return self.calibration_a + self.calibration_b * predictions
        else:
            logger.warning("No calibration parameters found, returning raw predictions")
            return predictions
    
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
