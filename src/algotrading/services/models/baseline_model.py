"""Baseline models for comparison."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class RidgeBaseline:
    """Ridge regression baseline model."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeBaseline':
        """Fit the baseline model."""
        # Flatten sequences for Ridge regression
        X_flat = X.reshape(X.shape[0], -1)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_flat)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Fitted Ridge baseline with alpha={self.alpha}")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Flatten sequences
        X_flat = X.reshape(X.shape[0], -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X_flat)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        return predictions
        
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (coefficients)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.coef_


def train_baseline_model(train_data, val_data, test_data, config) -> Dict[str, Any]:
    """Train a Ridge baseline model and return results."""
    from algotrading.core.types import calculate_metrics, calculate_daily_ic, calculate_daily_rank_ic
    
    logger.info("Training Ridge baseline model...")
    
    # Initialize model
    baseline = RidgeBaseline(alpha=1.0)
    
    # Flatten training data
    X_train = train_data.features.reshape(train_data.features.shape[0], -1)
    y_train = train_data.targets
    
    # Fit model
    baseline.fit(train_data.features, train_data.targets)
    
    # Evaluate on validation set
    val_predictions = baseline.predict(val_data.features)
    val_metrics = calculate_metrics(val_predictions, val_data.targets)
    
    # Calculate daily IC for validation
    val_ic_mean, val_ic_std, val_ic_count = calculate_daily_ic(
        val_predictions, val_data.targets, val_data.dates
    )
    val_rank_ic_mean, val_rank_ic_std, val_rank_ic_count = calculate_daily_rank_ic(
        val_predictions, val_data.targets, val_data.dates
    )
    
    # Evaluate on test set
    test_predictions = baseline.predict(test_data.features)
    test_metrics = calculate_metrics(test_predictions, test_data.targets)
    
    # Calculate daily IC for test
    test_ic_mean, test_ic_std, test_ic_count = calculate_daily_ic(
        test_predictions, test_data.targets, test_data.dates
    )
    test_rank_ic_mean, test_rank_ic_std, test_rank_ic_count = calculate_daily_rank_ic(
        test_predictions, test_data.targets, test_data.dates
    )
    
    # Log results
    logger.info(f"Baseline Val IC: {val_ic_mean:.4f} +/- {val_ic_std:.4f} (n={val_ic_count})")
    logger.info(f"Baseline Val Rank-IC: {val_rank_ic_mean:.4f} +/- {val_rank_ic_std:.4f} (n={val_rank_ic_count})")
    logger.info(f"Baseline Test IC: {test_ic_mean:.4f} +/- {test_ic_std:.4f} (n={test_ic_count})")
    logger.info(f"Baseline Test Rank-IC: {test_rank_ic_mean:.4f} +/- {test_rank_ic_std:.4f} (n={test_rank_ic_count})")
    logger.info(f"Baseline Test MSE: {test_metrics.mse:.6f}, RMSE: {test_metrics.rmse:.6f}")
    logger.info(f"Baseline Prediction std: {test_metrics.prediction_std:.6f}")
    
    return {
        "model": baseline,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_ic": (val_ic_mean, val_ic_std, val_ic_count),
        "val_rank_ic": (val_rank_ic_mean, val_rank_ic_std, val_rank_ic_count),
        "test_ic": (test_ic_mean, test_ic_std, test_ic_count),
        "test_rank_ic": (test_rank_ic_mean, test_rank_ic_std, test_rank_ic_count),
        "predictions": test_predictions
    }
