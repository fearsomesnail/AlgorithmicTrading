"""Feature and target scaling with proper train-only fitting."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
import logging
import joblib
import os

logger = logging.getLogger(__name__)


class ScalerManager:
    """Manages feature and target scaling with proper train-only fitting."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit scalers on training data only."""
        logger.info("Fitting scalers on training data only...")
        
        # Reshape for scaler (samples, features)
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        
        # Fit scalers
        self.feature_scaler.fit(X_train_2d)
        self.target_scaler.fit(y_train.reshape(-1, 1))
        
        self.fitted = True
        
        # Log scaling statistics
        logger.info(f"Feature scaling - Mean: {self.feature_scaler.mean_[:5]}, Std: {self.feature_scaler.scale_[:5]}")
        logger.info(f"Target scaling - Mean: {self.target_scaler.mean_[0]:.6f}, Std: {self.target_scaler.scale_[0]:.6f}")
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        X_2d = X.reshape(X.shape[0], -1)
        X_scaled = self.feature_scaler.transform(X_2d)
        return X_scaled.reshape(X.shape)
    
    def transform_targets(self, y: np.ndarray) -> np.ndarray:
        """Transform targets using fitted scaler."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        y_scaled = self.target_scaler.transform(y.reshape(-1, 1))
        return y_scaled.flatten()
    
    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform targets back to original scale."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        y_original = self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1))
        return y_original.flatten()
    
    def save(self, run_dir: str):
        """Save scalers to disk."""
        scaler_dir = os.path.join(run_dir, "scalers")
        os.makedirs(scaler_dir, exist_ok=True)
        
        joblib.dump(self.feature_scaler, os.path.join(scaler_dir, "feature_scaler.pkl"))
        joblib.dump(self.target_scaler, os.path.join(scaler_dir, "target_scaler.pkl"))
        
        logger.info(f"Scalers saved to {scaler_dir}")
    
    def load(self, run_dir: str):
        """Load scalers from disk."""
        scaler_dir = os.path.join(run_dir, "scalers")
        
        self.feature_scaler = joblib.load(os.path.join(scaler_dir, "feature_scaler.pkl"))
        self.target_scaler = joblib.load(os.path.join(scaler_dir, "target_scaler.pkl"))
        self.fitted = True
        
        logger.info(f"Scalers loaded from {scaler_dir}")


def validate_feature_target_alignment(features: pd.DataFrame, targets: pd.Series, 
                                    horizon_days: int = 5) -> bool:
    """Validate that features and targets are properly aligned with horizon."""
    
    logger.info("Validating feature-target alignment...")
    
    # Check that target index is properly shifted
    target_max_date = targets.index.max()
    feature_max_date = features.index.max()
    
    expected_gap = pd.Timedelta(days=horizon_days)
    actual_gap = feature_max_date - target_max_date
    
    if actual_gap < expected_gap:
        logger.warning(f"Feature-target gap too small: {actual_gap} < {expected_gap}")
        return False
    
    # Check for data leakage (correlation at same time)
    aligned_data = pd.concat([features, targets], axis=1, join='inner')
    if len(aligned_data) == 0:
        logger.error("No aligned data found between features and targets")
        return False
    
    # Check for NaN values after alignment
    nan_count = aligned_data.isnull().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values after alignment")
    
    logger.info(f"Alignment validation passed. Aligned samples: {len(aligned_data)}")
    return True


def enforce_feature_config(features: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    """Enforce that features exactly match configuration."""
    
    # Check for missing features
    missing = [c for c in expected_features if c not in features.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    # Check for extra features
    extra = [c for c in features.columns if c not in expected_features]
    if extra:
        logger.warning(f"Extra features found: {extra}. Dropping them.")
    
    # Return features in exact expected order
    features = features[expected_features]
    
    logger.info(f"Feature validation passed. Using {len(features.columns)} features: {list(features.columns)}")
    return features
