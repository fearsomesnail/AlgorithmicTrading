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
    
    def __init__(self, results_dir: str = "results", winsorize_pct: float = 0.01):
        self.results_dir = results_dir
        self.winsorize_pct = winsorize_pct
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.fitted = False
        self.feature_names = None
        self.symbol_winsorize_stats = {}  # Store per-symbol winsorization stats
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, symbols: np.ndarray = None, feature_names: list = None):
        """Fit scalers on training data only."""
        logger.info("Fitting scalers on training data only...")
        
        self.feature_names = feature_names
        
        # Reshape for scaler (samples, features)
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        
        # Apply per-symbol winsorization and z-scoring if symbols provided
        if symbols is not None:
            X_train_2d = self._apply_per_symbol_preprocessing(X_train_2d, symbols, fit=True)
        
        # Fit scalers
        self.feature_scaler.fit(X_train_2d)
        self.target_scaler.fit(y_train.reshape(-1, 1))
        
        self.fitted = True
        
        # Log scaling statistics with full arrays
        if self.feature_names:
            logger.info(f"Feature scaling - Mean: {self.feature_scaler.mean_}")
            logger.info(f"Feature scaling - Std: {self.feature_scaler.scale_}")
            for i, name in enumerate(self.feature_names):
                logger.info(f"  {name}: mean={self.feature_scaler.mean_[i]:.6f}, std={self.feature_scaler.scale_[i]:.6f}")
        else:
            logger.info(f"Feature scaling - Mean: {self.feature_scaler.mean_}, Std: {self.feature_scaler.scale_}")
        logger.info(f"Target scaling - Mean: {self.target_scaler.mean_[0]:.6f}, Std: {self.target_scaler.scale_[0]:.6f}")
    
    def _apply_per_symbol_preprocessing(self, X: np.ndarray, symbols: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply per-symbol winsorization and z-scoring."""
        X_processed = X.copy()
        
        for symbol in np.unique(symbols):
            symbol_mask = symbols == symbol
            symbol_data = X[symbol_mask]
            
            if fit:
                # Store winsorization stats for this symbol
                self.symbol_winsorize_stats[symbol] = {}
            
            for feat_idx in range(X.shape[1]):
                feature_data = symbol_data[:, feat_idx]
                
                if fit:
                    # Winsorize at specified percentiles
                    lower_pct = np.percentile(feature_data, self.winsorize_pct * 100)
                    upper_pct = np.percentile(feature_data, (1 - self.winsorize_pct) * 100)
                    
                    # Store stats for later use
                    self.symbol_winsorize_stats[symbol][feat_idx] = {
                        'lower': lower_pct,
                        'upper': upper_pct,
                        'mean': np.mean(feature_data),
                        'std': np.std(feature_data)
                    }
                    
                    # Apply winsorization
                    feature_data = np.clip(feature_data, lower_pct, upper_pct)
                    
                    # Z-score
                    feature_data = (feature_data - self.symbol_winsorize_stats[symbol][feat_idx]['mean']) / self.symbol_winsorize_stats[symbol][feat_idx]['std']
                else:
                    # Use stored stats for transform
                    if symbol in self.symbol_winsorize_stats and feat_idx in self.symbol_winsorize_stats[symbol]:
                        stats = self.symbol_winsorize_stats[symbol][feat_idx]
                        # Apply winsorization
                        feature_data = np.clip(feature_data, stats['lower'], stats['upper'])
                        # Z-score
                        feature_data = (feature_data - stats['mean']) / stats['std']
                
                X_processed[symbol_mask, feat_idx] = feature_data
        
        return X_processed
    
    def transform_features(self, X: np.ndarray, symbols: np.ndarray = None) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        X_2d = X.reshape(X.shape[0], -1)
        
        # Apply per-symbol preprocessing if symbols provided
        if symbols is not None and self.symbol_winsorize_stats:
            X_2d = self._apply_per_symbol_preprocessing(X_2d, symbols, fit=False)
        
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
    
    def validate_feature_stds(self, X_val: np.ndarray, symbols: np.ndarray = None) -> bool:
        """Validate that validation feature standard deviations are reasonable."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        X_val_2d = X_val.reshape(X_val.shape[0], -1)
        
        # Apply per-symbol preprocessing if symbols provided
        if symbols is not None and self.symbol_winsorize_stats:
            X_val_2d = self._apply_per_symbol_preprocessing(X_val_2d, symbols, fit=False)
        
        # Transform using fitted scaler
        X_val_scaled = self.feature_scaler.transform(X_val_2d)
        
        # Check standard deviations
        val_stds = np.std(X_val_scaled, axis=0)
        
        logger.info(f"Validation feature stds: {val_stds}")
        
        # Check if any std is outside reasonable range [0.3, 1.7]
        bad_features = []
        for i, std in enumerate(val_stds):
            if std < 0.3 or std > 1.7:
                bad_features.append((i, std))
        
        if bad_features:
            logger.warning(f"Features with poor validation stds: {bad_features}")
            if self.feature_names:
                for feat_idx, std in bad_features:
                    if feat_idx < len(self.feature_names):
                        logger.warning(f"  {self.feature_names[feat_idx]}: std={std:.3f}")
            return False
        
        logger.info("All validation feature stds are within reasonable range [0.3, 1.7]")
        return True
    
    def save(self, run_dir: str):
        """Save scalers to disk."""
        scaler_dir = os.path.join(run_dir, "scalers")
        os.makedirs(scaler_dir, exist_ok=True)
        
        joblib.dump(self.feature_scaler, os.path.join(scaler_dir, "feature_scaler.pkl"))
        joblib.dump(self.target_scaler, os.path.join(scaler_dir, "target_scaler.pkl"))
        joblib.dump(self.symbol_winsorize_stats, os.path.join(scaler_dir, "symbol_winsorize_stats.pkl"))
        joblib.dump(self.feature_names, os.path.join(scaler_dir, "feature_names.pkl"))
        
        logger.info(f"Scalers saved to {scaler_dir}")
    
    def load(self, run_dir: str):
        """Load scalers from disk."""
        scaler_dir = os.path.join(run_dir, "scalers")
        
        self.feature_scaler = joblib.load(os.path.join(scaler_dir, "feature_scaler.pkl"))
        self.target_scaler = joblib.load(os.path.join(scaler_dir, "target_scaler.pkl"))
        self.symbol_winsorize_stats = joblib.load(os.path.join(scaler_dir, "symbol_winsorize_stats.pkl"))
        self.feature_names = joblib.load(os.path.join(scaler_dir, "feature_names.pkl"))
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
