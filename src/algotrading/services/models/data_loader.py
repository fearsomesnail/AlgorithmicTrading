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
        # Expanded ASX200 universe - major liquid stocks
        self.universe = [
            # Top 20 by market cap
            "BHP.AX", "CBA.AX", "CSL.AX", "WES.AX", "WBC.AX", "TLS.AX", "ANZ.AX", "NAB.AX", 
            "RIO.AX", "FMG.AX", "WOW.AX", "WDS.AX", "STO.AX", "ALL.AX", "QAN.AX", "SUN.AX",
            "TCL.AX", "BXB.AX", "S32.AX", "WPL.AX",
            # Additional liquid ASX200 stocks
            "AGL.AX", "AMP.AX", "ASX.AX", "BEN.AX", "BOQ.AX", "BSL.AX", "CAR.AX", "CCL.AX",
            "COH.AX", "CPU.AX", "CSR.AX", "CTD.AX", "DMP.AX", "DXS.AX", "ELD.AX", "EVN.AX",
            "FLT.AX", "FPH.AX", "GMA.AX", "GPT.AX", "HUB.AX", "IEL.AX", "IGO.AX", "JBH.AX",
            "JHX.AX", "LLC.AX", "MGR.AX", "MPL.AX", "NCM.AX", "NEC.AX", "NST.AX", "ORG.AX",
            "OSH.AX", "PLS.AX", "PTM.AX", "QBE.AX", "REA.AX", "RHC.AX", "RMD.AX", "SDF.AX",
            "SEK.AX", "SGM.AX", "SGR.AX", "SHL.AX", "SIG.AX", "SKC.AX", "SLR.AX", "SPK.AX",
            "STG.AX", "SUL.AX", "SUN.AX", "TCL.AX", "TLS.AX", "TPG.AX", "TWE.AX", "VEA.AX",
            "VOC.AX", "WBC.AX", "WDS.AX", "WES.AX", "WOW.AX", "WPL.AX", "XRO.AX", "ZIM.AX"
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
            
            # Check if symbol has valid data
            symbol_prices = prices[symbol].dropna()
            if len(symbol_prices) < 50:  # Need at least 50 data points for meaningful features
                logger.warning(f"Skipping {symbol}: insufficient data ({len(symbol_prices)} points)")
                continue
                
            symbol_data = pd.DataFrame(index=prices.index)
            
            # Price returns
            symbol_data["ret1"] = returns[symbol]
            symbol_data["ret5"] = prices[symbol].pct_change(5)
            symbol_data["ret21"] = prices[symbol].pct_change(21)
            
            # RSI calculations
            symbol_data["rsi14"] = self._calculate_rsi(returns[symbol], window=14)
            symbol_data["rsi2"] = self._calculate_rsi(returns[symbol], window=2)
            symbol_data["rsi50"] = self._calculate_rsi(returns[symbol], window=50)
            
            # MACD
            macd_line, macd_signal, macd_hist = self._calculate_macd(prices[symbol])
            symbol_data["macd"] = macd_line
            symbol_data["macd_signal"] = macd_signal
            symbol_data["macd_hist"] = macd_hist
            
            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(prices[symbol], volumes[symbol] if symbol in volumes.columns else None)
            symbol_data["stoch_k"] = stoch_k
            symbol_data["stoch_d"] = stoch_d
            
            # ATR
            symbol_data["atr14"] = self._calculate_atr(prices[symbol], volumes[symbol] if symbol in volumes.columns else None)
            
            # Donchian width
            donchian_high = prices[symbol].rolling(20).max()
            donchian_low = prices[symbol].rolling(20).min()
            symbol_data["donchian_width"] = (donchian_high - donchian_low) / prices[symbol]
            
            # Rolling skewness and kurtosis
            symbol_data["skew_21"] = returns[symbol].rolling(21).skew()
            symbol_data["kurt_21"] = returns[symbol].rolling(21).kurt()
            
            # Realized volatility
            symbol_data["realized_vol_5"] = returns[symbol].rolling(5).std() * np.sqrt(252)
            symbol_data["realized_vol_21"] = returns[symbol].rolling(21).std() * np.sqrt(252)
            
            # Volume indicators
            if symbol in volumes.columns:
                vol_mean = volumes[symbol].rolling(60).mean()
                vol_std = volumes[symbol].rolling(60).std()
                symbol_data["volz"] = (volumes[symbol] - vol_mean) / (vol_std + 1e-9)
                symbol_data["volz_5"] = (volumes[symbol] - volumes[symbol].rolling(5).mean()) / (volumes[symbol].rolling(5).std() + 1e-9)
                symbol_data["volz_21"] = (volumes[symbol] - volumes[symbol].rolling(21).mean()) / (volumes[symbol].rolling(21).std() + 1e-9)
                
                # OBV change
                obv = self._calculate_obv(prices[symbol], volumes[symbol])
                obv_change = obv.pct_change(5)
                # Handle inf/nan values in OBV change
                obv_change = obv_change.replace([np.inf, -np.inf], 0).fillna(0)
                symbol_data["obv_change"] = obv_change
                
                # Volume ratio
                symbol_data["volume_ratio"] = volumes[symbol] / volumes[symbol].rolling(21).mean()
            else:
                symbol_data["volz"] = 0.0
                symbol_data["volz_5"] = 0.0
                symbol_data["volz_21"] = 0.0
                symbol_data["obv_change"] = 0.0
                symbol_data["volume_ratio"] = 1.0
            
            # Volatility (re-enabled)
            symbol_data["volatility"] = returns[symbol].rolling(21).std() * np.sqrt(252)
            
            # Cross-sectional factors
            symbol_data["inv_price"] = 1.0 / prices[symbol]  # Proxy for size
            
            # Momentum (skip last 21 days to avoid look-ahead)
            symbol_data["mom_3m"] = prices[symbol].pct_change(63)  # 3 months
            symbol_data["mom_6m"] = prices[symbol].pct_change(126)  # 6 months
            symbol_data["mom_12m"] = prices[symbol].pct_change(252)  # 12 months
            
            # Short-term reversal
            symbol_data["reversal_1d"] = -returns[symbol]
            symbol_data["reversal_3d"] = -prices[symbol].pct_change(3)
            
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
                    if feature == 'ret1' and abs(mean_val) > 0.02:
                        raise ValueError(f"FEATURE ORDER MISMATCH: {feature} mean {mean_val:.6f} too high - likely wrong column order")
                    elif feature == 'ret5' and abs(mean_val) > 0.05:
                        raise ValueError(f"FEATURE ORDER MISMATCH: {feature} mean {mean_val:.6f} too high - likely wrong column order")
                    elif feature == 'ret21' and abs(mean_val) > 0.15:
                        raise ValueError(f"FEATURE ORDER MISMATCH: {feature} mean {mean_val:.6f} too high - likely wrong column order")
                    elif feature == 'rsi14' and not np.isnan(mean_val) and not (35 <= mean_val <= 65):
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
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD line, signal, and histogram."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_hist = macd_line - macd_signal
        return macd_line, macd_signal, macd_hist
    
    def _calculate_stochastic(self, prices: pd.Series, volumes: pd.Series = None, k_period: int = 14, d_period: int = 3) -> tuple:
        """Calculate Stochastic %K and %D."""
        if volumes is None:
            # Use high/low approximation from close prices
            high = prices.rolling(2).max()
            low = prices.rolling(2).min()
        else:
            # For now, use close as proxy for high/low
            high = prices
            low = prices
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        stoch_k = 100 * (prices - lowest_low) / (highest_high - lowest_low + 1e-9)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k, stoch_d
    
    def _calculate_atr(self, prices: pd.Series, volumes: pd.Series = None, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        if volumes is None:
            # Use close as proxy for high/low
            high = prices
            low = prices
        else:
            # For now, use close as proxy for high/low
            high = prices
            low = prices
        
        tr1 = high - low
        tr2 = abs(high - prices.shift(1))
        tr3 = abs(low - prices.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _calculate_obv(self, prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        price_change = prices.diff()
        obv = pd.Series(index=prices.index, dtype=float)
        obv.iloc[0] = 0
        
        for i in range(1, len(prices)):
            if pd.isna(price_change.iloc[i]) or pd.isna(volumes.iloc[i]):
                obv.iloc[i] = obv.iloc[i-1]
            elif price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volumes.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volumes.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        # Replace any inf/nan values with 0
        obv = obv.replace([np.inf, -np.inf], 0).fillna(0)
        return obv
    
    def calculate_targets(self, prices: pd.DataFrame, benchmark_prices: pd.Series, 
                         horizon: int = 10) -> Dict[str, pd.Series]:
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
        
        # Convert symbols to integer indices - only include symbols that have data
        unique_symbols = list(set(symbols_list))
        symbol_to_idx = {symbol: idx for idx, symbol in enumerate(unique_symbols)}
        symbols = np.array([symbol_to_idx[symbol] for symbol in symbols_list], dtype=np.int32)
        
        # Store the actual universe for later use
        self.actual_universe = unique_symbols
        logger.info(f"Actual universe size: {len(unique_symbols)} symbols")
        
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
        features_to_demean = ['ret1', 'ret5', 'ret21', 'rsi14', 'rsi2', 'rsi50', 'macd', 'macd_signal', 'macd_hist', 
                             'stoch_k', 'stoch_d', 'skew_21', 'kurt_21', 'realized_vol_5', 'realized_vol_21',
                             'volz', 'volz_5', 'volz_21', 'obv_change', 'volatility', 'volume_ratio',
                             'mom_3m', 'mom_6m', 'mom_12m', 'reversal_1d', 'reversal_3d']
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
            EXPECTED = self.config.features
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
