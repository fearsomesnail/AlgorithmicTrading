"""Backtesting metrics and evaluation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import spearmanr, pearsonr
import logging

logger = logging.getLogger(__name__)


def validate_backtest_integrity(predictions: pd.DataFrame, returns: pd.DataFrame, 
                               train_dates: pd.DatetimeIndex, val_dates: pd.DatetimeIndex,
                               test_dates: pd.DatetimeIndex, horizon_days: int = 5) -> Dict[str, Any]:
    """
    Validate backtest for leakage and integrity issues.
    
    Args:
        predictions: DataFrame with predictions (index=date, columns=symbols)
        returns: DataFrame with forward returns (index=date, columns=symbols) 
        train_dates: Training date range
        val_dates: Validation date range
        test_dates: Test date range
        horizon_days: Prediction horizon
        
    Returns:
        Dict with validation results and warnings
    """
    warnings = []
    errors = []
    
    # Check 1: No overlap between train/val/test dates in backtest
    backtest_dates = predictions.index
    train_overlap = backtest_dates.intersection(train_dates)
    val_overlap = backtest_dates.intersection(val_dates)
    
    if len(train_overlap) > 0:
        errors.append(f"Backtest contains {len(train_overlap)} training dates: {train_overlap[:5].tolist()}")
    if len(val_overlap) > 0:
        errors.append(f"Backtest contains {len(val_overlap)} validation dates: {val_overlap[:5].tolist()}")
    
    # Check 2: Predictions use only past information
    for date in backtest_dates:
        if date in train_dates or date in val_dates:
            errors.append(f"Backtest date {date} overlaps with training/validation")
    
    # Check 3: Returns are forward-looking (t -> t+H)
    for date in backtest_dates:
        expected_return_date = date + pd.Timedelta(days=horizon_days)
        if expected_return_date not in returns.index:
            warnings.append(f"No forward return available for {date} -> {expected_return_date}")
    
    # Check 4: Reasonable number of rebalance dates and symbols
    n_rebalance_dates = len(backtest_dates)
    avg_symbols_per_date = predictions.count(axis=1).mean()
    total_trades = predictions.count().sum()
    
    if n_rebalance_dates == 0:
        errors.append("No rebalance dates found in backtest")
    elif n_rebalance_dates < 10:
        warnings.append(f"Very few rebalance dates: {n_rebalance_dates}")
    
    if avg_symbols_per_date < 2:
        errors.append(f"Too few symbols per date: {avg_symbols_per_date:.1f}")
    
    # Check 5: IC vs Sharpe sanity check
    if len(predictions) > 0 and len(returns) > 0:
        # Calculate IC on backtest panel
        pred_flat = predictions.values.flatten()
        ret_flat = returns.values.flatten()
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(ret_flat))
        
        if np.sum(valid_mask) > 10:
            backtest_ic = calculate_ic(pred_flat[valid_mask], ret_flat[valid_mask])
            
            # Calculate Sharpe from returns
            if len(ret_flat[valid_mask]) > 1:
                returns_std = np.std(ret_flat[valid_mask])
                if returns_std > 0:
                    returns_mean = np.mean(ret_flat[valid_mask])
                    backtest_sharpe = returns_mean / returns_std * np.sqrt(252)  # Annualized
                    
                    # Sanity check: IC ~0.05 shouldn't yield Sharpe >2 with 6 names
                    if abs(backtest_ic) < 0.01 and backtest_sharpe > 2.0:
                        errors.append(f"Suspected leakage: IC={backtest_ic:.4f} but Sharpe={backtest_sharpe:.2f}")
    
    # Check 6: Turnover calculation
    if len(predictions) > 1:
        # Simple turnover calculation (change in positions)
        position_changes = predictions.diff().abs().sum(axis=1)
        avg_turnover = position_changes.mean()
        
        if avg_turnover > 2.0:  # More than 200% turnover
            warnings.append(f"High turnover: {avg_turnover:.1f} (may indicate overfitting)")
    
    return {
        "n_rebalance_dates": n_rebalance_dates,
        "avg_symbols_per_date": avg_symbols_per_date,
        "total_trades": total_trades,
        "warnings": warnings,
        "errors": errors,
        "is_valid": len(errors) == 0
    }


def calculate_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Information Coefficient (Pearson correlation)."""
    if len(predictions) < 2:
        return 0.0
    return float(pearsonr(predictions, targets)[0])


def calculate_rank_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Rank Information Coefficient (Spearman correlation)."""
    if len(predictions) < 2:
        return 0.0
    return float(spearmanr(predictions, targets)[0])


def calculate_rolling_ic(predictions: pd.Series, targets: pd.Series, 
                        window: int = 252) -> pd.Series:
    """Calculate rolling Information Coefficient."""
    return predictions.rolling(window).corr(targets)


def calculate_rolling_rank_ic(predictions: pd.Series, targets: pd.Series, 
                             window: int = 252) -> pd.Series:
    """Calculate rolling Rank Information Coefficient."""
    def rolling_spearman(x, y):
        if len(x) < 2:
            return np.nan
        return spearmanr(x, y)[0]
    
    return predictions.rolling(window).apply(
        lambda x: rolling_spearman(x, targets.loc[x.index]), raw=False
    )


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    if len(returns) == 0:
        return 0.0
    
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(np.min(drawdown))


def calculate_calmar_ratio(returns: np.ndarray) -> float:
    """Calculate Calmar ratio (annual return / max drawdown)."""
    if len(returns) == 0:
        return 0.0
    
    annual_return = np.mean(returns) * 252
    max_dd = abs(calculate_max_drawdown(returns))
    
    if max_dd == 0:
        return 0.0
    
    return float(annual_return / max_dd)


def calculate_volatility(returns: np.ndarray) -> float:
    """Calculate annualized volatility."""
    if len(returns) == 0:
        return 0.0
    
    return float(np.std(returns) * np.sqrt(252))


def calculate_win_rate(returns: np.ndarray) -> float:
    """Calculate win rate (percentage of positive returns)."""
    if len(returns) == 0:
        return 0.0
    
    return float(np.mean(returns > 0))


def calculate_turnover(weights: np.ndarray) -> float:
    """Calculate portfolio turnover."""
    if len(weights) < 2:
        return 0.0
    
    return float(np.mean(np.abs(np.diff(weights, axis=0)).sum(axis=1)))


def calculate_hit_rate(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate hit rate (percentage of correct directional predictions)."""
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0
    
    pred_direction = np.sign(predictions)
    target_direction = np.sign(targets)
    
    return float(np.mean(pred_direction == target_direction))


def calculate_equity_curve(returns: np.ndarray, initial_value: float = 1.0) -> np.ndarray:
    """Calculate equity curve from returns."""
    if len(returns) == 0:
        return np.array([initial_value])
    
    return initial_value * np.cumprod(1 + returns)


def calculate_rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling Sharpe ratio."""
    return returns.rolling(window).apply(
        lambda x: calculate_sharpe_ratio(x.values), raw=False
    )


def calculate_rolling_volatility(returns: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling volatility."""
    return returns.rolling(window).std() * np.sqrt(252)


def calculate_rolling_max_drawdown(returns: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling maximum drawdown."""
    def rolling_max_dd(x):
        if len(x) == 0:
            return np.nan
        return calculate_max_drawdown(x.values)
    
    return returns.rolling(window).apply(rolling_max_dd, raw=False)


class BacktestMetrics:
    """Comprehensive backtesting metrics calculator."""
    
    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_metrics(self, returns: np.ndarray, 
                             predictions: Optional[np.ndarray] = None,
                             targets: Optional[np.ndarray] = None,
                             weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate all backtesting metrics."""
        metrics = {}
        
        # Basic return metrics
        metrics["total_return"] = float(np.prod(1 + returns) - 1)
        metrics["annual_return"] = float(np.mean(returns) * 252)
        metrics["volatility"] = calculate_volatility(returns)
        metrics["sharpe_ratio"] = calculate_sharpe_ratio(returns, self.risk_free_rate)
        metrics["calmar_ratio"] = calculate_calmar_ratio(returns)
        metrics["max_drawdown"] = calculate_max_drawdown(returns)
        metrics["win_rate"] = calculate_win_rate(returns)
        
        # Prediction metrics
        if predictions is not None and targets is not None:
            metrics["ic"] = calculate_ic(predictions, targets)
            metrics["rank_ic"] = calculate_rank_ic(predictions, targets)
            metrics["hit_rate"] = calculate_hit_rate(predictions, targets)
        
        # Turnover metrics
        if weights is not None:
            metrics["turnover"] = calculate_turnover(weights)
        
        return metrics
    
    def calculate_rolling_metrics(self, returns: pd.Series, 
                                 predictions: Optional[pd.Series] = None,
                                 targets: Optional[pd.Series] = None,
                                 window: int = 252) -> pd.DataFrame:
        """Calculate rolling metrics."""
        metrics_df = pd.DataFrame(index=returns.index)
        
        # Rolling return metrics
        metrics_df["rolling_sharpe"] = calculate_rolling_sharpe(returns, window)
        metrics_df["rolling_volatility"] = calculate_rolling_volatility(returns, window)
        metrics_df["rolling_max_drawdown"] = calculate_rolling_max_drawdown(returns, window)
        
        # Rolling prediction metrics
        if predictions is not None and targets is not None:
            metrics_df["rolling_ic"] = calculate_rolling_ic(predictions, targets, window)
            metrics_df["rolling_rank_ic"] = calculate_rolling_rank_ic(predictions, targets, window)
        
        return metrics_df


def create_performance_summary(returns: np.ndarray, 
                              predictions: Optional[np.ndarray] = None,
                              targets: Optional[np.ndarray] = None,
                              weights: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Create a comprehensive performance summary."""
    metrics_calc = BacktestMetrics()
    metrics = metrics_calc.calculate_all_metrics(returns, predictions, targets, weights)
    
    summary_df = pd.DataFrame([
        {"Metric": "Total Return", "Value": f"{metrics['total_return']:.2%}"},
        {"Metric": "Annual Return", "Value": f"{metrics['annual_return']:.2%}"},
        {"Metric": "Volatility", "Value": f"{metrics['volatility']:.2%}"},
        {"Metric": "Sharpe Ratio", "Value": f"{metrics['sharpe_ratio']:.3f}"},
        {"Metric": "Calmar Ratio", "Value": f"{metrics['calmar_ratio']:.3f}"},
        {"Metric": "Max Drawdown", "Value": f"{metrics['max_drawdown']:.2%}"},
        {"Metric": "Win Rate", "Value": f"{metrics['win_rate']:.2%}"},
    ])
    
    if predictions is not None and targets is not None:
        pred_metrics = pd.DataFrame([
            {"Metric": "IC", "Value": f"{metrics['ic']:.4f}"},
            {"Metric": "Rank-IC", "Value": f"{metrics['rank_ic']:.4f}"},
            {"Metric": "Hit Rate", "Value": f"{metrics['hit_rate']:.2%}"},
        ])
        summary_df = pd.concat([summary_df, pred_metrics], ignore_index=True)
    
    if weights is not None:
        turnover_metrics = pd.DataFrame([
            {"Metric": "Turnover", "Value": f"{metrics['turnover']:.2%}"},
        ])
        summary_df = pd.concat([summary_df, turnover_metrics], ignore_index=True)
    
    return summary_df


def analyze_prediction_quality(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Analyze prediction quality metrics."""
    metrics = {}
    
    # Correlation metrics
    metrics["ic"] = calculate_ic(predictions, targets)
    metrics["rank_ic"] = calculate_rank_ic(predictions, targets)
    
    # Directional accuracy
    metrics["hit_rate"] = calculate_hit_rate(predictions, targets)
    
    # Prediction statistics
    metrics["prediction_std"] = float(np.std(predictions))
    metrics["prediction_mean"] = float(np.mean(predictions))
    metrics["prediction_skew"] = float(pd.Series(predictions).skew())
    metrics["prediction_kurtosis"] = float(pd.Series(predictions).kurtosis())
    
    # Target statistics
    metrics["target_std"] = float(np.std(targets))
    metrics["target_mean"] = float(np.mean(targets))
    
    # Prediction vs target alignment
    metrics["mse"] = float(np.mean((predictions - targets) ** 2))
    metrics["mae"] = float(np.mean(np.abs(predictions - targets)))
    
    return metrics
