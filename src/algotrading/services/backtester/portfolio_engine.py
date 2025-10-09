"""Portfolio construction and backtesting engine with turnover caps and cost reduction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Portfolio construction configuration."""
    # Portfolio construction
    long_quantile: float = 0.2  # Top 20% for long positions
    short_quantile: float = 0.2  # Bottom 20% for short positions
    min_signal_threshold: float = 0.0  # Minimum signal strength to trade
    rebalance_frequency: int = 5  # Rebalance every N days
    
    # Turnover and cost controls
    max_turnover: float = 0.5  # Maximum 50% turnover per rebalance
    transaction_cost_bps: float = 7.5  # 7.5 bps per trade
    buffer_band: float = 0.1  # 10% buffer to reduce unnecessary trades
    
    # Risk controls
    max_position_size: float = 0.1  # Maximum 10% position size
    min_liquidity: int = 3  # Minimum 3 symbols per side


class PortfolioEngine:
    """Portfolio construction and backtesting engine."""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.current_weights = None
        self.last_rebalance_date = None
        
    def construct_portfolio(self, predictions: np.ndarray, symbols: np.ndarray, 
                          dates: np.ndarray, current_date: pd.Timestamp) -> Dict[str, Any]:
        """Construct portfolio weights based on predictions."""
        
        # Check if we should rebalance
        if (self.last_rebalance_date is not None and 
            (current_date - self.last_rebalance_date).days < self.config.rebalance_frequency):
            return self._apply_buffer_bands(predictions, symbols)
        
        # Calculate target weights
        target_weights = self._calculate_target_weights(predictions, symbols)
        
        # Apply turnover constraints
        if self.current_weights is not None:
            target_weights = self._apply_turnover_constraints(target_weights, symbols)
        
        # Update state
        self.current_weights = target_weights.copy()
        self.last_rebalance_date = current_date
        
        return {
            'weights': target_weights,
            'symbols': symbols,
            'rebalanced': True,
            'turnover': self._calculate_turnover(target_weights, symbols) if self.current_weights is not None else 0.0
        }
    
    def _calculate_target_weights(self, predictions: np.ndarray, symbols: np.ndarray) -> Dict[str, float]:
        """Calculate target portfolio weights based on predictions."""
        weights = {}
        
        # Filter out weak signals
        strong_signals = np.abs(predictions) >= self.config.min_signal_threshold
        if np.sum(strong_signals) < self.config.min_liquidity:
            # Not enough strong signals, return equal weight
            for symbol in symbols:
                weights[symbol] = 0.0
            return weights
        
        # Calculate quantile thresholds
        valid_preds = predictions[strong_signals]
        valid_symbols = symbols[strong_signals]
        
        if len(valid_preds) < self.config.min_liquidity:
            for symbol in symbols:
                weights[symbol] = 0.0
            return weights
        
        long_threshold = np.percentile(valid_preds, (1 - self.config.long_quantile) * 100)
        short_threshold = np.percentile(valid_preds, self.config.short_quantile * 100)
        
        # Initialize weights
        for symbol in symbols:
            weights[symbol] = 0.0
        
        # Long positions (top quantile)
        long_mask = (predictions >= long_threshold) & strong_signals
        long_symbols = symbols[long_mask]
        if len(long_symbols) > 0:
            long_weight = 1.0 / len(long_symbols)  # Equal weight
            for symbol in long_symbols:
                weights[symbol] = min(long_weight, self.config.max_position_size)
        
        # Short positions (bottom quantile)
        short_mask = (predictions <= short_threshold) & strong_signals
        short_symbols = symbols[short_mask]
        if len(short_symbols) > 0:
            short_weight = -1.0 / len(short_symbols)  # Equal weight, negative
            for symbol in short_symbols:
                weights[symbol] = max(short_weight, -self.config.max_position_size)
        
        return weights
    
    def _apply_turnover_constraints(self, target_weights: Dict[str, float], 
                                  symbols: np.ndarray) -> Dict[str, float]:
        """Apply turnover constraints to target weights."""
        if self.current_weights is None:
            return target_weights
        
        # Calculate current turnover
        current_turnover = self._calculate_turnover(target_weights, symbols)
        
        if current_turnover <= self.config.max_turnover:
            return target_weights
        
        # Scale down positions to meet turnover constraint
        scale_factor = self.config.max_turnover / current_turnover
        
        adjusted_weights = {}
        for symbol in symbols:
            current_w = self.current_weights.get(symbol, 0.0)
            target_w = target_weights.get(symbol, 0.0)
            
            # Scale the change
            change = target_w - current_w
            adjusted_change = change * scale_factor
            adjusted_weights[symbol] = current_w + adjusted_change
        
        return adjusted_weights
    
    def _apply_buffer_bands(self, predictions: np.ndarray, symbols: np.ndarray) -> Dict[str, Any]:
        """Apply buffer bands to reduce unnecessary trades."""
        if self.current_weights is None:
            return self.construct_portfolio(predictions, symbols, None, None)
        
        # Only rebalance if signal changes significantly
        target_weights = self._calculate_target_weights(predictions, symbols)
        
        # Check if any position changes exceed buffer band
        significant_changes = False
        for symbol in symbols:
            current_w = self.current_weights.get(symbol, 0.0)
            target_w = target_weights.get(symbol, 0.0)
            
            if abs(target_w - current_w) > self.config.buffer_band:
                significant_changes = True
                break
        
        if significant_changes:
            return self.construct_portfolio(predictions, symbols, None, None)
        else:
            return {
                'weights': self.current_weights.copy(),
                'symbols': symbols,
                'rebalanced': False,
                'turnover': 0.0
            }
    
    def _calculate_turnover(self, target_weights: Dict[str, float], symbols: np.ndarray) -> float:
        """Calculate portfolio turnover."""
        if self.current_weights is None:
            return 0.0
        
        total_turnover = 0.0
        for symbol in symbols:
            current_w = self.current_weights.get(symbol, 0.0)
            target_w = target_weights.get(symbol, 0.0)
            total_turnover += abs(target_w - current_w)
        
        return total_turnover / 2.0  # Divide by 2 for one-way turnover


class BacktestEngine:
    """Main backtesting engine with cost reduction and turnover controls."""
    
    def __init__(self, portfolio_config: PortfolioConfig):
        self.portfolio_engine = PortfolioEngine(portfolio_config)
        self.config = portfolio_config
        
    def run_backtest(self, predictions: np.ndarray, targets: np.ndarray, 
                    symbols: np.ndarray, dates: np.ndarray) -> Dict[str, Any]:
        """Run backtest with turnover caps and cost reduction."""
        
        # Group by date
        df = pd.DataFrame({
            'date': dates,
            'symbol': symbols,
            'prediction': predictions,
            'target': targets
        })
        
        portfolio_returns = []
        turnover_history = []
        rebalance_dates = []
        
        for date, group in df.groupby('date'):
            date_predictions = group['prediction'].values
            date_symbols = group['symbol'].values
            date_targets = group['target'].values
            
            # Construct portfolio
            portfolio_result = self.portfolio_engine.construct_portfolio(
                date_predictions, date_symbols, dates, pd.Timestamp(date)
            )
            
            weights = portfolio_result['weights']
            rebalanced = portfolio_result['rebalanced']
            turnover = portfolio_result['turnover']
            
            # Calculate portfolio return
            portfolio_return = 0.0
            for symbol in date_symbols:
                weight = weights.get(symbol, 0.0)
                symbol_return = date_targets[date_symbols == symbol][0] if len(date_targets[date_symbols == symbol]) > 0 else 0.0
                portfolio_return += weight * symbol_return
            
            # Apply transaction costs
            if rebalanced and turnover > 0:
                cost_drag = turnover * self.config.transaction_cost_bps / 10000
                portfolio_return -= cost_drag
            
            portfolio_returns.append(portfolio_return)
            turnover_history.append(turnover)
            if rebalanced:
                rebalance_dates.append(date)
        
        # Calculate metrics
        portfolio_returns = np.array(portfolio_returns)
        
        metrics = {
            'total_return': float(np.prod(1 + portfolio_returns) - 1),
            'annual_return': float(np.mean(portfolio_returns) * 252),
            'volatility': float(np.std(portfolio_returns) * np.sqrt(252)),
            'sharpe_ratio': float(np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)) if np.std(portfolio_returns) > 0 else 0.0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'win_rate': float(np.mean(portfolio_returns > 0)),
            'avg_turnover': float(np.mean(turnover_history)),
            'rebalance_frequency': len(rebalance_dates) / len(df['date'].unique()) if len(df['date'].unique()) > 0 else 0.0,
            'total_trades': len(rebalance_dates)
        }
        
        return {
            'metrics': metrics,
            'returns': portfolio_returns,
            'turnover_history': turnover_history,
            'rebalance_dates': rebalance_dates
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))
