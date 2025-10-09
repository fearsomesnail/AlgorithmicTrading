"""Diagnostic tools for model validation and sanity checks."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def analyze_ic_by_month(predictions: np.ndarray, targets: np.ndarray, 
                       dates: np.ndarray) -> Dict[str, Any]:
    """Analyze IC by month to detect regime dependence."""
    
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'prediction': predictions,
        'target': targets
    })
    
    # Add month column
    df['month'] = df['date'].dt.to_period('M')
    
    monthly_ics = []
    monthly_rank_ics = []
    monthly_counts = []
    months = []
    
    for month, group in df.groupby('month'):
        if len(group) < 3:  # Need at least 3 samples
            continue
            
        preds = group['prediction'].values
        targs = group['target'].values
        
        # Calculate IC and Rank-IC
        ic = np.corrcoef(preds, targs)[0, 1] if len(preds) > 1 else np.nan
        rank_ic = np.corrcoef(np.argsort(np.argsort(preds)), np.argsort(np.argsort(targs)))[0, 1] if len(preds) > 1 else np.nan
        
        if not np.isnan(ic) and not np.isnan(rank_ic):
            monthly_ics.append(ic)
            monthly_rank_ics.append(rank_ic)
            monthly_counts.append(len(group))
            months.append(str(month))
    
    return {
        'months': months,
        'ics': monthly_ics,
        'rank_ics': monthly_rank_ics,
        'counts': monthly_counts,
        'ic_mean': np.mean(monthly_ics) if monthly_ics else np.nan,
        'ic_std': np.std(monthly_ics) if monthly_ics else np.nan,
        'rank_ic_mean': np.mean(monthly_rank_ics) if monthly_rank_ics else np.nan,
        'rank_ic_std': np.std(monthly_rank_ics) if monthly_rank_ics else np.nan
    }


def analyze_ic_by_symbol(predictions: np.ndarray, targets: np.ndarray, 
                        symbols: np.ndarray) -> Dict[str, Any]:
    """Analyze IC by symbol to detect name dependence."""
    
    df = pd.DataFrame({
        'symbol': symbols,
        'prediction': predictions,
        'target': targets
    })
    
    symbol_ics = []
    symbol_rank_ics = []
    symbol_counts = []
    symbol_names = []
    
    for symbol, group in df.groupby('symbol'):
        if len(group) < 3:  # Need at least 3 samples
            continue
            
        preds = group['prediction'].values
        targs = group['target'].values
        
        # Calculate IC and Rank-IC
        ic = np.corrcoef(preds, targs)[0, 1] if len(preds) > 1 else np.nan
        rank_ic = np.corrcoef(np.argsort(np.argsort(preds)), np.argsort(np.argsort(targs)))[0, 1] if len(preds) > 1 else np.nan
        
        if not np.isnan(ic) and not np.isnan(rank_ic):
            symbol_ics.append(ic)
            symbol_rank_ics.append(rank_ic)
            symbol_counts.append(len(group))
            symbol_names.append(symbol)
    
    return {
        'symbols': symbol_names,
        'ics': symbol_ics,
        'rank_ics': symbol_rank_ics,
        'counts': symbol_counts,
        'ic_mean': np.mean(symbol_ics) if symbol_ics else np.nan,
        'ic_std': np.std(symbol_ics) if symbol_ics else np.nan,
        'rank_ic_mean': np.mean(symbol_rank_ics) if symbol_rank_ics else np.nan,
        'rank_ic_std': np.std(symbol_rank_ics) if symbol_rank_ics else np.nan
    }


def analyze_prediction_dispersion(predictions: np.ndarray, targets: np.ndarray, 
                                 dates: np.ndarray) -> Dict[str, Any]:
    """Analyze prediction dispersion over time."""
    
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'prediction': predictions,
        'target': targets
    })
    
    daily_stats = []
    dates_list = []
    
    for date, group in df.groupby('date'):
        if len(group) < 3:  # Need at least 3 samples
            continue
            
        preds = group['prediction'].values
        targs = group['target'].values
        
        daily_stats.append({
            'pred_std': np.std(preds),
            'target_std': np.std(targs),
            'pred_mean': np.mean(preds),
            'target_mean': np.mean(targs),
            'count': len(group)
        })
        dates_list.append(date)
    
    daily_stats = pd.DataFrame(daily_stats)
    daily_stats['date'] = dates_list
    
    # Calculate dispersion ratio
    daily_stats['dispersion_ratio'] = daily_stats['pred_std'] / daily_stats['target_std']
    
    return {
        'daily_stats': daily_stats,
        'avg_dispersion_ratio': daily_stats['dispersion_ratio'].mean(),
        'dispersion_ratio_std': daily_stats['dispersion_ratio'].std(),
        'pred_std_mean': daily_stats['pred_std'].mean(),
        'target_std_mean': daily_stats['target_std'].mean()
    }


def feature_importance_permutation(model, X_val: np.ndarray, y_val: np.ndarray, 
                                 feature_names: List[str], n_permutations: int = 10) -> Dict[str, float]:
    """Calculate feature importance using permutation method."""
    
    # Get baseline score
    baseline_score = model.score(X_val, y_val) if hasattr(model, 'score') else 0.0
    
    feature_importance = {}
    
    for i, feature_name in enumerate(feature_names):
        scores = []
        
        for _ in range(n_permutations):
            # Create a copy of validation data
            X_perm = X_val.copy()
            
            # Permute the feature
            np.random.shuffle(X_perm[:, i])
            
            # Calculate score with permuted feature
            if hasattr(model, 'score'):
                perm_score = model.score(X_perm, y_val)
                scores.append(baseline_score - perm_score)
            else:
                # For models without score method, use prediction correlation
                pred_perm = model.predict(X_perm)
                perm_corr = np.corrcoef(pred_perm, y_val)[0, 1] if len(pred_perm) > 1 else 0.0
                pred_orig = model.predict(X_val)
                orig_corr = np.corrcoef(pred_orig, y_val)[0, 1] if len(pred_orig) > 1 else 0.0
                scores.append(orig_corr - perm_corr)
        
        feature_importance[feature_name] = np.mean(scores)
    
    return feature_importance


def plot_diagnostic_charts(predictions: np.ndarray, targets: np.ndarray, 
                          dates: np.ndarray, symbols: np.ndarray, 
                          save_path: Optional[str] = None) -> None:
    """Create diagnostic plots for model validation."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Diagnostic Charts', fontsize=16)
    
    # 1. Prediction vs Target scatter
    axes[0, 0].scatter(predictions, targets, alpha=0.6, s=20)
    axes[0, 0].plot([predictions.min(), predictions.max()], 
                    [predictions.min(), predictions.max()], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('Predictions')
    axes[0, 0].set_ylabel('Targets')
    axes[0, 0].set_title('Predictions vs Targets')
    
    # Add correlation text
    corr = np.corrcoef(predictions, targets)[0, 1]
    axes[0, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=axes[0, 0].transAxes, verticalalignment='top')
    
    # 2. Prediction dispersion over time
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'prediction': predictions,
        'target': targets
    })
    
    daily_pred_std = df.groupby('date')['prediction'].std()
    daily_target_std = df.groupby('date')['target'].std()
    
    axes[0, 1].plot(daily_pred_std.index, daily_pred_std.values, label='Prediction Std', alpha=0.7)
    axes[0, 1].plot(daily_target_std.index, daily_target_std.values, label='Target Std', alpha=0.7)
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].set_title('Prediction Dispersion Over Time')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. IC by month
    monthly_ic_data = analyze_ic_by_month(predictions, targets, dates)
    if monthly_ic_data['months']:
        axes[1, 0].bar(range(len(monthly_ic_data['months'])), monthly_ic_data['ics'], alpha=0.7)
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('IC')
        axes[1, 0].set_title('IC by Month')
        axes[1, 0].set_xticks(range(len(monthly_ic_data['months'])))
        axes[1, 0].set_xticklabels(monthly_ic_data['months'], rotation=45)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 4. IC by symbol
    symbol_ic_data = analyze_ic_by_symbol(predictions, targets, symbols)
    if symbol_ic_data['symbols']:
        axes[1, 1].bar(range(len(symbol_ic_data['symbols'])), symbol_ic_data['ics'], alpha=0.7)
        axes[1, 1].set_xlabel('Symbol')
        axes[1, 1].set_ylabel('IC')
        axes[1, 1].set_title('IC by Symbol')
        axes[1, 1].set_xticks(range(len(symbol_ic_data['symbols'])))
        axes[1, 1].set_xticklabels(symbol_ic_data['symbols'], rotation=45)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Diagnostic charts saved to {save_path}")
    
    plt.show()


def run_comprehensive_diagnostics(predictions: np.ndarray, targets: np.ndarray, 
                                dates: np.ndarray, symbols: np.ndarray,
                                model=None, X_val: np.ndarray = None, 
                                feature_names: List[str] = None) -> Dict[str, Any]:
    """Run comprehensive diagnostic analysis."""
    
    logger.info("Running comprehensive diagnostics...")
    
    results = {}
    
    # 1. Monthly IC analysis
    monthly_analysis = analyze_ic_by_month(predictions, targets, dates)
    results['monthly_ic'] = monthly_analysis
    logger.info(f"Monthly IC: {monthly_analysis['ic_mean']:.4f} +/- {monthly_analysis['ic_std']:.4f}")
    
    # 2. Symbol IC analysis
    symbol_analysis = analyze_ic_by_symbol(predictions, targets, symbols)
    results['symbol_ic'] = symbol_analysis
    logger.info(f"Symbol IC: {symbol_analysis['ic_mean']:.4f} +/- {symbol_analysis['ic_std']:.4f}")
    
    # 3. Dispersion analysis
    dispersion_analysis = analyze_prediction_dispersion(predictions, targets, dates)
    results['dispersion'] = dispersion_analysis
    logger.info(f"Dispersion ratio: {dispersion_analysis['avg_dispersion_ratio']:.3f}")
    
    # 4. Feature importance (if model and features provided)
    if model is not None and X_val is not None and feature_names is not None:
        feature_importance = feature_importance_permutation(model, X_val, targets, feature_names)
        results['feature_importance'] = feature_importance
        logger.info("Feature importance calculated")
    
    # 5. Create diagnostic plots
    try:
        plot_diagnostic_charts(predictions, targets, dates, symbols)
        results['plots_created'] = True
    except Exception as e:
        logger.warning(f"Failed to create diagnostic plots: {e}")
        results['plots_created'] = False
    
    return results
