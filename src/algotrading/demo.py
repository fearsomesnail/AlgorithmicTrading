"""Demo script for ALGOTRADING system."""

import sys
import os
import argparse

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from algotrading.core.types import TrainingConfig
from algotrading.services.models.data_loader import ASXDataLoader
from algotrading.services.models.trainer import ModelTrainer
from algotrading.services.backtester.metrics import BacktestMetrics
from algotrading.services.results_manager import ResultsManager
from algotrading.utils.determinism import set_seeds
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_demo(quick=False, no_backtest=False, no_calibration=True):
    """Run the complete ALGOTRADING demo."""
    print("=" * 60)
    print("ALGOTRADING - ASX Equity Return Prediction Demo")
    print("=" * 60)
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Configuration
    config = TrainingConfig(
        sequence_length=30,
        horizon_days=10,  # Use 10-day horizon as per improvements
        hidden_size=64,
        num_layers=2,
        learning_rate=1e-3,
        batch_size=256,
        max_epochs=10 if quick else 20,
        early_stopping_patience=5
    )
    
    if quick:
        print("Running in QUICK mode (reduced epochs, data, and universe)")
        config.max_epochs = 10
        config.batch_size = 512  # Larger batches for speed
        # Use 20 stocks for better cross-sectional learning (faster than 50)
        config.universe_size = 20
    
    print(f"Configuration: {config}")
    
    # Load data
    print("\n1. Loading data...")
    loader = ASXDataLoader(config)
    
    # Limit universe size for quick mode
    if quick and hasattr(config, 'universe_size'):
        original_universe = loader.universe.copy()
        loader.universe = loader.universe[:config.universe_size]
        print(f"   Using only {len(loader.universe)} stocks for quick testing")
        print(f"   Selected stocks: {loader.universe}")
    
    training_data = loader.load_training_data(start_date="2018-01-01")
    
    print(f"Loaded {len(training_data.features)} training sequences")
    print(f"Features shape: {training_data.features.shape}")
    print(f"Targets shape: {training_data.targets.shape}")
    
    # Create data splits
    print("\n2. Creating data splits...")
    from algotrading.services.models.temporal_splitter import create_temporal_splits
    train_data, val_data, test_data = create_temporal_splits(training_data, config)
    
    print(f"Train: {len(train_data.features)} samples")
    print(f"Validation: {len(val_data.features)} samples")
    print(f"Test: {len(test_data.features)} samples")
    
    # Train baseline model for comparison
    print("\n3. Training baseline model...")
    from algotrading.services.models.baseline_model import train_baseline_model
    baseline_results = train_baseline_model(train_data, val_data, test_data, config)
    print(f"Baseline training completed!")
    
    # Train LSTM model
    print("\n4. Training LSTM model...")
    print("   This may take several minutes. Progress will be shown below...")
    results_manager = ResultsManager(enable_file_logging=True)
    trainer = ModelTrainer(config, model_family="lstm", results_manager=results_manager)
    
    # Get actual number of symbols from the data
    actual_num_symbols = len(set(train_data.symbols))
    print(f"   Training with {actual_num_symbols} symbols...")
    
    # Update the trainer's symbol mapping to match the actual data
    trainer.symbol_mapping = {symbol: idx for idx, symbol in enumerate(set(train_data.symbols))}
    
    history = trainer.train(train_data, val_data, num_symbols=actual_num_symbols)
    
    print("LSTM training completed!")
    print(f"Final train loss: {history['train_losses'][-1]:.6f}")
    print(f"Final val loss: {history['val_losses'][-1]:.6f}")
    
    # Evaluate models
    print("\n5. Evaluating models...")
    
    # LSTM evaluation
    test_metrics = trainer.evaluate(test_data)
    print(f"\nLSTM Results:")
    print(f"  Test IC: {test_metrics.ic:.4f}")
    print(f"  Test Rank-IC: {test_metrics.rank_ic:.4f}")
    print(f"  Test MSE: {test_metrics.mse:.6f}")
    print(f"  Test RMSE: {test_metrics.rmse:.6f}")
    print(f"  Prediction std: {test_metrics.prediction_std:.6f}")
    
    # Baseline evaluation
    baseline_test_ic, baseline_test_ic_std, baseline_test_ic_count = baseline_results['test_ic']
    baseline_test_rank_ic, baseline_test_rank_ic_std, baseline_test_rank_ic_count = baseline_results['test_rank_ic']
    baseline_test_metrics = baseline_results['test_metrics']
    
    print(f"\nBaseline Results:")
    print(f"  Test IC: {baseline_test_ic:.4f} +/- {baseline_test_ic_std:.4f} (n={baseline_test_ic_count})")
    print(f"  Test Rank-IC: {baseline_test_rank_ic:.4f} +/- {baseline_test_rank_ic_std:.4f} (n={baseline_test_rank_ic_count})")
    print(f"  Test MSE: {baseline_test_metrics.mse:.6f}")
    print(f"  Test RMSE: {baseline_test_metrics.rmse:.6f}")
    print(f"  Prediction std: {baseline_test_metrics.prediction_std:.6f}")
    
    # Compare results
    print(f"\nComparison:")
    print(f"  IC improvement: {test_metrics.ic - baseline_test_ic:.4f}")
    print(f"  Rank-IC improvement: {test_metrics.rank_ic - baseline_test_rank_ic:.4f}")
    print(f"  MSE improvement: {baseline_test_metrics.mse - test_metrics.mse:.6f}")
    print(f"  Prediction std ratio: {test_metrics.prediction_std / baseline_test_metrics.prediction_std:.3f}")
    
    # Store final metrics
    final_metrics = {
        "test_ic": float(test_metrics.ic),
        "test_rank_ic": float(test_metrics.rank_ic),
        "test_mse": float(test_metrics.mse),
        "test_rmse": float(test_metrics.rmse),
        "final_train_loss": float(history['train_losses'][-1]),
        "final_val_loss": float(history['val_losses'][-1])
    }
    results_manager.store_final_metrics(final_metrics)
    
    # Save baseline vs LSTM comparison
    baseline_test_ic, baseline_test_ic_std, baseline_test_ic_count = baseline_results['test_ic']
    baseline_test_rank_ic, baseline_test_rank_ic_std, baseline_test_rank_ic_count = baseline_results['test_rank_ic']
    baseline_test_metrics = baseline_results['test_metrics']
    
    model_comparison = {
        "lstm": {
            "ic": float(test_metrics.ic),
            "rank_ic": float(test_metrics.rank_ic),
            "rmse": float(test_metrics.rmse),
            "pred_std": float(test_metrics.prediction_std),
            "mse": float(test_metrics.mse)
        },
        "baseline": {
            "ic": float(baseline_test_ic),
            "rank_ic": float(baseline_test_rank_ic),
            "rmse": float(baseline_test_metrics.rmse),
            "pred_std": float(baseline_test_metrics.prediction_std),
            "mse": float(baseline_test_metrics.mse)
        },
        "improvement": {
            "ic": float(test_metrics.ic - baseline_test_ic),
            "rank_ic": float(test_metrics.rank_ic - baseline_test_rank_ic),
            "rmse": float(baseline_test_metrics.rmse - test_metrics.rmse),
            "pred_std_ratio": float(test_metrics.prediction_std / baseline_test_metrics.prediction_std)
        }
    }
    results_manager.save_model_comparison(model_comparison)
    
    # Make predictions
    print("\n6. Making predictions...")
    predictions = trainer.predict(test_data)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction stats: mean={predictions.mean():.4f}, std={predictions.std():.4f}")
    
    # Save predictions
    results_manager.save_predictions(
        predictions=predictions,
        targets=test_data.targets,
        dates=test_data.dates,
        symbols=test_data.symbols
    )
    
    # Simple backtest
    print("\n7. Running backtest...")
    
    # Create simple backtest
    def simple_backtest(predictions, targets, symbols, dates, top_k=2):
        """Simple backtest with top-k selection."""
        df = pd.DataFrame({
            'date': dates,
            'symbol': symbols,
            'prediction': predictions,
            'target': targets
        })
        
        daily_returns = []
        for date, group in df.groupby('date'):
            if len(group) < top_k:
                continue
            top_symbols = group.nlargest(top_k, 'prediction')['symbol'].tolist()
            avg_return = group[group['symbol'].isin(top_symbols)]['target'].mean()
            daily_returns.append(avg_return)
        
        return np.array(daily_returns)
    
    # Run backtest
    test_returns = simple_backtest(
        predictions, 
        test_data.targets, 
        test_data.symbols, 
        test_data.dates
    )
    
    # Model quality validation using cross-sectional dispersion
    from algotrading.services.metrics import check_prediction_collapse
    
    is_collapsed, ratio, pred_cs, tgt_cs = check_prediction_collapse(
        predictions, test_data.targets, test_data.dates, threshold=0.25
    )
    
    if is_collapsed:
        print(f"\n[ERROR] MODEL COLLAPSE DETECTED!")
        print(f"   Cross-sectional dispersion ratio: {ratio:.3f}")
        print(f"   Prediction CS std: {pred_cs:.6f}")
        print(f"   Target CS std: {tgt_cs:.6f}")
        print(f"   Ratio: {ratio:.3f} (should be > 0.25)")
        print(f"   Model outputs are nearly flat - not suitable for trading!")
        print(f"\n[WARNING] RUN MARKED AS NON-TRADEABLE")
        print(f"   Consider: reducing regularization, checking data quality, or using simpler baseline")
        
        # Save non-tradeable status
        results_manager.save_non_tradeable_status(ratio, pred_cs, tgt_cs)
        
        # Save training plots and finalize results
        results_manager.save_training_plots(history.get('training_history', []))
        results_dir = results_manager.finalize_results()
        print(f"\nResults saved to: {results_dir}")
        
        # Results summary
        print("\n" + "=" * 60)
        print("DEMO COMPLETED - MODEL NON-TRADEABLE")
        print("=" * 60)
        return
    
    # Calculate backtest metrics (only if not collapsed)
    metrics_calc = BacktestMetrics()
    backtest_metrics = metrics_calc.calculate_all_metrics(test_returns)
    
    # Calculate IC magnitude for validation
    ic_magnitude = abs(test_metrics.ic)
    
    if ic_magnitude < 0.01:
        print(f"\n[WARNING] Very low IC magnitude ({ic_magnitude:.4f})")
        print(f"   Model shows minimal predictive power - results may not be reliable")
    
    # Backtest validation
    if backtest_metrics['sharpe_ratio'] > 2.0 and ic_magnitude < 0.02:
        print(f"[WARNING] High Sharpe ({backtest_metrics['sharpe_ratio']:.2f}) with low IC ({test_metrics.ic:.4f}) - possible leakage!")
    
    print(f"Backtest Results:")
    print(f"  Total Return: {backtest_metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {backtest_metrics['max_drawdown']:.2%}")
    print(f"  Win Rate: {backtest_metrics['win_rate']:.2%}")
    print(f"  Volatility: {backtest_metrics['volatility']:.2%}")
    
    # Store backtest results
    results_manager.store_backtest_results(backtest_metrics)
    
    # Results summary
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print(f"\nKey Results:")
    print(f"  - LSTM IC: {test_metrics.ic:.4f} vs Baseline IC: {baseline_test_ic:.4f}")
    print(f"  - LSTM Rank-IC: {test_metrics.rank_ic:.4f} vs Baseline Rank-IC: {baseline_test_rank_ic:.4f}")
    print(f"  - LSTM RMSE: {test_metrics.rmse:.6f} vs Baseline RMSE: {baseline_test_metrics.rmse:.6f}")
    print(f"  - LSTM Prediction std: {test_metrics.prediction_std:.6f} vs Baseline: {baseline_test_metrics.prediction_std:.6f}")
    return_sign = "positive" if backtest_metrics['total_return'] >= 0 else "negative"
    print(f"  - Backtest shows {return_sign} returns: {backtest_metrics['total_return']:.2%}")
    print(f"  - Risk-adjusted performance: Sharpe {backtest_metrics['sharpe_ratio']:.3f}")
    
    print(f"\nNext Steps:")
    print(f"  - Run the Colab notebook for interactive analysis")
    print(f"  - Read PROJECT_JOURNAL.md for detailed documentation")
    print(f"  - Modify configuration for different experiments")
    print(f"  - Add more symbols to the universe")
    
    # Save training plots and finalize results
    results_manager.save_training_plots(history.get('training_history', []))
    results_dir = results_manager.finalize_results()
    print(f"\nResults saved to: {results_dir}")
    
    return {
        'test_metrics': test_metrics,
        'baseline_metrics': baseline_test_metrics,
        'backtest_metrics': backtest_metrics,
        'history': history,
        'results_dir': results_dir
    }


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="ALGOTRADING Demo")
    parser.add_argument("--quick", action="store_true", help="Run in quick mode (fewer epochs)")
    parser.add_argument("--no-backtest", action="store_true", help="Skip backtesting")
    parser.add_argument("--no-calibration", action="store_true", default=True, help="Skip calibration")
    
    args = parser.parse_args()
    
    try:
        results = run_demo(
            quick=args.quick,
            no_backtest=args.no_backtest,
            no_calibration=args.no_calibration
        )
        print("\nDemo completed successfully!")
        return results
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nDemo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()