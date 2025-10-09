"""Results management and logging utilities."""

import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

class ResultsManager:
    """Manages experiment results, logging, and file output."""
    
    def __init__(self, results_dir: str = "results", enable_file_logging: bool = True):
        self.results_dir = results_dir
        self.enable_file_logging = enable_file_logging
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(results_dir, f"run_{self.run_id}")
        
        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Results storage
        self.results = {
            "run_info": {
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(),
                "results_dir": self.run_dir
            },
            "config": {},
            "data_stats": {},
            "training_history": [],
            "final_metrics": {},
            "backtest_results": {},
            "warnings": []
        }
    
    def _setup_logging(self):
        """Setup file and console logging."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if self.enable_file_logging:
            log_file = os.path.join(self.run_dir, "training.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def log_warning(self, message: str):
        """Log a warning and store it."""
        warning_msg = f"WARNING: {message}"
        logging.warning(warning_msg)
        self.results["warnings"].append({
            "timestamp": datetime.now().isoformat(),
            "message": warning_msg
        })
    
    def log_error(self, message: str):
        """Log an error and store it."""
        error_msg = f"ERROR: {message}"
        logging.error(error_msg)
        self.results["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "message": error_msg
        })
    
    def log_info(self, message: str):
        """Log an info message."""
        logging.info(message)
    
    def store_config(self, config: Dict[str, Any]):
        """Store configuration."""
        self.results["config"] = config
        config_file = os.path.join(self.run_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def store_data_stats(self, stats: Dict[str, Any]):
        """Store data statistics."""
        self.results["data_stats"] = stats
        stats_file = os.path.join(self.run_dir, "data_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  val_ic: float, val_rank_ic: float, train_ic: float = None, 
                  train_rank_ic: float = None):
        """Log epoch results."""
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ic": val_ic,
            "val_rank_ic": val_rank_ic
        }
        
        if train_ic is not None:
            epoch_data["train_ic"] = train_ic
        if train_rank_ic is not None:
            epoch_data["train_rank_ic"] = train_rank_ic
        
        self.results["training_history"].append(epoch_data)
        
        # Log to console
        logging.info(f"Epoch {epoch}/20: Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, Val IC: {val_ic:.4f}, "
                    f"Val Rank-IC: {val_rank_ic:.4f}")
    
    def store_final_metrics(self, metrics: Dict[str, Any]):
        """Store final evaluation metrics."""
        self.results["final_metrics"] = metrics
        metrics_file = os.path.join(self.run_dir, "final_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    def store_backtest_results(self, backtest_results: Dict[str, Any]):
        """Store backtest results."""
        self.results["backtest_results"] = backtest_results
    
    def save_model_comparison(self, comparison: Dict[str, Any]):
        """Save baseline vs LSTM model comparison."""
        comparison_path = os.path.join(self.run_dir, "model_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        self.log_info(f"Model comparison saved to {comparison_path}")
    
    def save_non_tradeable_status(self, ratio: float, pred_cs: float, tgt_cs: float):
        """Save non-tradeable status with reasons."""
        status = {
            "tradeable": False,
            "reason": "Cross-sectional dispersion too low",
            "ratio": float(ratio),
            "pred_cs_std": float(pred_cs),
            "tgt_cs_std": float(tgt_cs),
            "threshold": 0.25
        }
        
        # Save as JSON
        status_path = os.path.join(self.run_dir, "non_tradeable_status.json")
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=2)
        
        # Save as text for easy reading
        text_path = os.path.join(self.run_dir, "backtest_skipped.txt")
        with open(text_path, 'w') as f:
            f.write(f"Skipped due to CS dispersion ratio={ratio:.3f} < 0.25\n")
            f.write(f"Prediction CS std: {pred_cs:.6f}\n")
            f.write(f"Target CS std: {tgt_cs:.6f}\n")
        
        self.log_info(f"Non-tradeable status saved to {status_path}")
        backtest_file = os.path.join(self.run_dir, "backtest_results.json")
        with open(backtest_file, 'w') as f:
            json.dump(status, f, indent=2, default=str)
    
    def save_predictions(self, predictions: np.ndarray, targets: np.ndarray, 
                        dates: np.ndarray, symbols: np.ndarray):
        """Save predictions to CSV."""
        # Ensure all arrays are 1-dimensional
        predictions = np.asarray(predictions).flatten()
        targets = np.asarray(targets).flatten()
        dates = np.asarray(dates).flatten()
        symbols = np.asarray(symbols).flatten()
        
        df = pd.DataFrame({
            'date': dates,
            'symbol': symbols,
            'prediction': predictions,
            'target': targets,
            'error': predictions - targets
        })
        
        predictions_file = os.path.join(self.run_dir, "predictions.csv")
        df.to_csv(predictions_file, index=False)
        
        # Store summary stats
        self.results["prediction_stats"] = {
            "mean_prediction": float(np.mean(predictions)),
            "std_prediction": float(np.std(predictions)),
            "mean_target": float(np.mean(targets)),
            "std_target": float(np.std(targets)),
            "mse": float(np.mean((predictions - targets) ** 2)),
            "rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
            "correlation": float(np.corrcoef(predictions, targets)[0, 1])
        }
    
    def save_training_plots(self, history: Dict):
        """Save training history plots."""
        try:
            import matplotlib.pyplot as plt
            
            # Handle both list of dicts and dict of lists formats
            if isinstance(history, dict) and "train_losses" in history:
                # Format: {"train_losses": [...], "val_losses": [...], ...}
                train_losses = history["train_losses"]
                val_losses = history["val_losses"]
                epochs = list(range(1, len(train_losses) + 1))
                
                # Extract IC data if available
                val_ics = []
                val_rank_ics = []
                train_ics = []
                train_rank_ics = []
                
                if "val_metrics" in history:
                    val_ics = [m.ic for m in history["val_metrics"]]
                    val_rank_ics = [m.rank_ic for m in history["val_metrics"]]
                
                if "train_metrics" in history:
                    train_ics = [m.ic for m in history["train_metrics"]]
                    train_rank_ics = [m.rank_ic for m in history["train_metrics"]]
                    
            else:
                # Format: [{"epoch": 1, "train_loss": 0.5, ...}, ...]
                epochs = [h["epoch"] for h in history]
                train_losses = [h["train_loss"] for h in history]
                val_losses = [h["val_loss"] for h in history]
                val_ics = [h.get("val_ic", 0) for h in history]
                val_rank_ics = [h.get("val_rank_ic", 0) for h in history]
                train_ics = [h.get("train_ic", 0) for h in history]
                train_rank_ics = [h.get("train_rank_ic", 0) for h in history]
            
            plt.figure(figsize=(15, 5))
            
            # Loss plot
            plt.subplot(1, 3, 1)
            plt.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
            plt.plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # IC plot
            plt.subplot(1, 3, 2)
            has_val_data = False
            if val_ics:
                plt.plot(epochs, val_ics, label='Val IC', marker='o', markersize=3)
                has_val_data = True
            if val_rank_ics:
                plt.plot(epochs, val_rank_ics, label='Val Rank-IC', marker='s', markersize=3)
                has_val_data = True
            plt.xlabel('Epoch')
            plt.ylabel('IC')
            plt.title('Validation Information Coefficient')
            if has_val_data:
                plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Train IC plot
            plt.subplot(1, 3, 3)
            has_train_data = False
            if train_ics:
                plt.plot(epochs, train_ics, label='Train IC', marker='o', markersize=3)
                has_train_data = True
            if train_rank_ics:
                plt.plot(epochs, train_rank_ics, label='Train Rank-IC', marker='s', markersize=3)
                has_train_data = True
            plt.xlabel('Epoch')
            plt.ylabel('IC')
            plt.title('Training Information Coefficient')
            if has_train_data:
                plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = os.path.join(self.run_dir, "training_plots.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log_info(f"Training plots saved to {plot_file}")
            
        except ImportError:
            self.log_warning("Matplotlib not available, skipping plots")
        except Exception as e:
            self.log_error(f"Error creating training plots: {e}")
    
    def finalize_results(self):
        """Save complete results bundle."""
        # Save complete results
        results_file = os.path.join(self.run_dir, "results_bundle.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        summary_file = os.path.join(self.run_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"ALGOTRADING Results Summary - Run {self.run_id}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Configuration:\n")
            for key, value in self.results["config"].items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nData Statistics:\n")
            for key, value in self.results["data_stats"].items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nFinal Metrics:\n")
            for key, value in self.results["final_metrics"].items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nBacktest Results:\n")
            if self.results["backtest_results"]:
                for key, value in self.results["backtest_results"].items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  No backtest results (model marked as non-tradeable)\n")
            
            if self.results["warnings"]:
                f.write(f"\nWarnings ({len(self.results['warnings'])}):\n")
                for warning in self.results["warnings"]:
                    f.write(f"  {warning['timestamp']}: {warning['message']}\n")
        
        logging.info(f"Results saved to: {self.run_dir}")
        return self.run_dir
