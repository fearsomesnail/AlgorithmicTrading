# ALGOTRADING - ASX Equity Return Prediction System

A comprehensive machine learning system for predicting 5-day forward excess returns on ASX equities using LSTM neural networks with temporal validation and rigorous evaluation.

## Project Overview

This project implements a complete algorithmic trading research pipeline that:

- **Downloads** historical ASX equity data using `yfinance`
- **Engineers** technical features (returns, RSI, volatility)
- **Builds** temporal sequences for LSTM training
- **Validates** using proper temporal splits with embargo periods
- **Trains** both LSTM and Ridge baseline models
- **Evaluates** using Information Coefficient (IC) and Rank-IC metrics
- **Backtests** trading strategies with risk controls
- **Provides** comprehensive logging and result management

## Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ALGOTRADING_A2_CLEAN

# Install dependencies
pip install -r requirements.txt
```

### Run the Demo

```bash
# Run the complete end-to-end demo
python -m src.algotrading.demo
```

This will:
1. Download ASX data for 6 major stocks (BHP, CBA, CSL, WES, WBC, TLS)
2. Engineer technical features
3. Train both Ridge baseline and LSTM models
4. Compare performance metrics
5. Run backtesting simulation
6. Generate comprehensive reports

## Recent Results

### Model Performance (Latest Run)

| Metric | LSTM | Ridge Baseline | Improvement |
|--------|------|----------------|-------------|
| **Test IC** | 0.0648 | 0.0199 | +0.0449 |
| **Test Rank-IC** | 0.0649 | 0.0088 | +0.0562 |
| **Test RMSE** | 0.028417 | 0.028485 | +0.000004 |
| **Prediction Std** | 0.005399 | 0.003033 | +1.78x |

### Key Insights

- **LSTM significantly outperforms baseline** on ranking metrics
- **Positive IC and Rank-IC** indicate meaningful predictive power
- **Model quality validation** correctly identifies prediction variance issues
- **Temporal validation** prevents data leakage with proper embargo periods

## System Architecture

### Core Components

```
src/algotrading/
├── core/
│   └── types.py              # Data structures and metrics
├── services/
│   ├── models/
│   │   ├── data_loader.py    # Data loading and feature engineering
│   │   ├── trainer.py        # LSTM model training
│   │   ├── baseline_model.py # Ridge regression baseline
│   │   ├── model_nn.py       # LSTM architecture
│   │   ├── scaler_manager.py # Feature/target scaling
│   │   └── temporal_splitter.py # Temporal data splitting
│   ├── backtester/
│   │   └── metrics.py        # Backtesting and performance metrics
│   └── results_manager.py    # Experiment logging and result storage
└── demo.py                   # Main demo script
```

### Data Pipeline

1. **Data Loading**: Downloads ASX equity data using `yfinance`
2. **Feature Engineering**: Calculates technical indicators (returns, RSI, volatility)
3. **Target Creation**: Computes 5-day forward excess returns
4. **Sequence Building**: Creates 30-day sequences for LSTM training
5. **Temporal Splitting**: Splits data with embargo periods to prevent leakage
6. **Scaling**: Standardizes features and targets using training data only

### Model Architecture

#### LSTM Model
- **Input**: 30-day sequences of 5 technical features
- **Architecture**: 2-layer LSTM with symbol embeddings
- **Output**: Single value prediction for 5-day forward return
- **Regularization**: Dropout (0.1), Weight decay (5e-5)
- **Optimization**: Adam with learning rate scheduling

#### Ridge Baseline
- **Input**: Flattened 30-day sequences (150 features)
- **Architecture**: Ridge regression with L2 regularization
- **Purpose**: Provides baseline comparison for LSTM performance

## 🔬 Key Features

### 1. Temporal Validation
- **Proper time-based splits** (Train/Val/Test)
- **Embargo periods** (21 trading days) to prevent leakage
- **No future data** used in training or validation

### 2. Feature Engineering
- **Technical indicators**: 1-day, 5-day, 21-day returns
- **RSI (14-day)**: Relative Strength Index
- **Volatility**: Z-score normalized rolling volatility
- **Feature validation**: Ensures correct column ordering and ranges

### 3. Model Quality Controls
- **Prediction variance monitoring**: Flags models with collapsed predictions
- **IC/Rank-IC validation**: Ensures meaningful predictive power
- **Early stopping**: Prevents overfitting using validation Rank-IC
- **Dispersion penalty**: Encourages prediction variance during training

### 4. Comprehensive Evaluation
- **Daily cross-sectional IC**: Computes IC across symbols per date
- **Rank-IC**: Spearman correlation for ranking quality
- **Baseline comparison**: Ridge regression for performance context
- **Backtesting**: Simple top-K selection strategy simulation

### 5. Experiment Management
- **Timestamped runs**: Each experiment gets unique directory
- **Comprehensive logging**: Training progress, metrics, warnings
- **Result storage**: Config, metrics, predictions, plots saved automatically
- **Reproducibility**: All parameters and random seeds logged

## 📈 Usage Examples

### Basic Training

```python
from algotrading.core.types import TrainingConfig
from algotrading.services.models.data_loader import ASXDataLoader
from algotrading.services.models.trainer import ModelTrainer

# Configure training
config = TrainingConfig(
    sequence_length=30,
    horizon_days=5,
    hidden_size=64,
    dropout=0.1,
    learning_rate=1e-3,
    max_epochs=30
)

# Load data
loader = ASXDataLoader(config)
data = loader.load_training_data(start_date="2018-01-01")

# Train model
trainer = ModelTrainer(config)
results = trainer.train(train_data, val_data)
```

### Custom Configuration

```python
# Modify configuration for different experiments
config = TrainingConfig(
    sequence_length=60,        # Longer sequences
    hidden_size=128,          # Larger model
    dropout=0.2,              # More regularization
    weight_decay=1e-4,        # Higher weight decay
    max_epochs=50,            # More training
    embargo_days=30           # Longer embargo
)
```

### Baseline Comparison

```python
from algotrading.services.models.baseline_model import train_baseline_model

# Train Ridge baseline
baseline_results = train_baseline_model(train_data, val_data, test_data, config)

# Compare with LSTM
print(f"LSTM IC: {lstm_ic:.4f}")
print(f"Baseline IC: {baseline_ic:.4f}")
print(f"Improvement: {lstm_ic - baseline_ic:.4f}")
```

## 📊 Understanding the Results

### Metrics Explained

- **IC (Information Coefficient)**: Pearson correlation between predictions and actual returns
- **Rank-IC**: Spearman correlation measuring ranking quality
- **MSE/RMSE**: Mean squared error for regression quality
- **Prediction Std**: Standard deviation of predictions (higher = more dispersion)
- **Sharpe Ratio**: Risk-adjusted return measure

### Model Quality Flags

The system automatically flags potential issues:

- **PREDICTION COLLAPSE**: When prediction std < 0.25 × target std
- **LOW IC**: When IC magnitude < 0.01
- **HIGH SHARPE + LOW IC**: Possible data leakage
- **FEATURE MISMATCH**: When feature columns don't match configuration

### Interpreting Results

**Good Model**:
- IC > 0.05, Rank-IC > 0.05
- Prediction std > 0.25 × target std
- LSTM outperforms baseline
- No quality flags triggered

**Problematic Model**:
- IC near zero or negative
- Very low prediction variance
- Quality flags triggered
- LSTM performs worse than baseline

## 🔧 Configuration Options

### TrainingConfig Parameters

```python
@dataclass
class TrainingConfig:
    # Model architecture
    sequence_length: int = 30      # Length of input sequences
    hidden_size: int = 64          # LSTM hidden units
    num_layers: int = 2            # Number of LSTM layers
    embedding_dim: int = 12        # Symbol embedding dimension
    dropout: float = 0.1           # Dropout rate
    
    # Training parameters
    learning_rate: float = 1e-3    # Learning rate
    batch_size: int = 256          # Batch size
    max_epochs: int = 30           # Maximum epochs
    early_stopping_patience: int = 8  # Early stopping patience
    weight_decay: float = 5e-5     # L2 regularization
    
    # Data parameters
    horizon_days: int = 5          # Prediction horizon
    train_ratio: float = 0.8       # Training set ratio
    val_ratio: float = 0.1         # Validation set ratio
    test_ratio: float = 0.1        # Test set ratio
    embargo_days: int = 21         # Days between splits
    
    # Features
    features: List[str] = [        # Technical features to use
        "ret1", "ret5", "ret21", "rsi14", "volz"
    ]
```

## 📁 Project Structure

```
ALGOTRADING_A2_CLEAN/
├── src/algotrading/              # Main source code
├── results/                      # Experiment results
│   └── run_YYYYMMDD_HHMMSS/     # Timestamped run directories
├── ALGOTRADING_DEMO.ipynb       # Colab notebook
├── PROJECT_JOURNAL.md            # Detailed project documentation
├── IMPLEMENTATION_SUMMARY.md     # Technical implementation details
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🎓 Academic Context

This project was developed for **Machine Learning Assignment 3 (A2/A3)** and demonstrates:

### A2 Requirements Met
- ✅ **Self-contained demo**: Complete Colab notebook with one-click execution
- ✅ **Project Journal**: Comprehensive PDF-ready documentation
- ✅ **Task Definition**: Clear I/O specifications and data schemas
- ✅ **Model/Algorithm Mapping**: Theory-to-code connections documented
- ✅ **Evaluation & Results**: Real metrics with sensitivity analysis
- ✅ **Validation Scheme**: Proper temporal splits with embargo
- ✅ **AI Tool Documentation**: Transparent about tool usage

### Key Learning Outcomes
- **Temporal validation** in financial ML
- **Feature engineering** for equity prediction
- **Model evaluation** using IC and Rank-IC
- **Baseline comparison** methodology
- **Prediction quality** assessment
- **Reproducible research** practices

## 🚨 Known Limitations

1. **Prediction Variance**: Models tend to predict near-constant values
2. **Limited Universe**: Only 6 ASX stocks (small cross-sectional diversity)
3. **Simple Features**: Basic technical indicators only
4. **No Market Regime**: No adaptation to different market conditions
5. **Transaction Costs**: Backtesting doesn't include realistic trading costs

## 🔮 Future Improvements

1. **More Symbols**: Expand to 50+ ASX stocks
2. **Advanced Features**: Add fundamental, sentiment, macro features
3. **Ensemble Methods**: Combine multiple models
4. **Online Learning**: Adapt to new data continuously
5. **Risk Management**: Add position sizing and risk controls
6. **Alternative Architectures**: Try Transformers, GRUs, or hybrid models

## 📚 References

- **LSTM for Financial Prediction**: Hochreiter & Schmidhuber (1997)
- **Information Coefficient**: Grinold & Kahn (2000)
- **Temporal Validation**: Prado (2018) "Advances in Financial Machine Learning"
- **Feature Engineering**: Guido (2017) "Python for Finance"

## 🤝 Contributing

This is an academic project, but suggestions for improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please cite appropriately if used in research.

## 📞 Contact

For questions about this project, please refer to the `PROJECT_JOURNAL.md` for detailed documentation or create an issue in the repository.

---

**Last Updated**: October 2025  
**Version**: 1.0.0  
**Python**: 3.8+  
**Status**: Ready for A2/A3 Submission ✅