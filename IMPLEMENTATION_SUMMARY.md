# ALGOTRADING – Implementation Summary

## Project Overview

This repository contains a production-minded research pipeline for ASX equity 5-day forward excess return prediction. It is aligned to A2/A3 requirements: cloud-runnable, reproducible, temporally validated, and finance-native in evaluation (IC/Rank-IC, dispersion, backtest).

- **Latest submitted run**: results/run_20251009_193928/
- **Universe**: BHP.AX, CBA.AX, CSL.AX, WES.AX, WBC.AX, TLS.AX
- **Sequence length**: 30 days  Horizon: 5 days

## Repository Structure

```
AlgorithmicTrading/
├─ src/algotrading/
│  ├─ core/                  # datatypes, metrics, configs
│  ├─ services/
│  │  ├─ models/             # data loader, splitter, scalers, LSTM, baseline, trainer
│  │  └─ backtester/         # backtest + performance metrics
│  └─ demo.py                # end-to-end demo
├─ ALGOTRADING_DEMO.ipynb    # Colab notebook (cloud runnable)
├─ PROJECT_JOURNAL.md        # Detailed journal (assignment-ready)
├─ IMPLEMENTATION_SUMMARY.md # This file
├─ README.md                 # Overview + how to run + latest results
└─ requirements.txt
```

## Key Components

- **Core types & metrics** (core/): TrainingConfig, IC/Rank-IC, Sharpe, drawdown, dispersion.
- **Data & features** (models/data_loader.py): yfinance download; 18 technical features; forward excess return target; strict feature-order validation; cross-sectional de-meaning (where configured).
- **Temporal splitting** (models/temporal_splitter.py): train/val/test with 21-day embargo; date-sorted; leakage checks.

## Models

- **LSTMRegressor** (models/model_nn.py): 1-layer LSTM, symbol embeddings, linear head.
- **Ridge baseline** (models/baseline_model.py): L2 on flattened sequences.
- **Trainer** (models/trainer.py): train-only scaling, mini-batches grouped by date (cross-sectional), early stop on Val Rank-IC (patience=4), gradient clipping, dispersion checks, optional calibration (skipped if correlation ≤0).
- **Backtester** (backtester/metrics.py): simple top-K (equal-weight) context metrics with quality gates.

## Technical Implementation

### Model (current submitted configuration)

- **Architecture**: 1× LSTM (hidden_size=32) + symbol embeddings (dim=12) + linear head
- **Regularization**: dropout=0.2, weight_decay=1e-5, grad clip=1.0
- **Features (18)**: ret1, ret5, ret21, rsi14, rsi2, rsi50, macd, macd_signal, atr14, realized_vol_5, realized_vol_21, volz, volume_ratio, inv_price, mom_3m, mom_6m, mom_12m, reversal_1d
- **Training**: Adam (lr=3e-4), batch_size=6, max_epochs=15, early_stopping_patience=4 on Val Rank-IC

### Data/Target

- **Target**: 5-day forward excess returns vs benchmark (ASX200 proxy).
- **Scaling**: train-only standardization (features/target). Inverse available for reporting.
- **Integrity**: column order assertion; horizon alignment checks; symbol coverage per split.

## Results (latest run: run_20251009_193928)

### Test set

- **LSTM**: IC 0.0367, Rank-IC 0.0363, MSE 0.000778, RMSE 0.027889, Prediction std 0.005757
- **Ridge**: IC −0.0025, Rank-IC −0.0485, MSE 0.000835, RMSE 0.028903, Prediction std 0.007875
- **Dispersion (per-date cross-section)**: PASS (ratio 0.297 ≥ 0.25)

### Backtest (simple top-K equal-weight, no costs)

- **Total Return** 76.33%, **Sharpe** 3.820, **Max DD** −15.21%, **Win Rate** 60.65%, **Vol** 24.99%

### Interpretation (concise):

- Positive IC/Rank-IC and better RMSE than baseline on the same split.
- Dispersion passes anti-collapse gate → signals not flat.
- Strong backtest is encouraging but small universe + no costs ⇒ not live-ready evidence; use for academic demonstration.

## Assignment Mapping (A2/A3)

- **A2**: Clear I/O; temporal splits with embargo; finance-native metrics; baseline comparison; reproducibility (run dir & config); Colab demo.
- **A3**: Communicates loss vs business objective; interprets IC/Rank-IC vs PnL; explains controls; shows improvement vs prior run.

## Strengths & Current Limitations

**Strengths**: temporal validation; train-only scaling; dispersion checks; date-grouped batching; clean logging & artifacts.

**Limitations**: 6-name universe; only technical features; no costs/turnover/risk budget; modest IC magnitudes (typical at short horizons).

## Future Enhancements (work explored, to be formalized later)

- Variance/rank-preserving calibration at evaluation (gated to avoid inversion).
- Broader universe (≥50 names) and sector/market neutralization.
- Feature expansion (fundamentals, sentiment, macro); model diversity (GRU/TCN/Transformers, boosted trees) and ensembles.
- Portfolio realism (costs, slippage, turnover/risk budgets) and stability checks (rolling IC dashboards, purged CV variants).

## Reproduce This Run

```bash
git checkout be4006e         # commit referenced in README "Recent Results"
python -m src.algotrading.demo
# artifacts: results/run_20251009_193928/
```