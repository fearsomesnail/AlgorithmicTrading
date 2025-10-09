# ALGOTRADING – A2/A3 Project Journal (Practical ML System)

**Author**: Harrison White  
**Project**: ASX Equity 5-Day Excess Return Prediction (LSTM vs Ridge)  
**Cloud-Runnable Demo**: ALGOTRADING_DEMO.ipynb (top-level)  
**Submitted Run**: results/run_20251009_193928/ (commit be4006e)

## 0) Executive Summary

I built a reproducible, temporally validated pipeline to forecast 5-day forward excess returns for a small ASX universe using an LSTM with symbol embeddings and a Ridge baseline. The system enforces train-only scaling, embargoed splits, date-grouped mini-batches, and finance-native metrics (IC/Rank-IC + dispersion).

On the latest split, the LSTM achieved IC 0.0367 and Rank-IC 0.0363, with RMSE 0.027889, and passed the dispersion quality gate (0.297 ≥ 0.25). A simple top-K backtest (no costs) reached Sharpe 3.82. These results improve on a previous attempt that had weaker/negative correlation metrics.

## 1) Task Definition (I/O)

**Input (training)**: per-symbol sequences of 30 days × 18 features, with aligned 5-day forward excess return target vs ASX200 proxy.

**Output**: standardized prediction ŷ, plus ranks used for top-K selection in backtest.

**Constraints**: strict temporal order; 21-day embargo between splits; train-only scaling.

**Features (18)**: ret1, ret5, ret21, rsi14, rsi2, rsi50, macd, macd_signal, atr14, realized_vol_5, realized_vol_21, volz, volume_ratio, inv_price, mom_3m, mom_6m, mom_12m, reversal_1d.

## 2) Theory → Code Mapping

**Hypothesis class**: sequence-to-scalar fθ(X_{t−L+1:t}, s) → ŷ_t with symbol embedding.

**Model**: 1-layer LSTM (hidden=32) → linear head.

**Loss**: MSE on standardized targets.

**Optimization/controls**: Adam (lr=3e-4), dropout=0.2, wd=1e-5, early stop on Val Rank-IC (patience=4), grad clip=1.0, optional calibration (skipped when val correlation ≤0).

**Code pointers**: services/models/data_loader.py (features/targets) • services/models/temporal_splitter.py • services/models/model_nn.py • services/models/trainer.py • services/backtester/metrics.py.

## 3) Data & Validation

**Universe**: 6 liquid ASX tickers; Date range: 2018-01-01 → 2025-10-02 (aligned).

**Splits**: train/val/test = 0.8/0.1/0.1 with 21-day embargo; each split covers all symbols.

**Quality checks**: column-order assertion; horizon alignment; per-split feature/target stds; date-grouped batching (compute cross-sectional stats per day).

## 4) Results

### 4.1 Predictive Metrics (Test)

- **LSTM**: IC 0.0367, Rank-IC 0.0363, MSE 0.000778, RMSE 0.027889, pred std 0.005757
- **Ridge**: IC −0.0025, Rank-IC −0.0485, MSE 0.000835, RMSE 0.028903, pred std 0.007875
- **Dispersion (per-date)**: PASS (ratio 0.297 ≥ 0.25)

**Interpretation**: LSTM shows positive, non-trivial correlation at short horizon and improves RMSE vs baseline on the same split. Dispersion passing indicates non-collapsed signals.

### 4.2 Backtest (context only; no costs)

**Top-K equal-weight selection (simple rule)**

- **Total Return** 76.33%, **Sharpe** 3.820, **Max DD** −15.21%, **Win** 60.65%, **Vol** 24.99%

**Caveats**: No trading costs or turnover/risk budgets; small universe → IC variance is high. Treat as academic context, not live-trading evidence.

## 5) Comparison to Prior Attempt (what changed & why it helped)

**Prior attempt (older run)**:
- Narrow 5-feature set; some negative/near-zero IC/Rank-IC; backtest positive but less stable.

**Newest run (submitted)**:
- 18 features (added RSI variants, MACD+signal, ATR, realized vol(5/21), volume ratio, inverse price, momentum(3/6/12m), reversal(1d)).
- Tightened early stopping to Rank-IC (patience=4); kept date-grouped batches; preserved train-only scaling and embargo.

**Outcome**: IC/Rank-IC turned positive; RMSE improved; dispersion PASS; Sharpe notably higher in the simple backtest.

**Insight**: At short horizons, modest positive IC (≈0.03–0.05) is realistic; improving feature richness and validation alignment often matters more than deeper models.

## 6) Controls that protect validity

- **Temporal integrity**: sorted sequences, 21-day embargo, no train-on-future.
- **Scaling discipline**: fit on train only; inverse available for reporting.
- **Finance-native early stop**: Val Rank-IC rather than pure loss.
- **Anti-collapse checks**: prediction std and per-date dispersion gate (≥0.25 ratio).
- **Post-hoc calibration**: gated (skipped if correlation ≤0 to avoid inversion).

## 7) Limitations (honest accounting)

- **Breadth**: 6 symbols → noisy cross-section, Rank-IC variance high.
- **Signals**: still small IC magnitudes (typical).
- **Backtest realism**: no costs/slippage/turnover constraints; equal-weight top-K.
- **Scope**: only technical features; no sector neutralization or factor controls.

## 8) Future Enhancements (work explored; positioned as "future")

- **Breadth & neutralization**: expand to ≥50 ASX names; sector/market beta neutral.
- **Features**: fundamentals, sentiment/news, macro; rolling factor exposures.
- **Models**: GRU/TCN/Transformers; gradient-boosted trees; ensembles.
- **Calibration**: variance/rank-preserving calibration turned on when correlation > 0.
- **Portfolio realism**: costs, slippage; turnover & risk budgets; simple optimizer.
- **Stability**: rolling IC dashboards; purged time-series K-fold; multi-seed windows.

## 9) Reproducibility (assessor path)

```bash
# Python 3.8+; install deps
pip install -r requirements.txt

# Run end-to-end with current defaults
python -m src.algotrading.demo

# Verify artifacts
results/run_20251009_193928/
# Contains: metrics JSON, training plots, predictions, comparison summary
```

**Colab**: open ALGOTRADING_DEMO.ipynb and "Run all".

## 10) Conclusion

The submission demonstrates a clean, leakage-aware pipeline with finance-native evaluation and a documented improvement over a prior run. Results are realistic for short-horizon equity prediction and are presented with appropriate caveats for breadth and costs. The code, logs, and artifacts support marker reproduction and align with A2/A3 expectations.

## Acknowledgements / References

- Hochreiter & Schmidhuber (1997) — LSTM
- Grinold & Kahn (2000) — IC & breadth
- Prado (2018) — Temporal validation & embargo

---

**What I changed relative to your originals (so you don't have to hunt)**

- Updated all metrics to the newest run (IC/Rank-IC/RMSE/Sharpe/dispersion).
- Switched config to 18 features, hidden=32, num_layers=1, dropout=0.2, patience=4.
- Removed older placeholder numbers (e.g., "7 features", "patience=5", generic metrics).
- Added an explicit "Prior attempt vs newest run" section.
- Framed your extra investigations as future enhancements (per your request).