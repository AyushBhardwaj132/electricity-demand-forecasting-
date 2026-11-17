# TerraWatt Methodological Evaluation Report

## 1. Executive Summary
This report presents the verified empirical backtesting results for **TerraWatt**, a production-grade hierarchical electricity demand forecasting platform evaluated on the Indian power grid dataset (POSOCO / GRID-INDIA).

---

## 2. Methodology & Backtest Classification
- **Evaluation Methodology**: **Expanding-Window Rolling-Origin Evaluation** across 6 historical forecast origins:
  - `2025-01-01`
  - `2025-02-01`
  - `2025-03-01`
  - `2025-04-01`
  - `2025-05-01`
  - `2025-06-01`
- **Dataset**: POSOCO / GRID-INDIA Daily Electricity Consumption Data (`2013-01-02` to Present).
- **Training Period**: Expanding window strictly prior to each origin $T$ (data $< T$).
- **Forecast Horizons**: 7-day, 14-day, and 30-day ahead out-of-sample predictions from each origin.
- **Hierarchy Nodes**: 51 total nodes across 3 geographic levels:
  - Level 0: `India` (National Aggregate)
  - Level 1: `India/NR`, `India/WR`, `India/SR`, `India/ER`, `India/NER` (5 Regions)
  - Level 2: 45 State/Bulk Consumer & Synthetic Residual (`Other_<Region>`) Nodes

---

## 3. Data Leakage & Test Isolation Audit
- **Calendar & Holiday Features**: Day of week, month, quarter, day of year, year, weekend, and India national holidays (`holidays`).
- **Lag & Rolling Features**: Lags ($t-1, t-7, t-30, t-365$) and rolling stats ($7d, 30d$).
- **Leakage Audit**: Verified that all lag and rolling features apply `.shift(1)` prior to window calculations. Observation $T$ is **never** present in feature vectors for day $T$. Verified via `tests/test_features.py`.
- **Test Isolation**: Residual covariance matrix $W$ estimation and model fitting strictly use in-sample data prior to each origin $T$.

---

## 4. Empirical Benchmark Results (National Level: `India`)

Pooled metrics across 6 rolling forecast origins (`2025-01-01` to `2025-06-01`):

| Forecast Horizon | Method | National MAE (MU) | National RMSE (MU) | WMAPE (%) | Coherence Error (MU) | Coherent? |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **7 Days** | Base LightGBM | 77.86 | 90.91 | 1.66% | 104.22 | ❌ |
| **7 Days** | Bottom-Up | 77.82 | 87.98 | 1.66% | 0.00 | ✅ |
| **7 Days** | Top-Down | 77.86 | 90.91 | 1.66% | 0.00 | ✅ |
| **7 Days** | **MinT-Shrink** | **76.72** | **86.12** | **1.63%** | **0.00** | **✅ (Best)** |
| **14 Days** | Base LightGBM | 91.78 | 113.31 | 1.90% | 173.61 | ❌ |
| **14 Days** | Bottom-Up | 80.67 | 100.63 | 1.68% | 0.00 | ✅ |
| **14 Days** | Top-Down | 91.78 | 113.31 | 1.90% | 0.00 | ✅ |
| **14 Days** | **MinT-Shrink** | **80.36** | **100.29** | **1.67%** | **0.00** | **✅ (Best)** |
| **30 Days** | Base LightGBM | 79.46 | 102.74 | 1.65% | 173.61 | ❌ |
| **30 Days** | Bottom-Up | 76.60 | 96.81 | 1.59% | 0.00 | ✅ |
| **30 Days** | Top-Down | 79.46 | 102.74 | 1.65% | 0.00 | ✅ |
| **30 Days** | **MinT-Shrink** | **74.53** | **94.97** | **1.55%** | **0.00** | **✅ (Best)** |

---

## 5. Statistical & Methodological Insights
1. **Hierarchy Coherence**: Both Bottom-Up and MinT-Shrink achieve exact mathematical sum consistency ($\text{Coherence Error} < 10^{-11} \approx 0.0000000000$), while Base LightGBM suffers large aggregation discrepancies ($104.22$ to $173.61$ MU).
2. **MinT-Shrink Dominance Across Horizons**: Across the 6-origin expanding backtest, **MinT-Shrink consistently achieves the lowest National MAE and RMSE** across all evaluation horizons (7-day MAE: 76.72 MU; 14-day MAE: 80.36 MU; 30-day MAE: 74.53 MU), outperforming both unreconciled Base LightGBM and Bottom-Up aggregation.
3. **Top-Down Equivalence**: Top-Down (forecast proportions) scales the Base National forecast downward across the hierarchy. Summing these proportions back to the National root produces a total mathematically identical to Base LightGBM, explaining identical National MAE (77.86, 91.78, 79.46 MU).

---

## 6. Milestone 2: Classical Time Series, Prophet & AutoTS Benchmark

### 6.1 Stationarity & Parameter Justification
- **Raw Series Augmented Dickey-Fuller (ADF)**: Statistic = **$-1.5788$** ($p = 0.4942$), non-stationary.
- **First-Difference Series ADF ($\Delta y_t$)**: Statistic = **$-14.8693$** ($p = 1.67 \times 10^{-27}$), stationary ($d=1$).
- **Parameters**: AR order $p \in \{2, 7\}$, differencing $d=1$, MA order $q \in \{2, 7\}$, seasonal period $s=7$ (weekly operational grid cycle).

### 6.2 30-Day Out-of-Sample Benchmark Results

Evaluated across the 30-day forecast horizon (`2026-08-27` to `2026-09-25`) against actual India aggregate demand:

| Model | RMSE (MU) | MAE (MU) | MAPE (%) | Model Class |
| :--- | :---: | :---: | :---: | :--- |
| **AutoTS** | **120.79** | **92.18** | **1.69%** | AutoML Ensemble |
| **ARMA(2, 2)** | 148.37 | 111.59 | 2.06% | Classical TS |
| **Holt-Winters** | 154.21 | 116.76 | 2.17% | Classical TS |
| **ARIMAX(2, 1, 2)** | 155.97 | 118.23 | 2.19% | Classical TS (with exog) |
| **AR(7)** | 159.79 | 120.81 | 2.25% | Classical TS |
| **ARIMA(2, 1, 2)** | 162.23 | 122.08 | 2.27% | Classical TS |
| **VARIMA / VAR** | 163.74 | 126.82 | 2.35% | Multivariate (5 Regions) |
| **SARIMA** | 173.63 | 129.65 | 2.42% | Classical TS |
| **SARIMAX** | 173.63 | 129.65 | 2.42% | Classical TS (with exog) |
| **Prophet** | 338.08 | 321.64 | 5.82% | Additive/Bayesian |
| **MA(7)** | 1745.97 | 1700.79 | 31.02% | Classical TS |

### 6.3 Milestone 2 Generated Artifacts
- **Benchmark Script**: `src/run_milestone2.py`
- **Metrics Table**: `results/milestone2_model_comparison.csv`
- **EDA & Stationarity Plot**: `results/figures/milestone2_eda_stationarity.png`
- **Forecast Curves Comparison**: `results/figures/milestone2_models_comparison.png`
- **Error Metrics Bar Chart**: `results/figures/milestone2_metrics_barplot.png`

---

## 7. Reproducibility Commands
```bash
# 1. Run Milestone 2 time-series benchmark (Classical, Prophet, AutoTS)
python src/run_milestone2.py

# 2. Run rolling-origin backtest (6 origins x 3 horizons x 4 methods)
python -m src.backtest

# 3. Build frontend bundle
cd frontend && npm run build
```
