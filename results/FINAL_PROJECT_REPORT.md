# TerraWatt: Hierarchical Machine Learning and Classical Time Series Forecasting with Deviation Settlement Mechanism (DSM) Optimization for the Indian Power Grid

**Course**: CSET 369 – Time Series Analysis  
**Institution**: School of Computer Science Engineering and Technology, Bennett University  
**Course Coordinator**: Dr. Anurag Choubey  
**Student Name**: Ayush Bhardwaj  
**Enrollment Number**: S24CSEU0132  
**Batch**: 44  
**Date of Submission**: November 2026  

---

## Abstract
Power grid operations require accurate electricity demand forecasts to ensure grid frequency stability, reserve commitment, and financial compliance with electricity grid regulations. In India, the Central Electricity Regulatory Commission (CERC) enforces the Deviation Settlement Mechanism (DSM), which imposes severe asymmetric financial penalties on scheduling deviations. This paper presents **TerraWatt**, an end-to-end hierarchical electricity demand forecasting and dispatch optimization engine developed for the Indian national power grid. The system models a 3-tier geographic hierarchy consisting of 51 time series nodes (India National Root $\rightarrow$ 5 Regional Grids $\rightarrow$ 45 States and bulk consumers) using 5,015 daily observations (2013–2026) from POSOCO / GRID-INDIA. We evaluate classical statistical models (AR, MA, ARMA, ARIMA, SARIMA, Holt-Winters, ARIMAX, SARIMAX, VARIMA) against advanced machine learning and deep architectures (Facebook Prophet, LightGBM, and AutoTS AutoML ensembles) reconciled through Minimum Trace (MinT-Shrink) optimal projection. Experimental results demonstrate that AutoTS and MinT-reconciled LightGBM outperform classical baselines, achieving a National MAPE of $1.50\%\text{--}1.69\%$ and reducing spatial aggregation coherence error to zero. Furthermore, an asymmetric Newsvendor dispatch scheduler reduces state-level DSM financial penalties by $55.36\%$ while regional pooling delivers ₹2.09 crore in annual error-cancellation savings.

**Keywords**—Hierarchical Time Series, Power Grid Demand Forecasting, Minimum Trace (MinT) Reconciliation, Deviation Settlement Mechanism (DSM), ARIMA, AutoTS, Prophet, LightGBM.

---

## 1. Introduction
Electricity cannot be stored cost-effectively at national scale in grid batteries; electrical generation must instantaneously match demand at every microsecond. Under-forecasting demand forces grid operators to procure expensive peak-power reserves or face load-shedding and frequency excursions below 49.90 Hz. Conversely, over-forecasting causes grid over-frequency ($>50.05\text{ Hz}$), forcing generators to back down with significant thermodynamic cycling losses.

In India, power dispatch is governed by the Central Electricity Regulatory Commission (CERC) Deviation Settlement Mechanism (DSM) Regulations 2024. Under DSM rules, scheduling errors are penalized asymmetrically: over-drawal during stressed grid conditions is penalized at punitive escalation rates up to $300\%$ of the Normal Rate (NR), whereas under-drawal receives discounted or zero settlement credits.

Standard forecasting models fail to satisfy Indian utility requirements for two critical reasons:
1. **Spatial Incoherence**: When 45 Indian states and 5 regional grids are forecasted independently, bottom-level predictions do not sum to top-level regional and national totals ($\sum \hat{y}_{\text{states}} \ne \hat{y}_{\text{national}}$), generating multi-hundred megawatt accounting discrepancies.
2. **Loss Asymmetry Mismatch**: Conventional models optimize symmetric $L_2$ loss (RMSE), assuming over- and under-predictions carry identical consequences. Under CERC regulations, this symmetry exposes distribution companies (DISCOMs) to millions of rupees in avoidable over-drawal penalties.

### Project Objectives
- Conduct rigorous statistical preprocessing and stationarity verification (ADF, KPSS) on long-horizon Indian grid consumption data.
- Benchmark classical linear and seasonal time series models (AR, MA, ARMA, ARIMA, SARIMA, Holt-Winters, ARIMAX, SARIMAX, VARIMA) to establish baseline forecasting capabilities.
- Implement advanced Machine Learning, Deep Learning, and automated frameworks (Facebook Prophet, AutoTS, LightGBM) to capture non-linear calendar harmonics and multi-regional dependencies.
- Reconcile independent multi-level forecasts using optimal Minimum Trace (MinT-Shrink) linear projection to enforce exact spatial coherence.
- Implement an asymmetric Newsvendor-based dispatch scheduler aligned with CERC 2024 DSM penalty curves to minimize expected financial penalties and quantify regional pooling benefits.

---

## 2. Literature Review
Time series forecasting for electric power systems has evolved from linear parametric methods to non-linear hierarchical ensembles.

1. **Box, Jenkins, and Reinsel (2015)** established the mathematical foundations for Auto-Regressive Integrated Moving Average (ARIMA) models. While effective for single stationary series with low-frequency seasonality, standard ARIMA models scale poorly when extended to multi-seasonal or spatial hierarchies without severe parameter explosion.
2. **Taylor (2003)** introduced double and multi-seasonal exponential smoothing formulations for short-term load forecasting, extending Holt-Winters models to handle both intraday and intraweek cycles. However, smoothing formulations lack native support for exogenous weather regressors and inter-regional transmission constraints.
3. **Hyndman, Ahmed, Athanasopoulos, and Shang (2011)** formalized optimal reconciliation for hierarchical time series. They proved that classical Top-Down (proportional) and Bottom-Up methods are sub-optimal because they discard correlation information present at intermediate aggregation levels.
4. **Wickramasuriya, Athanasopoulos, and Hyndman (2019)** proposed the Minimum Trace (MinT) estimator. By utilizing the in-sample residual covariance matrix $\mathbf{W}$ with Ledoit-Wolf shrinkage, MinT finds the minimum variance linear unbiased reconciled forecast, outperforming all prior hierarchical combinations.
5. **Taylor and Letham (2018)** developed Facebook Prophet, framing time series forecasting as a generalized additive curve-fitting problem with decomposable trend, seasonality, and holiday matrices, demonstrating robustness against missing points and holiday shifts.
6. **Ke et al. (2017)** introduced LightGBM, utilizing Gradient-Based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). LightGBM has dominated recent load forecasting competitions due to its ability to handle hundreds of lag features and non-linear calendar interactions with low memory overhead.

### Research Gaps Addressed by TerraWatt
While the literature extensively examines symmetric RMSE minimization and hierarchical reconciliation independently, no prior published benchmark has integrated **hierarchical MinT reconciliation** directly with **CERC DSM asymmetric regulatory penalty schedules** and **regional error-cancellation pooling** across the complete 51-node Indian national grid.

---

## 3. Dataset Description & Preprocessing

### 3.1 Data Source & Scope
The empirical dataset is sourced from daily operational logbooks published by the Power System Operation Corporation (POSOCO, now **GRID-INDIA**), the national load dispatch center (NLDC) of India.
- **Time Range**: January 2, 2013 to September 25, 2026 (5,015 consecutive daily observations).
- **Spatial Coverage**: 51 operational hierarchy nodes spanning three geographic tiers:
  - **Level 0 (Root)**: National Total (`India: EnergyMet`)
  - **Level 1 (Regions)**: 5 Synchronous Regional Grids: Northern (`NR`), Western (`WR`), Southern (`SR`), Eastern (`ER`), North-Eastern (`NER`).
  - **Level 2 (States & Utilities)**: 45 constituent state electricity boards, union territories, synthetic residual balancing nodes (`Other_<Region>`), and dedicated industrial bulk consumers (e.g., AMNSIL, BALCO).
- **Target Metric**: Daily Energy Met in Million Units ($\text{MU} = 10^6\text{ kWh} = 1\text{ GWh}$).

### 3.2 Data Cleaning & Anomaly Correction
- **Missing Date Imputation**: Evaluated calendar continuity; missing dates were checked and indexed on continuous daily frequencies (`D`).
- **Telemetry Spike Filtering**: An extreme telemetry transcription error in Assam on `2014-11-25` ($1,190\text{ MU}$ recorded vs. true baseline $\approx 15\text{ MU}$) was identified via rolling $4\sigma$ z-score anomaly detection and corrected via linear interpolation.
- **Regional Normalization**: Western Region telemetry uncalibrated state scaling on `2015-01-19` was reconciled to match state-level sum totals.

### 3.3 Statistical Stationarity Diagnostics
To select appropriate statistical model orders ($d$), the national demand series was tested for stationarity:

```
+-----------------------------------------------------------------------------------------+
|                                STATIONARITY TEST MATRIX                                 |
+------------------------------+--------------------+-----------+-------------------------+
| Series Tested                | Test Statistic     | p-value   | Diagnostic Conclusion   |
+------------------------------+--------------------+-----------+-------------------------+
| Raw Series (India Load)      | ADF:  -1.5788      | 0.4942    | Non-Stationary (Unit    |
|                              | KPSS: 10.1883      | 0.0100    | Root Present, I(1))     |
+------------------------------+--------------------+-----------+-------------------------+
| First-Difference (\Delta y_t)| ADF: -14.8693      | 1.67e-27  | Stationary (I(0), d=1   |
|                              | KPSS:  0.1245      | > 0.10    | Mathematically Proven)  |
+------------------------------+--------------------+-----------+-------------------------+
```

The Augmented Dickey-Fuller (ADF) null hypothesis of a unit root is rejected at $p < 0.001$ for the differenced series $\Delta y_t = y_t - y_{t-1}$, confirming that **first-order differencing ($d=1$)** renders the Indian national demand series strictly stationary.

### 3.4 Leak-Free Feature Engineering
To avoid data leakage during walk-forward validation:
- All lag features ($t-1, t-7, t-30, t-365$) and rolling statistical windows ($7\text{d}, 30\text{d}$ mean and standard deviation) are strictly computed after applying a `.shift(1)` operation. Observation at day $T$ is never used to predict day $T$.
- Calendar indicators (`day_of_week`, `day_of_year`, `month`, `year_trend`, `is_weekend`) and Indian national statutory holidays (`Republic Day`, `Independence Day`, `Diwali`, `Eid-ul-Fitr`, `Gandhi Jayanti`) were generated using the `holidays` library.

---

## 4. Methodology & Mathematical Framework

### 4.1 Classical Time Series Formulations (Phase 2)
1. **Autoregressive AR(7)**:
   $$y_t = c + \sum_{i=1}^7 \phi_i y_{t-i} + \epsilon_t$$
   Parametrized with $p=7$ to capture weekly cyclicity directly through historical lags.
2. **Moving Average MA(7)**:
   $$y_t = \mu + \epsilon_t + \sum_{j=1}^7 \theta_j \epsilon_{t-j}$$
   Evaluates shock dissipation across short error windows.
3. **ARMA(2, 2) & ARIMA(2, 1, 2)**:
   $$\Phi(L)(1 - L)^d y_t = c + \Theta(L)\epsilon_t$$
   Where $L$ is the lag operator, $d=1$ based on ADF test verification, and $p=2, q=2$ based on PACF/ACF cutoff signatures.
4. **SARIMA(1, 1, 1)(1, 1, 1)[7]**:
   $$\Phi_p(L)\tilde{\Phi}_P(L^s)(1 - L)^d(1 - L^s)^D y_t = \Theta_q(L)\tilde{\Theta}_Q(L^s)\epsilon_t$$
   Accounts for joint non-seasonal ($d=1$) and weekly seasonal ($D=1, s=7$) variations.
5. **Holt-Winters Exponential Smoothing**:
   Triple exponential smoothing with additive trend and additive 7-day seasonality:
   $$\hat{y}_{t+h|t} = l_t + h b_t + s_{t+h-m(k+1)}$$
6. **ARIMAX & SARIMAX**:
   Augments the seasonal ARIMA structure with exogenous binary weekend flags:
   $$\Phi(L)(1-L)y_t = \boldsymbol{\beta}^T \mathbf{X}_t + \Theta(L)\epsilon_t$$
7. **Vector Autoregression (VAR / VARIMA across 5 Regions)**:
   $$\mathbf{Y}_t = \mathbf{c} + \sum_{i=1}^p \mathbf{\Gamma}_i \mathbf{Y}_{t-i} + \mathbf{U}_t$$
   Where $\mathbf{Y}_t = [y_{\text{NR},t}, y_{\text{WR},t}, y_{\text{SR},t}, y_{\text{ER},t}, y_{\text{NER},t}]^T$, modeling cross-regional power transfer dynamics.

### 4.2 Advanced Machine Learning & Deep Frameworks (Phase 3)
1. **Facebook Prophet**:
   $$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$
   Where $g(t)$ is a piecewise linear trend, $s(t)$ models weekly and yearly Fourier harmonics, and $h(t)$ models irregular holiday shocks with holiday window priors.
2. **AutoTS AutoML**:
   Genetic search optimization over hundreds of candidate transformations, decomposition mechanisms, and multi-model ensembling.
3. **Gradient Boosted Decision Trees (LightGBM)**:
   Leaf-wise regression trees trained with asymmetric Huber loss, capturing high-order non-linear interactions across 51 individual node models.

### 4.3 Hierarchical Minimum Trace (MinT) Reconciliation
Let $\mathbf{y}_t$ be the vector of all 51 series at time $t$, and $\mathbf{b}_t$ be the 45 bottom-level series. The structural summing matrix $\mathbf{S} \in \{0, 1\}^{51 \times 45}$ defines the hierarchy:
$$\mathbf{y}_t = \mathbf{S} \mathbf{b}_t$$
Independent base forecasts $\hat{\mathbf{y}}_t$ are incoherent ($\hat{\mathbf{y}}_t \ne \mathbf{S} \hat{\mathbf{b}}_t$). The MinT optimal linear unbiased projection computes:
$$\tilde{\mathbf{y}}_t = \mathbf{S} (\mathbf{S}^T \mathbf{W}^{-1} \mathbf{S})^{-1} \mathbf{S}^T \mathbf{W}^{-1} \hat{\mathbf{y}}_t$$
Where $\mathbf{W} = \text{Cov}(\hat{\mathbf{e}}_t)$ is the in-sample residual covariance matrix estimated using **Ledoit-Wolf shrinkage** to guarantee positive definiteness.

### 4.4 Cost-Aware DSM Dispatch Optimization
Under CERC DSM Regulations 2024, deviation volume is $\Delta y = y_{\text{actual}} - y_{\text{schedule}}$. Penalties are asymmetric:
- Over-drawal ($\Delta y > 0$): Charged at $1.0\times, 1.5\times, 2.0\times, 3.0\times$ Normal Rate (NR) across escalating percentage deviation bands.
- Under-drawal ($\Delta y < 0$): Credited at descending haircut rates ($0.9\times, 0.75\times, 0.0\times$ NR).

Using the Newsvendor loss formulation, the cost-minimizing dispatch target corresponds to the critical quantile:
$$q^* = \frac{c_{\text{under-schedule}}}{c_{\text{under-schedule}} + c_{\text{over-schedule}}} \approx 0.5263$$
The optimal dispatch schedule shifts by a protective bias:
$$Y^* = \hat{\mu} + \Phi^{-1}(q^*) \cdot \hat{\sigma} \approx \hat{\mu} + 0.066\hat{\sigma}$$

---

## 5. Experimental Results & Comparative Discussion

All models were evaluated strictly out-of-sample on a **30-day forecast horizon** (`2026-08-27` to `2026-09-25`) against verified actual national demand:

### 5.1 Comprehensive Benchmark Table

```
+=======================================================================================================+
|                            COMPREHENSIVE MODEL BENCHMARK RESULTS TABLE                                |
+======+==================+===========+===========+==========+======================+===================+
| Rank | Model Name       | RMSE (MU) | MAE (MU)  | MAPE (%) | Paradigm / Class     | Phase             |
+======+==================+===========+===========+==========+======================+===================+
| 1    | AutoTS (Ensemble)| 120.79    | 92.18     | 1.69%    | AutoML Pipeline      | Phase 3 (Advanced)|
| 2    | ARMA(2, 2)       | 148.37    | 111.59    | 2.06%    | Classical TS         | Phase 2 (Baseline)|
| 3    | Holt-Winters     | 154.21    | 116.76    | 2.17%    | Exponential Smoothing| Phase 2 (Baseline)|
| 4    | ARIMAX(2, 1, 2)  | 155.97    | 118.23    | 2.19%    | Classical + Exog     | Phase 2 (Baseline)|
| 5    | AR(7)            | 159.79    | 120.81    | 2.25%    | Classical Autoregress| Phase 2 (Baseline)|
| 6    | ARIMA(2, 1, 2)   | 162.23    | 122.08    | 2.27%    | Classical Differenced| Phase 2 (Baseline)|
| 7    | VARIMA / VAR     | 163.74    | 126.82    | 2.35%    | Multivariate (5 Grids| Phase 2 (Baseline)|
| 8    | SARIMA           | 173.63    | 129.65    | 2.42%    | Seasonal Classical   | Phase 2 (Baseline)|
| 9    | SARIMAX          | 173.63    | 129.65    | 2.42%    | Seasonal + Exog      | Phase 2 (Baseline)|
| 10   | Facebook Prophet | 338.08    | 321.64    | 5.82%    | Bayesian Additive    | Phase 3 (Advanced)|
| 11   | MA(7)            | 1745.97   | 1700.79   | 31.02%   | Pure Moving Average  | Phase 2 (Baseline)|
+======+==================+===========+===========+==========+======================+===================+
```

### 5.2 National Hierarchy Reconciled Performance Across Horizons
Evaluated across 6 rolling forecast origins (7-day, 14-day, 30-day horizons):

```
+========================================================================================================+
|                       HIERARCHICAL RECONCILIATION & DSM PERFORMANCE COMPARISON                         |
+==========+====================+==========+===========+===========+=================+===================+
| Horizon  | Method             | MAE (MU) | RMSE (MU) | WMAPE (%) | Coherence Error | Total Penalty (₹) |
+==========+====================+==========+===========+===========+=================+===================+
| 7-Day    | Base LightGBM      | 77.86    | 101.32    | 1.67%     | 104.22 MU (❌)  | ₹3,611,089        |
| 7-Day    | MinT-Shrink        | 76.73    | 98.79     | 1.65%     | 0.00 MU (✅)    | ₹1,500,750        |
| 7-Day    | Cost-Aware MinT    | 74.53    | 98.83     | 1.60%     | 0.00 MU (✅)    | -₹182,678 (Credit)|
+----------+--------------------+----------+-----------+-----------+-----------------+-------------------+
| 14-Day   | Base LightGBM      | 91.78    | 127.11    | 1.93%     | 173.61 MU (❌)  | ₹11,628,109       |
| 14-Day   | MinT-Shrink        | 80.36    | 109.07    | 1.69%     | 0.00 MU (✅)    | ₹8,104,482        |
| 14-Day   | Cost-Aware MinT    | 77.42    | 107.42    | 1.63%     | 0.00 MU (✅)    | ₹4,683,993 (-42%) |
+----------+--------------------+----------+-----------+-----------+-----------------+-------------------+
| 30-Day   | Base LightGBM      | 79.46    | 111.00    | 1.67%     | 173.61 MU (❌)  | ₹14,456,161       |
| 30-Day   | MinT-Shrink        | 74.52    | 100.60    | 1.56%     | 0.00 MU (✅)    | ₹15,651,114       |
| 30-Day   | Cost-Aware MinT    | 71.67    | 99.13     | 1.50%     | 0.00 MU (✅)    | ₹8,283,634 (-47%) |
+==========+====================+==========+===========+===========+=================+===================+
```

### 5.3 Comparative Analysis: Classical vs. Advanced Techniques

#### 1. Why AutoTS & MinT-LightGBM Outperform Classical Models
- **AutoTS** achieved the lowest single-model error (**RMSE: 120.79 MU, MAPE: 1.69%**). It discovered that ensembling a seasonal decomposition transformer with an autoregressive rolling window regressor neutralized both weekly calendar drops and gradual weather-driven level shifts.
- **MinT-Reconciled LightGBM** achieved the lowest overall 30-day National error (**MAPE: 1.50%**). By leveraging covariance shrinkage across all 51 nodes, errors in individual states cancel out during matrix projection, strictly eliminating the $173.61\text{ MU}$ incoherence present in standalone base forecasts.

#### 2. Why Pure Moving Average MA(7) Fails Catastrophically
- The MA(7) model exhibited a crippling **31.02% MAPE (1,745.97 MU RMSE)**. By mathematical definition, the moving average terms $\theta_j \epsilon_{t-j}$ vanish to zero for all forecast horizons $h > q=7$. Consequently, for days 8 through 30, the model forecasts the unconditional sample mean ($\approx 3,700\text{ MU}$), completely missing the 2026 economic baseline ($\approx 5,400\text{ MU}$). In electrical grid dispatch, this would trigger catastrophic emergency over-drawal penalties under CERC norms.

#### 3. Autoregressive Persistence & Exogenous Value (AR vs. ARMAX)
- Both **AR(7)** ($2.25\%$ MAPE) and **ARMA(2,2)** ($2.06\%$ MAPE) proved resilient because autoregressive coefficients capture load inertia.
- Incorporating binary weekend regressors in **ARIMAX** lowered MAE from $122.08$ to $118.23\text{ MU}$, correcting Monday morning load surges that classical univariate ARIMA misinterprets as sudden trend changes.

#### 4. Facebook Prophet's Performance Characteristics
- Prophet recorded a **5.82% MAPE** ($338.08\text{ MU}$ RMSE). While Prophet accurately captured statutory Indian holidays and long-term trend changepoints, its smooth Fourier seasonal components were too rigid to capture sharp weather-induced daily variance compared to gradient-boosted decision trees.

### 5.4 Domain Impact: CERC DSM Deviation Savings & Regional Pooling
1. **Direct Penalty Reduction via Cost-Aware Dispatch**:
   - Baseline MinT Total Penalty across 45 states: **₹2,694,019**
   - Cost-Aware MinT Penalty: **₹1,202,664**
   - **Direct Financial Savings**: **₹1,491,355 saved (55.36% reduction)** without increasing true volumetric error.
2. **Regional Pooling Benefit (Error Cancellation)**:
   - When states coordinate schedules within regional grids, over-drawals in one state offset under-drawals in neighboring states.
   - **Standalone State Penalties**: ₹240,025,040
   - **Pooled Regional Penalties**: ₹219,123,124
   - **Total Savings from Regional Coordination**: **₹20,901,916 (8.7% net savings)**. Northern Region (NR) achieved the highest error cancellation ratio ($25.6\%$), saving ₹65.2 lakh alone.

---

## 6. Limitations and Future Improvements

### 6.1 Limitations of Current Work
1. **Daily Resolution**: Operating on daily energy met ($\text{MU}$) rather than 15-minute time blocks precludes modeling sub-hourly ramp rates and peak-shaving storage dispatches.
2. **Absence of Numerical Weather Regressors**: Temperature, heat index, and monsoon precipitation were proxied through calendar lags and seasonal Fourier harmonics rather than direct satellite weather feeds.
3. **Linearity of Summing Matrix**: While MinT provides optimal linear unbiased reconciliation, extreme transmission congestion can introduce localized physical non-linearities in inter-state power corridors.

### 6.2 Future Improvements (Phase 4 / Production Roadmap)
1. **15-Minute SCADA Telemetry Stream**: Ingest live 96 time-block SCADA telemetry from State Load Despatch Centres (SLDCs) for intra-day rolling revision scheduling.
2. **Temporal Fusion Transformers (TFT)**: Implement multi-horizon attention networks to model explicit weather reanalysis features (ERA5 temperature, humidity, solar irradiance).
3. **Constrained AC-OPF Integration**: Integrate optimal power flow network impedance constraints directly into the MinT shrinkage projection.

---

## 7. Conclusion
This project developed and validated an end-to-end hierarchical time series forecasting and dispatch optimization engine for the Indian national power grid. By benchmarking classical statistical models against modern advanced architectures across 5,015 daily observations, we verified that:
1. First-order differencing ($d=1$) is mathematically required to eliminate non-stationarity ($p = 1.67 \times 10^{-27}$).
2. Advanced AutoML (AutoTS) and MinT-reconciled LightGBM outperform classical baselines, delivering state-of-the-art accuracy ($1.50\%\text{--}1.69\%$ MAPE) while eliminating hierarchy spatial incoherence.
3. Aligning forecasting objectives with regulatory tariff structures through cost-aware Newsvendor dispatch reduces DSM financial penalties by **$55.36\%$**, and regional grid pooling unlocks **₹2.09 crore** in annual error-cancellation benefits.

---

## 8. References (IEEE Format)

1. G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, *Time Series Analysis: Forecasting and Control*, 5th ed. Hoboken, NJ, USA: John Wiley & Sons, 2015.
2. R. J. Hyndman, R. A. Ahmed, G. Athanasopoulos, and H. L. Shang, "Optimal combination forecasts for hierarchical time series," *Computational Statistics & Data Analysis*, vol. 55, no. 9, pp. 2579–2589, Sep. 2011.
3. S. L. Wickramasuriya, G. Athanasopoulos, and R. J. Hyndman, "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization," *Journal of the American Statistical Association*, vol. 114, no. 526, pp. 804–819, Apr. 2019.
4. J. W. Taylor, "Short-term electricity demand forecasting using double seasonal exponential smoothing," *Journal of the Operational Research Society*, vol. 54, no. 8, pp. 799–805, Aug. 2003.
5. S. J. Taylor and B. Letham, "Forecasting at scale," *The American Statistician*, vol. 72, no. 1, pp. 37–45, Jan. 2018.
6. G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 3146–3154.
7. O. Ledoit and M. Wolf, "A well-conditioned estimator for large-dimensional covariance matrices," *Journal of Multivariate Analysis*, vol. 88, no. 2, pp. 365–411, Feb. 2004.
8. Central Electricity Regulatory Commission (CERC), *Deviation Settlement Mechanism and Related Matters Regulations, 2024*, New Delhi, India: Government of India Gazette, 2024.
9. Power System Operation Corporation (POSOCO) / GRID-INDIA, *National Electricity Demand and Grid Operational Reports (2013–2026)*, New Delhi, India: NLDC, 2026.
10. C. Catlett et al., "AutoTS: Automated Time Series Forecasting for Python," *Python Package Index*, 2024. [Online]. Available: https://github.com/winedarksea/AutoTS

---

## 9. Appendix

### Appendix A: Reproducibility Commands
All experiments, models, and figures are fully reproducible from the repository:
```bash
# 1. Run Complete Milestone 2 & Phase 3 Classical vs Advanced Model Benchmarks
python src/run_milestone2.py

# 2. Run 6-Origin Rolling Backtest & MinT Reconciliation
python -m src.backtest

# 3. Start Backend REST API and Interactive Operations Dashboard
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
npm --prefix frontend run dev
```

### Appendix B: File Artifact Locations
- **Final Project Report (IEEE Format)**: `results/FINAL_PROJECT_REPORT.md`
- **Milestone 2 & 3 Metrics CSV**: `results/milestone2_model_comparison.csv`
- **EDA & Stationarity Diagnostics Plot**: `results/figures/milestone2_eda_stationarity.png`
- **11-Model Out-of-Sample Forecast Curves Plot**: `results/figures/milestone2_models_comparison.png`
- **Model Evaluation Error Barplot (RMSE & MAPE)**: `results/figures/milestone2_metrics_barplot.png`
- **Interactive Jupyter Notebook**: `notebooks/02_milestone2_models.ipynb`

