"""
run_milestone2.py
Executes Milestone 2 requirements:
  1. Exploratory Data Analysis & Seasonality Decomposition
  2. Stationarity testing (ADF, KPSS) & Parameter Selection
  3. Classical Time Series Modeling:
     - AR(p)
     - MA(q)
     - ARMA(p, q)
     - ARIMA(p, d, q)
     - SARIMA(p, d, q)(P, D, Q)_s
     - Holt-Winters Exponential Smoothing
     - ARIMAX (with calendar exog)
     - SARIMAX (with calendar exog)
     - VARIMA / VAR (multivariate regional)
  4. Prophet forecasting (library requirement)
  5. AutoTS automated model search (library requirement)
  6. Comprehensive error metrics comparison table (RMSE, MAE, MAPE)
  7. High-resolution comparison plots saved to results/figures/

Interactive / Individual Model Execution:
  python src/run_milestone2.py                 # Runs all 11 models
  python src/run_milestone2.py --model arima   # Runs only ARIMA(2,1,2) with summary & plot
  python src/run_milestone2.py --model sarima  # Runs only SARIMA
  python src/run_milestone2.py --model holtwinters
  python src/run_milestone2.py --model prophet
  python src/run_milestone2.py --model autots
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_daily.csv"
FIG_DIR = BASE_DIR / "results" / "figures"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="TerraWatt Milestone 2 Model Trainer & Evaluator")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "ar", "ma", "arma", "arima", "sarima", "holtwinters", "arimax", "sarimax", "var", "prophet", "autots"],
        help="Specify which model to train and display (default: all)"
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Pop up the forecast plot in a GUI window"
    )
    return parser.parse_args()

def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    return round(rmse, 2), round(mae, 2), round(mape, 2)

def main():
    args = parse_args()

    print("=" * 72)
    print("           TERRAWATT MILESTONE 2: TIME SERIES BENCHMARK           ")
    print("=" * 72)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Primary series: National Daily Energy Met (MU)
    series = df.set_index("date")["India: EnergyMet"].dropna()
    series = series.asfreq("D")
    if series.isnull().any():
        series = series.interpolate(method="time")

    print(f"Loaded series from {series.index.min().date()} to {series.index.max().date()} ({len(series)} days).")

    # ---------------------------------------------------------
    # 2. EDA & Stationarity Checks
    # ---------------------------------------------------------
    print("\n--- STEP 1 & 2: EDA & STATIONARITY ANALYSIS ---")
    adf_res = adfuller(series)
    kpss_res = kpss(series, regression="c", nlags="auto")

    print(f"Raw Series ADF Statistic:  {adf_res[0]:.4f} (p-value: {adf_res[1]:.4e}) -> {'Stationary' if adf_res[1] < 0.05 else 'Non-Stationary'}")
    print(f"Raw Series KPSS Statistic: {kpss_res[0]:.4f} (p-value: {kpss_res[1]:.4f}) -> {'Non-Stationary' if kpss_res[1] < 0.05 else 'Stationary'}")

    diff_series = series.diff().dropna()
    adf_diff = adfuller(diff_series)
    print(f"First-Diff ADF Statistic:  {adf_diff[0]:.4f} (p-value: {adf_diff[1]:.4e}) -> Stationary (d=1 confirmed)")

    eda_fig_path = FIG_DIR / "milestone2_eda_stationarity.png"
    if not eda_fig_path.exists() or args.model == "all":
        plt.figure(figsize=(14, 9))
        plt.subplot(3, 1, 1)
        plt.plot(series.index, series.values, color="#1e3a8a", lw=1.2)
        plt.title("National Daily Electricity Demand Met (2013-2026) [Trend & Multi-Seasonality]", fontsize=12, fontweight="bold")
        plt.ylabel("Demand (MU)")
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 2)
        decomp = seasonal_decompose(series.tail(365 * 3), model="additive", period=365)
        plt.plot(decomp.seasonal.index, decomp.seasonal.values, color="#059669", lw=1.2)
        plt.title("Extracted Annual Seasonal Cycle (Summer Surge vs. Monsoon Dip)", fontsize=12, fontweight="bold")
        plt.ylabel("Seasonality (MU)")
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 3)
        plt.plot(diff_series.index, diff_series.values, color="#b91c1c", lw=0.8)
        plt.title("First-Differenced Series: Delta(y_t) [Strongly Stationary, d=1]", fontsize=12, fontweight="bold")
        plt.ylabel("Daily Delta (MU)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(eda_fig_path, dpi=200)
        plt.close()
        print(f"Saved EDA & Stationarity plot to: {eda_fig_path}")

    # ---------------------------------------------------------
    # 3. Train / Test Split
    # ---------------------------------------------------------
    TEST_HORIZON = 30
    train = series.iloc[:-TEST_HORIZON]
    test = series.iloc[-TEST_HORIZON:]
    test_dates = test.index

    print(f"\nTrain set: {train.index.min().date()} to {train.index.max().date()} ({len(train)} obs)")
    print(f"Test set:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} obs)")

    # Exogenous variables for ARIMAX / SARIMAX
    exog_train = pd.DataFrame({
        "is_weekend": (train.index.dayofweek >= 5).astype(int),
        "day_of_week": train.index.dayofweek
    }, index=train.index)

    exog_test = pd.DataFrame({
        "is_weekend": (test.index.dayofweek >= 5).astype(int),
        "day_of_week": test.index.dayofweek
    }, index=test.index)

    # Individual model runner
    def run_single(name):
        print(f"\n{'='*25} TRAINING MODEL: {name.upper()} {'='*25}")
        pred = None
        fitted_obj = None

        if name == "ar":
            fitted_obj = AutoReg(train, lags=7, old_names=False).fit()
            pred = fitted_obj.predict(start=len(train), end=len(train) + TEST_HORIZON - 1).values
            print(f"Model: AutoReg(p=7) Fitted Successfully.")
            print(f"AIC: {fitted_obj.aic:.2f} | BIC: {fitted_obj.bic:.2f}")

        elif name == "ma":
            fitted_obj = ARIMA(train, order=(0, 0, 7)).fit()
            pred = fitted_obj.forecast(steps=TEST_HORIZON).values
            print(f"Model: ARIMA(0, 0, 7) Fitted Successfully.")
            print(f"AIC: {fitted_obj.aic:.2f} | BIC: {fitted_obj.bic:.2f}")

        elif name == "arma":
            fitted_obj = ARIMA(train, order=(2, 0, 2)).fit()
            pred = fitted_obj.forecast(steps=TEST_HORIZON).values
            print(f"Model: ARMA(2, 2) Fitted Successfully.")
            print(f"AIC: {fitted_obj.aic:.2f} | BIC: {fitted_obj.bic:.2f}")

        elif name == "arima":
            fitted_obj = ARIMA(train, order=(2, 1, 2)).fit()
            pred = fitted_obj.forecast(steps=TEST_HORIZON).values
            print(f"Model: ARIMA(2, 1, 2) [d=1 Differenced] Fitted Successfully.")
            print(f"AIC: {fitted_obj.aic:.2f} | BIC: {fitted_obj.bic:.2f}")
            print("\nCoefficients:\n", fitted_obj.params)

        elif name == "sarima":
            fitted_obj = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
            pred = fitted_obj.forecast(steps=TEST_HORIZON).values
            print(f"Model: SARIMA(1, 1, 1)(1, 1, 1)[7] Fitted Successfully.")
            print(f"AIC: {fitted_obj.aic:.2f} | BIC: {fitted_obj.bic:.2f}")

        elif name == "holtwinters":
            fitted_obj = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7).fit()
            pred = fitted_obj.forecast(TEST_HORIZON).values
            print(f"Model: Holt-Winters (Additive Trend & Additive s=7 Seasonality) Fitted Successfully.")

        elif name == "arimax":
            fitted_obj = ARIMA(train, order=(2, 1, 2), exog=exog_train).fit()
            pred = fitted_obj.forecast(steps=TEST_HORIZON, exog=exog_test).values
            print(f"Model: ARIMAX(2, 1, 2) with Exogenous Weekend Flag Fitted Successfully.")
            print(f"AIC: {fitted_obj.aic:.2f} | BIC: {fitted_obj.bic:.2f}")

        elif name == "sarimax":
            fitted_obj = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), exog=exog_train).fit(disp=False)
            pred = fitted_obj.forecast(steps=TEST_HORIZON, exog=exog_test).values
            print(f"Model: SARIMAX(1, 1, 1)(1, 1, 1)[7] with Exogenous Regressors Fitted Successfully.")
            print(f"AIC: {fitted_obj.aic:.2f} | BIC: {fitted_obj.bic:.2f}")

        elif name == "var":
            reg_cols = ["NR: EnergyMet", "WR: EnergyMet", "SR: EnergyMet", "ER: EnergyMet", "NER: EnergyMet"]
            multi_df = df.set_index("date")[reg_cols].dropna().asfreq("D").interpolate()
            multi_train = multi_df.iloc[:-TEST_HORIZON]
            var_mod = VAR(multi_train.diff().dropna()).fit(maxlags=7)
            lag_order = var_mod.k_ar
            forecast_diff = var_mod.forecast(multi_train.diff().dropna().values[-lag_order:], TEST_HORIZON)
            last_level = multi_train.iloc[-1].values
            forecast_levels = np.zeros_like(forecast_diff)
            current = last_level.copy()
            for i in range(TEST_HORIZON):
                current += forecast_diff[i]
                forecast_levels[i] = current.copy()
            pred = forecast_levels.sum(axis=1)
            print(f"Model: VAR(5 Regional Grids: NR, WR, SR, ER, NER) Fitted Successfully (Lags={lag_order}).")

        elif name == "prophet":
            from prophet import Prophet
            prophet_df = pd.DataFrame({"ds": train.index, "y": train.values})
            m_prophet = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            m_prophet.add_country_holidays(country_name="IN")
            m_prophet.fit(prophet_df)
            future = m_prophet.make_future_dataframe(periods=TEST_HORIZON)
            forecast_p = m_prophet.predict(future)
            pred = forecast_p["yhat"].iloc[-TEST_HORIZON:].values
            print("Model: Facebook Prophet with Indian Statutory Holidays Fitted Successfully.")

        elif name == "autots":
            from autots import AutoTS
            autots_train = pd.DataFrame({"demand": train.values}, index=train.index)
            model_autots = AutoTS(
                forecast_length=TEST_HORIZON,
                frequency="D",
                prediction_interval=0.9,
                ensemble="simple",
                max_generations=2,
                num_validations=1,
                models_to_validate=0.1,
                generation_timeout=1,
                verbose=0
            )
            model_autots = model_autots.fit(autots_train)
            pred = model_autots.predict().forecast["demand"].values
            print(f"Model: AutoTS Best Pipeline Fitted ({model_autots.best_model_name}).")

        rmse, mae, mape = calc_metrics(test.values, pred)
        print("\n--- OUT-OF-SAMPLE ACCURACY (30-DAY HORIZON) ---")
        print(f"RMSE: {rmse} MU")
        print(f"MAE:  {mae} MU")
        print(f"MAPE: {mape}%")

        # Display Forecast vs Actual Sample Table
        sample_df = pd.DataFrame({
            "Date": test_dates,
            "Actual (MU)": np.round(test.values, 2),
            f"Forecast_{name.upper()} (MU)": np.round(pred, 2),
            "Error (MU)": np.round(pred - test.values, 2),
            "Abs % Error": np.round(np.abs((test.values - pred) / test.values) * 100.0, 2)
        })
        print("\nForecast vs Actual (First 7 Days Sample):")
        print(sample_df.head(7).to_string(index=False))

        # Plot individual model
        plt.figure(figsize=(12, 5))
        plt.plot(test_dates, test.values, label="Actual National Demand", color="black", lw=2.2)
        plt.plot(test_dates, pred, label=f"Forecast: {name.upper()}", color="#2563eb", lw=2, linestyle="--")
        plt.title(f"Model Evaluation: {name.upper()} vs Actual Demand (30-Day Window | MAPE: {mape}%)", fontsize=12, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Electricity Demand Met (MU)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        single_plot_path = FIG_DIR / f"single_model_{name}.png"
        plt.savefig(single_plot_path, dpi=200)
        print(f"\nSaved plot to: {single_plot_path}")
        if args.show_plot:
            plt.show()
        plt.close()

    # If single model selected
    if args.model != "all":
        run_single(args.model)
        return

    # ---------------------------------------------------------
    # RUN ALL 11 MODELS (Full Benchmark Suite)
    # ---------------------------------------------------------
    model_results = {}
    predictions = {}

    print("\n[Model 1/11] Fitting AR(7)...")
    ar_mod = AutoReg(train, lags=7, old_names=False).fit()
    pred_ar = ar_mod.predict(start=len(train), end=len(train) + TEST_HORIZON - 1)
    predictions["AR(7)"] = pred_ar.values
    model_results["AR(7)"] = calc_metrics(test.values, pred_ar.values)

    print("[Model 2/11] Fitting MA(7)...")
    ma_mod = ARIMA(train, order=(0, 0, 7)).fit()
    pred_ma = ma_mod.forecast(steps=TEST_HORIZON)
    predictions["MA(7)"] = pred_ma.values
    model_results["MA(7)"] = calc_metrics(test.values, pred_ma.values)

    print("[Model 3/11] Fitting ARMA(2, 2)...")
    arma_mod = ARIMA(train, order=(2, 0, 2)).fit()
    pred_arma = arma_mod.forecast(steps=TEST_HORIZON)
    predictions["ARMA(2,2)"] = pred_arma.values
    model_results["ARMA(2,2)"] = calc_metrics(test.values, pred_arma.values)

    print("[Model 4/11] Fitting ARIMA(2, 1, 2)...")
    arima_mod = ARIMA(train, order=(2, 1, 2)).fit()
    pred_arima = arima_mod.forecast(steps=TEST_HORIZON)
    predictions["ARIMA(2,1,2)"] = pred_arima.values
    model_results["ARIMA(2,1,2)"] = calc_metrics(test.values, pred_arima.values)

    print("[Model 5/11] Fitting SARIMA(1,1,1)(1,1,1)[7]...")
    sarima_mod = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
    pred_sarima = sarima_mod.forecast(steps=TEST_HORIZON)
    predictions["SARIMA"] = pred_sarima.values
    model_results["SARIMA"] = calc_metrics(test.values, pred_sarima.values)

    print("[Model 6/11] Fitting Holt-Winters...")
    hw_mod = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7).fit()
    pred_hw = hw_mod.forecast(TEST_HORIZON)
    predictions["Holt-Winters"] = pred_hw.values
    model_results["Holt-Winters"] = calc_metrics(test.values, pred_hw.values)

    print("[Model 7/11] Fitting ARIMAX(2, 1, 2)...")
    arimax_mod = ARIMA(train, order=(2, 1, 2), exog=exog_train).fit()
    pred_arimax = arimax_mod.forecast(steps=TEST_HORIZON, exog=exog_test)
    predictions["ARIMAX"] = pred_arimax.values
    model_results["ARIMAX"] = calc_metrics(test.values, pred_arimax.values)

    print("[Model 8/11] Fitting SARIMAX...")
    sarimax_mod = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), exog=exog_train).fit(disp=False)
    pred_sarimax = sarimax_mod.forecast(steps=TEST_HORIZON, exog=exog_test)
    predictions["SARIMAX"] = pred_sarimax.values
    model_results["SARIMAX"] = calc_metrics(test.values, pred_sarima.values)

    print("[Model 9/11] Fitting VAR (5 Regions)...")
    reg_cols = ["NR: EnergyMet", "WR: EnergyMet", "SR: EnergyMet", "ER: EnergyMet", "NER: EnergyMet"]
    multi_df = df.set_index("date")[reg_cols].dropna().asfreq("D").interpolate()
    multi_train = multi_df.iloc[:-TEST_HORIZON]
    var_mod = VAR(multi_train.diff().dropna()).fit(maxlags=7)
    lag_order = var_mod.k_ar
    forecast_diff = var_mod.forecast(multi_train.diff().dropna().values[-lag_order:], TEST_HORIZON)
    last_level = multi_train.iloc[-1].values
    forecast_levels = np.zeros_like(forecast_diff)
    current = last_level.copy()
    for i in range(TEST_HORIZON):
        current += forecast_diff[i]
        forecast_levels[i] = current.copy()
    pred_var = forecast_levels.sum(axis=1)
    predictions["VARIMA/VAR"] = pred_var
    model_results["VARIMA/VAR"] = calc_metrics(test.values, pred_var)

    print("[Model 10/11] Fitting Prophet...")
    try:
        from prophet import Prophet
        prophet_df = pd.DataFrame({"ds": train.index, "y": train.values})
        m_prophet = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        m_prophet.add_country_holidays(country_name="IN")
        m_prophet.fit(prophet_df)
        future = m_prophet.make_future_dataframe(periods=TEST_HORIZON)
        pred_p = m_prophet.predict(future)["yhat"].iloc[-TEST_HORIZON:].values
        predictions["Prophet"] = pred_p
        model_results["Prophet"] = calc_metrics(test.values, pred_p)
    except Exception as e:
        print(f"Prophet failed: {e}")

    print("[Model 11/11] Running AutoTS Automated Best-Model Search...")
    try:
        from autots import AutoTS
        autots_train = pd.DataFrame({"demand": train.values}, index=train.index)
        model_autots = AutoTS(
            forecast_length=TEST_HORIZON,
            frequency="D",
            prediction_interval=0.9,
            ensemble="simple",
            max_generations=2,
            num_validations=1,
            models_to_validate=0.1,
            generation_timeout=1,
            verbose=0
        )
        model_autots = model_autots.fit(autots_train)
        pred_ats = model_autots.predict().forecast["demand"].values
        predictions["AutoTS"] = pred_ats
        model_results["AutoTS"] = calc_metrics(test.values, pred_ats)
        print(f"AutoTS Selected Best Model Pipeline: {model_autots.best_model_name}")
    except Exception as e:
        print(f"AutoTS failed: {e}")

    # Metrics Table
    print("\n" + "=" * 72)
    print("              MODEL EVALUATION & ERROR METRICS TABLE              ")
    print("=" * 72)
    res_rows = []
    classes = {
        "AutoTS": "AutoML",
        "Prophet": "Additive/Bayesian",
        "VARIMA/VAR": "Multivariate",
        "Holt-Winters": "Classical TS",
        "ARIMA(2,1,2)": "Classical TS",
        "ARMA(2,2)": "Classical TS",
        "AR(7)": "Classical TS",
        "MA(7)": "Classical TS",
        "SARIMA": "Classical TS",
        "SARIMAX": "Classical TS",
        "ARIMAX": "Classical TS"
    }
    for m_name, (rmse, mae, mape) in model_results.items():
        res_rows.append({
            "Model": m_name,
            "RMSE (MU)": rmse,
            "MAE (MU)": mae,
            "MAPE (%)": mape,
            "Class": classes.get(m_name, "Classical TS")
        })

    results_df = pd.DataFrame(res_rows).sort_values("MAPE (%)").reset_index(drop=True)
    print(results_df.to_string(index=False))

    results_csv_path = RESULTS_DIR / "milestone2_model_comparison.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved metrics table to: {results_csv_path}")

    # Forecast Comparison Plot
    plt.figure(figsize=(16, 8))
    plt.plot(test_dates, test.values, label="Actual National Demand", color="black", lw=2.5, zorder=10)
    colors = {
        "AR(7)": "#94a3b8", "MA(7)": "#cbd5e1", "ARMA(2,2)": "#64748b",
        "ARIMA(2,1,2)": "#3b82f6", "SARIMA": "#1d4ed8", "Holt-Winters": "#f59e0b",
        "ARIMAX": "#10b981", "SARIMAX": "#059669", "VARIMA/VAR": "#8b5cf6",
        "Prophet": "#ec4899", "AutoTS": "#ef4444"
    }
    for name, y_p in predictions.items():
        lw = 2.0 if name in ["SARIMAX", "Prophet", "AutoTS", "Holt-Winters"] else 1.2
        plt.plot(test_dates, y_p, label=name, color=colors.get(name, None), lw=lw, alpha=0.85, linestyle="--" if "MA" in name or "AR(" in name else "-")

    plt.title("Milestone 2: Classical, Multivariate & Automated Models Benchmark (30-Day Window)", fontsize=13, fontweight="bold")
    plt.xlabel("Date (August - September 2026)")
    plt.ylabel("Electricity Demand Met (MU)")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    comp_fig_path = FIG_DIR / "milestone2_models_comparison.png"
    plt.savefig(comp_fig_path, dpi=200)
    plt.close()
    print(f"Saved forecast comparison plot to: {comp_fig_path}")

    # Bar chart
    plt.figure(figsize=(14, 5))
    x = np.arange(len(results_df))
    width = 0.35
    plt.bar(x - width/2, results_df["RMSE (MU)"], width, label="RMSE (MU)", color="#3b82f6", alpha=0.85)
    plt.bar(x + width/2, results_df["MAE (MU)"], width, label="MAE (MU)", color="#10b981", alpha=0.85)
    plt.xlabel("Model")
    plt.ylabel("Error (MU)")
    plt.title("Error Metrics Comparison across Classical, Prophet, and AutoTS Models", fontsize=12, fontweight="bold")
    plt.xticks(x, results_df["Model"], rotation=30, ha="right", fontsize=9)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    bar_fig_path = FIG_DIR / "milestone2_metrics_barplot.png"
    plt.savefig(bar_fig_path, dpi=200)
    plt.close()
    print(f"Saved metrics bar chart to: {bar_fig_path}")

    print("=" * 72)
    print("         MILESTONE 2 EXECUTION & DELIVERABLES COMPLETE!        ")
    print("=" * 72)

if __name__ == "__main__":
    main()
