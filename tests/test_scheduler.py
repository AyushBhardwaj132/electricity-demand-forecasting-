"""
tests/test_scheduler.py
Unit tests for the schedule recommendation engine.
"""

import pytest
import numpy as np
from src.scheduler import (
    compute_optimal_schedule,
    compute_schedule_recommendations,
    estimate_forecast_uncertainty,
)
from src.penalties import DEFAULT_NORMAL_RATE


def test_zero_uncertainty():
    """With zero uncertainty, optimal schedule matches point forecast exactly."""
    res = compute_optimal_schedule(point_forecast_mw=500.0, forecast_std_mw=0.0)
    assert res["optimal_schedule_mw"] == 500.0
    assert res["schedule_bias_mw"] == 0.0
    assert res["bias_direction"] == "neutral"


def test_positive_uncertainty_bias():
    """Under asymmetric DSM penalties, uncertainty induces a positive over-scheduling bias."""
    res = compute_optimal_schedule(point_forecast_mw=1000.0, forecast_std_mw=50.0)
    assert res["optimal_schedule_mw"] >= res["point_forecast_mw"]
    assert res["bias_direction"] in ["over_schedule", "neutral"]
    assert res["penalty_savings_inr"] >= 0.0
    assert res["optimal_quantile"] > 0.5


def test_larger_uncertainty_larger_bias():
    """Greater forecast uncertainty causes a larger optimal bias in MW."""
    res_low = compute_optimal_schedule(point_forecast_mw=1000.0, forecast_std_mw=20.0)
    res_high = compute_optimal_schedule(point_forecast_mw=1000.0, forecast_std_mw=100.0)
    assert res_high["schedule_approx_mw"] > res_low["schedule_approx_mw"]


def test_batch_schedule_recommendations():
    """Batch calculation works for multiple nodes."""
    forecasts = {"India/NR/Delhi": 1200.0, "India/WR/Maharashtra": 4500.0}
    stds = {"India/NR/Delhi": 60.0, "India/WR/Maharashtra": 180.0}
    results = compute_schedule_recommendations(forecasts, stds)
    assert len(results) == 2
    assert "India/NR/Delhi" in results
    assert results["India/NR/Delhi"]["point_forecast_mw"] == 1200.0
    assert results["India/WR/Maharashtra"]["optimal_schedule_mw"] >= 4500.0


def test_estimate_forecast_uncertainty():
    """Check estimation of residuals uncertainty."""
    np.random.seed(42)
    residuals = np.random.normal(0, 25.0, 1000)
    std_emp = estimate_forecast_uncertainty(residuals, method="empirical_std")
    std_mad = estimate_forecast_uncertainty(residuals, method="mad")
    std_iqr = estimate_forecast_uncertainty(residuals, method="iqr")
    assert 20.0 < std_emp < 30.0
    assert 20.0 < std_mad < 30.0
    assert 20.0 < std_iqr < 30.0

