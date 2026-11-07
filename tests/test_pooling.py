"""
tests/test_pooling.py
Unit tests for the regional pooling benefit calculation module.
"""

import pytest
import numpy as np
from src.pooling import (
    compute_standalone_penalties,
    compute_hierarchy_penalties,
    compute_pooling_benefit,
    compute_error_cancellation_ratio,
)


def test_error_cancellation_ratio():
    # Two states: State A errors = +10, State B errors = -10 (perfect cancellation)
    errors = {
        "State_A": np.array([10.0, 10.0, 10.0]),
        "State_B": np.array([-10.0, -10.0, -10.0]),
    }
    ratio = compute_error_cancellation_ratio(errors)
    # Total sum of errors = 0, sum of abs errors = 60 -> cancellation = 1.0
    assert pytest.approx(ratio, 0.01) == 1.0

    # No cancellation: both positive
    errors_same = {
        "State_A": np.array([10.0, 10.0]),
        "State_B": np.array([10.0, 10.0]),
    }
    ratio_same = compute_error_cancellation_ratio(errors_same)
    assert pytest.approx(ratio_same, 0.01) == 0.0


def test_pooling_benefit_with_cancellation():
    actuals = {
        "India/NR/Delhi": np.array([110.0, 110.0]),     # +10 MW over-drawal
        "India/NR/Punjab": np.array([90.0, 90.0]),       # -10 MW under-drawal
    }
    forecasts = {
        "India/NR/Delhi": np.array([100.0, 100.0]),
        "India/NR/Punjab": np.array([100.0, 100.0]),
    }
    region_members = {"NR": ["Delhi: EnergyMet", "Punjab: EnergyMet"]}

    res = compute_pooling_benefit(actuals, forecasts, region_members=region_members, normal_rate=4000.0)
    summary = res["summary"]

    # Standalone: Delhi pays 40k x 2 = 80k INR, Punjab gets credit 36k x 2 = 72k INR. Net = 8k INR
    # Pooled: Region net deviation = 0. Pooled penalty = 0 INR!
    # Savings = 8k INR!
    assert summary["total_savings_inr"] >= 0.0
    assert summary["total_pooled_penalty_inr"] <= summary["total_standalone_penalty_inr"]
    assert "NR" in res["by_region"]
    assert res["by_region"]["NR"]["error_cancellation_ratio"] > 0.9

