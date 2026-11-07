"""
tests/test_penalties.py
Unit tests for CERC DSM 2024 penalty calculation functions.
"""

import pytest
import numpy as np
from src.penalties import (
    dsm_penalty,
    dsm_penalty_vector,
    total_dsm_penalty,
    optimal_schedule_quantile,
    DEFAULT_NORMAL_RATE,
)


def test_zero_deviation():
    res = dsm_penalty(actual_mw=100.0, scheduled_mw=100.0)
    assert res["deviation_mw"] == 0.0
    assert res["penalty_inr"] == 0.0
    assert res["penalty_rate_effective"] == 0.0


def test_small_over_drawal():
    # 5% over-drawal: within 0-12% band -> rate multiplier 1.0 * NR
    res = dsm_penalty(actual_mw=105.0, scheduled_mw=100.0, normal_rate=4000.0)
    assert res["direction"] == "over_drawal"
    assert res["deviation_mw"] == 5.0
    # 5 MWh * 1.0 * 4000 = 20,000 INR
    assert pytest.approx(res["penalty_inr"], 0.1) == 20000.0
    assert pytest.approx(res["penalty_rate_effective"], 0.1) == 4000.0


def test_medium_over_drawal():
    # 15% over-drawal: 12% in slab 1 (1.0x), 3% in slab 2 (1.5x)
    # 12 MWh * 4000 = 48,000; 3 MWh * 1.5 * 4000 = 18,000. Total = 66,000 INR
    res = dsm_penalty(actual_mw=115.0, scheduled_mw=100.0, normal_rate=4000.0)
    assert res["direction"] == "over_drawal"
    assert pytest.approx(res["penalty_inr"], 0.1) == 66000.0


def test_under_drawal_credit():
    # 5% under-drawal: within 0-12% band -> credit multiplier 0.9 * NR
    # 5 MWh * 0.9 * 4000 = 18,000 INR credit (penalty = -18,000)
    res = dsm_penalty(actual_mw=95.0, scheduled_mw=100.0, normal_rate=4000.0)
    assert res["direction"] == "under_drawal"
    assert res["deviation_mw"] == -5.0
    assert pytest.approx(res["penalty_inr"], 0.1) == -18000.0


def test_penalty_asymmetry():
    # +10% over-drawal vs -10% under-drawal
    over = dsm_penalty(actual_mw=110.0, scheduled_mw=100.0, normal_rate=4000.0)
    under = dsm_penalty(actual_mw=90.0, scheduled_mw=100.0, normal_rate=4000.0)
    # Over-drawal costs 40,000 INR, under-drawal returns credit of 36,000 INR
    assert abs(over["penalty_inr"]) > abs(under["penalty_inr"])


def test_vectorized_penalty():
    actuals = np.array([100.0, 105.0, 95.0])
    schedules = np.array([100.0, 100.0, 100.0])
    vec = dsm_penalty_vector(actuals, schedules, normal_rate=4000.0)
    assert pytest.approx(vec["penalty_inr"][0], 0.1) == 0.0
    assert pytest.approx(vec["penalty_inr"][1], 0.1) == 20000.0
    assert pytest.approx(vec["penalty_inr"][2], 0.1) == -18000.0


def test_total_dsm_penalty():
    actuals = np.array([105.0, 95.0])
    schedules = np.array([100.0, 100.0])
    tot = total_dsm_penalty(actuals, schedules, normal_rate=4000.0)
    # Net: 20,000 - 18,000 = 2,000 INR
    assert pytest.approx(tot["total_penalty_inr"], 0.1) == 2000.0
    assert pytest.approx(tot["total_over_drawal_penalty"], 0.1) == 20000.0
    assert pytest.approx(tot["total_under_drawal_credit"], 0.1) == -18000.0


def test_optimal_schedule_quantile():
    q = optimal_schedule_quantile(normal_rate=4000.0)
    # With 1.0 vs 0.9, q* = 1.0 / 1.9 ≈ 0.5263
    assert 0.5 < q < 0.6

