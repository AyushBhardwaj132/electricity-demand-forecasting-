"""
scheduler.py
Schedule recommendation module based on cost-optimal newsvendor logic and Monte Carlo
optimization under asymmetric CERC DSM (Deviation Settlement Mechanism) penalties.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.stats import norm

from src.penalties import (
    dsm_penalty,
    dsm_penalty_vector,
    total_dsm_penalty,
    optimal_schedule_quantile,
    DEFAULT_NORMAL_RATE,
    DEFAULT_PENALTY_SLABS,
)


def compute_expected_penalty(
    actuals: np.ndarray,
    schedule: float,
    normal_rate: float = DEFAULT_NORMAL_RATE,
    penalty_slabs: Optional[Dict[str, List[Dict]]] = None,
    mode: str = "penalty_only",
) -> float:
    """
    Compute expected penalty or net deviation loss of submitting `schedule` against realized `actuals`.
    
    Modes:
      - 'penalty_only': Penalizes over-drawal at slab rates; penalizes under-drawal for unrecovered energy.
      - 'asymmetric_loss': Standard newsvendor loss where under-scheduling penalty > over-scheduling haircut.
    """
    if penalty_slabs is None:
        penalty_slabs = DEFAULT_PENALTY_SLABS

    # Dynamically extract marginal penalty multipliers from the provided slabs
    over_mult = penalty_slabs.get("over_drawal", [{}])[0].get("rate_multiplier", 1.0)
    under_mult = penalty_slabs.get("under_drawal", [{}])[0].get("rate_multiplier", 0.9)

    c_over = over_mult * normal_rate
    c_under = under_mult * normal_rate

    diff = actuals - schedule
    loss = np.where(diff > 0, c_over * diff, c_under * (-diff))
    return float(np.mean(loss))


def compute_optimal_schedule(
    point_forecast_mw: float,
    forecast_std_mw: float,
    normal_rate: float = DEFAULT_NORMAL_RATE,
    penalty_slabs: Optional[Dict[str, List[Dict]]] = None,
    n_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Given a point forecast and its uncertainty (std dev), compute the
    cost-optimal schedule to submit.

    Uses Monte Carlo simulation and grid search, alongside a closed-form
    newsvendor quantile approximation.
    """
    if point_forecast_mw < 0:
        point_forecast_mw = 0.0

    # Optimal quantile from newsvendor formulation
    q_star = optimal_schedule_quantile(normal_rate=normal_rate, penalty_slabs=penalty_slabs)
    
    # Closed-form approximation using normal distribution inverse CDF
    z_q = float(norm.ppf(q_star)) if forecast_std_mw > 1e-6 else 0.0
    schedule_approx = max(0.0, float(point_forecast_mw + z_q * forecast_std_mw))

    if forecast_std_mw <= 1e-6:
        # Zero uncertainty case
        return {
            "point_forecast_mw": round(float(point_forecast_mw), 2),
            "forecast_std_mw": round(float(forecast_std_mw), 2),
            "optimal_schedule_mw": round(float(point_forecast_mw), 2),
            "schedule_approx_mw": round(float(point_forecast_mw), 2),
            "schedule_bias_mw": 0.0,
            "schedule_bias_pct": 0.0,
            "expected_penalty_at_optimal_inr": 0.0,
            "expected_penalty_at_forecast_inr": 0.0,
            "penalty_savings_inr": 0.0,
            "penalty_savings_pct": 0.0,
            "optimal_quantile": round(float(q_star), 4),
            "bias_direction": "neutral",
        }

    # Monte Carlo simulation of actual realized demand scenarios
    rng = np.random.default_rng(seed)
    actual_scenarios = rng.normal(loc=point_forecast_mw, scale=forecast_std_mw, size=n_samples)
    actual_scenarios = np.clip(actual_scenarios, 0.0, None)

    # Grid search around point forecast: [-1*std, +2*std] with dense candidate points
    lower_bound = max(0.0, point_forecast_mw - 1.0 * forecast_std_mw)
    upper_bound = point_forecast_mw + 2.0 * forecast_std_mw
    candidate_schedules = np.linspace(lower_bound, upper_bound, 61)

    # Ensure point_forecast and schedule_approx are evaluated
    candidate_schedules = np.unique(np.append(candidate_schedules, [point_forecast_mw, schedule_approx]))

    best_schedule = schedule_approx
    min_penalty = compute_expected_penalty(actual_scenarios, best_schedule, normal_rate=normal_rate, penalty_slabs=penalty_slabs)

    # Evaluate candidates
    for sched in candidate_schedules:
        pen = compute_expected_penalty(actual_scenarios, sched, normal_rate=normal_rate, penalty_slabs=penalty_slabs)
        if pen < min_penalty:
            min_penalty = pen
            best_schedule = sched

    # Penalty at baseline point forecast
    base_penalty = compute_expected_penalty(actual_scenarios, point_forecast_mw, normal_rate=normal_rate, penalty_slabs=penalty_slabs)

    savings_inr = max(0.0, base_penalty - min_penalty)
    savings_pct = (savings_inr / base_penalty * 100.0) if base_penalty > 1e-6 else 0.0

    bias_mw = float(best_schedule - point_forecast_mw)
    bias_pct = (bias_mw / point_forecast_mw * 100.0) if point_forecast_mw > 0 else 0.0

    if abs(bias_mw) < 1e-4:
        bias_dir = "neutral"
    elif bias_mw > 0:
        bias_dir = "over_schedule"
    else:
        bias_dir = "under_schedule"

    return {
        "point_forecast_mw": round(float(point_forecast_mw), 2),
        "forecast_std_mw": round(float(forecast_std_mw), 2),
        "optimal_schedule_mw": round(float(best_schedule), 2),
        "schedule_approx_mw": round(float(schedule_approx), 2),
        "schedule_bias_mw": round(bias_mw, 2),
        "schedule_bias_pct": round(bias_pct, 2),
        "expected_penalty_at_optimal_inr": round(float(min_penalty), 2),
        "expected_penalty_at_forecast_inr": round(float(base_penalty), 2),
        "penalty_savings_inr": round(float(savings_inr), 2),
        "penalty_savings_pct": round(float(savings_pct), 2),
        "optimal_quantile": round(float(q_star), 4),
        "bias_direction": bias_dir,
    }


def compute_schedule_recommendations(
    forecasts: Dict[str, float],
    forecast_stds: Dict[str, float],
    normal_rate: float = DEFAULT_NORMAL_RATE,
) -> Dict[str, Dict]:
    """
    Batch schedule recommendations for multiple hierarchy nodes.
    """
    recommendations = {}
    for node_id, fcst in forecasts.items():
        std_mw = forecast_stds.get(node_id, fcst * 0.05)  # default 5% std if not provided
        recommendations[node_id] = compute_optimal_schedule(
            point_forecast_mw=fcst,
            forecast_std_mw=std_mw,
            normal_rate=normal_rate,
        )
    return recommendations


def estimate_forecast_uncertainty(
    backtest_errors: np.ndarray,
    method: str = "empirical_std",
) -> float:
    """
    Estimate forecast uncertainty (std dev) from historical backtest residuals.

    Methods:
        - 'empirical_std': standard deviation
        - 'mad': Median Absolute Deviation scaled by 1.4826
        - 'iqr': Interquartile Range divided by 1.349
    """
    clean_errors = backtest_errors[~np.isnan(backtest_errors)]
    if len(clean_errors) < 2:
        return 1.0

    if method == "empirical_std":
        return float(np.std(clean_errors, ddof=1))
    elif method == "mad":
        median = np.median(clean_errors)
        mad = np.median(np.abs(clean_errors - median))
        return float(mad * 1.4826)
    elif method == "iqr":
        q75, q25 = np.percentile(clean_errors, [75, 25])
        return float((q75 - q25) / 1.349)
    else:
        return float(np.std(clean_errors, ddof=1))
