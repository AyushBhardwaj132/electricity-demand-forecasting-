"""
evaluate.py
Objective evaluation metrics for base and reconciled time series forecasts.
Supports MAE, RMSE, MAPE, WMAPE, Hierarchy Coherence Error, and DSM Penalty Cost.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, min_threshold: float = 1.0) -> float:
    """
    Mean Absolute Percentage Error.
    Returns NaN if actual values fall below min_threshold to avoid division by ~zero.
    """
    if np.mean(np.abs(y_true)) < min_threshold:
        return float("nan")
    nonzero_mask = np.abs(y_true) > 1e-5
    if not np.any(nonzero_mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)


def calculate_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error (WMAPE):
    WMAPE = sum(|y_true - y_pred|) / sum(|y_true|) * 100
    Robust against zero/near-zero actual demand.
    """
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def calculate_coherence_error(
    reconciled_df: pd.DataFrame,
    forecast_col: str,
    parent_child_map: Dict[str, List[str]]
) -> float:
    """
    Calculates the maximum absolute hierarchy coherence error across all parent-child relationships:
    max_{t, parent} | y_parent,t - sum_{child} y_child,t |
    """
    max_error = 0.0
    for parent, children in parent_child_map.items():
        parent_series = reconciled_df[reconciled_df["unique_id"] == parent].set_index("ds")[forecast_col]
        children_sum = reconciled_df[reconciled_df["unique_id"].isin(children)].groupby("ds")[forecast_col].sum()
        diff = (parent_series - children_sum).abs().max()
        if not np.isnan(diff):
            max_error = max(max_error, float(diff))
    return max_error


def evaluate_forecasts(
    df: pd.DataFrame,
    actual_col: str,
    forecast_col: str,
    parent_child_map: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Evaluates a forecast series against ground truth actuals across all nodes.

    Returns structured dict:
      {
        "MAE": ...,
        "RMSE": ...,
        "MAPE": ...,
        "WMAPE": ...,
        "coherence_error": ...
      }
    """
    valid = df.dropna(subset=[actual_col, forecast_col])
    y_true = valid[actual_col].values
    y_pred = valid[forecast_col].values

    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    wmape = calculate_wmape(y_true, y_pred)
    coh_err = calculate_coherence_error(df, forecast_col, parent_child_map)

    result = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4) if not np.isnan(mape) else None,
        "WMAPE": round(wmape, 4),
        "coherence_error": round(coh_err, 8)
    }

    return result


# ---------------------------------------------------------------------------
# DSM Penalty Metrics
# ---------------------------------------------------------------------------

def calculate_dsm_penalty_cost(
    y_true: np.ndarray,
    y_schedule: np.ndarray,
    normal_rate: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate total DSM penalty cost when y_schedule is submitted as the
    drawal schedule and y_true is the actual demand realized.

    Uses the CERC DSM 2024 penalty slabs from src.penalties.

    Parameters:
        y_true: Actual demand values (MW or MU)
        y_schedule: Submitted schedule / forecast values
        normal_rate: Normal Rate in INR/MWh (defaults to penalties.DEFAULT_NORMAL_RATE)

    Returns:
        Dictionary with:
            - total_penalty_inr: Net penalty cost in INR
            - mean_penalty_per_block: Average penalty per time block
            - total_over_drawal_penalty_inr: Total penalty from over-drawal events
            - total_under_drawal_credit_inr: Total credit from under-drawal events
            - num_over_drawal_blocks: Count of over-drawal time blocks
            - num_under_drawal_blocks: Count of under-drawal time blocks
            - penalty_per_mu: Penalty normalized by total actual demand
    """
    from src.penalties import total_dsm_penalty, DEFAULT_NORMAL_RATE

    nr = normal_rate if normal_rate is not None else DEFAULT_NORMAL_RATE
    result = total_dsm_penalty(y_true, y_schedule, normal_rate=nr)

    # Add intensity metric: penalty per MU of actual demand
    total_actual = float(np.sum(np.abs(y_true)))
    result["penalty_per_mu"] = (
        round(result["total_penalty_inr"] / total_actual, 2)
        if total_actual > 0 else 0.0
    )

    return result


def calculate_penalty_asymmetry_ratio(
    y_true: np.ndarray,
    y_schedule: np.ndarray,
    normal_rate: Optional[float] = None
) -> float:
    """
    Measures how asymmetric the realized penalties are.

    Ratio = over_drawal_penalty / (over_drawal_penalty + |under_drawal_credit|)

    Returns:
        Float in [0, 1]. Values > 0.5 indicate over-drawal dominates costs.
        Returns 0.5 if both are zero.
    """
    from src.penalties import total_dsm_penalty, DEFAULT_NORMAL_RATE

    nr = normal_rate if normal_rate is not None else DEFAULT_NORMAL_RATE
    result = total_dsm_penalty(y_true, y_schedule, normal_rate=nr)

    over_cost = abs(result.get("total_over_drawal_penalty_inr", 0.0))
    under_credit = abs(result.get("total_under_drawal_credit_inr", 0.0))
    total = over_cost + under_credit

    if total < 1e-10:
        return 0.5

    return round(over_cost / total, 4)
