"""
reconcile.py
Hierarchical time series reconciliation layer.

Implements mathematical reconciliation methods (Bottom-Up, Top-Down, MinT-Shrink, MinT-Cov)
to guarantee hierarchy coherence: parent forecast == sum(children forecasts).

Mathematics of MinT (Minimum Trace):
  Reconciled forecasts Y_tilde are given by:
    Y_tilde = S * (S^T * W^-1 * S)^-1 * S^T * W^-1 * Y_hat
  where:
    S = Summing matrix (n_total x n_bottom)
    Y_hat = Unreconciled base forecasts vector (n_total x 1)
    W = In-sample residual covariance matrix (n_total x n_total)
    In MinT-Shrink, W is estimated using Ledoit-Wolf shrinkage to ensure positive-definiteness.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, TopDown, MinTrace


def reconcile_with_hierarchicalforecast(
    Y_hat_df: pd.DataFrame,
    S_df: pd.DataFrame,
    tags: dict,
    Y_df_train: Optional[pd.DataFrame] = None,
    methods: Optional[List] = None
) -> pd.DataFrame:
    """
    Reconciles base forecasts using hierarchicalforecast package implementations.

    Parameters:
      Y_hat_df: DataFrame containing ['unique_id', 'ds', 'y_hat']
      S_df: Summing matrix DataFrame produced by aggregate()
      tags: Hierarchy level tags dictionary
      Y_df_train: Training DataFrame containing in-sample actuals (needed for MinT covariance estimation)
      methods: List of reconciliation method instances. Defaults to BottomUp, TopDown, MinTrace(mint_shrink).

    Returns:
      reconciled_df: DataFrame with base and reconciled forecast columns.
    """
    if methods is None:
        methods = [
            BottomUp(),
            TopDown(method="forecast_proportions"),
            MinTrace(method="mint_shrink"),
        ]

    reconciler = HierarchicalReconciliation(reconcilers=methods)

    df_forecast = Y_hat_df.copy()
    if "y_hat" in df_forecast.columns and "LGBM" not in df_forecast.columns:
        df_forecast = df_forecast.rename(columns={"y_hat": "LGBM"})

    reconciled_df = reconciler.reconcile(
        Y_hat_df=df_forecast,
        tags=tags,
        S_df=S_df,
        Y_df=Y_df_train
    )

    return reconciled_df


def reconcile_matrix_bottom_up(
    y_hat_dict: Dict[str, np.ndarray],
    S_df: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """
    Direct matrix implementation of Bottom-Up reconciliation:
    Y_tilde = S * Y_bottom
    """
    bottom_uids = S_df.columns.tolist()
    all_uids = S_df.index.tolist()

    horizon = len(next(iter(y_hat_dict.values())))
    Y_bottom = np.vstack([y_hat_dict[uid] for uid in bottom_uids])

    S_matrix = S_df.values
    Y_reconciled = S_matrix @ Y_bottom

    reconciled_dict = {}
    for idx, uid in enumerate(all_uids):
        reconciled_dict[uid] = Y_reconciled[idx, :]

    return reconciled_dict


def verify_coherence(
    reconciled_df: pd.DataFrame,
    parent_uid: str,
    child_uids: List[str],
    forecast_col: str,
    tolerance: float = 1e-6
) -> Tuple[bool, float]:
    """
    Verifies parent forecast equals sum of child forecasts across all timestamps.

    Returns:
      (is_coherent, max_absolute_error)
    """
    parent = reconciled_df[reconciled_df["unique_id"] == parent_uid].set_index("ds")[forecast_col]
    children = reconciled_df[reconciled_df["unique_id"].isin(child_uids)]
    children_sum = children.groupby("ds")[forecast_col].sum()

    diff = (parent - children_sum).abs()
    max_error = float(diff.max())
    is_coherent = max_error < tolerance

    return is_coherent, max_error


# ---------------------------------------------------------------------------
# Cost-Aware Hierarchical Reconciliation
# ---------------------------------------------------------------------------

def reconcile_cost_aware(
    reconciled_df: pd.DataFrame,
    S_df: pd.DataFrame,
    base_col: str = "LGBM/MinTrace_method-mint_shrink",
    normal_rate: Optional[float] = None,
    uncertainty_dict: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Cost-Aware Hierarchical Reconciliation under India CERC DSM deviation penalties.

    Takes mathematically coherent MinT-Shrink forecasts and adjusts the bottom-level
    forecasts with the cost-optimal scheduling quantile bias to protect against asymmetric
    penalties, then projects back through the Summing Matrix S:
        Y_cost_aware = S @ Y_cost_aware_bottom

    This mathematically guarantees:
    1. Exact hierarchy sum coherence (coherence error == 0.0)
    2. Bottom-level and aggregate schedules are protected against expensive over-drawal deviations.

    Parameters:
      reconciled_df: DataFrame with reconciled forecasts (must contain unique_id, ds, and base_col)
      S_df: Summing matrix DataFrame (columns = bottom level nodes, index = all nodes)
      base_col: Source reconciled forecast column to adjust (default: MinT-Shrink)
      normal_rate: Normal Rate in INR/MWh
      uncertainty_dict: Optional {unique_id: std_dev_mw} dictionary
    """
    from src.penalties import optimal_schedule_quantile, DEFAULT_NORMAL_RATE
    from scipy.stats import norm

    nr = normal_rate if normal_rate is not None else DEFAULT_NORMAL_RATE
    q_star = optimal_schedule_quantile(normal_rate=nr)
    z_q = float(norm.ppf(q_star))  # ~0.066 for q* ≈ 0.5263

    df_out = reconciled_df.copy()
    cost_aware_col = "CostAware_MinT"

    # S_df may have 'unique_id' as a column or as the index
    if "unique_id" in S_df.columns:
        all_uids = S_df["unique_id"].tolist()
        bottom_uids = [c for c in S_df.columns if c != "unique_id"]
        S_matrix = S_df[bottom_uids].values.astype(np.float64)
    else:
        all_uids = S_df.index.tolist()
        bottom_uids = S_df.columns.tolist()
        S_matrix = S_df.values.astype(np.float64)

    timestamps = sorted(df_out["ds"].unique())

    # Build matrix of bottom forecasts: shape (n_bottom, n_timestamps)
    pivot_df = df_out.pivot(index="unique_id", columns="ds", values=base_col)
    
    # Ensure bottom nodes are present
    missing_bottom = [b for b in bottom_uids if b not in pivot_df.index]
    if missing_bottom:
        # Fallback if pivot doesn't match
        df_out[cost_aware_col] = df_out[base_col]
        return df_out

    Y_bottom = pivot_df.loc[bottom_uids, timestamps].values.astype(np.float64)

    # Apply asymmetric newsvendor shift to bottom nodes:
    # shift = z_q * std_dev
    shifts = np.zeros(len(bottom_uids))
    for idx, b_uid in enumerate(bottom_uids):
        if uncertainty_dict and b_uid in uncertainty_dict:
            std_val = uncertainty_dict[b_uid]
        else:
            # Default ~3% standard deviation relative to mean demand
            mean_demand = np.mean(Y_bottom[idx, :])
            std_val = max(1.0, 0.03 * mean_demand)
        shifts[idx] = z_q * std_val

    # Broadcast shift across timestamps
    Y_bottom_cost_aware = Y_bottom + shifts[:, np.newaxis]

    # Re-project through summing matrix S to guarantee exact coherence:
    # Y_all = S @ Y_bottom_cost_aware
    Y_all_cost_aware = S_matrix @ Y_bottom_cost_aware

    # Flatten back into dictionary for fast lookup
    cost_aware_map = {}
    for row_idx, uid in enumerate(all_uids):
        for col_idx, ts in enumerate(timestamps):
            cost_aware_map[(uid, ts)] = float(Y_all_cost_aware[row_idx, col_idx])

    # Assign column
    df_out[cost_aware_col] = [
        cost_aware_map.get((row["unique_id"], row["ds"]), row[base_col])
        for _, row in df_out.iterrows()
    ]

    return df_out
