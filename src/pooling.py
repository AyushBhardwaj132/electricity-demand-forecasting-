"""
pooling.py
Pooling benefit analysis for TerraWatt's hierarchical forecasting system.

Measures how much DSM (Deviation Settlement Mechanism) penalty the hierarchical
forecasting approach saves compared to each state scheduling independently.
When states within a region pool their deviations, forecast errors between
neighbouring states partially cancel, reducing the net regional deviation and
the resulting financial penalty.

Key metric produced:
    Pooling Benefit_r = Σ Penalty_s^standalone  −  Penalty_r^hierarchy

References
----------
- CERC DSM Regulations (5th amendment), 2022
- POSOCO Scheduling & Despatch Code

Functions
---------
compute_standalone_penalties
    DSM penalties when each state uses its own forecast as schedule.
compute_hierarchy_penalties
    DSM penalties when regional deviations are pooled across member states.
compute_pooling_benefit
    Headline savings from hierarchical pooling, broken down by region and state.
compute_error_cancellation_ratio
    Scalar metric of how much errors cancel within a group.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.hierarchy import REGION_MEMBERS, REGIONS
from src.penalties import (
    DEFAULT_NORMAL_RATE,
    dsm_penalty_vector,
    total_dsm_penalty,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _energymet_to_node_id(region: str, col_name: str) -> str:
    """Convert a REGION_MEMBERS column name to the ``unique_id`` format.

    Examples
    --------
    >>> _energymet_to_node_id("NR", "Delhi: EnergyMet")
    'India/NR/Delhi'
    """
    state_name = col_name.replace(": EnergyMet", "")
    return f"India/{region}/{state_name}"


def _build_region_state_ids() -> Dict[str, List[str]]:
    """Map each region code to its list of ``unique_id``-format state IDs.

    Returns
    -------
    Dict[str, List[str]]
        ``{"NR": ["India/NR/Punjab", "India/NR/Haryana", ...], ...}``
    """
    return {
        region: [
            _energymet_to_node_id(region, col)
            for col in members
        ]
        for region, members in REGION_MEMBERS.items()
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_standalone_penalties(
    actuals_by_node: Dict[str, np.ndarray],
    forecasts_by_node: Dict[str, np.ndarray],
    normal_rate: float = DEFAULT_NORMAL_RATE,
) -> Dict[str, Dict[str, float]]:
    """Compute DSM penalties for each state scheduling independently.

    Each state uses its own forecast as its submitted schedule, and is
    penalised for the deviation between its actual demand and that schedule.

    Parameters
    ----------
    actuals_by_node : Dict[str, np.ndarray]
        ``{node_id: actual_mw_array}`` for every state-level node.
    forecasts_by_node : Dict[str, np.ndarray]
        ``{node_id: forecast_mw_array}`` (used as submitted schedule).
    normal_rate : float, optional
        Normal-rate tariff in INR / MWh (default: ``DEFAULT_NORMAL_RATE``).

    Returns
    -------
    Dict[str, Dict[str, float]]
        ``{node_id: {"total_penalty_inr", "mean_penalty", "num_blocks", ...}}``
    """
    results: Dict[str, Dict[str, float]] = {}

    for node_id in actuals_by_node:
        if node_id not in forecasts_by_node:
            continue

        actuals = np.asarray(actuals_by_node[node_id], dtype=np.float64)
        schedules = np.asarray(forecasts_by_node[node_id], dtype=np.float64)

        penalty_info = total_dsm_penalty(actuals, schedules, normal_rate)
        num_blocks = len(actuals)
        total_pen = float(penalty_info["total_penalty_inr"])

        results[node_id] = {
            "total_penalty_inr": total_pen,
            "mean_penalty": total_pen / num_blocks if num_blocks > 0 else 0.0,
            "num_blocks": num_blocks,
        }

    return results


def compute_hierarchy_penalties(
    actuals_by_node: Dict[str, np.ndarray],
    forecasts_by_node: Dict[str, np.ndarray],
    region_members: Dict[str, List[str]] | None = None,
    normal_rate: float = DEFAULT_NORMAL_RATE,
) -> Dict[str, Dict[str, float]]:
    """Compute DSM penalties when states pool deviations within a region.

    Regional deviation is calculated as:

        deviation_r(t) = Σ_s actual_s(t)  −  Σ_s schedule_s(t)

    Because individual state errors partially cancel, the regional penalty is
    typically smaller than the sum of state-level standalone penalties.

    The pooled penalty is allocated back to each member state proportionally
    to its share of total regional demand (energy-weighted allocation).

    Parameters
    ----------
    actuals_by_node : Dict[str, np.ndarray]
        ``{node_id: actual_mw_array}`` for every state-level node.
    forecasts_by_node : Dict[str, np.ndarray]
        ``{node_id: forecast_mw_array}`` (used as submitted schedule).
    region_members : Dict[str, List[str]] | None, optional
        ``{region_code: [node_id, ...]}`` override.  Falls back to the
        canonical mapping derived from :data:`src.hierarchy.REGION_MEMBERS`.
    normal_rate : float, optional
        Normal-rate tariff in INR / MWh.

    Returns
    -------
    Dict[str, Dict[str, float]]
        ``{node_id: {"pooled_penalty_inr", "demand_share", ...}}``
    """
    if region_members is None:
        region_members = _build_region_state_ids()
    else:
        # Normalize region_members keys/values if provided in EnergyMet format
        normalized = {}
        for reg, members in region_members.items():
            norm_members = []
            for m in members:
                if ": EnergyMet" in m or not m.startswith("India/"):
                    norm_members.append(_energymet_to_node_id(reg, m))
                else:
                    norm_members.append(m)
            normalized[reg] = norm_members
        region_members = normalized

    results: Dict[str, Dict[str, float]] = {}

    for region, state_ids in region_members.items():
        # Filter to states present in both dictionaries
        present_ids = [
            sid for sid in state_ids
            if sid in actuals_by_node and sid in forecasts_by_node
        ]
        if not present_ids:
            continue

        # Build regional aggregate arrays
        actuals_list = [
            np.asarray(actuals_by_node[sid], dtype=np.float64)
            for sid in present_ids
        ]
        schedules_list = [
            np.asarray(forecasts_by_node[sid], dtype=np.float64)
            for sid in present_ids
        ]

        regional_actuals = np.sum(actuals_list, axis=0)
        regional_schedules = np.sum(schedules_list, axis=0)

        # Regional pooled penalty
        penalty_info = total_dsm_penalty(
            regional_actuals, regional_schedules, normal_rate,
        )
        regional_penalty = float(penalty_info["total_penalty_inr"])

        # Demand-share allocation
        total_demand = sum(float(np.sum(np.abs(a))) for a in actuals_list)
        for sid, act_arr in zip(present_ids, actuals_list):
            state_demand = float(np.sum(np.abs(act_arr)))
            demand_share = state_demand / total_demand if total_demand > 0 else 0.0
            allocated = regional_penalty * demand_share

            results[sid] = {
                "pooled_penalty_inr": allocated,
                "demand_share": demand_share,
                "region": region,
                "regional_penalty_inr": regional_penalty,
            }

    return results


def compute_error_cancellation_ratio(
    errors_by_state: Dict[str, np.ndarray],
) -> float:
    """Measure how much neighbouring states' errors cancel within a region.

    .. math::

        \\text{ratio} = 1 - \\frac{\\sum_t |\\sum_s e_{s,t}|}
                                    {\\sum_t \\sum_s |e_{s,t}|}

    Parameters
    ----------
    errors_by_state : Dict[str, np.ndarray]
        ``{state_id: error_array}`` where ``error = actual - forecast``.

    Returns
    -------
    float
        * **0.0** → no cancellation (all errors same direction every block).
        * **1.0** → perfect cancellation (errors perfectly offset).
    """
    if not errors_by_state:
        return 0.0

    error_arrays = [np.asarray(e, dtype=np.float64) for e in errors_by_state.values()]

    # sum of absolute individual errors across time and states
    sum_abs_individual = sum(float(np.sum(np.abs(e))) for e in error_arrays)
    if sum_abs_individual == 0.0:
        return 0.0  # no errors at all → ratio undefined, return 0

    # absolute value of aggregated error per block, then sum over time
    aggregated_error = np.sum(error_arrays, axis=0)  # shape: (T,)
    abs_aggregated = float(np.sum(np.abs(aggregated_error)))

    return 1.0 - abs_aggregated / sum_abs_individual


def compute_pooling_benefit(
    actuals_by_node: Dict[str, np.ndarray],
    forecasts_by_node: Dict[str, np.ndarray],
    region_members: Dict[str, List[str]] | None = None,
    normal_rate: float = DEFAULT_NORMAL_RATE,
) -> Dict[str, Any]:
    """Compute headline savings from hierarchical pooling.

    Calls :func:`compute_standalone_penalties` and
    :func:`compute_hierarchy_penalties`, then derives the pooling benefit
    broken down by region and state.

    Parameters
    ----------
    actuals_by_node : Dict[str, np.ndarray]
        ``{node_id: actual_mw_array}`` for state-level nodes.
    forecasts_by_node : Dict[str, np.ndarray]
        ``{node_id: forecast_mw_array}`` (schedules).
    region_members : Dict[str, List[str]] | None, optional
        Override mapping; defaults to canonical hierarchy.
    normal_rate : float, optional
        Normal-rate tariff in INR / MWh.

    Returns
    -------
    Dict[str, Any]
        Structured result with ``"summary"``, ``"by_region"``, and
        ``"by_state"`` sections.  See module docstring for schema.
    """
    if region_members is None:
        region_members = _build_region_state_ids()
    else:
        # Normalize region_members keys/values if provided in EnergyMet format
        normalized = {}
        for reg, members in region_members.items():
            norm_members = []
            for m in members:
                if ": EnergyMet" in m or not m.startswith("India/"):
                    norm_members.append(_energymet_to_node_id(reg, m))
                else:
                    norm_members.append(m)
            normalized[reg] = norm_members
        region_members = normalized

    standalone = compute_standalone_penalties(
        actuals_by_node, forecasts_by_node, normal_rate,
    )
    pooled = compute_hierarchy_penalties(
        actuals_by_node, forecasts_by_node, region_members, normal_rate,
    )

    # ---- by_region ---------------------------------------------------------
    by_region: Dict[str, Dict[str, Any]] = {}
    total_standalone = 0.0
    total_pooled = 0.0

    for region, state_ids in region_members.items():
        present_ids = [
            sid for sid in state_ids
            if sid in standalone and sid in pooled
        ]
        if not present_ids:
            continue

        region_standalone_pen = sum(
            standalone[sid]["total_penalty_inr"] for sid in present_ids
        )
        region_pooled_pen = float(
            pooled[present_ids[0]]["regional_penalty_inr"]
        ) if present_ids else 0.0

        savings = region_standalone_pen - region_pooled_pen
        savings_pct = (savings / region_standalone_pen * 100.0
                       if region_standalone_pen > 0 else 0.0)

        # Error cancellation ratio for this region
        errors_by_state = {
            sid: (
                np.asarray(actuals_by_node[sid], dtype=np.float64)
                - np.asarray(forecasts_by_node[sid], dtype=np.float64)
            )
            for sid in present_ids
        }
        ecr = compute_error_cancellation_ratio(errors_by_state)

        by_region[region] = {
            "standalone_penalty_inr": region_standalone_pen,
            "pooled_penalty_inr": region_pooled_pen,
            "savings_inr": savings,
            "savings_pct": savings_pct,
            "error_cancellation_ratio": ecr,
            "num_states": len(present_ids),
        }

        total_standalone += region_standalone_pen
        total_pooled += region_pooled_pen

    # ---- by_state ----------------------------------------------------------
    by_state: Dict[str, Dict[str, float]] = {}
    for region, state_ids in region_members.items():
        for sid in state_ids:
            if sid not in standalone or sid not in pooled:
                continue
            standalone_pen = standalone[sid]["total_penalty_inr"]
            allocated_pen = pooled[sid]["pooled_penalty_inr"]
            by_state[sid] = {
                "standalone_penalty_inr": standalone_pen,
                "allocated_pooled_penalty_inr": allocated_pen,
                "savings_inr": standalone_pen - allocated_pen,
                "demand_share_in_region": pooled[sid]["demand_share"],
            }

    # ---- summary -----------------------------------------------------------
    total_savings = total_standalone - total_pooled
    summary = {
        "total_standalone_penalty_inr": total_standalone,
        "total_pooled_penalty_inr": total_pooled,
        "total_savings_inr": total_savings,
        "savings_pct": (
            total_savings / total_standalone * 100.0
            if total_standalone > 0 else 0.0
        ),
    }

    return {
        "summary": summary,
        "by_region": by_region,
        "by_state": by_state,
    }
