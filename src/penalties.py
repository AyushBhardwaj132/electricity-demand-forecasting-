"""
penalties.py
DSM (Deviation Settlement Mechanism) penalty engine based on CERC DSM
Regulations 2024 for buyer (DISCOM / state) deviations from scheduled drawal.

Penalties are computed *marginally* across slabs — analogous to income-tax
brackets — so that each incremental percentage band of deviation attracts
its own rate multiplier applied to the Normal Rate (NR).

Key conventions
---------------
* **Over-drawal** (actual > scheduled): the buyer drew more than scheduled,
  i.e. *under-scheduled*.  Penalised at escalating multiples of NR.
* **Under-drawal** (actual < scheduled): the buyer drew less than scheduled,
  i.e. *over-scheduled*.  Credited at diminishing fractions of NR.
* ``penalty_inr`` is **positive** when the buyer owes money (over-drawal)
  and **negative** when the buyer receives credit (under-drawal).
"""

from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_NORMAL_RATE: float = 4000.0  # INR per MWh (₹4/kWh)

DEFAULT_PENALTY_SLABS: Dict[str, List[Dict]] = {
    "over_drawal": [
        {"min_pct": 0, "max_pct": 12, "rate_multiplier": 1.0},
        {"min_pct": 12, "max_pct": 20, "rate_multiplier": 1.5},
        {"min_pct": 20, "max_pct": 25, "rate_multiplier": 2.0},
        {"min_pct": 25, "max_pct": float("inf"), "rate_multiplier": 3.0},
    ],
    "under_drawal": [
        {"min_pct": 0, "max_pct": 12, "rate_multiplier": 0.9},
        {"min_pct": 12, "max_pct": 20, "rate_multiplier": 0.6},
        {"min_pct": 20, "max_pct": 25, "rate_multiplier": 0.3},
        {"min_pct": 25, "max_pct": float("inf"), "rate_multiplier": 0.0},
    ],
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _marginal_penalty(
    deviation_pct: float,
    slabs: List[Dict],
    normal_rate: float,
    deviation_mwh: float,
    scheduled_mwh: float,
) -> float:
    """Compute marginal (slab-wise) penalty amount in INR.

    Each slab is applied to its own band of deviation.  For example, with a
    15 % deviation the first 12 % is charged at the first slab's multiplier
    and the remaining 3 % at the second slab's multiplier.

    Parameters
    ----------
    deviation_pct:
        Absolute deviation as a percentage of scheduled (non-negative).
    slabs:
        Ordered list of slab dicts with ``min_pct``, ``max_pct``,
        ``rate_multiplier``.
    normal_rate:
        Normal Rate in INR / MWh.
    deviation_mwh:
        Absolute deviation in MWh (used only to derive energy from pct).
    scheduled_mwh:
        Scheduled energy in MWh (used to convert pct bands → MWh).

    Returns
    -------
    float
        Total penalty INR (always ≥ 0).
    """
    total = 0.0
    remaining_pct = deviation_pct

    for slab in slabs:
        if remaining_pct <= 0:
            break

        band_width = slab["max_pct"] - slab["min_pct"]
        applicable_pct = min(remaining_pct, band_width)

        # Convert percentage band back to MWh
        band_mwh = (applicable_pct / 100.0) * scheduled_mwh
        total += slab["rate_multiplier"] * normal_rate * band_mwh

        remaining_pct -= applicable_pct

    return total


# ---------------------------------------------------------------------------
# Core scalar function
# ---------------------------------------------------------------------------


def dsm_penalty(
    actual_mw: float,
    scheduled_mw: float,
    normal_rate: float = DEFAULT_NORMAL_RATE,
    penalty_slabs: Optional[Dict[str, List[Dict]]] = None,
) -> Dict[str, float]:
    """Calculate the DSM penalty / credit for a single 15-minute time block.

    Parameters
    ----------
    actual_mw:
        Actual metered drawal in MW for the block.
    scheduled_mw:
        Scheduled drawal in MW for the block.
    normal_rate:
        Normal Rate in INR / MWh.  Defaults to ``DEFAULT_NORMAL_RATE``.
    penalty_slabs:
        Custom slab structure; falls back to ``DEFAULT_PENALTY_SLABS``.

    Returns
    -------
    dict
        ``deviation_mw``  – signed (positive = over-drawal).
        ``deviation_pct`` – absolute, as % of scheduled.
        ``direction``     – ``'over_drawal'`` or ``'under_drawal'``.
        ``penalty_inr``   – net amount (positive = cost to buyer).
        ``penalty_rate_effective`` – effective blended ₹ / MWh rate.
    """
    if penalty_slabs is None:
        penalty_slabs = DEFAULT_PENALTY_SLABS

    deviation_mw: float = actual_mw - scheduled_mw

    # Handle zero-schedule edge case
    if scheduled_mw == 0.0:
        return {
            "deviation_mw": deviation_mw,
            "deviation_pct": 0.0 if deviation_mw == 0.0 else float("inf"),
            "direction": "over_drawal" if deviation_mw >= 0 else "under_drawal",
            "penalty_inr": 0.0,
            "penalty_rate_effective": 0.0,
        }

    deviation_pct = abs(deviation_mw) / scheduled_mw * 100.0

    if deviation_mw == 0.0:
        return {
            "deviation_mw": 0.0,
            "deviation_pct": 0.0,
            "direction": "over_drawal",
            "penalty_inr": 0.0,
            "penalty_rate_effective": 0.0,
        }

    # In a 15-minute block, MW ≈ MWh × 4.  But DSM charges are
    # conventionally stated per-MWh-of-deviation.  For a single time-block
    # calculation we treat the deviation in MW as the energy quantum (one
    # block = 0.25 h, but slabs are defined over % deviation so the
    # conversion is consistent as long as caller is consistent).
    abs_dev = abs(deviation_mw)

    if deviation_mw > 0:
        direction = "over_drawal"
        slabs = penalty_slabs["over_drawal"]
        raw_penalty = _marginal_penalty(
            deviation_pct, slabs, normal_rate, abs_dev, scheduled_mw
        )
        penalty_inr = raw_penalty  # buyer pays
    else:
        direction = "under_drawal"
        slabs = penalty_slabs["under_drawal"]
        raw_credit = _marginal_penalty(
            deviation_pct, slabs, normal_rate, abs_dev, scheduled_mw
        )
        penalty_inr = -raw_credit  # buyer receives credit (negative)

    effective_rate = abs(penalty_inr) / abs_dev if abs_dev > 0 else 0.0

    return {
        "deviation_mw": deviation_mw,
        "deviation_pct": deviation_pct,
        "direction": direction,
        "penalty_inr": penalty_inr,
        "penalty_rate_effective": effective_rate,
    }


# ---------------------------------------------------------------------------
# Vectorised function
# ---------------------------------------------------------------------------


def dsm_penalty_vector(
    actuals: np.ndarray,
    schedules: np.ndarray,
    normal_rate: float = DEFAULT_NORMAL_RATE,
    penalty_slabs: Optional[Dict[str, List[Dict]]] = None,
) -> Dict[str, np.ndarray]:
    """Vectorised DSM penalty across multiple time blocks.

    Iterates over blocks internally (marginal slab logic is not trivially
    vectorisable).  For large arrays consider batching in Cython / Numba in
    the future.

    Parameters
    ----------
    actuals:
        1-D array of actual MW values per block.
    schedules:
        1-D array of scheduled MW values per block (same length).
    normal_rate:
        Normal Rate in INR / MWh.
    penalty_slabs:
        Custom slab structure.

    Returns
    -------
    dict of np.ndarray
        Same keys as ``dsm_penalty`` but each value is an array aligned
        with the input blocks.
    """
    n = len(actuals)
    deviation_mw = np.empty(n, dtype=np.float64)
    deviation_pct = np.empty(n, dtype=np.float64)
    direction = np.empty(n, dtype=object)
    penalty_inr = np.empty(n, dtype=np.float64)
    penalty_rate_effective = np.empty(n, dtype=np.float64)

    for i in range(n):
        result = dsm_penalty(
            actual_mw=float(actuals[i]),
            scheduled_mw=float(schedules[i]),
            normal_rate=normal_rate,
            penalty_slabs=penalty_slabs,
        )
        deviation_mw[i] = result["deviation_mw"]
        deviation_pct[i] = result["deviation_pct"]
        direction[i] = result["direction"]
        penalty_inr[i] = result["penalty_inr"]
        penalty_rate_effective[i] = result["penalty_rate_effective"]

    return {
        "deviation_mw": deviation_mw,
        "deviation_pct": deviation_pct,
        "direction": direction,
        "penalty_inr": penalty_inr,
        "penalty_rate_effective": penalty_rate_effective,
    }


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------


def total_dsm_penalty(
    actuals: np.ndarray,
    schedules: np.ndarray,
    normal_rate: float = DEFAULT_NORMAL_RATE,
) -> Dict[str, float]:
    """Total DSM settlement across all time blocks.

    Parameters
    ----------
    actuals:
        1-D array of actual MW per block.
    schedules:
        1-D array of scheduled MW per block (same length).
    normal_rate:
        Normal Rate in INR / MWh.

    Returns
    -------
    dict
        ``total_penalty_inr``         – net across all blocks (signed).
        ``mean_penalty_per_block``    – average signed penalty per block.
        ``total_over_drawal_penalty`` – sum of positive penalties.
        ``total_under_drawal_credit`` – sum of negative credits (≤ 0).
        ``net_penalty``               – same as ``total_penalty_inr``.
    """
    vec = dsm_penalty_vector(actuals, schedules, normal_rate=normal_rate)
    penalties = vec["penalty_inr"]

    over_mask = penalties > 0
    under_mask = penalties < 0

    total = float(np.sum(penalties))
    return {
        "total_penalty_inr": total,
        "mean_penalty_per_block": total / len(penalties) if len(penalties) > 0 else 0.0,
        "total_over_drawal_penalty": float(np.sum(penalties[over_mask])),
        "total_under_drawal_credit": float(np.sum(penalties[under_mask])),
        "net_penalty": total,
    }


# ---------------------------------------------------------------------------
# Newsvendor optimal quantile
# ---------------------------------------------------------------------------


def optimal_schedule_quantile(
    normal_rate: float = DEFAULT_NORMAL_RATE,
    penalty_slabs: Optional[Dict[str, List[Dict]]] = None,
    deviation_pct: float = 10.0,
) -> float:
    """Compute the cost-minimising scheduling quantile (newsvendor ratio).

    For asymmetric penalty structures the optimal point forecast to submit
    as a schedule is **not** the expected value (mean) but a specific
    quantile of the demand distribution.

    The newsvendor ratio is::

        q* = c_over / (c_over + c_under)

    where ``c_over`` is the marginal cost per unit of over-drawal
    (under-scheduling) and ``c_under`` is the marginal *opportunity cost*
    per unit of under-drawal (over-scheduling) — i.e. the credit that is
    **not** received relative to full NR.

    At the default slabs for < 12 % deviation::

        c_over  = 1.0 × NR  →  4000
        c_under = (1.0 − 0.9) × NR = 0.1 × NR  →  400   (haircut)

    Wait — but the true economic cost of under-drawal is the *lost credit*.
    If you under-draw by 1 MWh you receive 0.9 × NR rather than selling at
    NR on the exchange, so the opportunity cost is ``(1 − credit_mult) × NR``.

    However for scheduling optimisation the relevant marginal costs are:
    * Over-drawal cost  = penalty multiplier × NR  (what you pay)
    * Under-drawal cost = credit multiplier × NR   (what you forgo by
      scheduling less, since the credit shrinks)

    So::

        q* = c_over / (c_over + c_under)

    where ``c_over = over_mult × NR`` and ``c_under = under_mult × NR``.

    Parameters
    ----------
    normal_rate:
        Normal Rate (used for consistency; cancels in the ratio).
    penalty_slabs:
        Custom slab structure.
    deviation_pct:
        Representative deviation percentage to select the applicable slab
        pair (default 10 %, which falls in the first band for both sides).

    Returns
    -------
    float
        Optimal quantile in [0, 1].  Values > 0.5 indicate a bias toward
        *over-scheduling* (submitting a schedule higher than the median
        demand) to avoid the costlier over-drawal penalties.
    """
    if penalty_slabs is None:
        penalty_slabs = DEFAULT_PENALTY_SLABS

    # Find the applicable slab for the given deviation_pct
    over_mult = 1.0
    for slab in penalty_slabs["over_drawal"]:
        if slab["min_pct"] <= deviation_pct < slab["max_pct"]:
            over_mult = slab["rate_multiplier"]
            break

    under_mult = 0.9
    for slab in penalty_slabs["under_drawal"]:
        if slab["min_pct"] <= deviation_pct < slab["max_pct"]:
            under_mult = slab["rate_multiplier"]
            break

    c_over = over_mult * normal_rate
    c_under = under_mult * normal_rate

    if c_over + c_under == 0:
        return 0.5

    return c_over / (c_over + c_under)
