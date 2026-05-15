"""Constraint repair helpers for optimizer starting points.

This module intentionally contains no candidate portfolio construction or
ranking. It only repairs a weight vector into hard asset/group caps so SLSQP can
start from a feasible point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _constraint_inputs(index: pd.Index, constraints: dict[str, object]) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    asset_caps = pd.Series(constraints.get("asset_max_weights", {}), dtype=float)
    asset_caps.index = pd.Index([str(t) for t in asset_caps.index], name="ticker")
    asset_caps = asset_caps.reindex(index).fillna(1.0).clip(lower=0.0)
    group_map = pd.Series(constraints.get("group_map", {}), dtype=object)
    group_map.index = pd.Index([str(t) for t in group_map.index], name="ticker")
    group_map = group_map.reindex(index)
    group_limits = {str(k): float(v) for k, v in dict(constraints.get("group_limits", {})).items()}
    return asset_caps, group_map, group_limits


def _constraint_validation(weights: pd.Series, constraints: dict[str, object], tolerance: float) -> dict[str, object]:
    index = pd.Index([str(t) for t in weights.index], name="ticker")
    cleaned = weights.reindex(index).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    asset_caps, group_map, group_limits = _constraint_inputs(index, constraints)
    errors: list[str] = []
    asset_violations: list[dict[str, object]] = []
    group_violations: list[dict[str, object]] = []

    if (cleaned < -tolerance).any():
        errors.append("negative_weights")
    for ticker, weight in cleaned.items():
        limit = float(asset_caps.get(ticker, 1.0))
        if float(weight) > limit + tolerance:
            asset_violations.append(
                {
                    "ticker": str(ticker),
                    "actual_weight": float(weight),
                    "limit": limit,
                    "excess": float(weight) - limit,
                }
            )
    if asset_violations:
        errors.append("asset_limit")

    for group, tickers in group_map.groupby(group_map).groups.items():
        group_name = str(group)
        if group_name not in group_limits:
            continue
        group_weight = float(cleaned[list(tickers)].sum())
        limit = float(group_limits[group_name])
        if group_weight > limit + tolerance:
            group_violations.append(
                {
                    "group": group_name,
                    "actual_weight": group_weight,
                    "limit": limit,
                    "excess": group_weight - limit,
                }
            )
    if group_violations:
        errors.append("group_limit")
    return {
        "ok": not errors,
        "errors": errors,
        "asset_limit_violations": asset_violations,
        "group_limit_violations": group_violations,
    }


def _capacity(weights: pd.Series, constraints: dict[str, object]) -> pd.Series:
    index = pd.Index([str(t) for t in weights.index], name="ticker")
    asset_caps, group_map, group_limits = _constraint_inputs(index, constraints)
    capacities = (asset_caps - weights.reindex(index).fillna(0.0)).clip(lower=0.0)
    for group, tickers in group_map.groupby(group_map).groups.items():
        group_name = str(group)
        if group_name not in group_limits:
            continue
        group_free = max(float(group_limits[group_name]) - float(weights[list(tickers)].sum()), 0.0)
        capacities.loc[list(tickers)] = np.minimum(capacities.loc[list(tickers)].to_numpy(dtype=float), group_free)
    return capacities.astype(float)


def _redistribute_with_capacity(
    weights: pd.Series,
    amount: float,
    constraints: dict[str, object],
    preferred: list[str],
    tolerance: float,
) -> tuple[pd.Series, float]:
    repaired = weights.astype(float).copy()
    remaining = max(float(amount), 0.0)
    if remaining <= tolerance:
        return repaired, 0.0
    ordered = [ticker for ticker in preferred if ticker in repaired.index]
    ordered.extend([str(ticker) for ticker in repaired.index if str(ticker) not in ordered])
    for ticker in ordered:
        if remaining <= tolerance:
            break
        capacities = _capacity(repaired, constraints)
        free = float(capacities.get(ticker, 0.0))
        if free <= tolerance:
            continue
        add = min(free, remaining)
        repaired.loc[ticker] += add
        remaining -= add
    return repaired, remaining


def repair_weights_to_constraints(
    weights: pd.Series,
    constraints: dict[str, object],
    tolerance: float = 1e-8,
) -> dict[str, object]:
    """Repair long-only weights to configured asset and group caps."""

    index = pd.Index([str(t) for t in weights.index], name="ticker")
    repaired = weights.reindex(index).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if float(repaired.sum()) > 1.0 + tolerance:
        repaired = repaired / float(repaired.sum())
    asset_caps, group_map, group_limits = _constraint_inputs(index, constraints)
    cash_ticker = str(constraints.get("effective_cash_ticker") or constraints.get("cash_ticker") or "")
    defensive_groups = set(map(str, constraints.get("defensive_groups", [])))
    preferred = [ticker for ticker in [cash_ticker, "SGOV", "SHY", "IEF", "AGG"] if ticker]
    preferred.extend(
        str(ticker)
        for ticker, group in group_map.items()
        if str(group) in defensive_groups and str(ticker) not in preferred
    )
    preferred.extend([str(ticker) for ticker in index if str(ticker) not in preferred])
    freed = 0.0

    for _ in range(6):
        changed = False
        over_asset = repaired - asset_caps
        for ticker, excess in over_asset[over_asset > tolerance].items():
            reduction = float(excess)
            repaired.loc[ticker] -= reduction
            freed += reduction
            changed = True

        for group, tickers in group_map.groupby(group_map).groups.items():
            group_name = str(group)
            if group_name not in group_limits:
                continue
            group_weight = float(repaired[list(tickers)].sum())
            limit = float(group_limits[group_name])
            if group_weight <= limit + tolerance:
                continue
            excess = group_weight - limit
            group_weights = repaired[list(tickers)].clip(lower=0.0)
            denominator = max(float(group_weights.sum()), tolerance)
            reduction = group_weights * (excess / denominator)
            repaired.loc[list(tickers)] = (repaired.loc[list(tickers)] - reduction).clip(lower=0.0)
            freed += float(reduction.sum())
            changed = True

        if freed > tolerance:
            repaired, leftover = _redistribute_with_capacity(repaired, freed, constraints, preferred, tolerance)
            freed = leftover
            changed = True
        if not changed:
            break

    total = float(repaired.sum())
    if total < 1.0 - tolerance:
        repaired, leftover = _redistribute_with_capacity(repaired, 1.0 - total, constraints, preferred, tolerance)
        freed = max(freed, leftover)
    validation = _constraint_validation(repaired, constraints, tolerance)
    repair_possible = bool(validation["ok"] and float(repaired.sum()) >= 1.0 - 1e-6 and freed <= 1e-6)
    return {
        "weights": repaired.clip(lower=0.0),
        "valid": bool(validation["ok"]),
        "errors": list(validation["errors"]),
        "asset_limit_violations": list(validation["asset_limit_violations"]),
        "group_limit_violations": list(validation["group_limit_violations"]),
        "repair_possible": repair_possible,
        "unallocated_weight": max(1.0 - float(repaired.sum()), 0.0),
    }
