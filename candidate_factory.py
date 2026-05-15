"""Candidate portfolio generation for the daily decision bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from optimizer import build_feasible_initial_weights


@dataclass(slots=True)
class CandidatePortfolio:
    """A named candidate allocation considered by the robust scorer."""

    name: str
    weights: pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize(weights: pd.Series) -> pd.Series:
    aligned = weights.astype(float).fillna(0.0).clip(lower=0.0)
    total = float(aligned.sum())
    if total <= 0.0:
        aligned[:] = 1.0 / len(aligned)
        return aligned
    return aligned / total


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
    asset_metadata: dict[str, object] | None = None,
    tolerance: float = 1e-8,
) -> dict[str, object]:
    """Repair long-only weights to configured asset and group caps.

    The function is intentionally deterministic and conservative: it clips
    violations, then redistributes freed weight only to assets with remaining
    asset and group capacity. If no capacity remains, it reports that the repair
    is incomplete instead of silently re-normalizing into a new violation.
    """

    del asset_metadata  # reserved for future richer asset-class preferences
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


def _defensive_cash_weights(index: pd.Index, params: dict[str, object]) -> pd.Series:
    defensive = pd.Series(params.get("defensive_weights", {}), dtype=float)
    defensive.index = pd.Index([str(t) for t in defensive.index], name="ticker")
    defensive = defensive.reindex(index).fillna(0.0)
    if float(defensive.sum()) <= 0.0:
        defensive = build_feasible_initial_weights(index.tolist(), params)
    return _normalize(defensive)


def _greedy_score_weights(
    score: pd.Series,
    index: pd.Index,
    params: dict[str, object],
) -> pd.Series:
    """Allocate weights greedily by descending score while respecting static limits best-effort."""

    score = score.reindex(index).fillna(0.0).astype(float)
    score = score.where(score > 0.0, 0.0)
    if float(score.sum()) <= 0.0:
        return _defensive_cash_weights(index, params)

    asset_caps = pd.Series(params.get("asset_max_weights", {}), dtype=float).reindex(index).fillna(0.0)
    group_map = pd.Series(params.get("group_map", {}), dtype=object).reindex(index)
    group_limits = {str(k): float(v) for k, v in dict(params.get("group_limits", {})).items()}
    weights = pd.Series(0.0, index=index, dtype=float)
    remaining = 1.0

    for ticker in score.sort_values(ascending=False).index:
        group = str(group_map.get(ticker, ""))
        group_used = float(weights[group_map[group_map == group].index].sum()) if group else 0.0
        group_cap = max(group_limits.get(group, 1.0) - group_used, 0.0)
        cap = min(float(asset_caps.get(ticker, 0.0)), group_cap, remaining)
        if cap <= 0.0:
            continue
        desired = remaining * float(score.loc[ticker] / max(score.sum(), 1e-12))
        alloc = min(cap, desired)
        if alloc <= 0.0:
            continue
        weights.loc[ticker] += alloc
        remaining = max(remaining - alloc, 0.0)
        if remaining <= 1e-8:
            break

    if remaining > 1e-8:
        fallback = build_feasible_initial_weights(index.tolist(), params)
        preferred = fallback.sort_values(ascending=False).index.astype(str).tolist()
        weights, leftover = _redistribute_with_capacity(weights, remaining, params, preferred, 1e-8)
        if leftover > 1e-8:
            repair_result = repair_weights_to_constraints(weights.add(fallback * leftover, fill_value=0.0), params)
            weights = pd.Series(repair_result["weights"], dtype=float).reindex(index).fillna(0.0)
    repair_result = repair_weights_to_constraints(weights, params)
    repaired = pd.Series(repair_result["weights"], dtype=float).reindex(index).fillna(0.0)
    return repaired if float(repaired.sum()) > 0.0 else _defensive_cash_weights(index, params)


def build_candidate_portfolios(
    w_current: pd.Series,
    w_target: pd.Series,
    forecast_table: pd.DataFrame,
    params: dict[str, object],
    conditional_factor_target: pd.Series | None = None,
) -> dict[str, CandidatePortfolio]:
    """Build the required portfolio alternatives for the daily bot."""

    ordered = list(dict.fromkeys([*w_current.index.tolist(), *w_target.index.tolist(), *forecast_table.index.tolist()]))
    index = pd.Index([str(t) for t in ordered], name="ticker")
    current = _normalize(w_current.reindex(index).fillna(0.0))
    target = _normalize(w_target.reindex(index).fillna(0.0))
    defensive = _defensive_cash_weights(index, params)
    signal_score = (
        forecast_table.get("expected_return_3m", pd.Series(0.0, index=index))
        * forecast_table.get("signal_confidence", pd.Series(1.0, index=index))
    )
    momentum_tilt = _greedy_score_weights(signal_score, index, params)
    momentum_repair = repair_weights_to_constraints(momentum_tilt, params)
    momentum_repaired = pd.Series(momentum_repair["weights"], dtype=float).reindex(index).fillna(0.0)

    partial_25 = _normalize(current + 0.25 * (target - current))
    partial_50 = _normalize(current + 0.50 * (target - current))
    min_turnover_repair = repair_weights_to_constraints(partial_25, params)
    compliance_repair = repair_weights_to_constraints(current, params)

    def repaired_metadata(kind: str, source: str, repair_result: dict[str, object]) -> dict[str, Any]:
        return {
            "kind": kind,
            "repaired_from": source,
            "repair_possible": bool(repair_result.get("repair_possible", False)),
            "repair_errors": "; ".join(map(str, repair_result.get("errors", []))),
            "score_lost_due_to_repair": np.nan,
        }

    candidates: dict[str, CandidatePortfolio] = {
        "HOLD": CandidatePortfolio(name="HOLD", weights=current, metadata={"kind": "hold"}),
        "SCENARIO_WEIGHTED_RF_SHARPE_OPTIMAL": CandidatePortfolio(
            name="SCENARIO_WEIGHTED_RF_SHARPE_OPTIMAL",
            weights=target,
            metadata={"kind": "scenario_weighted_solver", "final_solver_source": True},
        ),
        "DIRECT_SCENARIO_OPTIMIZER": CandidatePortfolio(
            name="DIRECT_SCENARIO_OPTIMIZER",
            weights=target,
            metadata={
                "kind": "direct_optimizer_legacy_alias",
                "final_solver_source": False,
                "legacy_alias_for": "SCENARIO_WEIGHTED_RF_SHARPE_OPTIMAL",
            },
        ),
        "OPTIMIZER_TARGET": CandidatePortfolio(name="OPTIMIZER_TARGET", weights=target, metadata={"kind": "optimizer"}),
        "PARTIAL_25": CandidatePortfolio(name="PARTIAL_25", weights=partial_25, metadata={"kind": "partial", "fraction": 0.25}),
        "PARTIAL_50": CandidatePortfolio(name="PARTIAL_50", weights=partial_50, metadata={"kind": "partial", "fraction": 0.50}),
        "DEFENSIVE_CASH": CandidatePortfolio(name="DEFENSIVE_CASH", weights=defensive, metadata={"kind": "defensive"}),
        "MOMENTUM_TILT_SIMPLE": CandidatePortfolio(name="MOMENTUM_TILT_SIMPLE", weights=momentum_tilt, metadata={"kind": "momentum_tilt"}),
        "MOMENTUM_TILT_REPAIRED": CandidatePortfolio(
            name="MOMENTUM_TILT_REPAIRED",
            weights=momentum_repaired,
            metadata=repaired_metadata("momentum_tilt_repaired", "MOMENTUM_TILT_SIMPLE", momentum_repair),
        ),
        "MOMENTUM_TILT_CAP_AWARE": CandidatePortfolio(
            name="MOMENTUM_TILT_CAP_AWARE",
            weights=momentum_repaired,
            metadata=repaired_metadata("momentum_tilt_cap_aware", "MOMENTUM_TILT_SIMPLE", momentum_repair),
        ),
        "MIN_TURNOVER_ACTIVE_REPAIR": CandidatePortfolio(
            name="MIN_TURNOVER_ACTIVE_REPAIR",
            weights=pd.Series(min_turnover_repair["weights"], dtype=float).reindex(index).fillna(0.0),
            metadata=repaired_metadata("min_turnover_active_repair", "PARTIAL_25", min_turnover_repair),
        ),
    }

    current_validation = _constraint_validation(current, params, 1e-8)
    if not bool(current_validation["ok"]):
        candidates["CURRENT_COMPLIANCE_REPAIR"] = CandidatePortfolio(
            name="CURRENT_COMPLIANCE_REPAIR",
            weights=pd.Series(compliance_repair["weights"], dtype=float).reindex(index).fillna(0.0),
            metadata={
                **repaired_metadata("current_compliance_repair", "HOLD", compliance_repair),
                "fixes_current_constraints": bool(compliance_repair.get("valid", False)),
            },
        )

    if conditional_factor_target is not None and not conditional_factor_target.empty:
        conditional = _normalize(conditional_factor_target.reindex(index).fillna(0.0))
        candidates["CONDITIONAL_FACTOR_TARGET"] = CandidatePortfolio(
            name="CONDITIONAL_FACTOR_TARGET",
            weights=conditional,
            metadata={"kind": "conditional_factor"},
        )
        conditional_validation = _constraint_validation(conditional, params, 1e-8)
        if not bool(conditional_validation["ok"]):
            conditional_repair = repair_weights_to_constraints(conditional, params)
            candidates["FACTOR_TARGET_REPAIRED"] = CandidatePortfolio(
                name="FACTOR_TARGET_REPAIRED",
                weights=pd.Series(conditional_repair["weights"], dtype=float).reindex(index).fillna(0.0),
                metadata=repaired_metadata("factor_target_repaired", "CONDITIONAL_FACTOR_TARGET", conditional_repair),
            )
    return candidates
