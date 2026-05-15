"""Allocation optimizer with optional Gurobi support and SciPy fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd
try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency in stabilization mode
    minimize = None
    SCIPY_AVAILABLE = False

from asset_universe import (
    AssetDefinition,
    CRYPTO_MAX_NORMAL,
    CRYPTO_MAX_RISK_OFF,
    DEFENSIVE_GROUPS,
    EQUITY_LIKE_GROUPS,
    MAX_EQUITY_LIKE_TOTAL_NORMAL,
    MAX_EQUITY_LIKE_TOTAL_RISK_OFF,
    MIN_DEFENSIVE_WEIGHT_NORMAL,
    MIN_DEFENSIVE_WEIGHT_RISK_OFF,
    get_asset_max_weights,
    get_cash_ticker,
    get_group_limits,
    get_group_map,
)
from config import CONCENTRATION_PENALTY, MAX_TURNOVER, RISK_AVERSION, TURNOVER_PENALTY, OptimizationConfig
from risk import RiskRegime, RiskSnapshot
from risk_free import risk_free_return_for_horizon

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    gp = None
    GRB = None
    GUROBI_AVAILABLE = False


TOLERANCE = 1e-8
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OptimizerInput:
    """Compatibility input structure used by the existing backtest pipeline."""

    feature_scores: pd.Series
    covariance: pd.DataFrame
    current_weights: pd.Series
    universe: Sequence[AssetDefinition]
    risk_snapshot: RiskSnapshot


@dataclass(slots=True)
class OptimizerResult:
    """Outputs returned by the allocation optimizer."""

    target_weights: pd.Series
    solver_name: str
    success: bool
    objective_value: float | None
    status: str
    diagnostics: dict[str, float | str] = field(default_factory=dict)


@dataclass(slots=True)
class ScenarioSharpeProblem:
    """Aligned scenario-wise return/covariance inputs for direct allocation."""

    tickers: list[str]
    probabilities: np.ndarray
    mu_matrix: np.ndarray
    covariance_matrices: list[np.ndarray]
    risk_free_returns: np.ndarray
    scenario_names: list[str]


def align_inputs(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    w_current: pd.Series,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Align forecast, covariance and current weights to a common ticker order."""

    if mu.empty:
        raise ValueError("mu must not be empty.")
    if Sigma.empty:
        raise ValueError("Sigma must not be empty.")

    mu_aligned = mu.astype(float).copy()
    mu_aligned.index = pd.Index([str(ticker) for ticker in mu_aligned.index], name="ticker")

    sigma_aligned = Sigma.astype(float).copy()
    sigma_aligned.index = pd.Index([str(ticker) for ticker in sigma_aligned.index], name="ticker")
    sigma_aligned.columns = pd.Index([str(ticker) for ticker in sigma_aligned.columns], name="ticker")
    if sigma_aligned.shape[0] != sigma_aligned.shape[1]:
        raise ValueError("Sigma must be a square covariance matrix.")

    current_aligned = w_current.astype(float).copy()
    current_aligned.index = pd.Index([str(ticker) for ticker in current_aligned.index], name="ticker")

    sigma_universe = set(sigma_aligned.index).intersection(sigma_aligned.columns)
    missing_sigma = [ticker for ticker in mu_aligned.index if ticker not in sigma_universe]
    if missing_sigma:
        raise ValueError(
            "Sigma is missing rows or columns for forecast tickers: " + ", ".join(missing_sigma)
        )

    ordered_tickers: list[str] = []
    seen: set[str] = set()
    for collection in (mu_aligned.index, current_aligned.index, sigma_aligned.index):
        for ticker in collection:
            if ticker not in sigma_universe:
                continue
            if ticker not in seen:
                seen.add(ticker)
                ordered_tickers.append(ticker)

    if not ordered_tickers:
        raise ValueError("No overlapping tickers were found across mu, Sigma and w_current.")

    mu_aligned = mu_aligned.reindex(ordered_tickers).fillna(0.0)
    current_aligned = current_aligned.reindex(ordered_tickers).fillna(0.0)
    sigma_aligned = sigma_aligned.reindex(index=ordered_tickers, columns=ordered_tickers).fillna(0.0)
    sigma_aligned = 0.5 * (sigma_aligned + sigma_aligned.T)

    return mu_aligned, sigma_aligned, current_aligned


def clean_weights(weights: pd.Series) -> pd.Series:
    """Clip tiny negatives to zero and normalize weights to sum to one."""

    if weights.empty:
        raise ValueError("Weight vector must not be empty.")

    cleaned = weights.astype(float).copy()
    cleaned[np.abs(cleaned) < TOLERANCE] = 0.0
    cleaned = cleaned.clip(lower=0.0)

    total = float(cleaned.sum())
    if total <= 0.0:
        cleaned[:] = 1.0 / len(cleaned)
        return cleaned

    return cleaned / total


def _emit_warning(message: str) -> None:
    """Log and emit a runtime warning without interrupting the run."""

    LOGGER.warning(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _prepare_params(tickers: Sequence[str], params: dict[str, Any]) -> dict[str, Any]:
    """Normalize optimizer parameters and align registry-derived mappings."""

    tickers_list = [str(ticker) for ticker in tickers]
    if not tickers_list:
        raise ValueError("The optimizer requires at least one ticker.")

    asset_max_source = params.get("asset_max_weights", get_asset_max_weights())
    asset_max_weights = pd.Series(asset_max_source, dtype=float).reindex(tickers_list)
    missing_asset_caps = asset_max_weights[asset_max_weights.isna()].index.tolist()
    if missing_asset_caps:
        raise ValueError(
            "Missing asset max weights for tickers: " + ", ".join(missing_asset_caps)
        )

    group_map_source = params.get("group_map", get_group_map())
    group_map = pd.Series(group_map_source, dtype=object).reindex(tickers_list)
    missing_groups = group_map[group_map.isna()].index.tolist()
    if missing_groups:
        raise ValueError("Missing group assignments for tickers: " + ", ".join(missing_groups))

    group_limits = {
        str(group): float(limit)
        for group, limit in dict(params.get("group_limits", get_group_limits())).items()
    }
    present_groups = sorted({str(group) for group in group_map.tolist()})
    missing_group_limits = [group for group in present_groups if group not in group_limits]
    if missing_group_limits:
        raise ValueError(
            "Missing group limits for groups: " + ", ".join(missing_group_limits)
        )

    cash_ticker = params.get("cash_ticker", get_cash_ticker())
    cash_ticker = str(cash_ticker) if cash_ticker is not None else None
    if cash_ticker is not None and cash_ticker not in tickers_list:
        cash_ticker = None

    equity_like_groups = [str(group) for group in params.get("equity_like_groups", EQUITY_LIKE_GROUPS)]
    defensive_groups = [str(group) for group in params.get("defensive_groups", DEFENSIVE_GROUPS)]

    group_to_tickers: dict[str, list[str]] = {}
    for ticker in tickers_list:
        group = str(group_map[ticker])
        group_to_tickers.setdefault(group, []).append(ticker)

    equity_like_tickers = [
        ticker for ticker in tickers_list if str(group_map[ticker]) in equity_like_groups
    ]
    defensive_tickers = [
        ticker for ticker in tickers_list if str(group_map[ticker]) in defensive_groups
    ]

    return {
        "tickers": tickers_list,
        "risk_aversion": float(params.get("risk_aversion", RISK_AVERSION)),
        "turnover_penalty": float(params.get("turnover_penalty", TURNOVER_PENALTY)),
        "concentration_penalty": float(
            params.get("concentration_penalty", CONCENTRATION_PENALTY)
        ),
        "asset_max_weights": asset_max_weights.astype(float),
        "group_map": group_map.astype(str),
        "group_limits": group_limits,
        "group_to_tickers": group_to_tickers,
        "equity_like_groups": equity_like_groups,
        "defensive_groups": defensive_groups,
        "equity_like_tickers": equity_like_tickers,
        "defensive_tickers": defensive_tickers,
        "max_equity_like_total": float(
            params.get("max_equity_like_total", MAX_EQUITY_LIKE_TOTAL_NORMAL)
        ),
        "min_defensive_weight": float(
            params.get("min_defensive_weight", MIN_DEFENSIVE_WEIGHT_NORMAL)
        ),
        "cash_ticker": cash_ticker,
        "min_cash_weight": float(params.get("min_cash_weight", 0.0) or 0.0),
        "max_turnover": float(params.get("max_turnover", MAX_TURNOVER)),
    }


def _objective_value(
    weights: np.ndarray,
    mu_values: np.ndarray,
    sigma_values: np.ndarray,
    current_values: np.ndarray,
    params: dict[str, Any],
) -> float:
    """Compute the maximization objective value for a weight vector."""

    expected_return = float(mu_values @ weights)
    variance = float(weights @ sigma_values @ weights)
    turnover = float(np.abs(weights - current_values).sum())
    concentration = float(np.square(weights).sum())
    return (
        expected_return
        - params["risk_aversion"] * variance
        - params["turnover_penalty"] * turnover
        - params["concentration_penalty"] * concentration
    )


def _negative_objective(
    weights: np.ndarray,
    mu_values: np.ndarray,
    sigma_values: np.ndarray,
    current_values: np.ndarray,
    params: dict[str, Any],
) -> float:
    """Return the minimization form of the optimizer objective."""

    return -_objective_value(
        weights=weights,
        mu_values=mu_values,
        sigma_values=sigma_values,
        current_values=current_values,
        params=params,
    )


def _scenario_sharpe_objective_value(
    weights: np.ndarray,
    current_values: np.ndarray,
    problem: ScenarioSharpeProblem,
    params: dict[str, Any],
) -> float:
    """Objective requested for direct scenario-aware allocation optimization."""

    epsilon = float(params.get("eps_variance", params.get("sharpe_epsilon", 1e-8)))
    lambda_turnover = float(
        params.get("direct_scenario_lambda_turnover", params.get("lambda_turnover_sharpe", 0.0))
    )
    lambda_concentration = float(
        params.get("direct_scenario_lambda_concentration", params.get("lambda_concentration_sharpe", 0.0))
    )
    lambda_downside = float(
        params.get("direct_scenario_lambda_downside", params.get("lambda_downside_sharpe", 0.25))
    )

    weighted_sharpe = 0.0
    weighted_downside = 0.0
    for probability, mu_values, sigma_values, rf_value in zip(
        problem.probabilities,
        problem.mu_matrix,
        problem.covariance_matrices,
        problem.risk_free_returns,
        strict=True,
    ):
        scenario_return = float(weights @ mu_values)
        scenario_variance = max(float(weights @ sigma_values @ weights), 0.0)
        scenario_volatility = float(np.sqrt(scenario_variance + epsilon))
        weighted_sharpe += float(probability) * ((scenario_return - float(rf_value)) / scenario_volatility)
        weighted_downside += float(probability) * max(0.0, float(rf_value) - scenario_return)

    turnover = float(np.abs(weights - current_values).sum())
    concentration = float(np.square(weights).sum())
    return (
        weighted_sharpe
        - lambda_turnover * turnover
        - lambda_concentration * concentration
        - lambda_downside * weighted_downside
    )


def _negative_scenario_sharpe_objective(
    weights: np.ndarray,
    current_values: np.ndarray,
    problem: ScenarioSharpeProblem,
    params: dict[str, Any],
) -> float:
    """Return minimization form of the direct scenario Sharpe objective."""

    return -_scenario_sharpe_objective_value(
        weights=weights,
        current_values=current_values,
        problem=problem,
        params=params,
    )


def _residual_capacity(weights: pd.Series, ticker: str, params: dict[str, Any]) -> float:
    """Return remaining static capacity for a ticker under upper-bound constraints."""

    capacity = float(params["asset_max_weights"][ticker] - weights[ticker])
    group = str(params["group_map"][ticker])
    capacity = min(
        capacity,
        float(params["group_limits"][group] - weights[params["group_to_tickers"][group]].sum()),
    )
    if ticker in params["equity_like_tickers"]:
        capacity = min(
            capacity,
            float(params["max_equity_like_total"] - weights[params["equity_like_tickers"]].sum()),
        )
    return max(0.0, capacity)


def _remove_from_donors(
    weights: pd.Series,
    donor_tickers: Sequence[str],
    amount: float,
    floor_map: dict[str, float] | None = None,
) -> float:
    """Remove weight from donor assets while respecting optional floors."""

    remaining = float(amount)
    floor_map = floor_map or {}

    donors = sorted(
        donor_tickers,
        key=lambda ticker: float(weights[ticker] - floor_map.get(ticker, 0.0)),
        reverse=True,
    )
    for ticker in donors:
        floor = float(floor_map.get(ticker, 0.0))
        removable = max(0.0, float(weights[ticker] - floor))
        if removable <= TOLERANCE:
            continue
        delta = min(removable, remaining)
        weights[ticker] -= delta
        remaining -= delta
        if remaining <= TOLERANCE:
            break

    return float(amount - remaining)


def _allocate_evenly(
    weights: pd.Series,
    eligible_tickers: Sequence[str],
    amount: float,
    params: dict[str, Any],
) -> float:
    """Distribute additional weight evenly across eligible tickers."""

    remaining = float(amount)
    eligible = [ticker for ticker in eligible_tickers if ticker in weights.index]

    while remaining > TOLERANCE:
        active = [
            ticker
            for ticker in eligible
            if _residual_capacity(weights=weights, ticker=ticker, params=params) > TOLERANCE
        ]
        if not active:
            break

        share = remaining / len(active)
        allocated = 0.0
        for ticker in active:
            cap = _residual_capacity(weights=weights, ticker=ticker, params=params)
            delta = min(share, cap)
            if delta <= TOLERANCE:
                continue
            weights[ticker] += delta
            allocated += delta

        if allocated <= TOLERANCE:
            break
        remaining -= allocated

    return float(amount - remaining)


def _adjust_budget_residual(weights: pd.Series, params: dict[str, Any]) -> pd.Series:
    """Adjust a small budget residual without renormalizing the whole portfolio."""

    adjusted = weights.copy().astype(float)
    residual = 1.0 - float(adjusted.sum())

    if residual > TOLERANCE:
        preferred = (
            [params["cash_ticker"]] + [ticker for ticker in params["tickers"] if ticker != params["cash_ticker"]]
            if params["cash_ticker"] in adjusted.index
            else list(params["tickers"])
        )
        _allocate_evenly(
            weights=adjusted,
            eligible_tickers=preferred,
            amount=residual,
            params=params,
        )
    elif residual < -TOLERANCE:
        floor_map: dict[str, float] = {}
        cash_ticker = params["cash_ticker"]
        if cash_ticker is not None:
            floor_map[cash_ticker] = float(params["min_cash_weight"])
        _remove_from_donors(
            weights=adjusted,
            donor_tickers=params["tickers"],
            amount=-residual,
            floor_map=floor_map,
        )

    return adjusted


def _enforce_basic_upper_constraints(weights: pd.Series, params: dict[str, Any]) -> pd.Series:
    """Clip weights to static upper constraints and refill remaining budget if possible."""

    adjusted = weights.copy().astype(float)
    adjusted = adjusted.clip(lower=0.0, upper=params["asset_max_weights"])

    for group, group_tickers in params["group_to_tickers"].items():
        group_sum = float(adjusted[group_tickers].sum())
        group_limit = float(params["group_limits"][group])
        if group_sum > group_limit + TOLERANCE and group_sum > 0.0:
            adjusted.loc[group_tickers] *= group_limit / group_sum

    if params["equity_like_tickers"]:
        equity_like_sum = float(adjusted[params["equity_like_tickers"]].sum())
        equity_limit = float(params["max_equity_like_total"])
        if equity_like_sum > equity_limit + TOLERANCE and equity_like_sum > 0.0:
            adjusted.loc[params["equity_like_tickers"]] *= equity_limit / equity_like_sum

    total = float(adjusted.sum())
    if total < 1.0 - TOLERANCE:
        _allocate_evenly(
            weights=adjusted,
            eligible_tickers=params["tickers"],
            amount=1.0 - total,
            params=params,
        )

    return adjusted


def _enforce_lower_constraints(weights: pd.Series, params: dict[str, Any]) -> pd.Series:
    """Shift weight to satisfy cash and defensive minimum requirements."""

    adjusted = weights.copy().astype(float)
    cash_ticker = params["cash_ticker"]
    min_cash_weight = float(params["min_cash_weight"])

    if cash_ticker is not None and min_cash_weight > float(adjusted.get(cash_ticker, 0.0)) + TOLERANCE:
        cash_gap = min_cash_weight - float(adjusted[cash_ticker])
        removed = _remove_from_donors(
            weights=adjusted,
            donor_tickers=[ticker for ticker in params["tickers"] if ticker != cash_ticker],
            amount=min(cash_gap, _residual_capacity(adjusted, cash_ticker, params)),
        )
        adjusted[cash_ticker] += removed

    defensive_tickers = params["defensive_tickers"]
    defensive_sum = float(adjusted[defensive_tickers].sum()) if defensive_tickers else 0.0
    defensive_gap = float(params["min_defensive_weight"]) - defensive_sum
    if defensive_tickers and defensive_gap > TOLERANCE:
        donors = [ticker for ticker in params["tickers"] if ticker not in defensive_tickers]
        funded = _remove_from_donors(
            weights=adjusted,
            donor_tickers=donors,
            amount=defensive_gap,
        )
        _allocate_evenly(
            weights=adjusted,
            eligible_tickers=(
                [cash_ticker] + [ticker for ticker in defensive_tickers if ticker != cash_ticker]
                if cash_ticker in defensive_tickers
                else defensive_tickers
            ),
            amount=funded,
            params=params,
        )

    return adjusted


def _repair_static_constraints(weights: pd.Series, params: dict[str, Any]) -> pd.Series:
    """Repair static constraints for guesses and fallbacks without naive renormalization."""

    adjusted = weights.copy().astype(float)
    adjusted[np.abs(adjusted) < TOLERANCE] = 0.0
    adjusted = adjusted.clip(lower=0.0)
    adjusted = _enforce_basic_upper_constraints(weights=adjusted, params=params)
    adjusted = _enforce_lower_constraints(weights=adjusted, params=params)
    adjusted = _enforce_basic_upper_constraints(weights=adjusted, params=params)
    adjusted = _adjust_budget_residual(weights=adjusted, params=params)
    adjusted = _enforce_lower_constraints(weights=adjusted, params=params)
    adjusted = _enforce_basic_upper_constraints(weights=adjusted, params=params)
    adjusted = _adjust_budget_residual(weights=adjusted, params=params)
    adjusted[np.abs(adjusted) < TOLERANCE] = 0.0
    return adjusted.clip(lower=0.0)


def _constraint_violations(
    weights: pd.Series,
    w_current: pd.Series,
    params: dict[str, Any],
) -> list[str]:
    """Return a list of violated optimization constraints for the supplied weights."""

    violations: list[str] = []

    total = float(weights.sum())
    if abs(total - 1.0) > 1e-6:
        violations.append(f"budget={total:.8f}")

    upper_violations = weights - params["asset_max_weights"]
    if float(upper_violations.max()) > 1e-6:
        violations.append("asset_max")

    if float(weights.min()) < -1e-6:
        violations.append("non_negative")

    turnover = float(np.abs(weights - w_current).sum())
    if turnover > float(params["max_turnover"]) + 1e-6:
        violations.append(f"turnover={turnover:.6f}")

    for group, group_tickers in params["group_to_tickers"].items():
        group_sum = float(weights[group_tickers].sum())
        if group_sum > float(params["group_limits"][group]) + 1e-6:
            violations.append(f"group_limit:{group}")

    if params["equity_like_tickers"]:
        equity_sum = float(weights[params["equity_like_tickers"]].sum())
        if equity_sum > float(params["max_equity_like_total"]) + 1e-6:
            violations.append("equity_like_total")

    if params["defensive_tickers"]:
        defensive_sum = float(weights[params["defensive_tickers"]].sum())
        if defensive_sum < float(params["min_defensive_weight"]) - 1e-6:
            violations.append("defensive_total")

    cash_ticker = params["cash_ticker"]
    if cash_ticker is not None and float(weights[cash_ticker]) < float(params["min_cash_weight"]) - 1e-6:
        violations.append("cash_min")

    return violations


def validate_weights(weights: pd.Series, params: dict[str, Any], w_current: pd.Series | None = None) -> list[str]:
    """Validate a weight vector against the optimizer constraints."""

    prepared = _prepare_params(tickers=weights.index.tolist(), params=params)
    reference = (
        w_current.reindex(weights.index).fillna(0.0).astype(float)
        if w_current is not None
        else weights.astype(float).copy()
    )
    aligned_weights = weights.reindex(prepared["tickers"]).fillna(0.0).astype(float)
    return _constraint_violations(weights=aligned_weights, w_current=reference, params=prepared)


def build_feasible_initial_weights(tickers: Sequence[str], params: dict[str, Any]) -> pd.Series:
    """Build a static feasible equal-weight-like portfolio within the optimizer limits."""

    prepared = _prepare_params(tickers=tickers, params=params)
    weights = pd.Series(0.0, index=prepared["tickers"], dtype=float)

    cash_ticker = prepared["cash_ticker"]
    min_cash_weight = float(prepared["min_cash_weight"])
    if cash_ticker is not None and min_cash_weight > TOLERANCE:
        cash_capacity = _residual_capacity(weights, cash_ticker, prepared)
        if cash_capacity + TOLERANCE < min_cash_weight:
            raise ValueError("Static constraints are infeasible because min_cash_weight exceeds cash capacity.")
        weights[cash_ticker] += min_cash_weight

    defensive_gap = float(prepared["min_defensive_weight"]) - float(weights[prepared["defensive_tickers"]].sum())
    if prepared["defensive_tickers"] and defensive_gap > TOLERANCE:
        defensive_priority = (
            [cash_ticker] + [ticker for ticker in prepared["defensive_tickers"] if ticker != cash_ticker]
            if cash_ticker in prepared["defensive_tickers"]
            else prepared["defensive_tickers"]
        )
        funded = _allocate_evenly(
            weights=weights,
            eligible_tickers=defensive_priority,
            amount=defensive_gap,
            params=prepared,
        )
        if funded + TOLERANCE < defensive_gap:
            raise ValueError(
                "Static constraints are infeasible because defensive minimum weight cannot be funded."
            )

    remaining_budget = 1.0 - float(weights.sum())
    if remaining_budget > TOLERANCE:
        funded = _allocate_evenly(
            weights=weights,
            eligible_tickers=prepared["tickers"],
            amount=remaining_budget,
            params=prepared,
        )
        if funded + TOLERANCE < remaining_budget:
            raise ValueError(
                "Static constraints are infeasible because upper bounds do not permit a fully invested portfolio."
            )

    weights = clean_weights(weights)
    violations = _constraint_violations(weights=weights, w_current=weights, params=prepared)
    non_turnover_violations = [violation for violation in violations if not violation.startswith("turnover=")]
    if non_turnover_violations:
        raise ValueError(
            "Unable to build a static feasible portfolio: " + ", ".join(non_turnover_violations)
        )
    return weights


def _build_initial_guess(
    tickers: Sequence[str],
    w_current: pd.Series,
    params: dict[str, Any],
) -> pd.Series:
    """Build the SciPy starting point from current weights or a feasible equal-weight-like fallback."""

    prepared = _prepare_params(tickers=tickers, params=params)
    candidate = clean_weights(w_current.reindex(prepared["tickers"]).fillna(0.0))
    candidate = _repair_static_constraints(weights=candidate, params=prepared)

    total = float(candidate.sum())
    if total <= 0.0:
        return build_feasible_initial_weights(tickers=prepared["tickers"], params=prepared)

    if not _constraint_violations(weights=candidate, w_current=w_current.reindex(candidate.index).fillna(0.0), params=prepared):
        return candidate

    return build_feasible_initial_weights(tickers=prepared["tickers"], params=prepared)


def _build_feasible_fallback_portfolio(
    tickers: Sequence[str],
    w_current: pd.Series,
    params: dict[str, Any],
) -> pd.Series:
    """Return the fallback portfolio after an optimizer failure."""

    prepared = _prepare_params(tickers=tickers, params=params)
    try:
        fallback = build_feasible_initial_weights(tickers=prepared["tickers"], params=prepared)
    except Exception as exc:
        _emit_warning(
            f"Unable to build a feasible equal-weight-like fallback portfolio: {exc}. "
            "Returning cleaned current weights instead."
        )
        fallback = clean_weights(w_current.reindex(prepared["tickers"]).fillna(0.0))
        fallback = _repair_static_constraints(weights=fallback, params=prepared)
        if float(fallback.sum()) <= 0.0:
            fallback = clean_weights(fallback)

    turnover = float(np.abs(fallback - w_current.reindex(fallback.index).fillna(0.0)).sum())
    if turnover > float(prepared["max_turnover"]) + 1e-6:
        _emit_warning(
            "Fallback portfolio satisfies static allocation constraints but exceeds max_turnover. "
            "Returning best-effort fallback allocation."
        )
    return fallback


def _finalize_solution_weights(
    weights: pd.Series | Sequence[float] | np.ndarray,
    tickers: Sequence[str],
    w_current: pd.Series,
    params: dict[str, Any],
) -> pd.Series:
    """Return a best-effort long-only, normalized weight vector on the requested ticker index."""

    prepared = _prepare_params(tickers=tickers, params=params)
    current_aligned = w_current.reindex(prepared["tickers"]).fillna(0.0).astype(float)

    finalized = pd.Series(weights, index=prepared["tickers"], dtype=float)
    finalized = finalized.reindex(prepared["tickers"]).fillna(0.0).astype(float)
    finalized[np.abs(finalized) < TOLERANCE] = 0.0
    finalized = finalized.clip(lower=0.0)
    finalized = _repair_static_constraints(weights=finalized, params=prepared)
    finalized = _adjust_budget_residual(weights=finalized, params=prepared)
    finalized = finalized.clip(lower=0.0)

    total = float(finalized.sum())
    if total <= 0.0:
        _emit_warning(
            "Optimizer post-processing produced a zero-weight portfolio. "
            "Falling back to a feasible equal-weight-like portfolio."
        )
        return _build_feasible_fallback_portfolio(
            tickers=prepared["tickers"],
            w_current=current_aligned,
            params=prepared,
        )

    if abs(total - 1.0) > 1e-8:
        finalized = finalized / total
        finalized = _repair_static_constraints(weights=finalized, params=prepared)
        finalized = _adjust_budget_residual(weights=finalized, params=prepared)

    violations = _constraint_violations(weights=finalized, w_current=current_aligned, params=prepared)
    critical_violations = [violation for violation in violations if not violation.startswith("turnover=")]
    if critical_violations:
        _emit_warning(
            "Optimizer post-processing still violates static constraints ("
            + ", ".join(critical_violations)
            + "). Falling back to a feasible equal-weight-like portfolio."
        )
        return _build_feasible_fallback_portfolio(
            tickers=prepared["tickers"],
            w_current=current_aligned,
            params=prepared,
        )

    if violations:
        _emit_warning(
            "Optimizer returned a best-effort allocation with remaining non-critical violations: "
            + ", ".join(violations)
        )

    return finalized.reindex(prepared["tickers"]).fillna(0.0).astype(float)


def optimize_with_gurobi(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    w_current: pd.Series,
    params: dict[str, Any],
) -> OptimizerResult:
    """Optimize the allocation with Gurobi."""

    if not GUROBI_AVAILABLE:
        raise RuntimeError("gurobipy is not available.")

    mu_aligned, sigma_aligned, current_aligned = align_inputs(mu=mu, Sigma=Sigma, w_current=w_current)
    prepared = _prepare_params(tickers=mu_aligned.index.tolist(), params=params)
    tickers = prepared["tickers"]

    model = gp.Model("robust_3m_active_allocation")
    model.Params.OutputFlag = 0

    w = {
        ticker: model.addVar(
            lb=0.0,
            ub=float(prepared["asset_max_weights"][ticker]),
            vtype=GRB.CONTINUOUS,
            name=f"w_{ticker}",
        )
        for ticker in tickers
    }
    u = {
        ticker: model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"u_{ticker}")
        for ticker in tickers
    }

    model.addConstr(gp.quicksum(w[ticker] for ticker in tickers) == 1.0, name="budget")

    for ticker in tickers:
        current_value = float(current_aligned[ticker])
        model.addConstr(u[ticker] >= w[ticker] - current_value, name=f"u_pos_{ticker}")
        model.addConstr(u[ticker] >= -(w[ticker] - current_value), name=f"u_neg_{ticker}")

    model.addConstr(
        gp.quicksum(u[ticker] for ticker in tickers) <= float(prepared["max_turnover"]),
        name="max_turnover",
    )

    for group, group_tickers in prepared["group_to_tickers"].items():
        model.addConstr(
            gp.quicksum(w[ticker] for ticker in group_tickers) <= float(prepared["group_limits"][group]),
            name=f"group_limit_{group}",
        )

    if prepared["equity_like_tickers"]:
        model.addConstr(
            gp.quicksum(w[ticker] for ticker in prepared["equity_like_tickers"])
            <= float(prepared["max_equity_like_total"]),
            name="equity_like_total",
        )

    if prepared["defensive_tickers"]:
        model.addConstr(
            gp.quicksum(w[ticker] for ticker in prepared["defensive_tickers"])
            >= float(prepared["min_defensive_weight"]),
            name="defensive_total",
        )

    cash_ticker = prepared["cash_ticker"]
    if cash_ticker is not None:
        model.addConstr(
            w[cash_ticker] >= float(prepared["min_cash_weight"]),
            name="cash_min",
        )

    expected_return_expr = gp.quicksum(float(mu_aligned[ticker]) * w[ticker] for ticker in tickers)
    turnover_expr = gp.quicksum(u[ticker] for ticker in tickers)
    concentration_expr = gp.quicksum(w[ticker] * w[ticker] for ticker in tickers)

    variance_expr = gp.QuadExpr()
    for left in tickers:
        for right in tickers:
            variance_expr += float(sigma_aligned.loc[left, right]) * w[left] * w[right]

    objective = (
        expected_return_expr
        - float(prepared["risk_aversion"]) * variance_expr
        - float(prepared["turnover_penalty"]) * turnover_expr
        - float(prepared["concentration_penalty"]) * concentration_expr
    )
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        raise RuntimeError(f"Gurobi solve failed with status code {model.Status}.")

    raw_weights = pd.Series({ticker: float(w[ticker].X) for ticker in tickers}, dtype=float)
    weights = _finalize_solution_weights(
        weights=raw_weights,
        tickers=tickers,
        w_current=current_aligned,
        params=prepared,
    )
    violations = _constraint_violations(weights=weights, w_current=current_aligned, params=prepared)
    critical_violations = [violation for violation in violations if not violation.startswith("turnover=")]
    if critical_violations:
        raise RuntimeError(
            "Gurobi returned a solution that violates optimizer constraints after post-processing: "
            + ", ".join(critical_violations)
        )

    return OptimizerResult(
        target_weights=weights,
        solver_name="gurobi",
        success=model.Status == GRB.OPTIMAL,
        objective_value=float(model.ObjVal),
        status="OPTIMAL" if model.Status == GRB.OPTIMAL else "SUBOPTIMAL",
        diagnostics={
            "turnover": float(np.abs(weights - current_aligned).sum()),
            "status_code": float(model.Status),
        },
    )


def optimize_with_scipy(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    w_current: pd.Series,
    params: dict[str, Any],
) -> OptimizerResult:
    """Optimize the allocation with SciPy SLSQP and a robust fallback."""

    mu_aligned, sigma_aligned, current_aligned = align_inputs(mu=mu, Sigma=Sigma, w_current=w_current)
    prepared = _prepare_params(tickers=mu_aligned.index.tolist(), params=params)
    tickers = prepared["tickers"]
    mu_values = mu_aligned.to_numpy(dtype=float)
    sigma_values = sigma_aligned.to_numpy(dtype=float)
    current_values = current_aligned.reindex(tickers).fillna(0.0).to_numpy(dtype=float)

    if not SCIPY_AVAILABLE or minimize is None:
        warning_message = (
            "SciPy is not available in the current Python environment. "
            "Falling back to a feasible equal-weight-like portfolio."
        )
        _emit_warning(warning_message)
        fallback_weights = _build_feasible_fallback_portfolio(
            tickers=tickers,
            w_current=current_aligned,
            params=prepared,
        )
        return OptimizerResult(
            target_weights=fallback_weights,
            solver_name="scipy_missing_fallback",
            success=False,
            objective_value=_objective_value(
                weights=fallback_weights.to_numpy(dtype=float),
                mu_values=mu_values,
                sigma_values=sigma_values,
                current_values=current_values,
                params=prepared,
            ),
            status="SciPy unavailable",
            diagnostics={
                "turnover": float(np.abs(fallback_weights - current_aligned).sum()),
                "warning": warning_message,
            },
        )

    initial_guess = _build_initial_guess(
        tickers=tickers,
        w_current=current_aligned,
        params=prepared,
    ).to_numpy(dtype=float)

    bounds = [
        (0.0, float(prepared["asset_max_weights"][ticker]))
        for ticker in tickers
    ]

    group_indices = {
        group: [tickers.index(ticker) for ticker in group_tickers]
        for group, group_tickers in prepared["group_to_tickers"].items()
    }
    equity_like_indices = [tickers.index(ticker) for ticker in prepared["equity_like_tickers"]]
    defensive_indices = [tickers.index(ticker) for ticker in prepared["defensive_tickers"]]
    cash_index = tickers.index(prepared["cash_ticker"]) if prepared["cash_ticker"] in tickers else None

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)},
        {
            "type": "ineq",
            "fun": lambda x: float(prepared["max_turnover"] - np.abs(x - current_values).sum()),
        },
    ]

    for group, indices in group_indices.items():
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, idx=indices, limit=float(prepared["group_limits"][group]): float(
                    limit - np.sum(x[idx])
                ),
            }
        )

    if equity_like_indices:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: float(
                    float(prepared["max_equity_like_total"]) - np.sum(x[equity_like_indices])
                ),
            }
        )

    if defensive_indices:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: float(
                    np.sum(x[defensive_indices]) - float(prepared["min_defensive_weight"])
                ),
            }
        )

    if cash_index is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: float(x[cash_index] - float(prepared["min_cash_weight"])),
            }
        )

    result = minimize(
        fun=_negative_objective,
        x0=initial_guess,
        args=(mu_values, sigma_values, current_values, prepared),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if result.success and result.x is not None:
        weights = _finalize_solution_weights(
            weights=result.x,
            tickers=tickers,
            w_current=current_aligned,
            params=prepared,
        )
        violations = _constraint_violations(weights=weights, w_current=current_aligned, params=prepared)
        critical_violations = [violation for violation in violations if not violation.startswith("turnover=")]
        if not critical_violations:
            diagnostics: dict[str, float | str] = {
                "iterations": float(getattr(result, "nit", 0)),
                "turnover": float(np.abs(weights - current_aligned).sum()),
            }
            if violations:
                diagnostics["warning"] = "non_critical_violations:" + ",".join(violations)
            return OptimizerResult(
                target_weights=weights,
                solver_name="scipy_slsqp",
                success=True,
                objective_value=_objective_value(
                    weights=weights.to_numpy(dtype=float),
                    mu_values=mu_values,
                    sigma_values=sigma_values,
                    current_values=current_values,
                    params=prepared,
                ),
                status=str(result.message),
                diagnostics=diagnostics,
            )

    warning_message = (
        f"SciPy optimization failed or returned an infeasible solution: "
        f"{getattr(result, 'message', 'unknown failure')}. "
        "Falling back to a feasible equal-weight-like portfolio."
    )
    _emit_warning(warning_message)

    fallback_weights = _build_feasible_fallback_portfolio(
        tickers=tickers,
        w_current=current_aligned,
        params=prepared,
    )

    return OptimizerResult(
        target_weights=fallback_weights,
        solver_name="scipy_slsqp_fallback",
        success=False,
        objective_value=_objective_value(
            weights=fallback_weights.to_numpy(dtype=float),
            mu_values=mu_values,
            sigma_values=sigma_values,
            current_values=current_values,
            params=prepared,
        ),
        status=str(getattr(result, "message", "SciPy failed")),
        diagnostics={
            "iterations": float(getattr(result, "nit", 0)),
            "turnover": float(np.abs(fallback_weights - current_aligned).sum()),
            "warning": warning_message,
        },
    )


def _scenario_risk_free_returns(
    *,
    expected_returns: pd.DataFrame,
    params: dict[str, Any],
) -> pd.Series:
    """Resolve scenario-wise risk-free returns on the same horizon as scenario mu."""

    configured = params.get("scenario_risk_free_returns")
    if isinstance(configured, dict) and configured:
        rf = pd.Series(configured, dtype=float).reindex(expected_returns.index)
        if rf.notna().any():
            return rf.fillna(float(rf.dropna().mean())).astype(float)

    cash_ticker = params.get("cash_ticker") or params.get("effective_cash_ticker")
    if cash_ticker is not None and str(cash_ticker) in expected_returns.columns:
        return expected_returns[str(cash_ticker)].astype(float)

    annual_rf = float(params.get("risk_free_rate_annual", 0.0) or 0.0)
    horizon_days = int(params.get("effective_horizon_days", params.get("horizon_days", 63)) or 63)
    horizon_rf = risk_free_return_for_horizon(
        annual_rf,
        horizon_days,
        int(params.get("trading_days_per_year", 252) or 252),
    )
    return pd.Series(horizon_rf, index=expected_returns.index, dtype=float)


def _build_scenario_sharpe_problem(
    *,
    distribution: Any,
    params: dict[str, Any],
) -> ScenarioSharpeProblem:
    """Align scenario distribution inputs for direct nonlinear optimization."""

    assets = pd.Index([str(ticker) for ticker in distribution.assets], name="ticker")
    expected_returns = distribution.expected_returns.reindex(columns=assets).astype(float)
    probabilities = distribution.probabilities.reindex(expected_returns.index).fillna(0.0).astype(float)
    probabilities = probabilities.clip(lower=0.0)
    total_probability = float(probabilities.sum())
    if total_probability <= 0.0:
        raise ValueError("Scenario probabilities must contain positive mass.")
    probabilities = probabilities / total_probability
    risk_free = _scenario_risk_free_returns(expected_returns=expected_returns, params=params)
    covariance_matrices: list[np.ndarray] = []
    for scenario_name in expected_returns.index:
        covariance = distribution.covariance_matrices[str(scenario_name)].reindex(
            index=assets,
            columns=assets,
        ).fillna(0.0)
        values = covariance.to_numpy(dtype=float, copy=True)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        covariance_matrices.append(0.5 * (values + values.T))

    return ScenarioSharpeProblem(
        tickers=assets.astype(str).tolist(),
        probabilities=probabilities.to_numpy(dtype=float),
        mu_matrix=expected_returns.to_numpy(dtype=float),
        covariance_matrices=covariance_matrices,
        risk_free_returns=risk_free.reindex(expected_returns.index).fillna(0.0).to_numpy(dtype=float),
        scenario_names=[str(name) for name in expected_returns.index],
    )


def _dedupe_start_vectors(starts: list[pd.Series], tickers: Sequence[str]) -> list[pd.Series]:
    unique: list[pd.Series] = []
    seen: set[tuple[float, ...]] = set()
    for start in starts:
        aligned = start.reindex(tickers).fillna(0.0).astype(float)
        signature = tuple(np.round(aligned.to_numpy(dtype=float), 10))
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(aligned)
    return unique


def _build_greedy_scenario_start(
    *,
    problem: ScenarioSharpeProblem,
    w_current: pd.Series,
    params: dict[str, Any],
    direction: str,
) -> pd.Series:
    """Build deterministic extra starts to reduce local-solver path dependence."""

    prepared = _prepare_params(tickers=problem.tickers, params=params)
    weights = pd.Series(0.0, index=prepared["tickers"], dtype=float)
    probability_values = problem.probabilities.reshape((-1, 1))
    expected_excess = ((problem.mu_matrix - problem.risk_free_returns.reshape((-1, 1))) * probability_values).sum(axis=0)
    scores = pd.Series(expected_excess, index=prepared["tickers"], dtype=float)
    if direction == "low_risk":
        average_variance = np.zeros(len(prepared["tickers"]), dtype=float)
        for probability, covariance in zip(problem.probabilities, problem.covariance_matrices, strict=True):
            average_variance += float(probability) * np.maximum(np.diag(covariance), 0.0)
        scores = -pd.Series(average_variance, index=prepared["tickers"], dtype=float)
    elif direction == "current_blend":
        base = w_current.reindex(prepared["tickers"]).fillna(0.0).astype(float)
        repaired = _repair_static_constraints(weights=base, params=prepared)
        blended = clean_weights(0.5 * repaired + 0.5 * build_feasible_initial_weights(prepared["tickers"], prepared))
        blended = _repair_static_constraints(weights=blended, params=prepared)
        return _adjust_budget_residual(weights=blended, params=prepared)

    for ticker in scores.sort_values(ascending=False).index:
        residual = 1.0 - float(weights.sum())
        if residual <= TOLERANCE:
            break
        capacity = _residual_capacity(weights, str(ticker), prepared)
        if capacity <= TOLERANCE:
            continue
        weights.loc[str(ticker)] += min(capacity, residual)

    weights = _repair_static_constraints(weights=weights, params=prepared)
    weights = _adjust_budget_residual(weights=weights, params=prepared)
    if float(weights.sum()) <= TOLERANCE:
        return build_feasible_initial_weights(prepared["tickers"], prepared)
    return weights.reindex(prepared["tickers"]).fillna(0.0).astype(float)


def optimize_scenario_sharpe_allocation(
    *,
    distribution: Any,
    w_current: pd.Series,
    params: dict[str, Any],
) -> OptimizerResult:
    """Directly solve the scenario-probability Sharpe objective over weights.

    This replaces final model selection across hand-built continuous candidate
    portfolios. HOLD_CURRENT remains an execution benchmark outside this solver.
    """

    problem = _build_scenario_sharpe_problem(distribution=distribution, params=params)
    current_aligned = w_current.reindex(problem.tickers).fillna(0.0).astype(float)
    prepared = _prepare_params(tickers=problem.tickers, params=params)
    prepared.update(
        {
            "eps_variance": float(params.get("eps_variance", params.get("sharpe_epsilon", 1e-8))),
            "sharpe_epsilon": float(params.get("sharpe_epsilon", params.get("eps_variance", 1e-8))),
            "direct_scenario_lambda_turnover": float(
                params.get("direct_scenario_lambda_turnover", params.get("lambda_turnover_sharpe", 0.0))
            ),
            "direct_scenario_lambda_concentration": float(
                params.get("direct_scenario_lambda_concentration", params.get("lambda_concentration_sharpe", 0.0))
            ),
            "direct_scenario_lambda_downside": float(
                params.get("direct_scenario_lambda_downside", params.get("lambda_downside_sharpe", 0.25))
            ),
            "scenario_solver_objective": str(params.get("scenario_solver_objective", "scenario_weighted_rf_sharpe")),
        }
    )
    current_values = current_aligned.to_numpy(dtype=float)

    if not SCIPY_AVAILABLE or minimize is None:
        warning_message = (
            "SciPy is not available; direct scenario Sharpe optimization cannot run. "
            "Using a feasible static fallback target."
        )
        _emit_warning(warning_message)
        fallback_weights = _build_feasible_fallback_portfolio(
            tickers=problem.tickers,
            w_current=current_aligned,
            params=prepared,
        )
        return OptimizerResult(
            target_weights=fallback_weights,
            solver_name="direct_scenario_sharpe_scipy_missing_fallback",
            success=False,
            objective_value=_scenario_sharpe_objective_value(
                fallback_weights.to_numpy(dtype=float),
                current_values,
                problem,
                prepared,
            ),
            status="SciPy unavailable",
            diagnostics={
                "warning": warning_message,
                "objective": str(params.get("scenario_solver_objective", "scenario_weighted_rf_sharpe")),
                "turnover": float(np.abs(fallback_weights - current_aligned).sum()),
            },
        )

    bounds = [(0.0, float(prepared["asset_max_weights"][ticker])) for ticker in problem.tickers]
    group_indices = {
        group: [problem.tickers.index(ticker) for ticker in group_tickers]
        for group, group_tickers in prepared["group_to_tickers"].items()
    }
    equity_like_indices = [problem.tickers.index(ticker) for ticker in prepared["equity_like_tickers"]]
    defensive_indices = [problem.tickers.index(ticker) for ticker in prepared["defensive_tickers"]]
    cash_index = problem.tickers.index(prepared["cash_ticker"]) if prepared["cash_ticker"] in problem.tickers else None

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)},
        {
            "type": "ineq",
            "fun": lambda x: float(prepared["max_turnover"] - np.abs(x - current_values).sum()),
        },
    ]
    for group, indices in group_indices.items():
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, idx=indices, limit=float(prepared["group_limits"][group]): float(
                    limit - np.sum(x[idx])
                ),
            }
        )
    if equity_like_indices:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: float(
                    float(prepared["max_equity_like_total"]) - np.sum(x[equity_like_indices])
                ),
            }
        )
    if defensive_indices:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: float(
                    np.sum(x[defensive_indices]) - float(prepared["min_defensive_weight"])
                ),
            }
        )
    if cash_index is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: float(x[cash_index] - float(prepared["min_cash_weight"])),
            }
        )

    starts = _dedupe_start_vectors(
        [
            _build_initial_guess(tickers=problem.tickers, w_current=current_aligned, params=prepared),
            build_feasible_initial_weights(problem.tickers, prepared),
            _build_greedy_scenario_start(problem=problem, w_current=current_aligned, params=prepared, direction="return"),
            _build_greedy_scenario_start(problem=problem, w_current=current_aligned, params=prepared, direction="low_risk"),
            _build_greedy_scenario_start(problem=problem, w_current=current_aligned, params=prepared, direction="current_blend"),
        ],
        problem.tickers,
    )
    max_starts = max(1, int(params.get("direct_scenario_optimizer_max_starts", len(starts)) or len(starts)))
    optimizer_method = str(params.get("direct_scenario_optimizer_method", "SLSQP") or "SLSQP").upper()

    best_result: Any | None = None
    best_weights: pd.Series | None = None
    best_objective = float("-inf")
    statuses: list[str] = []
    for start in starts[:max_starts]:
        result = minimize(
            fun=_negative_scenario_sharpe_objective,
            x0=start.to_numpy(dtype=float),
            args=(current_values, problem, prepared),
            method=optimizer_method,
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": int(params.get("direct_scenario_optimizer_maxiter", 800)),
                "ftol": float(params.get("direct_scenario_optimizer_ftol", 1e-10)),
            },
        )
        statuses.append(str(getattr(result, "message", "unknown")))
        if not result.success or result.x is None:
            continue
        candidate_weights = _finalize_solution_weights(
            weights=result.x,
            tickers=problem.tickers,
            w_current=current_aligned,
            params=prepared,
        )
        violations = _constraint_violations(candidate_weights, current_aligned, prepared)
        critical_violations = [violation for violation in violations if not violation.startswith("turnover=")]
        if critical_violations:
            continue
        objective = _scenario_sharpe_objective_value(
            candidate_weights.to_numpy(dtype=float),
            current_values,
            problem,
            prepared,
        )
        if objective > best_objective:
            best_objective = float(objective)
            best_result = result
            best_weights = candidate_weights

    if best_weights is not None:
        return OptimizerResult(
            target_weights=best_weights,
            solver_name="direct_scenario_sharpe_slsqp",
            success=True,
            objective_value=best_objective,
            status=str(getattr(best_result, "message", "OK")),
            diagnostics={
                "objective": str(params.get("scenario_solver_objective", "scenario_weighted_rf_sharpe")),
                "optimizer_method": optimizer_method,
                "scenario_count": float(len(problem.scenario_names)),
                "start_count": float(len(starts[:max_starts])),
                "turnover": float(np.abs(best_weights - current_aligned).sum()),
                "risk_free_mode": str(params.get("direct_scenario_rf_mode", "cash_ticker")),
            },
        )

    warning_message = (
        "Direct scenario Sharpe optimization failed for all deterministic starts. "
        "Using a feasible static fallback target; execution gates remain active."
    )
    _emit_warning(warning_message)
    fallback_weights = _build_feasible_fallback_portfolio(
        tickers=problem.tickers,
        w_current=current_aligned,
        params=prepared,
    )
    fallback_objective = _scenario_sharpe_objective_value(
        fallback_weights.to_numpy(dtype=float),
        current_values,
        problem,
        prepared,
    )
    return OptimizerResult(
        target_weights=fallback_weights,
        solver_name="direct_scenario_sharpe_slsqp_fallback",
        success=False,
        objective_value=fallback_objective,
        status=" | ".join(statuses[-3:]) if statuses else f"{optimizer_method} failed",
        diagnostics={
            "warning": warning_message,
            "objective": str(params.get("scenario_solver_objective", "scenario_weighted_rf_sharpe")),
            "optimizer_method": optimizer_method,
            "scenario_count": float(len(problem.scenario_names)),
            "start_count": float(len(starts[:max_starts])),
            "turnover": float(np.abs(fallback_weights - current_aligned).sum()),
        },
    )


def _build_params_from_compatibility_inputs(
    input_data: OptimizerInput,
    config: OptimizationConfig | None,
) -> dict[str, Any]:
    """Translate the legacy optimizer inputs into the new parameter dictionary."""

    group_limits = get_group_limits()
    regime = input_data.risk_snapshot.regime

    if regime in {RiskRegime.RISK_OFF, RiskRegime.PAUSE}:
        max_equity_like_total = MAX_EQUITY_LIKE_TOTAL_RISK_OFF
        min_defensive_weight = MIN_DEFENSIVE_WEIGHT_RISK_OFF
        group_limits["crypto"] = min(group_limits.get("crypto", CRYPTO_MAX_RISK_OFF), CRYPTO_MAX_RISK_OFF)
    else:
        max_equity_like_total = MAX_EQUITY_LIKE_TOTAL_NORMAL
        min_defensive_weight = MIN_DEFENSIVE_WEIGHT_NORMAL
        group_limits["crypto"] = min(group_limits.get("crypto", CRYPTO_MAX_NORMAL), CRYPTO_MAX_NORMAL)

    return {
        "risk_aversion": getattr(config, "risk_aversion", RISK_AVERSION),
        "turnover_penalty": getattr(config, "turnover_penalty", TURNOVER_PENALTY),
        "concentration_penalty": getattr(config, "concentration_penalty", CONCENTRATION_PENALTY),
        "asset_max_weights": get_asset_max_weights(),
        "group_map": get_group_map(),
        "group_limits": group_limits,
        "equity_like_groups": EQUITY_LIKE_GROUPS,
        "defensive_groups": DEFENSIVE_GROUPS,
        "max_equity_like_total": max_equity_like_total,
        "min_defensive_weight": min_defensive_weight,
        "cash_ticker": get_cash_ticker(),
        "min_cash_weight": getattr(config, "min_cash_buffer", 0.0),
        "max_turnover": getattr(config, "max_turnover", MAX_TURNOVER),
    }


def optimize_allocation(
    mu: pd.Series | None = None,
    Sigma: pd.DataFrame | None = None,
    w_current: pd.Series | None = None,
    params: dict[str, Any] | None = None,
    *,
    input_data: OptimizerInput | None = None,
    config: OptimizationConfig | None = None,
) -> OptimizerResult:
    """Optimize the target allocation using Gurobi when possible, else SciPy."""

    if input_data is not None:
        mu = input_data.feature_scores
        Sigma = input_data.covariance
        w_current = input_data.current_weights
        params = _build_params_from_compatibility_inputs(input_data=input_data, config=config)

    if mu is None or Sigma is None or w_current is None:
        raise ValueError("optimize_allocation requires mu, Sigma, w_current and params.")

    effective_params = dict(params or {})

    if GUROBI_AVAILABLE:
        try:
            return optimize_with_gurobi(
                mu=mu,
                Sigma=Sigma,
                w_current=w_current,
                params=effective_params,
            )
        except Exception as exc:  # pragma: no cover - exercised only when gurobipy is installed
            LOGGER.warning("Gurobi optimization failed, switching to SciPy fallback: %s", exc)
            scipy_result = optimize_with_scipy(
                mu=mu,
                Sigma=Sigma,
                w_current=w_current,
                params=effective_params,
            )
            scipy_result.diagnostics["gurobi_fallback_reason"] = str(exc)
            return scipy_result

    return optimize_with_scipy(
        mu=mu,
        Sigma=Sigma,
        w_current=w_current,
        params=effective_params,
    )
