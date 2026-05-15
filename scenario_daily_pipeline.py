"""Slim scenario-weighted daily solve helper.

This module contains the active Daily Bot target-allocation solve. It is not a
candidate factory and it does not write files or execute orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src_new.regimes.regime_probability_model import compute_regime_indicators

from constraint_repair import repair_weights_to_constraints
from scenarios import build_scenario_inputs
from scenario_weighted_solver import (
    ScenarioInput,
    SolverConfig,
    SolverResult,
    solve_scenario_weighted_sharpe,
    validate_solver_result,
)


@dataclass(slots=True)
class ScenarioWeightedDailySolveResult:
    """Result bundle for the active scenario-weighted Daily solve."""

    final_target_source: str
    scenario_inputs: list[ScenarioInput]
    solver_config: SolverConfig
    solver_constraints: dict[str, Any]
    solver_current_weights: pd.Series
    solver_result: SolverResult
    solver_validation: dict[str, Any]
    optimal_weights: pd.Series
    executable_weights: pd.Series
    target_weights: pd.Series
    solver_assets: pd.Index
    warnings: list[str] = field(default_factory=list)


def run_scenario_weighted_daily_solve(
    *,
    forecast_table: pd.DataFrame,
    returns: pd.DataFrame,
    current_weights: pd.Series,
    active_tickers: list[str],
    params: dict[str, Any],
    optimizer_constraint_params: dict[str, Any],
    effective_cash_ticker: str,
    execution_fraction: float,
    prices: pd.DataFrame | None = None,
    market_ticker: str | None = None,
    success_target_source: str = "SCENARIO_WEIGHTED_RF_SHARPE_OPTIMAL",
    failure_target_source: str = "HOLD_SOLVER_FAILED",
) -> ScenarioWeightedDailySolveResult:
    """Build scenarios, solve the direct optimizer, validate, and fallback.

    The final target is determined only by the scenario-weighted RF-adjusted
    Sharpe optimizer. If the solve or post-solver validation fails, the target
    falls back to current weights and reports ``failure_target_source``.
    """

    solver_assets = pd.Index([str(ticker) for ticker in forecast_table.index], name="ticker")
    active_index = pd.Index([str(ticker) for ticker in active_tickers], name="ticker")
    solver_current_weights = _normalize_weight_series(
        current_weights.reindex(solver_assets).fillna(0.0),
        solver_assets,
        fallback_ticker=effective_cash_ticker,
    )
    solver_config = scenario_weighted_solver_config(params)
    solver_max_weights = {
        asset: float(dict(params.get("asset_max_weights", {})).get(asset, 1.0))
        for asset in solver_assets
    }
    solver_asset_groups = {
        asset: str(dict(params.get("group_map", {})).get(asset, "unknown"))
        for asset in solver_assets
    }
    solver_group_limits = {
        str(group): float(limit)
        for group, limit in dict(params.get("group_limits", {})).items()
    }
    solver_constraints: dict[str, Any] = {
        "max_weights": solver_max_weights,
        "asset_groups": solver_asset_groups,
        "group_limits": solver_group_limits,
        "current_weights": solver_current_weights,
        "max_turnover": float(solver_config.max_turnover),
        "config": solver_config,
    }

    warnings: list[str] = []
    scenario_inputs: list[ScenarioInput] = []
    solver_result = solver_failure_result("Scenario-weighted solver was not run.", solver_current_weights)
    try:
        scenario_config = {**optimizer_constraint_params, "solver": params.get("solver", {})}
        if prices is not None and not prices.empty:
            dynamic_probabilities = _dynamic_scenario_weighted_probabilities(
                prices=prices,
                returns=returns.reindex(columns=solver_assets),
                params=params,
                market_ticker=market_ticker,
            )
            scenario_config["scenario_weighted_probabilities"] = dynamic_probabilities.to_dict()
            scenario_config["scenario_weighted_probabilities_source"] = "dynamic_regime_probability_model"

        scenario_inputs = build_scenario_inputs(
            forecast_table=forecast_table.reindex(solver_assets),
            returns=returns.reindex(columns=solver_assets),
            config=scenario_config,
        )
        repaired_start = repair_weights_to_constraints(
            solver_current_weights,
            {**optimizer_constraint_params, "cash_ticker": effective_cash_ticker},
        )
        solver_x0 = None
        if bool(repaired_start.get("valid", False)):
            solver_x0 = _normalize_weight_series(
                pd.Series(repaired_start["weights"], dtype=float).reindex(solver_assets).fillna(0.0),
                solver_assets,
                fallback_ticker=effective_cash_ticker,
            )
        solver_result = solve_scenario_weighted_sharpe(
            current_weights=solver_current_weights,
            scenarios=scenario_inputs,
            max_weights=solver_max_weights,
            asset_groups=solver_asset_groups,
            group_limits=solver_group_limits,
            config=solver_config,
            x0=solver_x0,
        )
    except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        message = f"Scenario-weighted RF Sharpe solver failed: {exc}"
        warnings.append(message)
        solver_result = solver_failure_result(str(exc), solver_current_weights)
        scenario_inputs = []

    solver_validation = validate_solver_result(
        solver_result,
        solver_constraints,
        tolerance=1.0e-6,
    )
    solver_validation_ok = bool(solver_validation.get("ok", False))
    solver_failure_reason = "" if solver_validation_ok else str(
        solver_validation.get("failure_reason")
        or "; ".join(map(str, solver_validation.get("errors", [])))
        or solver_result.message
        or "post_solver_validation_failed"
    )
    solver_result.constraint_diagnostics.update(
        {
            "post_solver_validation": solver_validation,
            "solver_failed": not solver_validation_ok,
            "failure_reason": solver_failure_reason,
        }
    )
    if not solver_validation_ok:
        solver_result.success = False
        solver_result.status = int(solver_result.status) if solver_result.status else -2
        solver_result.message = solver_failure_reason
        warnings.append(f"Scenario-weighted solver post-validation failed: {solver_failure_reason}")

    if bool(solver_result.success):
        optimal_weights = _normalize_weight_series(
            solver_result.weights.reindex(active_index).fillna(0.0).astype(float),
            active_index,
            fallback_ticker=effective_cash_ticker,
        )
        executable_weights = apply_execution_fraction(
            current_weights=current_weights.reindex(active_index).fillna(0.0),
            optimal_weights=optimal_weights,
            execution_fraction=execution_fraction,
            fallback_ticker=effective_cash_ticker,
        ).reindex(active_index).fillna(0.0)
        final_target_source = success_target_source
    else:
        optimal_weights = _normalize_weight_series(
            current_weights.reindex(active_index).fillna(0.0).astype(float),
            active_index,
            fallback_ticker=effective_cash_ticker,
        )
        executable_weights = optimal_weights.copy()
        final_target_source = failure_target_source

    return ScenarioWeightedDailySolveResult(
        final_target_source=final_target_source,
        scenario_inputs=scenario_inputs,
        solver_config=solver_config,
        solver_constraints=solver_constraints,
        solver_current_weights=solver_current_weights,
        solver_result=solver_result,
        solver_validation=solver_validation,
        optimal_weights=optimal_weights,
        executable_weights=executable_weights,
        target_weights=executable_weights.copy(),
        solver_assets=solver_assets,
        warnings=warnings,
    )



def _dynamic_scenario_weighted_probabilities(
    *,
    returns: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
    market_ticker: str | None = None,
) -> pd.Series:
    """Map current market regime indicators to active six-scenario solver probabilities.

    Accepts either price history or return history. Tests and diagnostics may pass
    returns directly; the daily bot normally passes prices.
    """

    params = dict(params or {})
    effective_market_ticker = str(params.get("market_ticker") or market_ticker or "") or None

    if prices is None or prices.empty:
        if returns is None or returns.empty:
            price_proxy = pd.DataFrame()
        else:
            clean_returns = (
                returns.copy()
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            price_proxy = 100.0 * (1.0 + clean_returns).cumprod()
    else:
        price_proxy = prices

    indicators = compute_regime_indicators(
        prices=price_proxy,
        market_ticker=effective_market_ticker,
    )

    stress = float(indicators.stress_score)
    risk_on = float(indicators.risk_on_score)
    momentum = float(indicators.momentum_score)
    volatility = float(indicators.volatility_score)
    drawdown = float(indicators.drawdown_score)
    correlation = float(indicators.correlation_score)

    raw = pd.Series(
        {
            "bull_momentum": 0.12 + 0.30 * risk_on + 0.10 * momentum,
            "soft_landing": 0.18 + 0.18 * risk_on + 0.08 * (1.0 - volatility),
            "sideways_choppy": 0.14 + 0.12 * (1.0 - abs(risk_on - stress)),
            "inflation_shock": 0.08 + 0.10 * volatility + 0.04 * correlation,
            "growth_selloff": 0.08 + 0.22 * stress + 0.08 * drawdown,
            "liquidity_stress": 0.05 + 0.18 * stress + 0.08 * correlation,
        },
        dtype=float,
    )

    if stress > 0.65:
        raw.loc["bull_momentum"] *= 0.65
        raw.loc["soft_landing"] *= 0.80
        raw.loc["growth_selloff"] *= 1.20
        raw.loc["liquidity_stress"] *= 1.25

    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(raw.sum())
    if total <= 0.0:
        raw = pd.Series(
            {
                "bull_momentum": 0.25,
                "soft_landing": 0.25,
                "sideways_choppy": 0.20,
                "inflation_shock": 0.15,
                "growth_selloff": 0.10,
                "liquidity_stress": 0.05,
            },
            dtype=float,
        )
        total = float(raw.sum())

    result = raw / total
    result.attrs["probability_source"] = "dynamic_regime_probability_model"
    result.attrs["stress_score"] = stress
    result.attrs["risk_on_score"] = risk_on
    result.attrs["momentum_score"] = momentum
    result.attrs["volatility_score"] = volatility
    result.attrs["drawdown_score"] = drawdown
    result.attrs["correlation_score"] = correlation
    return result


def scenario_weighted_solver_config(params: dict[str, Any]) -> SolverConfig:
    """Build the solver config from flat params plus optional nested solver block."""

    solver_block = params.get("solver", {})
    solver_params = dict(solver_block) if isinstance(solver_block, dict) else {}
    merged = {**solver_params, **params}
    return SolverConfig(
        lambda_turnover=float(merged.get("lambda_turnover", merged.get("direct_scenario_lambda_turnover", 0.03))),
        lambda_concentration=float(
            merged.get("lambda_concentration", merged.get("direct_scenario_lambda_concentration", 0.01))
        ),
        lambda_downside=float(merged.get("lambda_downside", merged.get("direct_scenario_lambda_downside", 0.15))),
        eps_variance=float(merged.get("eps_variance", merged.get("sharpe_epsilon", 1.0e-10))),
        max_turnover=float(merged.get("max_turnover", 0.75)),
        ftol=float(merged.get("optimizer_ftol", merged.get("direct_scenario_optimizer_ftol", 1.0e-9))),
        maxiter=int(merged.get("optimizer_maxiter", merged.get("direct_scenario_optimizer_maxiter", 1000))),
    )


def solver_failure_result(message: str, weights: pd.Series) -> SolverResult:
    """Return a structured fail-closed solver result at current weights."""

    weights = weights.astype(float).copy()
    return SolverResult(
        success=False,
        status=-1,
        message=str(message),
        weights=weights,
        objective_value=0.0,
        weighted_sharpe=0.0,
        turnover=0.0,
        concentration=float(np.square(weights.to_numpy(dtype=float)).sum()),
        downside_penalty=0.0,
        per_scenario_metrics=pd.DataFrame(
            columns=[
                "scenario",
                "probability",
                "portfolio_return",
                "risk_free_return",
                "excess_return",
                "portfolio_volatility",
                "rf_adjusted_sharpe",
                "downside_shortfall",
            ]
        ),
        constraint_diagnostics={"solver_failed": True, "failure_reason": str(message), "warnings": [str(message)]},
    )


def apply_execution_fraction(
    *,
    current_weights: pd.Series,
    optimal_weights: pd.Series,
    execution_fraction: float,
    fallback_ticker: str | None,
) -> pd.Series:
    """Apply execution damping without creating a new candidate decision."""

    index = pd.Index([str(asset) for asset in optimal_weights.index], name="ticker")
    current = current_weights.reindex(index).fillna(0.0).astype(float)
    optimal = optimal_weights.reindex(index).fillna(0.0).astype(float)
    fraction = min(max(float(execution_fraction), 0.0), 1.0)
    blended = current + fraction * (optimal - current)
    return _normalize_weight_series(blended, index, fallback_ticker=fallback_ticker)


def _normalize_weight_series(
    weights: pd.Series,
    index: pd.Index,
    *,
    fallback_ticker: str | None = None,
) -> pd.Series:
    aligned = (
        weights.copy()
        .rename(index=str)
        .reindex(index)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    total = float(aligned.sum())
    if total > 0.0:
        return aligned / total
    fallback = str(fallback_ticker or "")
    if fallback and fallback in aligned.index:
        aligned.loc[fallback] = 1.0
    elif len(aligned.index) > 0:
        aligned.iloc[0] = 1.0
    return aligned
