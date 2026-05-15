"""Strict scenario-weighted RF-adjusted Sharpe solver.

This module is deliberately side-effect free. It performs no order execution,
no email sending and no file writes. Its job is to validate scenario inputs,
solve the continuous allocation problem, and return structured diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    minimize = None
    SCIPY_AVAILABLE = False


@dataclass(slots=True)
class ScenarioInput:
    """Single scenario with scenario-specific mu, Sigma, probability and RF."""

    name: str
    probability: float
    expected_returns: pd.Series
    covariance: pd.DataFrame
    risk_free_return: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SolverConfig:
    """Numerical and penalty configuration for the strict solver."""

    lambda_turnover: float
    lambda_concentration: float
    lambda_downside: float
    eps_variance: float
    max_turnover: float
    ftol: float
    maxiter: int


@dataclass(slots=True)
class SolverResult:
    """Structured result and diagnostics for a scenario-weighted solve."""

    success: bool
    status: int
    message: str
    weights: pd.Series
    objective_value: float
    weighted_sharpe: float
    turnover: float
    concentration: float
    downside_penalty: float
    per_scenario_metrics: pd.DataFrame
    constraint_diagnostics: dict[str, Any] = field(default_factory=dict)


def solve_scenario_weighted_sharpe(
    current_weights: pd.Series,
    scenarios: list[ScenarioInput],
    max_weights: dict[str, float],
    asset_groups: dict[str, str] | dict[str, list[str]],
    group_limits: dict[str, float],
    config: SolverConfig,
    min_group_weights: dict[str, float] | None = None,
    x0: pd.Series | None = None,
) -> SolverResult:
    """Solve the strict continuous scenario-weighted Sharpe allocation.

    Objective maximized internally:
        sum_s p_s * ((w^T mu_s - rf_s) / sqrt(w^T Sigma_s w + eps))
        - lambda_turnover * ||w - w_current||_1
        - lambda_concentration * sum_i w_i^2
        - lambda_downside * sum_s p_s * max(0, rf_s - w^T mu_s)

    SciPy minimizes the negative objective.
    """

    assets = _strict_asset_index(current_weights)
    scenario_inputs, probability_diagnostics = _prepare_scenarios(
        scenarios=scenarios,
        assets=assets,
        config=config,
    )
    current = _strict_numeric_series(current_weights, assets, "current_weights")
    max_weight_series = _required_numeric_series(pd.Series(max_weights), assets, "max_weights").clip(lower=0.0, upper=1.0)
    if float(max_weight_series.sum()) < 1.0 - 1.0e-10:
        raise ValueError(
            f"sum(max_weights) must be >= 1.0 for full investment; got {float(max_weight_series.sum()):.10f}."
        )
    if (max_weight_series <= 0.0).all():
        raise ValueError("max_weights must allow positive capacity for at least one asset.")

    asset_to_group = _normalize_asset_groups(asset_groups, assets)
    group_limit_series = pd.Series(group_limits, dtype=float).rename(index=str)
    min_group_series = (
        pd.Series(dtype=float)
        if min_group_weights is None
        else pd.Series(min_group_weights, dtype=float).rename(index=str)
    )
    start = _initial_weights(
        x0=x0,
        current=current,
        max_weights=max_weight_series,
        config=config,
        asset_to_group=asset_to_group,
        group_limits=group_limit_series,
        min_group_weights=min_group_series,
    )
    if not SCIPY_AVAILABLE or minimize is None:
        return _build_result(
            success=False,
            status=2,
            message="SciPy unavailable; returned validated initial weights.",
            weights=start,
            current=current,
            scenarios=scenario_inputs,
            config=config,
            max_weights=max_weight_series,
            asset_to_group=asset_to_group,
            group_limits=group_limit_series,
            min_group_weights=min_group_series,
            extra_diagnostics=probability_diagnostics,
        )

    current_values = current.to_numpy(dtype=float)
    asset_count = len(assets)
    start_values = start.to_numpy(dtype=float)
    start_delta = start_values - current_values
    # Represent exact L1 turnover with auxiliary variables:
    #   w - w_current = d_plus - d_minus
    #   ||w-w_current||_1 = sum(d_plus + d_minus)
    # This keeps SLSQP away from the non-differentiable abs() kink while
    # preserving the exact mathematical turnover definition.
    z0 = np.concatenate(
        [
            start_values,
            np.maximum(start_delta, 0.0),
            np.maximum(-start_delta, 0.0),
        ]
    )
    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda z, n=asset_count: float(np.sum(z[:n]) - 1.0)},
        {
            "type": "ineq",
            "fun": lambda z, n=asset_count: float(config.max_turnover - np.sum(z[n : 3 * n])),
        },
    ]
    for asset_index in range(asset_count):
        constraints.append(
            {
                "type": "eq",
                "fun": (
                    lambda z, idx=asset_index, n=asset_count: float(
                        z[idx] - current_values[idx] - z[n + idx] + z[2 * n + idx]
                    )
                ),
            }
        )
    group_indices = _group_indices(assets, asset_to_group)
    for group, indices in group_indices.items():
        if group in group_limit_series.index:
            limit = float(group_limit_series.loc[group])
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda z, idx=indices, lim=limit: float(lim - np.sum(z[:asset_count][idx])),
                }
            )
        if group in min_group_series.index:
            minimum = float(min_group_series.loc[group])
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda z, idx=indices, minimum=minimum: float(
                        np.sum(z[:asset_count][idx]) - minimum
                    ),
                }
            )

    try:
        result = minimize(
            fun=_negative_score_with_l1_auxiliary,
            x0=z0,
            args=(current_values, scenario_inputs, config),
            method="SLSQP",
            bounds=[
                *[(0.0, float(max_weight_series.loc[asset])) for asset in assets],
                *[(0.0, float(config.max_turnover)) for _ in assets],
                *[(0.0, float(config.max_turnover)) for _ in assets],
            ],
            constraints=constraints,
            options={"ftol": float(config.ftol), "maxiter": int(config.maxiter)},
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        return _build_result(
            success=False,
            status=3,
            message=f"SLSQP failed before returning a solution: {exc}",
            weights=start,
            current=current,
            scenarios=scenario_inputs,
            config=config,
            max_weights=max_weight_series,
            asset_to_group=asset_to_group,
            group_limits=group_limit_series,
            min_group_weights=min_group_series,
            extra_diagnostics=probability_diagnostics,
        )

    if getattr(result, "x", None) is None:
        return _build_result(
            success=False,
            status=int(getattr(result, "status", 4)),
            message="SLSQP did not return a solution vector.",
            weights=start,
            current=current,
            scenarios=scenario_inputs,
            config=config,
            max_weights=max_weight_series,
            asset_to_group=asset_to_group,
            group_limits=group_limit_series,
            min_group_weights=min_group_series,
            extra_diagnostics=probability_diagnostics,
        )

    weights = pd.Series(np.asarray(result.x, dtype=float)[:asset_count], index=assets, dtype=float)
    diagnostics = constraint_diagnostics(
        weights=weights,
        current_weights=current,
        max_weights=max_weight_series,
        asset_to_group=asset_to_group,
        group_limits=group_limit_series,
        min_group_weights=min_group_series,
        config=config,
    )
    diagnostics.update(probability_diagnostics)
    success = bool(result.success) and bool(diagnostics["feasible"])
    return _build_result(
        success=success,
        status=int(getattr(result, "status", 1)),
        message=str(getattr(result, "message", "")),
        weights=weights,
        current=current,
        scenarios=scenario_inputs,
        config=config,
        max_weights=max_weight_series,
        asset_to_group=asset_to_group,
        group_limits=group_limit_series,
        min_group_weights=min_group_series,
        extra_diagnostics=diagnostics,
    )


def evaluate_weights(
    weights: pd.Series,
    current_weights: pd.Series,
    scenarios: list[ScenarioInput],
    config: SolverConfig,
) -> SolverResult:
    """Evaluate the strict objective for a fixed weight vector."""

    assets = _strict_asset_index(current_weights)
    scenario_inputs, probability_diagnostics = _prepare_scenarios(
        scenarios=scenarios,
        assets=assets,
        config=config,
    )
    current = _strict_numeric_series(current_weights, assets, "current_weights")
    weights_aligned = _strict_numeric_series(weights, assets, "weights")
    return _build_result(
        success=True,
        status=0,
        message="evaluated",
        weights=weights_aligned,
        current=current,
        scenarios=scenario_inputs,
        config=config,
        max_weights=pd.Series(1.0, index=assets),
        asset_to_group={asset: "" for asset in assets},
        group_limits=pd.Series(dtype=float),
        min_group_weights=pd.Series(dtype=float),
        extra_diagnostics=probability_diagnostics,
    )


def _strict_asset_index(current_weights: pd.Series) -> pd.Index:
    if current_weights.empty:
        raise ValueError("current_weights must not be empty.")
    assets = pd.Index([str(asset) for asset in current_weights.index], name="ticker")
    if assets.has_duplicates:
        duplicates = assets[assets.duplicated()].unique().tolist()
        raise ValueError(f"current_weights contains duplicate assets: {duplicates}")
    return assets


def _strict_numeric_series(values: pd.Series, assets: pd.Index, label: str) -> pd.Series:
    series = values.copy()
    series.index = pd.Index([str(asset) for asset in series.index], name="ticker")
    if list(series.index) != list(assets):
        missing = [asset for asset in assets if asset not in set(series.index)]
        extra = [asset for asset in series.index if asset not in set(assets)]
        raise ValueError(
            f"{label} must use the exact asset order {list(assets)}; missing={missing}; extra={extra}."
        )
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        bad_assets = numeric.index[numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=float))].tolist()
        raise ValueError(f"{label} contains non-finite values for assets: {bad_assets}")
    return numeric.astype(float)


def _required_numeric_series(values: pd.Series, assets: pd.Index, label: str) -> pd.Series:
    """Validate that all required assets exist, then return in solver order."""

    series = values.copy()
    series.index = pd.Index([str(asset) for asset in series.index], name="ticker")
    missing = [asset for asset in assets if asset not in set(series.index)]
    extra = [asset for asset in series.index if asset not in set(assets)]
    if missing or extra:
        raise ValueError(f"{label} assets mismatch; missing={missing}; extra={extra}.")
    numeric = pd.to_numeric(series.reindex(assets), errors="coerce")
    bad_mask = numeric.isna() | pd.Series(~np.isfinite(numeric.to_numpy(dtype=float)), index=assets)
    if bad_mask.any():
        raise ValueError(f"{label} contains non-finite values for assets: {bad_mask[bad_mask].index.tolist()}")
    return numeric.astype(float)


def _prepare_scenarios(
    *,
    scenarios: list[ScenarioInput],
    assets: pd.Index,
    config: SolverConfig,
) -> tuple[list[ScenarioInput], dict[str, Any]]:
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    raw_probabilities = np.array([float(scenario.probability) for scenario in scenarios], dtype=float)
    if (raw_probabilities < 0.0).any() or not np.isfinite(raw_probabilities).all():
        raise ValueError("Scenario probabilities must be finite and non-negative.")
    probability_sum = float(raw_probabilities.sum())
    if probability_sum <= 0.0:
        raise ValueError("Scenario probabilities must contain positive probability mass.")
    normalized_probabilities = raw_probabilities / probability_sum
    diagnostics = {
        "probability_sum_before_normalization": probability_sum,
        "probabilities_normalized": abs(probability_sum - 1.0) > 1.0e-10,
        "warnings": [],
    }
    if diagnostics["probabilities_normalized"]:
        diagnostics["warnings"].append(
            f"scenario_probabilities_normalized_from_sum_{probability_sum:.12f}"
        )

    prepared: list[ScenarioInput] = []
    for scenario, probability in zip(scenarios, normalized_probabilities, strict=True):
        mu = _strict_numeric_series(scenario.expected_returns, assets, f"{scenario.name}.expected_returns")
        covariance = _strict_covariance(
            covariance=scenario.covariance,
            assets=assets,
            eps_variance=float(config.eps_variance),
            label=f"{scenario.name}.covariance",
        )
        prepared.append(
            ScenarioInput(
                name=str(scenario.name),
                probability=float(probability),
                expected_returns=mu,
                covariance=covariance,
                risk_free_return=float(scenario.risk_free_return),
                metadata=dict(getattr(scenario, "metadata", {}) or {}),
            )
        )
    return prepared, diagnostics


def _strict_covariance(
    *,
    covariance: pd.DataFrame,
    assets: pd.Index,
    eps_variance: float,
    label: str,
) -> pd.DataFrame:
    matrix = covariance.copy()
    matrix.index = pd.Index([str(asset) for asset in matrix.index], name="ticker")
    matrix.columns = pd.Index([str(asset) for asset in matrix.columns], name="ticker")
    if list(matrix.index) != list(assets) or list(matrix.columns) != list(assets):
        missing_rows = [asset for asset in assets if asset not in set(matrix.index)]
        missing_cols = [asset for asset in assets if asset not in set(matrix.columns)]
        extra_rows = [asset for asset in matrix.index if asset not in set(assets)]
        extra_cols = [asset for asset in matrix.columns if asset not in set(assets)]
        raise ValueError(
            f"{label} must use exact asset order {list(assets)} for rows and columns; "
            f"missing_rows={missing_rows}; missing_cols={missing_cols}; "
            f"extra_rows={extra_rows}; extra_cols={extra_cols}."
        )
    values = matrix.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite covariance values.")
    values = 0.5 * (values + values.T)
    try:
        min_eigenvalue = float(np.linalg.eigvalsh(values).min()) if values.size else 0.0
    except np.linalg.LinAlgError:
        min_eigenvalue = float("-inf")
    if min_eigenvalue < max(float(eps_variance), 0.0):
        values = values + float(eps_variance) * np.eye(len(assets))
    return pd.DataFrame(values, index=assets, columns=assets)


def _normalize_asset_groups(
    asset_groups: dict[str, str] | dict[str, list[str]],
    assets: pd.Index,
) -> dict[str, str]:
    """Accept either asset->group or group->assets and return asset->group."""

    if not asset_groups:
        return {asset: "" for asset in assets}
    first_value = next(iter(asset_groups.values()))
    asset_to_group: dict[str, str] = {}
    if isinstance(first_value, (list, tuple, set)):
        for group, group_assets in asset_groups.items():  # type: ignore[union-attr]
            for asset in group_assets:
                asset_to_group[str(asset)] = str(group)
    else:
        asset_to_group = {str(asset): str(group) for asset, group in asset_groups.items()}  # type: ignore[union-attr]
    missing_assets = [asset for asset in assets if asset not in asset_to_group]
    if missing_assets:
        raise ValueError(f"asset_groups missing assets: {missing_assets}")
    return {asset: asset_to_group[asset] for asset in assets}


def _group_indices(assets: pd.Index, asset_to_group: dict[str, str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, asset in enumerate(assets):
        group = str(asset_to_group.get(str(asset), ""))
        if not group:
            continue
        groups.setdefault(group, []).append(index)
    return groups


def _initial_weights(
    *,
    x0: pd.Series | None,
    current: pd.Series,
    max_weights: pd.Series,
    config: SolverConfig,
    asset_to_group: dict[str, str],
    group_limits: pd.Series,
    min_group_weights: pd.Series,
) -> pd.Series:
    # 1. Explicit x0: normalize and require feasibility. A bad user-provided
    # start is safer as a clear error than as a hidden fallback.
    if x0 is not None:
        start = _normalize_to_budget(_required_numeric_series(x0, current.index, "x0"), "x0")
        diagnostics = constraint_diagnostics(
            weights=start,
            current_weights=current,
            max_weights=max_weights,
            asset_to_group=asset_to_group,
            group_limits=group_limits,
            min_group_weights=min_group_weights,
            config=config,
        )
        if diagnostics["feasible"]:
            return start
        raise ValueError(f"x0 is not feasible after normalization: {diagnostics['errors']}")

    # 2. Use current weights if they already satisfy all constraints.
    current_diagnostics = constraint_diagnostics(
        weights=current,
        current_weights=current,
        max_weights=max_weights,
        asset_to_group=asset_to_group,
        group_limits=group_limits,
        min_group_weights=min_group_weights,
        config=config,
    )
    if current_diagnostics["feasible"]:
        return current

    # 3. Clip current weights to bounds and renormalize within caps.
    clipped_current = _normalize_with_caps(current.clip(lower=0.0), max_weights, "clipped current_weights")
    clipped_diagnostics = constraint_diagnostics(
        weights=clipped_current,
        current_weights=current,
        max_weights=max_weights,
        asset_to_group=asset_to_group,
        group_limits=group_limits,
        min_group_weights=min_group_weights,
        config=config,
    )
    if clipped_diagnostics["feasible"]:
        return clipped_current

    # 4. Build an equal-weight allocation within caps.
    equal_capped = _equal_weight_within_caps(max_weights)
    equal_diagnostics = constraint_diagnostics(
        weights=equal_capped,
        current_weights=current,
        max_weights=max_weights,
        asset_to_group=asset_to_group,
        group_limits=group_limits,
        min_group_weights=min_group_weights,
        config=config,
    )
    if equal_diagnostics["feasible"]:
        return equal_capped

    # 5. Fail clearly if no robust starting point exists.
    raise ValueError(
        "Unable to construct feasible x0. "
        f"current_errors={current_diagnostics['errors']}; "
        f"clipped_errors={clipped_diagnostics['errors']}; "
        f"equal_weight_errors={equal_diagnostics['errors']}"
    )


def _normalize_to_budget(weights: pd.Series, label: str) -> pd.Series:
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{label} must have positive finite weight sum before normalization.")
    normalized = weights.astype(float) / total
    if abs(float(normalized.sum()) - 1.0) > 1.0e-10:
        raise ValueError(f"{label} could not be normalized to full investment.")
    return normalized


def _normalize_with_caps(weights: pd.Series, max_weights: pd.Series, label: str) -> pd.Series:
    if float(max_weights.sum()) < 1.0 - 1.0e-10:
        raise ValueError(
            f"sum(max_weights) must be >= 1.0 for {label}; got {float(max_weights.sum()):.10f}."
        )
    clipped = np.minimum(weights.reindex(max_weights.index).fillna(0.0).astype(float), max_weights.astype(float))
    clipped = pd.Series(clipped, index=max_weights.index, dtype=float).clip(lower=0.0)
    residual = 1.0 - float(clipped.sum())
    while residual > 1.0e-10:
        capacity = (max_weights - clipped).clip(lower=0.0)
        available = capacity[capacity > 1.0e-12]
        if available.empty:
            raise ValueError(f"No remaining capacity while normalizing {label}.")
        addition = min(residual / float(len(available)), float(available.min()))
        clipped.loc[available.index] += addition
        residual = 1.0 - float(clipped.sum())
    if float(clipped.sum()) > 1.0 + 1.0e-10:
        clipped = clipped / float(clipped.sum())
    return clipped.astype(float)


def _equal_weight_within_caps(max_weights: pd.Series) -> pd.Series:
    equal = pd.Series(1.0 / len(max_weights.index), index=max_weights.index, dtype=float)
    return _normalize_with_caps(equal, max_weights, "equal-weight allocation")


def _score_components(
    *,
    values: np.ndarray,
    current_values: np.ndarray,
    scenarios: list[ScenarioInput],
    config: SolverConfig,
    turnover_override: float | None = None,
) -> dict[str, Any]:
    """Compute exactly the scenario-weighted RF-adjusted Sharpe objective.

    This is intentionally not a mixture mean-return divided by mixture
    volatility. The same single weight vector is evaluated in each scenario
    with that scenario's own expected returns, covariance matrix and risk-free
    return, then the per-scenario RF-adjusted Sharpes are probability-weighted.
    """

    weighted_sharpe = 0.0
    downside_penalty = 0.0
    rows: list[dict[str, float | str]] = []
    for scenario in scenarios:
        mu = scenario.expected_returns.to_numpy(dtype=float)
        sigma = scenario.covariance.to_numpy(dtype=float)
        rf = float(scenario.risk_free_return)
        probability = float(scenario.probability)

        port_ret = float(values @ mu)
        port_var = float(values @ sigma @ values)
        port_vol = float(np.sqrt(max(port_var, 0.0) + float(config.eps_variance)))

        sharpe_s = (port_ret - rf) / port_vol
        weighted_sharpe += probability * sharpe_s

        downside_s = max(0.0, rf - port_ret)
        downside_penalty += probability * downside_s
        rows.append(
            {
                "scenario": scenario.name,
                "probability": probability,
                "portfolio_return": port_ret,
                "risk_free_return": rf,
                "excess_return": port_ret - rf,
                "variance": port_var,
                "variance_for_volatility": max(port_var, 0.0),
                "volatility": port_vol,
                "rf_adjusted_sharpe": sharpe_s,
                "rf_adjusted_sharpe_alias": sharpe_s,
                "downside_shortfall": downside_s,
                "weighted_sharpe_contribution": probability * sharpe_s,
            }
        )
    turnover = _turnover_values(values, current_values) if turnover_override is None else float(turnover_override)
    concentration = float(np.square(values).sum())
    final_score = (
        weighted_sharpe
        - float(config.lambda_turnover) * turnover
        - float(config.lambda_concentration) * concentration
        - float(config.lambda_downside) * downside_penalty
    )
    return {
        "objective_value": final_score,
        "weighted_sharpe": weighted_sharpe,
        "turnover": turnover,
        "concentration": concentration,
        "downside_penalty": downside_penalty,
        "per_scenario_metrics": pd.DataFrame(rows),
    }


def _negative_score(
    values: np.ndarray,
    current_values: np.ndarray,
    scenarios: list[ScenarioInput],
    config: SolverConfig,
) -> float:
    """SciPy minimizes this negative maximization objective."""

    return -float(
        _score_components(
            values=np.asarray(values, dtype=float),
            current_values=current_values,
            scenarios=scenarios,
            config=config,
        )["objective_value"]
    )


def _negative_score_with_l1_auxiliary(
    values_with_auxiliaries: np.ndarray,
    current_values: np.ndarray,
    scenarios: list[ScenarioInput],
    config: SolverConfig,
) -> float:
    """SLSQP objective with exact L1 turnover represented by auxiliary vars."""

    values = np.asarray(values_with_auxiliaries, dtype=float)
    n = len(current_values)
    turnover = float(np.sum(values[n : 3 * n]))
    return -float(
        _score_components(
            values=values[:n],
            current_values=current_values,
            scenarios=scenarios,
            config=config,
            turnover_override=turnover,
        )["objective_value"]
    )


def _turnover_values(weights: np.ndarray, current_weights: np.ndarray) -> float:
    """Exact L1 turnover: ||w - w_current||_1."""

    return float(np.abs(np.asarray(weights, dtype=float) - np.asarray(current_weights, dtype=float)).sum())


def constraint_diagnostics(
    *,
    weights: pd.Series,
    current_weights: pd.Series,
    max_weights: pd.Series,
    asset_to_group: dict[str, str],
    group_limits: pd.Series,
    min_group_weights: pd.Series,
    config: SolverConfig,
    tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    weights = _strict_numeric_series(weights, current_weights.index, "weights")
    max_weights = _strict_numeric_series(max_weights, current_weights.index, "max_weights")
    current_weights = _strict_numeric_series(current_weights, current_weights.index, "current_weights")
    errors: list[str] = []
    budget_sum = float(weights.sum())
    turnover = _turnover_values(weights.to_numpy(dtype=float), current_weights.to_numpy(dtype=float))
    if abs(budget_sum - 1.0) > tolerance:
        errors.append(f"budget_sum={budget_sum:.10f}")
    if (weights < -tolerance).any():
        errors.append("negative_weight")
    asset_excess = weights - max_weights
    if (asset_excess > tolerance).any():
        errors.append("asset_max_violation")
    if turnover > float(config.max_turnover) + tolerance:
        errors.append(f"turnover={turnover:.10f}")

    group_weights: dict[str, float] = {}
    group_violations: dict[str, dict[str, float]] = {}
    all_groups = sorted(
        set(asset_to_group.values())
        | set(group_limits.index.astype(str).tolist())
        | set(min_group_weights.index.astype(str).tolist())
    )
    for group in all_groups:
        if not group:
            continue
        group_assets = [asset for asset in weights.index if asset_to_group.get(str(asset), "") == group]
        actual = float(weights.reindex(group_assets).sum()) if group_assets else 0.0
        group_weights[group] = actual
        if group in group_limits.index and actual > float(group_limits.loc[group]) + tolerance:
            errors.append(f"group_max_{group}")
            group_violations[group] = {
                "actual": actual,
                "limit": float(group_limits.loc[group]),
                "excess": actual - float(group_limits.loc[group]),
            }
        if group in min_group_weights.index and actual < float(min_group_weights.loc[group]) - tolerance:
            errors.append(f"group_min_{group}")
            group_violations[group] = {
                "actual": actual,
                "limit": float(min_group_weights.loc[group]),
                "shortfall": float(min_group_weights.loc[group]) - actual,
            }

    return {
        "feasible": not errors,
        "errors": errors,
        "budget_sum": budget_sum,
        "turnover": turnover,
        "max_turnover": float(config.max_turnover),
        "turnover_definition": "sum(abs(w-current_w))",
        "asset_max_violations": {
            str(asset): {"actual": float(weights.loc[asset]), "limit": float(max_weights.loc[asset])}
            for asset in weights.index
            if float(asset_excess.loc[asset]) > tolerance
        },
        "group_weights": group_weights,
        "group_violations": group_violations,
    }


def validate_solver_result(
    result: SolverResult,
    constraints: dict[str, Any],
    tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Validate a solver result as the final post-SLSQP safety gate.

    The solver already enforces constraints internally. This function is the
    explicit downstream guard used by the daily bot before an optimized target
    is allowed to become the final executable target.
    """

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    failure_reason = "" if bool(result.success) else str(result.message or "solver_success_false")
    if not bool(result.success):
        warnings.append(f"solver_success_false: {failure_reason}")

    weights = result.weights.copy()
    try:
        assets = pd.Index([str(asset) for asset in weights.index], name="ticker")
        weights.index = assets
        numeric_weights = pd.to_numeric(weights, errors="coerce").astype(float)
    except (TypeError, ValueError) as exc:
        return {
            "ok": False,
            "solver_failed": True,
            "failure_reason": failure_reason or str(exc),
            "errors": [f"invalid_weights: {exc}"],
            "warnings": warnings,
            "checks": checks,
        }

    finite_weights = bool(np.isfinite(numeric_weights.to_numpy(dtype=float)).all())
    checks["weights_finite"] = finite_weights
    if not finite_weights:
        errors.append("weights_non_finite")

    budget_sum = float(numeric_weights.sum()) if finite_weights else float("nan")
    checks["budget_sum"] = budget_sum
    if not np.isfinite(budget_sum) or abs(budget_sum - 1.0) > tolerance:
        errors.append(f"budget_sum={budget_sum:.10f}")

    min_weight = float(numeric_weights.min()) if len(numeric_weights) else float("nan")
    checks["min_weight"] = min_weight
    if np.isfinite(min_weight) and min_weight < -tolerance:
        errors.append("negative_weight")

    max_weights_obj = constraints.get("max_weights")
    asset_max_violations: dict[str, dict[str, float]] = {}
    if max_weights_obj is not None:
        try:
            max_weights = pd.Series(max_weights_obj, dtype=float).rename(index=str).reindex(assets)
            if max_weights.isna().any() or not np.isfinite(max_weights.to_numpy(dtype=float)).all():
                missing = max_weights[max_weights.isna()].index.astype(str).tolist()
                errors.append(f"max_weights_invalid_or_missing={missing}")
            else:
                excess = numeric_weights - max_weights
                for asset in assets:
                    if float(excess.loc[asset]) > tolerance:
                        asset_max_violations[str(asset)] = {
                            "actual": float(numeric_weights.loc[asset]),
                            "limit": float(max_weights.loc[asset]),
                            "excess": float(excess.loc[asset]),
                        }
                if asset_max_violations:
                    errors.append("asset_max_violation")
        except (TypeError, ValueError) as exc:
            errors.append(f"max_weights_invalid: {exc}")
    else:
        warnings.append("max_weights_missing_from_post_solver_validation")
    checks["asset_max_violations"] = asset_max_violations

    group_violations: dict[str, dict[str, float]] = {}
    raw_asset_groups = constraints.get("asset_groups", constraints.get("group_map", {}))
    raw_group_limits = constraints.get("group_limits", {})
    raw_min_group_weights = constraints.get("min_group_weights", {})
    if raw_asset_groups and raw_group_limits is not None:
        try:
            asset_to_group = _normalize_asset_groups(raw_asset_groups, assets)
            group_limits = pd.Series(raw_group_limits, dtype=float).rename(index=str)
            min_group_weights = pd.Series(raw_min_group_weights, dtype=float).rename(index=str)
            all_groups = sorted(
                set(asset_to_group.values())
                | set(group_limits.index.astype(str).tolist())
                | set(min_group_weights.index.astype(str).tolist())
            )
            group_weights: dict[str, float] = {}
            for group in all_groups:
                group_assets = [asset for asset in assets if asset_to_group.get(str(asset), "") == group]
                actual = float(numeric_weights.reindex(group_assets).fillna(0.0).sum()) if group_assets else 0.0
                group_weights[str(group)] = actual
                if group in group_limits.index and actual > float(group_limits.loc[group]) + tolerance:
                    group_violations[str(group)] = {
                        "actual": actual,
                        "limit": float(group_limits.loc[group]),
                        "excess": actual - float(group_limits.loc[group]),
                    }
                if group in min_group_weights.index and actual < float(min_group_weights.loc[group]) - tolerance:
                    group_violations[str(group)] = {
                        "actual": actual,
                        "limit": float(min_group_weights.loc[group]),
                        "shortfall": float(min_group_weights.loc[group]) - actual,
                    }
            checks["group_weights"] = group_weights
            if group_violations:
                errors.append("group_limit_violation")
        except (TypeError, ValueError) as exc:
            errors.append(f"group_limits_invalid: {exc}")
    else:
        warnings.append("group_limits_missing_from_post_solver_validation")
    checks["group_violations"] = group_violations

    current_weights_obj = constraints.get("current_weights")
    max_turnover_obj = constraints.get("max_turnover")
    config_obj = constraints.get("config")
    if max_turnover_obj is None and isinstance(config_obj, SolverConfig):
        max_turnover_obj = config_obj.max_turnover
    if current_weights_obj is not None and max_turnover_obj is not None:
        try:
            current_weights = pd.Series(current_weights_obj, dtype=float).rename(index=str).reindex(assets)
            if current_weights.isna().any() or not np.isfinite(current_weights.to_numpy(dtype=float)).all():
                errors.append("current_weights_invalid_or_missing")
            else:
                turnover = _turnover_values(
                    numeric_weights.to_numpy(dtype=float),
                    current_weights.to_numpy(dtype=float),
                )
                checks["turnover"] = turnover
                checks["max_turnover"] = float(max_turnover_obj)
                if turnover > float(max_turnover_obj) + tolerance:
                    errors.append(f"turnover={turnover:.10f}")
        except (TypeError, ValueError) as exc:
            errors.append(f"turnover_validation_invalid: {exc}")
    else:
        warnings.append("turnover_inputs_missing_from_post_solver_validation")

    metrics = result.per_scenario_metrics.copy()
    if metrics.empty:
        errors.append("per_scenario_metrics_empty")
    else:
        volatility_column = "portfolio_volatility" if "portfolio_volatility" in metrics.columns else "volatility"
        if volatility_column not in metrics.columns:
            errors.append("scenario_volatility_missing")
        else:
            vols = pd.to_numeric(metrics[volatility_column], errors="coerce")
            vols_ok = bool((vols > 0.0).all() and np.isfinite(vols.to_numpy(dtype=float)).all())
            checks["scenario_vols_positive"] = vols_ok
            if not vols_ok:
                errors.append("scenario_volatility_non_positive_or_non_finite")

        sharpe_column = (
            "rf_adjusted_sharpe"
            if "rf_adjusted_sharpe" in metrics.columns
            else "rf_adjusted_sharpe_alias"
        )
        if sharpe_column not in metrics.columns:
            errors.append("scenario_sharpe_missing")
        else:
            sharpes = pd.to_numeric(metrics[sharpe_column], errors="coerce")
            sharpes_ok = bool(np.isfinite(sharpes.to_numpy(dtype=float)).all())
            checks["scenario_sharpes_finite"] = sharpes_ok
            if not sharpes_ok:
                errors.append("scenario_sharpe_non_finite")

    objective_finite = bool(np.isfinite(float(result.objective_value)))
    checks["objective_finite"] = objective_finite
    if not objective_finite:
        errors.append("objective_non_finite")

    ok = bool(result.success) and not errors
    return {
        "ok": ok,
        "solver_failed": not bool(result.success),
        "failure_reason": failure_reason,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "budget_sum": budget_sum,
        "asset_max_violations": asset_max_violations,
        "group_violations": group_violations,
    }


def _build_result(
    *,
    success: bool,
    status: int,
    message: str,
    weights: pd.Series,
    current: pd.Series,
    scenarios: list[ScenarioInput],
    config: SolverConfig,
    max_weights: pd.Series,
    asset_to_group: dict[str, str],
    group_limits: pd.Series,
    min_group_weights: pd.Series,
    extra_diagnostics: dict[str, Any],
) -> SolverResult:
    components = _score_components(
        values=weights.to_numpy(dtype=float),
        current_values=current.to_numpy(dtype=float),
        scenarios=scenarios,
        config=config,
    )
    diagnostics = constraint_diagnostics(
        weights=weights,
        current_weights=current,
        max_weights=max_weights,
        asset_to_group=asset_to_group,
        group_limits=group_limits,
        min_group_weights=min_group_weights,
        config=config,
    )
    diagnostics.update(extra_diagnostics)
    return SolverResult(
        success=bool(success) and bool(diagnostics["feasible"]),
        status=int(status),
        message=str(message),
        weights=weights.astype(float),
        objective_value=float(components["objective_value"]),
        weighted_sharpe=float(components["weighted_sharpe"]),
        turnover=float(components["turnover"]),
        concentration=float(components["concentration"]),
        downside_penalty=float(components["downside_penalty"]),
        per_scenario_metrics=components["per_scenario_metrics"],
        constraint_diagnostics=diagnostics,
    )
