"""Scenario-dependent covariance and portfolio risk/return diagnostics.

This module is report-first by default: it computes covariance-aware scenario
mixture metrics without changing the production selection objective unless the
caller explicitly opts into the scenario objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asset_universe import DEFENSIVE_GROUPS, EQUITY_LIKE_GROUPS, get_group_map
from risk_free import risk_free_return_from_params


DEFAULT_SCENARIO_PROBABILITIES: dict[str, float] = {
    "base": 0.35,
    "risk_on": 0.15,
    "risk_off": 0.15,
    "rates_up": 0.10,
    "rates_down": 0.07,
    "commodity_up": 0.08,
    "equity_stress": 0.10,
}


@dataclass(slots=True)
class ScenarioRiskDistribution:
    """Scenario distribution with scenario-specific mu and covariance."""

    as_of: pd.Timestamp
    expected_returns: pd.DataFrame
    probabilities: pd.Series
    covariance_matrices: dict[str, pd.DataFrame]
    baseline_daily_covariance: pd.DataFrame
    baseline_covariance_horizon: pd.DataFrame
    baseline_correlation: pd.DataFrame
    summary: pd.DataFrame
    warnings: list[str]

    @property
    def scenario_names(self) -> list[str]:
        return [str(name) for name in self.probabilities.index.tolist()]

    @property
    def assets(self) -> pd.Index:
        return pd.Index(self.expected_returns.columns, name="ticker")


def _to_float_series(values: pd.Series, index: pd.Index, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(values.reindex(index), errors="coerce").fillna(float(default)).astype(float)


def _clean_covariance(matrix: pd.DataFrame, *, jitter: float) -> pd.DataFrame:
    values = matrix.to_numpy(dtype=float, copy=True)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = 0.5 * (values + values.T)
    if values.size:
        eigvals, eigvecs = np.linalg.eigh(values)
        eigvals = np.clip(eigvals, float(jitter), None)
        values = (eigvecs * eigvals) @ eigvecs.T
        values = 0.5 * (values + values.T)
        values[np.diag_indices_from(values)] = np.maximum(np.diag(values), float(jitter))
    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns)


def _cov_to_corr(covariance: pd.DataFrame) -> pd.DataFrame:
    values = covariance.to_numpy(dtype=float, copy=True)
    diag = np.sqrt(np.maximum(np.diag(values), 0.0))
    denom = np.outer(diag, diag)
    corr_values = np.divide(values, denom, out=np.zeros_like(values), where=denom > 0.0)
    np.fill_diagonal(corr_values, 1.0)
    corr_values = np.clip(corr_values, -1.0, 1.0)
    return pd.DataFrame(corr_values, index=covariance.index, columns=covariance.columns)


def compute_baseline_covariance(
    returns: pd.DataFrame,
    *,
    as_of: pd.Timestamp | str,
    assets: list[str] | pd.Index,
    lookback: int,
    horizon_days: int,
    shrink_alpha: float,
    jitter: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Compute daily and horizon covariance from trailing daily returns."""

    warnings: list[str] = []
    index = pd.Index([str(asset) for asset in assets], name="ticker")
    as_of_ts = pd.Timestamp(as_of)
    if returns.empty:
        raise ValueError("returns must not be empty for covariance estimation.")
    window = (
        returns.reindex(columns=index)
        .loc[:as_of_ts]
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
        .tail(max(int(lookback), 2))
    )
    if len(window) < max(20, min(int(lookback), 20)):
        warnings.append(f"low_covariance_history: observations={len(window)}")
    window = window.fillna(0.0)
    if len(window) < 2:
        raise ValueError("At least two return observations are required for covariance estimation.")

    sample_daily = window.cov().reindex(index=index, columns=index).fillna(0.0)
    diag_daily = pd.DataFrame(
        np.diag(np.diag(sample_daily.to_numpy(dtype=float))),
        index=index,
        columns=index,
    )
    alpha = min(max(float(shrink_alpha), 0.0), 1.0)
    daily = alpha * sample_daily + (1.0 - alpha) * diag_daily
    daily = _clean_covariance(daily, jitter=float(jitter))
    horizon = _clean_covariance(daily * max(float(horizon_days), 0.0), jitter=float(jitter))
    correlation = _cov_to_corr(horizon)
    return daily, horizon, correlation, warnings


def _normalized_probabilities(params: dict[str, Any]) -> tuple[pd.Series, list[str]]:
    raw = params.get("scenario_probabilities", DEFAULT_SCENARIO_PROBABILITIES)
    probabilities = pd.Series(dict(raw), dtype=float)
    warnings: list[str] = []
    probabilities = probabilities.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(probabilities.sum())
    if total <= 0.0:
        probabilities = pd.Series(DEFAULT_SCENARIO_PROBABILITIES, dtype=float)
        total = float(probabilities.sum())
        warnings.append("scenario_probabilities_invalid_used_defaults")
    if abs(total - 1.0) > 1e-8:
        warnings.append(f"scenario_probabilities_normalized_from_sum_{total:.6f}")
        probabilities = probabilities / max(total, 1e-12)
    return probabilities, warnings


def _scenario_shocks(index: pd.Index, params: dict[str, Any]) -> dict[str, pd.Series]:
    group_map = pd.Series(get_group_map(), dtype=object).reindex(index).fillna("")
    equity_groups = set(map(str, params.get("equity_like_groups", EQUITY_LIKE_GROUPS)))
    defensive_groups = set(map(str, params.get("defensive_groups", DEFENSIVE_GROUPS)))
    commodity_groups = {"commodities"}
    hedge_groups = {"hedge"}
    bond_groups = {"bonds"}
    cash_groups = {"cash"}

    shocks = {name: pd.Series(0.0, index=index, dtype=float) for name in DEFAULT_SCENARIO_PROBABILITIES}
    risk_on = float(params.get("scenario_risk_on_equity_shock", 0.012))
    risk_off = float(params.get("scenario_risk_off_equity_shock", -0.018))
    rates_up = float(params.get("scenario_rates_up_duration_shock", -0.010))
    rates_down = float(params.get("scenario_rates_down_duration_shock", 0.008))
    commodity_up = float(params.get("scenario_commodity_up_shock", 0.018))
    equity_stress = float(params.get("scenario_equity_stress_shock", -0.035))
    defensive_carry = float(params.get("scenario_defensive_carry_shock", 0.001))

    for ticker, group in group_map.items():
        group_name = str(group)
        if group_name in equity_groups:
            shocks["risk_on"].loc[ticker] += risk_on
            shocks["risk_off"].loc[ticker] += risk_off
            shocks["rates_up"].loc[ticker] += 0.35 * rates_up
            shocks["rates_down"].loc[ticker] += 0.25 * abs(rates_down)
            shocks["equity_stress"].loc[ticker] += equity_stress
        if group_name in defensive_groups or group_name in cash_groups:
            shocks["risk_on"].loc[ticker] -= 0.25 * defensive_carry
            shocks["risk_off"].loc[ticker] += defensive_carry
            shocks["equity_stress"].loc[ticker] += defensive_carry
        if group_name in bond_groups:
            shocks["rates_up"].loc[ticker] += rates_up
            shocks["rates_down"].loc[ticker] += rates_down
        if group_name in commodity_groups:
            shocks["commodity_up"].loc[ticker] += commodity_up
            shocks["risk_off"].loc[ticker] -= 0.25 * abs(risk_off)
            shocks["equity_stress"].loc[ticker] -= 0.20 * abs(equity_stress)
        if group_name in hedge_groups:
            shocks["risk_on"].loc[ticker] -= 0.005
            shocks["risk_off"].loc[ticker] += abs(risk_off)
            shocks["equity_stress"].loc[ticker] += abs(equity_stress)
    return shocks


def _adjust_covariance_for_scenario(
    baseline_covariance: pd.DataFrame,
    scenario_name: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    index = pd.Index(baseline_covariance.index, name="ticker")
    group_map = pd.Series(get_group_map(), dtype=object).reindex(index).fillna("")
    corr = _cov_to_corr(baseline_covariance).to_numpy(dtype=float, copy=True)
    vols = np.sqrt(np.maximum(np.diag(baseline_covariance.to_numpy(dtype=float)), 0.0))
    vol_scale = pd.Series(1.0, index=index, dtype=float)

    scenario = str(scenario_name)
    equity_groups = set(map(str, params.get("equity_like_groups", EQUITY_LIKE_GROUPS)))

    if scenario == "risk_on":
        for ticker, group in group_map.items():
            if str(group) in equity_groups:
                vol_scale.loc[ticker] = 0.90
    elif scenario == "risk_off":
        for ticker, group in group_map.items():
            if str(group) in equity_groups:
                vol_scale.loc[ticker] = 1.35
            elif str(group) in {"commodities", "crypto"}:
                vol_scale.loc[ticker] = 1.25
            elif str(group) in set(DEFENSIVE_GROUPS):
                vol_scale.loc[ticker] = 0.90
    elif scenario == "equity_stress":
        for ticker, group in group_map.items():
            if str(group) in equity_groups:
                vol_scale.loc[ticker] = 1.65
            elif str(group) in {"commodities", "crypto"}:
                vol_scale.loc[ticker] = 1.35
    elif scenario in {"rates_up", "rates_down"}:
        for ticker, group in group_map.items():
            if str(group) == "bonds":
                vol_scale.loc[ticker] = 1.25
    elif scenario == "commodity_up":
        for ticker, group in group_map.items():
            if str(group) == "commodities":
                vol_scale.loc[ticker] = 1.35

    stress_targets = {
        "risk_on": (equity_groups, 0.45, 0.15),
        "risk_off": (equity_groups, 0.75, 0.35),
        "equity_stress": (equity_groups, 0.85, 0.55),
        "rates_up": ({"bonds"}, 0.70, 0.35),
        "rates_down": ({"bonds"}, 0.65, 0.30),
        "commodity_up": ({"commodities"}, 0.70, 0.35),
    }
    if scenario in stress_targets:
        groups, target_corr, blend = stress_targets[scenario]
        groups = set(map(str, groups))
        for i, left in enumerate(index):
            for j, right in enumerate(index):
                if i == j:
                    continue
                if str(group_map.get(left, "")) in groups and str(group_map.get(right, "")) in groups:
                    corr[i, j] = (1.0 - blend) * corr[i, j] + blend * target_corr
    np.fill_diagonal(corr, 1.0)
    scaled_vols = vols * vol_scale.to_numpy(dtype=float)
    adjusted = corr * np.outer(scaled_vols, scaled_vols)
    covariance = pd.DataFrame(adjusted, index=index, columns=index)
    return _clean_covariance(covariance, jitter=float(params.get("cov_jitter", 1e-8)))


def build_scenario_risk_distribution(
    *,
    forecast_table: pd.DataFrame,
    returns: pd.DataFrame,
    as_of: pd.Timestamp | str,
    params: dict[str, Any],
    effective_horizon_days: int | None = None,
) -> ScenarioRiskDistribution:
    """Build a probability-weighted scenario distribution with covariance."""

    if forecast_table.empty:
        raise ValueError("forecast_table must not be empty.")
    index = pd.Index([str(ticker) for ticker in forecast_table.index], name="ticker")
    horizon_days = int(effective_horizon_days or params.get("effective_horizon_days", params.get("horizon_days", 63)))
    daily_cov, horizon_cov, baseline_corr, warnings = compute_baseline_covariance(
        returns=returns,
        as_of=as_of,
        assets=index,
        lookback=int(params.get("scenario_covariance_lookback", params.get("cov_window", 126))),
        horizon_days=horizon_days,
        shrink_alpha=float(params.get("scenario_covariance_shrink_alpha", params.get("cov_shrink_alpha", 0.75))),
        jitter=float(params.get("cov_jitter", 1e-8)),
    )
    probabilities, probability_warnings = _normalized_probabilities(params)
    warnings.extend(probability_warnings)

    base_mu_column = "expected_return_horizon" if "expected_return_horizon" in forecast_table.columns else "expected_return_3m"
    base_mu = _to_float_series(forecast_table[base_mu_column], index=index)
    shocks = _scenario_shocks(index, params)
    expected_returns = pd.DataFrame(index=probabilities.index, columns=index, dtype=float)
    covariance_matrices: dict[str, pd.DataFrame] = {}
    for scenario_name in probabilities.index:
        shock = shocks.get(str(scenario_name), pd.Series(0.0, index=index, dtype=float))
        expected_returns.loc[scenario_name] = (base_mu + shock.reindex(index).fillna(0.0)).reindex(index)
        covariance_matrices[str(scenario_name)] = _adjust_covariance_for_scenario(horizon_cov, str(scenario_name), params)

    summary_rows: list[dict[str, Any]] = []
    for scenario_name in probabilities.index:
        cov = covariance_matrices[str(scenario_name)]
        corr = _cov_to_corr(cov)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        summary_rows.append(
            {
                "scenario_name": str(scenario_name),
                "probability": float(probabilities.loc[scenario_name]),
                "scenario_horizon_days": horizon_days,
                "mean_expected_return": float(expected_returns.loc[scenario_name].mean()),
                "median_expected_return": float(expected_returns.loc[scenario_name].median()),
                "average_pairwise_correlation": float(upper.mean()) if not upper.empty else 0.0,
                "max_pairwise_correlation": float(upper.max()) if not upper.empty else 0.0,
                "uses_scenario_dependent_covariance": True,
                "probability_source": "config_default",
            }
        )
    summary = pd.DataFrame(summary_rows)
    return ScenarioRiskDistribution(
        as_of=pd.Timestamp(as_of),
        expected_returns=expected_returns,
        probabilities=probabilities,
        covariance_matrices=covariance_matrices,
        baseline_daily_covariance=daily_cov,
        baseline_covariance_horizon=horizon_cov,
        baseline_correlation=baseline_corr,
        summary=summary,
        warnings=warnings,
    )


def weighted_quantile(values: np.ndarray, probabilities: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_probabilities = probabilities[order]
    cdf = np.cumsum(sorted_probabilities) / max(float(np.sum(sorted_probabilities)), 1e-12)
    idx = int(np.searchsorted(cdf, float(quantile), side="left"))
    idx = min(max(idx, 0), len(sorted_values) - 1)
    return float(sorted_values[idx])


def evaluate_portfolio_scenario_mixture(
    *,
    weights: pd.Series,
    distribution: ScenarioRiskDistribution,
    current_weights: pd.Series | None = None,
    defensive_cash_weights: pd.Series | None = None,
    hold_weights: pd.Series | None = None,
    cost_pct_nav: float = 0.0,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a portfolio using probability-weighted scenario return/variance."""

    params = dict(params or {})
    assets = distribution.assets
    weights_aligned = weights.reindex(assets).fillna(0.0).astype(float)
    current = (
        current_weights.reindex(assets).fillna(0.0).astype(float)
        if current_weights is not None
        else pd.Series(0.0, index=assets, dtype=float)
    )
    defensive = (
        defensive_cash_weights.reindex(assets).fillna(0.0).astype(float)
        if defensive_cash_weights is not None
        else pd.Series(0.0, index=assets, dtype=float)
    )
    if float(defensive.sum()) <= 0.0 and str(params.get("cash_ticker", "")) in assets:
        defensive.loc[str(params.get("cash_ticker"))] = 1.0
    hold = hold_weights.reindex(assets).fillna(0.0).astype(float) if hold_weights is not None else current.copy()

    probabilities = distribution.probabilities.reindex(distribution.expected_returns.index).fillna(0.0).to_numpy(dtype=float)
    probabilities = probabilities / max(float(probabilities.sum()), 1e-12)
    scenario_returns: list[float] = []
    scenario_variances: list[float] = []
    cash_returns: list[float] = []
    hold_returns: list[float] = []
    for scenario_name in distribution.expected_returns.index:
        mu = distribution.expected_returns.loc[scenario_name].reindex(assets).fillna(0.0).astype(float)
        cov = distribution.covariance_matrices[str(scenario_name)].reindex(index=assets, columns=assets).fillna(0.0)
        w = weights_aligned.to_numpy(dtype=float)
        c = cov.to_numpy(dtype=float)
        scenario_return = float(w @ mu.to_numpy(dtype=float))
        scenario_variance = max(float(w @ c @ w), 0.0)
        scenario_returns.append(scenario_return)
        scenario_variances.append(scenario_variance)
        cash_returns.append(float(defensive.to_numpy(dtype=float) @ mu.to_numpy(dtype=float)))
        hold_returns.append(float(hold.to_numpy(dtype=float) @ mu.to_numpy(dtype=float)))

    scenario_returns_array = np.asarray(scenario_returns, dtype=float)
    scenario_variances_array = np.asarray(scenario_variances, dtype=float)
    cash_returns_array = np.asarray(cash_returns, dtype=float)
    hold_returns_array = np.asarray(hold_returns, dtype=float)
    mixture_expected_return = float(probabilities @ scenario_returns_array)
    defensive_cash_return = float(probabilities @ cash_returns_array)
    hold_return = float(probabilities @ hold_returns_array)
    within_variance = float(probabilities @ scenario_variances_array)
    between_variance = float(probabilities @ np.square(scenario_returns_array - mixture_expected_return))
    mixture_variance = max(within_variance + between_variance, 0.0)
    mixture_volatility = float(np.sqrt(mixture_variance))
    epsilon = float(params.get("sharpe_epsilon", 1e-8))
    excess_vs_cash = mixture_expected_return - defensive_cash_return
    excess_vs_current = mixture_expected_return - hold_return
    risk_free_return = risk_free_return_from_params(params)
    excess_vs_risk_free = mixture_expected_return - risk_free_return
    scenario_mixture_sharpe = float(excess_vs_risk_free / max(mixture_volatility, epsilon))
    scenario_mixture_excess_vs_cash_ratio = float(excess_vs_cash / max(mixture_volatility, epsilon))
    downside_base = defensive_cash_return if str(params.get("sortino_target", "defensive_cash")) == "defensive_cash" else 0.0
    downside = np.minimum(scenario_returns_array - downside_base, 0.0)
    downside_deviation = float(np.sqrt(probabilities @ np.square(downside)))
    sortino = float(excess_vs_cash / max(downside_deviation, epsilon))
    var_5 = weighted_quantile(scenario_returns_array, probabilities, 0.05)
    tail_mask = scenario_returns_array <= var_5 + 1e-12
    tail_prob = float(probabilities[tail_mask].sum())
    cvar_5 = float((probabilities[tail_mask] @ scenario_returns_array[tail_mask]) / max(tail_prob, 1e-12))
    probability_loss = float(probabilities[scenario_returns_array < 0.0].sum())
    probability_beats_current = float(probabilities[scenario_returns_array > hold_returns_array].sum())
    probability_beats_cash = float(probabilities[scenario_returns_array > cash_returns_array].sum())
    turnover = float(np.abs(weights_aligned - current).sum())
    concentration_hhi = float(np.square(weights_aligned.to_numpy(dtype=float)).sum())
    lambda_cvar = float(params.get("lambda_cvar_sharpe", params.get("lambda_cvar", 0.25)))
    lambda_turnover = float(params.get("lambda_turnover_sharpe", 0.0))
    lambda_cost = float(params.get("lambda_cost_sharpe", 1.0))
    lambda_concentration = float(params.get("lambda_concentration_sharpe", 0.0))
    tail_penalty = max(0.0, -float(cvar_5))
    turnover_penalty = lambda_turnover * turnover
    cost_penalty = lambda_cost * max(float(cost_pct_nav), 0.0)
    concentration_penalty = lambda_concentration * concentration_hhi
    objective = scenario_mixture_sharpe - lambda_cvar * tail_penalty - turnover_penalty - cost_penalty - concentration_penalty
    average_corr = _weighted_average_pairwise_correlation(weights_aligned, distribution.baseline_correlation)
    diversification_ratio = _diversification_ratio(weights_aligned, distribution.baseline_covariance_horizon, mixture_volatility)
    return {
        "mixture_expected_return": mixture_expected_return,
        "defensive_cash_mixture_return": defensive_cash_return,
        "risk_free_horizon_return": risk_free_return,
        "excess_return_vs_defensive_cash": excess_vs_cash,
        "excess_return_vs_risk_free": excess_vs_risk_free,
        "excess_return_vs_current": excess_vs_current,
        "within_scenario_variance": within_variance,
        "between_scenario_variance": between_variance,
        "mixture_variance": mixture_variance,
        "mixture_volatility": mixture_volatility,
        "scenario_mixture_sharpe": scenario_mixture_sharpe,
        "scenario_mixture_excess_vs_cash_ratio": scenario_mixture_excess_vs_cash_ratio,
        "sortino_like_ratio": sortino,
        "probability_weighted_var": var_5,
        "probability_weighted_cvar": cvar_5,
        "probability_loss": probability_loss,
        "probability_beats_current": probability_beats_current,
        "probability_beats_defensive_cash": probability_beats_cash,
        "concentration_hhi": concentration_hhi,
        "turnover": turnover,
        "cost_pct_nav": float(cost_pct_nav),
        "tail_penalty": tail_penalty,
        "turnover_penalty": turnover_penalty,
        "cost_penalty": cost_penalty,
        "concentration_penalty": concentration_penalty,
        "robust_scenario_sharpe_objective": objective,
        "average_pairwise_correlation_weighted": average_corr,
        "diversification_ratio": diversification_ratio,
        "scenario_returns": scenario_returns_array,
        "scenario_variances": scenario_variances_array,
        "cash_returns": cash_returns_array,
        "hold_returns": hold_returns_array,
    }


def _weighted_average_pairwise_correlation(weights: pd.Series, correlation: pd.DataFrame) -> float:
    aligned = weights.reindex(correlation.index).fillna(0.0).astype(float)
    values = correlation.reindex(index=aligned.index, columns=aligned.index).fillna(0.0).to_numpy(dtype=float)
    w = aligned.to_numpy(dtype=float)
    pair_weights = np.outer(w, w)
    mask = ~np.eye(len(w), dtype=bool)
    denom = float(pair_weights[mask].sum())
    if denom <= 0.0:
        return 0.0
    return float((pair_weights[mask] * values[mask]).sum() / denom)


def _diversification_ratio(weights: pd.Series, covariance: pd.DataFrame, portfolio_volatility: float) -> float:
    aligned = weights.reindex(covariance.index).fillna(0.0).astype(float)
    asset_vols = np.sqrt(np.maximum(np.diag(covariance.to_numpy(dtype=float)), 0.0))
    weighted_vol_sum = float(aligned.to_numpy(dtype=float) @ asset_vols)
    if portfolio_volatility <= 1e-12:
        return 0.0
    return float(weighted_vol_sum / portfolio_volatility)


def build_candidate_risk_return_frame(
    *,
    candidate_weights: dict[str, pd.Series],
    distribution: ScenarioRiskDistribution,
    current_weights: pd.Series,
    defensive_cash_weights: pd.Series,
    hold_weights: pd.Series,
    params: dict[str, Any],
    scores_frame: pd.DataFrame | None = None,
    selected_name: str = "",
    selected_reason: str = "",
) -> pd.DataFrame:
    """Build the candidate risk/return report frame."""

    scores = scores_frame.copy() if scores_frame is not None else pd.DataFrame()
    if not scores.empty:
        name_column = "discrete_candidate" if "discrete_candidate" in scores.columns else "candidate"
        scores = scores.set_index(scores[name_column].astype(str), drop=False)

    rows: list[dict[str, Any]] = []
    for name, weights in candidate_weights.items():
        score_row = scores.loc[str(name)] if not scores.empty and str(name) in scores.index else pd.Series(dtype=object)
        cost_pct_nav = _safe_score_value(score_row, ["total_order_cost_pct_nav", "estimated_cost", "cost_pct_nav"], 0.0)
        metrics = evaluate_portfolio_scenario_mixture(
            weights=weights,
            distribution=distribution,
            current_weights=current_weights,
            defensive_cash_weights=defensive_cash_weights,
            hold_weights=hold_weights,
            cost_pct_nav=cost_pct_nav,
            params=params,
        )
        failed_constraints = str(score_row.get("validation_errors", "") if not score_row.empty else "")
        valid_constraints = bool(score_row.get("valid_constraints", True)) if not score_row.empty else True
        rows.append(
            {
                "candidate": str(name),
                "candidate_family": str(name).split("::", 1)[0],
                "mixture_expected_return": metrics["mixture_expected_return"],
                "defensive_cash_mixture_return": metrics["defensive_cash_mixture_return"],
                "risk_free_horizon_return": metrics["risk_free_horizon_return"],
                "excess_return_vs_defensive_cash": metrics["excess_return_vs_defensive_cash"],
                "excess_return_vs_risk_free": metrics["excess_return_vs_risk_free"],
                "excess_return_vs_current": metrics["excess_return_vs_current"],
                "within_scenario_variance": metrics["within_scenario_variance"],
                "between_scenario_variance": metrics["between_scenario_variance"],
                "mixture_variance": metrics["mixture_variance"],
                "mixture_volatility": metrics["mixture_volatility"],
                "scenario_mixture_sharpe": metrics["scenario_mixture_sharpe"],
                "scenario_mixture_excess_vs_cash_ratio": metrics["scenario_mixture_excess_vs_cash_ratio"],
                "sortino_like_ratio": metrics["sortino_like_ratio"],
                "probability_weighted_var": metrics["probability_weighted_var"],
                "probability_weighted_cvar": metrics["probability_weighted_cvar"],
                "probability_loss": metrics["probability_loss"],
                "probability_beats_current": metrics["probability_beats_current"],
                "probability_beats_defensive_cash": metrics["probability_beats_defensive_cash"],
                "concentration_hhi": metrics["concentration_hhi"],
                "turnover": _safe_score_value(score_row, ["turnover_vs_current", "turnover"], metrics["turnover"]),
                "cost_pct_nav": cost_pct_nav,
                "tail_penalty": metrics["tail_penalty"],
                "turnover_penalty": metrics["turnover_penalty"],
                "cost_penalty": metrics["cost_penalty"],
                "concentration_penalty": metrics["concentration_penalty"],
                "robust_scenario_sharpe_objective": metrics["robust_scenario_sharpe_objective"],
                "average_pairwise_correlation_weighted": metrics["average_pairwise_correlation_weighted"],
                "diversification_ratio": metrics["diversification_ratio"],
                "robust_score": _safe_score_value(score_row, ["robust_score", "gross_robust_score"], np.nan),
                "net_robust_score": _safe_score_value(score_row, ["net_robust_score"], np.nan),
                "valid_constraints": valid_constraints,
                "failed_constraints": failed_constraints,
                "selected": str(name) == str(selected_name),
                "selected_reason": selected_reason if str(name) == str(selected_name) else "",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["robust_scenario_sharpe_objective", "scenario_mixture_sharpe"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def build_scenario_attribution_frame(
    *,
    candidate_weights: dict[str, pd.Series],
    distribution: ScenarioRiskDistribution,
    hold_weights: pd.Series,
    defensive_cash_weights: pd.Series,
) -> pd.DataFrame:
    """Explain which scenarios and assets drive each candidate."""

    assets = distribution.assets
    hold = hold_weights.reindex(assets).fillna(0.0).astype(float)
    cash = defensive_cash_weights.reindex(assets).fillna(0.0).astype(float)
    rows: list[dict[str, Any]] = []
    for candidate_name, weights in candidate_weights.items():
        w = weights.reindex(assets).fillna(0.0).astype(float)
        for scenario_name in distribution.scenario_names:
            mu = distribution.expected_returns.loc[scenario_name].reindex(assets).fillna(0.0).astype(float)
            cov = distribution.covariance_matrices[scenario_name].reindex(index=assets, columns=assets).fillna(0.0)
            contributions = w * mu
            scenario_return = float(contributions.sum())
            hold_return = float((hold * mu).sum())
            cash_return = float((cash * mu).sum())
            variance = max(float(w.to_numpy(dtype=float) @ cov.to_numpy(dtype=float) @ w.to_numpy(dtype=float)), 0.0)
            top_positive = contributions.sort_values(ascending=False).head(3)
            top_negative = contributions.sort_values(ascending=True).head(3)
            probability = float(distribution.probabilities.loc[scenario_name])
            rows.append(
                {
                    "candidate": str(candidate_name),
                    "scenario_name": str(scenario_name),
                    "scenario_probability": probability,
                    "scenario_return": scenario_return,
                    "scenario_variance": variance,
                    "scenario_volatility": float(np.sqrt(variance)),
                    "return_vs_hold": scenario_return - hold_return,
                    "return_vs_defensive_cash": scenario_return - cash_return,
                    "contribution_to_mixture_return": probability * scenario_return,
                    "contribution_to_mixture_variance": probability * variance,
                    "top_positive_asset_contributions": "; ".join(f"{idx}:{val:.6f}" for idx, val in top_positive.items()),
                    "top_negative_asset_contributions": "; ".join(f"{idx}:{val:.6f}" for idx, val in top_negative.items()),
                }
            )
    return pd.DataFrame(rows)


def _safe_score_value(score_row: pd.Series, columns: list[str], default: float) -> float:
    for column in columns:
        if column in score_row.index:
            try:
                value = float(score_row[column])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
    return float(default)


def write_scenario_risk_reports(
    *,
    distribution: ScenarioRiskDistribution,
    output_dir: Path,
    write_pairwise_relationships: bool = True,
    write_scenario_matrices: bool = True,
) -> None:
    """Write covariance, correlation and scenario probability reports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if write_scenario_matrices:
        distribution.baseline_correlation.to_csv(output_dir / "asset_correlation_matrix.csv")
        distribution.baseline_covariance_horizon.to_csv(output_dir / "asset_covariance_horizon.csv")
        probability_report = distribution.summary.copy()
        if not probability_report.empty:
            probability_report.insert(0, "model_scope", "scenario_risk_distribution")
            probability_report.insert(1, "active_for_final_allocation", False)
            probability_report.insert(2, "source_module", "scenario_risk_model")
        probability_report.to_csv(output_dir / "scenario_probability_report.csv", index=False)
        probability_report.to_csv(output_dir / "scenario_risk_probability_report.csv", index=False)
        covariance_summary = distribution.summary[
            [
                "scenario_name",
                "probability",
                "scenario_horizon_days",
                "average_pairwise_correlation",
                "max_pairwise_correlation",
                "uses_scenario_dependent_covariance",
            ]
        ].copy()
        covariance_summary.to_csv(output_dir / "scenario_covariance_summary.csv", index=False)
    if not write_pairwise_relationships:
        return

    corr = distribution.baseline_correlation.copy()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
    highest = upper.head(8)
    lowest = upper.sort_values(ascending=True).head(8)
    avg_corr = corr.where(~np.eye(len(corr), dtype=bool)).mean(axis=1).sort_values(ascending=False)
    lines = [
        "Correlation Cluster Report",
        "",
        f"as_of: {distribution.as_of.date()}",
        f"scenario_count: {len(distribution.scenario_names)}",
        f"scenario_names: {', '.join(distribution.scenario_names)}",
        "uses_scenario_dependent_covariance: true",
        "warnings: " + (" | ".join(distribution.warnings) if distribution.warnings else "none"),
        "",
        "Highest positive correlations:",
    ]
    lines.extend([f"- {left}/{right}: {value:.4f}" for (left, right), value in highest.items()])
    lines.append("")
    lines.append("Lowest correlations:")
    lines.extend([f"- {left}/{right}: {value:.4f}" for (left, right), value in lowest.items()])
    lines.append("")
    lines.append("Assets with high average correlation:")
    lines.extend([f"- {ticker}: {value:.4f}" for ticker, value in avg_corr.head(8).items()])
    (output_dir / "correlation_cluster_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_candidate_risk_return_reports(
    *,
    risk_return_frame: pd.DataFrame,
    selected_candidate: str,
    objective_used: str,
    output_dir: Path,
) -> None:
    """Write candidate risk/return CSV and readable summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    risk_return_frame.to_csv(output_dir / "candidate_risk_return_report.csv", index=False)
    if risk_return_frame.empty:
        text = "Candidate Risk/Return Report\n\nNo candidates available.\n"
        (output_dir / "candidate_risk_return_report.txt").write_text(text, encoding="utf-8")
        return

    top_sharpe = risk_return_frame.sort_values("scenario_mixture_sharpe", ascending=False).iloc[0]
    top_objective = risk_return_frame.sort_values("robust_scenario_sharpe_objective", ascending=False).iloc[0]
    robust_rank = risk_return_frame.dropna(subset=["net_robust_score"]).sort_values("net_robust_score", ascending=False)
    top_robust = robust_rank.iloc[0] if not robust_rank.empty else pd.Series({"candidate": "n/a", "net_robust_score": np.nan})
    selected_rows = risk_return_frame.loc[risk_return_frame["candidate"].astype(str) == str(selected_candidate)]
    selected = selected_rows.iloc[0] if not selected_rows.empty else pd.Series(dtype=object)
    hold_rows = risk_return_frame.loc[risk_return_frame["candidate"].astype(str).str.contains("HOLD", case=False, regex=False)]
    hold_note = "HOLD not present in risk/return frame."
    if not hold_rows.empty:
        hold_best = hold_rows.sort_values("robust_scenario_sharpe_objective", ascending=False).iloc[0]
        hold_note = (
            "HOLD has the highest robust scenario Sharpe objective."
            if str(hold_best["candidate"]) == str(top_objective["candidate"])
            else "HOLD does not have the highest robust scenario Sharpe objective; selection may be due to constraints/gates/fallback."
        )
    lines = [
        "Candidate Risk/Return Report",
        "",
        f"objective_used: {objective_used}",
        f"highest_scenario_mixture_sharpe_candidate: {top_sharpe['candidate']} ({float(top_sharpe['scenario_mixture_sharpe']):.6f})",
        f"highest_robust_scenario_sharpe_objective_candidate: {top_objective['candidate']} ({float(top_objective['robust_scenario_sharpe_objective']):.6f})",
        f"highest_net_robust_score_candidate: {top_robust['candidate']} ({float(top_robust.get('net_robust_score', np.nan)):.6f})",
        f"final_selected_candidate: {selected_candidate}",
        f"selected_candidate_scenario_mixture_sharpe: {float(selected.get('scenario_mixture_sharpe', np.nan)):.6f}",
        f"selected_candidate_robust_scenario_sharpe_objective: {float(selected.get('robust_scenario_sharpe_objective', np.nan)):.6f}",
        f"selected_candidate_valid_constraints: {bool(selected.get('valid_constraints', False)) if not selected.empty else 'unknown'}",
        "",
        hold_note,
        "",
        "Why rankings may differ:",
        "- robust_score still drives default selection unless OPTIMIZATION_OBJECTIVE is changed.",
        "- scenario_mixture_sharpe is RF-adjusted and includes within-scenario covariance plus between-scenario return uncertainty.",
        "- robust_scenario_sharpe_objective additionally penalizes tail risk, cost, turnover and concentration.",
    ]
    (output_dir / "candidate_risk_return_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
