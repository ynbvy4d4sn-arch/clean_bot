"""Scenario input construction for direct RF-adjusted Sharpe optimization.

The adjustments in this module are explicitly default heuristics. They are
derived from the existing forecast table, historical returns, asset groups and
momentum/volatility fields when available. They are not hard-coded market
truths, and callers can override probabilities and shock magnitudes through
the config dictionary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from asset_universe import get_group_map
from risk_free import risk_free_return_for_horizon
from scenario_weighted_solver import ScenarioInput


DEFAULT_SCENARIO_PROBABILITIES: dict[str, float] = {
    "bull_momentum": 0.25,
    "soft_landing": 0.25,
    "sideways_choppy": 0.20,
    "inflation_shock": 0.15,
    "growth_selloff": 0.10,
    "liquidity_stress": 0.05,
}

SCENARIO_NAMES: tuple[str, ...] = tuple(DEFAULT_SCENARIO_PROBABILITIES)
EQUITY_LIKE_GROUPS = {"us_sector", "factor", "international", "equity"}
DEFENSIVE_GROUPS = {"cash", "bonds", "hedge"}
COMMODITY_GROUPS = {"commodities"}
CRYPTO_GROUPS = {"crypto"}
GROWTH_TICKERS = {"XLK", "SPMO", "XLC", "IBIT", "XLY"}
DEFENSIVE_EQUITY_TICKERS = {"SPLV", "XLP", "XLU", "XLV"}
DURATION_BOND_TICKERS = {"IEF", "TLT", "AGG", "LQD", "TIP"}
CASH_LIKE_TICKERS = {"SGOV", "SHY"}
METALS_TICKERS = {"GLD", "SLV"}


def build_scenario_inputs(
    forecast_table: pd.DataFrame,
    returns: pd.DataFrame,
    config: dict[str, Any],
) -> list[ScenarioInput]:
    """Build strict scenario inputs for the scenario-weighted solver.

    The returned scenarios all use the exact same asset order, defined by the
    forecast table. Missing forecast or returns inputs raise ``ValueError``
    instead of silently filling with zeros.
    """

    params = _resolve_params(config)
    forecast = _normalize_forecast_table(forecast_table)
    assets = pd.Index([str(asset) for asset in forecast.index], name="ticker")
    if assets.empty:
        raise ValueError("forecast_table must contain at least one asset.")

    baseline_mu = _baseline_expected_returns(forecast, params, assets)
    returns_aligned = _strict_returns(returns, assets)
    baseline_covariance = _baseline_covariance(returns_aligned, params, assets)
    baseline_corr = _cov_to_corr(baseline_covariance)
    baseline_vol = pd.Series(
        np.sqrt(np.maximum(np.diag(baseline_covariance.to_numpy(dtype=float)), 0.0)),
        index=assets,
        dtype=float,
    )
    baseline_vol.attrs["volatility_source"] = "historical_returns_covariance"
    group_map = _asset_group_map(params, assets)
    probabilities = _scenario_probabilities(params)
    risk_free = risk_free_return_for_horizon(
        risk_free_rate_annual=float(params.get("risk_free_rate_annual", 0.02)),
        horizon_days=int(params.get("effective_horizon_days", params.get("horizon_days", 63))),
        trading_days_per_year=int(params.get("trading_days_per_year", 252)),
    )

    scenarios: list[ScenarioInput] = []
    for scenario_name in SCENARIO_NAMES:
        expected_returns = _scenario_expected_returns(
            scenario_name=scenario_name,
            baseline_mu=baseline_mu,
            forecast=forecast,
            baseline_vol=baseline_vol,
            group_map=group_map,
            params=params,
        )
        covariance = _scenario_covariance(
            scenario_name=scenario_name,
            baseline_vol=baseline_vol,
            baseline_corr=baseline_corr,
            group_map=group_map,
            params=params,
        )
        scenarios.append(
            ScenarioInput(
                name=scenario_name,
                probability=float(probabilities.loc[scenario_name]),
                expected_returns=expected_returns.reindex(assets).astype(float),
                covariance=covariance.reindex(index=assets, columns=assets).astype(float),
                risk_free_return=risk_free,
                metadata={
                    "assumption_type": "default_configurable_heuristic",
                    "probability_source": str(probabilities.attrs.get("probability_source", "default_configurable_heuristic")),
                    "expected_return_source": str(baseline_mu.attrs.get("expected_return_source", "forecast_table")),
                    "volatility_source": _volatility_source_label(forecast),
                    "covariance_source": "historical_returns_shrunk_horizon_covariance",
                    "adjustment_source": "asset_groups_momentum_volatility_default_heuristic",
                    "diagnostic_warnings": "; ".join(map(str, probabilities.attrs.get("warnings", []))) or "none",
                },
            )
        )
    return scenarios


def _resolve_params(config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(config or {})
    solver_block = raw.get("solver")
    params = dict(solver_block) if isinstance(solver_block, dict) else {}
    params.update(raw)
    return params


def _asset_group_map(params: dict[str, Any], assets: pd.Index) -> pd.Series:
    raw_group_map = params.get("asset_groups") or params.get("group_map") or get_group_map()
    mapped: dict[str, str] = {}
    for key, value in dict(raw_group_map).items():
        if isinstance(value, (list, tuple, set, pd.Index)):
            for asset in value:
                mapped[str(asset)] = str(key)
        else:
            mapped[str(key)] = str(value)
    mapped_series = pd.Series(mapped, dtype=object).reindex(assets)
    missing = mapped_series[mapped_series.isna()].index.astype(str).tolist()
    if missing and not bool(params.get("allow_unknown_asset_groups", False)):
        raise ValueError(
            "asset group map missing assets required for scenario adjustments: "
            f"{missing}. Provide asset_groups/group_map or set allow_unknown_asset_groups=true for research diagnostics only."
        )
    return mapped_series.fillna("unknown").astype(str)


def _normalize_forecast_table(forecast_table: pd.DataFrame) -> pd.DataFrame:
    if forecast_table.empty:
        raise ValueError("forecast_table must not be empty.")
    forecast = forecast_table.copy()
    if "ticker" in forecast.columns:
        forecast = forecast.set_index("ticker", drop=True)
    forecast.index = pd.Index([str(asset) for asset in forecast.index], name="ticker")
    if forecast.index.has_duplicates:
        duplicates = forecast.index[forecast.index.duplicated()].unique().tolist()
        raise ValueError(f"forecast_table contains duplicate assets: {duplicates}")
    return forecast


def _baseline_expected_returns(
    forecast: pd.DataFrame,
    params: dict[str, Any],
    assets: pd.Index,
) -> pd.Series:
    if "expected_return_horizon" in forecast.columns:
        values = forecast["expected_return_horizon"]
        source = "forecast_table.expected_return_horizon"
    elif "expected_return_3m" in forecast.columns:
        horizon_days = float(params.get("effective_horizon_days", params.get("horizon_days", 63)) or 63)
        default_days = float(params.get("default_forecast_horizon_days", params.get("horizon_days", 63)) or 63)
        values = forecast["expected_return_3m"].astype(float) * horizon_days / max(default_days, 1.0)
        source = "forecast_table.expected_return_3m_scaled_to_horizon"
    else:
        raise ValueError("forecast_table requires expected_return_horizon or expected_return_3m.")
    numeric = pd.to_numeric(values.reindex(assets), errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        bad_assets = numeric.index[numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=float))].tolist()
        raise ValueError(f"forecast expected returns contain non-finite values for: {bad_assets}")
    result = numeric.astype(float)
    result.attrs["expected_return_source"] = source
    return result


def _horizon_scale(params: dict[str, Any]) -> float:
    horizon_days = float(params.get("effective_horizon_days", params.get("horizon_days", 63)) or 63)
    default_days = float(params.get("default_forecast_horizon_days", params.get("horizon_days", 63)) or 63)
    return horizon_days / max(default_days, 1.0)


def _strict_returns(returns: pd.DataFrame, assets: pd.Index) -> pd.DataFrame:
    if returns.empty:
        raise ValueError("returns must not be empty.")
    missing = [asset for asset in assets if asset not in returns.columns.astype(str)]
    if missing:
        raise ValueError(f"returns missing assets required by forecast_table: {missing}")
    aligned = returns.copy()
    aligned.columns = pd.Index([str(asset) for asset in aligned.columns], name="ticker")
    aligned = aligned.reindex(columns=assets).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(aligned) < 2:
        raise ValueError("At least two complete return observations are required for scenario covariance.")
    return aligned.astype(float)


def _baseline_covariance(
    returns: pd.DataFrame,
    params: dict[str, Any],
    assets: pd.Index,
) -> pd.DataFrame:
    lookback = int(params.get("scenario_covariance_lookback", params.get("cov_window", 126)) or 126)
    horizon_days = int(params.get("effective_horizon_days", params.get("horizon_days", 63)) or 63)
    shrink_alpha = min(max(float(params.get("scenario_covariance_shrink_alpha", params.get("cov_shrink_alpha", 0.75))), 0.0), 1.0)
    sample = returns.tail(max(lookback, 2)).cov().reindex(index=assets, columns=assets)
    if sample.isna().any().any():
        raise ValueError("baseline covariance contains NaN after alignment.")
    diagonal = pd.DataFrame(np.diag(np.diag(sample.to_numpy(dtype=float))), index=assets, columns=assets)
    daily_covariance = shrink_alpha * sample + (1.0 - shrink_alpha) * diagonal
    horizon_covariance = daily_covariance * max(float(horizon_days), 1.0)
    return _nearest_psd(horizon_covariance, jitter=float(params.get("cov_jitter", params.get("eps_variance", 1.0e-10))))


def _scenario_probabilities(params: dict[str, Any]) -> pd.Series:
    raw_key = ""
    raw = None
    for key in ("scenario_weighted_probabilities", "scenario_input_probabilities", "scenario_probabilities"):
        if params.get(key) is not None:
            raw_key = key
            raw = params.get(key)
            break
    warnings: list[str] = []
    if raw is None:
        raw = DEFAULT_SCENARIO_PROBABILITIES
        probability_source = "default_configurable_heuristic"
    else:
        probability_source = f"config.{raw_key}"
    raw_series = pd.Series(raw, dtype=float)
    if not set(SCENARIO_NAMES).issubset(set(raw_series.index.astype(str))):
        if raw_key == "scenario_probabilities":
            warnings.append(
                "config.scenario_probabilities uses the legacy scenario schema; "
                "scenario-weighted solver probabilities fell back to default configurable heuristics."
            )
            raw_series = pd.Series(DEFAULT_SCENARIO_PROBABILITIES, dtype=float)
            probability_source = "default_configurable_heuristic_due_to_legacy_config_schema"
        else:
            missing = sorted(set(SCENARIO_NAMES) - set(raw_series.index.astype(str)))
            raise ValueError(f"{raw_key} missing required scenario probabilities: {missing}")
    raw_series.index = raw_series.index.astype(str)
    probabilities = raw_series.reindex(SCENARIO_NAMES).astype(float).replace([np.inf, -np.inf], np.nan)
    if probabilities.isna().any():
        missing = probabilities[probabilities.isna()].index.astype(str).tolist()
        raise ValueError(f"{raw_key or 'default_scenario_probabilities'} contains non-finite probabilities: {missing}")
    if (probabilities < 0.0).any():
        negative = probabilities[probabilities < 0.0].index.astype(str).tolist()
        raise ValueError(f"{raw_key or 'default_scenario_probabilities'} contains negative probabilities: {negative}")
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError(f"{raw_key or 'default_scenario_probabilities'} must contain positive probability mass.")
    if abs(total - 1.0) > 1.0e-12:
        warnings.append(f"scenario probabilities normalized from sum {total:.12f}")
    result = (probabilities / total).astype(float)
    result.attrs["probability_source"] = probability_source
    result.attrs["warnings"] = warnings
    return result


def _scenario_expected_returns(
    *,
    scenario_name: str,
    baseline_mu: pd.Series,
    forecast: pd.DataFrame,
    baseline_vol: pd.Series,
    group_map: pd.Series,
    params: dict[str, Any],
) -> pd.Series:
    mu = baseline_mu.copy().astype(float)
    momentum = _momentum_signal(forecast, baseline_mu.index)
    momentum_intensity = _positive_signal_intensity(momentum)
    confidence = _optional_forecast_series(forecast, "signal_confidence", baseline_mu.index, default=1.0).clip(0.0, 1.0)
    horizon_scale = _horizon_scale(params)
    vol = _forecast_volatility_horizon(forecast, baseline_mu.index, params, baseline_vol=baseline_vol).clip(lower=0.0)
    median_abs_mu = max(float(mu.abs().median()), 0.001)
    momentum_boost = float(params.get("scenario_bull_momentum_multiplier", 0.35))
    moderate_boost = float(params.get("scenario_soft_landing_return_boost", 0.20))
    choppy_multiplier = float(params.get("scenario_sideways_return_multiplier", 0.35))
    inflation_commodity_shock = float(params.get("scenario_inflation_commodity_shock", 0.012)) * horizon_scale
    inflation_bond_shock = float(params.get("scenario_inflation_bond_shock", -0.010)) * horizon_scale
    growth_stress_scale = float(params.get("scenario_growth_selloff_vol_scale", 0.45))
    liquidity_stress_scale = float(params.get("scenario_liquidity_stress_vol_scale", 0.60))
    defensive_carry = float(params.get("scenario_defensive_carry_shock", 0.001)) * horizon_scale

    for asset in baseline_mu.index:
        group = str(group_map.get(asset, "unknown"))
        if scenario_name == "bull_momentum":
            if momentum.loc[asset] > 0.0:
                mu.loc[asset] += momentum_boost * max(float(mu.loc[asset]), 0.0) * confidence.loc[asset] * momentum_intensity.loc[asset]
            if group in DEFENSIVE_GROUPS:
                mu.loc[asset] -= 0.25 * defensive_carry
        elif scenario_name == "soft_landing":
            if group in EQUITY_LIKE_GROUPS:
                mu.loc[asset] += moderate_boost * median_abs_mu
            if group in {"bonds", "cash"}:
                mu.loc[asset] += 0.50 * defensive_carry
        elif scenario_name == "sideways_choppy":
            mu.loc[asset] *= choppy_multiplier
            if group in {"cash", "bonds"} or asset in CASH_LIKE_TICKERS:
                mu.loc[asset] = max(float(mu.loc[asset] + defensive_carry), float(baseline_mu.loc[asset]))
        elif scenario_name == "inflation_shock":
            if group in COMMODITY_GROUPS or asset in METALS_TICKERS:
                mu.loc[asset] += inflation_commodity_shock
            if group == "bonds" or asset in DURATION_BOND_TICKERS:
                mu.loc[asset] += inflation_bond_shock
            if asset in {"XLK", "SPMO", "XLC", "IBIT"}:
                mu.loc[asset] -= 0.50 * abs(inflation_bond_shock)
        elif scenario_name == "growth_selloff":
            if group in EQUITY_LIKE_GROUPS or group in CRYPTO_GROUPS:
                mu.loc[asset] -= growth_stress_scale * max(float(vol.loc[asset]), median_abs_mu)
            if asset in DEFENSIVE_EQUITY_TICKERS:
                mu.loc[asset] += 0.25 * growth_stress_scale * max(float(vol.loc[asset]), median_abs_mu)
            if group == "bonds" or asset in METALS_TICKERS:
                mu.loc[asset] += defensive_carry + 0.10 * median_abs_mu
        elif scenario_name == "liquidity_stress":
            if group in EQUITY_LIKE_GROUPS or group in CRYPTO_GROUPS or group in COMMODITY_GROUPS:
                mu.loc[asset] -= liquidity_stress_scale * max(float(vol.loc[asset]), median_abs_mu)
            if asset in CASH_LIKE_TICKERS:
                mu.loc[asset] = max(float(mu.loc[asset]), defensive_carry)
            if asset in {"IEF", "GLD", "SGOV", "SHY"}:
                mu.loc[asset] += defensive_carry
    clip_lower = float(params.get("scenario_return_clip_lower", -0.50))
    clip_upper = float(params.get("scenario_return_clip_upper", 0.50))
    return mu.clip(lower=clip_lower, upper=clip_upper).astype(float)


def _scenario_covariance(
    *,
    scenario_name: str,
    baseline_vol: pd.Series,
    baseline_corr: pd.DataFrame,
    group_map: pd.Series,
    params: dict[str, Any],
) -> pd.DataFrame:
    assets = pd.Index(baseline_vol.index, name="ticker")
    vol_scale = pd.Series(1.0, index=assets, dtype=float)
    corr = baseline_corr.reindex(index=assets, columns=assets).to_numpy(dtype=float, copy=True)

    def group_mask(groups: set[str]) -> pd.Series:
        return group_map.reindex(assets).astype(str).isin(groups)

    equity_mask = group_mask(EQUITY_LIKE_GROUPS)
    risk_mask = group_map.reindex(assets).astype(str).isin(EQUITY_LIKE_GROUPS | COMMODITY_GROUPS | CRYPTO_GROUPS)
    bond_mask = group_mask({"bonds"})
    commodity_mask = group_mask(COMMODITY_GROUPS)
    cash_mask = group_mask({"cash"})
    crypto_mask = group_mask(CRYPTO_GROUPS)
    metals_mask = pd.Series([asset in METALS_TICKERS for asset in assets], index=assets)
    commodity_or_metals_mask = commodity_mask | metals_mask
    risk_stress_mask = risk_mask & ~metals_mask

    if scenario_name == "bull_momentum":
        vol_scale.loc[equity_mask] *= float(params.get("scenario_bull_equity_vol_scale", 0.95))
        corr = _blend_corr(corr, equity_mask.to_numpy(), equity_mask.to_numpy(), target=0.65, blend=0.25)
        corr = _blend_corr(corr, bond_mask.to_numpy(), equity_mask.to_numpy(), target=-0.05, blend=0.15)
    elif scenario_name == "soft_landing":
        vol_scale.loc[equity_mask] *= float(params.get("scenario_soft_landing_equity_vol_scale", 0.90))
        vol_scale.loc[bond_mask] *= 0.95
        corr = _blend_corr(corr, equity_mask.to_numpy(), equity_mask.to_numpy(), target=0.50, blend=0.15)
    elif scenario_name == "sideways_choppy":
        vol_scale.loc[risk_mask] *= float(params.get("scenario_sideways_risk_vol_scale", 1.15))
        corr = _blend_corr(corr, equity_mask.to_numpy(), equity_mask.to_numpy(), target=0.55, blend=0.15)
    elif scenario_name == "inflation_shock":
        vol_scale.loc[commodity_or_metals_mask] *= float(params.get("scenario_inflation_commodity_vol_scale", 1.30))
        vol_scale.loc[bond_mask] *= float(params.get("scenario_inflation_bond_vol_scale", 1.25))
        corr = _blend_corr(corr, commodity_or_metals_mask.to_numpy(), commodity_or_metals_mask.to_numpy(), target=0.70, blend=0.30)
        corr = _blend_corr(corr, bond_mask.to_numpy(), equity_mask.to_numpy(), target=0.20, blend=0.20)
    elif scenario_name == "growth_selloff":
        vol_scale.loc[equity_mask] *= float(params.get("scenario_growth_equity_vol_scale", 1.55))
        vol_scale.loc[crypto_mask] *= float(params.get("scenario_growth_crypto_vol_scale", 1.80))
        corr = _blend_corr(corr, equity_mask.to_numpy(), equity_mask.to_numpy(), target=0.85, blend=0.45)
        corr = _blend_corr(corr, crypto_mask.to_numpy(), equity_mask.to_numpy(), target=0.75, blend=0.40)
        corr = _blend_corr(corr, bond_mask.to_numpy(), equity_mask.to_numpy(), target=-0.25, blend=0.25)
    elif scenario_name == "liquidity_stress":
        vol_scale.loc[risk_stress_mask] *= float(params.get("scenario_liquidity_risk_vol_scale", 1.80))
        vol_scale.loc[cash_mask] *= float(params.get("scenario_liquidity_cash_vol_scale", 0.50))
        vol_scale.loc[bond_mask] *= 1.10
        vol_scale.loc[metals_mask] *= float(params.get("scenario_liquidity_metals_vol_scale", 0.95))
        corr = _blend_corr(corr, risk_stress_mask.to_numpy(), risk_stress_mask.to_numpy(), target=0.90, blend=0.55)
        corr = _blend_corr(corr, bond_mask.to_numpy(), risk_stress_mask.to_numpy(), target=-0.20, blend=0.20)
        corr = _blend_corr(
            corr,
            metals_mask.to_numpy(),
            risk_stress_mask.to_numpy(),
            target=float(params.get("scenario_liquidity_metals_risk_corr_target", -0.20)),
            blend=float(params.get("scenario_liquidity_metals_risk_corr_blend", 0.60)),
        )

    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -0.95, 0.95)
    np.fill_diagonal(corr, 1.0)
    scenario_vol = baseline_vol.to_numpy(dtype=float) * vol_scale.to_numpy(dtype=float)
    covariance = pd.DataFrame(
        np.diag(scenario_vol) @ corr @ np.diag(scenario_vol),
        index=assets,
        columns=assets,
    )
    return _nearest_psd(covariance, jitter=float(params.get("cov_jitter", params.get("eps_variance", 1.0e-10))))


def _momentum_signal(forecast: pd.DataFrame, assets: pd.Index) -> pd.Series:
    for column in (
        "momentum_score",
        "momentum_63",
        "momentum_126",
        "relative_strength",
        "trend_score",
        "raw_signal",
        "expected_return_horizon",
        "expected_return_3m",
    ):
        if column in forecast.columns:
            signal = pd.to_numeric(forecast[column].reindex(assets), errors="coerce")
            if signal.isna().any() or not np.isfinite(signal.to_numpy(dtype=float)).all():
                continue
            return signal.astype(float)
    raise ValueError("forecast_table requires a momentum or expected return column for scenario adjustments.")


def _positive_signal_intensity(signal: pd.Series) -> pd.Series:
    positive = signal.astype(float).clip(lower=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scale = float(positive.max())
    if scale <= 0.0:
        return positive * 0.0
    return (positive / scale).clip(0.0, 1.0)


def _forecast_volatility_horizon(
    forecast: pd.DataFrame,
    assets: pd.Index,
    params: dict[str, Any],
    *,
    baseline_vol: pd.Series,
) -> pd.Series:
    if "volatility_horizon" in forecast.columns:
        values = pd.to_numeric(forecast["volatility_horizon"].reindex(assets), errors="coerce")
        source = "forecast_table.volatility_horizon"
    elif "volatility_3m" in forecast.columns:
        values = pd.to_numeric(forecast["volatility_3m"].reindex(assets), errors="coerce") * np.sqrt(max(_horizon_scale(params), 0.0))
        source = "forecast_table.volatility_3m_scaled_to_horizon"
    else:
        values = baseline_vol.reindex(assets)
        source = "historical_returns_covariance"
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        bad_assets = values.index[values.isna() | ~np.isfinite(values.to_numpy(dtype=float))].tolist()
        raise ValueError(f"forecast/historical volatility contains non-finite values for: {bad_assets}")
    result = values.astype(float)
    result.attrs["volatility_source"] = source
    return result


def _volatility_source_label(forecast: pd.DataFrame) -> str:
    if "volatility_horizon" in forecast.columns:
        return "forecast_table.volatility_horizon"
    if "volatility_3m" in forecast.columns:
        return "forecast_table.volatility_3m_scaled_to_horizon"
    return "historical_returns_covariance"


def _optional_forecast_series(
    forecast: pd.DataFrame,
    column: str,
    assets: pd.Index,
    default: float,
) -> pd.Series:
    if column not in forecast.columns:
        return pd.Series(default, index=assets, dtype=float)
    values = pd.to_numeric(forecast[column].reindex(assets), errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        bad_assets = values.index[values.isna() | ~np.isfinite(values.to_numpy(dtype=float))].tolist()
        raise ValueError(f"forecast column {column} contains non-finite values for: {bad_assets}")
    return values.astype(float)


def _cov_to_corr(covariance: pd.DataFrame) -> pd.DataFrame:
    values = covariance.to_numpy(dtype=float, copy=True)
    vols = np.sqrt(np.maximum(np.diag(values), 0.0))
    denom = np.outer(vols, vols)
    corr = np.divide(values, denom, out=np.zeros_like(values), where=denom > 0.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(np.clip(corr, -1.0, 1.0), index=covariance.index, columns=covariance.columns)


def _blend_corr(
    corr: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    target: float,
    blend: float,
) -> np.ndarray:
    adjusted = corr.copy()
    for i, left in enumerate(mask_a):
        for j, right in enumerate(mask_b):
            if i == j or not (left and right):
                continue
            adjusted[i, j] = (1.0 - blend) * adjusted[i, j] + blend * target
            adjusted[j, i] = adjusted[i, j]
    return adjusted


def _nearest_psd(covariance: pd.DataFrame, *, jitter: float) -> pd.DataFrame:
    values = covariance.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("covariance contains non-finite values before PSD repair.")
    values = 0.5 * (values + values.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(values)
        eigvals = np.clip(eigvals, max(float(jitter), 0.0), None)
        values = (eigvecs * eigvals) @ eigvecs.T
    except np.linalg.LinAlgError:
        values = np.diag(np.maximum(np.diag(values), max(float(jitter), 0.0)))
    values = 0.5 * (values + values.T)
    values[np.diag_indices_from(values)] = np.maximum(np.diag(values), max(float(jitter), 0.0))
    return pd.DataFrame(values, index=covariance.index, columns=covariance.columns)


__all__ = [
    "DEFAULT_SCENARIO_PROBABILITIES",
    "SCENARIO_NAMES",
    "build_scenario_inputs",
    "risk_free_return_for_horizon",
]
