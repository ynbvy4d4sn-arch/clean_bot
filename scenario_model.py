"""Direct 3M scenario generation for candidate evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from asset_universe import DEFENSIVE_GROUPS, EQUITY_LIKE_GROUPS, get_group_map


@dataclass(slots=True)
class ScenarioSet:
    """Scenario container for robust candidate evaluation."""

    as_of: pd.Timestamp
    scenario_returns_matrix: pd.DataFrame
    scenario_names: list[str]
    scenario_probabilities: pd.Series
    summary: pd.DataFrame
    risk_state: str


def build_3m_scenarios(
    forecast_table: pd.DataFrame,
    covariance_3m: pd.DataFrame,
    risk_state: str,
    as_of: pd.Timestamp | str,
    random_seed: int = 42,
) -> ScenarioSet:
    """Build a small forward-looking direct scenario set."""

    del random_seed  # deterministic rules only for now

    if forecast_table.empty:
        raise ValueError("forecast_table must not be empty.")

    tickers = [str(t) for t in forecast_table.index.tolist()]
    group_map = get_group_map()
    mu = forecast_table["expected_return_3m"].reindex(tickers).fillna(0.0)
    vol = forecast_table["volatility_3m"].reindex(tickers).fillna(0.0)
    raw_signal = forecast_table.get("raw_signal", pd.Series(0.0, index=tickers, dtype=float)).reindex(tickers).fillna(0.0)
    uncertainty = forecast_table.get("uncertainty_multiplier", pd.Series(1.0, index=tickers, dtype=float)).reindex(tickers).fillna(1.0)

    base = mu.copy()
    bull = mu + 0.50 * vol / uncertainty
    bear = mu - 0.75 * vol * uncertainty
    corr = mu - 0.90 * vol * uncertainty
    mean_reversion = mu - 0.35 * raw_signal

    for ticker in tickers:
        group = group_map.get(ticker, "")
        if group in DEFENSIVE_GROUPS:
            bear.loc[ticker] = mu.loc[ticker] - 0.15 * vol.loc[ticker]
            corr.loc[ticker] = mu.loc[ticker] - 0.10 * vol.loc[ticker]
            bull.loc[ticker] = mu.loc[ticker] + 0.10 * vol.loc[ticker]
        elif group in EQUITY_LIKE_GROUPS or group in {"commodities", "crypto"}:
            bear.loc[ticker] = mu.loc[ticker] - 1.00 * vol.loc[ticker] * uncertainty.loc[ticker]
            corr.loc[ticker] = mu.loc[ticker] - 1.15 * vol.loc[ticker] * uncertainty.loc[ticker]

    matrix = pd.DataFrame(
        {
            "base": base,
            "bull_momentum": bull,
            "bear_risk_off": bear,
            "correlation_stress": corr,
            "mean_reversion": mean_reversion,
        }
    ).T.reindex(columns=tickers)

    probabilities = pd.Series(
        {
            "base": 0.50,
            "bull_momentum": 0.15,
            "bear_risk_off": 0.15,
            "correlation_stress": 0.10,
            "mean_reversion": 0.10,
        },
        dtype=float,
    )
    if risk_state == "risk_off":
        probabilities.loc["base"] = 0.40
        probabilities.loc["bull_momentum"] = 0.08
        probabilities.loc["bear_risk_off"] = 0.24
        probabilities.loc["correlation_stress"] = 0.18
        probabilities.loc["mean_reversion"] = 0.10
    probabilities = probabilities / probabilities.sum()

    summary = pd.DataFrame(
        {
            "scenario_name": probabilities.index,
            "probability": probabilities.values,
            "mean_asset_return": [float(matrix.loc[name].mean()) for name in probabilities.index],
            "median_asset_return": [float(matrix.loc[name].median()) for name in probabilities.index],
        }
    )
    return ScenarioSet(
        as_of=pd.Timestamp(as_of),
        scenario_returns_matrix=matrix,
        scenario_names=list(probabilities.index),
        scenario_probabilities=probabilities,
        summary=summary,
        risk_state=risk_state,
    )
