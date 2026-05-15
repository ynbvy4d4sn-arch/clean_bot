from __future__ import annotations

import numpy as np
import pandas as pd

from scenario_model import build_3m_scenarios


def _forecast_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expected_return_3m": [0.03, 0.01, 0.02],
            "volatility_3m": [0.12, 0.08, 0.10],
            "raw_signal": [0.5, 0.1, 0.2],
            "uncertainty_multiplier": [1.0, 1.0, 1.0],
        },
        index=["SPY", "TLT", "GLD"],
    )


def _covariance() -> pd.DataFrame:
    tickers = ["SPY", "TLT", "GLD"]
    return pd.DataFrame(np.eye(3) * 0.02, index=tickers, columns=tickers)


def _prices(daily_return: float) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=140, freq="B")
    values = 100.0 * np.cumprod(np.full(len(idx), 1.0 + daily_return))
    return pd.DataFrame(
        {
            "SPY": values,
            "TLT": values * 0.8,
            "GLD": values * 1.1,
        },
        index=idx,
    )


def test_build_3m_scenarios_keeps_static_fallback_without_prices():
    scenarios = build_3m_scenarios(
        forecast_table=_forecast_table(),
        covariance_3m=_covariance(),
        risk_state="normal",
        as_of="2026-01-01",
    )

    assert abs(float(scenarios.scenario_probabilities.sum()) - 1.0) < 1e-12
    assert set(scenarios.scenario_returns_matrix.index) == set(scenarios.scenario_probabilities.index)
    assert "probability_source" in scenarios.summary.columns
    assert set(scenarios.summary["probability_source"]) == {"static_default"}


def test_build_3m_scenarios_uses_dynamic_probabilities_when_prices_available():
    scenarios = build_3m_scenarios(
        forecast_table=_forecast_table(),
        covariance_3m=_covariance(),
        risk_state="normal",
        as_of="2026-01-01",
        prices=_prices(0.001),
        market_ticker="SPY",
    )

    assert abs(float(scenarios.scenario_probabilities.sum()) - 1.0) < 1e-12
    assert "probability_source" in scenarios.summary.columns
    assert set(scenarios.summary["probability_source"]) == {"dynamic_regime_probability_model"}
    assert "stress_score" in scenarios.summary.columns
    assert "risk_on_score" in scenarios.summary.columns
