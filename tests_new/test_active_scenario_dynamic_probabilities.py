from __future__ import annotations

import numpy as np
import pandas as pd

from scenario_daily_pipeline import _dynamic_scenario_weighted_probabilities
from scenarios import SCENARIO_NAMES, build_scenario_inputs


def _returns_frame(days: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=days, freq="B")
    return pd.DataFrame(
        {
            "SPY": np.full(days, 0.0010),
            "XLK": np.full(days, 0.0012),
            "IEF": np.full(days, 0.0001),
        },
        index=index,
    )


def _forecast_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expected_return_3m": [0.05, 0.06, 0.01],
            "volatility_3m": [0.10, 0.12, 0.03],
            "raw_signal": [0.02, 0.03, 0.00],
        },
        index=pd.Index(["SPY", "XLK", "IEF"], name="ticker"),
    )


def test_dynamic_mapper_returns_active_six_scenario_schema():
    probabilities = _dynamic_scenario_weighted_probabilities(
        returns=_returns_frame(),
        params={"market_ticker": "SPY"},
    )

    assert probabilities is not None
    assert list(probabilities.index) == list(SCENARIO_NAMES)
    assert abs(float(probabilities.sum()) - 1.0) < 1e-12
    assert (probabilities >= 0.0).all()


def test_build_scenario_inputs_prefers_dynamic_probabilities():
    dynamic = _dynamic_scenario_weighted_probabilities(
        returns=_returns_frame(),
        params={"market_ticker": "SPY"},
    )
    assert dynamic is not None

    scenarios = build_scenario_inputs(
        forecast_table=_forecast_table(),
        returns=_returns_frame(),
        config={
            "dynamic_scenario_weighted_probabilities": dynamic.to_dict(),
            "allow_unknown_asset_groups": True,
            "risk_free_rate_annual": 0.02,
            "horizon_days": 63,
        },
    )

    assert [s.name for s in scenarios] == list(SCENARIO_NAMES)
    assert abs(sum(float(s.probability) for s in scenarios) - 1.0) < 1e-12
    assert {s.metadata["probability_source"] for s in scenarios} == {
        "dynamic_regime_probability_model"
    }
