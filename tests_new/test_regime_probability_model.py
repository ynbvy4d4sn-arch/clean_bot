from __future__ import annotations

import numpy as np
import pandas as pd

from src_new.regimes.regime_probability_model import (
    SCENARIO_NAMES,
    estimate_scenario_probabilities,
)


def _price_frame(daily_return: float, days: int = 140, assets: int = 4) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=days, freq="B")
    data = {}
    for i in range(assets):
        drift = daily_return + i * 0.00005
        values = 100.0 * np.cumprod(np.full(days, 1.0 + drift))
        data[f"A{i}"] = values
    return pd.DataFrame(data, index=index)


def test_probabilities_sum_to_one_and_have_expected_names():
    prices = _price_frame(0.0005)
    result = estimate_scenario_probabilities(prices)

    assert list(result.probabilities.index) == SCENARIO_NAMES
    assert abs(float(result.probabilities.sum()) - 1.0) < 1e-12
    assert (result.probabilities >= 0.0).all()


def test_positive_momentum_increases_bull_relative_to_stress_case():
    up_prices = _price_frame(0.0010)
    down_prices = _price_frame(-0.0025)

    up = estimate_scenario_probabilities(up_prices).probabilities
    down = estimate_scenario_probabilities(down_prices).probabilities

    assert float(up["bull_momentum"]) > float(down["bull_momentum"])
    assert float(down["bear_risk_off"]) >= float(up["bear_risk_off"])


def test_empty_prices_returns_valid_default_probabilities():
    result = estimate_scenario_probabilities(pd.DataFrame())

    assert abs(float(result.probabilities.sum()) - 1.0) < 1e-12
    assert (result.probabilities >= 0.0).all()
