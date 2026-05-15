"""3M factor forecast layer built from proxy factor time series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_factor_forecast(
    factor_data: pd.DataFrame,
    date: pd.Timestamp | str | None = None,
    risk_state: str = "normal",
    horizon_days: int = 63,
) -> pd.DataFrame:
    """Build a simple 3M factor forecast from recent factor history."""

    if factor_data.empty:
        return pd.DataFrame(
            columns=[
                "factor",
                "expected_change_3m",
                "volatility_3m",
                "confidence",
                "uncertainty_multiplier",
                "diagnostics",
            ]
        )

    as_of = pd.Timestamp(date if date is not None else factor_data.index[-1])
    history = factor_data.loc[:as_of].sort_index()
    short = history.tail(min(63, len(history)))
    long = history.tail(min(126, len(history)))

    rows: list[dict[str, object]] = []
    for factor in history.columns:
        short_mean = float(short[factor].mean()) if not short.empty else 0.0
        long_mean = float(long[factor].mean()) if not long.empty else 0.0
        current = float(history[factor].iloc[-1])
        vol = float(short[factor].std(ddof=0) * np.sqrt(max(horizon_days / 63.0, 1e-12))) if not short.empty else 0.0
        expected = 0.6 * short_mean + 0.2 * long_mean - 0.2 * current
        if risk_state == "risk_off" and factor in {"equity_beta", "growth", "risk_appetite", "crypto_beta"}:
            expected -= 0.15 * abs(expected) + 0.05
        confidence = min(max(0.35 + 0.25 * (abs(short_mean) > abs(long_mean) / 2), 0.05), 1.0)
        uncertainty = min(max(1.0 + 0.5 * (1.0 - confidence) + max(vol - 1.0, 0.0), 1.0), 3.0)
        rows.append(
            {
                "factor": factor,
                "expected_change_3m": expected,
                "volatility_3m": vol,
                "confidence": confidence,
                "uncertainty_multiplier": uncertainty,
                "diagnostics": f"current={current:.4f}",
            }
        )
    return pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)
