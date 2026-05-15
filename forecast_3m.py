"""Forward-looking 3M direct asset forecast layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features import compute_market_risk_state, compute_returns


@dataclass(slots=True)
class Forecast3M:
    """Container for the direct 3M forecast output."""

    as_of: pd.Timestamp
    table: pd.DataFrame
    diagnostics: pd.DataFrame
    risk_state: str


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def build_forecast_3m(
    prices: pd.DataFrame,
    returns: pd.DataFrame | None = None,
    date: pd.Timestamp | str | None = None,
    params: dict[str, object] | None = None,
    tickers: list[str] | None = None,
) -> Forecast3M:
    """Build a direct 3M forecast using only information available up to `date`."""

    if prices.empty:
        raise ValueError("prices must not be empty.")

    params = dict(params or {})
    as_of = pd.Timestamp(date if date is not None else prices.index[-1])
    active_tickers = [str(t) for t in (tickers or prices.columns.tolist()) if str(t) in prices.columns]
    price_history = prices.reindex(columns=active_tickers).loc[:as_of].sort_index().ffill(limit=3)
    if returns is None:
        returns = compute_returns(price_history)
    else:
        returns = returns.reindex(columns=active_tickers).loc[:as_of].sort_index().fillna(0.0)

    if len(price_history) < 127:
        raise ValueError("At least 127 price observations are required for the 3M forecast layer.")

    risk_state = compute_market_risk_state(prices=prices, date=as_of)

    latest = price_history.iloc[-1]
    px_63 = price_history.iloc[-64]
    px_126 = price_history.iloc[-127]
    ma_126 = price_history.tail(126).mean()
    rolling_peak_126 = price_history.tail(126).cummax().iloc[-1]

    momentum_63 = latest / px_63 - 1.0
    momentum_126 = latest / px_126 - 1.0
    relative_strength = momentum_63 - float(momentum_63.median())
    trend_score = latest / ma_126 - 1.0
    drawdown_126 = latest / rolling_peak_126 - 1.0

    vol_63 = returns.tail(63).std(ddof=0) * np.sqrt(63.0)
    downside_63 = returns.tail(63).where(returns.tail(63) < 0.0, 0.0).std(ddof=0) * np.sqrt(63.0)
    median_vol = float(vol_63.median()) if not vol_63.empty else 0.0

    kappa = float(params.get("kappa", 0.20))
    clip_lower = float(params.get("forecast_clip_lower", -0.20))
    clip_upper = float(params.get("forecast_clip_upper", 0.20))

    rows: list[dict[str, object]] = []
    for ticker in active_tickers:
        m63 = float(momentum_63.get(ticker, 0.0))
        m126 = float(momentum_126.get(ticker, 0.0))
        rs = float(relative_strength.get(ticker, 0.0))
        trend = float(trend_score.get(ticker, 0.0))
        drawdown = float(drawdown_126.get(ticker, 0.0))
        vol = max(float(vol_63.get(ticker, 0.0)), 0.0)
        downside = max(float(downside_63.get(ticker, 0.0)), 0.0)

        raw_signal = 0.45 * m63 + 0.30 * m126 + 0.15 * rs + 0.10 * trend

        confidence = 0.35
        if m63 > 0.0 and m126 > 0.0:
            confidence += 0.20
        if np.sign(m63) == np.sign(m126) and abs(m63) + abs(m126) > 0.0:
            confidence += 0.10
        if float(latest.get(ticker, 0.0)) > float(ma_126.get(ticker, 0.0)):
            confidence += 0.15
        if rs > 0.0:
            confidence += 0.10
        if drawdown < -0.10:
            confidence -= 0.15
        if vol > median_vol and median_vol > 0.0:
            confidence -= 0.15
        confidence = _clip(confidence, 0.05, 1.00)

        drawdown_penalty = max(abs(min(drawdown + 0.10, 0.0)), 0.0)
        volatility_penalty = max((vol - median_vol), 0.0) if median_vol > 0.0 else 0.0
        uncertainty_multiplier = 1.0 + 0.5 * (1.0 - confidence) + drawdown_penalty + volatility_penalty
        uncertainty_multiplier = _clip(uncertainty_multiplier, 1.0, 3.0)

        expected_return_3m = _clip(kappa * confidence * raw_signal, clip_lower, clip_upper)

        rows.append(
            {
                "ticker": ticker,
                "expected_return_3m": expected_return_3m,
                "volatility_3m": vol,
                "downside_risk_3m": downside,
                "signal_confidence": confidence,
                "uncertainty_multiplier": uncertainty_multiplier,
                "risk_state": risk_state,
                "raw_signal": raw_signal,
                "momentum_63": m63,
                "momentum_126": m126,
                "relative_strength": rs,
                "trend_score": trend,
                "drawdown_126": drawdown,
            }
        )

    table = pd.DataFrame(rows).set_index("ticker").sort_values(
        ["expected_return_3m", "signal_confidence"],
        ascending=[False, False],
    )
    diagnostics = table[
        ["momentum_63", "momentum_126", "relative_strength", "trend_score", "drawdown_126", "raw_signal"]
    ].copy()
    return Forecast3M(as_of=as_of, table=table, diagnostics=diagnostics, risk_state=risk_state)
