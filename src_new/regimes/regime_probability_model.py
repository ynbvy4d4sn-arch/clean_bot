from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


SCENARIO_NAMES = [
    "base",
    "bull_momentum",
    "bear_risk_off",
    "correlation_stress",
    "mean_reversion",
]


@dataclass(frozen=True)
class RegimeIndicators:
    momentum_score: float
    volatility_score: float
    drawdown_score: float
    correlation_score: float
    stress_score: float
    risk_on_score: float


@dataclass(frozen=True)
class RegimeProbabilityResult:
    probabilities: pd.Series
    indicators: RegimeIndicators
    diagnostics: dict[str, float | str]


def _safe_last_valid(series: pd.Series, default: float = 0.0) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float(default)
    return float(valid.iloc[-1])


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _normalize_probability_dict(values: dict[str, float]) -> pd.Series:
    s = pd.Series(values, dtype=float).reindex(SCENARIO_NAMES).fillna(0.0)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(s.sum())
    if total <= 0.0:
        s = pd.Series(
            {
                "base": 0.50,
                "bull_momentum": 0.15,
                "bear_risk_off": 0.15,
                "correlation_stress": 0.10,
                "mean_reversion": 0.10,
            },
            dtype=float,
        )
        total = float(s.sum())
    return s / total


def compute_regime_indicators(
    prices: pd.DataFrame,
    market_ticker: str | None = None,
    lookback_short: int = 21,
    lookback_medium: int = 63,
    volatility_lookback: int = 63,
    correlation_lookback: int = 63,
) -> RegimeIndicators:
    """
    Computes simple, robust market-state indicators from price history.

    Scores are clipped to [0, 1]:
    - momentum_score: high when recent market/index momentum is positive
    - volatility_score: high when recent annualized volatility is elevated
    - drawdown_score: high when market is far below recent high
    - correlation_score: high when average asset correlation is elevated
    - stress_score: combined risk-off pressure
    - risk_on_score: combined risk-on pressure
    """
    if prices is None or prices.empty:
        return RegimeIndicators(
            momentum_score=0.0,
            volatility_score=0.0,
            drawdown_score=0.0,
            correlation_score=0.0,
            stress_score=0.0,
            risk_on_score=0.0,
        )

    clean = prices.copy()
    clean = clean.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    clean = clean.ffill().dropna(axis=1, how="all")

    if clean.empty or len(clean) < max(lookback_short, 10):
        return RegimeIndicators(
            momentum_score=0.0,
            volatility_score=0.0,
            drawdown_score=0.0,
            correlation_score=0.0,
            stress_score=0.0,
            risk_on_score=0.0,
        )

    returns = clean.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    if market_ticker and market_ticker in clean.columns:
        market_price = clean[market_ticker].dropna()
    else:
        market_price = clean.mean(axis=1).dropna()

    market_return_short = (
        market_price.iloc[-1] / market_price.iloc[-lookback_short] - 1.0
        if len(market_price) > lookback_short and market_price.iloc[-lookback_short] != 0
        else 0.0
    )

    market_return_medium = (
        market_price.iloc[-1] / market_price.iloc[-lookback_medium] - 1.0
        if len(market_price) > lookback_medium and market_price.iloc[-lookback_medium] != 0
        else market_return_short
    )

    momentum_raw = 0.50 * market_return_short + 0.50 * market_return_medium
    momentum_score = _clip01((momentum_raw + 0.08) / 0.16)

    recent_returns = returns.tail(volatility_lookback)
    market_returns = market_price.pct_change(fill_method=None).dropna().tail(volatility_lookback)

    realized_vol = float(market_returns.std(ddof=0) * np.sqrt(252.0)) if len(market_returns) > 5 else 0.0
    volatility_score = _clip01((realized_vol - 0.12) / 0.25)

    rolling_high = market_price.tail(max(lookback_medium, lookback_short)).max()
    last_price = _safe_last_valid(market_price, default=float("nan"))
    drawdown = 0.0
    if math.isfinite(last_price) and rolling_high and rolling_high > 0:
        drawdown = float(last_price / rolling_high - 1.0)
    drawdown_score = _clip01(abs(min(drawdown, 0.0)) / 0.20)

    corr_score = 0.0
    recent_corr_returns = recent_returns.dropna(axis=1, how="all")
    if recent_corr_returns.shape[1] >= 2 and len(recent_corr_returns.dropna(how="all")) >= 10:
        corr = recent_corr_returns.corr().replace([np.inf, -np.inf], np.nan)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        avg_corr = float(upper.mean()) if not upper.empty else 0.0
        corr_score = _clip01((avg_corr - 0.25) / 0.50)

    stress_score = _clip01(
        0.35 * volatility_score
        + 0.35 * drawdown_score
        + 0.20 * corr_score
        + 0.10 * (1.0 - momentum_score)
    )

    risk_on_score = _clip01(
        0.55 * momentum_score
        + 0.25 * (1.0 - volatility_score)
        + 0.20 * (1.0 - drawdown_score)
    )

    return RegimeIndicators(
        momentum_score=float(momentum_score),
        volatility_score=float(volatility_score),
        drawdown_score=float(drawdown_score),
        correlation_score=float(corr_score),
        stress_score=float(stress_score),
        risk_on_score=float(risk_on_score),
    )


def estimate_scenario_probabilities(
    prices: pd.DataFrame,
    market_ticker: str | None = None,
) -> RegimeProbabilityResult:
    """
    Converts market indicators into scenario probabilities.

    This is intentionally transparent and conservative:
    - base remains the anchor
    - bull_momentum rises with risk_on_score
    - bear_risk_off rises with stress_score and drawdown
    - correlation_stress rises with correlation and volatility
    - mean_reversion rises when drawdown is high but momentum stabilizes
    """
    ind = compute_regime_indicators(prices=prices, market_ticker=market_ticker)

    base = 0.42
    bull = 0.10 + 0.25 * ind.risk_on_score
    bear = 0.08 + 0.28 * ind.stress_score
    corr = 0.06 + 0.22 * (0.60 * ind.correlation_score + 0.40 * ind.volatility_score)
    mean_rev = 0.06 + 0.18 * (0.65 * ind.drawdown_score + 0.35 * ind.momentum_score)

    # When stress is very high, reduce bull and base.
    base *= 1.0 - 0.30 * ind.stress_score
    bull *= 1.0 - 0.50 * ind.stress_score

    probs = _normalize_probability_dict(
        {
            "base": base,
            "bull_momentum": bull,
            "bear_risk_off": bear,
            "correlation_stress": corr,
            "mean_reversion": mean_rev,
        }
    )

    diagnostics = {
        "momentum_score": ind.momentum_score,
        "volatility_score": ind.volatility_score,
        "drawdown_score": ind.drawdown_score,
        "correlation_score": ind.correlation_score,
        "stress_score": ind.stress_score,
        "risk_on_score": ind.risk_on_score,
        "dominant_scenario": str(probs.idxmax()),
        "dominant_probability": float(probs.max()),
    }

    return RegimeProbabilityResult(
        probabilities=probs,
        indicators=ind,
        diagnostics=diagnostics,
    )
