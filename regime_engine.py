"""Rule-based regime detection using available market proxy series."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _safe_trend(prices: pd.DataFrame, ticker: str, window: int = 63) -> float:
    if ticker not in prices.columns:
        return 0.0
    series = prices[ticker].dropna()
    if len(series) <= window:
        return 0.0
    return float(series.iloc[-1] / series.iloc[-(window + 1)] - 1.0)


def _average_trend(prices: pd.DataFrame, tickers: list[str], window: int = 63) -> float:
    available = [ticker for ticker in tickers if ticker in prices.columns]
    if not available:
        return 0.0
    values = [_safe_trend(prices, ticker, window) for ticker in available]
    return float(sum(values) / len(values))


def detect_regime(prices: pd.DataFrame, date: pd.Timestamp | str) -> dict[str, Any]:
    """Detect a simple market regime from proxy price series up to `date`."""

    as_of = pd.Timestamp(date)
    history = prices.loc[:as_of].sort_index()
    core_equity_trend = _average_trend(history, ["SPHQ", "SPMO", "SPLV", "XLK", "XLI", "XLV"], 63)
    cyclical_vs_defensive = _average_trend(history, ["SPMO", "XLK", "XLC", "XLI"], 63) - _average_trend(
        history,
        ["SPLV", "XLP", "XLU", "XLV"],
        63,
    )
    lqd_vs_ief = _safe_trend(history, "LQD", 63) - _safe_trend(history, "IEF", 63)
    duration_trend = _average_trend(history, ["IEF", "AGG"], 63)
    commodity_trend = _average_trend(history, ["PDBC", "GLD", "SLV"], 63)
    crypto_trend = _safe_trend(history, "IBIT", 63)
    hedge_trend = _safe_trend(history, "SH", 21)
    inflation_trend = _safe_trend(history, "TIP", 63) - _safe_trend(history, "IEF", 63)

    risk_off_score = 0.0
    risk_off_score += 0.6 if core_equity_trend < -0.08 else 0.0
    risk_off_score += 0.2 if lqd_vs_ief < -0.02 else 0.0
    risk_off_score += 0.2 if hedge_trend > 0.05 else 0.0
    risk_state = "risk_off" if risk_off_score >= 0.5 else "normal"

    regime_scores = {
        "risk_on": max(core_equity_trend, 0.0) + max(cyclical_vs_defensive, 0.0) + max(crypto_trend, 0.0),
        "neutral": 0.20,
        "risk_off": max(risk_off_score, 0.0),
        "inflation_pressure": max(commodity_trend, 0.0) + max(-duration_trend, 0.0) + max(inflation_trend, 0.0),
        "rates_down": max(duration_trend, 0.0),
        "liquidity_stress": max(hedge_trend, 0.0) + max(-lqd_vs_ief, 0.0),
        "commodity_shock": max(commodity_trend, 0.0),
        "crypto_risk_on": max(crypto_trend, 0.0),
    }
    primary_regime = max(regime_scores, key=regime_scores.get)

    scenario_probability_adjustments = {
        "bull_momentum": -0.05 if risk_state == "risk_off" else 0.03,
        "bear_risk_off": 0.07 if risk_state == "risk_off" else -0.02,
        "correlation_stress": 0.05 if risk_state == "risk_off" else -0.01,
    }
    diagnostics = {
        "core_equity_63d_momentum": core_equity_trend,
        "cyclical_vs_defensive_relative_strength": cyclical_vs_defensive,
        "lqd_vs_ief_relative_strength": lqd_vs_ief,
        "duration_trend": duration_trend,
        "commodity_trend": commodity_trend,
        "crypto_trend": crypto_trend,
        "hedge_trend": hedge_trend,
        "inflation_trend": inflation_trend,
    }
    return {
        "primary_regime": primary_regime,
        "regime_scores": regime_scores,
        "risk_state": risk_state,
        "scenario_probability_adjustments": scenario_probability_adjustments,
        "diagnostics": diagnostics,
    }


def save_regime_report(regime_result: dict[str, Any], csv_path: str | Path, txt_path: str | Path) -> None:
    """Persist regime outputs to CSV and TXT."""

    csv_file = Path(csv_path)
    txt_file = Path(txt_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    txt_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "primary_regime": regime_result.get("primary_regime", "neutral"),
                "risk_state": regime_result.get("risk_state", "normal"),
                **dict(regime_result.get("regime_scores", {})),
            }
        ]
    ).to_csv(csv_file, index=False)
    txt_file.write_text(
        "\n".join(
            [
                f"Primary Regime: {regime_result.get('primary_regime', 'neutral')}",
                f"Risk State: {regime_result.get('risk_state', 'normal')}",
                "Diagnostics:",
                *[
                    f"- {key}: {value}"
                    for key, value in dict(regime_result.get("diagnostics", {})).items()
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
