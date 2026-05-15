"""Multi-horizon tactical forecast and asset scoring layer.

This module is intentionally report-only at first. It does not change the
Daily Bot's final allocation or order decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from features import compute_returns


DEFAULT_PROJECT_END_DATE = "2026-07-24"


@dataclass(slots=True)
class TacticalForecastResult:
    """Container for tactical multi-horizon forecast outputs."""

    as_of: pd.Timestamp
    project_end_date: pd.Timestamp
    remaining_trading_days: int
    table: pd.DataFrame
    report_text: str


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _clip_series(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=lower, upper=upper)


def _zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float).replace([np.inf, -np.inf], np.nan)
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    mean = float(values.mean())
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _trailing_return(prices: pd.DataFrame, window: int) -> pd.Series:
    if len(prices) <= window:
        return pd.Series(0.0, index=prices.columns)
    latest = prices.iloc[-1]
    base = prices.iloc[-(window + 1)]
    return (latest / base - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _annualized_like_horizon_vol(returns: pd.DataFrame, window: int, horizon: int) -> pd.Series:
    if returns.empty:
        return pd.Series(0.0, index=returns.columns)
    sample = returns.tail(max(window, 2))
    return (sample.std(ddof=0) * np.sqrt(float(horizon))).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _prob_up_from_signal(signal: pd.Series, vol: pd.Series) -> pd.Series:
    scaled = signal.astype(float) / vol.astype(float).replace(0.0, np.nan)
    scaled = scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prob = 0.50 + 0.25 * np.tanh(scaled)
    return prob.clip(lower=0.05, upper=0.95)


def _remaining_trading_days(as_of: pd.Timestamp, project_end_date: pd.Timestamp) -> int:
    start = (as_of + pd.Timedelta(days=1)).date()
    end = (project_end_date + pd.Timedelta(days=1)).date()
    if pd.Timestamp(start) > project_end_date:
        return 0
    return int(max(np.busday_count(start, end), 0))


def _reason_for_row(row: pd.Series) -> str:
    reasons: list[str] = []
    if _safe_float(row.get("momentum_5d")) > 0.0:
        reasons.append("positive_5d_momentum")
    if _safe_float(row.get("momentum_20d")) > 0.0:
        reasons.append("positive_20d_momentum")
    if _safe_float(row.get("prob_up_5d")) >= 0.55:
        reasons.append("prob_up_5d_above_55pct")
    if _safe_float(row.get("vol_adjusted_momentum_20d")) > 0.0:
        reasons.append("positive_vol_adjusted_momentum")
    if _safe_float(row.get("overextension_score")) > 1.0:
        reasons.append("overextension_warning")
    if _safe_float(row.get("drawdown_60d")) < -0.10:
        reasons.append("drawdown_warning")
    if not reasons:
        reasons.append("neutral_or_mixed_signal")
    return ", ".join(reasons)


def build_multi_horizon_forecast(
    *,
    prices: pd.DataFrame,
    returns: pd.DataFrame | None = None,
    date: pd.Timestamp | str | None = None,
    params: dict[str, object] | None = None,
    tickers: Sequence[str] | None = None,
    project_end_date: str | pd.Timestamp | None = None,
) -> TacticalForecastResult:
    """Build a report-only multi-horizon tactical forecast table."""

    if prices.empty:
        raise ValueError("prices must not be empty.")

    params = dict(params or {})
    as_of = pd.Timestamp(date if date is not None else prices.index[-1])
    project_end = pd.Timestamp(project_end_date or params.get("project_end_date", DEFAULT_PROJECT_END_DATE))
    active_tickers = [str(ticker) for ticker in (tickers or prices.columns.tolist()) if str(ticker) in prices.columns]
    if not active_tickers:
        raise ValueError("No active tickers available for tactical forecast.")

    history = prices.reindex(columns=active_tickers).loc[:as_of].sort_index().ffill(limit=3)
    if len(history.dropna(how="all")) < 61:
        raise ValueError("At least 61 price observations are required for tactical forecast.")

    if returns is None:
        ret = compute_returns(history)
    else:
        ret = returns.reindex(columns=active_tickers).loc[:as_of].sort_index().fillna(0.0)

    latest = history.iloc[-1].astype(float)
    remaining_days = _remaining_trading_days(as_of, project_end)

    ret_1d = _trailing_return(history, 1)
    ret_3d = _trailing_return(history, 3)
    ret_5d = _trailing_return(history, 5)
    ret_10d = _trailing_return(history, 10)
    ret_20d = _trailing_return(history, 20)
    ret_60d = _trailing_return(history, 60)

    vol_5d = _annualized_like_horizon_vol(ret, 20, 5)
    vol_10d = _annualized_like_horizon_vol(ret, 30, 10)
    vol_20d = _annualized_like_horizon_vol(ret, 60, 20)
    vol_60d = _annualized_like_horizon_vol(ret, 60, 60)

    mean_daily_20d = ret.tail(20).mean().reindex(active_tickers).fillna(0.0)
    mean_daily_60d = ret.tail(60).mean().reindex(active_tickers).fillna(0.0)

    ma_20 = history.tail(20).mean()
    ma_60 = history.tail(60).mean()
    rolling_peak_60 = history.tail(60).cummax().iloc[-1]

    trend_score = 0.60 * (latest / ma_20 - 1.0) + 0.40 * (latest / ma_60 - 1.0)
    drawdown_60d = latest / rolling_peak_60 - 1.0
    overextension_score = _zscore(ret_5d - (ret_20d / 4.0)).clip(lower=0.0)

    kappa_short = float(params.get("tactical_kappa_short", 0.45))
    kappa_medium = float(params.get("tactical_kappa_medium", 0.35))
    forecast_clip = float(params.get("tactical_forecast_clip", 0.15))

    raw_1d = 0.60 * ret_1d + 0.25 * ret_3d / 3.0 + 0.15 * mean_daily_20d
    raw_3d = 0.45 * ret_3d + 0.35 * ret_5d * (3.0 / 5.0) + 0.20 * mean_daily_20d * 3.0
    raw_5d = 0.40 * ret_5d + 0.35 * ret_10d * 0.5 + 0.25 * mean_daily_20d * 5.0
    raw_10d = 0.35 * ret_10d + 0.35 * ret_20d * 0.5 + 0.30 * mean_daily_60d * 10.0
    raw_20d = 0.40 * ret_20d + 0.35 * ret_60d * (20.0 / 60.0) + 0.25 * mean_daily_60d * 20.0

    expected_return_1d = _clip_series(kappa_short * raw_1d, -forecast_clip, forecast_clip)
    expected_return_3d = _clip_series(kappa_short * raw_3d, -forecast_clip, forecast_clip)
    expected_return_5d = _clip_series(kappa_short * raw_5d, -forecast_clip, forecast_clip)
    expected_return_10d = _clip_series(kappa_medium * raw_10d, -forecast_clip, forecast_clip)
    expected_return_20d = _clip_series(kappa_medium * raw_20d, -forecast_clip, forecast_clip)

    daily_project_signal = 0.35 * mean_daily_20d + 0.35 * mean_daily_60d + 0.20 * ret_20d / 20.0 + 0.10 * trend_score / 20.0
    expected_return_to_project_end = _clip_series(
        kappa_medium * daily_project_signal * float(max(remaining_days, 1)),
        -forecast_clip,
        forecast_clip,
    )

    prob_up_1d = _prob_up_from_signal(expected_return_1d, _annualized_like_horizon_vol(ret, 20, 1))
    prob_up_3d = _prob_up_from_signal(expected_return_3d, vol_5d)
    prob_up_5d = _prob_up_from_signal(expected_return_5d, vol_5d)
    prob_up_10d = _prob_up_from_signal(expected_return_10d, vol_10d)

    vol_adjusted_momentum_20d = ret_20d / vol_20d.replace(0.0, np.nan)
    vol_adjusted_momentum_20d = vol_adjusted_momentum_20d.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    signal_agreement = (
        (ret_3d > 0.0).astype(float)
        + (ret_5d > 0.0).astype(float)
        + (ret_20d > 0.0).astype(float)
        + (trend_score > 0.0).astype(float)
    ) / 4.0
    forecast_confidence = (
        0.30
        + 0.35 * signal_agreement
        + 0.20 * prob_up_5d
        - 0.10 * _zscore(vol_20d).clip(lower=0.0)
        - 0.10 * overextension_score
    ).clip(lower=0.05, upper=1.00)

    relative_strength_rank = ret_20d.rank(ascending=False, method="dense")
    risk_adjusted_forecast = expected_return_5d / vol_5d.replace(0.0, np.nan)
    risk_adjusted_forecast = risk_adjusted_forecast.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    tactical_score = (
        0.25 * _zscore(ret_3d)
        + 0.25 * _zscore(ret_5d)
        + 0.20 * _zscore(ret_20d)
        + 0.15 * _zscore(prob_up_5d)
        + 0.10 * _zscore(trend_score)
        + 0.10 * _zscore(expected_return_to_project_end)
        - 0.15 * _zscore(vol_20d)
        - 0.10 * overextension_score
    )
    tactical_score = tactical_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    table = pd.DataFrame(
        {
            "ticker": active_tickers,
            "as_of": str(as_of.date()),
            "project_end_date": str(project_end.date()),
            "remaining_trading_days": int(remaining_days),
            "latest_price": latest.reindex(active_tickers).to_numpy(dtype=float),
            "return_1d": ret_1d.reindex(active_tickers).to_numpy(dtype=float),
            "return_3d": ret_3d.reindex(active_tickers).to_numpy(dtype=float),
            "return_5d": ret_5d.reindex(active_tickers).to_numpy(dtype=float),
            "return_10d": ret_10d.reindex(active_tickers).to_numpy(dtype=float),
            "return_20d": ret_20d.reindex(active_tickers).to_numpy(dtype=float),
            "return_60d": ret_60d.reindex(active_tickers).to_numpy(dtype=float),
            "momentum_3d": ret_3d.reindex(active_tickers).to_numpy(dtype=float),
            "momentum_5d": ret_5d.reindex(active_tickers).to_numpy(dtype=float),
            "momentum_10d": ret_10d.reindex(active_tickers).to_numpy(dtype=float),
            "momentum_20d": ret_20d.reindex(active_tickers).to_numpy(dtype=float),
            "momentum_60d": ret_60d.reindex(active_tickers).to_numpy(dtype=float),
            "vol_5d": vol_5d.reindex(active_tickers).to_numpy(dtype=float),
            "vol_10d": vol_10d.reindex(active_tickers).to_numpy(dtype=float),
            "vol_20d": vol_20d.reindex(active_tickers).to_numpy(dtype=float),
            "vol_60d": vol_60d.reindex(active_tickers).to_numpy(dtype=float),
            "vol_adjusted_momentum_20d": vol_adjusted_momentum_20d.reindex(active_tickers).to_numpy(dtype=float),
            "mean_reversion_score": (-overextension_score).reindex(active_tickers).to_numpy(dtype=float),
            "overextension_score": overextension_score.reindex(active_tickers).to_numpy(dtype=float),
            "trend_score": trend_score.reindex(active_tickers).to_numpy(dtype=float),
            "drawdown_60d": drawdown_60d.reindex(active_tickers).to_numpy(dtype=float),
            "relative_strength_rank": relative_strength_rank.reindex(active_tickers).to_numpy(dtype=float),
            "expected_return_1d": expected_return_1d.reindex(active_tickers).to_numpy(dtype=float),
            "expected_return_3d": expected_return_3d.reindex(active_tickers).to_numpy(dtype=float),
            "expected_return_5d": expected_return_5d.reindex(active_tickers).to_numpy(dtype=float),
            "expected_return_10d": expected_return_10d.reindex(active_tickers).to_numpy(dtype=float),
            "expected_return_20d": expected_return_20d.reindex(active_tickers).to_numpy(dtype=float),
            "expected_return_to_project_end": expected_return_to_project_end.reindex(active_tickers).to_numpy(dtype=float),
            "prob_up_1d": prob_up_1d.reindex(active_tickers).to_numpy(dtype=float),
            "prob_up_3d": prob_up_3d.reindex(active_tickers).to_numpy(dtype=float),
            "prob_up_5d": prob_up_5d.reindex(active_tickers).to_numpy(dtype=float),
            "prob_up_10d": prob_up_10d.reindex(active_tickers).to_numpy(dtype=float),
            "forecast_confidence": forecast_confidence.reindex(active_tickers).to_numpy(dtype=float),
            "risk_adjusted_forecast": risk_adjusted_forecast.reindex(active_tickers).to_numpy(dtype=float),
            "tactical_score": tactical_score.reindex(active_tickers).to_numpy(dtype=float),
        }
    )
    table = table.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    table["tactical_rank"] = table["tactical_score"].rank(ascending=False, method="dense").astype(int)
    table["reason"] = table.apply(_reason_for_row, axis=1)
    table = table.sort_values(["tactical_score", "forecast_confidence"], ascending=[False, False]).reset_index(drop=True)

    top = table.head(8)
    bottom = table.tail(8).sort_values("tactical_score", ascending=True)
    report_lines = [
        "Tactical Multi-Horizon Forecast Report",
        "",
        f"as_of: {as_of.date()}",
        f"project_end_date: {project_end.date()}",
        f"remaining_trading_days: {remaining_days}",
        f"asset_count: {len(table)}",
        "",
        "method:",
        "- Report-only tactical forecast layer; does not change final Daily Bot orders.",
        "- Combines 1d/3d/5d/10d/20d momentum, probability-up proxy, volatility, trend and overextension risk.",
        "- Designed for active paper-trading diagnostics before changing the optimizer objective.",
        "",
        "top_tactical_assets:",
    ]
    for row in top.itertuples(index=False):
        report_lines.append(
            f"- {row.ticker}: rank={row.tactical_rank}, score={row.tactical_score:.4f}, "
            f"prob_up_5d={row.prob_up_5d:.3f}, exp_5d={row.expected_return_5d:.4f}, reason={row.reason}"
        )
    report_lines.extend(["", "weakest_tactical_assets:"])
    for row in bottom.itertuples(index=False):
        report_lines.append(
            f"- {row.ticker}: rank={row.tactical_rank}, score={row.tactical_score:.4f}, "
            f"prob_up_5d={row.prob_up_5d:.3f}, exp_5d={row.expected_return_5d:.4f}, reason={row.reason}"
        )
    report_text = "\n".join(report_lines) + "\n"

    return TacticalForecastResult(
        as_of=as_of,
        project_end_date=project_end,
        remaining_trading_days=remaining_days,
        table=table,
        report_text=report_text,
    )


def write_tactical_forecast_outputs(
    result: TacticalForecastResult,
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write tactical forecast CSV and text outputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    multi_path = output_path / "multi_horizon_forecast.csv"
    score_path = output_path / "tactical_asset_scores.csv"
    report_path = output_path / "tactical_model_report.txt"

    result.table.to_csv(multi_path, index=False)

    score_columns = [
        "ticker",
        "tactical_rank",
        "tactical_score",
        "forecast_confidence",
        "prob_up_3d",
        "prob_up_5d",
        "prob_up_10d",
        "expected_return_3d",
        "expected_return_5d",
        "expected_return_10d",
        "expected_return_to_project_end",
        "momentum_3d",
        "momentum_5d",
        "momentum_20d",
        "vol_20d",
        "risk_adjusted_forecast",
        "reason",
    ]
    result.table.loc[:, [column for column in score_columns if column in result.table.columns]].to_csv(
        score_path,
        index=False,
    )
    report_path.write_text(result.report_text, encoding="utf-8")

    return {
        "multi_horizon_forecast": str(multi_path),
        "tactical_asset_scores": str(score_path),
        "tactical_model_report": str(report_path),
    }
