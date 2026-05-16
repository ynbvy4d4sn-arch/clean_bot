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

    risk_free_rate_annual = _safe_float(
        params.get("risk_free_rate_annual", params.get("RISK_FREE_RATE_ANNUAL", 0.02)),
        0.02,
    )

    def risk_free_return_for_days(days: int) -> pd.Series:
        rf_value = (1.0 + risk_free_rate_annual) ** (float(days) / 252.0) - 1.0
        return pd.Series(rf_value, index=active_tickers, dtype=float)

    risk_free_return_1d = risk_free_return_for_days(1)
    risk_free_return_3d = risk_free_return_for_days(3)
    risk_free_return_5d = risk_free_return_for_days(5)
    risk_free_return_10d = risk_free_return_for_days(10)
    risk_free_return_20d = risk_free_return_for_days(20)
    risk_free_return_to_project_end = risk_free_return_for_days(max(remaining_days, 1))

    excess_expected_return_1d = expected_return_1d - risk_free_return_1d
    excess_expected_return_3d = expected_return_3d - risk_free_return_3d
    excess_expected_return_5d = expected_return_5d - risk_free_return_5d
    excess_expected_return_10d = expected_return_10d - risk_free_return_10d
    excess_expected_return_20d = expected_return_20d - risk_free_return_20d
    excess_expected_return_to_project_end = expected_return_to_project_end - risk_free_return_to_project_end

    risk_adjusted_forecast = excess_expected_return_5d / vol_5d.replace(0.0, np.nan)
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
            "risk_free_rate_annual": float(risk_free_rate_annual),
            "risk_free_return_1d": risk_free_return_1d.reindex(active_tickers).to_numpy(dtype=float),
            "risk_free_return_3d": risk_free_return_3d.reindex(active_tickers).to_numpy(dtype=float),
            "risk_free_return_5d": risk_free_return_5d.reindex(active_tickers).to_numpy(dtype=float),
            "risk_free_return_10d": risk_free_return_10d.reindex(active_tickers).to_numpy(dtype=float),
            "risk_free_return_20d": risk_free_return_20d.reindex(active_tickers).to_numpy(dtype=float),
            "risk_free_return_to_project_end": risk_free_return_to_project_end.reindex(active_tickers).to_numpy(dtype=float),
            "expected_return_1d": expected_return_1d.reindex(active_tickers).to_numpy(dtype=float),
            "expected_return_3d": expected_return_3d.reindex(active_tickers).to_numpy(dtype=float),
            "expected_return_5d": expected_return_5d.reindex(active_tickers).to_numpy(dtype=float),
            "expected_return_10d": expected_return_10d.reindex(active_tickers).to_numpy(dtype=float),
            "expected_return_20d": expected_return_20d.reindex(active_tickers).to_numpy(dtype=float),
            "expected_return_to_project_end": expected_return_to_project_end.reindex(active_tickers).to_numpy(dtype=float),
            "excess_expected_return_1d": excess_expected_return_1d.reindex(active_tickers).to_numpy(dtype=float),
            "excess_expected_return_3d": excess_expected_return_3d.reindex(active_tickers).to_numpy(dtype=float),
            "excess_expected_return_5d": excess_expected_return_5d.reindex(active_tickers).to_numpy(dtype=float),
            "excess_expected_return_10d": excess_expected_return_10d.reindex(active_tickers).to_numpy(dtype=float),
            "excess_expected_return_20d": excess_expected_return_20d.reindex(active_tickers).to_numpy(dtype=float),
            "excess_expected_return_to_project_end": excess_expected_return_to_project_end.reindex(active_tickers).to_numpy(dtype=float),
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
        "- Combines 1d/3d/5d/10d/20d momentum, uncalibrated directional-up proxies, volatility, trend, overextension risk and 2%-RF excess-return diagnostics.",
        "- Designed for active paper-trading diagnostics before changing the optimizer objective.",
        "- prob_up_* fields are uncalibrated directional proxies, not calibrated statistical probabilities.",
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
        "excess_expected_return_5d",
        "risk_free_return_5d",
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


def write_tactical_order_alignment(
    *,
    scenario_preview: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, str]:
    """Compare executable order preview against tactical asset scores.

    Report-only diagnostic. Does not change orders.
    """

    output_path = Path(output_dir)
    scores_path = output_path / "tactical_asset_scores.csv"
    alignment_path = output_path / "tactical_order_alignment.csv"
    report_path = output_path / "tactical_order_alignment_report.txt"

    if not scores_path.exists():
        report_path.write_text(
            "Tactical Order Alignment Report\n\nstatus: tactical_asset_scores.csv missing\n",
            encoding="utf-8",
        )
        return {
            "tactical_order_alignment": str(alignment_path),
            "tactical_order_alignment_report": str(report_path),
        }

    scores = pd.read_csv(scores_path)
    if scores.empty or "ticker" not in scores.columns:
        report_path.write_text(
            "Tactical Order Alignment Report\n\nstatus: tactical_asset_scores.csv empty or malformed\n",
            encoding="utf-8",
        )
        return {
            "tactical_order_alignment": str(alignment_path),
            "tactical_order_alignment_report": str(report_path),
        }

    preview = scenario_preview.copy()
    if "asset" not in preview.columns:
        raise ValueError("scenario_preview must include asset column.")

    preview["ticker"] = preview["asset"].astype(str)
    preview["trade_side"] = preview.get("trade_side", "HOLD").astype(str).str.upper()

    scores["ticker"] = scores["ticker"].astype(str)
    merged = preview.merge(scores, on="ticker", how="left", suffixes=("", "_tactical"))

    asset_count = max(int(scores["ticker"].nunique()), 1)
    top_third_rank = max(int(np.ceil(asset_count / 3.0)), 1)
    bottom_third_rank = max(int(np.floor(asset_count * 2.0 / 3.0)), 1)
    cash_proxy_tickers = {"SGOV", "SHY", "BIL", "TFLO", "ICSH"}

    def classify(row: pd.Series) -> str:
        side = str(row.get("trade_side", "HOLD")).upper()
        ticker = str(row.get("ticker", ""))
        score = _safe_float(row.get("tactical_score"), 0.0)
        rank = int(_safe_float(row.get("tactical_rank"), asset_count))

        if ticker in cash_proxy_tickers:
            if side == "BUY":
                return "cash_proxy_buy_defensive_or_idle_cash"
            if side == "SELL":
                return "cash_proxy_sell_funds_risk_trade"
            return "cash_proxy_hold"

        if side == "BUY":
            if score > 0.0 and rank <= top_third_rank:
                return "tactical_confirms_buy"
            if score > 0.0:
                return "tactical_mildly_supports_buy"
            return "tactical_conflicts_buy"

        if side == "SELL":
            if score < 0.0 or rank >= bottom_third_rank:
                return "tactical_confirms_sell"
            if score <= 0.25:
                return "tactical_mildly_supports_sell"
            return "tactical_conflicts_sell"

        if score > 0.75 and rank <= top_third_rank:
            return "high_tactical_score_but_hold"
        if score < -0.75 and rank >= bottom_third_rank:
            return "weak_tactical_score_and_hold_ok"
        return "neutral_hold"

    merged["alignment"] = merged.apply(classify, axis=1)
    merged["is_cash_proxy"] = merged["ticker"].isin(cash_proxy_tickers)

    output_columns = [
        "ticker",
        "trade_side",
        "alignment",
        "is_cash_proxy",
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
        "excess_expected_return_5d",
        "risk_free_return_5d",
        "reason",
        "current_weight",
        "executable_weight",
        "trade_value_usd",
        "estimated_order_value_usd",
        "skipped_reason",
    ]
    existing_columns = [column for column in output_columns if column in merged.columns]
    alignment_df = merged.loc[:, existing_columns].copy()
    alignment_df.to_csv(alignment_path, index=False)

    actionable = alignment_df[alignment_df["trade_side"].isin(["BUY", "SELL"])].copy()
    conflict_mask = alignment_df["alignment"].astype(str).str.contains("conflicts|high_tactical_score_but_hold", case=False, regex=True)
    conflict_rows = alignment_df.loc[conflict_mask].copy()

    buy_confirmed = int((alignment_df["alignment"] == "tactical_confirms_buy").sum())
    buy_conflict = int((alignment_df["alignment"] == "tactical_conflicts_buy").sum())
    sell_confirmed = int((alignment_df["alignment"] == "tactical_confirms_sell").sum())
    sell_conflict = int((alignment_df["alignment"] == "tactical_conflicts_sell").sum())

    lines = [
        "Tactical Order Alignment Report",
        "",
        "status: report_only_no_order_change",
        f"asset_count: {asset_count}",
        f"actionable_order_count: {len(actionable)}",
        f"buy_confirmed_count: {buy_confirmed}",
        f"buy_conflict_count: {buy_conflict}",
        f"sell_confirmed_count: {sell_confirmed}",
        f"sell_conflict_count: {sell_conflict}",
        f"conflict_or_watch_count: {len(conflict_rows)}",
        "",
        "interpretation:",
        "- BUY should generally have positive tactical_score and top-third tactical_rank.",
        "- SELL should generally have negative tactical_score or weak tactical_rank.",
        "- Cash proxies are marked separately because low volatility can inflate tactical rank.",
        "- prob_up_* fields are uncalibrated directional proxies, not calibrated probabilities.",
        "- This report is diagnostic only and does not alter final orders.",
        "",
        "actionable_orders:",
    ]

    for row in actionable.sort_values(["trade_side", "tactical_rank"], ascending=[True, True]).itertuples(index=False):
        lines.append(
            f"- {row.trade_side} {row.ticker}: alignment={row.alignment}, "
            f"rank={getattr(row, 'tactical_rank', 'n/a')}, "
            f"score={_safe_float(getattr(row, 'tactical_score', 0.0)):.4f}, "
            f"prob_up_5d={_safe_float(getattr(row, 'prob_up_5d', 0.0)):.3f}"
        )

    lines.extend(["", "conflicts_or_watchlist:"])
    if conflict_rows.empty:
        lines.append("- none")
    else:
        for row in conflict_rows.sort_values("tactical_rank", ascending=True).itertuples(index=False):
            lines.append(
                f"- {row.ticker}: side={row.trade_side}, alignment={row.alignment}, "
                f"rank={getattr(row, 'tactical_rank', 'n/a')}, "
                f"score={_safe_float(getattr(row, 'tactical_score', 0.0)):.4f}, "
                f"reason={getattr(row, 'reason', 'n/a')}"
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "tactical_order_alignment": str(alignment_path),
        "tactical_order_alignment_report": str(report_path),
    }


def _forward_return(prices: pd.DataFrame, start_position: int, horizon: int) -> pd.Series:
    if start_position + horizon >= len(prices):
        return pd.Series(dtype=float)
    start_px = prices.iloc[start_position]
    end_px = prices.iloc[start_position + horizon]
    return (end_px / start_px - 1.0).replace([np.inf, -np.inf], np.nan)


def _spearman_rank_ic(x: pd.Series, y: pd.Series) -> float:
    frame = pd.concat([x.astype(float), y.astype(float)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 5:
        return float("nan")
    return float(frame.iloc[:, 0].rank().corr(frame.iloc[:, 1].rank()))


def build_tactical_forecast_calibration(
    *,
    prices: pd.DataFrame,
    params: dict[str, object] | None = None,
    tickers: Sequence[str] | None = None,
    as_of: pd.Timestamp | str | None = None,
    horizons: Sequence[int] = (3, 5, 10),
    min_history: int = 180,
    max_samples: int = 180,
    step: int = 5,
) -> tuple[pd.DataFrame, str]:
    """Walk-forward calibration of tactical scores against realized forward returns.

    Report-only diagnostic. Uses historical dates strictly before/as of the current run date.
    """

    if prices.empty:
        raise ValueError("prices must not be empty.")

    params = dict(params or {})
    end_date = pd.Timestamp(as_of if as_of is not None else prices.index[-1])
    active_tickers = [str(ticker) for ticker in (tickers or prices.columns.tolist()) if str(ticker) in prices.columns]
    history = prices.reindex(columns=active_tickers).loc[:end_date].sort_index().ffill(limit=3).dropna(how="all")

    max_horizon = max(int(h) for h in horizons)
    if len(history) < min_history + max_horizon + 2:
        empty = pd.DataFrame(
            columns=[
                "horizon_days",
                "sample_count",
                "rank_ic_mean",
                "rank_ic_median",
                "top_quintile_forward_return_mean",
                "bottom_quintile_forward_return_mean",
                "top_minus_bottom_mean",
                "top_hit_rate_mean",
                "all_assets_forward_return_mean",
            ]
        )
        report = (
            "Tactical Forecast Calibration Report\n\n"
            f"status: insufficient_history\n"
            f"available_rows: {len(history)}\n"
            f"required_rows: {min_history + max_horizon + 2}\n"
        )
        return empty, report

    returns = compute_returns(history)
    eligible_positions = list(range(min_history, len(history) - max_horizon, max(step, 1)))
    if len(eligible_positions) > max_samples:
        eligible_positions = eligible_positions[-max_samples:]

    rows: list[dict[str, object]] = []

    for pos in eligible_positions:
        decision_date = pd.Timestamp(history.index[pos])
        try:
            tactical = build_multi_horizon_forecast(
                prices=history.iloc[: pos + 1],
                returns=returns.loc[:decision_date],
                date=decision_date,
                params=params,
                tickers=active_tickers,
            )
        except Exception:
            continue

        score = tactical.table.set_index("ticker")["tactical_score"].astype(float)
        rank = tactical.table.set_index("ticker")["tactical_rank"].astype(float)

        for horizon in horizons:
            h = int(horizon)
            fwd = _forward_return(history, pos, h)
            if fwd.empty:
                continue

            frame = pd.DataFrame(
                {
                    "tactical_score": score,
                    "tactical_rank": rank,
                    "forward_return": fwd.reindex(score.index),
                }
            ).replace([np.inf, -np.inf], np.nan).dropna()

            if len(frame) < 10:
                continue

            rank_ic = _spearman_rank_ic(frame["tactical_score"], frame["forward_return"])
            quintile_size = max(int(np.ceil(len(frame) * 0.20)), 1)
            top = frame.sort_values("tactical_score", ascending=False).head(quintile_size)
            bottom = frame.sort_values("tactical_score", ascending=True).head(quintile_size)

            top_mean = float(top["forward_return"].mean())
            bottom_mean = float(bottom["forward_return"].mean())
            all_mean = float(frame["forward_return"].mean())
            rows.append(
                {
                    "decision_date": str(decision_date.date()),
                    "horizon_days": h,
                    "asset_count": int(len(frame)),
                    "rank_ic": rank_ic,
                    "top_quintile_forward_return": top_mean,
                    "bottom_quintile_forward_return": bottom_mean,
                    "top_minus_bottom": top_mean - bottom_mean,
                    "top_hit_rate": float((top["forward_return"] > 0.0).mean()),
                    "bottom_hit_rate": float((bottom["forward_return"] > 0.0).mean()),
                    "all_assets_forward_return_mean": all_mean,
                }
            )

    raw = pd.DataFrame(rows)
    if raw.empty:
        report = (
            "Tactical Forecast Calibration Report\n\n"
            "status: no_valid_walk_forward_samples\n"
        )
        return raw, report

    summary_rows: list[dict[str, object]] = []
    for horizon, group in raw.groupby("horizon_days"):
        summary_rows.append(
            {
                "horizon_days": int(horizon),
                "sample_count": int(len(group)),
                "rank_ic_mean": float(group["rank_ic"].mean()),
                "rank_ic_median": float(group["rank_ic"].median()),
                "top_quintile_forward_return_mean": float(group["top_quintile_forward_return"].mean()),
                "bottom_quintile_forward_return_mean": float(group["bottom_quintile_forward_return"].mean()),
                "top_minus_bottom_mean": float(group["top_minus_bottom"].mean()),
                "top_hit_rate_mean": float(group["top_hit_rate"].mean()),
                "bottom_hit_rate_mean": float(group["bottom_hit_rate"].mean()),
                "all_assets_forward_return_mean": float(group["all_assets_forward_return_mean"].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("horizon_days").reset_index(drop=True)

    lines = [
        "Tactical Forecast Calibration Report",
        "",
        "status: report_only_no_order_change",
        f"as_of: {end_date.date()}",
        f"asset_count: {len(active_tickers)}",
        f"walk_forward_samples: {len(raw)}",
        f"sample_step_days: {step}",
        "",
        "method:",
        "- Recomputes tactical_score on historical decision dates using only data available up to each date.",
        "- Compares tactical_score rank to realized forward returns.",
        "- rank_ic is Spearman rank correlation between tactical_score and forward return.",
        "- top_minus_bottom is top-quintile forward return minus bottom-quintile forward return.",
        "- This report is diagnostic only and does not alter final orders.",
        "",
        "summary_by_horizon:",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.horizon_days}d: samples={row.sample_count}, "
            f"rank_ic_mean={row.rank_ic_mean:.4f}, "
            f"top_mean={row.top_quintile_forward_return_mean:.4f}, "
            f"bottom_mean={row.bottom_quintile_forward_return_mean:.4f}, "
            f"top_minus_bottom={row.top_minus_bottom_mean:.4f}, "
            f"top_hit_rate={row.top_hit_rate_mean:.3f}"
        )

    best = summary.sort_values("top_minus_bottom_mean", ascending=False).head(1)
    if not best.empty:
        best_row = best.iloc[0]
        lines.extend(
            [
                "",
                "current_read:",
                f"- best_horizon_by_top_minus_bottom: {int(best_row['horizon_days'])}d",
                f"- best_top_minus_bottom_mean: {float(best_row['top_minus_bottom_mean']):.4f}",
                f"- best_rank_ic_mean: {float(best_row['rank_ic_mean']):.4f}",
            ]
        )

    return summary, "\n".join(lines) + "\n"


def write_tactical_forecast_calibration_outputs(
    *,
    prices: pd.DataFrame,
    params: dict[str, object] | None,
    tickers: Sequence[str],
    as_of: pd.Timestamp | str,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write walk-forward tactical forecast calibration outputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    calibration_path = output_path / "tactical_forecast_calibration.csv"
    report_path = output_path / "tactical_forecast_calibration_report.txt"

    calibration, report = build_tactical_forecast_calibration(
        prices=prices,
        params=params,
        tickers=tickers,
        as_of=as_of,
    )
    calibration.to_csv(calibration_path, index=False)
    report_path.write_text(report, encoding="utf-8")

    return {
        "tactical_forecast_calibration": str(calibration_path),
        "tactical_forecast_calibration_report": str(report_path),
    }


def build_tactical_component_calibration(
    *,
    prices: pd.DataFrame,
    params: dict[str, object] | None = None,
    tickers: Sequence[str] | None = None,
    as_of: pd.Timestamp | str | None = None,
    horizons: Sequence[int] = (3, 5, 10),
    min_history: int = 180,
    max_samples: int = 180,
    step: int = 5,
) -> tuple[pd.DataFrame, str]:
    """Walk-forward calibration for individual tactical signal components.

    Report-only diagnostic. This identifies which components have historical
    rank/return signal before changing tactical_score weights.
    """

    if prices.empty:
        raise ValueError("prices must not be empty.")

    params = dict(params or {})
    end_date = pd.Timestamp(as_of if as_of is not None else prices.index[-1])
    active_tickers = [str(ticker) for ticker in (tickers or prices.columns.tolist()) if str(ticker) in prices.columns]
    history = prices.reindex(columns=active_tickers).loc[:end_date].sort_index().ffill(limit=3).dropna(how="all")

    component_columns = [
        "tactical_score",
        "momentum_3d",
        "momentum_5d",
        "momentum_10d",
        "momentum_20d",
        "momentum_60d",
        "vol_adjusted_momentum_20d",
        "mean_reversion_score",
        "overextension_score",
        "trend_score",
        "drawdown_60d",
        "relative_strength_rank",
        "expected_return_3d",
        "expected_return_5d",
        "expected_return_10d",
        "expected_return_to_project_end",
        "forecast_confidence",
        "risk_adjusted_forecast",
        "vol_5d",
        "vol_20d",
    ]

    max_horizon = max(int(h) for h in horizons)
    if len(history) < min_history + max_horizon + 2:
        empty = pd.DataFrame(
            columns=[
                "component",
                "horizon_days",
                "sample_count",
                "rank_ic_mean",
                "rank_ic_median",
                "top_minus_bottom_mean",
                "top_hit_rate_mean",
            ]
        )
        report = (
            "Tactical Component Calibration Report\n\n"
            f"status: insufficient_history\n"
            f"available_rows: {len(history)}\n"
            f"required_rows: {min_history + max_horizon + 2}\n"
        )
        return empty, report

    returns = compute_returns(history)
    eligible_positions = list(range(min_history, len(history) - max_horizon, max(step, 1)))
    if len(eligible_positions) > max_samples:
        eligible_positions = eligible_positions[-max_samples:]

    rows: list[dict[str, object]] = []

    for pos in eligible_positions:
        decision_date = pd.Timestamp(history.index[pos])
        try:
            tactical = build_multi_horizon_forecast(
                prices=history.iloc[: pos + 1],
                returns=returns.loc[:decision_date],
                date=decision_date,
                params=params,
                tickers=active_tickers,
            )
        except Exception:
            continue

        table = tactical.table.set_index("ticker")

        for horizon in horizons:
            h = int(horizon)
            fwd = _forward_return(history, pos, h)
            if fwd.empty:
                continue

            for component in component_columns:
                if component not in table.columns:
                    continue

                signal = table[component].astype(float)

                # For rank columns and risk-only columns, higher raw value is not always "better".
                # Calibration remains descriptive: if IC/top-minus-bottom is negative, the component
                # is either harmful as-is or should be inverted/used as a penalty.
                frame = pd.DataFrame(
                    {
                        "signal": signal,
                        "forward_return": fwd.reindex(signal.index),
                    }
                ).replace([np.inf, -np.inf], np.nan).dropna()

                if len(frame) < 10:
                    continue

                rank_ic = _spearman_rank_ic(frame["signal"], frame["forward_return"])
                quintile_size = max(int(np.ceil(len(frame) * 0.20)), 1)
                top = frame.sort_values("signal", ascending=False).head(quintile_size)
                bottom = frame.sort_values("signal", ascending=True).head(quintile_size)

                top_mean = float(top["forward_return"].mean())
                bottom_mean = float(bottom["forward_return"].mean())

                rows.append(
                    {
                        "decision_date": str(decision_date.date()),
                        "component": component,
                        "horizon_days": h,
                        "asset_count": int(len(frame)),
                        "rank_ic": rank_ic,
                        "top_quintile_forward_return": top_mean,
                        "bottom_quintile_forward_return": bottom_mean,
                        "top_minus_bottom": top_mean - bottom_mean,
                        "top_hit_rate": float((top["forward_return"] > 0.0).mean()),
                        "bottom_hit_rate": float((bottom["forward_return"] > 0.0).mean()),
                    }
                )

    raw = pd.DataFrame(rows)
    if raw.empty:
        report = (
            "Tactical Component Calibration Report\n\n"
            "status: no_valid_walk_forward_samples\n"
        )
        return raw, report

    summary_rows: list[dict[str, object]] = []
    for (component, horizon), group in raw.groupby(["component", "horizon_days"]):
        rank_ic_mean = float(group["rank_ic"].mean())
        top_minus_bottom_mean = float(group["top_minus_bottom"].mean())
        summary_rows.append(
            {
                "component": str(component),
                "horizon_days": int(horizon),
                "sample_count": int(len(group)),
                "rank_ic_mean": rank_ic_mean,
                "rank_ic_median": float(group["rank_ic"].median()),
                "top_quintile_forward_return_mean": float(group["top_quintile_forward_return"].mean()),
                "bottom_quintile_forward_return_mean": float(group["bottom_quintile_forward_return"].mean()),
                "top_minus_bottom_mean": top_minus_bottom_mean,
                "top_hit_rate_mean": float(group["top_hit_rate"].mean()),
                "bottom_hit_rate_mean": float(group["bottom_hit_rate"].mean()),
                "signal_direction_suggestion": (
                    "use_positive"
                    if rank_ic_mean > 0.01 and top_minus_bottom_mean > 0.0
                    else (
                        "consider_inverse_or_penalty"
                        if rank_ic_mean < -0.01 and top_minus_bottom_mean < 0.0
                        else "weak_or_mixed"
                    )
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["horizon_days", "top_minus_bottom_mean", "rank_ic_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    lines = [
        "Tactical Component Calibration Report",
        "",
        "status: report_only_no_order_change",
        f"as_of: {end_date.date()}",
        f"asset_count: {len(active_tickers)}",
        f"walk_forward_samples: {len(raw)}",
        f"sample_step_days: {step}",
        "",
        "method:",
        "- Recomputes tactical components on historical decision dates using only data available up to each date.",
        "- Measures each component against realized 3d/5d/10d forward returns.",
        "- Positive rank_ic/top_minus_bottom means high component values historically ranked better assets.",
        "- Negative rank_ic/top_minus_bottom means the component may be harmful as a positive signal or useful as a penalty.",
        "- This report is diagnostic only and does not alter final orders.",
        "",
    ]

    for horizon in sorted(summary["horizon_days"].unique()):
        subset = summary[summary["horizon_days"] == horizon].copy()
        lines.append(f"top_components_{int(horizon)}d:")
        for row in subset.head(8).itertuples(index=False):
            lines.append(
                f"- {row.component}: rank_ic={row.rank_ic_mean:.4f}, "
                f"top_minus_bottom={row.top_minus_bottom_mean:.4f}, "
                f"hit_rate={row.top_hit_rate_mean:.3f}, suggestion={row.signal_direction_suggestion}"
            )
        lines.append("")
        lines.append(f"weakest_components_{int(horizon)}d:")
        for row in subset.tail(8).sort_values("top_minus_bottom_mean", ascending=True).itertuples(index=False):
            lines.append(
                f"- {row.component}: rank_ic={row.rank_ic_mean:.4f}, "
                f"top_minus_bottom={row.top_minus_bottom_mean:.4f}, "
                f"hit_rate={row.top_hit_rate_mean:.3f}, suggestion={row.signal_direction_suggestion}"
            )
        lines.append("")

    return summary, "\n".join(lines).rstrip() + "\n"


def write_tactical_component_calibration_outputs(
    *,
    prices: pd.DataFrame,
    params: dict[str, object] | None,
    tickers: Sequence[str],
    as_of: pd.Timestamp | str,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write component-level tactical signal calibration outputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    calibration_path = output_path / "tactical_component_calibration.csv"
    report_path = output_path / "tactical_component_calibration_report.txt"

    calibration, report = build_tactical_component_calibration(
        prices=prices,
        params=params,
        tickers=tickers,
        as_of=as_of,
    )
    calibration.to_csv(calibration_path, index=False)
    report_path.write_text(report, encoding="utf-8")

    return {
        "tactical_component_calibration": str(calibration_path),
        "tactical_component_calibration_report": str(report_path),
    }


def _build_tactical_score_v2_table(table: pd.DataFrame) -> pd.DataFrame:
    """Add a calibrated-report candidate score based on component diagnostics.

    This is intentionally conservative and report-only. It is not used for
    final allocation or execution.
    """

    out = table.copy()

    # Rank 1 means strongest relative strength, so convert to a higher-is-better score.
    if "relative_strength_rank" in out.columns:
        max_rank = float(out["relative_strength_rank"].max() or 1.0)
        out["relative_strength_score"] = 1.0 - ((out["relative_strength_rank"].astype(float) - 1.0) / max(max_rank - 1.0, 1.0))
    else:
        out["relative_strength_score"] = 0.0

    # A4 found raw vol components had the strongest positive historical top-minus-bottom.
    # Keep this as a participation/risk-on candidate, not as a final objective.
    vol_participation = 0.55 * _zscore(out.get("vol_5d", pd.Series(0.0, index=out.index))) + 0.45 * _zscore(
        out.get("vol_20d", pd.Series(0.0, index=out.index))
    )

    # Avoid blindly chasing the existing v1 score and overconfident forecast proxies,
    # which calibrated negatively in A3/A4.
    anti_overconfidence = -0.35 * _zscore(out.get("forecast_confidence", pd.Series(0.0, index=out.index))) - 0.30 * _zscore(
        out.get("risk_adjusted_forecast", pd.Series(0.0, index=out.index))
    )

    # Momentum was weak/mixed; include only a small relative-strength component.
    relative_strength_component = 0.25 * _zscore(out["relative_strength_score"])

    # Mild reversal: A4 suggested trend/momentum was often not reliable as a positive signal.
    reversal_component = -0.20 * _zscore(out.get("trend_score", pd.Series(0.0, index=out.index))) - 0.15 * _zscore(
        out.get("momentum_20d", pd.Series(0.0, index=out.index))
    )

    # Penalize very severe drawdowns less aggressively than v1; drawdown_60d is negative.
    # This term rewards less damaged charts mildly, but does not dominate.
    drawdown_quality = 0.10 * _zscore(out.get("drawdown_60d", pd.Series(0.0, index=out.index)))

    out["tactical_score_v2_candidate"] = (
        vol_participation
        + anti_overconfidence
        + relative_strength_component
        + reversal_component
        + drawdown_quality
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["tactical_rank_v2_candidate"] = out["tactical_score_v2_candidate"].rank(ascending=False, method="dense").astype(int)
    return out


def build_tactical_score_v2_comparison(
    *,
    prices: pd.DataFrame,
    params: dict[str, object] | None = None,
    tickers: Sequence[str] | None = None,
    as_of: pd.Timestamp | str | None = None,
    horizons: Sequence[int] = (3, 5, 10),
    min_history: int = 180,
    max_samples: int = 180,
    step: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Compare v1 tactical_score against report-only v2 candidate."""

    if prices.empty:
        raise ValueError("prices must not be empty.")

    params = dict(params or {})
    end_date = pd.Timestamp(as_of if as_of is not None else prices.index[-1])
    active_tickers = [str(ticker) for ticker in (tickers or prices.columns.tolist()) if str(ticker) in prices.columns]
    history = prices.reindex(columns=active_tickers).loc[:end_date].sort_index().ffill(limit=3).dropna(how="all")

    max_horizon = max(int(h) for h in horizons)
    returns = compute_returns(history)

    # Current v2 table
    current_result = build_multi_horizon_forecast(
        prices=history,
        returns=returns,
        date=end_date,
        params=params,
        tickers=active_tickers,
    )
    current_table = _build_tactical_score_v2_table(current_result.table)

    if len(history) < min_history + max_horizon + 2:
        comparison = pd.DataFrame()
        report = (
            "Tactical Score v2 Candidate Report\n\n"
            "status: insufficient_history\n"
            f"available_rows: {len(history)}\n"
            f"required_rows: {min_history + max_horizon + 2}\n"
        )
        return current_table, comparison, report

    eligible_positions = list(range(min_history, len(history) - max_horizon, max(step, 1)))
    if len(eligible_positions) > max_samples:
        eligible_positions = eligible_positions[-max_samples:]

    rows: list[dict[str, object]] = []
    for pos in eligible_positions:
        decision_date = pd.Timestamp(history.index[pos])
        try:
            tactical = build_multi_horizon_forecast(
                prices=history.iloc[: pos + 1],
                returns=returns.loc[:decision_date],
                date=decision_date,
                params=params,
                tickers=active_tickers,
            )
            table = _build_tactical_score_v2_table(tactical.table).set_index("ticker")
        except Exception:
            continue

        for horizon in horizons:
            h = int(horizon)
            fwd = _forward_return(history, pos, h)
            if fwd.empty:
                continue

            frame = pd.DataFrame(
                {
                    "v1": table["tactical_score"].astype(float),
                    "v2": table["tactical_score_v2_candidate"].astype(float),
                    "forward_return": fwd.reindex(table.index),
                }
            ).replace([np.inf, -np.inf], np.nan).dropna()

            if len(frame) < 10:
                continue

            quintile_size = max(int(np.ceil(len(frame) * 0.20)), 1)

            for score_name in ["v1", "v2"]:
                top = frame.sort_values(score_name, ascending=False).head(quintile_size)
                bottom = frame.sort_values(score_name, ascending=True).head(quintile_size)
                top_mean = float(top["forward_return"].mean())
                bottom_mean = float(bottom["forward_return"].mean())
                rows.append(
                    {
                        "decision_date": str(decision_date.date()),
                        "score_name": score_name,
                        "horizon_days": h,
                        "rank_ic": _spearman_rank_ic(frame[score_name], frame["forward_return"]),
                        "top_quintile_forward_return": top_mean,
                        "bottom_quintile_forward_return": bottom_mean,
                        "top_minus_bottom": top_mean - bottom_mean,
                        "top_hit_rate": float((top["forward_return"] > 0.0).mean()),
                    }
                )

    raw = pd.DataFrame(rows)
    if raw.empty:
        report = "Tactical Score v2 Candidate Report\n\nstatus: no_valid_walk_forward_samples\n"
        return current_table, raw, report

    summary_rows: list[dict[str, object]] = []
    for (score_name, horizon), group in raw.groupby(["score_name", "horizon_days"]):
        summary_rows.append(
            {
                "score_name": str(score_name),
                "horizon_days": int(horizon),
                "sample_count": int(len(group)),
                "rank_ic_mean": float(group["rank_ic"].mean()),
                "rank_ic_median": float(group["rank_ic"].median()),
                "top_quintile_forward_return_mean": float(group["top_quintile_forward_return"].mean()),
                "bottom_quintile_forward_return_mean": float(group["bottom_quintile_forward_return"].mean()),
                "top_minus_bottom_mean": float(group["top_minus_bottom"].mean()),
                "top_hit_rate_mean": float(group["top_hit_rate"].mean()),
            }
        )

    comparison = pd.DataFrame(summary_rows).sort_values(["horizon_days", "score_name"]).reset_index(drop=True)

    lines = [
        "Tactical Score v2 Candidate Report",
        "",
        "status: report_only_no_order_change",
        f"as_of: {end_date.date()}",
        f"asset_count: {len(active_tickers)}",
        f"walk_forward_rows: {len(raw)}",
        "",
        "method:",
        "- Builds tactical_score_v2_candidate from component calibration findings.",
        "- Compares v1 tactical_score and v2 candidate against realized 3d/5d/10d forward returns.",
        "- v2 is a candidate diagnostic only; it does not alter final orders.",
        "- Positive top_minus_bottom and rank_ic are better.",
        "",
        "score_comparison:",
    ]

    for row in comparison.itertuples(index=False):
        lines.append(
            f"- {row.score_name} {row.horizon_days}d: "
            f"rank_ic_mean={row.rank_ic_mean:.4f}, "
            f"top_minus_bottom={row.top_minus_bottom_mean:.4f}, "
            f"top_hit_rate={row.top_hit_rate_mean:.3f}"
        )

    lines.extend(["", "current_top_v2_assets:"])
    for row in current_table.sort_values("tactical_score_v2_candidate", ascending=False).head(10).itertuples(index=False):
        lines.append(
            f"- {row.ticker}: v2_rank={row.tactical_rank_v2_candidate}, "
            f"v2_score={row.tactical_score_v2_candidate:.4f}, "
            f"v1_rank={row.tactical_rank}, v1_score={row.tactical_score:.4f}"
        )

    return current_table, comparison, "\n".join(lines) + "\n"


def write_tactical_score_v2_outputs(
    *,
    prices: pd.DataFrame,
    params: dict[str, object] | None,
    tickers: Sequence[str],
    as_of: pd.Timestamp | str,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write report-only tactical score v2 candidate outputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_path = output_path / "tactical_score_v2_comparison.csv"
    current_path = output_path / "tactical_score_v2_current.csv"
    report_path = output_path / "tactical_score_v2_report.txt"

    current_table, comparison, report = build_tactical_score_v2_comparison(
        prices=prices,
        params=params,
        tickers=tickers,
        as_of=as_of,
    )

    keep_cols = [
        "ticker",
        "tactical_rank",
        "tactical_score",
        "tactical_rank_v2_candidate",
        "tactical_score_v2_candidate",
        "relative_strength_score",
        "vol_5d",
        "vol_20d",
        "forecast_confidence",
        "risk_adjusted_forecast",
        "trend_score",
        "momentum_20d",
        "drawdown_60d",
        "reason",
    ]
    current_table.loc[:, [c for c in keep_cols if c in current_table.columns]].sort_values(
        "tactical_score_v2_candidate",
        ascending=False,
    ).to_csv(current_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    report_path.write_text(report, encoding="utf-8")

    return {
        "tactical_score_v2_current": str(current_path),
        "tactical_score_v2_comparison": str(comparison_path),
        "tactical_score_v2_report": str(report_path),
    }


TACTICAL_SCORE_V3_WEIGHTS: dict[str, float] = {
    # Robust candidate 223 from optimize_tactical_weights.py, 500 random trials.
    # Selection objective: robust 2%-RF Sharpe across train/test split.
    "excess_expected_return_10d": 0.21066017774082196,
    "relative_strength_score": -0.13537910011730786,
    "momentum_10d": -0.13358578798390355,
    "forecast_confidence": 0.10256507441395955,
    "trend_score": 0.09747045804801821,
    "excess_expected_return_3d": 0.06308841546412342,
    "excess_expected_return_to_project_end": -0.05624748992633874,
    "vol_5d": 0.054734356212967,
    "mean_reversion_score": -0.050291162339511586,
    "momentum_5d": -0.041313276003039884,
    "vol_adjusted_momentum_20d": -0.02883775503555471,
    "excess_expected_return_5d": -0.025826946714453437,
}


def _build_tactical_score_v3_table(table: pd.DataFrame) -> pd.DataFrame:
    """Add robust constant-weight tactical score v3 candidate.

    This score is derived from optimize_tactical_weights.py candidate 223.
    It is report-only and does not alter final production orders.
    """

    out = _build_tactical_score_v2_table(table)

    score = pd.Series(0.0, index=out.index, dtype=float)
    for feature, weight in TACTICAL_SCORE_V3_WEIGHTS.items():
        if feature not in out.columns:
            continue
        score = score + float(weight) * _zscore(out[feature].astype(float))

    out["tactical_score_v3_candidate"] = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["tactical_rank_v3_candidate"] = out["tactical_score_v3_candidate"].rank(
        ascending=False,
        method="dense",
    ).astype(int)
    return out


def write_tactical_score_v3_outputs(
    *,
    tactical_forecast: TacticalForecastResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write report-only tactical score v3 candidate outputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    current_path = output_path / "tactical_score_v3_current.csv"
    report_path = output_path / "tactical_score_v3_report.txt"

    table = _build_tactical_score_v3_table(tactical_forecast.table)

    keep_cols = [
        "ticker",
        "tactical_rank",
        "tactical_score",
        "tactical_rank_v2_candidate",
        "tactical_score_v2_candidate",
        "tactical_rank_v3_candidate",
        "tactical_score_v3_candidate",
        "expected_return_3d",
        "expected_return_5d",
        "expected_return_10d",
        "excess_expected_return_3d",
        "excess_expected_return_5d",
        "excess_expected_return_10d",
        "risk_adjusted_forecast",
        "forecast_confidence",
        "trend_score",
        "momentum_5d",
        "momentum_10d",
        "relative_strength_score",
        "vol_5d",
        "reason",
    ]
    cols = [col for col in keep_cols if col in table.columns]
    table.loc[:, cols].sort_values("tactical_score_v3_candidate", ascending=False).to_csv(current_path, index=False)

    lines = [
        "Tactical Score v3 Candidate Report",
        "",
        "status: report_only_no_order_change",
        "source: optimize_tactical_weights.py candidate 223",
        "selection: robust train/test 2%-RF Sharpe",
        "",
        "method:",
        "- Applies fixed constant feature weights learned from historical tactical weight search.",
        "- Uses cross-sectional z-scores per feature.",
        "- Uses excess-return diagnostics with 2% annual risk-free rate where applicable.",
        "- This report does not alter final Daily Bot orders.",
        "",
        "weights:",
    ]

    for feature, weight in sorted(TACTICAL_SCORE_V3_WEIGHTS.items(), key=lambda item: abs(item[1]), reverse=True):
        lines.append(f"- {feature}: {weight:+.6f}")

    lines.extend(["", "current_top_v3_assets:"])
    for row in table.sort_values("tactical_score_v3_candidate", ascending=False).head(12).itertuples(index=False):
        lines.append(
            f"- {row.ticker}: v3_rank={row.tactical_rank_v3_candidate}, "
            f"v3_score={row.tactical_score_v3_candidate:.4f}, "
            f"v2_rank={row.tactical_rank_v2_candidate}, "
            f"v1_rank={row.tactical_rank}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "tactical_score_v3_current": str(current_path),
        "tactical_score_v3_report": str(report_path),
    }
