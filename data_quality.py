"""Data-quality scoring for the active universe used by main and the daily bot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _clip_score(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def compute_data_quality_report(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    active_tickers: list[str],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Compute per-ticker and global data-quality scores."""

    del params

    warnings: list[str] = []
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    if prices.empty:
        return {
            "report_df": pd.DataFrame(),
            "global_data_quality_score": 0.0,
            "removed_tickers": [],
            "warnings": warnings,
            "errors": ["Price data is empty."],
        }

    for ticker in [str(ticker) for ticker in active_tickers]:
        series = prices[ticker] if ticker in prices.columns else pd.Series(index=prices.index, dtype=float)
        returns_series = returns[ticker] if ticker in returns.columns else pd.Series(dtype=float)
        missing_ratio = float(series.isna().mean()) if len(series) else 1.0
        latest_valid = series.dropna()
        latest_price_available = not latest_valid.empty and float(latest_valid.iloc[-1]) > 0.0
        history_length = int(latest_valid.shape[0])
        stale_price_flag = False
        if history_length >= 5:
            stale_price_flag = bool(latest_valid.tail(5).nunique() <= 1)
        extreme_return_count = int((returns_series.abs() > 0.15).sum()) if not returns_series.empty else 0
        last_valid_date = (
            pd.Timestamp(latest_valid.index[-1]).date().isoformat()
            if latest_price_available
            else ""
        )

        score = 1.0
        score -= min(missing_ratio, 1.0) * 0.60
        score -= 0.15 if stale_price_flag else 0.0
        score -= 0.15 if not latest_price_available else 0.0
        if history_length < 252:
            score -= 0.15
        if history_length < 126:
            score -= 0.15
        score -= min(extreme_return_count / max(len(returns_series), 1), 0.20)
        score = _clip_score(score)

        rows.append(
            {
                "ticker": ticker,
                "missing_ratio": missing_ratio,
                "stale_price_flag": stale_price_flag,
                "latest_price_available": latest_price_available,
                "history_length": history_length,
                "extreme_return_count": extreme_return_count,
                "last_valid_date": last_valid_date,
                "data_quality_score": score,
            }
        )

    report_df = pd.DataFrame(rows).sort_values(["data_quality_score", "ticker"], ascending=[True, True]).reset_index(drop=True)
    global_score = float(report_df["data_quality_score"].mean()) if not report_df.empty else 0.0
    removed_tickers = list(prices.attrs.get("removed_tickers", []))
    if bool(prices.attrs.get("synthetic_data", False)):
        warnings.append("Synthetic fallback prices are active; execution should remain blocked.")
    if global_score < 0.50:
        errors.append("Global data quality score is below 0.50.")
    elif global_score < 0.70:
        warnings.append("Global data quality score is below 0.70; only defensive or no-trade actions are advisable.")
    elif global_score < 0.90:
        warnings.append("Global data quality score is below 0.90; add extra model-uncertainty buffer.")

    report_df["global_data_quality_score"] = global_score
    report_df["synthetic_data"] = bool(prices.attrs.get("synthetic_data", False))
    return {
        "report_df": report_df,
        "global_data_quality_score": global_score,
        "removed_tickers": removed_tickers,
        "warnings": warnings,
        "errors": errors,
    }


def save_data_quality_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Persist the data-quality report to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report_df = report.get("report_df", pd.DataFrame()).copy()
    if report_df.empty:
        report_df = pd.DataFrame(
            [
                {
                    "ticker": "SUMMARY",
                    "missing_ratio": np.nan,
                    "stale_price_flag": False,
                    "latest_price_available": False,
                    "history_length": 0,
                    "extreme_return_count": 0,
                    "last_valid_date": "",
                    "data_quality_score": 0.0,
                    "global_data_quality_score": float(report.get("global_data_quality_score", 0.0)),
                    "synthetic_data": False,
                }
            ]
        )
    report_df.to_csv(path, index=False)
    return path
