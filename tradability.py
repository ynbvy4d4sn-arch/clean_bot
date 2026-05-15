"""Tradability checks for data availability and execution-mode eligibility."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from asset_universe import ASSET_UNIVERSE, EXPECTED_CASH_TICKER, get_enabled_tickers
from database import save_tradability_status as _save_tradability_status


SHORT_OR_INVERSE_TICKERS = {"SH", "PSQ"}


@dataclass(slots=True)
class TradabilityStatus:
    """Tradability state for one ticker."""

    ticker: str
    has_price_data: bool
    has_latest_price: bool
    is_enabled_in_universe: bool
    is_tradable_in_local_paper: bool
    is_tradable_in_investopedia: bool
    is_short_or_inverse: bool
    allowed_by_policy: bool
    reason: str
    final_allowed: bool


def check_data_tradability(tickers: list[str], prices: pd.DataFrame) -> pd.DataFrame:
    """Check whether tickers have usable price data."""

    enabled = set(get_enabled_tickers())
    rows: list[dict[str, Any]] = []
    for ticker in [str(t) for t in tickers]:
        has_column = ticker in prices.columns
        latest_price = None
        if has_column:
            series = prices[ticker].dropna()
            if not series.empty:
                latest_price = float(series.iloc[-1])
        has_latest_price = latest_price is not None and pd.notna(latest_price) and latest_price > 0.0
        is_enabled = ticker in enabled
        reasons: list[str] = []
        if not has_column:
            reasons.append("missing_price_column")
        if has_column and not has_latest_price:
            reasons.append("missing_or_nonpositive_latest_price")
        rows.append(
            asdict(
                TradabilityStatus(
                    ticker=ticker,
                    has_price_data=bool(has_column),
                    has_latest_price=bool(has_latest_price),
                    is_enabled_in_universe=bool(is_enabled),
                    is_tradable_in_local_paper=bool(has_latest_price),
                    is_tradable_in_investopedia=False,
                    is_short_or_inverse=ticker in SHORT_OR_INVERSE_TICKERS,
                    allowed_by_policy=bool(is_enabled),
                    reason=";".join(reasons) if reasons else "ok",
                    final_allowed=bool(has_column and has_latest_price and is_enabled),
                )
            )
        )
    return pd.DataFrame(rows)


def check_local_paper_tradability(tickers: list[str], prices: pd.DataFrame) -> pd.Series:
    """Check local paper tradability from price availability only."""

    data_df = check_data_tradability(tickers, prices)
    series = data_df.set_index("ticker")["has_latest_price"] & data_df.set_index("ticker")["allowed_by_policy"]
    return series.reindex([str(t) for t in tickers]).fillna(False)


def check_investopedia_tradability(
    tickers: list[str],
    adapter: object | None = None,
    dry_run: bool = True,
) -> pd.DataFrame:
    """Check tradability for the optional Investopedia adapter without forcing login."""

    rows: list[dict[str, Any]] = []
    enabled = bool(getattr(getattr(adapter, "settings", None), "enabled", False)) if adapter is not None else False
    for ticker in [str(t) for t in tickers]:
        reason = "disabled_or_not_checked"
        tradable = False
        if adapter is not None and enabled and not dry_run:
            try:
                preview = pd.DataFrame(
                    [{"ticker": ticker, "side": "HOLD", "estimated_shares": 0.0}]
                )
                adapter.preview_orders(preview)
                tradable = True
                reason = "preview_validated_only"
            except Exception as exc:
                tradable = False
                reason = f"adapter_preview_failed:{exc}"
        rows.append(
            {
                "ticker": ticker,
                "is_tradable_in_investopedia": tradable,
                "investopedia_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_tradability_report(
    tickers: list[str],
    prices: pd.DataFrame,
    *,
    enable_local_paper: bool = False,
    enable_investopedia: bool = False,
    adapter: object | None = None,
    dry_run: bool = True,
) -> pd.DataFrame:
    """Build the combined tradability report used by main and daily bot."""

    base = check_data_tradability(tickers, prices).set_index("ticker")
    local_paper = check_local_paper_tradability(tickers, prices).rename("is_tradable_in_local_paper")
    investopedia = check_investopedia_tradability(tickers, adapter=adapter, dry_run=dry_run).set_index("ticker")
    report = (
        base.join(local_paper, rsuffix="_local_override")
        .join(investopedia, how="left", rsuffix="_investopedia_override")
    )
    report["is_tradable_in_local_paper"] = report["is_tradable_in_local_paper_local_override"].fillna(
        report["is_tradable_in_local_paper"]
    )
    if "is_tradable_in_investopedia_investopedia_override" in report.columns:
        report["is_tradable_in_investopedia"] = report[
            "is_tradable_in_investopedia_investopedia_override"
        ].fillna(report["is_tradable_in_investopedia"])
    report = report.drop(columns=[column for column in report.columns if column.endswith("_local_override")], errors="ignore")
    report = report.drop(
        columns=[column for column in report.columns if column.endswith("_investopedia_override")],
        errors="ignore",
    )

    final_allowed: list[bool] = []
    reasons: list[str] = []
    for ticker, row in report.iterrows():
        reason_parts: list[str] = []
        if str(row.get("reason", "ok")) not in {"", "ok"}:
            reason_parts.append(str(row.get("reason")))
        if not bool(row.get("allowed_by_policy", False)):
            reason_parts.append("disallowed_by_policy")
        if enable_local_paper and not bool(row.get("is_tradable_in_local_paper", False)):
            reason_parts.append("not_tradable_in_local_paper")
        if enable_investopedia and not dry_run and not bool(row.get("is_tradable_in_investopedia", False)):
            reason_parts.append(str(row.get("investopedia_reason", "not_tradable_in_investopedia")))
        allow = bool(row.get("has_price_data", False) and row.get("has_latest_price", False) and row.get("allowed_by_policy", False))
        if enable_local_paper:
            allow = allow and bool(row.get("is_tradable_in_local_paper", False))
        if enable_investopedia and not dry_run:
            allow = allow and bool(row.get("is_tradable_in_investopedia", False))
        final_allowed.append(allow)
        reasons.append(";".join(part for part in reason_parts if part) or "ok")
    report["final_allowed"] = final_allowed
    report["reason"] = reasons
    report = report.reset_index().rename(columns={"index": "ticker"})
    return report[
        [
            "ticker",
            "has_price_data",
            "has_latest_price",
            "is_enabled_in_universe",
            "is_tradable_in_local_paper",
            "is_tradable_in_investopedia",
            "is_short_or_inverse",
            "allowed_by_policy",
            "reason",
            "final_allowed",
        ]
    ]


def select_cash_proxy(active_tickers: list[str], tradability_df: pd.DataFrame) -> str | None:
    """Return the effective cash proxy for the current run."""

    allowed = set(tradability_df.loc[tradability_df["final_allowed"], "ticker"].astype(str).tolist())
    if EXPECTED_CASH_TICKER in active_tickers and EXPECTED_CASH_TICKER in allowed:
        return EXPECTED_CASH_TICKER
    for fallback in ("BIL", "SHY"):
        if fallback in active_tickers and fallback in allowed:
            return fallback
    return None


def apply_tradability_filter(
    tickers: list[str],
    tradability_df: pd.DataFrame,
    min_assets: int = 10,
) -> list[str]:
    """Remove non-tradable assets and ensure a minimum investable set remains."""

    active = [
        str(ticker)
        for ticker in tickers
        if str(ticker) in set(tradability_df.loc[tradability_df["final_allowed"], "ticker"].astype(str))
    ]
    if len(active) < int(min_assets):
        raise ValueError(
            f"Too few tradable assets remain after tradability filtering: {len(active)} available, at least {min_assets} required."
        )
    cash_proxy = select_cash_proxy(active, tradability_df)
    if cash_proxy is None:
        raise ValueError("No tradable cash proxy is available after tradability filtering.")
    return active


def save_tradability_report(tradability_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist the tradability report to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tradability_df.to_csv(path, index=False)
    return path


def save_tradability_to_db(db_path: str | Path, run_id: str, tradability_df: pd.DataFrame) -> None:
    """Persist tradability rows to SQLite."""

    _save_tradability_status(db_path=db_path, run_id=run_id, tradability_df=tradability_df)
