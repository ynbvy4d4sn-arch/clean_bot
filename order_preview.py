"""Order preview utilities for hypothetical rebalance instructions.

Research/backtest previews and daily-bot discrete simulator previews are both
"order previews", but they are not interchangeable. The helpers below keep the
context explicit so research artifacts are not confused with final simulator
order sheets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_order_preview(
    w_current: pd.Series,
    w_target: pd.Series,
    latest_prices: pd.Series,
    portfolio_value: float,
    output_path: str | Path | None,
    min_order_value: float = 10.0,
    not_executable: bool = False,
    reason: str = "",
    price_basis: str = "adjusted_close_proxy",
) -> pd.DataFrame:
    """Generate and optionally persist a CSV preview of hypothetical rebalance orders."""

    ordered_tickers = list(dict.fromkeys([*w_target.index.tolist(), *w_current.index.tolist()]))
    tickers = pd.Index([str(ticker) for ticker in ordered_tickers], name="ticker")

    current = w_current.astype(float).copy()
    current.index = pd.Index([str(ticker) for ticker in current.index], name="ticker")
    target = w_target.astype(float).copy()
    target.index = pd.Index([str(ticker) for ticker in target.index], name="ticker")
    prices = latest_prices.astype(float).copy()
    prices.index = pd.Index([str(ticker) for ticker in prices.index], name="ticker")

    preview = pd.DataFrame(index=tickers)
    preview["current_weight"] = current.reindex(tickers).fillna(0.0)
    preview["target_weight"] = target.reindex(tickers).fillna(0.0)
    preview["delta_weight"] = preview["target_weight"] - preview["current_weight"]

    preview["estimated_order_value"] = preview["delta_weight"] * float(portfolio_value)
    aligned_prices = prices.reindex(tickers).fillna(0.0)
    preview["estimated_shares"] = preview.apply(
        lambda row: row["estimated_order_value"] / aligned_prices.loc[row.name]
        if aligned_prices.loc[row.name] > 0.0
        else 0.0,
        axis=1,
    )
    preview["price_basis"] = str(price_basis)
    preview["quote_note"] = "latest adjusted close is a proxy, not an executable quote"

    sides: list[str] = []
    not_executable_flags: list[bool] = []
    reasons: list[str] = []
    for _, row in preview.iterrows():
        delta_weight = float(row["delta_weight"])
        estimated_order_value = float(row["estimated_order_value"])
        row_reason = str(reason or "")
        side = "HOLD"
        too_small_or_no_change = abs(estimated_order_value) < float(min_order_value) or abs(delta_weight) <= 0.001
        if too_small_or_no_change:
            row_reason = row_reason or "too_small_or_no_change"
        elif delta_weight > 0.0:
            side = "BUY"
        else:
            side = "SELL"
        sides.append("HOLD" if too_small_or_no_change else side)
        not_executable_flags.append(bool(not_executable or too_small_or_no_change))
        reasons.append(row_reason)

    preview["side"] = sides
    preview["not_executable"] = not_executable_flags
    preview["reason"] = reasons

    result = preview.reset_index()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)

    return result


def build_order_preview(
    current_weights: pd.Series,
    target_weights: pd.Series,
    last_prices: pd.Series,
    portfolio_value: float,
) -> pd.DataFrame:
    """Compatibility wrapper that returns an in-memory order preview DataFrame."""

    return generate_order_preview(
        w_current=current_weights,
        w_target=target_weights,
        latest_prices=last_prices,
        portfolio_value=portfolio_value,
        output_path=None,
    )


def mark_research_preview(preview_df: pd.DataFrame) -> pd.DataFrame:
    """Mark a preview as research/backtest-only and never executable."""

    preview = preview_df.copy()
    preview["preview_context"] = "research_backtest_preview"
    preview["preview_role"] = "research_only"
    preview["preview_note"] = "Research/backtest preview only. Not final simulator orders."
    preview["executable"] = False
    preview["not_executable"] = True
    preview["not_executable_reason"] = "research_preview_only"
    return preview


def mark_daily_simulator_preview(preview_df: pd.DataFrame) -> pd.DataFrame:
    """Annotate the final daily-bot preview with explicit simulator context."""

    preview = preview_df.copy()
    preview["preview_context"] = "daily_bot_discrete_simulator"
    preview["preview_role"] = "final_discrete_preview"
    preview["preview_note"] = "Daily-bot discrete simulator preview. Use for manual simulator decisions only."

    side = preview.get("side", pd.Series(index=preview.index, dtype=object)).astype(str).str.upper()
    if "action" not in preview.columns:
        preview["action"] = side
    if "estimated_price" not in preview.columns:
        preview["estimated_price"] = pd.to_numeric(
            preview.get("latest_price", pd.Series(0.0, index=preview.index)),
            errors="coerce",
        ).fillna(0.0)
    if "preview_only" not in preview.columns:
        preview["preview_only"] = False
    actionable = side.isin(["BUY", "SELL"])
    not_executable = preview.get("not_executable", pd.Series(False, index=preview.index)).fillna(False).astype(bool)
    execution_block_reason = (
        preview.get("execution_block_reason", pd.Series("", index=preview.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    row_reason = (
        preview.get("reason", pd.Series("", index=preview.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    not_executable_reason = execution_block_reason.where(execution_block_reason.ne(""), row_reason)
    preview["not_executable_reason"] = not_executable_reason
    preview["executable"] = actionable & ~not_executable & not_executable_reason.eq("")
    return preview
