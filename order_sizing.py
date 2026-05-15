"""Deterministic conversion from target weights to trade-sized order rows.

This module is intentionally small and side-effect free. It does not decide
whether orders may execute; it only translates a target allocation into a
transparent whole-share or fractional-share preview.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


ORDER_SIZING_COLUMNS = [
    "asset",
    "current_weight",
    "target_weight",
    "delta_weight",
    "current_value_usd",
    "target_value_usd",
    "trade_value_usd",
    "latest_price",
    "current_shares",
    "target_shares",
    "share_delta",
    "estimated_order_value_usd",
    "trade_side",
    "skipped_reason",
]


def _string_index(series: pd.Series) -> pd.Series:
    normalized = series.copy()
    normalized.index = pd.Index([str(item).strip() for item in normalized.index], name="asset")
    return normalized


def _first_existing_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    lower_to_actual = {str(column).lower(): str(column) for column in frame.columns}
    for name in names:
        actual = lower_to_actual.get(name.lower())
        if actual is not None:
            return actual
    return None


def _positions_by_asset(current_positions: pd.DataFrame) -> pd.DataFrame:
    if current_positions is None:
        raise ValueError("current_positions is required")

    if current_positions.empty:
        return pd.DataFrame(columns=["asset", "current_shares", "current_value_usd"]).set_index("asset")

    frame = current_positions.copy()
    asset_col = _first_existing_column(frame, ["asset", "ticker", "symbol"])
    if asset_col is None:
        frame["_asset"] = [str(item).strip() for item in frame.index]
    else:
        frame["_asset"] = frame[asset_col].astype(str).str.strip()

    frame = frame.loc[frame["_asset"].ne("")]
    shares_col = _first_existing_column(frame, ["current_shares", "shares", "quantity", "qty"])
    value_col = _first_existing_column(
        frame,
        ["current_value_usd", "market_value_usd", "market_value", "value_usd", "value"],
    )

    grouped = pd.DataFrame(index=pd.Index(sorted(frame["_asset"].unique()), name="asset"))
    if shares_col is None:
        grouped["current_shares"] = 0.0
    else:
        shares = pd.to_numeric(frame[shares_col], errors="coerce")
        if shares.isna().any():
            raise ValueError(f"current_positions contains non-numeric shares in column {shares_col}")
        grouped["current_shares"] = shares.groupby(frame["_asset"]).sum().reindex(grouped.index).fillna(0.0)

    if value_col is None:
        grouped["current_value_usd"] = np.nan
    else:
        values = pd.to_numeric(frame[value_col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"current_positions contains non-numeric values in column {value_col}")
        grouped["current_value_usd"] = values.groupby(frame["_asset"]).sum().reindex(grouped.index).fillna(0.0)

    return grouped


def _validate_numeric_scalar(value: float, name: str, *, minimum: float | None = None) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return numeric


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def convert_weights_to_orders(
    current_positions: pd.DataFrame,
    target_weights: pd.Series,
    latest_prices: pd.Series,
    total_portfolio_value_usd: float,
    min_order_value_usd: float,
    fractional_shares: bool,
    cash_buffer_usd: float = 0.0,
) -> pd.DataFrame:
    """Convert target weights into BUY/SELL/HOLD rows with share deltas.

    The function deliberately does not apply execution-window, stale-data, or
    broker checks. Those belong in the execution guard. Missing prices are
    fail-closed at the row level by returning HOLD with ``missing_price``.
    """

    nav = _validate_numeric_scalar(total_portfolio_value_usd, "total_portfolio_value_usd", minimum=0.0)
    if nav <= 0.0:
        raise ValueError("total_portfolio_value_usd must be > 0")
    min_order_value = _validate_numeric_scalar(min_order_value_usd, "min_order_value_usd", minimum=0.0)
    cash_buffer = _validate_numeric_scalar(cash_buffer_usd, "cash_buffer_usd", minimum=0.0)

    targets = _string_index(pd.to_numeric(target_weights, errors="coerce"))
    if targets.isna().any():
        raise ValueError("target_weights contains non-numeric values")
    if (targets < -1e-12).any():
        bad_assets = ", ".join(targets.loc[targets < -1e-12].index.astype(str).tolist())
        raise ValueError(f"target_weights contains negative weights: {bad_assets}")
    targets = targets.clip(lower=0.0)

    prices = _string_index(pd.to_numeric(latest_prices, errors="coerce"))
    positions = _positions_by_asset(current_positions)

    ordered_assets: list[str] = []
    for asset in [*targets.index.astype(str).tolist(), *positions.index.astype(str).tolist()]:
        if asset and asset not in ordered_assets:
            ordered_assets.append(asset)

    rows: list[dict[str, object]] = []
    for asset in ordered_assets:
        target_weight = _as_float(targets.get(asset, 0.0))
        target_value_usd = target_weight * nav

        position = positions.loc[asset] if asset in positions.index else None
        current_shares = _as_float(position["current_shares"]) if position is not None else 0.0
        if current_shares < -1e-12:
            raise ValueError(f"current_positions contains negative shares for {asset}")
        current_shares = max(current_shares, 0.0)

        if not fractional_shares and abs(current_shares - round(current_shares)) > 1e-9:
            raise ValueError(
                f"fractional current shares for {asset} are incompatible with fractional_shares=False"
            )
        if not fractional_shares:
            current_shares = float(round(current_shares))

        price = _as_float(prices.get(asset, np.nan), default=np.nan)
        price_is_valid = math.isfinite(price) and price > 0.0

        fallback_current_value = (
            _as_float(position["current_value_usd"], default=np.nan) if position is not None else np.nan
        )
        if price_is_valid:
            current_value_usd = current_shares * price
            raw_target_shares = target_value_usd / price
            if fractional_shares:
                target_shares = max(raw_target_shares, 0.0)
            else:
                target_shares = float(math.floor(max(raw_target_shares, 0.0) + 1e-12))
            share_delta = target_shares - current_shares
            estimated_order_value_usd = abs(share_delta * price)
            latest_price = price
            skipped_reason = ""
        else:
            current_value_usd = fallback_current_value if math.isfinite(fallback_current_value) else 0.0
            raw_target_shares = 0.0
            target_shares = current_shares
            share_delta = 0.0
            estimated_order_value_usd = 0.0
            latest_price = np.nan
            skipped_reason = "missing_price"

        current_weight = current_value_usd / nav
        delta_weight = target_weight - current_weight
        trade_value_usd = target_value_usd - current_value_usd

        trade_side = "HOLD"
        if skipped_reason == "":
            if abs(share_delta) <= 1e-12:
                skipped_reason = "zero_share_delta_after_rounding"
            elif abs(trade_value_usd) < min_order_value:
                skipped_reason = "below_min_order_value"
            elif share_delta > 0.0:
                trade_side = "BUY"
            elif share_delta < 0.0:
                trade_side = "SELL"
            if trade_side == "HOLD":
                target_shares = current_shares
                share_delta = 0.0
                estimated_order_value_usd = 0.0

        rows.append(
            {
                "asset": asset,
                "current_weight": float(current_weight),
                "target_weight": float(target_weight),
                "delta_weight": float(delta_weight),
                "current_value_usd": float(current_value_usd),
                "target_value_usd": float(target_value_usd),
                "trade_value_usd": float(trade_value_usd),
                "latest_price": float(latest_price) if math.isfinite(latest_price) else np.nan,
                "current_shares": float(current_shares),
                "target_shares": float(target_shares),
                "share_delta": float(share_delta),
                "estimated_order_value_usd": float(estimated_order_value_usd),
                "trade_side": trade_side,
                "skipped_reason": skipped_reason,
            }
        )

    result = pd.DataFrame(rows, columns=ORDER_SIZING_COLUMNS)
    valid_price_mask = pd.to_numeric(result["latest_price"], errors="coerce").gt(0.0)
    rounded_target_value = (
        pd.to_numeric(result.loc[valid_price_mask, "target_shares"], errors="coerce")
        * pd.to_numeric(result.loc[valid_price_mask, "latest_price"], errors="coerce")
    ).sum()
    raw_target_value = pd.to_numeric(result.loc[valid_price_mask, "target_value_usd"], errors="coerce").sum()
    result.attrs["fractional_shares"] = bool(fractional_shares)
    result.attrs["cash_buffer_usd"] = cash_buffer
    result.attrs["rounding_cash_drift_usd"] = float(raw_target_value - rounded_target_value)
    result.attrs["estimated_order_notional_usd"] = float(
        pd.to_numeric(result["estimated_order_value_usd"], errors="coerce").fillna(0.0).sum()
    )
    return result
