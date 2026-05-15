"""Pre-trade validation for order previews and optional execution layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_ORDER_COLUMNS = {
    "ticker",
    "current_weight",
    "target_weight",
    "delta_weight",
    "side",
    "estimated_order_value",
    "estimated_shares",
}


def validate_weights_for_trading(
    w_current: pd.Series,
    w_target: pd.Series,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Validate that current and target weights are sensible for trading."""

    warnings: list[str] = []
    errors: list[str] = []
    if not isinstance(w_current, pd.Series) or not isinstance(w_target, pd.Series):
        errors.append("w_current and w_target must both be pandas Series.")
        return {"ok": False, "warnings": warnings, "errors": errors}

    current = w_current.astype(float).copy()
    target = w_target.astype(float).copy()
    current.index = pd.Index([str(t) for t in current.index], name="ticker")
    target.index = pd.Index([str(t) for t in target.index], name="ticker")
    current_cash_weight = float(params.get("__current_cash_weight__", 0.0) or 0.0)
    target_cash_weight = float(params.get("__target_cash_weight__", 0.0) or 0.0)
    if set(current.index) != set(target.index):
        errors.append("w_current and w_target must have identical ticker indices.")
    else:
        target = target.reindex(current.index)

    for name, series, cash_weight in (
        ("w_current", current, current_cash_weight),
        ("w_target", target, target_cash_weight),
    ):
        if not np.isfinite(series.to_numpy(dtype=float)).all():
            errors.append(f"{name} contains NaN or infinite values.")
        if (series < -1e-10).any():
            errors.append(f"{name} contains negative weights.")
        if cash_weight < -1e-10:
            errors.append(f"{name} contains negative literal cash weight.")
        if abs(float(series.sum()) + float(cash_weight) - 1.0) > 1e-3:
            errors.append(f"{name} does not sum approximately to 1.0.")

    asset_caps = pd.Series(params.get("asset_max_weights", {}), dtype=float).reindex(target.index).fillna(np.inf)
    if (target > asset_caps + 1e-8).any():
        errors.append("Target weights violate individual asset max-weight limits.")

    group_map = pd.Series(params.get("group_map", {}), dtype=object).reindex(target.index)
    group_limits = {str(k): float(v) for k, v in dict(params.get("group_limits", {})).items()}
    for group, tickers in group_map.groupby(group_map).groups.items():
        if str(group) not in group_limits:
            errors.append(f"Missing group limit for group {group}.")
            continue
        if float(target[list(tickers)].sum()) > float(group_limits[str(group)]) + 1e-8:
            errors.append(f"Target weights violate group limit for {group}.")

    equity_like_groups = set(params.get("equity_like_groups", []))
    defensive_groups = set(params.get("defensive_groups", []))
    equity_like_total = float(target[group_map[group_map.isin(equity_like_groups)].index].sum()) if not group_map.empty else 0.0
    defensive_total = (
        float(target[group_map[group_map.isin(defensive_groups)].index].sum()) + float(target_cash_weight)
        if not group_map.empty
        else float(target_cash_weight)
    )
    max_equity_like = float(params.get("max_equity_like_total", params.get("max_equity_like_total_normal", 1.0)))
    min_defensive = float(params.get("min_defensive_weight", params.get("min_defensive_weight_normal", 0.0)))
    if equity_like_total > max_equity_like + 1e-8:
        errors.append("Target weights violate equity-like aggregate limit.")
    if defensive_total + 1e-8 < min_defensive:
        errors.append("Target weights violate minimum defensive-weight requirement.")
    return {"ok": not errors, "warnings": warnings, "errors": errors}


def validate_prices_for_trading(latest_prices: pd.Series, tickers: list[str]) -> dict[str, Any]:
    """Validate latest prices used for order preview and execution."""

    warnings: list[str] = []
    errors: list[str] = []
    if not isinstance(latest_prices, pd.Series):
        errors.append("latest_prices must be a pandas Series.")
        return {"ok": False, "warnings": warnings, "errors": errors}
    prices = latest_prices.astype(float).copy()
    prices.index = pd.Index([str(t) for t in prices.index], name="ticker")
    for ticker in [str(t) for t in tickers]:
        if ticker not in prices.index:
            errors.append(f"Missing latest price for {ticker}.")
            continue
        value = float(prices.loc[ticker])
        if not np.isfinite(value):
            errors.append(f"Latest price for {ticker} is not finite.")
        elif value <= 0.0:
            errors.append(f"Latest price for {ticker} is non-positive.")
    return {"ok": not errors, "warnings": warnings, "errors": errors}


def validate_order_preview(order_preview_df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Validate structure and basic safety of the order preview."""

    warnings: list[str] = []
    errors: list[str] = []
    if order_preview_df.empty:
        warnings.append("Order preview is empty.")
        return {"ok": True, "warnings": warnings, "errors": errors}

    missing = sorted(REQUIRED_ORDER_COLUMNS - set(order_preview_df.columns))
    if missing:
        errors.append("Order preview is missing required columns: " + ", ".join(missing))
        return {"ok": False, "warnings": warnings, "errors": errors}

    preview = order_preview_df.copy()
    preview["ticker"] = preview["ticker"].astype(str)
    preview["side"] = preview["side"].astype(str).str.upper()
    invalid_sides = sorted(set(preview["side"]) - {"BUY", "SELL", "HOLD"})
    if invalid_sides:
        errors.append("Order preview contains invalid sides: " + ", ".join(invalid_sides))

    numeric_columns = ["current_weight", "target_weight", "delta_weight", "estimated_order_value", "estimated_shares"]
    for column in numeric_columns:
        preview[column] = pd.to_numeric(preview[column], errors="coerce")
        if preview[column].isna().any():
            errors.append(f"Order preview column {column} contains NaN or non-finite values.")

    min_order_value = float(params.get("min_order_value_usd", 10.0))
    inconsistent_buy_sell = preview[
        (preview["side"].isin(["BUY", "SELL"]))
        & (
            (preview["estimated_order_value"].abs() < min_order_value)
            | (preview["estimated_shares"].abs() <= 1e-9)
            | (preview["target_weight"] < -1e-10)
        )
    ]
    if not inconsistent_buy_sell.empty:
        errors.append("Order preview contains BUY/SELL rows that are inconsistent with the minimum actionable order thresholds.")

    blocked_tickers = set(str(t) for t in params.get("blocked_tickers", []))
    if blocked_tickers:
        offending = sorted(set(preview["ticker"]) & blocked_tickers)
        if offending:
            errors.append("Order preview contains blocked/non-tradable tickers: " + ", ".join(offending))

    if (preview["target_weight"] < -1e-10).any():
        errors.append("Order preview implies short exposure.")
    if preview["target_weight"].sum() > 1.05:
        errors.append("Order preview implies leverage above 100%.")
    return {"ok": not errors, "warnings": warnings, "errors": errors}


def validate_cash_and_positions(
    order_preview_df: pd.DataFrame,
    account_summary: dict[str, Any] | None,
    positions: pd.DataFrame | None,
    latest_prices: pd.Series,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Validate local paper cash and position constraints.

    When no broker/paper account is active, this check degrades to WARN/SKIP.
    """

    warnings: list[str] = []
    errors: list[str] = []
    adjusted = order_preview_df.copy()
    blocked_rows: list[pd.Series] = []

    if account_summary is None or positions is None:
        warnings.append("Cash/positions validation skipped because no broker or paper-account state was provided.")
        return {
            "ok": True,
            "warnings": warnings,
            "errors": errors,
            "adjusted_order_preview": adjusted,
            "blocked_orders": pd.DataFrame(columns=adjusted.columns.tolist()),
        }

    cash = float(account_summary.get("cash", 0.0))
    positions_df = positions.copy()
    if positions_df.empty:
        positions_df = pd.DataFrame(columns=["ticker", "shares"])
    positions_map = positions_df.set_index("ticker")["shares"].to_dict() if "ticker" in positions_df.columns else {}

    cost_rate = float(params.get("cost_rate", 0.0))
    cost_columns = [
        "estimated_commission",
        "estimated_spread_cost",
        "estimated_slippage_cost",
        "estimated_market_impact_cost",
        "estimated_total_order_cost",
        "cost_bps_used",
    ]

    def _row_total_cost(frame: pd.DataFrame, idx: object, shares: float, price: float) -> float:
        if "estimated_total_order_cost" in frame.columns:
            value = pd.to_numeric(frame.at[idx, "estimated_total_order_cost"], errors="coerce")
            if pd.notna(value):
                return max(float(value), 0.0)
        return max(shares * price * cost_rate, 0.0)

    def _zero_cost_fields(frame: pd.DataFrame, idx: object) -> None:
        for column in cost_columns:
            if column in frame.columns:
                frame.at[idx, column] = 0.0
        if "cost_model_used" in frame.columns:
            frame.at[idx, "cost_model_used"] = "blocked_or_hold"
        if "live_quote_used" in frame.columns:
            frame.at[idx, "live_quote_used"] = False
    # Process SELLs before BUYs so validation reflects the executable cash path
    # of a rebalance rather than the arbitrary CSV row order.
    ordered_indices = (
        adjusted.assign(_side_order=adjusted["side"].astype(str).str.upper().map({"SELL": 0, "BUY": 1, "HOLD": 2}).fillna(3))
        .sort_values(by=["_side_order", "ticker"])
        .index.tolist()
    )

    for idx in ordered_indices:
        row = adjusted.loc[idx]
        side = str(row["side"]).upper()
        ticker = str(row["ticker"])
        shares = abs(float(row["estimated_shares"]))
        price = float(latest_prices.get(ticker, 0.0))

        if side == "SELL":
            available = float(positions_map.get(ticker, 0.0))
            if shares > available + 1e-8:
                adjusted.at[idx, "side"] = "HOLD"
                adjusted.at[idx, "estimated_shares"] = 0.0
                adjusted.at[idx, "estimated_order_value"] = 0.0
                if "order_shares" in adjusted.columns:
                    adjusted.at[idx, "order_shares"] = 0.0
                if "order_value" in adjusted.columns:
                    adjusted.at[idx, "order_value"] = 0.0
                _zero_cost_fields(adjusted, idx)
                blocked_rows.append(adjusted.loc[idx].copy())
                errors.append(f"SELL order in {ticker} exceeds available shares.")
                continue
            proceeds = shares * price - _row_total_cost(adjusted, idx, shares, price)
            cash += proceeds
            positions_map[ticker] = max(available - shares, 0.0)
        elif side == "BUY":
            total_cost = shares * price + _row_total_cost(adjusted, idx, shares, price)
            if total_cost > cash + 1e-8:
                adjusted.at[idx, "side"] = "HOLD"
                adjusted.at[idx, "estimated_shares"] = 0.0
                adjusted.at[idx, "estimated_order_value"] = 0.0
                if "order_shares" in adjusted.columns:
                    adjusted.at[idx, "order_shares"] = 0.0
                if "order_value" in adjusted.columns:
                    adjusted.at[idx, "order_value"] = 0.0
                _zero_cost_fields(adjusted, idx)
                blocked_rows.append(adjusted.loc[idx].copy())
                errors.append(f"Not enough cash for BUY order in {ticker}.")
                continue
            cash -= total_cost
            positions_map[ticker] = float(positions_map.get(ticker, 0.0)) + shares

    blocked = pd.DataFrame(blocked_rows, columns=adjusted.columns.tolist()) if blocked_rows else pd.DataFrame(columns=adjusted.columns.tolist())
    return {
        "ok": not errors,
        "warnings": warnings,
        "errors": errors,
        "adjusted_order_preview": adjusted,
        "blocked_orders": blocked,
    }


def run_pre_trade_validation(
    *,
    w_current: pd.Series,
    w_target: pd.Series,
    latest_prices: pd.Series,
    order_preview_df: pd.DataFrame,
    params: dict[str, Any],
    account_summary: dict[str, Any] | None = None,
    positions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the full pre-trade validation stack."""

    weight_check = validate_weights_for_trading(w_current=w_current, w_target=w_target, params=params)
    price_check = validate_prices_for_trading(latest_prices=latest_prices, tickers=list(w_target.index))
    order_check = validate_order_preview(order_preview_df=order_preview_df, params=params)
    cash_check = validate_cash_and_positions(
        order_preview_df=order_preview_df,
        account_summary=account_summary,
        positions=positions,
        latest_prices=latest_prices,
        params=params,
    )

    warnings = [*weight_check["warnings"], *price_check["warnings"], *order_check["warnings"], *cash_check["warnings"]]
    errors = [*weight_check["errors"], *price_check["errors"], *order_check["errors"], *cash_check["errors"]]

    validation_rows = []
    for check_name, payload in (
        ("weights", weight_check),
        ("prices", price_check),
        ("order_preview", order_check),
        ("cash_positions", cash_check),
    ):
        status = "PASS" if not payload["errors"] else "FAIL"
        message = "; ".join([*payload["warnings"], *payload["errors"]]) or "ok"
        validation_rows.append({"check": check_name, "status": status, "message": message})

    adjusted_preview = cash_check["adjusted_order_preview"].copy()
    if "not_executable" in adjusted_preview.columns:
        adjusted_preview["not_executable"] = adjusted_preview["not_executable"].fillna(False).astype(bool)
    else:
        adjusted_preview["not_executable"] = False
    if not cash_check["blocked_orders"].empty:
        blocked_index = set(cash_check["blocked_orders"].index.tolist())
        adjusted_preview["not_executable"] = [
            bool(existing) or (idx in blocked_index)
            for idx, existing in zip(adjusted_preview.index, adjusted_preview["not_executable"].tolist(), strict=False)
        ]

    return {
        "ok": not errors,
        "warnings": warnings,
        "errors": errors,
        "adjusted_order_preview": adjusted_preview,
        "blocked_orders": cash_check["blocked_orders"],
        "validation_report": pd.DataFrame(validation_rows),
    }


def save_pre_trade_validation_report(validation_report: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist the pre-trade validation report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validation_report.to_csv(path, index=False)
    return path
