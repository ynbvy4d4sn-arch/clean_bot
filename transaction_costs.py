"""Central modeled transaction-cost estimation for discrete order previews."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


LOW_COST_TICKERS = {
    "SGOV",
    "SHY",
    "IEF",
    "AGG",
    "LQD",
    "TIP",
}
NORMAL_COST_TICKERS = {
    "XLC",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "SPHQ",
    "SPLV",
    "SPMO",
}
COMMODITY_COST_TICKERS = {
    "PDBC",
    "GLD",
    "SLV",
}
INVERSE_COST_TICKERS = {"SH"}
CRYPTO_COST_TICKERS = {"IBIT"}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _valid_live_quote(live_quote: dict[str, Any] | None, use_live_bid_ask: bool) -> tuple[bool, float | None, float | None]:
    if not use_live_bid_ask or not isinstance(live_quote, dict):
        return False, None, None
    bid = _safe_float(live_quote.get("bid"), default=np.nan)
    ask = _safe_float(live_quote.get("ask"), default=np.nan)
    if not np.isfinite(bid) or not np.isfinite(ask) or bid <= 0.0 or ask <= 0.0 or ask < bid:
        return False, None, None
    return True, float(bid), float(ask)


def get_cost_assumption_for_ticker(
    ticker: str,
    asset_metadata: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return modeled cost assumptions for a ticker.

    The system does not use live broker fees by default. Assumptions are grouped
    by asset liquidity and execution complexity.
    """

    cfg = dict(config or {})
    symbol = str(ticker).strip().upper()
    commission = _safe_float(cfg.get("default_commission_per_trade_usd"), 0.0)
    default_spread_bps = _safe_float(cfg.get("default_spread_bps"), 2.0)
    default_slippage_bps = _safe_float(cfg.get("default_slippage_bps"), 3.0)
    default_market_impact_bps = _safe_float(cfg.get("default_market_impact_bps"), 0.0)

    spread_bps = default_spread_bps
    slippage_bps = default_slippage_bps
    market_impact_bps = default_market_impact_bps
    bucket = "default"

    overrides = dict(cfg.get("asset_cost_overrides", {}))
    if symbol in overrides:
        entry = dict(overrides[symbol])
        spread_bps = _safe_float(entry.get("spread_bps"), spread_bps)
        slippage_bps = _safe_float(entry.get("slippage_bps"), slippage_bps)
        market_impact_bps = _safe_float(entry.get("market_impact_bps"), market_impact_bps)
        commission = _safe_float(entry.get("commission_per_trade_usd"), commission)
        bucket = str(entry.get("bucket", "override"))
    elif symbol in LOW_COST_TICKERS:
        spread_bps = 1.0
        slippage_bps = 1.0
        bucket = "low_cost_cash_bond_etf"
    elif symbol in COMMODITY_COST_TICKERS:
        spread_bps = 5.0
        slippage_bps = 8.0
        bucket = "higher_cost_commodity_etf"
    elif symbol in INVERSE_COST_TICKERS:
        spread_bps = 5.0
        slippage_bps = 10.0
        bucket = "higher_cost_inverse_etf"
    elif symbol in CRYPTO_COST_TICKERS:
        spread_bps = 8.0
        slippage_bps = 15.0
        bucket = "higher_cost_crypto_etf"
    elif symbol in NORMAL_COST_TICKERS:
        spread_bps = default_spread_bps
        slippage_bps = default_slippage_bps
        bucket = "normal_liquidity_etf"

    assumption = {
        "ticker": symbol,
        "bucket": bucket,
        "commission_per_trade_usd": max(commission, 0.0),
        "spread_bps": max(spread_bps, 0.0),
        "slippage_bps": max(slippage_bps, 0.0),
        "market_impact_bps": max(market_impact_bps, 0.0),
        "total_cost_bps_ex_commission": max(spread_bps + slippage_bps + market_impact_bps, 0.0),
        "cost_model_used": "modeled_bps_assumptions",
        "live_costs_available": False,
    }
    if asset_metadata:
        assumption["asset_metadata"] = dict(asset_metadata)
    return assumption


def estimate_order_cost(
    ticker: str,
    side: str,
    order_shares: float,
    latest_price: float,
    order_value: float | None = None,
    asset_metadata: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    live_quote: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate modeled cost breakdown for one order."""

    symbol = str(ticker).strip().upper()
    normalized_side = str(side).strip().upper()
    shares = abs(_safe_float(order_shares))
    price = _safe_float(latest_price, default=np.nan)
    if normalized_side not in {"BUY", "SELL", "HOLD"}:
        raise ValueError(f"Unsupported order side for cost estimation: {side}")
    if normalized_side == "HOLD" or shares <= 0.0:
        return {
            "ticker": symbol,
            "side": normalized_side,
            "order_shares": 0.0,
            "latest_price": _safe_float(latest_price),
            "order_value": 0.0,
            "estimated_commission": 0.0,
            "estimated_spread_cost": 0.0,
            "estimated_slippage_cost": 0.0,
            "estimated_market_impact_cost": 0.0,
            "estimated_total_order_cost": 0.0,
            "cost_bps_used": 0.0,
            "cost_model_used": "no_trade",
            "live_bid": np.nan,
            "live_ask": np.nan,
            "live_quote_used": False,
            "live_costs_available": False,
            "assumption_bucket": "no_trade",
        }
    if not np.isfinite(price) or price <= 0.0:
        raise ValueError(f"Invalid latest_price for cost estimation in {symbol}: {latest_price}")

    gross_value = abs(_safe_float(order_value, default=shares * price))
    assumption = get_cost_assumption_for_ticker(symbol, asset_metadata=asset_metadata, config=config)
    use_live_bid_ask = bool(dict(config or {}).get("use_live_bid_ask_if_available", True))
    live_quote_used, live_bid, live_ask = _valid_live_quote(live_quote, use_live_bid_ask)

    estimated_commission = float(assumption["commission_per_trade_usd"])
    if live_quote_used and live_bid is not None and live_ask is not None:
        estimated_spread_cost = shares * ((live_ask - live_bid) / 2.0)
        live_costs_available = True
        cost_model_used = "modeled_slippage_with_live_bid_ask_spread"
    else:
        estimated_spread_cost = gross_value * float(assumption["spread_bps"]) / 10000.0
        live_costs_available = False
        cost_model_used = str(assumption["cost_model_used"])
    estimated_slippage_cost = gross_value * float(assumption["slippage_bps"]) / 10000.0
    estimated_market_impact_cost = gross_value * float(assumption["market_impact_bps"]) / 10000.0
    estimated_total_order_cost = (
        estimated_commission
        + estimated_spread_cost
        + estimated_slippage_cost
        + estimated_market_impact_cost
    )
    cost_bps_used = (estimated_total_order_cost / gross_value) * 10000.0 if gross_value > 0.0 else 0.0

    return {
        "ticker": symbol,
        "side": normalized_side,
        "order_shares": shares,
        "latest_price": price,
        "order_value": gross_value,
        "estimated_commission": float(estimated_commission),
        "estimated_spread_cost": float(estimated_spread_cost),
        "estimated_slippage_cost": float(estimated_slippage_cost),
        "estimated_market_impact_cost": float(estimated_market_impact_cost),
        "estimated_total_order_cost": float(estimated_total_order_cost),
        "cost_bps_used": float(cost_bps_used),
        "cost_model_used": cost_model_used,
        "live_bid": live_bid if live_bid is not None else np.nan,
        "live_ask": live_ask if live_ask is not None else np.nan,
        "live_quote_used": bool(live_quote_used),
        "live_costs_available": bool(live_costs_available),
        "assumption_bucket": str(assumption["bucket"]),
    }


def estimate_order_list_costs(
    order_preview_df: pd.DataFrame,
    latest_prices: pd.Series,
    asset_metadata: dict[str, dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    live_quotes: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Estimate modeled costs for an order-preview DataFrame."""

    preview = order_preview_df.copy()
    if preview.empty:
        return preview, {
            "total_estimated_transaction_cost": 0.0,
            "total_estimated_commission": 0.0,
            "total_estimated_spread_cost": 0.0,
            "total_estimated_slippage_cost": 0.0,
            "total_estimated_market_impact_cost": 0.0,
            "total_order_value": 0.0,
            "total_order_cost_pct_nav": 0.0,
            "weighted_average_cost_bps": 0.0,
            "cost_model_used": "no_orders",
            "live_costs_available": False,
            "cost_assumptions": format_cost_assumptions_summary(config),
            "orders_considered": 0,
            "skipped_small_orders": 0,
            "cash_before_orders": _safe_float(dict(config or {}).get("current_cash"), np.nan),
            "cash_after_orders": _safe_float(dict(config or {}).get("current_cash"), np.nan),
            "cash_buffer_usd": _safe_float(dict(config or {}).get("cash_buffer_usd"), 0.0),
            "no_negative_cash": True,
        }

    cfg = dict(config or {})
    prices = latest_prices.astype(float).copy()
    prices.index = pd.Index([str(ticker).upper() for ticker in prices.index], name="ticker")
    nav = _safe_float(cfg.get("nav", cfg.get("portfolio_nav_usd")), 0.0)
    current_cash = _safe_float(cfg.get("current_cash"), np.nan)
    cash_buffer = _safe_float(cfg.get("cash_buffer_usd"), 0.0)
    min_order_value = _safe_float(cfg.get("min_order_value_usd"), 10.0)
    asset_metadata = asset_metadata or {}
    live_quotes = live_quotes or {}

    for column, default in (
        ("estimated_commission", 0.0),
        ("estimated_spread_cost", 0.0),
        ("estimated_slippage_cost", 0.0),
        ("estimated_market_impact_cost", 0.0),
        ("estimated_total_order_cost", 0.0),
        ("cost_bps_used", 0.0),
        ("cost_model_used", "modeled_bps_assumptions"),
        ("live_bid", np.nan),
        ("live_ask", np.nan),
        ("live_quote_used", False),
        ("assumption_bucket", "default"),
    ):
        if column not in preview.columns:
            preview[column] = default

    total_commission = 0.0
    total_spread = 0.0
    total_slippage = 0.0
    total_market_impact = 0.0
    total_cost = 0.0
    total_order_value = 0.0
    orders_considered = 0
    skipped_small_orders = 0
    live_costs_available = False
    cost_models_used: set[str] = set()
    cash_running = current_cash if np.isfinite(current_cash) else np.nan

    ordered_indices = (
        preview.assign(
            __side_order__=preview.get("side", pd.Series("HOLD", index=preview.index))
            .astype(str)
            .str.upper()
            .map({"SELL": 0, "HOLD": 1, "BUY": 2})
            .fillna(1)
        )
        .sort_values(["__side_order__"], kind="stable")
        .index.tolist()
    )

    for idx in ordered_indices:
        row = preview.loc[idx]
        ticker = str(row.get("ticker", "")).strip().upper()
        side = str(row.get("side", "HOLD")).strip().upper()
        shares = abs(_safe_float(row.get("order_shares", row.get("estimated_shares")), 0.0))
        price = _safe_float(row.get("latest_price"), prices.get(ticker, np.nan))
        signed_order_value = _safe_float(
            row.get("order_value", row.get("estimated_order_value")),
            shares * price if side == "BUY" else -shares * price if side == "SELL" else 0.0,
        )
        gross_order_value = abs(signed_order_value)

        if side in {"BUY", "SELL"} and 0.0 < gross_order_value < min_order_value:
            skipped_small_orders += 1
        breakdown = estimate_order_cost(
            ticker=ticker,
            side=side,
            order_shares=shares,
            latest_price=price,
            order_value=gross_order_value,
            asset_metadata=asset_metadata.get(ticker),
            config=cfg,
            live_quote=live_quotes.get(ticker),
        )
        preview.at[idx, "estimated_commission"] = breakdown["estimated_commission"]
        preview.at[idx, "estimated_spread_cost"] = breakdown["estimated_spread_cost"]
        preview.at[idx, "estimated_slippage_cost"] = breakdown["estimated_slippage_cost"]
        preview.at[idx, "estimated_market_impact_cost"] = breakdown["estimated_market_impact_cost"]
        preview.at[idx, "estimated_total_order_cost"] = breakdown["estimated_total_order_cost"]
        preview.at[idx, "cost_bps_used"] = breakdown["cost_bps_used"]
        preview.at[idx, "cost_model_used"] = breakdown["cost_model_used"]
        preview.at[idx, "live_bid"] = breakdown["live_bid"]
        preview.at[idx, "live_ask"] = breakdown["live_ask"]
        preview.at[idx, "live_quote_used"] = breakdown["live_quote_used"]
        preview.at[idx, "assumption_bucket"] = breakdown["assumption_bucket"]

        if side not in {"BUY", "SELL"} or shares <= 0.0 or gross_order_value <= 0.0:
            continue

        orders_considered += 1
        total_commission += float(breakdown["estimated_commission"])
        total_spread += float(breakdown["estimated_spread_cost"])
        total_slippage += float(breakdown["estimated_slippage_cost"])
        total_market_impact += float(breakdown["estimated_market_impact_cost"])
        total_cost += float(breakdown["estimated_total_order_cost"])
        total_order_value += gross_order_value
        live_costs_available = live_costs_available or bool(breakdown["live_costs_available"])
        cost_models_used.add(str(breakdown["cost_model_used"]))

        if np.isfinite(cash_running):
            if side == "BUY":
                cash_running -= gross_order_value + float(breakdown["estimated_total_order_cost"])
            elif side == "SELL":
                cash_running += gross_order_value - float(breakdown["estimated_total_order_cost"])

    if not np.isfinite(cash_running):
        cash_running = np.nan
    total_cost_pct_nav = (total_cost / nav) if nav > 0.0 else 0.0
    weighted_average_cost_bps = (total_cost / total_order_value) * 10000.0 if total_order_value > 0.0 else 0.0
    cost_model_used = (
        list(cost_models_used)[0]
        if len(cost_models_used) == 1
        else "mixed_modeled_and_live_spread"
        if cost_models_used
        else "no_orders"
    )

    summary = {
        "total_estimated_transaction_cost": float(total_cost),
        "total_estimated_commission": float(total_commission),
        "total_estimated_spread_cost": float(total_spread),
        "total_estimated_slippage_cost": float(total_slippage),
        "total_estimated_market_impact_cost": float(total_market_impact),
        "total_order_value": float(total_order_value),
        "total_order_cost_pct_nav": float(total_cost_pct_nav),
        "weighted_average_cost_bps": float(weighted_average_cost_bps),
        "cost_model_used": str(cost_model_used),
        "live_costs_available": bool(live_costs_available),
        "cost_assumptions": format_cost_assumptions_summary(cfg),
        "orders_considered": int(orders_considered),
        "skipped_small_orders": int(skipped_small_orders),
        "cash_before_orders": float(current_cash) if np.isfinite(current_cash) else np.nan,
        "cash_after_orders": float(cash_running) if np.isfinite(cash_running) else np.nan,
        "cash_buffer_usd": float(cash_buffer),
        "no_negative_cash": bool((not np.isfinite(cash_running)) or (cash_running + 1e-9 >= cash_buffer)),
    }
    return preview, summary


def format_cost_assumptions_summary(config: dict[str, Any] | None = None) -> str:
    """Format the current modeled cost assumptions for reporting."""

    cfg = dict(config or {})
    return (
        "commission_per_trade_usd="
        f"{_safe_float(cfg.get('default_commission_per_trade_usd'), 0.0):.2f}; "
        f"default_turnover_bps={_safe_float(cfg.get('default_bps_per_turnover'), 5.0):.2f}; "
        f"default_spread_bps={_safe_float(cfg.get('default_spread_bps'), 2.0):.2f}; "
        f"default_slippage_bps={_safe_float(cfg.get('default_slippage_bps'), 3.0):.2f}; "
        f"default_market_impact_bps={_safe_float(cfg.get('default_market_impact_bps'), 0.0):.2f}; "
        "live_bid_ask_used_if_available="
        f"{bool(cfg.get('use_live_bid_ask_if_available', True))}"
    )


def build_transaction_cost_review_summary(
    order_cost_summary: dict[str, Any] | None,
    *,
    nav: float | None = None,
    config: dict[str, Any] | None = None,
    trade_edge_summary: dict[str, Any] | None = None,
) -> dict[str, float | str]:
    """Split direct simulator fees from conservative modeled trading frictions."""

    cfg = dict(config or {})
    summary = dict(order_cost_summary or {})
    nav_value = _safe_float(
        nav,
        _safe_float(summary.get("nav"), _safe_float(cfg.get("nav", cfg.get("portfolio_nav_usd")), 0.0)),
    )
    commission_per_trade_usd = max(_safe_float(cfg.get("default_commission_per_trade_usd"), 0.0), 0.0)
    simulator_order_fee_usd = commission_per_trade_usd
    total_simulator_order_fees_usd = max(_safe_float(summary.get("total_estimated_commission"), 0.0), 0.0)
    total_estimated_transaction_cost_usd = max(
        _safe_float(summary.get("total_estimated_transaction_cost"), 0.0),
        0.0,
    )
    modeled_transaction_costs_usd = max(
        total_estimated_transaction_cost_usd - total_simulator_order_fees_usd,
        0.0,
    )
    modeled_transaction_costs_pct_nav = (
        modeled_transaction_costs_usd / nav_value if nav_value > 0.0 else 0.0
    )
    total_transaction_costs_pct_nav = (
        total_estimated_transaction_cost_usd / nav_value if nav_value > 0.0 else 0.0
    )
    trade_now_edge_after_all_costs = _safe_float((trade_edge_summary or {}).get("trade_now_edge"), 0.0)
    simulator_fee_edge_drag = (
        total_simulator_order_fees_usd / nav_value if nav_value > 0.0 else 0.0
    )
    trade_now_edge_after_modeled_costs = trade_now_edge_after_all_costs + simulator_fee_edge_drag

    return {
        "commission_per_trade_usd": float(commission_per_trade_usd),
        "simulator_order_fee_usd": float(simulator_order_fee_usd),
        "total_simulator_order_fees_usd": float(total_simulator_order_fees_usd),
        "modeled_spread_bps": max(_safe_float(cfg.get("default_spread_bps"), 2.0), 0.0),
        "modeled_slippage_bps": max(_safe_float(cfg.get("default_slippage_bps"), 3.0), 0.0),
        "modeled_market_impact_bps": max(_safe_float(cfg.get("default_market_impact_bps"), 0.0), 0.0),
        "modeled_bps_per_turnover": max(_safe_float(cfg.get("default_bps_per_turnover"), 5.0), 0.0),
        "modeled_transaction_costs_usd": float(modeled_transaction_costs_usd),
        "modeled_transaction_costs_pct_nav": float(modeled_transaction_costs_pct_nav),
        "total_transaction_costs_usd": float(total_estimated_transaction_cost_usd),
        "total_transaction_costs_pct_nav": float(total_transaction_costs_pct_nav),
        "trade_now_edge_after_all_costs": float(trade_now_edge_after_all_costs),
        "trade_now_edge_after_modeled_costs": float(trade_now_edge_after_modeled_costs),
        "trade_now_edge_without_direct_simulator_fees": float(trade_now_edge_after_modeled_costs),
        "cost_model_used": str(summary.get("cost_model_used", "unknown")),
    }


def compute_trade_now_edge(
    current_score: float,
    target_score_after_costs: float,
    total_order_cost: float,
    execution_buffer: float,
    model_uncertainty_buffer: float,
    other_penalties: float = 0.0,
) -> dict[str, float]:
    """Compute the final trade-now edge from the executable order list."""

    current_score_value = _safe_float(current_score, 0.0)
    target_score_after_costs_value = _safe_float(target_score_after_costs, 0.0)
    total_order_cost_value = max(_safe_float(total_order_cost, 0.0), 0.0)
    execution_buffer_value = max(_safe_float(execution_buffer, 0.0), 0.0)
    model_uncertainty_buffer_value = max(_safe_float(model_uncertainty_buffer, 0.0), 0.0)
    other_penalties_value = max(_safe_float(other_penalties, 0.0), 0.0)
    trade_now_edge = (
        target_score_after_costs_value
        - current_score_value
        - execution_buffer_value
        - model_uncertainty_buffer_value
        - other_penalties_value
    )
    return {
        "current_score": current_score_value,
        "target_score_after_costs": target_score_after_costs_value,
        "total_order_cost": total_order_cost_value,
        "execution_buffer": execution_buffer_value,
        "model_uncertainty_buffer": model_uncertainty_buffer_value,
        "other_penalties": other_penalties_value,
        "trade_now_edge": trade_now_edge,
    }
