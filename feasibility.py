"""Static feasibility checks for the constrained portfolio problem."""

from __future__ import annotations

from typing import Any

import pandas as pd

from optimizer import build_feasible_initial_weights


def check_portfolio_feasibility(active_tickers: list[str], params: dict[str, Any]) -> dict[str, Any]:
    """Check whether the static allocation constraints appear feasible."""

    warnings: list[str] = []
    errors: list[str] = []
    tickers = [str(ticker) for ticker in active_tickers]
    if len(tickers) < 10:
        errors.append("Fewer than 10 active tradable assets remain.")

    asset_caps = pd.Series(params.get("asset_max_weights", {}), dtype=float).reindex(tickers).fillna(0.0)
    group_map = pd.Series(params.get("group_map", {}), dtype=object).reindex(tickers)
    group_limits = {str(key): float(value) for key, value in dict(params.get("group_limits", {})).items()}
    max_possible_by_group: dict[str, float] = {}
    for group in sorted({str(group_map.get(ticker, "")) for ticker in tickers}):
        members = [ticker for ticker in tickers if str(group_map.get(ticker, "")) == group]
        max_possible_by_group[group] = min(float(asset_caps.reindex(members).sum()), float(group_limits.get(group, 0.0)))

    total_asset_capacity = float(asset_caps.sum())
    total_group_capacity = float(sum(max_possible_by_group.values()))
    if total_asset_capacity + 1e-12 < 1.0:
        errors.append("Individual asset max weights cannot fund a fully invested portfolio.")
    if total_group_capacity + 1e-12 < 1.0:
        errors.append("Group limits cannot fund a fully invested portfolio.")

    defensive_groups = set(str(group) for group in params.get("defensive_groups", []))
    defensive_capacity = float(
        sum(capacity for group, capacity in max_possible_by_group.items() if group in defensive_groups)
    )
    min_defensive = float(params.get("min_defensive_weight", params.get("min_defensive_weight_normal", 0.0)))
    if defensive_capacity + 1e-12 < min_defensive:
        errors.append("Defensive assets cannot satisfy the minimum defensive weight.")

    cash_ticker = params.get("cash_ticker")
    min_cash_weight = float(params.get("min_cash_weight", 0.0) or 0.0)
    if cash_ticker is not None:
        cash_capacity = float(asset_caps.get(str(cash_ticker), 0.0))
        if cash_capacity + 1e-12 < min_cash_weight:
            errors.append("Cash asset cannot satisfy the configured minimum cash weight.")

    max_equity_like_total = float(
        params.get("max_equity_like_total", params.get("max_equity_like_total_normal", 1.0))
    )
    if max_equity_like_total + min_defensive > 1.000001:
        warnings.append("Equity-like maximum plus defensive minimum leave very little free capital.")

    try:
        build_feasible_initial_weights(tickers=tickers, params=params)
    except Exception as exc:
        errors.append(f"Static feasibility check failed in optimizer helper: {exc}")

    return {
        "feasible": not errors,
        "warnings": warnings,
        "errors": errors,
        "max_possible_by_group": max_possible_by_group,
        "min_required_by_group": {
            "defensive_groups": min_defensive,
            "cash": min_cash_weight,
        },
    }
