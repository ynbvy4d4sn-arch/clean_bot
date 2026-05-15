"""Discrete whole-share portfolio construction and re-scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from candidate_factory import CandidatePortfolio
from pre_trade_validation import validate_weights_for_trading
from robust_scorer import evaluate_candidate
from scenario_model import ScenarioSet
from transaction_costs import estimate_order_list_costs


@dataclass(slots=True)
class CurrentPortfolioState:
    """Current portfolio state used for HOLD semantics and order previews."""

    source: str
    nav: float
    current_cash: float
    actual_cash_weight: float
    current_shares: pd.Series
    current_values: pd.Series
    current_weights_actual: pd.Series
    current_weights_proxy: pd.Series
    hold_proxy_ticker: str | None
    schema_description: str = ""
    cash_input_method: str = ""
    allow_fractional_shares: bool = False
    parser_warnings: list[str] = field(default_factory=list)
    parser_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DiscreteCandidate:
    """One discrete whole-share portfolio candidate."""

    name: str
    shares: pd.Series
    values: pd.Series
    weights_actual: pd.Series
    weights_proxy: pd.Series
    cash_left: float
    cash_weight: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize(weights: pd.Series) -> pd.Series:
    cleaned = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(cleaned.sum())
    if total <= 0.0:
        return cleaned
    return cleaned / total


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _prepare_series(index: pd.Index, values: pd.Series | dict[str, float] | None = None) -> pd.Series:
    if values is None:
        series = pd.Series(dtype=float)
    else:
        series = pd.Series(values, dtype=float)
    series.index = pd.Index([str(t) for t in series.index], name="ticker")
    return series.reindex(index).fillna(0.0).astype(float)


def validate_portfolio_constraints(
    *,
    weights_actual: pd.Series,
    cash_weight: float,
    params: dict[str, Any],
    index: pd.Index | None = None,
    label: str = "Portfolio",
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Validate actual invested weights plus literal cash against hard limits."""

    resolved_index = index if index is not None else pd.Index([str(t) for t in weights_actual.index], name="ticker")
    actual = _prepare_series(resolved_index, weights_actual)
    cash = float(cash_weight)
    errors: list[str] = []
    warnings: list[str] = []
    asset_violations: list[dict[str, Any]] = []
    group_violations: list[dict[str, Any]] = []

    if not np.isfinite(actual.to_numpy(dtype=float)).all():
        errors.append(f"{label} contains NaN or infinite weights.")
    if not np.isfinite(cash):
        errors.append(f"{label} contains NaN or infinite cash weight.")
    if (actual < -tolerance).any():
        errors.append(f"{label} contains negative weights.")
    if cash < -tolerance:
        errors.append(f"{label} contains negative cash.")

    total_with_cash = float(actual.sum()) + max(cash, 0.0)
    if abs(total_with_cash - 1.0) > 1e-3:
        errors.append(f"{label} plus literal cash does not sum approximately to 1.0.")

    asset_caps = pd.Series(params.get("asset_max_weights", {}), dtype=float)
    asset_caps.index = pd.Index([str(t) for t in asset_caps.index], name="ticker")
    asset_caps = asset_caps.reindex(actual.index).fillna(np.inf)
    for ticker, weight in actual.items():
        limit = float(asset_caps.get(ticker, np.inf))
        if np.isfinite(limit) and float(weight) > limit + tolerance:
            excess = float(weight) - limit
            asset_violations.append(
                {
                    "ticker": str(ticker),
                    "actual_weight": float(weight),
                    "limit": limit,
                    "excess": excess,
                }
            )
    if asset_violations:
        errors.append(f"{label} violates individual asset max-weight limits.")

    group_map = pd.Series(params.get("group_map", {}), dtype=object)
    group_map.index = pd.Index([str(t) for t in group_map.index], name="ticker")
    group_map = group_map.reindex(actual.index)
    group_limits = {str(k): float(v) for k, v in dict(params.get("group_limits", {})).items()}
    for group, tickers in group_map.groupby(group_map).groups.items():
        group_name = str(group)
        if group_name not in group_limits:
            errors.append(f"Missing group limit for group {group_name}.")
            continue
        group_weight = float(actual[list(tickers)].sum())
        limit = float(group_limits[group_name])
        if group_weight > limit + tolerance:
            excess = group_weight - limit
            group_violations.append(
                {
                    "group": group_name,
                    "actual_weight": group_weight,
                    "limit": limit,
                    "excess": excess,
                }
            )
    if group_violations:
        errors.append(f"{label} violates configured group limits.")

    equity_like_groups = set(params.get("equity_like_groups", []))
    defensive_groups = set(params.get("defensive_groups", []))
    equity_like_total = float(actual[group_map[group_map.isin(equity_like_groups)].index].sum()) if not group_map.empty else 0.0
    defensive_total = (
        float(actual[group_map[group_map.isin(defensive_groups)].index].sum()) + max(cash, 0.0)
        if not group_map.empty
        else max(cash, 0.0)
    )
    max_equity_like = float(params.get("max_equity_like_total", params.get("max_equity_like_total_normal", 1.0)))
    min_defensive = float(params.get("min_defensive_weight", params.get("min_defensive_weight_normal", 0.0)))
    if equity_like_total > max_equity_like + tolerance:
        errors.append(f"{label} violates equity-like aggregate limit.")
        group_violations.append(
            {
                "group": "__equity_like_total__",
                "actual_weight": equity_like_total,
                "limit": max_equity_like,
                "excess": equity_like_total - max_equity_like,
            }
        )
    if defensive_total + tolerance < min_defensive:
        errors.append(f"{label} violates minimum defensive-weight requirement.")
        group_violations.append(
            {
                "group": "__minimum_defensive__",
                "actual_weight": defensive_total,
                "limit": min_defensive,
                "excess": min_defensive - defensive_total,
            }
        )

    max_gross_exposure = float(params.get("max_gross_exposure", 1.0))
    if float(actual.sum()) > max_gross_exposure + tolerance:
        errors.append(f"{label} implies gross exposure above the configured maximum.")

    return {
        "ok": not errors,
        "warnings": warnings,
        "errors": errors,
        "asset_limit_violations": asset_violations,
        "group_limit_violations": group_violations,
    }


def load_current_portfolio_state(
    *,
    params: dict[str, Any],
    active_tickers: list[str],
    latest_prices: pd.Series,
    cash_proxy_ticker: str | None,
    nav: float | None = None,
) -> CurrentPortfolioState:
    """Load the actual current portfolio from CSV or fall back to all-cash."""

    index = pd.Index([str(t) for t in active_tickers], name="ticker")
    prices = latest_prices.reindex(index).astype(float)
    nav_value = float(nav or params.get("portfolio_nav_usd") or params.get("default_portfolio_value", 10000.0))
    path = Path(params.get("current_portfolio_path", "data/current_portfolio.csv"))
    default_mode = str(params.get("default_current_portfolio", "cash")).lower()

    shares = pd.Series(0.0, index=index, dtype=float)
    values = pd.Series(0.0, index=index, dtype=float)
    current_cash = nav_value
    notes: list[str] = []
    parser_warnings: list[str] = []
    parser_errors: list[str] = []
    source = "default_cash"
    schema_description = "default_cash_only"
    cash_input_method = "default_all_cash"
    allow_fractional_shares = bool(params.get("allow_fractional_shares", False))

    has_explicit_cash_value = False
    has_share_based_positions = False
    if str(params.get("current_portfolio_source", "csv")).lower() == "csv" and path.exists():
        df = pd.read_csv(path)
        source = "csv"
        explicit_cash_value: float | None = None
        explicit_cash_weight: float | None = None
        for row in df.to_dict(orient="records"):
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            row_shares = _float_or_none(row.get("shares"))
            row_weight = _float_or_none(row.get("current_weight"))
            row_cash_value = _float_or_none(row.get("cash_value"))
            row_cash_usd = _float_or_none(row.get("cash_usd"))

            if ticker == "CASH":
                if row_cash_usd is not None:
                    explicit_cash_value = row_cash_usd
                    has_explicit_cash_value = True
                    cash_input_method = "CASH row with cash_usd column"
                elif row_cash_value is not None:
                    explicit_cash_value = row_cash_value
                    has_explicit_cash_value = True
                    cash_input_method = "CASH row with cash_value column"
                elif row_weight is not None:
                    explicit_cash_weight = row_weight
                    cash_input_method = "CASH row with current_weight column"
                continue

            if ticker not in shares.index:
                parser_warnings.append(
                    f"Unknown or inactive current-portfolio ticker {ticker}; row ignored."
                )
                continue

            price = float(prices.get(ticker, 0.0))
            if row_shares is not None:
                if row_shares < 0.0:
                    parser_errors.append(
                        f"Negative shares for {ticker} are not allowed; row ignored."
                    )
                    continue
                if abs(row_shares - round(row_shares)) > 1e-9:
                    if allow_fractional_shares:
                        parser_warnings.append(
                            f"Fractional shares for {ticker} accepted because allow_fractional_shares=true: {row_shares}."
                        )
                    else:
                        parser_errors.append(
                            f"Fractional shares for {ticker} are not allowed by configuration; row ignored."
                        )
                        continue
                shares.loc[ticker] = row_shares
                values.loc[ticker] = shares.loc[ticker] * max(price, 0.0)
                has_share_based_positions = True
            elif row_weight is not None and price > 0.0:
                if row_weight < 0.0:
                    parser_errors.append(
                        f"Negative current_weight for {ticker} is not allowed; row ignored."
                    )
                    continue
                implied_value = max(row_weight, 0.0) * nav_value
                shares.loc[ticker] = implied_value / price
                values.loc[ticker] = implied_value

        if explicit_cash_value is not None:
            current_cash = max(explicit_cash_value, 0.0)
        elif explicit_cash_weight is not None:
            current_cash = max(explicit_cash_weight, 0.0) * nav_value
        else:
            if has_share_based_positions:
                current_cash = 0.0
                cash_input_method = "missing explicit cash -> assumed cash_usd=0.00"
                parser_warnings.append(
                    "No explicit cash found in current_portfolio.csv; assumed cash_usd=0.00."
                )
            else:
                current_cash = max(nav_value - float(values.sum()), 0.0)
                cash_input_method = "implicit residual cash from current_weight rows"

        if has_share_based_positions and cash_input_method == "CASH row with cash_value column":
            schema_description = "ticker,shares rows plus CASH row with cash_value"
        elif has_share_based_positions and cash_input_method == "CASH row with cash_usd column":
            schema_description = "ticker,shares rows plus CASH row with cash_usd"
        elif has_share_based_positions and explicit_cash_weight is not None:
            schema_description = "ticker,shares rows plus CASH row with current_weight"
        elif has_share_based_positions:
            schema_description = "ticker,shares rows with missing explicit cash"
        elif explicit_cash_weight is not None:
            schema_description = "ticker,current_weight rows plus CASH row with current_weight"
        elif cash_input_method == "CASH row with cash_usd column":
            schema_description = "ticker,current_weight rows plus CASH row with cash_usd"
        elif explicit_cash_value is not None:
            schema_description = "ticker,current_weight rows plus CASH row with cash_value"
        else:
            schema_description = "ticker,current_weight rows with implicit residual cash"
    else:
        if default_mode != "cash":
            notes.append("No current_portfolio.csv found; falling back to all-cash default because only cash mode is supported safely.")
        source = "default_cash"
        current_cash = nav_value

    total_assets_value = float(values.sum())
    if has_share_based_positions and (
        has_explicit_cash_value or cash_input_method == "missing explicit cash -> assumed cash_usd=0.00"
    ):
        implied_nav = total_assets_value + current_cash
        if implied_nav > 0.0:
            nav_value = implied_nav
            notes.append("NAV was derived from current shares plus cash.")
    elif has_explicit_cash_value and total_assets_value > 0.0:
        implied_nav = total_assets_value + current_cash
        if implied_nav > 0.0:
            nav_value = implied_nav
            notes.append("NAV was derived from explicit cash plus mark-to-market position values.")
    if total_assets_value + current_cash <= 0.0:
        current_cash = nav_value
        notes.append("Current portfolio values were empty; reset to all-cash default.")

    current_weights_actual = (values / nav_value).fillna(0.0)
    actual_cash_weight = max(current_cash / nav_value, 0.0) if nav_value > 0.0 else 0.0
    current_weights_proxy = current_weights_actual.copy()
    if cash_proxy_ticker is not None and cash_proxy_ticker in current_weights_proxy.index:
        current_weights_proxy.loc[cash_proxy_ticker] += actual_cash_weight
    current_weights_proxy = current_weights_proxy.clip(lower=0.0)
    proxy_total = float(current_weights_proxy.sum())
    if proxy_total > 0.0:
        current_weights_proxy = current_weights_proxy / proxy_total

    return CurrentPortfolioState(
        source=source,
        nav=nav_value,
        current_cash=current_cash,
        actual_cash_weight=actual_cash_weight,
        current_shares=shares,
        current_values=values,
        current_weights_actual=current_weights_actual,
        current_weights_proxy=current_weights_proxy,
        hold_proxy_ticker=cash_proxy_ticker,
        schema_description=schema_description,
        cash_input_method=cash_input_method,
        allow_fractional_shares=allow_fractional_shares,
        parser_warnings=parser_warnings,
        parser_errors=parser_errors,
        notes=notes,
    )


def write_current_portfolio_report(
    state: CurrentPortfolioState,
    output_path: str | Path,
    *,
    latest_prices: pd.Series | None = None,
    latest_price_date: object | None = None,
    price_basis: str | None = None,
    data_source: str | None = None,
    data_freshness_ok: bool | None = None,
    constraint_validation: dict[str, Any] | None = None,
) -> Path:
    """Write a compact report about the current portfolio source and assumptions."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    non_zero_weights = state.current_weights_actual[state.current_weights_actual.abs() > 1e-8]
    non_zero_positions = state.current_shares[state.current_shares.abs() > 1e-8]
    positions_count = int(non_zero_positions.shape[0])
    invested_market_value = float(state.current_values.sum())
    current_weights_sum_without_cash = float(state.current_weights_actual.sum())
    current_weights_sum_including_cash = current_weights_sum_without_cash + float(state.actual_cash_weight)
    current_portfolio_100pct_cash = positions_count == 0 and float(state.current_cash) > 0.0
    report_prices = (
        latest_prices.reindex(state.current_shares.index).astype(float)
        if latest_prices is not None
        else pd.Series(np.nan, index=state.current_shares.index, dtype=float)
    )
    latest_price_date_text = str(latest_price_date or "n/a")
    price_basis_text = str(price_basis or "n/a")
    data_source_text = str(data_source or state.source or "unknown")
    stale_price_warning = bool(data_freshness_ok is not None and not data_freshness_ok)
    missing_price_tickers = [
        str(ticker)
        for ticker, shares in non_zero_positions.items()
        if _float_or_none(report_prices.get(ticker)) is None or float(report_prices.get(ticker, 0.0)) <= 0.0
    ]
    negative_market_value_tickers = [
        str(ticker)
        for ticker, value in state.current_values.items()
        if abs(float(state.current_shares.get(ticker, 0.0))) > 1e-9 and float(value) < -1e-9
    ]
    negative_share_tickers = [
        str(ticker)
        for ticker, shares in state.current_shares.items()
        if float(shares) < -1e-9
    ]
    cash_is_nan = not np.isfinite(float(state.current_cash))
    nav_positive = float(state.nav) > 0.0
    weights_close_to_one = abs(current_weights_sum_including_cash - 1.0) <= 0.01 if nav_positive else False
    constraint_validation = constraint_validation or {
        "ok": True,
        "errors": [],
        "asset_limit_violations": [],
        "group_limit_violations": [],
    }
    current_portfolio_constraint_valid = bool(constraint_validation.get("ok", True))
    current_portfolio_constraint_errors = list(constraint_validation.get("errors", []))
    current_portfolio_asset_limit_violations = list(constraint_validation.get("asset_limit_violations", []))
    current_portfolio_group_limit_violations = list(constraint_validation.get("group_limit_violations", []))
    lines = [
        f"current_portfolio_source: {state.source}",
        f"recognized_schema: {state.schema_description or 'unknown'}",
        f"cash_input_method: {state.cash_input_method or 'unknown'}",
        f"allow_fractional_shares: {state.allow_fractional_shares}",
        f"default_current_portfolio_used: {state.source != 'csv'}",
        f"current_portfolio_constraint_valid: {current_portfolio_constraint_valid}",
        f"current_portfolio_constraint_violation: {not current_portfolio_constraint_valid}",
        "current_portfolio_constraint_errors: "
        + (" | ".join(map(str, current_portfolio_constraint_errors)) if current_portfolio_constraint_errors else "none"),
        f"current_portfolio_asset_limit_violations: {len(current_portfolio_asset_limit_violations)}",
        f"current_portfolio_group_limit_violations: {len(current_portfolio_group_limit_violations)}",
        f"nav_usd: {state.nav:.2f}",
        f"cash_usd: {state.current_cash:.2f}",
        f"actual_cash_value: {state.current_cash:.2f}",
        f"invested_market_value_usd: {invested_market_value:.2f}",
        f"actual_cash_weight: {state.actual_cash_weight:.6f}",
        f"positions_count: {positions_count}",
        f"current_portfolio_100pct_cash: {current_portfolio_100pct_cash}",
        f"current_weights_sum_without_cash: {current_weights_sum_without_cash:.6f}",
        f"current_weights_sum_including_cash: {current_weights_sum_including_cash:.6f}",
        f"hold_proxy_ticker: {state.hold_proxy_ticker or 'none'}",
        f"parser_warning_count: {len(state.parser_warnings)}",
        f"parser_error_count: {len(state.parser_errors)}",
        "current_positions:",
    ]
    if non_zero_positions.empty:
        lines.append("  none")
    else:
        lines.extend(
            f"  {ticker}: shares={float(shares):.6f}, latest_price={float(report_prices.get(ticker, 0.0)):.4f}, "
            f"latest_price_date={latest_price_date_text}, price_basis={price_basis_text}, data_source={data_source_text}, "
            f"stale_price_warning={stale_price_warning}, value={float(state.current_values.loc[ticker]):.2f}"
            for ticker, shares in non_zero_positions.items()
        )
    lines.extend(
        [
            "current_weights:",
        ]
    )
    if non_zero_weights.empty:
        lines.append("  none (all value currently held as cash)")
    else:
        lines.extend(f"  {ticker}: {weight:.6f}" for ticker, weight in non_zero_weights.items())
    lines.extend(
        [
        "assumed_current_weights:",
        ]
    )
    if non_zero_weights.empty:
        lines.append("  none (all value currently held as cash)")
    else:
        lines.extend(f"  {ticker}: {weight:.6f}" for ticker, weight in non_zero_weights.items())
    lines.extend(
        [
            "consistency_checks:",
            f"  nav_positive: {nav_positive}",
            f"  cash_not_nan: {not cash_is_nan}",
            f"  missing_latest_price_count: {len(missing_price_tickers)}",
            f"  missing_latest_price_tickers: {', '.join(missing_price_tickers) if missing_price_tickers else 'none'}",
            f"  negative_market_value_count: {len(negative_market_value_tickers)}",
            f"  negative_market_value_tickers: {', '.join(negative_market_value_tickers) if negative_market_value_tickers else 'none'}",
            f"  negative_shares_count: {len(negative_share_tickers)}",
            f"  negative_share_tickers: {', '.join(negative_share_tickers) if negative_share_tickers else 'none'}",
            f"  weights_including_cash_close_to_1: {weights_close_to_one}",
            f"  positions_present_not_100pct_cash: {not (positions_count > 0 and current_portfolio_100pct_cash)}",
        ]
    )
    if state.parser_warnings:
        lines.append("parser_warnings:")
        lines.extend(f"  - {warning}" for warning in state.parser_warnings)
    if state.parser_errors:
        lines.append("parser_errors:")
        lines.extend(f"  - {error}" for error in state.parser_errors)
    if state.notes:
        lines.append("notes:")
        lines.extend(f"  - {note}" for note in state.notes)
    if current_portfolio_asset_limit_violations:
        lines.append("current_portfolio_asset_limit_violation_details:")
        lines.extend(
            f"  - {item.get('ticker')}: actual={float(item.get('actual_weight', 0.0)):.6f}, "
            f"limit={float(item.get('limit', 0.0)):.6f}, excess={float(item.get('excess', 0.0)):.6f}"
            for item in current_portfolio_asset_limit_violations
        )
    if current_portfolio_group_limit_violations:
        lines.append("current_portfolio_group_limit_violation_details:")
        lines.extend(
            f"  - {item.get('group')}: actual={float(item.get('actual_weight', 0.0)):.6f}, "
            f"limit={float(item.get('limit', 0.0)):.6f}, excess={float(item.get('excess', 0.0)):.6f}"
            for item in current_portfolio_group_limit_violations
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _position_signature(shares: pd.Series, cash_left: float) -> tuple[tuple[str, int], ...]:
    items = [(str(t), int(round(v))) for t, v in shares.sort_index().items()]
    items.append(("__cash_cents__", int(round(float(cash_left) * 100))))
    return tuple(items)


def _count_discrete_orders(
    *,
    candidate_shares: pd.Series,
    current_positions: pd.Series,
    prices: pd.Series,
    min_order_value: float,
) -> tuple[int, int]:
    order_count = 0
    skipped_small_orders = 0
    deltas = candidate_shares.reindex(current_positions.index).fillna(0.0) - current_positions.fillna(0.0)
    for ticker, delta in deltas.items():
        if abs(float(delta)) <= 1e-9:
            continue
        order_value = abs(float(delta) * float(prices.get(ticker, 0.0)))
        if order_value < float(min_order_value):
            skipped_small_orders += 1
            continue
        order_count += 1
    return order_count, skipped_small_orders


def _position_count(candidate_shares: pd.Series) -> int:
    return int(candidate_shares.fillna(0.0).abs().gt(1e-9).sum())


def _finalize_candidate(
    *,
    name: str,
    shares: pd.Series,
    latest_prices: pd.Series,
    nav: float,
    cash_proxy_ticker: str | None,
    current_positions: pd.Series | None,
    min_order_value: float,
    metadata: dict[str, Any] | None = None,
) -> DiscreteCandidate | None:
    prices = latest_prices.reindex(shares.index).astype(float)
    candidate_shares = shares.astype(float).clip(lower=0.0)
    current_positions = _prepare_series(shares.index, current_positions)
    skipped_small_orders = 0

    for ticker in candidate_shares.index:
        if current_positions.loc[ticker] <= 1e-12 and candidate_shares.loc[ticker] * prices.loc[ticker] < min_order_value:
            candidate_shares.loc[ticker] = 0.0
            skipped_small_orders += 1

    values = (candidate_shares * prices).astype(float)
    cash_left = float(nav - values.sum())
    if cash_left < -1e-6:
        return None
    cash_left = max(cash_left, 0.0)
    weights_actual = (values / nav).fillna(0.0) if nav > 0.0 else values * 0.0
    cash_weight = float(cash_left / nav) if nav > 0.0 else 0.0
    weights_proxy = weights_actual.copy()
    if cash_proxy_ticker is not None and cash_proxy_ticker in weights_proxy.index:
        weights_proxy.loc[cash_proxy_ticker] += cash_weight
    weights_proxy = weights_proxy.clip(lower=0.0)
    total_proxy = float(weights_proxy.sum())
    if total_proxy > 0.0:
        weights_proxy = weights_proxy / total_proxy
    final_metadata = dict(metadata or {})
    number_of_orders, residual_small_orders = _count_discrete_orders(
        candidate_shares=candidate_shares,
        current_positions=current_positions,
        prices=prices,
        min_order_value=min_order_value,
    )
    final_metadata["number_of_positions"] = _position_count(candidate_shares)
    final_metadata["number_of_orders"] = number_of_orders
    final_metadata["skipped_small_orders"] = skipped_small_orders + residual_small_orders

    return DiscreteCandidate(
        name=name,
        shares=candidate_shares,
        values=values,
        weights_actual=weights_actual,
        weights_proxy=weights_proxy,
        cash_left=cash_left,
        cash_weight=cash_weight,
        metadata=final_metadata,
    )


def _greedy_fill(
    *,
    base_shares: pd.Series,
    target_values: pd.Series,
    latest_prices: pd.Series,
    nav: float,
    reserved_cash: float,
    priority: pd.Series,
) -> pd.Series:
    shares = base_shares.astype(float).copy()
    prices = latest_prices.reindex(shares.index).astype(float)
    priority = priority.reindex(shares.index).fillna(0.0).astype(float)
    max_loops = 20000
    loops = 0

    while loops < max_loops:
        current_values = shares * prices
        cash_left = float(nav - current_values.sum())
        affordable = prices[prices <= max(cash_left - reserved_cash, 0.0) + 1e-9]
        if affordable.empty:
            break
        gaps = (target_values - current_values).clip(lower=0.0)
        candidates = [ticker for ticker in affordable.index if gaps.loc[ticker] > 0.0]
        if not candidates:
            break
        metric = (
            (gaps.loc[candidates] / affordable.loc[candidates]).astype(float)
            + 0.05 * priority.loc[candidates].rank(method="average", ascending=False)
        )
        best = str(metric.sort_values(ascending=False).index[0])
        if float(metric.loc[best]) <= 0.0:
            break
        shares.loc[best] += 1.0
        loops += 1
    return shares


def _repair_negative_cash(
    *,
    shares: pd.Series,
    target_values: pd.Series,
    latest_prices: pd.Series,
    nav: float,
    reserved_cash: float,
    priority: pd.Series,
) -> pd.Series:
    repaired = shares.astype(float).clip(lower=0.0).copy()
    prices = latest_prices.reindex(repaired.index).astype(float)
    priority = priority.reindex(repaired.index).fillna(0.0).astype(float)
    max_loops = 20000
    loops = 0
    while loops < max_loops and float((repaired * prices).sum()) > nav - reserved_cash + 1e-9:
        current_values = repaired * prices
        overweight = (current_values - target_values).clip(lower=0.0)
        candidates = repaired[repaired > 0.0].index.tolist()
        if not candidates:
            break
        metric = (
            overweight.reindex(candidates).fillna(0.0).astype(float)
            - 0.05 * priority.reindex(candidates).fillna(0.0)
        )
        best = str(metric.sort_values(ascending=False).index[0])
        repaired.loc[best] = max(repaired.loc[best] - 1.0, 0.0)
        loops += 1
    return repaired


def _repair_candidate_shares_to_constraints(
    *,
    candidate: DiscreteCandidate,
    latest_prices: pd.Series,
    nav: float,
    cash_proxy_ticker: str | None,
    current_positions: pd.Series,
    min_order_value: float,
    params: dict[str, Any],
) -> DiscreteCandidate | None:
    """Trim whole-share positions until the discrete candidate satisfies hard caps."""

    shares = candidate.shares.astype(float).clip(lower=0.0).copy()
    prices = latest_prices.reindex(shares.index).astype(float)
    group_map = pd.Series(params.get("group_map", {}), dtype=object)
    group_map.index = pd.Index([str(t) for t in group_map.index], name="ticker")
    group_map = group_map.reindex(shares.index)
    repaired = candidate
    for _ in range(10000):
        validation = _validate_discrete_candidate_constraints(
            repaired,
            current_weights_index=shares.index,
            params=params,
        )
        if validation["ok"]:
            repaired.metadata["share_repair_applied"] = True
            return repaired
        trim_ticker = ""
        asset_violations = list(validation.get("asset_limit_violations", []))
        group_violations = list(validation.get("group_limit_violations", []))
        if asset_violations:
            worst = max(asset_violations, key=lambda item: float(item.get("excess", 0.0)))
            trim_ticker = str(worst.get("ticker", ""))
        elif group_violations:
            worst_group = max(group_violations, key=lambda item: float(item.get("excess", 0.0)))
            group_name = str(worst_group.get("group", ""))
            if group_name.startswith("__"):
                eligible = shares[shares > 0.0].index
            else:
                eligible = group_map[group_map.astype(str).eq(group_name)].index
            values = (shares.reindex(eligible).fillna(0.0) * prices.reindex(eligible).fillna(0.0)).sort_values(ascending=False)
            trim_ticker = str(values.index[0]) if not values.empty else ""
        if not trim_ticker or trim_ticker not in shares.index or shares.loc[trim_ticker] <= 0.0:
            break
        shares.loc[trim_ticker] = max(float(shares.loc[trim_ticker]) - 1.0, 0.0)
        next_candidate = _finalize_candidate(
            name=candidate.name,
            shares=shares,
            latest_prices=prices,
            nav=float(nav),
            cash_proxy_ticker=cash_proxy_ticker,
            current_positions=current_positions,
            min_order_value=float(min_order_value),
            metadata={**candidate.metadata, "share_repair_trimmed": trim_ticker},
        )
        if next_candidate is None:
            break
        repaired = next_candidate
    repaired.metadata["share_repair_failed"] = True
    return repaired


def _validate_discrete_candidate_constraints(
    candidate: DiscreteCandidate,
    current_weights_index: pd.Index,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Validate discrete candidates using actual invested weights plus literal cash.

    The scenario/scoring layer maps literal cash into the cash proxy ticker for
    expected-return modeling. Constraint validation must not do that, otherwise
    a 100% actual-cash state can be misclassified as an overweight SGOV
    position.
    """

    return validate_portfolio_constraints(
        weights_actual=candidate.weights_actual.reindex(current_weights_index).fillna(0.0),
        cash_weight=float(candidate.cash_weight),
        params=params,
        index=current_weights_index,
        label="Discrete target",
    )


def generate_discrete_candidates(
    target_weights: pd.Series,
    latest_prices: pd.Series,
    nav: float,
    current_positions: pd.Series | None = None,
    current_cash: float | None = None,
    min_order_value: float = 10.0,
    cash_buffer: float = 0.0,
    max_candidates: int = 50,
    allow_fractional_shares: bool = False,
    marginal_priority: pd.Series | None = None,
    cash_proxy_ticker: str | None = None,
    constraint_params: dict[str, Any] | None = None,
) -> list[DiscreteCandidate]:
    """Generate several whole-share variants around a continuous target."""

    index = pd.Index([str(t) for t in target_weights.index], name="ticker")
    weights = _normalize(target_weights.reindex(index).fillna(0.0))
    prices = latest_prices.reindex(index).astype(float)
    prices = prices.where(prices > 0.0).dropna()
    weights = weights.reindex(prices.index).fillna(0.0)
    weights = _normalize(weights)
    if weights.empty:
        return []

    priority = _prepare_series(weights.index, marginal_priority).add(weights, fill_value=0.0)
    current_positions = _prepare_series(weights.index, current_positions)
    target_values_full = weights * float(nav)

    candidates: list[DiscreteCandidate] = []
    seen: set[tuple[tuple[str, int], ...]] = set()

    def add_candidate(name: str, shares: pd.Series, reserved_cash: float, metadata: dict[str, Any] | None = None) -> None:
        candidate = _finalize_candidate(
            name=name,
            shares=shares.reindex(weights.index).fillna(0.0),
            latest_prices=prices,
            nav=float(nav),
            cash_proxy_ticker=cash_proxy_ticker,
            current_positions=current_positions,
            min_order_value=float(min_order_value),
            metadata=metadata,
        )
        if candidate is None:
            return
        if constraint_params is not None:
            repaired_candidate = _repair_candidate_shares_to_constraints(
                candidate=candidate,
                latest_prices=prices,
                nav=float(nav),
                cash_proxy_ticker=cash_proxy_ticker,
                current_positions=current_positions,
                min_order_value=float(min_order_value),
                params=constraint_params,
            )
            if repaired_candidate is None:
                return
            candidate = repaired_candidate
        if candidate.cash_left + 1e-9 < reserved_cash:
            return
        signature = _position_signature(candidate.shares, candidate.cash_left)
        if signature in seen:
            return
        seen.add(signature)
        candidates.append(candidate)

    cash_buffer_variants = [float(cash_buffer)]
    for buffer_pct in (0.0, 0.0025, 0.005, 0.01):
        reserved = max(float(nav) * buffer_pct, float(cash_buffer))
        if reserved not in cash_buffer_variants:
            cash_buffer_variants.append(reserved)

    top_n = min(15, len(weights))
    top_weights = _normalize(weights.sort_values(ascending=False).head(top_n))

    for reserved_cash in cash_buffer_variants:
        spendable_nav = max(float(nav) - reserved_cash, 0.0)
        if spendable_nav <= 0.0:
            continue
        reserved_cash_token = f"{int(round(float(reserved_cash) * 100))}c"
        target_values = weights * spendable_nav
        floor_shares = np.floor(target_values / prices).astype(float)
        nearest_shares = np.round(target_values / prices).astype(float)
        target_values_top = top_weights.reindex(weights.index).fillna(0.0) * spendable_nav
        top_floor = np.floor(target_values_top / prices).astype(float)

        add_candidate(
            f"FLOOR_BASE_{reserved_cash_token}",
            floor_shares,
            reserved_cash,
            {"kind": "floor", "reserved_cash": reserved_cash},
        )
        greedy_fill = _greedy_fill(
            base_shares=floor_shares,
            target_values=target_values,
            latest_prices=prices,
            nav=float(nav),
            reserved_cash=reserved_cash,
            priority=weights,
        )
        add_candidate(
            f"GREEDY_FILL_{reserved_cash_token}",
            greedy_fill,
            reserved_cash,
            {"kind": "greedy_fill", "reserved_cash": reserved_cash},
        )

        nearest_repaired = _repair_negative_cash(
            shares=nearest_shares,
            target_values=target_values,
            latest_prices=prices,
            nav=float(nav),
            reserved_cash=reserved_cash,
            priority=weights,
        )
        add_candidate(
            f"ROUND_NEAREST_REPAIR_{reserved_cash_token}",
            nearest_repaired,
            reserved_cash,
            {"kind": "round_nearest_repair", "reserved_cash": reserved_cash},
        )

        underweight_priority = _greedy_fill(
            base_shares=floor_shares,
            target_values=target_values,
            latest_prices=prices,
            nav=float(nav),
            reserved_cash=reserved_cash,
            priority=(target_values / prices).fillna(0.0),
        )
        add_candidate(
            f"UNDERWEIGHT_PRIORITY_{reserved_cash_token}",
            underweight_priority,
            reserved_cash,
            {"kind": "underweight_priority", "reserved_cash": reserved_cash},
        )

        score_aware = _greedy_fill(
            base_shares=floor_shares,
            target_values=target_values,
            latest_prices=prices,
            nav=float(nav),
            reserved_cash=reserved_cash,
            priority=priority,
        )
        add_candidate(
            f"SCORE_AWARE_GREEDY_{reserved_cash_token}",
            score_aware,
            reserved_cash,
            {"kind": "score_aware_greedy", "reserved_cash": reserved_cash},
        )

        low_turnover = floor_shares.copy()
        for ticker in low_turnover.index:
            desired = floor_shares.loc[ticker]
            current = current_positions.reindex(low_turnover.index).fillna(0.0).loc[ticker]
            if abs(desired - current) * prices.loc[ticker] < float(min_order_value):
                low_turnover.loc[ticker] = current
        low_turnover = _repair_negative_cash(
            shares=low_turnover,
            target_values=target_values,
            latest_prices=prices,
            nav=float(nav),
            reserved_cash=reserved_cash,
            priority=priority,
        )
        add_candidate(
            f"LOW_TURNOVER_VARIANT_{reserved_cash_token}",
            low_turnover,
            reserved_cash,
            {"kind": "low_turnover", "reserved_cash": reserved_cash},
        )

        add_candidate(
            f"TOP_{top_n}_REDUCED_{reserved_cash_token}",
            top_floor,
            reserved_cash,
            {"kind": "top_n_reduced", "reserved_cash": reserved_cash},
        )

    if allow_fractional_shares:
        exact_values = target_values_full.astype(float)
        exact_shares = exact_values / prices
        add_candidate("FRACTIONAL_EXACT", exact_shares, float(cash_buffer), {"kind": "fractional_exact"})

    hold_current = _finalize_candidate(
        name="HOLD_CURRENT",
        shares=current_positions.reindex(weights.index).fillna(0.0),
        latest_prices=prices,
        nav=float(nav),
        cash_proxy_ticker=cash_proxy_ticker,
        current_positions=current_positions,
        min_order_value=float(min_order_value),
        metadata={"kind": "hold_current"},
    )
    if hold_current is not None:
        signature = _position_signature(hold_current.shares, hold_current.cash_left)
        if signature not in seen:
            seen.add(signature)
            candidates.append(hold_current)

    return candidates[: int(max_candidates)]


def score_discrete_candidates(
    discrete_candidates: list[DiscreteCandidate],
    scenario_returns: ScenarioSet,
    scorer_config: dict[str, Any],
    current_weights: pd.Series | None = None,
    transaction_cost_config: dict[str, Any] | None = None,
    current_shares: pd.Series | None = None,
    current_cash: float | None = None,
    latest_prices: pd.Series | None = None,
    nav: float | None = None,
) -> dict[str, Any]:
    """Re-score discrete whole-share portfolios with the same robust logic."""

    params = dict(scorer_config.get("params", {}))
    if transaction_cost_config:
        params.update(transaction_cost_config)
    current_weights = _prepare_series(pd.Index(scenario_returns.scenario_returns_matrix.columns, name="ticker"), current_weights)
    current_shares_series = _prepare_series(current_weights.index, current_shares)
    latest_prices_series = None if latest_prices is None else latest_prices.reindex(current_weights.index).astype(float)
    nav_value = float(nav) if nav is not None else _float_or_none(params.get("portfolio_nav_usd"))
    hold_weights = _prepare_series(current_weights.index, scorer_config.get("hold_weights"))
    cash_weights = _prepare_series(current_weights.index, scorer_config.get("cash_weights"))
    continuous_target = _prepare_series(current_weights.index, scorer_config.get("continuous_target"))

    hold_candidate = CandidatePortfolio(name="HOLD_REFERENCE", weights=hold_weights)
    cash_candidate = CandidatePortfolio(name="CASH_REFERENCE", weights=cash_weights)
    hold_score = evaluate_candidate(hold_candidate, scenario_returns, current_weights, params)
    cash_score = evaluate_candidate(cash_candidate, scenario_returns, current_weights, params)

    matrix = scenario_returns.scenario_returns_matrix.reindex(columns=current_weights.index).fillna(0.0)
    probs = scenario_returns.scenario_probabilities.reindex(matrix.index).fillna(0.0).to_numpy(dtype=float)
    probs = probs / max(probs.sum(), 1e-12)
    hold_returns = matrix.to_numpy(dtype=float) @ hold_weights.reindex(matrix.columns).fillna(0.0).to_numpy(dtype=float)
    cash_returns = matrix.to_numpy(dtype=float) @ cash_weights.reindex(matrix.columns).fillna(0.0).to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    score_objects: dict[str, Any] = {}
    candidate_map = {candidate.name: candidate for candidate in discrete_candidates}
    selection_config = {
        "hurdle": float(params.get("hurdle", 0.001)),
        "risk_premium_hurdle": float(params.get("risk_premium_hurdle", 0.0005)),
        "p_hold_min": float(params.get("p_hold_min", 0.55)),
        "p_cash_min": float(params.get("p_cash_min", 0.52)),
        "optimization_objective": str(params.get("optimization_objective", "robust_score")),
    }
    for candidate in discrete_candidates:
        drift_target = _prepare_series(
            current_weights.index,
            candidate.metadata.get("continuous_target_weights", continuous_target),
        )
        score = evaluate_candidate(
            CandidatePortfolio(name=candidate.name, weights=candidate.weights_proxy),
            scenario_returns,
            current_weights,
            params,
        )
        if latest_prices_series is not None and nav_value is not None and nav_value > 0.0:
            candidate_preview = build_discrete_order_preview(
                current_shares=current_shares_series,
                target_shares=candidate.shares.reindex(current_weights.index).fillna(0.0),
                latest_prices=latest_prices_series,
                nav=nav_value,
                min_order_value=float(params.get("min_order_value_usd", 10.0)),
            )
            _, cost_summary = estimate_order_list_costs(
                order_preview_df=candidate_preview,
                latest_prices=latest_prices_series,
                config={
                    **params,
                    "nav": nav_value,
                    "current_cash": float(current_cash) if current_cash is not None else np.nan,
                },
            )
            score.estimated_cost = float(cost_summary["total_order_cost_pct_nav"])
            score.estimated_commission = float(cost_summary["total_estimated_commission"])
            score.estimated_spread_cost = float(cost_summary["total_estimated_spread_cost"])
            score.estimated_slippage_cost = float(cost_summary["total_estimated_slippage_cost"])
            score.estimated_market_impact_cost = float(cost_summary["total_estimated_market_impact_cost"])
            score.estimated_total_order_cost = float(cost_summary["total_estimated_transaction_cost"])
            score.cost_bps_used = float(cost_summary["weighted_average_cost_bps"])
            score.cost_model_used = str(cost_summary["cost_model_used"])
            score.live_costs_available = bool(cost_summary["live_costs_available"])
            score.net_robust_score = score.robust_score - score.estimated_cost - score.dynamic_buffer
        candidate_returns = matrix.to_numpy(dtype=float) @ candidate.weights_proxy.reindex(matrix.columns).fillna(0.0).to_numpy(dtype=float)
        score.delta_vs_hold = score.net_robust_score - hold_score.net_robust_score
        score.delta_vs_cash = score.net_robust_score - cash_score.net_robust_score
        score.probability_beats_hold = float(np.sum(probs[candidate_returns > hold_returns]))
        score.probability_beats_cash = float(np.sum(probs[candidate_returns > cash_returns]))
        drift = candidate.weights_proxy.reindex(current_weights.index).fillna(0.0) - drift_target
        weight_validation = _validate_discrete_candidate_constraints(
            candidate=candidate,
            current_weights_index=current_weights.index,
            params=params,
        )
        asset_violations = list(weight_validation.get("asset_limit_violations", []))
        group_violations = list(weight_validation.get("group_limit_violations", []))
        rows.append(
            {
                "discrete_candidate": candidate.name,
                "mean_return": score.mean_return,
                "median_return": score.median_return,
                "volatility": score.volatility,
                "risk_free_return": score.risk_free_return,
                "robust_sharpe": score.robust_sharpe,
                "return_over_volatility_legacy": score.return_over_volatility_legacy,
                "robust_score": score.robust_score,
                "gross_robust_score": score.robust_score,
                "net_robust_score": score.net_robust_score,
                "net_score_after_order_costs": score.net_robust_score,
                "cvar_5": score.cvar_5,
                "probability_loss": score.probability_loss,
                "probability_beats_hold": score.probability_beats_hold,
                "probability_beats_cash": score.probability_beats_cash,
                "turnover": score.turnover,
                "turnover_vs_current": score.turnover,
                "estimated_cost": score.estimated_cost,
                "estimated_transaction_cost": score.estimated_cost,
                "estimated_commission": score.estimated_commission,
                "estimated_spread_cost": score.estimated_spread_cost,
                "estimated_slippage_cost": score.estimated_slippage_cost,
                "estimated_market_impact_cost": score.estimated_market_impact_cost,
                "total_estimated_transaction_cost": score.estimated_total_order_cost,
                "total_order_cost": score.estimated_total_order_cost,
                "total_order_cost_pct_nav": score.estimated_cost,
                "cost_bps_used": score.cost_bps_used,
                "cost_model_used": score.cost_model_used,
                "live_costs_available": score.live_costs_available,
                "dynamic_buffer": score.dynamic_buffer,
                "cash_left": candidate.cash_left,
                "cash_weight": candidate.cash_weight,
                "continuous_source": str(candidate.metadata.get("continuous_source", "")),
                "delta_vs_hold": score.delta_vs_hold,
                "delta_vs_cash": score.delta_vs_cash,
                "total_abs_weight_drift": float(drift.abs().sum()),
                "total_abs_weight_drift_vs_continuous_target": float(drift.abs().sum()),
                "max_abs_weight_drift": float(drift.abs().max()),
                "max_abs_weight_drift_vs_continuous_target": float(drift.abs().max()),
                "number_of_positions": int(candidate.metadata.get("number_of_positions", _position_count(candidate.shares))),
                "number_of_orders": int(candidate.metadata.get("number_of_orders", 0)),
                "skipped_small_orders": int(candidate.metadata.get("skipped_small_orders", 0)),
                "valid_constraints": bool(weight_validation["ok"]),
                "validation_errors": "; ".join(weight_validation["errors"]) if weight_validation["errors"] else "",
                "asset_limit_violations": json.dumps(asset_violations, sort_keys=True) if asset_violations else "",
                "group_limit_violations": json.dumps(group_violations, sort_keys=True) if group_violations else "",
                "selection_failed_reason": "",
                "selected": False,
            }
        )
        score_objects[candidate.name] = score

    scores_frame = pd.DataFrame(rows).sort_values("net_robust_score", ascending=False).reset_index(drop=True)
    return {
        "scores_frame": scores_frame,
        "score_objects": score_objects,
        "candidate_map": candidate_map,
        "selection_config": selection_config,
    }


def select_best_discrete_portfolio(scored_discrete_candidates: dict[str, Any]) -> dict[str, Any]:
    """Select the best valid discrete portfolio with HOLD_CURRENT as safe default."""

    scores_frame = scored_discrete_candidates["scores_frame"].copy()
    candidate_map = scored_discrete_candidates["candidate_map"]
    selection_config = dict(scored_discrete_candidates.get("selection_config", {}))
    if scores_frame.empty:
        raise ValueError("No discrete candidates were generated.")

    hurdle = float(selection_config.get("hurdle", 0.001))
    risk_premium_hurdle = float(selection_config.get("risk_premium_hurdle", 0.0005))
    p_hold_min = float(selection_config.get("p_hold_min", 0.55))
    p_cash_min = float(selection_config.get("p_cash_min", 0.52))
    objective_used = str(selection_config.get("optimization_objective", "robust_score")).strip().lower()
    objective_score_column = "net_robust_score"
    if objective_used in {
        "scenario_mixture_sharpe",
        "robust_scenario_sharpe_objective",
        "direct_scenario_sharpe",
        "scenario_weighted_rf_sharpe",
    }:
        preferred_column = (
            "scenario_mixture_sharpe"
            if objective_used == "scenario_mixture_sharpe"
            else "robust_scenario_sharpe_objective"
        )
        if preferred_column in scores_frame.columns:
            objective_score_column = preferred_column
        else:
            objective_used = "robust_score"
    scores_frame["objective_used"] = objective_used
    scores_frame["objective_score_column"] = objective_score_column
    scores_frame["_selection_objective_score"] = pd.to_numeric(
        scores_frame.get(objective_score_column, scores_frame["net_robust_score"]),
        errors="coerce",
    ).fillna(float("-inf"))

    hold_rows = scores_frame[scores_frame["discrete_candidate"].astype(str) == "HOLD_CURRENT"]
    hold_row = hold_rows.iloc[0] if not hold_rows.empty else None
    hold_score = float(hold_row["_selection_objective_score"]) if hold_row is not None else float("-inf")
    hold_net_score = float(hold_row["net_robust_score"]) if hold_row is not None else float("-inf")
    hold_current_constraint_valid = bool(hold_row["valid_constraints"]) if hold_row is not None else False
    current_constraint_errors = str(hold_row.get("validation_errors", "") if hold_row is not None else "").strip()

    valid_scores = scores_frame[scores_frame["valid_constraints"] == True].copy()  # noqa: E712
    valid_scores = valid_scores.sort_values(
        by=[
            "_selection_objective_score",
            "cvar_5",
            "turnover_vs_current",
            "max_abs_weight_drift",
            "number_of_positions",
            "cash_left",
        ],
        ascending=[False, False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    failure_reasons: dict[str, str] = {}
    if not valid_scores.empty:
        for _, row in valid_scores.iterrows():
            name = str(row["discrete_candidate"])
            if name == "HOLD_CURRENT":
                failure_reasons[name] = ""
                continue
            if float(row["_selection_objective_score"]) <= hold_score + hurdle:
                failure_reasons[name] = "failed_hurdle"
                continue
            if float(row["delta_vs_cash"]) <= risk_premium_hurdle:
                failure_reasons[name] = "failed_costs"
                continue
            if float(row["probability_beats_hold"]) < p_hold_min or float(row["probability_beats_cash"]) < p_cash_min:
                failure_reasons[name] = "failed_probability"
                continue
            if float(row["cash_left"]) < -1e-8:
                failure_reasons[name] = "failed_cash"
                continue
            failure_reasons[name] = ""
    invalid_scores = scores_frame[scores_frame["valid_constraints"] != True].copy()  # noqa: E712
    if not invalid_scores.empty:
        for row in invalid_scores.itertuples(index=False):
            failure_reasons[str(row.discrete_candidate)] = "failed_constraints"

    scores_frame["selection_failed_reason"] = scores_frame["discrete_candidate"].astype(str).map(failure_reasons).fillna("")

    non_hold_scores = scores_frame[scores_frame["discrete_candidate"].astype(str) != "HOLD_CURRENT"].copy()
    best_non_hold_row = (
        non_hold_scores.sort_values("_selection_objective_score", ascending=False, kind="mergesort").iloc[0]
        if not non_hold_scores.empty
        else None
    )

    chosen_row = hold_row if hold_row is not None else None
    reason = "Defaulted to HOLD_CURRENT because no discrete candidate cleared all post-cost hurdles."
    for _, row in valid_scores.iterrows():
        name = str(row["discrete_candidate"])
        if name == "HOLD_CURRENT":
            continue
        if failure_reasons.get(name, ""):
            continue
        chosen_row = row
        reason = f"Selected the highest-ranked discrete candidate by {objective_used} that cleared HOLD/CASH probability and post-cost hurdles."
        break

    if chosen_row is None:
        chosen_row = valid_scores.iloc[0] if not valid_scores.empty else scores_frame.iloc[0]
        reason = "No HOLD_CURRENT row was available; used the best available discrete candidate as a fallback."

    best_name = str(chosen_row["discrete_candidate"])
    scores_frame.loc[scores_frame["discrete_candidate"] == best_name, "selected"] = True
    candidate = candidate_map[best_name]
    selected_reason = "selected_trade_candidate"
    if best_name == "HOLD_CURRENT":
        valid_non_hold = non_hold_scores.loc[non_hold_scores.get("valid_constraints", False) == True].copy()  # noqa: E712
        better_valid_non_hold = valid_non_hold.loc[
            pd.to_numeric(valid_non_hold.get("_selection_objective_score", pd.Series(dtype=float)), errors="coerce").fillna(float("-inf"))
            > hold_score
        ].copy()
        if not hold_current_constraint_valid:
            selected_reason = "constraint_invalid_hold_fallback"
        elif valid_non_hold.empty:
            selected_reason = "safe_hold_fallback_no_valid_trade_candidate"
        elif not better_valid_non_hold.empty:
            selected_reason = "gate_blocked_hold_fallback"
        else:
            selected_reason = "optimal_hold"
    final_selection_is_safe_fallback = bool(
        best_name == "HOLD_CURRENT" and selected_reason != "optimal_hold"
    )
    best_non_hold_candidate = str(best_non_hold_row["discrete_candidate"]) if best_non_hold_row is not None else ""
    best_non_hold_score = float(best_non_hold_row["net_robust_score"]) if best_non_hold_row is not None else np.nan
    best_non_hold_objective_score = (
        float(best_non_hold_row["_selection_objective_score"]) if best_non_hold_row is not None else np.nan
    )
    best_non_hold_valid_constraints = bool(best_non_hold_row["valid_constraints"]) if best_non_hold_row is not None else False
    best_non_hold_failed_reason = (
        str(best_non_hold_row.get("selection_failed_reason", "") or "")
        if best_non_hold_row is not None
        else ""
    )
    if best_non_hold_row is not None and not best_non_hold_failed_reason and not best_non_hold_valid_constraints:
        best_non_hold_failed_reason = "failed_constraints"
    best_model_candidate = str(best_non_hold_candidate or best_name)
    best_model_candidate_valid_constraints = bool(best_non_hold_valid_constraints if best_non_hold_candidate else hold_current_constraint_valid)
    scores_frame["selected_reason"] = selected_reason
    scores_frame["hold_current_constraint_valid"] = hold_current_constraint_valid
    scores_frame["current_portfolio_constraint_violation"] = not hold_current_constraint_valid
    scores_frame["current_constraint_errors"] = current_constraint_errors
    scores_frame["best_non_hold_candidate"] = best_non_hold_candidate
    scores_frame["best_non_hold_score"] = best_non_hold_score
    scores_frame["best_non_hold_objective_score"] = best_non_hold_objective_score
    scores_frame["best_non_hold_valid_constraints"] = best_non_hold_valid_constraints
    scores_frame["best_non_hold_failed_reason"] = best_non_hold_failed_reason
    scores_frame["best_model_candidate"] = best_model_candidate
    scores_frame["best_model_candidate_valid_constraints"] = best_model_candidate_valid_constraints
    scores_frame["final_selection_is_safe_fallback"] = final_selection_is_safe_fallback
    return {
        "best_discrete_candidate_name": best_name,
        "best_discrete_weights": candidate.weights_proxy.copy(),
        "best_discrete_shares": candidate.shares.copy(),
        "best_discrete_score": float(chosen_row["net_robust_score"]),
        "best_discrete_objective_score": float(chosen_row["_selection_objective_score"]),
        "objective_used": objective_used,
        "objective_score_column": objective_score_column,
        "hold_objective_score": hold_score,
        "hold_net_robust_score": hold_net_score,
        "best_discrete_allocation": candidate.weights_actual.copy(),
        "best_discrete_orders": candidate.shares.copy(),
        "reason": reason,
        "selected_reason": selected_reason,
        "hold_current_constraint_valid": hold_current_constraint_valid,
        "current_portfolio_constraint_violation": not hold_current_constraint_valid,
        "current_constraint_errors": current_constraint_errors,
        "best_non_hold_candidate": best_non_hold_candidate,
        "best_non_hold_score": best_non_hold_score,
        "best_non_hold_objective_score": best_non_hold_objective_score,
        "best_non_hold_valid_constraints": best_non_hold_valid_constraints,
        "best_non_hold_failed_reason": best_non_hold_failed_reason,
        "best_model_candidate": best_model_candidate,
        "best_model_candidate_valid_constraints": best_model_candidate_valid_constraints,
        "final_selection_is_safe_fallback": final_selection_is_safe_fallback,
        "scores_frame": scores_frame,
        "candidate": candidate,
    }


def build_discrete_order_preview(
    *,
    current_shares: pd.Series,
    target_shares: pd.Series,
    latest_prices: pd.Series,
    nav: float,
    min_order_value: float = 10.0,
    not_executable: bool = False,
    reason: str = "",
) -> pd.DataFrame:
    """Build a whole-share order preview from current and target positions."""

    tickers = pd.Index(
        list(dict.fromkeys([*current_shares.index.tolist(), *target_shares.index.tolist()])),
        name="ticker",
    )
    current_shares = current_shares.reindex(tickers).fillna(0.0).astype(float)
    target_shares = target_shares.reindex(tickers).fillna(0.0).astype(float)
    prices = latest_prices.reindex(tickers).fillna(0.0).astype(float)

    current_value = current_shares * prices
    target_value = target_shares * prices
    shares_delta_series = target_shares - current_shares
    order_shares = shares_delta_series.abs()
    order_value = shares_delta_series * prices

    rows = []
    for ticker in tickers:
        value = float(order_value.loc[ticker])
        shares_delta = float(shares_delta_series.loc[ticker])
        absolute_order_shares = float(order_shares.loc[ticker])
        side = "HOLD"
        row_reason = reason if not_executable else ""
        too_small_or_no_change = abs(value) < float(min_order_value) or abs(shares_delta) < 1e-9
        if too_small_or_no_change:
            row_reason = row_reason or "too_small_or_no_change"
        elif shares_delta > 0:
            side = "BUY"
        else:
            side = "SELL"
        rows.append(
            {
                "ticker": str(ticker),
                "current_shares": float(current_shares.loc[ticker]),
                "target_shares": float(target_shares.loc[ticker]),
                "order_shares": absolute_order_shares,
                "shares_delta": shares_delta,
                "latest_price": float(prices.loc[ticker]),
                "estimated_price": float(prices.loc[ticker]),
                "current_value": float(current_value.loc[ticker]),
                "target_value": float(target_value.loc[ticker]),
                "order_value": value,
                "current_weight": float(current_value.loc[ticker] / nav) if nav > 0 else 0.0,
                "target_weight": float(target_value.loc[ticker] / nav) if nav > 0 else 0.0,
                "delta_weight": float((target_value.loc[ticker] - current_value.loc[ticker]) / nav) if nav > 0 else 0.0,
                "estimated_order_value": abs(value),
                "estimated_shares": absolute_order_shares,
                "price_basis": "adjusted_close_proxy",
                "quote_note": "latest adjusted close is a proxy, not an executable quote",
                "side": "HOLD" if too_small_or_no_change else side,
                "action": "HOLD" if too_small_or_no_change else side,
                "not_executable": bool(not_executable or too_small_or_no_change),
                "reason": row_reason or "",
            }
        )

    return pd.DataFrame(rows)
