"""Run isolated HOLD-vs-BUY/SELL sensitivity experiments on the latest Daily-Bot context."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asset_exposure_model import estimate_asset_factor_exposures
from candidate_factory import CandidatePortfolio, build_candidate_portfolios
from calendar_utils import DEFAULT_PROJECT_CALENDAR_PATH, is_within_project_trading_window
from conditional_scenario_model import build_conditional_scenarios
from config import OUTPUT_DIR, PORTFOLIO_NAV_USD, PRICE_CACHE_PATH, build_params
from data import build_run_data_context, check_data_freshness, load_price_data
from discrete_portfolio_optimizer import (
    build_discrete_order_preview,
    generate_discrete_candidates,
    load_current_portfolio_state,
    score_discrete_candidates,
    select_best_discrete_portfolio,
)
from execution_gate import evaluate_execution_gate
from factor_data import build_factor_data
from factor_forecast import build_factor_forecast
from features import compute_returns
from forecast_3m import Forecast3M, build_forecast_3m
from macro_data import load_macro_proxy_data
from model_governance import compute_model_confidence
from notifications import sanitize_for_output
from optimizer import OptimizerResult, optimize_allocation
from regime_engine import detect_regime
from robust_scorer import RobustSelectionResult, select_robust_candidate
from risk import estimate_robust_covariance_at_date
from scenario_model import ScenarioSet, build_3m_scenarios
from tradability import apply_tradability_filter, build_tradability_report, select_cash_proxy
from transaction_costs import compute_trade_now_edge, estimate_order_list_costs

from daily_bot import _select_discrete_expansion_sources


BASE_EXECUTION_BUFFER = 0.001


@dataclass(slots=True)
class BaseContext:
    params: dict[str, Any]
    active_tickers: list[str]
    prices: pd.DataFrame
    returns: pd.DataFrame
    as_of: pd.Timestamp
    data_context: dict[str, Any]
    risk_state: str
    forecast: Forecast3M
    sigma: pd.DataFrame
    scenario_set: ScenarioSet
    factor_mode: str
    factor_forecasts_df: pd.DataFrame
    current_state: Any
    latest_prices_at_asof: pd.Series
    w_current: pd.Series
    model_governance: dict[str, Any]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number) or not np.isfinite(number):
        return default
    return number


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _current_total_bps(params: dict[str, Any]) -> float:
    return float(
        _safe_float(params.get("default_spread_bps"), 0.0)
        + _safe_float(params.get("default_slippage_bps"), 0.0)
        + _safe_float(params.get("default_market_impact_bps"), 0.0)
    )


def _scaled_cost_overrides(asset_cost_overrides: dict[str, Any], scale: float) -> dict[str, Any]:
    scaled: dict[str, Any] = {}
    for ticker, entry in dict(asset_cost_overrides).items():
        item = dict(entry)
        for key in ("spread_bps", "slippage_bps", "market_impact_bps"):
            item[key] = max(_safe_float(item.get(key), 0.0) * scale, 0.0)
        scaled[str(ticker)] = item
    return scaled


def _format_changed_positions(order_preview: pd.DataFrame, limit: int = 5) -> str:
    if order_preview.empty:
        return "none"
    actionable = order_preview[order_preview["side"].astype(str).isin(["BUY", "SELL"])].copy()
    if actionable.empty:
        return "none"
    actionable["estimated_order_value"] = pd.to_numeric(actionable["estimated_order_value"], errors="coerce").fillna(0.0)
    actionable = actionable.sort_values("estimated_order_value", ascending=False).head(limit)
    parts = []
    for row in actionable.itertuples(index=False):
        parts.append(f"{row.ticker} {row.side} {float(row.order_shares):.0f}")
    return "; ".join(parts) if parts else "none"


def _theoretical_final_action(
    *,
    best_discrete_candidate_name: str,
    order_count: int,
    gate_status: str,
) -> str:
    if best_discrete_candidate_name == "HOLD_CURRENT" or order_count <= 0:
        return "HOLD"
    if str(gate_status).upper() == "PASS":
        return "BUY_SELL_PREVIEW"
    return "WAIT"


def _relax_asset_caps(params: dict[str, Any], scale: float) -> None:
    caps = dict(params.get("asset_max_weights", {}))
    for ticker, value in caps.items():
        caps[str(ticker)] = min(max(_safe_float(value, 0.0) * scale, 0.0), 1.0)
    params["asset_max_weights"] = caps


def _apply_cost_bps(params: dict[str, Any], total_bps: float) -> None:
    base_total = max(_current_total_bps(params), 1e-12)
    scale = float(total_bps) / base_total if total_bps > 0.0 else 0.0
    params["cost_rate"] = float(total_bps) / 10000.0
    params["default_bps_per_turnover"] = float(total_bps)
    params["default_spread_bps"] = max(_safe_float(params.get("default_spread_bps"), 0.0) * scale, 0.0)
    params["default_slippage_bps"] = max(_safe_float(params.get("default_slippage_bps"), 0.0) * scale, 0.0)
    params["default_market_impact_bps"] = max(_safe_float(params.get("default_market_impact_bps"), 0.0) * scale, 0.0)
    params["asset_cost_overrides"] = _scaled_cost_overrides(params.get("asset_cost_overrides", {}), scale)


def _build_base_context() -> BaseContext:
    requested_tickers = list(build_params().get("tickers", []))
    prices = load_price_data(
        tickers=requested_tickers,
        start_date=str(build_params()["start_date"]),
        end_date=build_params()["end_date"],
        cache_path=PRICE_CACHE_PATH,
        use_cache=True,
        prefer_live=True,
        allow_cache_fallback=True,
        force_refresh=False,
    )
    data_freshness = check_data_freshness(prices)
    market_gate = is_within_project_trading_window(calendar_path=DEFAULT_PROJECT_CALENDAR_PATH)
    data_context = build_run_data_context(
        prices=prices,
        freshness=data_freshness,
        market_gate=market_gate,
        calendar_path=DEFAULT_PROJECT_CALENDAR_PATH,
        run_context="hold_sensitivity_runner",
    ).as_dict()

    tradability_df = build_tradability_report(
        tickers=requested_tickers,
        prices=prices,
        enable_local_paper=False,
        enable_investopedia=False,
        dry_run=True,
    )
    active_tickers = apply_tradability_filter(
        tickers=requested_tickers,
        tradability_df=tradability_df,
        min_assets=10,
    )
    params = build_params(tickers=active_tickers)
    effective_cash_ticker = select_cash_proxy(active_tickers, tradability_df)
    if effective_cash_ticker is not None:
        params["cash_ticker"] = effective_cash_ticker
        params["effective_cash_ticker"] = effective_cash_ticker

    latest_prices_for_run = prices.reindex(columns=active_tickers).iloc[-1].astype(float)
    current_state = load_current_portfolio_state(
        params=params,
        active_tickers=active_tickers,
        latest_prices=latest_prices_for_run,
        cash_proxy_ticker=effective_cash_ticker,
        nav=float(PORTFOLIO_NAV_USD),
    )

    returns = compute_returns(prices.reindex(columns=active_tickers))
    as_of = pd.Timestamp(returns.index[-1])
    latest_prices_at_asof = prices.reindex(columns=active_tickers).loc[as_of]
    regime_result = detect_regime(prices=prices, date=prices.index[-1])
    risk_state = str(regime_result.get("risk_state", "normal"))

    forecast = build_forecast_3m(prices=prices, returns=returns, date=as_of, params=params, tickers=active_tickers)
    sigma = estimate_robust_covariance_at_date(
        returns=returns,
        date=as_of,
        window=int(params["cov_window"]),
        horizon_days=int(params["horizon_days"]),
        alpha=float(params["cov_shrink_alpha"]),
        jitter=float(params["cov_jitter"]),
    )

    direct_scenarios = build_3m_scenarios(
        forecast_table=forecast.table,
        covariance_3m=sigma,
        risk_state=risk_state,
        as_of=as_of,
    )

    factor_mode = "direct_only"
    factor_forecasts_df = pd.DataFrame()
    scenario_set = direct_scenarios
    try:
        macro_bundle = load_macro_proxy_data(prices=prices, date=as_of)
        factor_data_result = build_factor_data(macro_bundle.proxy_prices)
        factor_data_df = factor_data_result.factor_data.copy()
        factor_forecasts_df = build_factor_forecast(
            factor_data_df,
            date=as_of,
            risk_state=risk_state,
            horizon_days=int(params["horizon_days"]),
        )
        exposure_result = estimate_asset_factor_exposures(
            asset_returns=returns.reindex(columns=active_tickers),
            factor_returns=factor_data_df,
            date=as_of,
        )
        conditional_result = build_conditional_scenarios(
            direct_scenarios=direct_scenarios,
            factor_forecast_df=factor_forecasts_df,
            exposure_matrix=exposure_result.exposure_matrix,
            residual_volatility=exposure_result.residual_volatility,
        )
        factor_mode = conditional_result.mode
        scenario_set = ScenarioSet(
            as_of=as_of,
            scenario_returns_matrix=conditional_result.scenario_returns_matrix.copy(),
            scenario_names=list(conditional_result.scenario_probabilities.index),
            scenario_probabilities=conditional_result.scenario_probabilities.copy(),
            summary=conditional_result.summary.copy(),
            risk_state=risk_state,
        )
    except Exception:
        factor_mode = "direct_only"

    w_current = current_state.current_weights_proxy.reindex(active_tickers).fillna(0.0).astype(float)
    base_optimizer_result = optimize_allocation(
        mu=forecast.table["expected_return_3m"],
        Sigma=sigma,
        w_current=w_current,
        params={
            **params,
            "max_equity_like_total": params["max_equity_like_total_risk_off"] if risk_state == "risk_off" else params["max_equity_like_total_normal"],
            "min_defensive_weight": params["min_defensive_weight_risk_off"] if risk_state == "risk_off" else params["min_defensive_weight_normal"],
        },
    )
    model_governance = compute_model_confidence(
        forecast_report=forecast,
        factor_report=factor_forecasts_df,
        scenario_report=scenario_set.summary,
        optimizer_result=base_optimizer_result,
        data_quality_report={"global_data_quality_score": 0.5},
    )

    return BaseContext(
        params=params,
        active_tickers=active_tickers,
        prices=prices,
        returns=returns,
        as_of=as_of,
        data_context=data_context,
        risk_state=risk_state,
        forecast=forecast,
        sigma=sigma,
        scenario_set=scenario_set,
        factor_mode=factor_mode,
        factor_forecasts_df=factor_forecasts_df,
        current_state=current_state,
        latest_prices_at_asof=latest_prices_at_asof,
        w_current=w_current,
        model_governance=model_governance,
    )


def _evaluate_variant(
    base: BaseContext,
    *,
    variant_group: str,
    variant_name: str,
    notes: str = "",
    execution_buffer_override: float | None = None,
    model_buffer_override: float | None = None,
    transaction_cost_bps_override: float | None = None,
    asset_cap_scale: float = 1.0,
    p_hold_override: float | None = None,
    p_cash_override: float | None = None,
    min_order_value_override: float | None = None,
    max_candidates_override: int | None = None,
) -> dict[str, Any]:
    params = deepcopy(base.params)
    if transaction_cost_bps_override is not None:
        _apply_cost_bps(params, transaction_cost_bps_override)
    if asset_cap_scale != 1.0:
        _relax_asset_caps(params, asset_cap_scale)
    if p_hold_override is not None:
        params["p_hold_min"] = max(float(p_hold_override), 0.0)
    if p_cash_override is not None:
        params["p_cash_min"] = max(float(p_cash_override), 0.0)
    optimization_params = {
        **params,
        "max_equity_like_total": params["max_equity_like_total_risk_off"] if base.risk_state == "risk_off" else params["max_equity_like_total_normal"],
        "min_defensive_weight": params["min_defensive_weight_risk_off"] if base.risk_state == "risk_off" else params["min_defensive_weight_normal"],
    }

    optimizer_result = optimize_allocation(
        mu=base.forecast.table["expected_return_3m"],
        Sigma=base.sigma,
        w_current=base.w_current,
        params=optimization_params,
    )
    w_target = optimizer_result.target_weights
    conditional_factor_target = w_target if base.factor_mode == "conditional_factor" else None
    candidates = build_candidate_portfolios(
        w_current=base.w_current,
        w_target=w_target,
        forecast_table=base.forecast.table,
        params=params,
        conditional_factor_target=conditional_factor_target,
    )
    selection = select_robust_candidate(
        candidates=candidates,
        scenario_set=base.scenario_set,
        w_current=base.w_current,
        params=params,
        mode=base.factor_mode,
    )
    continuous_model_optimal_candidate = str(selection.scores_frame.iloc[0]["candidate"])
    expansion_sources = _select_discrete_expansion_sources(selection.scores_frame, continuous_model_optimal_candidate)

    max_candidates = int(max_candidates_override if max_candidates_override is not None else 25)
    min_order_value = float(min_order_value_override if min_order_value_override is not None else params.get("min_order_value_usd", 10.0))
    discrete_candidates = []
    hold_current_added = False
    for source_name in expansion_sources:
        source_target_weights = candidates[source_name].weights.reindex(base.active_tickers).fillna(0.0)
        generated = generate_discrete_candidates(
            target_weights=source_target_weights,
            latest_prices=base.latest_prices_at_asof,
            nav=float(base.current_state.nav),
            current_positions=base.current_state.current_shares.reindex(base.active_tickers).fillna(0.0),
            current_cash=float(base.current_state.current_cash),
            min_order_value=min_order_value,
            cash_buffer=float(params.get("cash_buffer_usd", 0.0)),
            max_candidates=max_candidates,
            allow_fractional_shares=False,
            marginal_priority=source_target_weights,
            cash_proxy_ticker=params.get("cash_ticker"),
        )
        for candidate in generated:
            if candidate.name == "HOLD_CURRENT":
                if hold_current_added:
                    continue
                hold_current_added = True
                candidate.metadata["continuous_source"] = "HOLD"
                candidate.metadata["continuous_target_weights"] = candidates["HOLD"].weights.reindex(base.active_tickers).fillna(0.0).copy()
            else:
                candidate.name = f"{source_name}::{candidate.name}"
                candidate.metadata["continuous_source"] = source_name
                candidate.metadata["continuous_target_weights"] = source_target_weights.copy()
            discrete_candidates.append(candidate)

    discrete_scored = score_discrete_candidates(
        discrete_candidates=discrete_candidates,
        scenario_returns=base.scenario_set,
        scorer_config={
            "params": params,
            "hold_weights": candidates["HOLD"].weights,
            "cash_weights": candidates["DEFENSIVE_CASH"].weights,
            "continuous_target": candidates[continuous_model_optimal_candidate].weights.reindex(base.active_tickers).fillna(0.0),
        },
        current_weights=base.w_current,
        current_shares=base.current_state.current_shares.reindex(base.active_tickers).fillna(0.0),
        current_cash=float(base.current_state.current_cash),
        latest_prices=base.latest_prices_at_asof,
        nav=float(base.current_state.nav),
    )
    discrete_selection = select_best_discrete_portfolio(discrete_scored)
    best_discrete_candidate_name = str(discrete_selection["best_discrete_candidate_name"])
    best_discrete_candidate = discrete_selection["candidate"]
    discrete_selected_score = discrete_scored["score_objects"][best_discrete_candidate_name]

    hold_rows = discrete_selection["scores_frame"].loc[
        discrete_selection["scores_frame"]["discrete_candidate"].astype(str) == "HOLD_CURRENT"
    ]
    hold_row = hold_rows.iloc[0] if not hold_rows.empty else None
    current_portfolio_score = _safe_float(hold_row["net_robust_score"], 0.0) if hold_row is not None else 0.0
    tail_risk_current = _safe_float(hold_row["cvar_5"], 0.0) if hold_row is not None else 0.0

    order_preview = build_discrete_order_preview(
        current_shares=base.current_state.current_shares.reindex(base.active_tickers).fillna(0.0),
        target_shares=best_discrete_candidate.shares.reindex(base.active_tickers).fillna(0.0),
        latest_prices=base.latest_prices_at_asof,
        nav=float(base.current_state.nav),
        min_order_value=min_order_value,
        not_executable=False,
        reason="",
    )
    order_preview, final_order_cost_summary = estimate_order_list_costs(
        order_preview_df=order_preview,
        latest_prices=base.latest_prices_at_asof,
        config={
            **params,
            "nav": float(base.current_state.nav),
            "current_cash": float(base.current_state.current_cash),
        },
    )

    discrete_selected_score.estimated_cost = float(final_order_cost_summary["total_order_cost_pct_nav"])
    discrete_selected_score.estimated_commission = float(final_order_cost_summary["total_estimated_commission"])
    discrete_selected_score.estimated_spread_cost = float(final_order_cost_summary["total_estimated_spread_cost"])
    discrete_selected_score.estimated_slippage_cost = float(final_order_cost_summary["total_estimated_slippage_cost"])
    discrete_selected_score.estimated_market_impact_cost = float(final_order_cost_summary["total_estimated_market_impact_cost"])
    discrete_selected_score.estimated_total_order_cost = float(final_order_cost_summary["total_estimated_transaction_cost"])
    discrete_selected_score.cost_bps_used = float(final_order_cost_summary["weighted_average_cost_bps"])
    discrete_selected_score.cost_model_used = str(final_order_cost_summary["cost_model_used"])
    discrete_selected_score.live_costs_available = bool(final_order_cost_summary["live_costs_available"])
    discrete_selected_score.net_robust_score = (
        discrete_selected_score.robust_score
        - discrete_selected_score.estimated_cost
        - discrete_selected_score.dynamic_buffer
    )
    delta_score_vs_current = float(discrete_selected_score.net_robust_score - current_portfolio_score)
    discrete_selected_score.delta_vs_hold = delta_score_vs_current

    model_governance = compute_model_confidence(
        forecast_report=base.forecast,
        factor_report=base.factor_forecasts_df,
        scenario_report=base.scenario_set.summary,
        optimizer_result=optimizer_result,
        data_quality_report={"global_data_quality_score": 0.968},
    )

    execution_buffer = float(execution_buffer_override if execution_buffer_override is not None else BASE_EXECUTION_BUFFER)
    model_buffer = float(model_buffer_override if model_buffer_override is not None else model_governance.get("model_uncertainty_buffer", base.model_governance.get("model_uncertainty_buffer", 0.001)))
    trade_edge_summary = compute_trade_now_edge(
        current_score=float(current_portfolio_score),
        target_score_after_costs=float(discrete_selected_score.net_robust_score),
        total_order_cost=float(discrete_selected_score.estimated_total_order_cost),
        execution_buffer=execution_buffer,
        model_uncertainty_buffer=model_buffer,
        other_penalties=0.0,
    )
    final_selection = type(
        "SelectionLike",
        (),
        {
            "selected_candidate": type(
                "DiscreteCandidateLike",
                (),
                {"name": best_discrete_candidate_name, "weights": discrete_selection["best_discrete_weights"]},
            )(),
            "selected_score": discrete_selected_score,
        },
    )()
    gate = evaluate_execution_gate(
        selection_result=final_selection,
        synthetic_data=bool(base.data_context.get("synthetic_data", False)),
        data_freshness_ok=bool(base.data_context.get("data_freshness_ok", False)),
        broker_state_reconciled=True,
        open_orders_exist=False,
        estimated_spread_cost=float(final_order_cost_summary["total_estimated_spread_cost"]) / max(float(base.current_state.nav), 1e-12),
        estimated_slippage=float(final_order_cost_summary["total_estimated_slippage_cost"]) / max(float(base.current_state.nav), 1e-12),
        estimated_transaction_cost=float(discrete_selected_score.estimated_cost),
        delta_vs_hold_is_net=True,
        costs_include_spread_slippage=True,
        execution_uncertainty_buffer=execution_buffer,
        model_uncertainty_buffer=model_buffer,
    )

    buy_count = int((order_preview["side"].astype(str) == "BUY").sum()) if not order_preview.empty else 0
    sell_count = int((order_preview["side"].astype(str) == "SELL").sum()) if not order_preview.empty else 0
    order_count = buy_count + sell_count
    changed_positions = _format_changed_positions(order_preview)
    final_action_theoretical = _theoretical_final_action(
        best_discrete_candidate_name=best_discrete_candidate_name,
        order_count=order_count,
        gate_status=str(gate.gate_status),
    )

    return {
        "variant_group": variant_group,
        "variant_name": variant_name,
        "notes": notes,
        "execution_buffer": execution_buffer,
        "model_uncertainty_buffer": model_buffer,
        "transaction_cost_bps": float(transaction_cost_bps_override if transaction_cost_bps_override is not None else _safe_float(params.get("default_bps_per_turnover"), 0.0)),
        "asset_cap_scale": asset_cap_scale,
        "p_current_threshold": float(params.get("p_hold_min", 0.55)),
        "p_cash_threshold": float(params.get("p_cash_min", 0.52)),
        "min_order_value_usd": min_order_value,
        "max_candidates": max_candidates,
        "continuous_candidate": continuous_model_optimal_candidate,
        "final_discrete_candidate": best_discrete_candidate_name,
        "final_action_theoretical": final_action_theoretical,
        "gate_status": str(gate.gate_status),
        "gate_reason": str(gate.reason),
        "trade_now_edge": float(trade_edge_summary["trade_now_edge"]),
        "delta_score_vs_current": delta_score_vs_current,
        "total_order_cost": float(final_order_cost_summary["total_estimated_transaction_cost"]),
        "probability_beats_current": float(discrete_selected_score.probability_beats_hold),
        "probability_beats_cash": float(discrete_selected_score.probability_beats_cash),
        "tail_risk_current": tail_risk_current,
        "tail_risk_target": float(discrete_selected_score.cvar_5),
        "order_count": order_count,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "changed_positions": changed_positions,
    }


def run_hold_sensitivity(output_dir: str | Path = OUTPUT_DIR) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base = _build_base_context()

    baseline_model_buffer = float(base.model_governance.get("model_uncertainty_buffer", 0.001))
    baseline_cost_bps = float(base.params.get("default_bps_per_turnover", 5.0))
    baseline_p_current = float(base.params.get("p_hold_min", 0.55))
    baseline_p_cash = float(base.params.get("p_cash_min", 0.52))
    baseline_min_order = float(base.params.get("min_order_value_usd", 10.0))

    rows: list[dict[str, Any]] = []

    for value in [0.0, 0.00025, 0.0005, BASE_EXECUTION_BUFFER, BASE_EXECUTION_BUFFER * 2.0]:
        rows.append(
            _evaluate_variant(
                base,
                variant_group="execution_buffer",
                variant_name=f"execution_buffer={value:.6f}",
                execution_buffer_override=value,
                notes="isolated execution buffer variation",
            )
        )

    for value in [0.0, 0.00025, 0.0005, baseline_model_buffer, baseline_model_buffer * 2.0]:
        rows.append(
            _evaluate_variant(
                base,
                variant_group="model_uncertainty_buffer",
                variant_name=f"model_uncertainty_buffer={value:.6f}",
                model_buffer_override=value,
                notes="isolated model uncertainty buffer variation",
            )
        )

    for value in [0.0, 2.0, baseline_cost_bps, 10.0]:
        rows.append(
            _evaluate_variant(
                base,
                variant_group="transaction_cost_bps",
                variant_name=f"transaction_cost_bps={value:.2f}",
                transaction_cost_bps_override=value,
                notes="scaled turnover proxy and modeled order-cost bps assumptions",
            )
        )

    for scale in [1.0, 1.25, 1.5]:
        rows.append(
            _evaluate_variant(
                base,
                variant_group="asset_max_weights",
                variant_name=f"asset_caps_x{scale:.2f}",
                asset_cap_scale=scale,
                notes="isolated per-asset max-weight relaxation; group limits unchanged",
            )
        )

    probability_variants = [
        ("probabilities_current", baseline_p_current, baseline_p_cash, "current thresholds"),
        ("p_current_minus_5pp", max(baseline_p_current - 0.05, 0.0), baseline_p_cash, "lower p_current only"),
        ("p_cash_minus_5pp", baseline_p_current, max(baseline_p_cash - 0.05, 0.0), "lower p_cash only"),
        ("both_minus_5pp", max(baseline_p_current - 0.05, 0.0), max(baseline_p_cash - 0.05, 0.0), "lower both probability thresholds"),
    ]
    for name, p_current, p_cash, note in probability_variants:
        rows.append(
            _evaluate_variant(
                base,
                variant_group="probability_thresholds",
                variant_name=name,
                p_hold_override=p_current,
                p_cash_override=p_cash,
                notes=note,
            )
        )

    rows.append(
        _evaluate_variant(
            base,
            variant_group="rounding_variants",
            variant_name="rounding_current",
            min_order_value_override=baseline_min_order,
            max_candidates_override=25,
            notes="current discrete generation settings",
        )
    )
    rows.append(
        _evaluate_variant(
            base,
            variant_group="rounding_variants",
            variant_name="smaller_min_order",
            min_order_value_override=1.0,
            max_candidates_override=25,
            notes="lower minimum order size only",
        )
    )
    rows.append(
        _evaluate_variant(
            base,
            variant_group="rounding_variants",
            variant_name="broader_variant_search",
            min_order_value_override=baseline_min_order,
            max_candidates_override=100,
            notes="repair-step count is not separately configurable; expanded candidate-search breadth instead",
        )
    )

    df = pd.DataFrame(rows)
    csv_path = output_path / "codex_hold_sensitivity.csv"
    df.to_csv(csv_path, index=False)

    buy_sell_rows = df[df["final_action_theoretical"].astype(str) == "BUY_SELL_PREVIEW"].copy()
    non_hold_rows = df[df["final_discrete_candidate"].astype(str) != "HOLD_CURRENT"].copy()
    strongest_hold_driver = "asset_max_constraints"
    if not non_hold_rows.empty:
        best_non_hold = non_hold_rows.sort_values("trade_now_edge", ascending=False).iloc[0]
        if str(best_non_hold["variant_group"]) in {"execution_buffer", "model_uncertainty_buffer"}:
            strongest_hold_driver = "buffer_stack_after_constraints"
    if not non_hold_rows.empty and (non_hold_rows["variant_group"] == "asset_max_weights").any():
        strongest_hold_driver = "asset_max_constraints"

    if buy_sell_rows.empty:
        buy_sell_answer = "Keine einzelne isolierte Stellschraube erzeugte in dieser Matrix bereits BUY/SELL_PREVIEW."
    else:
        first_trade = buy_sell_rows.sort_values(["variant_group", "trade_now_edge"], ascending=[True, False]).iloc[0]
        buy_sell_answer = (
            f"BUY/SELL_PREVIEW entsteht erstmals in dieser Matrix bei {first_trade['variant_name']} "
            f"(trade_now_edge={_safe_float(first_trade['trade_now_edge']):.6f}, changed_positions={first_trade['changed_positions']})."
        )

    asset_rows = df[df["variant_group"] == "asset_max_weights"].copy()
    execution_rows = df[df["variant_group"] == "execution_buffer"].copy()
    model_rows = df[df["variant_group"] == "model_uncertainty_buffer"].copy()
    cost_rows = df[df["variant_group"] == "transaction_cost_bps"].copy()
    prob_rows = df[df["variant_group"] == "probability_thresholds"].copy()
    rounding_rows = df[df["variant_group"] == "rounding_variants"].copy()

    baseline_rows = df[(df["variant_group"] == "rounding_variants") & (df["variant_name"] == "rounding_current")]
    baseline = baseline_rows.iloc[0] if not baseline_rows.empty else df.iloc[0]

    lines = [
        "# Codex Hold Sensitivity Report",
        "",
        "## Executive Summary",
        f"- Baseline theoretical final action: {baseline['final_action_theoretical']}",
        f"- Baseline final discrete candidate: {baseline['final_discrete_candidate']}",
        f"- Baseline trade_now_edge: {_safe_float(baseline['trade_now_edge']):.6f}",
        f"- Baseline order_count: {_safe_int(baseline['order_count'])}",
        f"- Strongest single HOLD driver in this matrix: {strongest_hold_driver}",
        f"- BUY/SELL finding: {buy_sell_answer}",
        "",
        "## Execution Buffer",
    ]
    for row in execution_rows.itertuples(index=False):
        lines.append(
            f"- {row.variant_name}: action={row.final_action_theoretical}, candidate={row.final_discrete_candidate}, trade_now_edge={_safe_float(row.trade_now_edge):.6f}, orders={_safe_int(row.order_count)}"
        )
    lines.extend(
        [
            "",
            "## Model Uncertainty Buffer",
        ]
    )
    for row in model_rows.itertuples(index=False):
        lines.append(
            f"- {row.variant_name}: action={row.final_action_theoretical}, candidate={row.final_discrete_candidate}, trade_now_edge={_safe_float(row.trade_now_edge):.6f}, orders={_safe_int(row.order_count)}"
        )
    lines.extend(
        [
            "",
            "## Transaction Costs",
        ]
    )
    for row in cost_rows.itertuples(index=False):
        lines.append(
            f"- {row.variant_name}: action={row.final_action_theoretical}, candidate={row.final_discrete_candidate}, trade_now_edge={_safe_float(row.trade_now_edge):.6f}, total_order_cost={_safe_float(row.total_order_cost):.2f} USD, orders={_safe_int(row.order_count)}"
        )
    lines.extend(
        [
            "",
            "## Asset-Max Relaxation",
        ]
    )
    for row in asset_rows.itertuples(index=False):
        lines.append(
            f"- {row.variant_name}: action={row.final_action_theoretical}, candidate={row.final_discrete_candidate}, trade_now_edge={_safe_float(row.trade_now_edge):.6f}, delta_score_vs_current={_safe_float(row.delta_score_vs_current):.6f}, changed_positions={row.changed_positions}"
        )
    lines.extend(
        [
            "",
            "## Probability Thresholds",
        ]
    )
    for row in prob_rows.itertuples(index=False):
        lines.append(
            f"- {row.variant_name}: action={row.final_action_theoretical}, candidate={row.final_discrete_candidate}, p_current={_safe_float(row.p_current_threshold):.2%}, p_cash={_safe_float(row.p_cash_threshold):.2%}, orders={_safe_int(row.order_count)}"
        )
    lines.extend(
        [
            "",
            "## Rounding / Search Variants",
        ]
    )
    for row in rounding_rows.itertuples(index=False):
        lines.append(
            f"- {row.variant_name}: action={row.final_action_theoretical}, candidate={row.final_discrete_candidate}, trade_now_edge={_safe_float(row.trade_now_edge):.6f}, changed_positions={row.changed_positions}, note={row.notes}"
        )

    result_bucket = "korrekt konservatives HOLD"
    if not buy_sell_rows.empty:
        result_bucket = "Constraint-Problem plus zu strenge Gates"
    elif not asset_rows.empty and (asset_rows["final_discrete_candidate"].astype(str) != "HOLD_CURRENT").any():
        result_bucket = "Constraint-Problem"
    elif (execution_rows["final_action_theoretical"].astype(str) != "HOLD").any() or (model_rows["final_action_theoretical"].astype(str) != "HOLD").any():
        result_bucket = "zu strenge Gates"
    elif (rounding_rows["final_discrete_candidate"].astype(str) != "HOLD_CURRENT").any():
        result_bucket = "Solver-/Rounding-Problem"
    else:
        result_bucket = "schwache Alpha-Signale plus konservative Constraints"

    lines.extend(
        [
            "",
            "## Answers",
            f"1. Welche einzelne Stellschraube macht HOLD am staerksten wahrscheinlich? -> {strongest_hold_driver}.",
            f"2. Ab welcher Parameteraenderung entstehen BUY/SELL? -> {buy_sell_answer}",
            "3. Sind diese Parameteraenderungen finanzlogisch vertretbar? -> Eine moderate Lockerung der Asset-Caps oder der absoluten Huerden ist eher vertretbar als ein kompletter Kosten-/Buffer-Reset auf Null. Reine Null-Puffer-/Null-Kosten-Faelle sind Diagnosefaelle, keine sinnvolle Produktionskalibrierung.",
            f"4. Gesamtbild -> {result_bucket}.",
        ]
    )

    report_path = output_path / "codex_hold_sensitivity_report.md"
    report_path.write_text(sanitize_for_output("\n".join(lines).strip() + "\n"), encoding="utf-8")
    return csv_path, report_path


if __name__ == "__main__":
    csv_path, report_path = run_hold_sensitivity()
    print(csv_path)
    print(report_path)
