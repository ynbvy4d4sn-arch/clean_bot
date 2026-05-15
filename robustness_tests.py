"""Minimal robustness checks for the dry-run allocator stack."""

from __future__ import annotations

from datetime import datetime
from itertools import product
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from asset_universe import get_asset_max_weights, get_enabled_tickers, get_group_limits, get_group_map
from candidate_factory import CandidatePortfolio, build_candidate_portfolios, repair_weights_to_constraints
from codex_report import build_codex_debug_report, write_codex_debug_report, write_codex_next_prompt
from config import build_params, get_email_gate_status
import data as data_module
import notifications as notifications_module
import daily_review_rendering as daily_review_rendering_module
from data import build_run_data_context, check_data_freshness, load_price_data, write_data_freshness_report
from data_quality import compute_data_quality_report
from daily_analysis_report import (
    build_daily_analysis_report,
    build_daily_analysis_email_body,
    send_daily_analysis_email_if_needed,
    should_send_after_local_time,
    should_send_daily_analysis_email,
    write_daily_analysis_report,
)
from daily_bot import (
    _annotate_final_daily_preview,
    _apply_execution_fraction,
    _build_active_preview_files,
    _build_hold_vs_target_analysis,
    _build_manual_simulator_order_outputs,
    _build_constraint_pressure_reports,
    _correlation_label,
    _diversification_label,
    _select_active_preview_candidate,
    _select_discrete_expansion_sources,
    _write_scenario_weighted_solver_reports,
    _write_rebalance_sensitivity_matrix,
    compute_decision_id,
    compute_order_signature,
    reset_state_periods_if_needed,
    update_state_after_execution,
)
from daily_portfolio_review import send_daily_review_email_if_needed, write_daily_portfolio_review_outputs
from daily_portfolio_review import build_daily_email_briefing, build_decision_fingerprint, build_review_issues
from diagnostics import (
    create_run_diagnostics,
    log_error,
    log_final_action,
    log_data_context,
    log_rejected_order,
    log_warning,
    write_run_diagnostics,
)
from execution_gate import evaluate_execution_gate
from feasibility import check_portfolio_feasibility
from investopedia_adapter import InvestopediaSimulatorAdapter
from model_governance import compute_model_confidence
from notifications import sanitize_for_output, send_email_notification
from optimizer import build_feasible_initial_weights, optimize_allocation, optimize_scenario_sharpe_allocation
from order_sizing import convert_weights_to_orders
from order_preview import mark_daily_simulator_preview, mark_research_preview
from paper_broker_stub import initialize_paper_account
from discrete_portfolio_optimizer import (
    _finalize_candidate,
    build_discrete_order_preview,
    generate_discrete_candidates,
    load_current_portfolio_state,
    score_discrete_candidates,
    select_best_discrete_portfolio,
    validate_portfolio_constraints,
    write_current_portfolio_report,
)
from reconciliation import detect_open_orders, reconcile_before_execution
from report import write_latest_decision_report, write_output_file_guide
from risk_free import risk_free_return_for_horizon
from robust_scorer import evaluate_candidate as evaluate_robust_candidate, select_robust_candidate
from scenario_model import ScenarioSet
from scenario_risk_model import (
    ScenarioRiskDistribution,
    build_candidate_risk_return_frame,
    build_scenario_risk_distribution,
    compute_baseline_covariance,
    evaluate_portfolio_scenario_mixture,
)
from scenarios import SCENARIO_NAMES, build_scenario_inputs
from scenario_weighted_solver import (
    ScenarioInput as WeightedScenarioInput,
    SolverConfig as WeightedSolverConfig,
    SolverResult as WeightedSolverResult,
    evaluate_weights as evaluate_scenario_weighted_weights,
    solve_scenario_weighted_sharpe,
    validate_solver_result as validate_weighted_solver_result,
)
from trade_sizing import compute_trade_fraction
from transaction_costs import build_transaction_cost_review_summary, estimate_order_cost, estimate_order_list_costs
from tradability import apply_tradability_filter, select_cash_proxy


def _test_row(name: str, status: str, message: str) -> dict[str, str]:
    return {"test_name": name, "status": status, "message": message}


def _make_test_scenario_set(
    tickers: list[str],
    scenario_rows: list[list[float]],
    probabilities: list[float] | None = None,
    *,
    names: list[str] | None = None,
) -> ScenarioSet:
    scenario_names = names or [f"s{i}" for i in range(len(scenario_rows))]
    matrix = pd.DataFrame(
        scenario_rows,
        index=scenario_names,
        columns=pd.Index(tickers, name="ticker"),
        dtype=float,
    )
    probs = probabilities or [1.0 / len(scenario_rows)] * len(scenario_rows)
    probability_series = pd.Series(probs, index=scenario_names, dtype=float)
    summary = pd.DataFrame(
        {
            "scenario_name": scenario_names,
            "probability": probability_series.values,
            "mean_asset_return": matrix.mean(axis=1).values,
            "median_asset_return": matrix.median(axis=1).values,
        }
    )
    return ScenarioSet(
        as_of=pd.Timestamp("2026-05-07"),
        scenario_returns_matrix=matrix,
        scenario_names=scenario_names,
        scenario_probabilities=probability_series,
        summary=summary,
        risk_state="normal",
    )


def _make_test_optimizer_params(
    tickers: list[str],
    *,
    cash_ticker: str | None = None,
    asset_max_weights: dict[str, float] | None = None,
    group_map: dict[str, str] | None = None,
    group_limits: dict[str, float] | None = None,
    risk_aversion: float = 0.0,
    turnover_penalty: float = 0.0,
    concentration_penalty: float = 0.0,
    min_cash_weight: float = 0.0,
    max_turnover: float = 2.0,
    cost_rate: float = 0.0,
    base_buffer: float = 0.0,
    vol_buffer_multiplier: float = 0.0,
    hurdle: float = 0.0,
    risk_premium_hurdle: float = 0.0,
    p_hold_min: float = 0.0,
    p_cash_min: float = 0.0,
    min_order_value_usd: float = 0.0,
    default_spread_bps: float = 0.0,
    default_slippage_bps: float = 0.0,
    default_market_impact_bps: float = 0.0,
    default_commission_per_trade_usd: float = 0.0,
    allow_fractional_shares: bool = False,
    max_equity_like_total: float = 1.0,
    min_defensive_weight: float = 0.0,
) -> dict[str, object]:
    resolved_group_map = dict(group_map or {})
    if not resolved_group_map:
        for ticker in tickers:
            if cash_ticker is not None and ticker == cash_ticker:
                resolved_group_map[ticker] = "cash"
            else:
                resolved_group_map[ticker] = "risk"
    resolved_group_limits = dict(group_limits or {})
    if not resolved_group_limits:
        for group in set(resolved_group_map.values()):
            resolved_group_limits[str(group)] = 1.0
    resolved_asset_caps = {ticker: 1.0 for ticker in tickers}
    resolved_asset_caps.update(asset_max_weights or {})
    equity_groups = [group for group in set(resolved_group_map.values()) if group != "cash"]
    defensive_groups = ["cash"] if cash_ticker is not None else []
    return {
        "asset_max_weights": resolved_asset_caps,
        "group_map": resolved_group_map,
        "group_limits": resolved_group_limits,
        "cash_ticker": cash_ticker,
        "risk_aversion": float(risk_aversion),
        "turnover_penalty": float(turnover_penalty),
        "concentration_penalty": float(concentration_penalty),
        "min_cash_weight": float(min_cash_weight),
        "max_turnover": float(max_turnover),
        "cost_rate": float(cost_rate),
        "base_buffer": float(base_buffer),
        "vol_buffer_multiplier": float(vol_buffer_multiplier),
        "hurdle": float(hurdle),
        "risk_premium_hurdle": float(risk_premium_hurdle),
        "p_hold_min": float(p_hold_min),
        "p_cash_min": float(p_cash_min),
        "min_order_value_usd": float(min_order_value_usd),
        "default_spread_bps": float(default_spread_bps),
        "default_slippage_bps": float(default_slippage_bps),
        "default_market_impact_bps": float(default_market_impact_bps),
        "default_commission_per_trade_usd": float(default_commission_per_trade_usd),
        "allow_fractional_shares": bool(allow_fractional_shares),
        "equity_like_groups": equity_groups,
        "defensive_groups": defensive_groups,
        "max_equity_like_total": float(max_equity_like_total),
        "min_defensive_weight": float(min_defensive_weight),
        "portfolio_nav_usd": 0.0,
    }


def _proxy_weights_from_state(
    *,
    shares: pd.Series,
    prices: pd.Series,
    nav: float,
    cash_proxy_ticker: str | None,
    current_cash: float = 0.0,
) -> pd.Series:
    values = shares.reindex(prices.index).fillna(0.0).astype(float) * prices.astype(float)
    actual = (values / float(nav)).fillna(0.0) if nav > 0 else values * 0.0
    proxy = actual.copy()
    if cash_proxy_ticker is not None and cash_proxy_ticker in proxy.index:
        proxy.loc[cash_proxy_ticker] += max(float(current_cash), 0.0) / float(nav)
    total = float(proxy.sum())
    if total > 0.0:
        proxy = proxy / total
    return proxy.reindex(prices.index).fillna(0.0).astype(float)


def _cash_reference_weights(index: pd.Index, cash_proxy_ticker: str | None) -> pd.Series:
    weights = pd.Series(0.0, index=index, dtype=float)
    if cash_proxy_ticker is not None and cash_proxy_ticker in weights.index:
        weights.loc[cash_proxy_ticker] = 1.0
    elif len(weights) > 0:
        weights.iloc[0] = 1.0
    return weights


def _make_scenario_input_fixture() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.Index]:
    assets = pd.Index(["SGOV", "SPMO", "PDBC", "GLD", "IEF", "XLK"], name="ticker")
    dates = pd.date_range("2026-01-01", periods=80, freq="B")
    phase = np.linspace(0.0, 6.0, len(dates))
    returns = pd.DataFrame(
        {
            "SGOV": np.full(len(dates), 0.00005),
            "SPMO": 0.00020 + np.sin(phase) * 0.0030,
            "PDBC": 0.00010 + np.cos(phase) * 0.0025,
            "GLD": 0.00008 + np.sin(phase * 0.7 + 0.4) * 0.0018,
            "IEF": 0.00006 - np.sin(phase * 0.5) * 0.0012,
            "XLK": 0.00025 + np.sin(phase + 0.8) * 0.0035,
        },
        index=dates,
    )
    forecast = pd.DataFrame(
        {
            "expected_return_3m": [0.0030, 0.0220, 0.0120, 0.0090, 0.0050, 0.0240],
            "volatility_3m": [0.0020, 0.0600, 0.0450, 0.0350, 0.0250, 0.0700],
            "momentum_score": [0.05, 0.80, 0.35, 0.15, -0.10, 0.90],
            "signal_confidence": [0.60, 0.85, 0.70, 0.65, 0.55, 0.90],
        },
        index=assets,
    )
    params: dict[str, object] = {
        "horizon_days": 63,
        "effective_horizon_days": 63,
        "default_forecast_horizon_days": 63,
        "risk_free_rate_annual": 0.02,
        "scenario_covariance_lookback": 60,
        "scenario_covariance_shrink_alpha": 0.75,
        "cov_jitter": 1.0e-10,
    }
    return forecast, returns, params, assets


def _score_and_select_discrete_fixture(
    *,
    candidates: list,
    scenario_set: ScenarioSet,
    params: dict[str, object],
    current_shares: pd.Series,
    latest_prices: pd.Series,
    nav: float,
    current_cash: float,
    target_weights: pd.Series,
    cash_proxy_ticker: str | None,
) -> dict[str, object]:
    current_weights = _proxy_weights_from_state(
        shares=current_shares,
        prices=latest_prices,
        nav=nav,
        cash_proxy_ticker=cash_proxy_ticker,
        current_cash=current_cash,
    )
    scorer_config = {
        "params": {
            **params,
            "portfolio_nav_usd": float(nav),
        },
        "hold_weights": current_weights,
        "cash_weights": _cash_reference_weights(current_weights.index, cash_proxy_ticker),
        "continuous_target": target_weights.reindex(current_weights.index).fillna(0.0),
    }
    scored = score_discrete_candidates(
        candidates,
        scenario_set,
        scorer_config,
        current_weights=current_weights,
        transaction_cost_config=params,
        current_shares=current_shares,
        current_cash=float(current_cash),
        latest_prices=latest_prices,
        nav=float(nav),
    )
    selected = select_best_discrete_portfolio(scored)
    return {
        "scored": scored,
        "selected": selected,
        "current_weights": current_weights,
    }


def _bruteforce_discrete_candidates(
    *,
    prices: pd.Series,
    nav: float,
    current_shares: pd.Series,
    min_order_value: float,
    cash_proxy_ticker: str | None,
    max_shares_by_ticker: dict[str, int] | None = None,
) -> list:
    index = pd.Index([str(t) for t in prices.index], name="ticker")
    price_series = prices.reindex(index).astype(float)
    current_positions = current_shares.reindex(index).fillna(0.0).astype(float)
    share_ranges: list[range] = []
    for ticker in index:
        explicit_cap = None if max_shares_by_ticker is None else max_shares_by_ticker.get(str(ticker))
        if explicit_cap is None:
            explicit_cap = int(np.floor(float(nav) / max(float(price_series.loc[ticker]), 1e-12)))
        share_ranges.append(range(int(explicit_cap) + 1))
    candidates = []
    for combo in product(*share_ranges):
        shares = pd.Series(combo, index=index, dtype=float)
        candidate = _finalize_candidate(
            name=f"BRUTE_FORCE_{len(candidates)}",
            shares=shares,
            latest_prices=price_series,
            nav=float(nav),
            cash_proxy_ticker=cash_proxy_ticker,
            current_positions=current_positions,
            min_order_value=float(min_order_value),
            metadata={"kind": "bruteforce"},
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def run_robustness_tests() -> pd.DataFrame:
    """Run lightweight robustness checks without network access."""

    rows: list[dict[str, str]] = []
    params = build_params(tickers=["SGOV", "IEF", "XLK", "XLI", "TIP", "PDBC", "SPHQ", "SPLV", "AGG", "LQD"])

    try:
        build_params()
        rows.append(_test_row("missing_env_no_crash", "PASS", "build_params succeeded without .env"))
    except Exception as exc:
        rows.append(_test_row("missing_env_no_crash", "FAIL", str(exc)))

    try:
        daily_source = (Path(__file__).resolve().parent / "daily_bot.py").read_text(encoding="utf-8")
        top_level = daily_source.split("LOGGER = logging.getLogger", maxsplit=1)[0]
        legacy_imports_absent = all(
            forbidden not in top_level
            for forbidden in [
                "build_candidate_portfolios",
                "select_robust_candidate",
                "generate_discrete_candidates",
                "score_discrete_candidates",
                "select_best_discrete_portfolio",
                "optimize_allocation",
            ]
        )
        slim_return = daily_source.find("return _finalize_slim_scenario_daily_run(")
        legacy_fence = daily_source.find("Legacy candidate/discrete selection is disabled")
        candidate_block = daily_source.find('log_stage(diagnostics, "candidate construction", "START")')
        legacy_path_blocked = slim_return > 0 and legacy_fence > slim_return and candidate_block > legacy_fence
        final_sources_locked = (
            'FINAL_TARGET_SOURCE_SCENARIO = "SCENARIO_WEIGHTED_RF_SHARPE_OPTIMAL"' in daily_source
            and 'FINAL_TARGET_SOURCE_SOLVER_FAILED = "HOLD_SOLVER_FAILED"' in daily_source
        )
        status = "PASS" if legacy_imports_absent and legacy_path_blocked and final_sources_locked else "FAIL"
        message = (
            f"legacy_imports_absent={legacy_imports_absent}; "
            f"legacy_path_blocked={legacy_path_blocked}; "
            f"final_sources_locked={final_sources_locked}"
        )
        rows.append(
            _test_row(
                "daily_bot_does_not_import_legacy_candidate_decision_in_active_path",
                status,
                message,
            )
        )
        rows.append(
            _test_row(
                "test_daily_bot_does_not_use_legacy_candidate_decision",
                status,
                message,
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_bot_does_not_import_legacy_candidate_decision_in_active_path", "FAIL", str(exc)))
        rows.append(_test_row("test_daily_bot_does_not_use_legacy_candidate_decision", "FAIL", str(exc)))

    try:
        requested_tickers = {
            "XLY",
            "XLE",
            "XLF",
            "XLRE",
            "XLB",
            "SH",
            "VEA",
            "VWO",
            "RPV",
            "SIZE",
            "TLT",
            "HYG",
            "EMB",
            "VBR",
        }
        enabled = set(get_enabled_tickers())
        group_map = get_group_map()
        group_limits = get_group_limits()
        asset_caps = get_asset_max_weights()
        missing = sorted(requested_tickers - enabled)
        missing_groups = sorted({group_map[ticker] for ticker in requested_tickers if ticker in group_map} - set(group_limits))
        missing_caps = sorted(ticker for ticker in requested_tickers if float(asset_caps.get(ticker, 0.0)) <= 0.0)
        rows.append(
            _test_row(
                "requested_expanded_universe_tickers_enabled",
                "PASS" if not missing and not missing_groups and not missing_caps else "FAIL",
                f"missing={missing}; missing_groups={missing_groups}; missing_caps={missing_caps}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("requested_expanded_universe_tickers_enabled", "FAIL", str(exc)))

    try:
        gate_status = get_email_gate_status(
            {
                "ENABLE_EMAIL_NOTIFICATIONS": False,
                "EMAIL_SEND_ENABLED": False,
                "EMAIL_DRY_RUN": True,
                "EMAIL_RECIPIENT": "",
                "USER_CONFIRMED_EMAIL_PHASE": False,
                "PHASE": "DAILY_REVIEW_PREVIEW",
                "ENABLE_EXTERNAL_BROKER": False,
                "ENABLE_INVESTOPEDIA_SIMULATOR": False,
                "ENABLE_LOCAL_PAPER_TRADING": False,
            }
        )
        rows.append(
            _test_row(
                "email_gate_default_config_is_preview_only",
                "PASS"
                if (
                    not gate_status["real_email_send_allowed"]
                    and gate_status["reason"] == "preview_only"
                    and "EMAIL_DRY_RUN=true" in list(gate_status["blockers"])
                )
                else "FAIL",
                str(gate_status),
            )
        )
    except Exception as exc:
        rows.append(_test_row("email_gate_default_config_is_preview_only", "FAIL", str(exc)))

    try:
        gate_status = get_email_gate_status(
            {
                "ENABLE_EMAIL_NOTIFICATIONS": True,
                "EMAIL_SEND_ENABLED": True,
                "EMAIL_DRY_RUN": False,
                "EMAIL_RECIPIENT": "test@example.com",
                "USER_CONFIRMED_EMAIL_PHASE": True,
                "PHASE": "DAILY_REVIEW_SEND_READY",
                "ENABLE_EXTERNAL_BROKER": True,
                "ENABLE_INVESTOPEDIA_SIMULATOR": False,
                "ENABLE_LOCAL_PAPER_TRADING": False,
            }
        )
        rows.append(
            _test_row(
                "email_gate_external_broker_blocks_send",
                "PASS"
                if (
                    not gate_status["real_email_send_allowed"]
                    and gate_status["reason"] == "blocked_by_gate"
                    and "ENABLE_EXTERNAL_BROKER must remain false" in list(gate_status["blockers"])
                )
                else "FAIL",
                str(gate_status),
            )
        )
    except Exception as exc:
        rows.append(_test_row("email_gate_external_broker_blocks_send", "FAIL", str(exc)))

    try:
        gate_status = get_email_gate_status(
            {
                "ENABLE_EMAIL_NOTIFICATIONS": True,
                "EMAIL_SEND_ENABLED": True,
                "EMAIL_DRY_RUN": False,
                "EMAIL_RECIPIENT": "test@example.com",
                "USER_CONFIRMED_EMAIL_PHASE": True,
                "PHASE": "DAILY_REVIEW_SEND_READY",
                "ENABLE_EXTERNAL_BROKER": False,
                "ENABLE_INVESTOPEDIA_SIMULATOR": True,
                "ENABLE_LOCAL_PAPER_TRADING": False,
            }
        )
        rows.append(
            _test_row(
                "email_gate_investopedia_blocks_send",
                "PASS"
                if (
                    not gate_status["real_email_send_allowed"]
                    and gate_status["reason"] == "blocked_by_gate"
                    and "ENABLE_INVESTOPEDIA_SIMULATOR must remain false" in list(gate_status["blockers"])
                )
                else "FAIL",
                str(gate_status),
            )
        )
    except Exception as exc:
        rows.append(_test_row("email_gate_investopedia_blocks_send", "FAIL", str(exc)))

    try:
        gate_status = get_email_gate_status(
            {
                "ENABLE_EMAIL_NOTIFICATIONS": True,
                "EMAIL_SEND_ENABLED": True,
                "EMAIL_DRY_RUN": True,
                "EMAIL_RECIPIENT": "test@example.com",
                "USER_CONFIRMED_EMAIL_PHASE": True,
                "PHASE": "DAILY_REVIEW_SEND_READY",
                "ENABLE_EXTERNAL_BROKER": False,
                "ENABLE_INVESTOPEDIA_SIMULATOR": False,
                "ENABLE_LOCAL_PAPER_TRADING": False,
            }
        )
        rows.append(
            _test_row(
                "email_gate_dry_run_blocks_real_send",
                "PASS"
                if (
                    not gate_status["real_email_send_allowed"]
                    and gate_status["reason"] == "preview_only"
                    and "EMAIL_DRY_RUN=true" in list(gate_status["blockers"])
                )
                else "FAIL",
                str(gate_status),
            )
        )
    except Exception as exc:
        rows.append(_test_row("email_gate_dry_run_blocks_real_send", "FAIL", str(exc)))

    try:
        scores_frame = pd.DataFrame(
            {
                "candidate": [
                    "MOMENTUM_TILT_SIMPLE",
                    "OPTIMIZER_TARGET",
                    "CONDITIONAL_FACTOR_TARGET",
                    "PARTIAL_50",
                    "PARTIAL_25",
                    "HOLD",
                    "DEFENSIVE_CASH",
                ]
            }
        )
        expansion_sources = _select_discrete_expansion_sources(
            scores_frame,
            "MOMENTUM_TILT_SIMPLE",
        )
        rows.append(
            _test_row(
                "discrete_expansion_includes_all_non_hold_continuous_candidates",
                "PASS"
                if expansion_sources
                == [
                    "MOMENTUM_TILT_SIMPLE",
                    "OPTIMIZER_TARGET",
                    "CONDITIONAL_FACTOR_TARGET",
                    "PARTIAL_50",
                    "PARTIAL_25",
                ]
                else "FAIL",
                str(expansion_sources),
            )
        )
    except Exception as exc:
        rows.append(
            _test_row(
                "discrete_expansion_includes_all_non_hold_continuous_candidates",
                "FAIL",
                str(exc),
            )
        )

    try:
        params_constraints = _make_test_optimizer_params(
            ["SGOV", "SPHQ", "SPLV", "SPMO", "PDBC"],
            cash_ticker="SGOV",
            asset_max_weights={"SGOV": 0.70, "SPHQ": 0.15, "SPLV": 0.15, "SPMO": 0.15, "PDBC": 0.10},
            group_map={"SGOV": "cash", "SPHQ": "factor", "SPLV": "factor", "SPMO": "factor", "PDBC": "commodities"},
            group_limits={"cash": 0.70, "factor": 0.30, "commodities": 0.25},
        )
        current_weights = pd.Series({"SGOV": 0.50, "SPHQ": 0.11, "SPLV": 0.10, "SPMO": 0.0932, "PDBC": 0.1968}, dtype=float)
        validation = validate_portfolio_constraints(
            weights_actual=current_weights,
            cash_weight=0.0,
            params=params_constraints,
            index=pd.Index(current_weights.index, name="ticker"),
            label="Current portfolio",
        )
        rows.append(
            _test_row(
                "hold_current_constraint_violation_reported",
                "PASS"
                if not bool(validation["ok"])
                and bool(validation["group_limit_violations"])
                and any(item.get("group") == "factor" for item in validation["group_limit_violations"])
                else "FAIL",
                str(validation),
            )
        )
    except Exception as exc:
        rows.append(_test_row("hold_current_constraint_violation_reported", "FAIL", str(exc)))

    try:
        candidate_map = {
            "HOLD_CURRENT": type("CandidateLike", (), {"weights_proxy": pd.Series(dtype=float), "weights_actual": pd.Series(dtype=float), "shares": pd.Series(dtype=float), "cash_left": 1000.0})(),
            "MOMENTUM_TILT_SIMPLE::FLOOR_BASE_0": type("CandidateLike", (), {"weights_proxy": pd.Series(dtype=float), "weights_actual": pd.Series(dtype=float), "shares": pd.Series(dtype=float), "cash_left": 50.0})(),
        }
        scores_frame = pd.DataFrame(
            [
                {
                    "discrete_candidate": "MOMENTUM_TILT_SIMPLE::FLOOR_BASE_0",
                    "net_robust_score": 0.0009,
                    "cvar_5": -0.02,
                    "turnover_vs_current": 0.20,
                    "max_abs_weight_drift": 0.01,
                    "number_of_positions": 4,
                    "cash_left": 50.0,
                    "valid_constraints": False,
                    "validation_errors": "Target weights violate individual asset max-weight limits.",
                    "delta_vs_cash": 0.0009,
                    "probability_beats_hold": 0.80,
                    "probability_beats_cash": 0.80,
                },
                {
                    "discrete_candidate": "HOLD_CURRENT",
                    "net_robust_score": 0.0002,
                    "cvar_5": -0.01,
                    "turnover_vs_current": 0.00,
                    "max_abs_weight_drift": 0.00,
                    "number_of_positions": 0,
                    "cash_left": 1000.0,
                    "valid_constraints": True,
                    "validation_errors": "",
                    "delta_vs_cash": 0.0002,
                    "probability_beats_hold": 1.00,
                    "probability_beats_cash": 1.00,
                },
            ]
        )
        selection = select_best_discrete_portfolio(
            {
                "scores_frame": scores_frame,
                "candidate_map": candidate_map,
                "selection_config": {"hurdle": 0.001, "risk_premium_hurdle": 0.0005, "p_hold_min": 0.55, "p_cash_min": 0.52},
            }
        )
        rows.append(
            _test_row(
                "safe_hold_fallback_not_reported_as_optimal",
                "PASS"
                if selection["best_discrete_candidate_name"] == "HOLD_CURRENT"
                and selection["selected_reason"] == "safe_hold_fallback_no_valid_trade_candidate"
                else "FAIL",
                str({k: selection.get(k) for k in ["best_discrete_candidate_name", "selected_reason"]}),
            )
        )
    except Exception as exc:
        rows.append(_test_row("safe_hold_fallback_not_reported_as_optimal", "FAIL", str(exc)))

    try:
        params_constraints = _make_test_optimizer_params(
            ["SGOV", "SPMO", "PDBC", "SPHQ", "SPLV"],
            cash_ticker="SGOV",
            asset_max_weights={"SGOV": 0.70, "SPMO": 0.15, "PDBC": 0.10, "SPHQ": 0.15, "SPLV": 0.15},
            group_map={"SGOV": "cash", "SPMO": "factor", "PDBC": "commodities", "SPHQ": "factor", "SPLV": "factor"},
            group_limits={"cash": 0.70, "factor": 0.30, "commodities": 0.25},
        )
        raw = pd.Series({"SGOV": 0.60, "SPMO": 0.1708, "PDBC": 0.1208, "SPHQ": 0.07, "SPLV": 0.0384}, dtype=float)
        repaired = repair_weights_to_constraints(raw, params_constraints)
        weights = pd.Series(repaired["weights"], dtype=float)
        rows.append(
            _test_row(
                "momentum_repair_respects_asset_caps",
                "PASS"
                if float(weights["SPMO"]) <= 0.15000001
                and float(weights["PDBC"]) <= 0.10000001
                and bool(repaired["valid"])
                else "FAIL",
                weights.to_dict().__repr__(),
            )
        )
    except Exception as exc:
        rows.append(_test_row("momentum_repair_respects_asset_caps", "FAIL", str(exc)))

    try:
        tickers = ["SGOV", "SPMO", "PDBC", "SPHQ", "SPLV"]
        params_constraints = _make_test_optimizer_params(
            tickers,
            cash_ticker="SGOV",
            asset_max_weights={"SGOV": 0.70, "SPMO": 0.15, "PDBC": 0.10, "SPHQ": 0.15, "SPLV": 0.15},
            group_map={"SGOV": "cash", "SPMO": "factor", "PDBC": "commodities", "SPHQ": "factor", "SPLV": "factor"},
            group_limits={"cash": 0.70, "factor": 0.30, "commodities": 0.25},
        )
        current = pd.Series({"SGOV": 0.50, "SPMO": 0.10, "PDBC": 0.05, "SPHQ": 0.10, "SPLV": 0.25}, dtype=float)
        target = pd.Series({"SGOV": 0.20, "SPMO": 0.30, "PDBC": 0.20, "SPHQ": 0.15, "SPLV": 0.15}, dtype=float)
        forecast = pd.DataFrame(
            {
                "expected_return_3m": [0.0, 0.10, 0.08, 0.06, 0.05],
                "signal_confidence": [0.1, 1.0, 1.0, 0.8, 0.8],
            },
            index=tickers,
        )
        candidates = build_candidate_portfolios(current, target, forecast, params_constraints)
        momentum = candidates["MOMENTUM_TILT_SIMPLE"].weights
        validation = validate_portfolio_constraints(
            weights_actual=momentum,
            cash_weight=0.0,
            params=params_constraints,
            index=pd.Index(tickers, name="ticker"),
            label="Momentum candidate",
        )
        rows.append(
            _test_row(
                "fallback_residual_respects_asset_and_group_caps",
                "PASS" if bool(validation["ok"]) else "FAIL",
                str({"weights": momentum.to_dict(), "validation": validation}),
            )
        )
    except Exception as exc:
        rows.append(_test_row("fallback_residual_respects_asset_and_group_caps", "FAIL", str(exc)))

    try:
        tickers = ["SGOV", "SPHQ", "SPLV", "SPMO", "PDBC"]
        params_constraints = _make_test_optimizer_params(
            tickers,
            cash_ticker="SGOV",
            asset_max_weights={"SGOV": 0.70, "SPHQ": 0.15, "SPLV": 0.15, "SPMO": 0.15, "PDBC": 0.10},
            group_map={"SGOV": "cash", "SPHQ": "factor", "SPLV": "factor", "SPMO": "factor", "PDBC": "commodities"},
            group_limits={"cash": 0.70, "factor": 0.30, "commodities": 0.25},
        )
        current = pd.Series({"SGOV": 0.55, "SPHQ": 0.11, "SPLV": 0.10, "SPMO": 0.0932, "PDBC": 0.1468}, dtype=float)
        target = pd.Series({"SGOV": 0.60, "SPHQ": 0.10, "SPLV": 0.10, "SPMO": 0.10, "PDBC": 0.10}, dtype=float)
        forecast = pd.DataFrame(
            {"expected_return_3m": [0.0, 0.03, 0.02, 0.04, 0.01], "signal_confidence": [0.5, 1, 1, 1, 1]},
            index=tickers,
        )
        candidates = build_candidate_portfolios(current, target, forecast, params_constraints)
        repair = candidates.get("CURRENT_COMPLIANCE_REPAIR")
        validation = (
            validate_portfolio_constraints(
                weights_actual=repair.weights,
                cash_weight=0.0,
                params=params_constraints,
                index=pd.Index(tickers, name="ticker"),
                label="Compliance repair",
            )
            if repair is not None
            else {"ok": False}
        )
        rows.append(
            _test_row(
                "compliance_repair_candidate_created_when_current_invalid",
                "PASS" if repair is not None and bool(validation["ok"]) else "FAIL",
                str({"has_repair": repair is not None, "validation": validation}),
            )
        )
    except Exception as exc:
        rows.append(_test_row("compliance_repair_candidate_created_when_current_invalid", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            scores_frame = pd.DataFrame(
                [
                    {
                        "discrete_candidate": "MOMENTUM_TILT_SIMPLE::FLOOR_BASE_0",
                        "robust_score": 0.001,
                        "net_robust_score": 0.001,
                        "delta_vs_hold": 0.0008,
                        "valid_constraints": False,
                        "validation_errors": "asset cap",
                        "asset_limit_violations": json.dumps([{"ticker": "SPMO", "actual_weight": 0.1708, "limit": 0.15, "excess": 0.0208}]),
                        "group_limit_violations": "",
                        "number_of_orders": 1,
                        "turnover_vs_current": 0.2,
                        "selected": False,
                    },
                    {
                        "discrete_candidate": "HOLD_CURRENT",
                        "robust_score": 0.0,
                        "net_robust_score": 0.0,
                        "delta_vs_hold": 0.0,
                        "valid_constraints": True,
                        "validation_errors": "",
                        "asset_limit_violations": "",
                        "group_limit_violations": "",
                        "number_of_orders": 0,
                        "turnover_vs_current": 0.0,
                        "selected": True,
                    },
                ]
            )
            candidate_map = {
                "MOMENTUM_TILT_SIMPLE::FLOOR_BASE_0": type("CandidateLike", (), {"metadata": {}})(),
                "HOLD_CURRENT": type("CandidateLike", (), {"metadata": {}})(),
            }
            pressure = _build_constraint_pressure_reports(
                scores_frame=scores_frame,
                candidate_map=candidate_map,
                params={},
                selected_reason="safe_hold_fallback_no_valid_trade_candidate",
                output_dir=output_dir,
            )
            required = {
                "candidate",
                "candidate_family",
                "score",
                "net_score_after_order_costs",
                "delta_vs_hold",
                "valid_constraints",
                "failed_constraint_type",
                "asset_or_group",
                "selected_reason",
            }
            rows.append(
                _test_row(
                    "constraint_pressure_report_exists_and_has_required_columns",
                    "PASS"
                    if (output_dir / "constraint_pressure_report.csv").exists()
                    and (output_dir / "constraint_pressure_report.txt").exists()
                    and required.issubset(set(pressure.columns))
                    else "FAIL",
                    str(sorted(pressure.columns)),
                )
            )
    except Exception as exc:
        rows.append(_test_row("constraint_pressure_report_exists_and_has_required_columns", "FAIL", str(exc)))

    try:
        tickers = ["SGOV", "AAA", "BBB"]
        returns = pd.DataFrame(
            {
                "SGOV": [0.0001] * 40,
                "AAA": np.linspace(-0.01, 0.012, 40),
                "BBB": np.linspace(0.008, -0.009, 40),
            },
            index=pd.date_range("2026-01-01", periods=40, freq="B"),
        )
        daily_cov, horizon_cov, corr, warnings = compute_baseline_covariance(
            returns=returns,
            as_of=returns.index[-1],
            assets=tickers,
            lookback=30,
            horizon_days=10,
            shrink_alpha=1.0,
            jitter=1e-10,
        )
        ok = (
            list(daily_cov.index) == tickers
            and list(horizon_cov.columns) == tickers
            and np.allclose(horizon_cov.to_numpy(), horizon_cov.to_numpy().T)
            and np.allclose(np.diag(corr), np.ones(len(tickers)))
            and isinstance(warnings, list)
        )
        rows.append(_test_row("covariance_matrix_created_and_aligned", "PASS" if ok else "FAIL", str(horizon_cov.shape)))
    except Exception as exc:
        rows.append(_test_row("covariance_matrix_created_and_aligned", "FAIL", str(exc)))

    try:
        assets = pd.Index(["A", "B"], name="ticker")
        covariance = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=assets, columns=assets)
        distribution = ScenarioRiskDistribution(
            as_of=pd.Timestamp("2026-05-07"),
            expected_returns=pd.DataFrame([[0.03, 0.02]], index=["base"], columns=assets),
            probabilities=pd.Series({"base": 1.0}),
            covariance_matrices={"base": covariance},
            baseline_daily_covariance=covariance,
            baseline_covariance_horizon=covariance,
            baseline_correlation=pd.DataFrame([[1.0, 0.1666667], [0.1666667, 1.0]], index=assets, columns=assets),
            summary=pd.DataFrame({"scenario_name": ["base"], "probability": [1.0]}),
            warnings=[],
        )
        weights = pd.Series({"A": 0.25, "B": 0.75})
        metrics = evaluate_portfolio_scenario_mixture(
            weights=weights,
            distribution=distribution,
            current_weights=pd.Series({"A": 0.0, "B": 0.0}),
            defensive_cash_weights=pd.Series({"A": 1.0, "B": 0.0}),
            hold_weights=pd.Series({"A": 0.0, "B": 1.0}),
            params={"sharpe_epsilon": 1e-8},
        )
        expected_variance = float(weights.to_numpy() @ covariance.to_numpy() @ weights.to_numpy())
        rows.append(
            _test_row(
                "portfolio_variance_uses_w_transpose_sigma_w",
                "PASS" if abs(float(metrics["within_scenario_variance"]) - expected_variance) < 1e-12 else "FAIL",
                f"{float(metrics['within_scenario_variance']):.8f} vs {expected_variance:.8f}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("portfolio_variance_uses_w_transpose_sigma_w", "FAIL", str(exc)))

    try:
        rf = risk_free_return_for_horizon(0.02, 63)
        expected_rf = (1.0 + 0.02) ** (63.0 / 252.0) - 1.0
        rows.append(
            _test_row(
                "risk_free_return_for_horizon_2pct_63_days",
                "PASS" if abs(rf - expected_rf) < 1.0e-14 and abs(rf - 0.0049629315732038215) < 1.0e-12 else "FAIL",
                f"{rf:.12f}",
            )
        )
        rows.append(
            _test_row(
                "test_risk_free_return_for_horizon",
                "PASS" if abs(rf - expected_rf) < 1.0e-14 and abs(rf - 0.0049629315732038215) < 1.0e-12 else "FAIL",
                f"2pct annual over 63 trading days = {rf:.12f}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("risk_free_return_for_horizon_2pct_63_days", "FAIL", str(exc)))
        rows.append(_test_row("test_risk_free_return_for_horizon", "FAIL", str(exc)))

    try:
        scenario_set = _make_test_scenario_set(
            ["AAA"],
            [[0.03], [0.05]],
            probabilities=[0.5, 0.5],
            names=["slow", "fast"],
        )
        candidate = CandidatePortfolio("AAA_ONLY", pd.Series({"AAA": 1.0}))
        score = evaluate_robust_candidate(
            candidate=candidate,
            scenario_set=scenario_set,
            w_current=pd.Series({"AAA": 1.0}),
            params={"risk_free_rate_annual": 0.02, "horizon_days": 63},
        )
        expected_rf = risk_free_return_for_horizon(0.02, 63)
        expected_sharpe = (0.04 - expected_rf) / 0.01
        rows.append(
            _test_row(
                "robust_scorer_uses_rf_adjusted_sharpe",
                "PASS"
                if (
                    abs(score.risk_free_return - expected_rf) < 1.0e-14
                    and abs(score.robust_sharpe - expected_sharpe) < 1.0e-12
                    and abs(score.return_over_volatility_legacy - 4.0) < 1.0e-12
                )
                else "FAIL",
                f"rf={score.risk_free_return:.12f}; robust_sharpe={score.robust_sharpe:.12f}; legacy={score.return_over_volatility_legacy:.12f}",
            )
        )
        rows.append(
            _test_row(
                "test_rf_adjusted_sharpe_used",
                "PASS"
                if (
                    abs(score.risk_free_return - expected_rf) < 1.0e-14
                    and abs(score.robust_sharpe - expected_sharpe) < 1.0e-12
                    and abs(score.return_over_volatility_legacy - 4.0) < 1.0e-12
                )
                else "FAIL",
                f"rf={score.risk_free_return:.12f}; rf_adjusted={score.robust_sharpe:.12f}; legacy={score.return_over_volatility_legacy:.12f}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("robust_scorer_uses_rf_adjusted_sharpe", "FAIL", str(exc)))
        rows.append(_test_row("test_rf_adjusted_sharpe_used", "FAIL", str(exc)))

    try:
        repo_root = Path(__file__).resolve().parent
        forbidden = ["mean_return " + "/ volatility", "return " + "/ volatility"]
        hits: list[str] = []
        for path in repo_root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                if pattern in text:
                    hits.append(f"{path.name}:{pattern}")
        rows.append(
            _test_row(
                "no_old_mean_over_volatility_sharpe_formula_left",
                "PASS" if not hits else "FAIL",
                ", ".join(hits) if hits else "no exact legacy Sharpe formulas found",
            )
        )
        rows.append(
            _test_row(
                "test_no_mean_over_vol_sharpe_left",
                "PASS" if not hits else "FAIL",
                ", ".join(hits) if hits else "no decision-time mean_return/volatility Sharpe formulas found",
            )
        )
    except Exception as exc:
        rows.append(_test_row("no_old_mean_over_volatility_sharpe_formula_left", "FAIL", str(exc)))
        rows.append(_test_row("test_no_mean_over_vol_sharpe_left", "FAIL", str(exc)))

    try:
        forecast, returns, params, assets = _make_scenario_input_fixture()
        scenario_inputs = build_scenario_inputs(forecast, returns, params)
        probability_sum = sum(float(scenario.probability) for scenario in scenario_inputs)
        scenario_names = [scenario.name for scenario in scenario_inputs]
        rows.append(
            _test_row(
                "scenario_inputs_default_probabilities_sum_to_one",
                "PASS" if scenario_names == list(SCENARIO_NAMES) and abs(probability_sum - 1.0) < 1.0e-12 else "FAIL",
                f"sum={probability_sum:.12f}; scenarios={','.join(scenario_names)}",
            )
        )
        aligned_and_stable = True
        min_eigenvalue = 1.0
        for scenario in scenario_inputs:
            covariance = scenario.covariance.to_numpy(dtype=float)
            eigenvalues = np.linalg.eigvalsh(covariance)
            min_eigenvalue = min(min_eigenvalue, float(eigenvalues.min()))
            aligned_and_stable = aligned_and_stable and (
                scenario.expected_returns.index.equals(assets)
                and scenario.covariance.index.equals(assets)
                and scenario.covariance.columns.equals(assets)
                and np.allclose(covariance, covariance.T, atol=1.0e-12)
                and float(eigenvalues.min()) >= -1.0e-8
                and abs(float(scenario.risk_free_return) - 0.0049629315732038215) < 1.0e-12
            )
        rows.append(
            _test_row(
                "scenario_inputs_expected_returns_and_covariance_aligned",
                "PASS" if aligned_and_stable else "FAIL",
                f"min_eigenvalue={min_eigenvalue:.12e}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("scenario_inputs_default_probabilities_sum_to_one", "FAIL", str(exc)))
        rows.append(_test_row("scenario_inputs_expected_returns_and_covariance_aligned", "FAIL", str(exc)))

    try:
        forecast, returns, params, _assets = _make_scenario_input_fixture()
        scenario_inputs = build_scenario_inputs(forecast, returns, params)
        metadata = scenario_inputs[0].metadata
        rows.append(
            _test_row(
                "scenario_inputs_metadata_marks_default_heuristics",
                "PASS"
                if (
                    metadata.get("assumption_type") == "default_configurable_heuristic"
                    and metadata.get("probability_source") == "default_configurable_heuristic"
                    and metadata.get("expected_return_source") == "forecast_table.expected_return_3m_scaled_to_horizon"
                    and metadata.get("covariance_source") == "historical_returns_shrunk_horizon_covariance"
                )
                else "FAIL",
                str(metadata),
            )
        )
    except Exception as exc:
        rows.append(_test_row("scenario_inputs_metadata_marks_default_heuristics", "FAIL", str(exc)))

    try:
        forecast, returns, params, _assets = _make_scenario_input_fixture()
        try:
            build_scenario_inputs(forecast, returns.drop(columns=["XLK"]), params)
            status = "FAIL"
            message = "expected ValueError for missing returns asset"
        except ValueError as exc:
            status = "PASS" if "returns missing assets" in str(exc) and "XLK" in str(exc) else "FAIL"
            message = str(exc)
        rows.append(_test_row("scenario_inputs_reject_missing_returns_asset", status, message))
    except Exception as exc:
        rows.append(_test_row("scenario_inputs_reject_missing_returns_asset", "FAIL", str(exc)))

    try:
        forecast, returns, params, _assets = _make_scenario_input_fixture()
        partial_group_config = {
            **params,
            "asset_groups": {
                "cash": ["SGOV"],
                "factor": ["SPMO"],
            },
        }
        try:
            build_scenario_inputs(forecast, returns, partial_group_config)
            status = "FAIL"
            message = "expected ValueError for missing asset groups"
        except ValueError as exc:
            status = "PASS" if "asset group map missing assets" in str(exc) and "XLK" in str(exc) else "FAIL"
            message = str(exc)
        rows.append(_test_row("scenario_inputs_reject_missing_asset_groups", status, message))
    except Exception as exc:
        rows.append(_test_row("scenario_inputs_reject_missing_asset_groups", "FAIL", str(exc)))

    try:
        forecast, returns, params, _assets = _make_scenario_input_fixture()
        bad_probability_config = {
            **params,
            "scenario_input_probabilities": {"bull_momentum": 1.0},
        }
        try:
            build_scenario_inputs(forecast, returns, bad_probability_config)
            status = "FAIL"
            message = "expected ValueError for incomplete probability config"
        except ValueError as exc:
            status = "PASS" if "missing required scenario probabilities" in str(exc) else "FAIL"
            message = str(exc)
        rows.append(_test_row("scenario_inputs_reject_incomplete_probability_config", status, message))
    except Exception as exc:
        rows.append(_test_row("scenario_inputs_reject_incomplete_probability_config", "FAIL", str(exc)))

    try:
        forecast, returns, params, _assets = _make_scenario_input_fixture()
        legacy_probability_config = {
            **params,
            "scenario_probabilities": {"base": 0.35, "risk_on": 0.65},
        }
        scenario_inputs = build_scenario_inputs(forecast, returns, legacy_probability_config)
        metadata = scenario_inputs[0].metadata
        rows.append(
            _test_row(
                "scenario_inputs_legacy_probability_schema_reports_default_fallback",
                "PASS"
                if (
                    metadata.get("probability_source") == "default_configurable_heuristic_due_to_legacy_config_schema"
                    and "legacy scenario schema" in str(metadata.get("diagnostic_warnings"))
                )
                else "FAIL",
                str(metadata),
            )
        )
    except Exception as exc:
        rows.append(_test_row("scenario_inputs_legacy_probability_schema_reports_default_fallback", "FAIL", str(exc)))

    try:
        forecast, returns, params, _assets = _make_scenario_input_fixture()
        scenario_inputs = build_scenario_inputs(forecast.drop(columns=["volatility_3m"]), returns, params)
        metadata = scenario_inputs[0].metadata
        rows.append(
            _test_row(
                "scenario_inputs_use_historical_volatility_when_forecast_vol_missing",
                "PASS" if metadata.get("volatility_source") == "historical_returns_covariance" else "FAIL",
                str(metadata),
            )
        )
    except Exception as exc:
        rows.append(_test_row("scenario_inputs_use_historical_volatility_when_forecast_vol_missing", "FAIL", str(exc)))

    try:
        forecast, returns, params, assets = _make_scenario_input_fixture()
        nested_config = {
            "solver": {
                "horizon_days": params["horizon_days"],
                "risk_free_rate_annual": params["risk_free_rate_annual"],
                "scenario_covariance_lookback": params["scenario_covariance_lookback"],
                "scenario_covariance_shrink_alpha": params["scenario_covariance_shrink_alpha"],
            },
            "asset_groups": {
                "cash": ["SGOV"],
                "factor": ["SPMO"],
                "commodities": ["PDBC", "GLD"],
                "bonds": ["IEF"],
                "us_sector": ["XLK"],
            },
            "scenario_input_probabilities": {name: weight * 2.0 for name, weight in zip(SCENARIO_NAMES, [0.25, 0.25, 0.20, 0.15, 0.10, 0.05])},
        }
        scenario_inputs = build_scenario_inputs(forecast, returns, nested_config)
        probability_sum = sum(float(scenario.probability) for scenario in scenario_inputs)
        rows.append(
            _test_row(
                "scenario_inputs_accept_nested_solver_config_and_group_lists",
                "PASS"
                if (
                    abs(probability_sum - 1.0) < 1.0e-12
                    and all(scenario.expected_returns.index.equals(assets) for scenario in scenario_inputs)
                    and float(scenario_inputs[0].risk_free_return) > 0.0
                )
                else "FAIL",
                f"sum={probability_sum:.12f}; rf={float(scenario_inputs[0].risk_free_return):.12f}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("scenario_inputs_accept_nested_solver_config_and_group_lists", "FAIL", str(exc)))

    try:
        forecast, returns, params, _assets = _make_scenario_input_fixture()
        mu_config = {
            **params,
            "asset_groups": {
                "cash": ["SGOV"],
                "factor": ["SPMO"],
                "commodities": ["PDBC"],
                "hedge": ["GLD"],
                "bonds": ["IEF"],
                "us_sector": ["XLK"],
            },
        }
        scenario_inputs = {scenario.name: scenario for scenario in build_scenario_inputs(forecast, returns, mu_config)}
        baseline = forecast["expected_return_3m"].astype(float)
        checks = [
            scenario_inputs["bull_momentum"].expected_returns["XLK"] > baseline["XLK"],
            scenario_inputs["bull_momentum"].expected_returns["SGOV"] < baseline["SGOV"],
            scenario_inputs["soft_landing"].expected_returns["XLK"] > baseline["XLK"],
            scenario_inputs["soft_landing"].expected_returns["IEF"] > baseline["IEF"],
            scenario_inputs["sideways_choppy"].expected_returns["XLK"] < baseline["XLK"],
            scenario_inputs["sideways_choppy"].expected_returns["SGOV"] >= baseline["SGOV"],
            scenario_inputs["inflation_shock"].expected_returns["PDBC"] > baseline["PDBC"],
            scenario_inputs["inflation_shock"].expected_returns["GLD"] > baseline["GLD"],
            scenario_inputs["inflation_shock"].expected_returns["IEF"] < baseline["IEF"],
            scenario_inputs["inflation_shock"].expected_returns["XLK"] < baseline["XLK"],
            scenario_inputs["growth_selloff"].expected_returns["XLK"] < baseline["XLK"],
            scenario_inputs["growth_selloff"].expected_returns["SPMO"] < baseline["SPMO"],
            scenario_inputs["growth_selloff"].expected_returns["GLD"] > baseline["GLD"],
            scenario_inputs["liquidity_stress"].expected_returns["SGOV"] >= baseline["SGOV"],
            scenario_inputs["liquidity_stress"].expected_returns["GLD"] > baseline["GLD"],
            scenario_inputs["liquidity_stress"].expected_returns["SPMO"] < baseline["SPMO"],
            scenario_inputs["liquidity_stress"].expected_returns["PDBC"] < baseline["PDBC"],
        ]
        rows.append(
            _test_row(
                "scenario_inputs_expected_return_adjustments_applied",
                "PASS" if all(checks) else "FAIL",
                (
                    f"bull_xlk={scenario_inputs['bull_momentum'].expected_returns['XLK']:.6f}; "
                    f"infl_pdbc={scenario_inputs['inflation_shock'].expected_returns['PDBC']:.6f}; "
                    f"growth_xlk={scenario_inputs['growth_selloff'].expected_returns['XLK']:.6f}; "
                    f"liq_sgov={scenario_inputs['liquidity_stress'].expected_returns['SGOV']:.6f}"
                ),
            )
        )
    except Exception as exc:
        rows.append(_test_row("scenario_inputs_expected_return_adjustments_applied", "FAIL", str(exc)))

    try:
        forecast, returns, params, _assets = _make_scenario_input_fixture()
        forecast.loc[["SPMO", "XLK"], "expected_return_3m"] = 0.020
        forecast.loc["SPMO", "momentum_score"] = 0.10
        forecast.loc["XLK", "momentum_score"] = 1.00
        momentum_config = {
            **params,
            "asset_groups": {
                "cash": ["SGOV"],
                "factor": ["SPMO"],
                "commodities": ["PDBC"],
                "hedge": ["GLD"],
                "bonds": ["IEF"],
                "us_sector": ["XLK"],
            },
        }
        scenario_inputs = {scenario.name: scenario for scenario in build_scenario_inputs(forecast, returns, momentum_config)}
        bull = scenario_inputs["bull_momentum"].expected_returns
        spmo_boost = float(bull["SPMO"] - 0.020)
        xlk_boost = float(bull["XLK"] - 0.020)
        rows.append(
            _test_row(
                "scenario_inputs_bull_momentum_uses_momentum_strength",
                "PASS" if xlk_boost > spmo_boost + 0.001 else "FAIL",
                f"spmo_boost={spmo_boost:.6f}; xlk_boost={xlk_boost:.6f}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("scenario_inputs_bull_momentum_uses_momentum_strength", "FAIL", str(exc)))

    try:
        forecast, returns, params, _assets = _make_scenario_input_fixture()
        horizon_config = {
            **params,
            "asset_groups": {
                "cash": ["SGOV"],
                "factor": ["SPMO"],
                "commodities": ["PDBC"],
                "hedge": ["GLD"],
                "bonds": ["IEF"],
                "us_sector": ["XLK"],
            },
        }
        shorter_config = {**horizon_config, "effective_horizon_days": 21}
        full_scenarios = {scenario.name: scenario for scenario in build_scenario_inputs(forecast, returns, horizon_config)}
        short_scenarios = {scenario.name: scenario for scenario in build_scenario_inputs(forecast, returns, shorter_config)}
        full_baseline = float(forecast.loc["PDBC", "expected_return_3m"])
        short_baseline = full_baseline * 21.0 / 63.0
        full_delta = float(full_scenarios["inflation_shock"].expected_returns["PDBC"] - full_baseline)
        short_delta = float(short_scenarios["inflation_shock"].expected_returns["PDBC"] - short_baseline)
        expected_short_rf = risk_free_return_for_horizon(0.02, 21)
        rows.append(
            _test_row(
                "scenario_inputs_expected_return_shocks_scale_with_horizon",
                "PASS"
                if (
                    abs(short_delta - full_delta / 3.0) < 1.0e-10
                    and abs(float(short_scenarios["inflation_shock"].risk_free_return) - expected_short_rf) < 1.0e-14
                )
                else "FAIL",
                f"full_delta={full_delta:.6f}; short_delta={short_delta:.6f}; short_rf={float(short_scenarios['inflation_shock'].risk_free_return):.12f}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("scenario_inputs_expected_return_shocks_scale_with_horizon", "FAIL", str(exc)))

    try:
        forecast, returns, params, _assets = _make_scenario_input_fixture()
        stress_config = {
            **params,
            "asset_groups": {
                "cash": ["SGOV"],
                "factor": ["SPMO"],
                "commodities": ["PDBC"],
                "hedge": ["GLD"],
                "bonds": ["IEF"],
                "us_sector": ["XLK"],
            },
        }
        scenario_inputs = {scenario.name: scenario for scenario in build_scenario_inputs(forecast, returns, stress_config)}

        def _scenario_vol(scenario_name: str, asset: str) -> float:
            covariance = scenario_inputs[scenario_name].covariance
            return float(np.sqrt(max(float(covariance.loc[asset, asset]), 0.0)))

        def _scenario_corr(scenario_name: str, left: str, right: str) -> float:
            covariance = scenario_inputs[scenario_name].covariance
            denom = float(np.sqrt(max(float(covariance.loc[left, left] * covariance.loc[right, right]), 0.0)))
            return float(covariance.loc[left, right] / denom) if denom > 0.0 else 0.0

        min_eigenvalue = min(
            float(np.linalg.eigvalsh(scenario.covariance.to_numpy(dtype=float)).min())
            for scenario in scenario_inputs.values()
        )
        growth_equity_corr = _scenario_corr("growth_selloff", "SPMO", "XLK")
        bull_equity_corr = _scenario_corr("bull_momentum", "SPMO", "XLK")
        liquidity_gold_equity_corr = _scenario_corr("liquidity_stress", "GLD", "SPMO")
        liquidity_commodity_equity_corr = _scenario_corr("liquidity_stress", "PDBC", "SPMO")
        inflation_bond_equity_corr = _scenario_corr("inflation_shock", "IEF", "SPMO")
        soft_bond_equity_corr = _scenario_corr("soft_landing", "IEF", "SPMO")
        checks = [
            _scenario_vol("growth_selloff", "SPMO") > _scenario_vol("bull_momentum", "SPMO") * 1.20,
            growth_equity_corr > bull_equity_corr,
            _scenario_vol("liquidity_stress", "PDBC") > _scenario_vol("bull_momentum", "PDBC") * 1.50,
            _scenario_vol("liquidity_stress", "GLD") <= _scenario_vol("inflation_shock", "GLD"),
            liquidity_gold_equity_corr < 0.20,
            liquidity_commodity_equity_corr > liquidity_gold_equity_corr + 0.20,
            inflation_bond_equity_corr > soft_bond_equity_corr,
            min_eigenvalue >= -1.0e-8,
        ]
        rows.append(
            _test_row(
                "scenario_inputs_covariance_stress_factors_applied",
                "PASS" if all(checks) else "FAIL",
                (
                    f"growth_corr={growth_equity_corr:.4f}; bull_corr={bull_equity_corr:.4f}; "
                    f"liq_gold_corr={liquidity_gold_equity_corr:.4f}; liq_commodity_corr={liquidity_commodity_equity_corr:.4f}; "
                    f"infl_bond_equity={inflation_bond_equity_corr:.4f}; soft_bond_equity={soft_bond_equity_corr:.4f}; "
                    f"min_eigenvalue={min_eigenvalue:.12e}"
                ),
            )
        )
    except Exception as exc:
        rows.append(_test_row("scenario_inputs_covariance_stress_factors_applied", "FAIL", str(exc)))

    try:
        dates = pd.date_range("2026-01-01", periods=80, freq="B")
        returns = pd.DataFrame(
            {
                "SGOV": np.full(len(dates), 0.00005),
                "SPMO": np.sin(np.linspace(0, 4, len(dates))) * 0.004,
                "PDBC": np.cos(np.linspace(0, 4, len(dates))) * 0.003,
            },
            index=dates,
        )
        forecast = pd.DataFrame(
            {
                "expected_return_3m": [0.003, 0.020, 0.012],
                "volatility_3m": [0.002, 0.060, 0.040],
            },
            index=pd.Index(["SGOV", "SPMO", "PDBC"], name="ticker"),
        )
        params = _make_test_optimizer_params(
            ["SGOV", "SPMO", "PDBC"],
            cash_ticker="SGOV",
            group_map={"SGOV": "cash", "SPMO": "factor", "PDBC": "commodities"},
            group_limits={"cash": 1.0, "factor": 0.5, "commodities": 0.5},
        )
        distribution = build_scenario_risk_distribution(
            forecast_table=forecast,
            returns=returns,
            as_of=dates[-1],
            params={**params, "cov_window": 30, "horizon_days": 10},
            effective_horizon_days=10,
        )
        prob_sum = float(distribution.probabilities.sum())
        rows.append(
            _test_row(
                "scenario_probabilities_sum_to_one",
                "PASS" if abs(prob_sum - 1.0) < 1e-10 and {"base", "risk_on", "risk_off", "rates_up", "rates_down", "commodity_up", "equity_stress"}.issubset(set(distribution.scenario_names)) else "FAIL",
                f"sum={prob_sum:.8f}; scenarios={','.join(distribution.scenario_names)}",
            )
        )
        candidate_frame = build_candidate_risk_return_frame(
            candidate_weights={
                "HOLD_CURRENT": pd.Series({"SGOV": 1.0, "SPMO": 0.0, "PDBC": 0.0}),
                "ACTIVE": pd.Series({"SGOV": 0.2, "SPMO": 0.5, "PDBC": 0.3}),
            },
            distribution=distribution,
            current_weights=pd.Series({"SGOV": 1.0, "SPMO": 0.0, "PDBC": 0.0}),
            defensive_cash_weights=pd.Series({"SGOV": 1.0, "SPMO": 0.0, "PDBC": 0.0}),
            hold_weights=pd.Series({"SGOV": 1.0, "SPMO": 0.0, "PDBC": 0.0}),
            params=params,
            scores_frame=pd.DataFrame(
                [
                    {"discrete_candidate": "HOLD_CURRENT", "robust_score": 0.0, "net_robust_score": 0.0, "valid_constraints": True},
                    {"discrete_candidate": "ACTIVE", "robust_score": 0.1, "net_robust_score": 0.1, "valid_constraints": True},
                ]
            ),
            selected_name="ACTIVE",
            selected_reason="selected_trade_candidate",
        )
        required_columns = {
            "mixture_expected_return",
            "within_scenario_variance",
            "between_scenario_variance",
            "mixture_variance",
            "scenario_mixture_sharpe",
            "probability_weighted_cvar",
            "robust_scenario_sharpe_objective",
        }
        active_row = candidate_frame.loc[candidate_frame["candidate"] == "ACTIVE"].iloc[0]
        rows.append(
            _test_row(
                "mixture_variance_includes_within_and_between_components",
                "PASS" if abs(float(active_row["mixture_variance"]) - float(active_row["within_scenario_variance"]) - float(active_row["between_scenario_variance"])) < 1e-12 else "FAIL",
                f"{float(active_row['mixture_variance']):.8f}",
            )
        )
        rows.append(
            _test_row(
                "scenario_mixture_sharpe_computed",
                "PASS" if np.isfinite(float(active_row["scenario_mixture_sharpe"])) else "FAIL",
                f"{float(active_row['scenario_mixture_sharpe']):.6f}",
            )
        )
        rows.append(
            _test_row(
                "candidate_risk_return_report_contains_mixture_fields",
                "PASS" if required_columns.issubset(set(candidate_frame.columns)) else "FAIL",
                str(sorted(required_columns - set(candidate_frame.columns))),
            )
        )
    except Exception as exc:
        rows.append(_test_row("scenario_mixture_risk_return_smoke", "FAIL", str(exc)))

    try:
        assets = pd.Index(["CASHX", "A", "B"], name="ticker")
        expected_returns = pd.DataFrame(
            [[0.0, 0.04, 0.03], [0.0, 0.05, 0.02]],
            index=["base", "risk_on"],
            columns=assets,
        )
        base_cov = pd.DataFrame(
            np.diag([1e-8, 0.01, 0.09]),
            index=assets,
            columns=assets,
        )
        distribution = ScenarioRiskDistribution(
            as_of=pd.Timestamp("2026-05-07"),
            expected_returns=expected_returns,
            probabilities=pd.Series({"base": 0.6, "risk_on": 0.4}),
            covariance_matrices={"base": base_cov, "risk_on": base_cov},
            baseline_daily_covariance=base_cov,
            baseline_covariance_horizon=base_cov,
            baseline_correlation=pd.DataFrame(np.eye(3), index=assets, columns=assets),
            summary=pd.DataFrame({"scenario_name": ["base", "risk_on"], "probability": [0.6, 0.4]}),
            warnings=[],
        )
        params_direct = _make_test_optimizer_params(
            ["CASHX", "A", "B"],
            cash_ticker="CASHX",
            group_map={"CASHX": "cash", "A": "risk", "B": "risk"},
            group_limits={"cash": 1.0, "risk": 1.0},
            max_turnover=2.0,
            max_equity_like_total=1.0,
        )
        result = optimize_scenario_sharpe_allocation(
            distribution=distribution,
            w_current=pd.Series({"CASHX": 1.0, "A": 0.0, "B": 0.0}),
            params={**params_direct, "direct_scenario_optimizer_max_starts": 5},
        )
        rows.append(
            _test_row(
                "direct_scenario_optimizer_prefers_best_sharpe_asset",
                "PASS" if result.success and float(result.target_weights["A"]) > 0.90 else "FAIL",
                f"{result.solver_name}: {result.target_weights.to_dict()}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("direct_scenario_optimizer_prefers_best_sharpe_asset", "FAIL", str(exc)))

    try:
        assets = pd.Index(["CASHX", "A", "B"], name="ticker")
        expected_returns = pd.DataFrame([[0.0, 0.06, 0.03]], index=["base"], columns=assets)
        base_cov = pd.DataFrame(np.diag([1e-8, 0.01, 0.02]), index=assets, columns=assets)
        distribution = ScenarioRiskDistribution(
            as_of=pd.Timestamp("2026-05-07"),
            expected_returns=expected_returns,
            probabilities=pd.Series({"base": 1.0}),
            covariance_matrices={"base": base_cov},
            baseline_daily_covariance=base_cov,
            baseline_covariance_horizon=base_cov,
            baseline_correlation=pd.DataFrame(np.eye(3), index=assets, columns=assets),
            summary=pd.DataFrame({"scenario_name": ["base"], "probability": [1.0]}),
            warnings=[],
        )
        params_capped = _make_test_optimizer_params(
            ["CASHX", "A", "B"],
            cash_ticker="CASHX",
            asset_max_weights={"CASHX": 1.0, "A": 0.4, "B": 1.0},
            group_map={"CASHX": "cash", "A": "risk", "B": "risk"},
            group_limits={"cash": 1.0, "risk": 1.0},
            max_turnover=2.0,
            max_equity_like_total=1.0,
        )
        result = optimize_scenario_sharpe_allocation(
            distribution=distribution,
            w_current=pd.Series({"CASHX": 1.0, "A": 0.0, "B": 0.0}),
            params={**params_capped, "direct_scenario_optimizer_max_starts": 5},
        )
        cap_ok = float(result.target_weights["A"]) <= 0.400001 and abs(float(result.target_weights.sum()) - 1.0) < 1e-6
        rows.append(
            _test_row(
                "direct_scenario_optimizer_respects_asset_cap",
                "PASS" if result.success and cap_ok else "FAIL",
                f"{result.solver_name}: {result.target_weights.to_dict()}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("direct_scenario_optimizer_respects_asset_cap", "FAIL", str(exc)))

    try:
        assets = pd.Index(["CASHX", "A"], name="ticker")
        expected_returns = pd.DataFrame([[0.0, 0.08]], index=["base"], columns=assets)
        base_cov = pd.DataFrame(np.diag([1e-8, 0.01]), index=assets, columns=assets)
        distribution = ScenarioRiskDistribution(
            as_of=pd.Timestamp("2026-05-07"),
            expected_returns=expected_returns,
            probabilities=pd.Series({"base": 1.0}),
            covariance_matrices={"base": base_cov},
            baseline_daily_covariance=base_cov,
            baseline_covariance_horizon=base_cov,
            baseline_correlation=pd.DataFrame(np.eye(2), index=assets, columns=assets),
            summary=pd.DataFrame({"scenario_name": ["base"], "probability": [1.0]}),
            warnings=[],
        )
        params_turnover = _make_test_optimizer_params(
            ["CASHX", "A"],
            cash_ticker="CASHX",
            group_map={"CASHX": "cash", "A": "risk"},
            group_limits={"cash": 1.0, "risk": 1.0},
            max_turnover=0.20,
            max_equity_like_total=1.0,
        )
        current = pd.Series({"CASHX": 1.0, "A": 0.0})
        result = optimize_scenario_sharpe_allocation(
            distribution=distribution,
            w_current=current,
            params={**params_turnover, "direct_scenario_optimizer_max_starts": 5},
        )
        turnover = float((result.target_weights.reindex(assets).fillna(0.0) - current).abs().sum())
        rows.append(
            _test_row(
                "direct_scenario_optimizer_respects_turnover_constraint",
                "PASS" if result.success and turnover <= 0.200001 and float(result.target_weights["A"]) <= 0.100001 else "FAIL",
                f"turnover={turnover:.6f}; weights={result.target_weights.to_dict()}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("direct_scenario_optimizer_respects_turnover_constraint", "FAIL", str(exc)))

    try:
        params_direct_candidate = _make_test_optimizer_params(
            ["CASHX", "A"],
            cash_ticker="CASHX",
            group_map={"CASHX": "cash", "A": "risk"},
            group_limits={"cash": 1.0, "risk": 1.0},
        )
        candidates = build_candidate_portfolios(
            w_current=pd.Series({"CASHX": 1.0, "A": 0.0}),
            w_target=pd.Series({"CASHX": 0.2, "A": 0.8}),
            forecast_table=pd.DataFrame(
                {"expected_return_3m": [0.0, 0.05], "signal_confidence": [1.0, 1.0]},
                index=pd.Index(["CASHX", "A"], name="ticker"),
            ),
            params=params_direct_candidate,
        )
        direct = candidates.get("SCENARIO_WEIGHTED_RF_SHARPE_OPTIMAL")
        legacy_alias = candidates.get("DIRECT_SCENARIO_OPTIMIZER")
        rows.append(
            _test_row(
                "candidate_factory_creates_scenario_weighted_solver_target",
                "PASS"
                if direct is not None
                and bool(direct.metadata.get("final_solver_source", False))
                and abs(float(direct.weights["A"]) - 0.8) < 1e-8
                and legacy_alias is not None
                and legacy_alias.metadata.get("legacy_alias_for") == "SCENARIO_WEIGHTED_RF_SHARPE_OPTIMAL"
                else "FAIL",
                str(direct.metadata if direct is not None else "missing"),
            )
        )
    except Exception as exc:
        rows.append(_test_row("candidate_factory_creates_scenario_weighted_solver_target", "FAIL", str(exc)))

    try:
        params = build_params()
        rows.append(
            _test_row(
                "objective_mode_defaults_to_scenario_weighted_rf_sharpe",
                "PASS" if str(params.get("optimization_objective")) == "scenario_weighted_rf_sharpe" else "FAIL",
                str(params.get("optimization_objective")),
            )
        )
    except Exception as exc:
        rows.append(_test_row("objective_mode_defaults_to_scenario_weighted_rf_sharpe", "FAIL", str(exc)))

    try:
        daily_bot_source = (Path(__file__).resolve().parent / "daily_bot.py").read_text(encoding="utf-8")
        slim_pipeline_source = (Path(__file__).resolve().parent / "scenario_daily_pipeline.py").read_text(encoding="utf-8")
        forbidden_final_solver_calls = "optimize_scenario_sharpe_allocation(" not in daily_bot_source
        rows.append(
            _test_row(
                "daily_bot_final_target_uses_scenario_weighted_solver",
                "PASS"
                if (
                    "run_scenario_weighted_daily_solve(" in daily_bot_source
                    and "solve_scenario_weighted_sharpe(" in slim_pipeline_source
                    and "Final Target Source:" in daily_bot_source
                    and "SCENARIO_WEIGHTED_RF_SHARPE_OPTIMAL" in daily_bot_source
                    and forbidden_final_solver_calls
                )
                else "FAIL",
                "daily_bot.py delegates final target solve to scenario_daily_pipeline and no longer calls optimize_scenario_sharpe_allocation",
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_bot_final_target_uses_scenario_weighted_solver", "FAIL", str(exc)))

    try:
        assets = pd.Index(["AAA", "BBB"], name="ticker")
        weights = pd.Series([0.6, 0.4], index=assets, dtype=float)
        current = pd.Series([0.5, 0.5], index=assets, dtype=float)
        scenarios = [
            WeightedScenarioInput(
                name="base",
                probability=0.75,
                expected_returns=pd.Series([0.08, 0.02], index=assets, dtype=float),
                covariance=pd.DataFrame([[0.04, 0.01], [0.01, 0.02]], index=assets, columns=assets),
                risk_free_return=0.01,
            ),
            WeightedScenarioInput(
                name="stress",
                probability=0.25,
                expected_returns=pd.Series([-0.03, 0.015], index=assets, dtype=float),
                covariance=pd.DataFrame([[0.09, 0.00], [0.00, 0.01]], index=assets, columns=assets),
                risk_free_return=0.01,
            ),
        ]
        config = WeightedSolverConfig(
            lambda_turnover=0.03,
            lambda_concentration=0.01,
            lambda_downside=0.15,
            eps_variance=1.0e-10,
            max_turnover=1.0,
            ftol=1.0e-9,
            maxiter=1000,
        )
        result = evaluate_scenario_weighted_weights(weights, current, scenarios, config)
        w = weights.to_numpy(dtype=float)
        w_current = current.to_numpy(dtype=float)
        manual_score = 0.0
        manual_downside = 0.0
        for scenario in scenarios:
            mu = scenario.expected_returns.to_numpy(dtype=float)
            sigma = scenario.covariance.to_numpy(dtype=float)
            port_ret = float(w @ mu)
            port_var = float(w @ sigma @ w)
            port_vol = float(np.sqrt(max(port_var, 0.0) + config.eps_variance))
            manual_score += float(scenario.probability) * ((port_ret - scenario.risk_free_return) / port_vol)
            manual_downside += float(scenario.probability) * max(0.0, scenario.risk_free_return - port_ret)
        manual_turnover = float(np.abs(w - w_current).sum())
        manual_concentration = float(np.square(w).sum())
        manual_final_score = (
            manual_score
            - config.lambda_turnover * manual_turnover
            - config.lambda_concentration * manual_concentration
            - config.lambda_downside * manual_downside
        )
        rows.append(
            _test_row(
                "scenario_weighted_solver_objective_matches_formula",
                "PASS" if abs(float(result.objective_value) - manual_final_score) < 1.0e-12 else "FAIL",
                f"solver={result.objective_value:.12f} manual={manual_final_score:.12f}",
            )
        )
        rows.append(
            _test_row(
                "scenario_weighted_solver_turnover_uses_exact_l1_norm",
                "PASS" if abs(float(result.turnover) - float(np.abs(w - w_current).sum())) < 1.0e-14 else "FAIL",
                f"solver_turnover={result.turnover:.12f}; exact_l1={float(np.abs(w - w_current).sum()):.12f}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("scenario_weighted_solver_objective_matches_formula", "FAIL", str(exc)))
        rows.append(_test_row("scenario_weighted_solver_turnover_uses_exact_l1_norm", "FAIL", str(exc)))

    try:
        assets = pd.Index(["AAA", "BBB", "CCC"], name="ticker")
        covariance = pd.DataFrame(np.diag([0.04, 0.03, 0.02]), index=assets, columns=assets)
        scenarios = [
            WeightedScenarioInput(
                name="base",
                probability=1.0,
                expected_returns=pd.Series([0.08, 0.04, 0.02], index=assets, dtype=float),
                covariance=covariance,
                risk_free_return=0.01,
            )
        ]
        config = WeightedSolverConfig(
            lambda_turnover=0.01,
            lambda_concentration=0.01,
            lambda_downside=0.10,
            eps_variance=1.0e-10,
            max_turnover=2.0,
            ftol=1.0e-9,
            maxiter=500,
        )
        result = solve_scenario_weighted_sharpe(
            current_weights=pd.Series([0.90, 0.10, 0.00], index=assets, dtype=float),
            scenarios=scenarios,
            max_weights={"AAA": 0.60, "BBB": 0.60, "CCC": 0.60},
            asset_groups={"risk": ["AAA", "BBB"], "defensive": ["CCC"]},
            group_limits={"risk": 0.80, "defensive": 0.60},
            min_group_weights={"defensive": 0.20},
            config=config,
        )
        diagnostics = result.constraint_diagnostics
        rows.append(
            _test_row(
                "scenario_weighted_solver_enforces_bounds_groups_and_turnover",
                "PASS"
                if (
                    result.success
                    and diagnostics["feasible"]
                    and abs(float(result.weights.sum()) - 1.0) < 1.0e-7
                    and float(result.weights["AAA"]) <= 0.60 + 1.0e-7
                    and float(result.weights[["AAA", "BBB"]].sum()) <= 0.80 + 1.0e-7
                    and float(result.weights["CCC"]) >= 0.20 - 1.0e-7
                    and diagnostics["turnover"] <= config.max_turnover + 1.0e-7
                )
                else "FAIL",
                f"weights={result.weights.to_dict()} diagnostics={diagnostics}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("scenario_weighted_solver_enforces_bounds_groups_and_turnover", "FAIL", str(exc)))

    try:
        assets = pd.Index(["AAA", "BBB", "CCC"], name="ticker")
        covariance = pd.DataFrame(np.diag([0.04, 0.03, 0.02]), index=assets, columns=assets)
        scenarios = [
            WeightedScenarioInput(
                name="base",
                probability=1.0,
                expected_returns=pd.Series([0.08, 0.04, 0.02], index=assets, dtype=float),
                covariance=covariance,
                risk_free_return=0.01,
            )
        ]
        config = WeightedSolverConfig(0.01, 0.01, 0.10, 1.0e-10, 2.0, 1.0e-9, 500)
        result = solve_scenario_weighted_sharpe(
            current_weights=pd.Series([0.90, 0.10, 0.00], index=assets, dtype=float),
            scenarios=scenarios,
            max_weights={"AAA": 0.60, "BBB": 0.60, "CCC": 0.60},
            asset_groups={"risk": ["AAA", "BBB"], "defensive": ["CCC"]},
            group_limits={"risk": 0.80, "defensive": 0.60},
            min_group_weights={"defensive": 0.20},
            config=config,
        )
        rows.append(
            _test_row(
                "test_solver_weights_sum_to_one",
                "PASS" if result.success and abs(float(result.weights.sum()) - 1.0) < 1.0e-7 else "FAIL",
                f"sum={float(result.weights.sum()):.12f}; weights={result.weights.to_dict()}",
            )
        )
        rows.append(
            _test_row(
                "test_solver_respects_asset_caps",
                "PASS"
                if result.success and all(float(result.weights.loc[asset]) <= cap + 1.0e-7 for asset, cap in {"AAA": 0.60, "BBB": 0.60, "CCC": 0.60}.items())
                else "FAIL",
                f"weights={result.weights.to_dict()}",
            )
        )
        rows.append(
            _test_row(
                "test_solver_respects_group_limits",
                "PASS"
                if (
                    result.success
                    and float(result.weights[["AAA", "BBB"]].sum()) <= 0.80 + 1.0e-7
                    and float(result.weights["CCC"]) >= 0.20 - 1.0e-7
                    and float(result.weights["CCC"]) <= 0.60 + 1.0e-7
                )
                else "FAIL",
                f"risk={float(result.weights[['AAA','BBB']].sum()):.12f}; defensive={float(result.weights['CCC']):.12f}; weights={result.weights.to_dict()}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("test_solver_weights_sum_to_one", "FAIL", str(exc)))
        rows.append(_test_row("test_solver_respects_asset_caps", "FAIL", str(exc)))
        rows.append(_test_row("test_solver_respects_group_limits", "FAIL", str(exc)))

    try:
        assets = pd.Index(["MOM", "DEF"], name="ticker")
        covariance = pd.DataFrame(np.diag([0.04, 0.02]), index=assets, columns=assets)
        config = WeightedSolverConfig(0.0, 0.0, 0.0, 1.0e-10, 2.0, 1.0e-9, 1000)

        def _solve_bull_probability(probability: float) -> WeightedSolverResult:
            return solve_scenario_weighted_sharpe(
                current_weights=pd.Series([0.50, 0.50], index=assets, dtype=float),
                scenarios=[
                    WeightedScenarioInput(
                        name="bull_momentum",
                        probability=probability,
                        expected_returns=pd.Series([0.10, 0.02], index=assets, dtype=float),
                        covariance=covariance,
                        risk_free_return=0.0,
                    ),
                    WeightedScenarioInput(
                        name="stress",
                        probability=1.0 - probability,
                        expected_returns=pd.Series([-0.04, 0.02], index=assets, dtype=float),
                        covariance=covariance,
                        risk_free_return=0.0,
                    ),
                ],
                max_weights={"MOM": 1.0, "DEF": 1.0},
                asset_groups={"MOM": "risk", "DEF": "defensive"},
                group_limits={"risk": 1.0, "defensive": 1.0},
                config=config,
            )

        low_bull = _solve_bull_probability(0.20)
        high_bull = _solve_bull_probability(0.90)
        rows.append(
            _test_row(
                "test_solver_uses_scenario_probabilities",
                "PASS"
                if (
                    low_bull.success
                    and high_bull.success
                    and float(high_bull.weights["MOM"]) > float(low_bull.weights["MOM"]) + 0.25
                )
                else "FAIL",
                f"low_bull_MOM={float(low_bull.weights['MOM']):.6f}; high_bull_MOM={float(high_bull.weights['MOM']):.6f}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("test_solver_uses_scenario_probabilities", "FAIL", str(exc)))

    try:
        assets = pd.Index(["A", "B", "C"], name="ticker")
        vol = 0.20
        correlation = np.array(
            [
                [1.00, 0.95, 0.00],
                [0.95, 1.00, 0.00],
                [0.00, 0.00, 1.00],
            ],
            dtype=float,
        )
        covariance = pd.DataFrame(correlation * vol * vol, index=assets, columns=assets)
        config = WeightedSolverConfig(0.0, 0.0, 0.0, 1.0e-10, 2.0, 1.0e-9, 1000)
        result = solve_scenario_weighted_sharpe(
            current_weights=pd.Series([1 / 3, 1 / 3, 1 / 3], index=assets, dtype=float),
            scenarios=[
                WeightedScenarioInput(
                    name="base",
                    probability=1.0,
                    expected_returns=pd.Series([0.05, 0.05, 0.05], index=assets, dtype=float),
                    covariance=covariance,
                    risk_free_return=0.0,
                )
            ],
            max_weights={"A": 0.50, "B": 0.50, "C": 0.50},
            asset_groups={"A": "risk", "B": "risk", "C": "diversifier"},
            group_limits={"risk": 1.0, "diversifier": 1.0},
            config=config,
        )
        rows.append(
            _test_row(
                "test_solver_uses_covariance",
                "PASS"
                if (
                    result.success
                    and float(result.weights["A"] + result.weights["B"]) < 0.95
                    and float(result.weights["C"]) > 0.40
                )
                else "FAIL",
                f"weights={result.weights.to_dict()}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("test_solver_uses_covariance", "FAIL", str(exc)))

    try:
        assets = pd.Index(["RISK", "SAFE"], name="ticker")
        covariance = pd.DataFrame(np.diag([0.05**2, 0.02**2]), index=assets, columns=assets)
        scenarios = [
            WeightedScenarioInput(
                name="base",
                probability=0.90,
                expected_returns=pd.Series([0.20, 0.02], index=assets, dtype=float),
                covariance=covariance,
                risk_free_return=0.01,
            ),
            WeightedScenarioInput(
                name="stress",
                probability=0.10,
                expected_returns=pd.Series([-0.30, 0.018], index=assets, dtype=float),
                covariance=covariance,
                risk_free_return=0.01,
            ),
        ]

        def _solve_downside(lambda_downside: float) -> WeightedSolverResult:
            return solve_scenario_weighted_sharpe(
                current_weights=pd.Series([0.50, 0.50], index=assets, dtype=float),
                scenarios=scenarios,
                max_weights={"RISK": 1.0, "SAFE": 1.0},
                asset_groups={"RISK": "risk", "SAFE": "safe"},
                group_limits={"risk": 1.0, "safe": 1.0},
                config=WeightedSolverConfig(0.0, 0.0, lambda_downside, 1.0e-10, 2.0, 1.0e-9, 1000),
            )

        no_downside = _solve_downside(0.0)
        high_downside = _solve_downside(100.0)
        rows.append(
            _test_row(
                "test_solver_penalizes_downside",
                "PASS"
                if (
                    no_downside.success
                    and high_downside.success
                    and float(high_downside.weights["RISK"]) < float(no_downside.weights["RISK"]) - 0.15
                    and float(high_downside.downside_penalty) <= float(no_downside.downside_penalty) + 1.0e-7
                )
                else "FAIL",
                f"risk_weight_no_penalty={float(no_downside.weights['RISK']):.6f}; risk_weight_high_penalty={float(high_downside.weights['RISK']):.6f}; downside_no={float(no_downside.downside_penalty):.6f}; downside_high={float(high_downside.downside_penalty):.6f}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("test_solver_penalizes_downside", "FAIL", str(exc)))

    try:
        assets = pd.Index(["AAA", "BBB", "CCC"], name="ticker")
        covariance = pd.DataFrame(np.diag([0.04, 0.03, 0.02]), index=assets, columns=assets)
        scenarios = [
            WeightedScenarioInput(
                name="base",
                probability=1.0,
                expected_returns=pd.Series([0.08, 0.04, 0.02], index=assets, dtype=float),
                covariance=covariance,
                risk_free_return=0.01,
            )
        ]
        config = WeightedSolverConfig(0.01, 0.01, 0.10, 1.0e-10, 2.0, 1.0e-9, 500)
        result = solve_scenario_weighted_sharpe(
            current_weights=pd.Series([0.60, 0.20, 0.20], index=assets, dtype=float),
            scenarios=scenarios,
            max_weights={"AAA": 0.70, "BBB": 0.60, "CCC": 0.60},
            asset_groups={"risk": ["AAA", "BBB"], "defensive": ["CCC"]},
            group_limits={"risk": 0.85, "defensive": 0.60},
            min_group_weights={"defensive": 0.15},
            config=config,
        )
        validation = validate_weighted_solver_result(
            result,
            {
                "max_weights": {"AAA": 0.70, "BBB": 0.60, "CCC": 0.60},
                "asset_groups": {"risk": ["AAA", "BBB"], "defensive": ["CCC"]},
                "group_limits": {"risk": 0.85, "defensive": 0.60},
                "min_group_weights": {"defensive": 0.15},
                "current_weights": pd.Series([0.60, 0.20, 0.20], index=assets, dtype=float),
                "max_turnover": config.max_turnover,
                "config": config,
            },
        )
        rows.append(
            _test_row(
                "solver_result_validation_accepts_feasible_success",
                "PASS"
                if (
                    bool(validation["ok"])
                    and bool(validation["checks"]["objective_finite"])
                    and bool(validation["checks"]["scenario_vols_positive"])
                    and bool(validation["checks"]["scenario_sharpes_finite"])
                )
                else "FAIL",
                str(validation),
            )
        )
    except Exception as exc:
        rows.append(_test_row("solver_result_validation_accepts_feasible_success", "FAIL", str(exc)))

    try:
        assets = pd.Index(["AAA", "BBB"], name="ticker")
        metrics = pd.DataFrame(
            [
                {
                    "scenario": "base",
                    "probability": 1.0,
                    "portfolio_return": 0.02,
                    "risk_free_return": 0.01,
                    "portfolio_volatility": 0.10,
                    "rf_adjusted_sharpe": 0.10,
                    "downside_shortfall": 0.0,
                }
            ]
        )
        failed = WeightedSolverResult(
            success=False,
            status=9,
            message="infeasible constraints",
            weights=pd.Series([0.50, 0.50], index=assets, dtype=float),
            objective_value=0.01,
            weighted_sharpe=0.10,
            turnover=0.0,
            concentration=0.50,
            downside_penalty=0.0,
            per_scenario_metrics=metrics,
            constraint_diagnostics={},
        )
        validation = validate_weighted_solver_result(
            failed,
            {
                "max_weights": {"AAA": 0.60, "BBB": 0.60},
                "asset_groups": {"AAA": "risk", "BBB": "risk"},
                "group_limits": {"risk": 1.0},
                "current_weights": pd.Series([0.50, 0.50], index=assets, dtype=float),
                "max_turnover": 2.0,
            },
        )
        rows.append(
            _test_row(
                "solver_result_validation_flags_scipy_failure",
                "PASS"
                if (
                    not bool(validation["ok"])
                    and bool(validation["solver_failed"])
                    and validation["failure_reason"] == "infeasible constraints"
                    and any("solver_success_false" in warning for warning in validation["warnings"])
                )
                else "FAIL",
                str(validation),
            )
        )
    except Exception as exc:
        rows.append(_test_row("solver_result_validation_flags_scipy_failure", "FAIL", str(exc)))

    try:
        assets = pd.Index(["AAA", "BBB"], name="ticker")
        covariance = pd.DataFrame(np.diag([0.04, 0.02]), index=assets, columns=assets)
        scenarios = [
            WeightedScenarioInput(
                name="base",
                probability=1.0,
                expected_returns=pd.Series([0.08, 0.02], index=assets, dtype=float),
                covariance=covariance,
                risk_free_return=0.01,
            )
        ]
        config = WeightedSolverConfig(0.01, 0.01, 0.10, 1.0e-10, 2.0, 1.0e-9, 200)
        try:
            solve_scenario_weighted_sharpe(
                current_weights=pd.Series([0.50, 0.50], index=assets, dtype=float),
                scenarios=scenarios,
                max_weights={"AAA": 0.40, "BBB": 0.40},
                asset_groups={"AAA": "risk", "BBB": "risk"},
                group_limits={"risk": 1.0},
                config=config,
            )
            status = "FAIL"
            message = "expected ValueError"
        except ValueError as exc:
            status = "PASS" if "sum(max_weights)" in str(exc) else "FAIL"
            message = str(exc)
        rows.append(_test_row("scenario_weighted_solver_rejects_insufficient_max_weights", status, message))
    except Exception as exc:
        rows.append(_test_row("scenario_weighted_solver_rejects_insufficient_max_weights", "FAIL", str(exc)))

    try:
        assets = pd.Index(["AAA", "BBB"], name="ticker")
        covariance = pd.DataFrame(np.diag([0.04, 0.02]), index=assets, columns=assets)
        scenarios = [
            WeightedScenarioInput(
                name="base",
                probability=1.0,
                expected_returns=pd.Series([0.08, 0.02], index=assets, dtype=float),
                covariance=covariance,
                risk_free_return=0.01,
            )
        ]
        config = WeightedSolverConfig(0.01, 0.01, 0.10, 1.0e-10, 2.0, 1.0e-9, 200)
        try:
            solve_scenario_weighted_sharpe(
                current_weights=pd.Series([0.50, 0.50], index=assets, dtype=float),
                scenarios=scenarios,
                max_weights={"AAA": 0.60, "BBB": 1.00},
                asset_groups={"AAA": "risk", "BBB": "risk"},
                group_limits={"risk": 1.0},
                config=config,
                x0=pd.Series([0.90, 0.10], index=assets, dtype=float),
            )
            status = "FAIL"
            message = "expected infeasible x0 ValueError"
        except ValueError as exc:
            status = "PASS" if "x0 is not feasible" in str(exc) else "FAIL"
            message = str(exc)
        rows.append(_test_row("scenario_weighted_solver_rejects_invalid_x0", status, message))
    except Exception as exc:
        rows.append(_test_row("scenario_weighted_solver_rejects_invalid_x0", "FAIL", str(exc)))

    try:
        assets = pd.Index(["AAA", "BBB"], name="ticker")
        current = pd.Series([0.50, 0.50], index=assets, dtype=float)
        optimal = pd.Series([0.80, 0.20], index=assets, dtype=float)
        executable = _apply_execution_fraction(
            current_weights=current,
            optimal_weights=optimal,
            execution_fraction=0.50,
            fallback_ticker=None,
        ).reindex(assets)
        rows.append(
            _test_row(
                "execution_fraction_damps_solver_target",
                "PASS"
                if abs(float(executable["AAA"]) - 0.65) < 1.0e-12
                and abs(float(executable["BBB"]) - 0.35) < 1.0e-12
                and abs(float(executable.sum()) - 1.0) < 1.0e-12
                else "FAIL",
                executable.to_dict(),
            )
        )
    except Exception as exc:
        rows.append(_test_row("execution_fraction_damps_solver_target", "FAIL", str(exc)))

    try:
        relationship_cases = [
            (0.75, "strong_positive"),
            (0.40, "moderate_positive"),
            (0.00, "low_or_neutral"),
            (-0.199, "low_or_neutral"),
            (-0.20, "moderate_negative"),
            (-0.50, "strong_negative"),
        ]
        diversification_cases = [
            (0.19, 0.0019, 0.10, 0.10, "high"),
            (0.20, 0.0020, 0.10, 0.10, "medium"),
            (0.49, 0.0049, 0.10, 0.10, "medium"),
            (0.50, 0.0050, 0.10, 0.10, "low"),
            (-0.30, -0.0030, 0.10, 0.10, "high"),
        ]
        relationship_ok = all(_correlation_label(value) == expected for value, expected in relationship_cases)
        diversification_ok = all(
            _diversification_label(value, covariance=covariance, vol_i=vol_i, vol_j=vol_j) == expected
            for value, covariance, vol_i, vol_j, expected in diversification_cases
        )
        rows.append(
            _test_row(
                "pairwise_relationship_and_diversification_labels_follow_thresholds",
                "PASS" if relationship_ok and diversification_ok else "FAIL",
                f"relationship_ok={relationship_ok} diversification_ok={diversification_ok}",
            )
        )
    except Exception as exc:
        rows.append(
            _test_row(
                "pairwise_relationship_and_diversification_labels_follow_thresholds",
                "FAIL",
                str(exc),
            )
        )

    try:
        assets = pd.Index(["AAA", "BBB"], name="ticker")
        current = pd.Series([0.50, 0.50], index=assets, dtype=float)
        optimal = pd.Series([0.80, 0.20], index=assets, dtype=float)
        executable = pd.Series([0.65, 0.35], index=assets, dtype=float)
        scenarios = [
            WeightedScenarioInput(
                name="base",
                probability=1.0,
                expected_returns=pd.Series([0.08, 0.02], index=assets, dtype=float),
                covariance=pd.DataFrame([[0.04, 0.01], [0.01, 0.02]], index=assets, columns=assets),
                risk_free_return=0.01,
            )
        ]
        config = WeightedSolverConfig(0.01, 0.01, 0.10, 1.0e-10, 2.0, 1.0e-9, 200)
        result = evaluate_scenario_weighted_weights(optimal, current, scenarios, config)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _write_scenario_weighted_solver_reports(
                scenarios=scenarios,
                result=result,
                current_weights=current,
                target_weights=optimal,
                executable_weights=executable,
                execution_fraction=0.50,
                output_dir=tmp_dir,
                final_target_source="SCENARIO_WEIGHTED_RF_SHARPE_OPTIMAL",
            )
            solver_result = pd.read_csv(tmp_dir / "scenario_solver_result.csv")
            metrics = pd.read_csv(tmp_dir / "scenario_solver_metrics.csv")
            pairwise = pd.read_csv(tmp_dir / "pairwise_asset_relationships.csv")
            required_result_columns = {
                "asset",
                "current_weight",
                "optimal_weight",
                "executable_weight",
                "delta_optimal",
                "delta_executable",
            }
            required_pairwise_columns = {
                "scenario",
                "asset_i",
                "asset_j",
                "correlation",
                "covariance",
                "vol_i",
                "vol_j",
                "relationship_label",
                "diversification_label",
            }
            metric_names = set(metrics["metric"].astype(str))
            expected_paths = [
                tmp_dir / "scenario_probabilities.csv",
                tmp_dir / "scenario_expected_returns.csv",
                tmp_dir / "scenario_covariance_base.csv",
                tmp_dir / "scenario_correlation_base.csv",
                tmp_dir / "pairwise_asset_relationships.csv",
                tmp_dir / "scenario_solver_result.csv",
                tmp_dir / "scenario_solver_metrics.csv",
                tmp_dir / "scenario_solver_per_scenario_metrics.csv",
                tmp_dir / "scenario_solver_decision.md",
            ]
            rows.append(
                _test_row(
                    "scenario_solver_reports_include_executable_and_pairwise_outputs",
                    "PASS"
                    if (
                        all(path.exists() for path in expected_paths)
                        and required_result_columns.issubset(solver_result.columns)
                        and required_pairwise_columns.issubset(pairwise.columns)
                        and {"turnover_executable", "execution_fraction"}.issubset(metric_names)
                    )
                    else "FAIL",
                    f"result_cols={list(solver_result.columns)} metrics={sorted(metric_names)} pairwise_cols={list(pairwise.columns)}",
                )
            )
            rows.append(
                _test_row(
                    "test_report_files_created",
                    "PASS" if all(path.exists() for path in expected_paths) else "FAIL",
                    ", ".join(path.name for path in expected_paths if path.exists()),
                )
            )
    except Exception as exc:
        rows.append(_test_row("scenario_solver_reports_include_executable_and_pairwise_outputs", "FAIL", str(exc)))
        rows.append(_test_row("test_report_files_created", "FAIL", str(exc)))

    try:
        assets = pd.Index(["AAA", "BBB"], name="ticker")
        current = pd.Series([0.50, 0.50], index=assets, dtype=float)
        scenarios = [
            WeightedScenarioInput(
                name="base",
                probability=1.0,
                expected_returns=pd.Series([0.04, 0.03], index=assets, dtype=float),
                covariance=pd.DataFrame([[0.04, 0.00], [0.00, 0.02]], index=assets, columns=assets),
                risk_free_return=0.01,
            )
        ]
        failed_result = WeightedSolverResult(
            success=False,
            status=9,
            message="SLSQP infeasible",
            weights=current.copy(),
            objective_value=0.0,
            weighted_sharpe=0.0,
            turnover=0.0,
            concentration=float(np.square(current.to_numpy(dtype=float)).sum()),
            downside_penalty=0.0,
            per_scenario_metrics=pd.DataFrame(
                [
                    {
                        "scenario": "base",
                        "probability": 1.0,
                        "portfolio_return": 0.035,
                        "risk_free_return": 0.01,
                        "portfolio_volatility": 0.10,
                        "rf_adjusted_sharpe": 0.25,
                        "downside_shortfall": 0.0,
                    }
                ]
            ),
            constraint_diagnostics={"solver_failed": True, "failure_reason": "SLSQP infeasible"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _write_scenario_weighted_solver_reports(
                scenarios=scenarios,
                result=failed_result,
                current_weights=current,
                target_weights=current,
                executable_weights=current,
                execution_fraction=1.0,
                output_dir=tmp_dir,
                final_target_source="HOLD_SOLVER_FAILED",
            )
            metrics = pd.read_csv(tmp_dir / "scenario_solver_metrics.csv")
            decision_text = (tmp_dir / "scenario_solver_decision.md").read_text(encoding="utf-8")
            metric_values = dict(zip(metrics["metric"].astype(str), metrics["value"].astype(str), strict=False))
            rows.append(
                _test_row(
                    "solver_failure_report_marks_hold_solver_failed",
                    "PASS"
                    if (
                        metric_values.get("solver_failed", "").lower() == "true"
                        and metric_values.get("failure_reason") == "SLSQP infeasible"
                        and "Final Target Source: HOLD_SOLVER_FAILED" in decision_text
                        and "solver_failed: True" in decision_text
                    )
                    else "FAIL",
                    f"metrics={metric_values} decision={decision_text}",
                )
            )
            rows.append(
                _test_row(
                    "test_solver_failure_fallback",
                    "PASS"
                    if (
                        metric_values.get("final_target_source") == "HOLD_SOLVER_FAILED"
                        and metric_values.get("solver_failed", "").lower() == "true"
                        and metric_values.get("failure_reason") == "SLSQP infeasible"
                    )
                    else "FAIL",
                    f"metrics={metric_values}",
                )
            )
    except Exception as exc:
        rows.append(_test_row("solver_failure_report_marks_hold_solver_failed", "FAIL", str(exc)))
        rows.append(_test_row("test_solver_failure_fallback", "FAIL", str(exc)))

    try:
        idx = pd.Index(["SGOV", "AAA"], name="ticker")
        hold_candidate = type("CandidateLike", (), {})()
        hold_candidate.weights_proxy = pd.Series({"SGOV": 1.0, "AAA": 0.0})
        hold_candidate.weights_actual = hold_candidate.weights_proxy.copy()
        hold_candidate.shares = pd.Series({"SGOV": 1.0, "AAA": 0.0})
        active_candidate = type("CandidateLike", (), {})()
        active_candidate.weights_proxy = pd.Series({"SGOV": 0.0, "AAA": 1.0})
        active_candidate.weights_actual = active_candidate.weights_proxy.copy()
        active_candidate.shares = pd.Series({"SGOV": 0.0, "AAA": 1.0})
        scored = {
            "scores_frame": pd.DataFrame(
                [
                    {
                        "discrete_candidate": "HOLD_CURRENT",
                        "net_robust_score": 0.0,
                        "robust_scenario_sharpe_objective": 0.0,
                        "cvar_5": 0.0,
                        "turnover_vs_current": 0.0,
                        "max_abs_weight_drift": 0.0,
                        "number_of_positions": 1,
                        "cash_left": 0.0,
                        "delta_vs_cash": 0.0,
                        "probability_beats_hold": 0.0,
                        "probability_beats_cash": 0.0,
                        "valid_constraints": True,
                        "validation_errors": "",
                    },
                    {
                        "discrete_candidate": "ACTIVE",
                        "net_robust_score": -0.1,
                        "robust_scenario_sharpe_objective": 1.0,
                        "cvar_5": 0.0,
                        "turnover_vs_current": 1.0,
                        "max_abs_weight_drift": 1.0,
                        "number_of_positions": 1,
                        "cash_left": 0.0,
                        "delta_vs_cash": 1.0,
                        "probability_beats_hold": 1.0,
                        "probability_beats_cash": 1.0,
                        "valid_constraints": True,
                        "validation_errors": "",
                    },
                ]
            ),
            "candidate_map": {"HOLD_CURRENT": hold_candidate, "ACTIVE": active_candidate},
            "selection_config": {
                "optimization_objective": "robust_scenario_sharpe_objective",
                "hurdle": 0.0,
                "risk_premium_hurdle": 0.0,
                "p_hold_min": 0.0,
                "p_cash_min": 0.0,
            },
        }
        selection = select_best_discrete_portfolio(scored)
        rows.append(
            _test_row(
                "robust_scenario_sharpe_objective_mode_selects_valid_candidate",
                "PASS" if selection["best_discrete_candidate_name"] == "ACTIVE" and selection["objective_score_column"] == "robust_scenario_sharpe_objective" else "FAIL",
                f"{selection['best_discrete_candidate_name']} via {selection['objective_score_column']}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("robust_scenario_sharpe_objective_mode_selects_valid_candidate", "FAIL", str(exc)))

    try:
        result = send_email_notification(
            subject="Default Gate",
            body="Body",
            recipient="test@example.com",
            dry_run=False,
            gate_status={"real_email_send_allowed": False, "reason": "preview_only"},
            settings={"EMAIL_PROVIDER": "smtp"},
        )
        rows.append(
            _test_row(
                "send_email_notification_default_gate_no_send",
                "PASS"
                if (
                    result["attempted"] is False
                    and result["sent"] is False
                    and result["reason"] == "preview_only_phase_gate"
                    and result["sanitized_error"] is None
                )
                else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("send_email_notification_default_gate_no_send", "FAIL", str(exc)))

    try:
        result = send_email_notification(
            subject="Dry Run",
            body="Body",
            recipient="test@example.com",
            dry_run=True,
            gate_status={"real_email_send_allowed": True, "reason": "send_allowed"},
            settings={"EMAIL_PROVIDER": "smtp"},
        )
        rows.append(
            _test_row(
                "send_email_notification_dry_run_no_send",
                "PASS"
                if (
                    result["attempted"] is False
                    and result["sent"] is False
                    and result["reason"] == "preview_only_phase_gate"
                )
                else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("send_email_notification_dry_run_no_send", "FAIL", str(exc)))

    try:
        result = send_email_notification(
            subject="Missing Recipient",
            body="Body",
            recipient="",
            dry_run=False,
            gate_status={"real_email_send_allowed": True, "reason": "send_allowed"},
            settings={
                "EMAIL_PROVIDER": "smtp",
                "SMTP_HOST": "smtp.example.com",
                "SMTP_PORT": 587,
                "EMAIL_FROM": "sender@example.com",
                "SMTP_USERNAME": "sender@example.com",
                "SMTP_PASSWORD": "placeholder",
            },
        )
        rows.append(
            _test_row(
                "send_email_notification_missing_recipient_no_send",
                "PASS"
                if (
                    result["attempted"] is False
                    and result["sent"] is False
                    and result["reason"] == "provider_rejected"
                    and result["delivery_status"] == "recipient_missing"
                )
                else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("send_email_notification_missing_recipient_no_send", "FAIL", str(exc)))

    smtp_failure_result: dict[str, object] | None = None
    smtp_failure_secret = "SUPER_SECRET_TEST_VALUE_12345"
    original_smtp_password = os.environ.get("SMTP_PASSWORD")
    original_email_password = os.environ.get("EMAIL_PASSWORD")
    original_smtp_factory = notifications_module.smtplib.SMTP
    try:
        os.environ["SMTP_PASSWORD"] = smtp_failure_secret
        os.environ["EMAIL_PASSWORD"] = smtp_failure_secret

        sanitized_text = sanitize_for_output(
            f"Authorization: Bearer {smtp_failure_secret}\npassword={smtp_failure_secret}\nraw={smtp_failure_secret}"
        )
        rows.append(
            _test_row(
                "email_secret_sanitizer_masks_env_values",
                "PASS"
                if (
                    smtp_failure_secret not in sanitized_text
                    and ("Authorization: [REDACTED]" in sanitized_text or "Authorization=[REDACTED]" in sanitized_text)
                    and "password=[REDACTED]" in sanitized_text
                    and sanitized_text.count("[REDACTED]") >= 3
                )
                else "FAIL",
                sanitized_text,
            )
        )

        class _FailingSMTP:
            def __init__(self, host: str, port: int, timeout: int = 30) -> None:
                self.host = host
                self.port = port
                self.timeout = timeout

            def __enter__(self) -> "_FailingSMTP":
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
                return False

            def ehlo(self) -> tuple[int, bytes]:
                return 250, b"ok"

            def starttls(self) -> None:
                raise RuntimeError(f"SMTP auth failed with password={smtp_failure_secret}")

        notifications_module.smtplib.SMTP = _FailingSMTP
        smtp_failure_result = send_email_notification(
            subject="SMTP Failure",
            body="Body",
            recipient="test@example.com",
            dry_run=False,
            gate_status={"real_email_send_allowed": True, "reason": "send_allowed"},
            settings={
                "EMAIL_PROVIDER": "smtp",
                "SMTP_HOST": "smtp.example.com",
                "SMTP_PORT": 587,
                "EMAIL_FROM": "sender@example.com",
                "EMAIL_SENDER": "sender@example.com",
                "SMTP_USERNAME": "sender@example.com",
                "SMTP_PASSWORD": smtp_failure_secret,
                "EMAIL_USE_SSL": False,
                "EMAIL_USE_STARTTLS": True,
            },
        )
        rows.append(
            _test_row(
                "send_email_notification_smtp_failure_is_visible_and_sanitized",
                "PASS"
                if (
                    smtp_failure_result["attempted"] is True
                    and smtp_failure_result["sent"] is False
                    and smtp_failure_result["reason"] == "delivery_unknown"
                    and bool(smtp_failure_result["sanitized_error"])
                    and smtp_failure_secret not in str(smtp_failure_result["sanitized_error"])
                )
                else "FAIL",
                str(smtp_failure_result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("send_email_notification_smtp_failure_is_visible_and_sanitized", "FAIL", str(exc)))
    finally:
        notifications_module.smtplib.SMTP = original_smtp_factory
        if original_smtp_password is None:
            os.environ.pop("SMTP_PASSWORD", None)
        else:
            os.environ["SMTP_PASSWORD"] = original_smtp_password
        if original_email_password is None:
            os.environ.pop("EMAIL_PASSWORD", None)
        else:
            os.environ["EMAIL_PASSWORD"] = original_email_password

    try:
        result = send_email_notification(
            subject="Brevo Missing Key",
            body="Body",
            recipient="test@example.com",
            dry_run=False,
            gate_status={"real_email_send_allowed": True, "reason": "send_allowed"},
            settings={
                "EMAIL_PROVIDER": "brevo",
                "EMAIL_SENDER": "sender@example.com",
                "BREVO_API_KEY": "",
            },
        )
        rows.append(
            _test_row(
                "send_email_notification_brevo_missing_api_key_no_send",
                "PASS"
                if (
                    result["attempted"] is False
                    and result["sent"] is False
                    and result["reason"] == "provider_rejected"
                    and result["delivery_status"] == "brevo_api_key_missing"
                    and result["provider"] == "brevo"
                )
                else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("send_email_notification_brevo_missing_api_key_no_send", "FAIL", str(exc)))

    try:
        result = send_email_notification(
            subject="Brevo Missing Recipient",
            body="Body",
            recipient="",
            dry_run=False,
            gate_status={"real_email_send_allowed": True, "reason": "send_allowed"},
            settings={
                "EMAIL_PROVIDER": "brevo",
                "EMAIL_SENDER": "sender@example.com",
                "BREVO_API_KEY": "test-api-key",
            },
        )
        rows.append(
            _test_row(
                "send_email_notification_brevo_missing_recipient_no_send",
                "PASS"
                if (
                    result["attempted"] is False
                    and result["sent"] is False
                    and result["reason"] == "provider_rejected"
                    and result["delivery_status"] == "recipient_missing"
                    and result["provider"] == "brevo"
                )
                else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("send_email_notification_brevo_missing_recipient_no_send", "FAIL", str(exc)))

    try:
        result = send_email_notification(
            subject="Brevo Dry Run",
            body="Body",
            recipient="test@example.com",
            dry_run=True,
            gate_status={"real_email_send_allowed": True, "reason": "send_allowed"},
            settings={
                "EMAIL_PROVIDER": "brevo",
                "EMAIL_SENDER": "sender@example.com",
                "BREVO_API_KEY": "test-api-key",
            },
        )
        rows.append(
            _test_row(
                "send_email_notification_brevo_dry_run_no_send",
                "PASS"
                if (
                    result["attempted"] is False
                    and result["sent"] is False
                    and result["reason"] == "preview_only_phase_gate"
                    and result["provider"] == "brevo"
                )
                else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("send_email_notification_brevo_dry_run_no_send", "FAIL", str(exc)))

    original_urlopen = notifications_module.urllib_request.urlopen
    try:
        class _BrevoSuccessResponse:
            status = 201

            def __enter__(self) -> "_BrevoSuccessResponse":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def getcode(self) -> int:
                return 201

            def read(self) -> bytes:
                return b'{\"messageId\":\"brevo-test-id\"}'

        def _fake_brevo_success(request: object, timeout: int = 30) -> _BrevoSuccessResponse:
            return _BrevoSuccessResponse()

        notifications_module.urllib_request.urlopen = _fake_brevo_success
        result = send_email_notification(
            subject="Brevo Success",
            body="Body",
            recipient="test@example.com",
            dry_run=False,
            gate_status={"real_email_send_allowed": True, "reason": "send_allowed"},
            settings={
                "EMAIL_PROVIDER": "brevo",
                "EMAIL_SENDER": "sender@example.com",
                "BREVO_API_KEY": "test-api-key",
            },
        )
        rows.append(
            _test_row(
                "send_email_notification_brevo_success",
                "PASS"
                if (
                    result["attempted"] is True
                    and result["sent"] is True
                    and result["provider_accepted"] is True
                    and result["delivery_confirmed"] is False
                    and result["reason"] == "provider_accepted_delivery_unconfirmed"
                    and result["provider_message_id"] == "brevo-test-id"
                    and result["provider"] == "brevo"
                )
                else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("send_email_notification_brevo_success", "FAIL", str(exc)))
    finally:
        notifications_module.urllib_request.urlopen = original_urlopen

    try:
        failure_secret = "BREVO_SECRET_FAILURE_TEST_67890"

        def _fake_brevo_failure(request: object, timeout: int = 30) -> object:
            raise notifications_module.urllib_error.HTTPError(
                url="https://api.brevo.com/v3/smtp/email",
                code=401,
                msg=f"Unauthorized api-key={failure_secret}",
                hdrs=None,
                fp=io.BytesIO(b'{"message":"invalid api key"}'),
            )

        notifications_module.urllib_request.urlopen = _fake_brevo_failure
        result = send_email_notification(
            subject="Brevo Failure",
            body="Body",
            recipient="test@example.com",
            dry_run=False,
            gate_status={"real_email_send_allowed": True, "reason": "send_allowed"},
            settings={
                "EMAIL_PROVIDER": "brevo",
                "EMAIL_SENDER": "sender@example.com",
                "BREVO_API_KEY": failure_secret,
            },
        )
        rows.append(
            _test_row(
                "send_email_notification_brevo_failure",
                "PASS"
                if (
                    result["attempted"] is True
                    and result["sent"] is False
                    and result["reason"] == "provider_rejected"
                    and result["delivery_status"] == "provider_rejected"
                    and result["error_type"] == "HTTPError"
                    and bool(result["sanitized_error"])
                    and failure_secret not in str(result["sanitized_error"])
                )
                else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("send_email_notification_brevo_failure", "FAIL", str(exc)))
    finally:
        notifications_module.urllib_request.urlopen = original_urlopen

    try:
        result = send_email_notification(
            subject="Fake Success",
            body="Body",
            recipient="test@example.com",
            dry_run=False,
            gate_status={"real_email_send_allowed": True, "reason": "send_allowed"},
            settings={"EMAIL_PROVIDER": "fake", "EMAIL_FAKE_SEND_SUCCESS": True},
        )
        rows.append(
            _test_row(
                "send_email_notification_fake_provider_success",
                "PASS"
                if (
                    result["attempted"] is True
                    and result["sent"] is True
                    and result["reason"] == "fake_send_success"
                )
                else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("send_email_notification_fake_provider_success", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            review_payload = {
                "run_status": {
                    "review_date": "2026-05-05",
                    "review_time_berlin": "16:00:00",
                    "current_date_berlin": "2026-05-05",
                    "current_time_berlin": "16:00:00",
                    "is_project_trading_day": True,
                    "within_allowed_window": True,
                    "execution_allowed_by_calendar": True,
                    "final_action": "HOLD",
                    "execution_mode": "order_preview_only",
                },
                "data_status": {
                    "data_source": "yfinance",
                    "cache_status": "live",
                    "synthetic_data": False,
                    "used_cache_fallback": False,
                    "latest_price_date": "2026-05-05",
                    "staleness_days": 0,
                    "data_freshness_ok": True,
                    "live_data_error": "",
                    "missing_prices": [],
                    "price_basis": "adjusted_close_proxy",
                },
                "current_portfolio": {
                    "current_portfolio_source": "csv",
                    "positions_count": 1,
                    "cash_usd": 10.0,
                    "invested_market_value_usd": 90.0,
                    "nav_usd": 100.0,
                    "current_portfolio_100pct_cash": False,
                    "current_weights_sum_including_cash": 1.0,
                    "current_weights_sum_without_cash": 0.9,
                    "parser_warnings": [],
                    "parser_errors": [],
                },
                "current_positions": [
                    {
                        "ticker": "SGOV",
                        "current_shares": 1.0,
                        "latest_price": 90.0,
                        "latest_price_date": "2026-05-05",
                        "market_value_usd": 90.0,
                        "current_weight": 0.9,
                        "price_basis": "adjusted_close_proxy",
                        "data_source": "yfinance",
                        "stale_price_warning": False,
                        "data_warning": "",
                    }
                ],
                "target_allocation": [],
                "delta_transactions": [],
                "cost_edge": {
                    "simulator_fee_usd": 0.0,
                    "total_simulator_fees_usd": 0.0,
                    "modeled_transaction_costs_usd": 0.0,
                    "modeled_transaction_costs_pct_nav": 0.0,
                    "current_portfolio_score": 0.1,
                    "target_score_before_costs": 0.1,
                    "target_score_after_costs": 0.1,
                    "delta_score_vs_current": 0.0,
                    "execution_buffer": 0.001,
                    "model_uncertainty_buffer": 0.001,
                    "trade_now_edge": -0.001,
                    "cost_model_used": "no_orders",
                },
                "decision_context": {
                    "why_this_target": "test",
                    "why_not_hold": "test",
                    "why_not_cash": "test",
                    "trade_decision_reason": "test",
                    "positive_drivers": [],
                    "negative_drivers": ["trade_now_edge negative"],
                    "rejected_candidates": [],
                    "main_blocker_category": "costs/edge",
                },
                "pre_trade_validation_status": "PASS",
                "preview_only": True,
                "manual_orders_preview_ready": False,
                "cash_after_orders": 10.0,
                "main_daily_scope_differs": False,
                "exception_message": "",
            }
            write_daily_portfolio_review_outputs(
                review_payload,
                output_dir=tmp_dir,
                email_result=dict(smtp_failure_result or {}),
            )
            written_text = "\n".join(
                file_path.read_text(encoding="utf-8")
                for file_path in sorted(tmp_dir.glob("*.txt"))
            )
            written_json = "\n".join(
                file_path.read_text(encoding="utf-8")
                for file_path in sorted(tmp_dir.glob("*.json"))
            )
        rows.append(
            _test_row(
                "email_outputs_do_not_contain_known_test_secret",
                "PASS"
                if (
                    smtp_failure_secret not in written_text
                    and smtp_failure_secret not in written_json
                    and "[REDACTED]" in written_text
                )
                else "FAIL",
                "checked generated txt/json outputs for secret leakage",
            )
        )
    except Exception as exc:
        rows.append(_test_row("email_outputs_do_not_contain_known_test_secret", "FAIL", str(exc)))

    def _sample_daily_review_payload(*, final_action: str = "HOLD", trade_reason: str = "test", manual_order_count: int = 0) -> dict[str, object]:
        return {
            "run_status": {
                "review_date": "2026-05-05",
                "review_time_berlin": "16:00:00",
                "current_date_berlin": "2026-05-05",
                "current_time_berlin": "16:00:00",
                "is_project_trading_day": True,
                "within_allowed_window": True,
                "execution_allowed_by_calendar": True,
                "final_action": final_action,
                "execution_mode": "order_preview_only",
            },
            "data_status": {
                "data_source": "yfinance",
                "cache_status": "live",
                "synthetic_data": False,
                "used_cache_fallback": False,
                "latest_price_date": "2026-05-05",
                "staleness_days": 0,
                "data_freshness_ok": True,
                "live_data_error": "",
                "missing_prices": [],
                "price_basis": "adjusted_close_proxy",
            },
            "current_portfolio": {
                "current_portfolio_source": "csv",
                "positions_count": 1,
                "cash_usd": 10.0,
                "invested_market_value_usd": 90.0,
                "nav_usd": 100.0,
                "current_portfolio_100pct_cash": False,
                "current_weights_sum_including_cash": 1.0,
                "current_weights_sum_without_cash": 0.9,
                "parser_warnings": [],
                "parser_errors": [],
            },
            "current_positions": [
                {
                    "ticker": "SGOV",
                    "current_shares": 1.0,
                    "latest_price": 90.0,
                    "latest_price_date": "2026-05-05",
                    "market_value_usd": 90.0,
                    "current_weight": 0.9,
                    "price_basis": "adjusted_close_proxy",
                    "data_source": "yfinance",
                    "stale_price_warning": False,
                    "data_warning": "",
                }
            ],
            "target_allocation": [],
            "delta_transactions": [],
            "order_summary": {
                "manual_eligible_order_count": manual_order_count,
                "order_count": manual_order_count,
            },
            "cost_edge": {
                "simulator_fee_usd": 0.0,
                "total_simulator_fees_usd": 0.0,
                "modeled_transaction_costs_usd": 0.0,
                "modeled_transaction_costs_pct_nav": 0.0,
                "current_portfolio_score": 0.1,
                "target_score_before_costs": 0.1,
                "target_score_after_costs": 0.1,
                "delta_score_vs_current": 0.0,
                "execution_buffer": 0.001,
                "model_uncertainty_buffer": 0.001,
                "trade_now_edge": -0.001,
                "cost_model_used": "no_orders",
            },
            "decision_context": {
                "why_this_target": "test",
                "why_not_hold": "test",
                "why_not_cash": "test",
                "trade_decision_reason": trade_reason,
                "positive_drivers": [],
                "negative_drivers": ["trade_now_edge negative"],
                "rejected_candidates": [],
                "main_blocker_category": "costs/edge",
            },
            "pre_trade_validation_status": "PASS",
            "preview_only": True,
            "manual_orders_preview_ready": False,
            "cash_after_orders": 10.0,
            "main_daily_scope_differs": False,
            "exception_message": "",
        }

    email_send_ready_settings = {
        "enable_email_notifications": True,
        "email_send_enabled": True,
        "email_dry_run": False,
        "daily_briefing_only": True,
        "max_emails_per_day": 1,
        "email_recipient": "test@example.com",
        "user_confirmed_email_phase": True,
        "phase": "DAILY_REVIEW_SEND_READY",
        "enable_external_broker": False,
        "enable_investopedia_simulator": False,
        "enable_local_paper_trading": False,
    }

    def _fake_provider_settings(fake_send_success: bool) -> dict[str, object]:
        return {
            **email_send_ready_settings,
            "email_provider": "fake",
            "EMAIL_PROVIDER": "fake",
            "EMAIL_FAKE_SEND_SUCCESS": fake_send_success,
            "EMAIL_RECIPIENT": str(email_send_ready_settings["email_recipient"]),
        }

    def _brevo_provider_settings(api_key: str = "test-api-key") -> dict[str, object]:
        return {
            **email_send_ready_settings,
            "email_provider": "brevo",
            "EMAIL_PROVIDER": "brevo",
            "EMAIL_RECIPIENT": str(email_send_ready_settings["email_recipient"]),
            "EMAIL_SENDER": "sender@example.com",
            "BREVO_API_KEY": api_key,
        }

    def _sample_trade_review_payload() -> dict[str, object]:
        payload = _sample_daily_review_payload(final_action="BUY", trade_reason="delta trade available", manual_order_count=1)
        payload["run_status"]["final_action"] = "BUY"
        payload["run_status"]["execution_mode"] = "order_preview_only"
        payload["target_allocation"] = [
            {
                "ticker": "SGOV",
                "target_weight": 0.50,
                "target_shares": 1.0,
                "target_market_value_usd": 90.0,
                "continuous_target_weight": 0.50,
            },
            {
                "ticker": "XLK",
                "target_weight": 0.40,
                "target_shares": 1.0,
                "target_market_value_usd": 40.0,
                "continuous_target_weight": 0.40,
            },
        ]
        payload["delta_transactions"] = [
            {
                "ticker": "XLK",
                "action": "BUY",
                "current_shares": 0.0,
                "target_shares": 1.0,
                "order_shares": 1.0,
                "estimated_price": 40.0,
                "estimated_order_value": 40.0,
                "simulator_fee_usd": 0.0,
                "modeled_transaction_cost_usd": 0.1,
                "preview_only": True,
                "not_executable": False,
                "execution_block_reason": "",
            }
        ]
        payload["order_summary"]["order_count"] = 1
        payload["order_summary"]["manual_eligible_order_count"] = 1
        payload["manual_orders_preview_ready"] = True
        payload["cost_edge"]["trade_now_edge"] = 0.002
        return payload

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            review_payload = _sample_daily_review_payload()
            write_daily_portfolio_review_outputs(review_payload, output_dir=tmp_dir)
            html_path = tmp_dir / "daily_review_email.html"
            tex_path = tmp_dir / "daily_review_report.tex"
            allocation_chart = tmp_dir / "charts" / "current_portfolio_allocation.png"
            weights_chart = tmp_dir / "charts" / "current_vs_target_weights.png"
            html_exists = html_path.exists()
            tex_exists = tex_path.exists()
            allocation_chart_exists = allocation_chart.exists()
            weights_chart_exists = weights_chart.exists()
            html_text = html_path.read_text(encoding="utf-8")
            pdf_expected = bool(daily_review_rendering_module.shutil.which("pdflatex"))
            pdf_exists = (tmp_dir / "daily_review_report.pdf").exists()
        rows.append(
            _test_row(
                "daily_review_render_artifacts_hold_preview_created",
                "PASS"
                if (
                    html_exists
                    and tex_exists
                    and allocation_chart_exists
                    and weights_chart_exists
                    and "Keine BUY/SELL-Delta-Orders." in html_text
                    and (pdf_exists if pdf_expected else True)
                )
                else "FAIL",
                str(
                    {
                        "html_exists": html_exists,
                        "tex_exists": tex_exists,
                        "allocation_chart_exists": allocation_chart_exists,
                        "weights_chart_exists": weights_chart_exists,
                        "pdf_expected": pdf_expected,
                        "pdf_exists": pdf_exists,
                    }
                ),
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_review_render_artifacts_hold_preview_created", "FAIL", str(exc)))

    try:
        original_which = daily_review_rendering_module.shutil.which
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            daily_review_rendering_module.shutil.which = lambda name: None if name == "pdflatex" else original_which(name)
            write_daily_portfolio_review_outputs(_sample_daily_review_payload(), output_dir=tmp_dir)
            tex_exists = (tmp_dir / "daily_review_report.tex").exists()
            pdf_exists = (tmp_dir / "daily_review_report.pdf").exists()
            safety_text = (tmp_dir / "email_safety_report.txt").read_text(encoding="utf-8")
        rows.append(
            _test_row(
                "daily_review_render_artifacts_tex_fallback_without_pdflatex",
                "PASS"
                if tex_exists and not pdf_exists and "pdflatex unavailable; PDF build skipped." in safety_text
                else "FAIL",
                str({"tex_exists": tex_exists, "pdf_exists": pdf_exists}),
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_review_render_artifacts_tex_fallback_without_pdflatex", "FAIL", str(exc)))
    finally:
        daily_review_rendering_module.shutil.which = original_which

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            write_daily_portfolio_review_outputs(_sample_trade_review_payload(), output_dir=tmp_dir)
            html_text = (tmp_dir / "daily_review_email.html").read_text(encoding="utf-8")
        rows.append(
            _test_row(
                "daily_review_render_artifacts_buy_sell_html_table",
                "PASS"
                if ("<table" in html_text and "BUY" in html_text and "XLK" in html_text)
                else "FAIL",
                "checked HTML order table rendering for BUY/SELL case",
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_review_render_artifacts_buy_sell_html_table", "FAIL", str(exc)))

    try:
        result = send_email_notification(
            subject="Fake Blocked",
            body="Body",
            recipient="test@example.com",
            dry_run=False,
            gate_status={"real_email_send_allowed": False, "reason": "preview_only"},
            settings={"EMAIL_PROVIDER": "fake", "EMAIL_FAKE_SEND_SUCCESS": True},
        )
        rows.append(
            _test_row(
                "send_email_notification_fake_provider_blocked_by_gate",
                "PASS"
                if (
                    result["attempted"] is False
                    and result["sent"] is False
                    and result["reason"] == "preview_only_phase_gate"
                    and result["provider"] == "fake"
                )
                else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("send_email_notification_fake_provider_blocked_by_gate", "FAIL", str(exc)))

    try:
        result = send_email_notification(
            subject="Fake Failure",
            body="Body",
            recipient="test@example.com",
            dry_run=False,
            gate_status={"real_email_send_allowed": True, "reason": "send_allowed"},
            settings={"EMAIL_PROVIDER": "fake", "EMAIL_FAKE_SEND_SUCCESS": False},
        )
        rows.append(
            _test_row(
                "send_email_notification_fake_provider_failure",
                "PASS"
                if (
                    result["attempted"] is True
                    and result["sent"] is False
                    and result["reason"] == "fake_send_failure"
                    and result["error_type"] == "FakeEmailProviderError"
                    and bool(result["sanitized_error"])
                )
                else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("send_email_notification_fake_provider_failure", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            call_count = {"count": 0}

            class _BrevoSuccessResponse:
                status = 201

                def __enter__(self) -> "_BrevoSuccessResponse":
                    return self

                def __exit__(self, exc_type, exc, tb) -> bool:
                    return False

                def getcode(self) -> int:
                    return 201

                def read(self) -> bytes:
                    return b'{\"messageId\":\"brevo-test-id\"}'

            def _counted_brevo_success(request: object, timeout: int = 30) -> _BrevoSuccessResponse:
                call_count["count"] += 1
                return _BrevoSuccessResponse()

            notifications_module.urllib_request.urlopen = _counted_brevo_success
            success_settings = _brevo_provider_settings()
            review_payload = _sample_daily_review_payload()
            first_result = send_daily_review_email_if_needed(review_payload, output_dir=tmp_dir, settings=success_settings)
            write_daily_portfolio_review_outputs(review_payload, output_dir=tmp_dir, email_result=first_result)
            state_after_first = json.loads((tmp_dir / "last_email_state.json").read_text(encoding="utf-8"))

            second_result = send_daily_review_email_if_needed(review_payload, output_dir=tmp_dir, settings=success_settings)
            write_daily_portfolio_review_outputs(review_payload, output_dir=tmp_dir, email_result=second_result)
            state_after_second = json.loads((tmp_dir / "last_email_state.json").read_text(encoding="utf-8"))
        rows.append(
            _test_row(
                "daily_review_email_brevo_success_updates_state_and_dedupe",
                "PASS"
                if (
                    first_result["attempted"] is True
                    and first_result["sent"] is True
                    and first_result["reason"] == "provider_accepted_delivery_unconfirmed"
                    and first_result["provider_accepted"] is True
                    and first_result["delivery_confirmed"] is False
                    and first_result["provider"] == "brevo"
                    and state_after_first.get("last_send_success") is True
                    and state_after_first.get("last_provider_accept_success") is True
                    and state_after_first.get("last_delivery_confirmed_success") is False
                    and second_result["attempted"] is False
                    and second_result["sent"] is False
                    and second_result["reason"] == "already_sent_same_decision_today"
                    and state_after_second.get("last_send_success") is True
                    and call_count["count"] == 1
                )
                else "FAIL",
                str(
                    {
                        "first_result": first_result,
                        "second_result": second_result,
                        "state_after_first": state_after_first,
                        "state_after_second": state_after_second,
                        "call_count": call_count,
                    }
                ),
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_review_email_brevo_success_updates_state_and_dedupe", "FAIL", str(exc)))
    finally:
        notifications_module.urllib_request.urlopen = original_urlopen

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            failure_secret = "BREVO_FAILURE_STATE_SECRET_999"

            def _counted_brevo_failure(request: object, timeout: int = 30) -> object:
                raise notifications_module.urllib_error.HTTPError(
                    url="https://api.brevo.com/v3/smtp/email",
                    code=500,
                    msg=f"Server error bearer {failure_secret}",
                    hdrs=None,
                    fp=io.BytesIO(b'{"message":"temporary provider failure"}'),
                )

            notifications_module.urllib_request.urlopen = _counted_brevo_failure
            failure_settings = _brevo_provider_settings(api_key=failure_secret)
            retry_settings = _brevo_provider_settings(api_key="retry-api-key")
            review_payload = _sample_daily_review_payload()
            failure_result = send_daily_review_email_if_needed(review_payload, output_dir=tmp_dir, settings=failure_settings)
            write_daily_portfolio_review_outputs(review_payload, output_dir=tmp_dir, email_result=failure_result)
            state_after_failure = json.loads((tmp_dir / "last_email_state.json").read_text(encoding="utf-8"))

            class _BrevoRetrySuccessResponse:
                status = 201

                def __enter__(self) -> "_BrevoRetrySuccessResponse":
                    return self

                def __exit__(self, exc_type, exc, tb) -> bool:
                    return False

                def getcode(self) -> int:
                    return 201

                def read(self) -> bytes:
                    return b'{\"messageId\":\"brevo-retry-id\"}'

            def _brevo_retry_success(request: object, timeout: int = 30) -> _BrevoRetrySuccessResponse:
                return _BrevoRetrySuccessResponse()

            notifications_module.urllib_request.urlopen = _brevo_retry_success
            retry_result = send_daily_review_email_if_needed(review_payload, output_dir=tmp_dir, settings=retry_settings)
            write_daily_portfolio_review_outputs(review_payload, output_dir=tmp_dir, email_result=retry_result)
            state_after_retry = json.loads((tmp_dir / "last_email_state.json").read_text(encoding="utf-8"))
        rows.append(
            _test_row(
                "daily_review_email_brevo_failure_does_not_mark_sent_and_allows_retry",
                "PASS"
                if (
                    failure_result["attempted"] is True
                    and failure_result["sent"] is False
                    and failure_result["reason"] == "provider_rejected"
                    and failure_result["provider"] == "brevo"
                    and failure_secret not in str(failure_result.get("sanitized_error", ""))
                    and state_after_failure.get("last_send_success") is False
                    and state_after_failure.get("last_provider_accept_success") is False
                    and state_after_failure.get("last_delivery_confirmed_success") is False
                    and state_after_failure.get("last_sent_date", "") in {"", None}
                    and retry_result["attempted"] is True
                    and retry_result["sent"] is True
                    and retry_result["reason"] == "provider_accepted_delivery_unconfirmed"
                    and state_after_retry.get("last_send_success") is True
                )
                else "FAIL",
                str(
                    {
                        "failure_result": failure_result,
                        "retry_result": retry_result,
                        "state_after_failure": state_after_failure,
                        "state_after_retry": state_after_retry,
                    }
                ),
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_review_email_brevo_failure_does_not_mark_sent_and_allows_retry", "FAIL", str(exc)))
    finally:
        notifications_module.urllib_request.urlopen = original_urlopen

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            success_settings = _fake_provider_settings(True)
            review_payload = _sample_daily_review_payload()
            first_result = send_daily_review_email_if_needed(review_payload, output_dir=tmp_dir, settings=success_settings)
            write_daily_portfolio_review_outputs(review_payload, output_dir=tmp_dir, email_result=first_result)
            state_after_first = json.loads((tmp_dir / "last_email_state.json").read_text(encoding="utf-8"))

            second_result = send_daily_review_email_if_needed(review_payload, output_dir=tmp_dir, settings=success_settings)
            write_daily_portfolio_review_outputs(review_payload, output_dir=tmp_dir, email_result=second_result)
            state_after_second = json.loads((tmp_dir / "last_email_state.json").read_text(encoding="utf-8"))

            changed_hash_payload = _sample_daily_review_payload(final_action="WAIT", trade_reason="changed body")
            third_result = send_daily_review_email_if_needed(changed_hash_payload, output_dir=tmp_dir, settings=success_settings)
        rows.append(
            _test_row(
                "daily_review_email_fake_provider_success_updates_state_and_dedupe",
                "PASS"
                if (
                    first_result["sent"] is True
                    and first_result["reason"] == "fake_send_success"
                    and first_result["provider"] == "fake"
                    and state_after_first.get("last_send_success") is True
                    and state_after_first.get("last_sent_date") == "2026-05-05"
                    and bool(state_after_first.get("last_body_hash"))
                    and second_result["attempted"] is False
                    and second_result["sent"] is False
                    and second_result["reason"] == "already_sent_same_decision_today"
                    and state_after_second.get("last_send_success") is True
                    and state_after_second.get("last_sent_date") == "2026-05-05"
                    and third_result["attempted"] is False
                    and third_result["sent"] is False
                    and third_result["reason"] == "max_emails_per_day_reached"
                )
                else "FAIL",
                str(
                    {
                        "first_result": first_result,
                        "second_result": second_result,
                        "third_result": third_result,
                        "state_after_first": state_after_first,
                        "state_after_second": state_after_second,
                    }
                ),
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_review_email_fake_provider_success_updates_state_and_dedupe", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            review_payload = _sample_daily_review_payload()
            failure_settings = _fake_provider_settings(False)
            retry_settings = _fake_provider_settings(True)
            failure_result = send_daily_review_email_if_needed(review_payload, output_dir=tmp_dir, settings=failure_settings)
            write_daily_portfolio_review_outputs(review_payload, output_dir=tmp_dir, email_result=failure_result)
            state_after_failure = json.loads((tmp_dir / "last_email_state.json").read_text(encoding="utf-8"))

            retry_result = send_daily_review_email_if_needed(review_payload, output_dir=tmp_dir, settings=retry_settings)
            write_daily_portfolio_review_outputs(review_payload, output_dir=tmp_dir, email_result=retry_result)
            state_after_retry = json.loads((tmp_dir / "last_email_state.json").read_text(encoding="utf-8"))
        rows.append(
            _test_row(
                "daily_review_email_fake_provider_failure_does_not_mark_sent_and_allows_retry",
                "PASS"
                if (
                    failure_result["attempted"] is True
                    and failure_result["sent"] is False
                    and failure_result["reason"] == "fake_send_failure"
                    and failure_result["provider"] == "fake"
                    and state_after_failure.get("last_send_success") is False
                    and state_after_failure.get("last_sent_date", "") in {"", None}
                    and state_after_failure.get("last_attempt_date") == "2026-05-05"
                    and state_after_failure.get("last_reason") == "fake_send_failure"
                    and state_after_failure.get("last_error_type") == "FakeEmailProviderError"
                    and retry_result["attempted"] is True
                    and retry_result["sent"] is True
                    and retry_result["reason"] == "fake_send_success"
                    and state_after_retry.get("last_send_success") is True
                    and state_after_retry.get("last_sent_date") == "2026-05-05"
                )
                else "FAIL",
                str(
                    {
                        "failure_result": failure_result,
                        "retry_result": retry_result,
                        "state_after_failure": state_after_failure,
                        "state_after_retry": state_after_retry,
                    }
                ),
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_review_email_fake_provider_failure_does_not_mark_sent_and_allows_retry", "FAIL", str(exc)))

    try:
        payload_a = _sample_daily_review_payload()
        payload_b = json.loads(json.dumps(payload_a))
        payload_b["run_status"]["current_time_berlin"] = "16:59:59"
        payload_b["run_status"]["review_time_berlin"] = "16:59:59"
        fp_a = build_decision_fingerprint(payload_a, settings={"email_recipient": "test@example.com"}, recipient="test@example.com")
        fp_b = build_decision_fingerprint(payload_b, settings={"email_recipient": "test@example.com"}, recipient="test@example.com")
        rows.append(
            _test_row(
                "current_time_change_does_not_change_decision_fingerprint",
                "PASS" if fp_a == fp_b else "FAIL",
                f"{fp_a} vs {fp_b}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("current_time_change_does_not_change_decision_fingerprint", "FAIL", str(exc)))

    try:
        payload_a = _sample_daily_review_payload()
        payload_b = json.loads(json.dumps(payload_a))
        payload_b["delta_transactions"] = [
            {
                "ticker": "XLK",
                "action": "BUY",
                "order_shares": 1.0,
                "estimated_price": 40.0,
                "estimated_order_value": 40.0,
            }
        ]
        payload_b["order_summary"]["order_count"] = 1
        payload_b["order_summary"]["manual_eligible_order_count"] = 1
        fp_a = build_decision_fingerprint(payload_a, settings={"email_recipient": "test@example.com"}, recipient="test@example.com")
        fp_b = build_decision_fingerprint(payload_b, settings={"email_recipient": "test@example.com"}, recipient="test@example.com")
        rows.append(
            _test_row(
                "changed_orders_change_decision_fingerprint",
                "PASS" if fp_a != fp_b else "FAIL",
                f"{fp_a} vs {fp_b}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("changed_orders_change_decision_fingerprint", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            settings = _fake_provider_settings(True)
            payload = _sample_daily_review_payload()
            first_result = send_daily_review_email_if_needed(payload, output_dir=tmp_dir, settings=settings)
            write_daily_portfolio_review_outputs(payload, output_dir=tmp_dir, email_result=first_result)
            second_payload = json.loads(json.dumps(payload))
            second_payload["run_status"]["current_time_berlin"] = "17:01:00"
            second_result = send_daily_review_email_if_needed(second_payload, output_dir=tmp_dir, settings=settings)
        rows.append(
            _test_row(
                "same_decision_same_day_does_not_resend",
                "PASS"
                if (
                    first_result["provider_accepted"] is True
                    and second_result["attempted"] is False
                    and second_result["reason"] == "already_sent_same_decision_today"
                )
                else "FAIL",
                str({"first_result": first_result, "second_result": second_result}),
            )
        )
    except Exception as exc:
        rows.append(_test_row("same_decision_same_day_does_not_resend", "FAIL", str(exc)))

    try:
        result = send_email_notification(
            subject="Fake Success",
            body="Body",
            recipient="test@example.com",
            dry_run=False,
            gate_status={"real_email_send_allowed": True, "reason": "send_allowed"},
            settings={"EMAIL_PROVIDER": "fake", "EMAIL_FAKE_SEND_SUCCESS": True},
        )
        rows.append(
            _test_row(
                "provider_accepted_is_not_delivery_confirmed",
                "PASS"
                if result["provider_accepted"] is True and result["delivery_confirmed"] is False
                else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("provider_accepted_is_not_delivery_confirmed", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            payload = _sample_daily_review_payload()
            failure_result = send_daily_review_email_if_needed(payload, output_dir=tmp_dir, settings=_fake_provider_settings(False))
            write_daily_portfolio_review_outputs(payload, output_dir=tmp_dir, email_result=failure_result)
            state_after_failure = json.loads((tmp_dir / "last_email_state.json").read_text(encoding="utf-8"))
        rows.append(
            _test_row(
                "failed_delivery_does_not_mark_delivery_success",
                "PASS"
                if (
                    failure_result["provider_accepted"] is False
                    and failure_result["delivery_confirmed"] is False
                    and state_after_failure.get("last_provider_accept_success") is False
                    and state_after_failure.get("last_delivery_confirmed_success") is False
                )
                else "FAIL",
                str({"failure_result": failure_result, "state": state_after_failure}),
            )
        )
    except Exception as exc:
        rows.append(_test_row("failed_delivery_does_not_mark_delivery_success", "FAIL", str(exc)))

    try:
        secret_value = "BREVO_CRON_SECRET_1234567890"
        previous_secret = os.environ.get("BREVO_API_KEY")
        os.environ["BREVO_API_KEY"] = secret_value
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            write_daily_portfolio_review_outputs(_sample_daily_review_payload(), output_dir=tmp_dir)
            safety_text = (tmp_dir / "email_safety_report.txt").read_text(encoding="utf-8")
            diagnosis_text = (tmp_dir / "email_delivery_diagnosis_report.txt").read_text(encoding="utf-8")
        rows.append(
            _test_row(
                "cron_env_diagnostics_hide_secrets",
                "PASS"
                if (
                    secret_value not in safety_text
                    and secret_value not in diagnosis_text
                    and "BREVO_API_KEY configured:" in safety_text
                    and "SMTP_PASSWORD configured:" in diagnosis_text
                )
                else "FAIL",
                diagnosis_text,
            )
        )
    except Exception as exc:
        rows.append(_test_row("cron_env_diagnostics_hide_secrets", "FAIL", str(exc)))
    finally:
        if "previous_secret" in locals():
            if previous_secret is None:
                os.environ.pop("BREVO_API_KEY", None)
            else:
                os.environ["BREVO_API_KEY"] = previous_secret

    try:
        payload = _sample_daily_review_payload()
        body = build_daily_email_briefing(payload, build_review_issues(payload))
        rows.append(
            _test_row(
                "email_body_references_briefing_files",
                "PASS"
                if "outputs/daily_portfolio_briefing.md" in body and "outputs/daily_portfolio_briefing.html" in body
                else "FAIL",
                body,
            )
        )
    except Exception as exc:
        rows.append(_test_row("email_body_references_briefing_files", "FAIL", str(exc)))

    try:
        mu = pd.Series(0.01, index=params["tickers"], dtype=float)
        sigma = pd.DataFrame(0.0, index=params["tickers"], columns=params["tickers"], dtype=float)
        for ticker in sigma.index:
            sigma.loc[ticker, ticker] = 0.02
        w0 = build_feasible_initial_weights(params["tickers"], params)
        result = optimize_allocation(
            mu=mu,
            Sigma=sigma,
            w_current=w0,
            params={**params, "max_equity_like_total": params["max_equity_like_total_normal"], "min_defensive_weight": params["min_defensive_weight_normal"]},
        )
        rows.append(_test_row("missing_gurobi_scipy_fallback", "PASS", result.solver_name))
        rows.append(_test_row("weights_sum_to_1", "PASS" if abs(float(result.target_weights.sum()) - 1.0) < 1e-6 else "FAIL", f"{float(result.target_weights.sum()):.6f}"))
        rows.append(_test_row("no_negative_weights", "PASS" if bool((result.target_weights >= -1e-10).all()) else "FAIL", "checked"))
    except Exception as exc:
        rows.append(_test_row("missing_gurobi_scipy_fallback", "FAIL", str(exc)))

    dummy_score = type("DummyScore", (), {"delta_vs_hold": 0.01, "estimated_cost": 0.0, "delta_vs_cash": 0.01, "probability_beats_hold": 0.7, "probability_beats_cash": 0.7})()
    dummy_candidate = type("DummyCandidate", (), {"name": "OPTIMIZER_TARGET"})()
    dummy_selection = type("DummySelection", (), {"selected_score": dummy_score, "selected_candidate": dummy_candidate})()
    gate = evaluate_execution_gate(dummy_selection, synthetic_data=True)
    rows.append(_test_row("synthetic_data_blocks_orders", "PASS" if gate.action == "PAUSE" else "FAIL", gate.reason))

    tradability_df = pd.DataFrame(
        [
            {"ticker": "SGOV", "final_allowed": False},
            {"ticker": "SHY", "final_allowed": True},
            {"ticker": "XLK", "final_allowed": True},
            {"ticker": "XLI", "final_allowed": True},
            {"ticker": "TIP", "final_allowed": True},
            {"ticker": "PDBC", "final_allowed": True},
            {"ticker": "SPHQ", "final_allowed": True},
            {"ticker": "SPLV", "final_allowed": True},
            {"ticker": "AGG", "final_allowed": True},
            {"ticker": "LQD", "final_allowed": True},
            {"ticker": "IEF", "final_allowed": True},
        ]
    )
    cash_proxy = select_cash_proxy(["SHY", "XLK", "XLI", "TIP", "PDBC", "SPHQ", "SPLV", "AGG", "LQD", "IEF"], tradability_df)
    rows.append(_test_row("missing_cash_ticker_fallback", "PASS" if cash_proxy == "SHY" else "FAIL", str(cash_proxy)))

    try:
        filtered = apply_tradability_filter(
            ["SGOV", "SHY", "XLK", "XLI", "TIP", "PDBC", "SPHQ", "SPLV", "AGG", "LQD", "IEF"],
            tradability_df,
            min_assets=10,
        )
        rows.append(_test_row("non_tradable_etf_removed", "PASS" if "SGOV" not in filtered else "FAIL", ",".join(filtered)))
    except Exception as exc:
        rows.append(_test_row("non_tradable_etf_removed", "FAIL", str(exc)))

    try:
        idx = pd.Index(params["tickers"], name="ticker")
        forecast_table = pd.DataFrame(
            {
                "expected_return_3m": pd.Series(-0.01, index=idx),
                "signal_confidence": pd.Series(0.3, index=idx),
                "volatility_3m": pd.Series(0.05, index=idx),
                "downside_risk_3m": pd.Series(0.04, index=idx),
            }
        )
        w_current = build_feasible_initial_weights(params["tickers"], params)
        candidates = build_candidate_portfolios(w_current, w_current, forecast_table, params)
        scenario_matrix = pd.DataFrame([pd.Series(-0.01, index=idx)], index=["base"])
        scenario_set = ScenarioSet(
            as_of=pd.Timestamp("2026-01-01"),
            scenario_returns_matrix=scenario_matrix,
            scenario_names=["base"],
            scenario_probabilities=pd.Series({"base": 1.0}),
            summary=pd.DataFrame({"scenario_name": ["base"], "probability": [1.0], "mean_asset_return": [-0.01], "median_asset_return": [-0.01]}),
            risk_state="risk_off",
        )
        selection = select_robust_candidate(candidates, scenario_set, w_current, params)
        rows.append(_test_row("all_negative_signals_defensive_or_hold", "PASS" if selection.selected_candidate.name in {"HOLD", "DEFENSIVE_CASH"} else "FAIL", selection.selected_candidate.name))
    except Exception as exc:
        rows.append(_test_row("all_negative_signals_defensive_or_hold", "FAIL", str(exc)))

    gate_spread = evaluate_execution_gate(dummy_selection, estimated_spread_cost=0.01)
    rows.append(_test_row("high_spread_wait_execution", "PASS" if gate_spread.action == "WAIT_EXECUTION" else "FAIL", gate_spread.action))
    gate_open = evaluate_execution_gate(dummy_selection, open_orders_exist=True)
    rows.append(_test_row("open_orders_wait", "PASS" if gate_open.action == "WAIT" else "FAIL", gate_open.action))
    hold_like_selection = type(
        "DummyHoldSelection",
        (),
        {
            "selected_score": dummy_score,
            "selected_candidate": type("DummyHoldCandidate", (), {"name": "HOLD_CURRENT"})(),
        },
    )()
    hold_gate = evaluate_execution_gate(hold_like_selection, trade_now_hurdle=0.02)
    rows.append(_test_row("hold_current_gate_action", "PASS" if hold_gate.action == "HOLD" else "FAIL", hold_gate.action))

    try:
        sgov_cost = estimate_order_cost("SGOV", "BUY", 10.0, 100.0, order_value=1000.0, config=params)
        ibit_cost = estimate_order_cost("IBIT", "BUY", 10.0, 100.0, order_value=1000.0, config=params)
        rows.append(
            _test_row(
                "cost_model_differentiates_crypto_vs_cashlike",
                "PASS" if float(ibit_cost["cost_bps_used"]) > float(sgov_cost["cost_bps_used"]) else "FAIL",
                f"SGOV={sgov_cost['cost_bps_used']:.2f}bps IBIT={ibit_cost['cost_bps_used']:.2f}bps",
            )
        )
    except Exception as exc:
        rows.append(_test_row("cost_model_differentiates_crypto_vs_cashlike", "FAIL", str(exc)))

    try:
        review_costs = build_transaction_cost_review_summary(
            {
                "total_estimated_transaction_cost": 7.50,
                "total_estimated_commission": 2.00,
                "cost_model_used": "modeled_bps_assumptions",
            },
            nav=1000.0,
            config={
                "default_commission_per_trade_usd": 0.0,
                "default_bps_per_turnover": 5.0,
                "default_spread_bps": 2.0,
                "default_slippage_bps": 3.0,
            },
            trade_edge_summary={"trade_now_edge": -0.015},
        )
        rows.append(
            _test_row(
                "transaction_cost_review_separates_simulator_and_modeled_costs",
                "PASS"
                if abs(float(review_costs["simulator_order_fee_usd"]) - 0.0) < 1e-9
                and abs(float(review_costs["total_simulator_order_fees_usd"]) - 2.0) < 1e-9
                and abs(float(review_costs["modeled_transaction_costs_usd"]) - 5.5) < 1e-9
                and abs(float(review_costs["modeled_transaction_costs_pct_nav"]) - 0.0055) < 1e-9
                and abs(float(review_costs["trade_now_edge_without_direct_simulator_fees"]) - (-0.013)) < 1e-9
                else "FAIL",
                str(review_costs),
            )
        )
    except Exception as exc:
        rows.append(_test_row("transaction_cost_review_separates_simulator_and_modeled_costs", "FAIL", str(exc)))

    try:
        hold_vs_text, hold_vs_summary = _build_hold_vs_target_analysis(
            as_of=pd.Timestamp("2026-05-05"),
            current_portfolio_score=0.0004,
            target_score_before_costs=0.0011,
            target_score_after_costs=0.0008,
            delta_score_vs_current=0.0004,
            total_order_cost=12.5,
            execution_buffer=0.0010,
            model_uncertainty_buffer=0.0009,
            trade_now_edge=-0.0011,
            trade_now_hurdle=0.0025,
            probability_beats_current=0.51,
            probability_beats_cash=0.56,
            tail_risk_current=-0.021,
            tail_risk_target=-0.028,
            current_weights=pd.Series({"SGOV": 0.5, "XLK": 0.5}, dtype=float),
            continuous_target_weights=pd.Series({"SGOV": 0.3, "XLK": 0.7}, dtype=float),
            final_discrete_weights=pd.Series({"SGOV": 0.4, "XLK": 0.6}, dtype=float),
            continuous_model_optimal_candidate="MOMENTUM_TILT_SIMPLE",
            best_discrete_candidate_name="HOLD_CURRENT",
            factor_forecast_df=pd.DataFrame(
                {
                    "factor": ["momentum", "quality", "value"],
                    "forecast": [0.03, 0.01, -0.01],
                }
            ),
            discrete_scores_frame=pd.DataFrame(
                [
                    {
                        "discrete_candidate": "HOLD_CURRENT",
                        "net_robust_score": 0.0004,
                        "delta_vs_hold": 0.0,
                        "delta_vs_cash": 0.0003,
                        "probability_beats_hold": 1.0,
                        "probability_beats_cash": 0.60,
                        "valid_constraints": True,
                        "validation_errors": "",
                    },
                    {
                        "discrete_candidate": "OPTIMIZER_TARGET::ROUND_NEAREST_0",
                        "net_robust_score": 0.0007,
                        "delta_vs_hold": 0.0003,
                        "delta_vs_cash": 0.0002,
                        "probability_beats_hold": 0.50,
                        "probability_beats_cash": 0.51,
                        "valid_constraints": True,
                        "validation_errors": "",
                    },
                ]
            ),
            gate_reason="execution_gate:trade_now_edge_below_hurdle",
            data_context={"used_cache_fallback": True},
            risk_premium_hurdle=0.0005,
            p_hold_min=0.55,
            p_cash_min=0.52,
        )
        rows.append(
            _test_row(
                "hold_vs_target_analysis_contains_requested_fields",
                "PASS"
                if "current_portfolio_score" in hold_vs_text
                and "probability_beats_cash" in hold_vs_text
                and "Was muesste sich aendern" in hold_vs_text
                and abs(float(hold_vs_summary["trade_now_edge"]) - (-0.0011)) < 1e-9
                else "FAIL",
                hold_vs_text.splitlines()[0] if hold_vs_text else "empty",
            )
        )
    except Exception as exc:
        rows.append(_test_row("hold_vs_target_analysis_contains_requested_fields", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "current_portfolio.csv"
            csv_path.write_text(
                "ticker,shares,cash_usd\nSGOV,10,\nXLK,5,\nCASH,,12.34\n",
                encoding="utf-8",
            )
            state = load_current_portfolio_state(
                params={
                    **params,
                    "current_portfolio_path": str(csv_path),
                    "current_portfolio_source": "csv",
                    "allow_fractional_shares": False,
                },
                active_tickers=["SGOV", "XLK"],
                latest_prices=pd.Series({"SGOV": 100.0, "XLK": 200.0}, dtype=float),
                cash_proxy_ticker="SGOV",
                nav=0.0,
            )
            rows.append(
                _test_row(
                    "current_portfolio_cash_usd_alias_supported",
                    "PASS"
                    if abs(float(state.current_cash) - 12.34) < 1e-9
                    and state.cash_input_method == "CASH row with cash_usd column"
                    else "FAIL",
                    f"cash={state.current_cash} method={state.cash_input_method}",
                )
            )
    except Exception as exc:
        rows.append(_test_row("current_portfolio_cash_usd_alias_supported", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "current_portfolio.csv"
            csv_path.write_text(
                "ticker,shares\nSGOV,10\nXLK,5\n",
                encoding="utf-8",
            )
            state = load_current_portfolio_state(
                params={
                    **params,
                    "current_portfolio_path": str(csv_path),
                    "current_portfolio_source": "csv",
                    "allow_fractional_shares": False,
                },
                active_tickers=["SGOV", "XLK"],
                latest_prices=pd.Series({"SGOV": 100.0, "XLK": 200.0}, dtype=float),
                cash_proxy_ticker="SGOV",
                nav=0.0,
            )
            rows.append(
                _test_row(
                    "current_portfolio_missing_cash_defaults_zero_with_warning",
                    "PASS"
                    if abs(float(state.current_cash) - 0.0) < 1e-9
                    and any("assumed cash_usd=0.00" in warning for warning in state.parser_warnings)
                    else "FAIL",
                    f"cash={state.current_cash} warnings={state.parser_warnings}",
                )
            )
    except Exception as exc:
        rows.append(_test_row("current_portfolio_missing_cash_defaults_zero_with_warning", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "current_portfolio.csv"
            report_path = Path(tmpdir) / "current_portfolio_report.txt"
            csv_path.write_text(
                "ticker,shares,cash_value\nSGOV,10,\nXLK,5,\nCASH,,7.50\n",
                encoding="utf-8",
            )
            state = load_current_portfolio_state(
                params={
                    **params,
                    "current_portfolio_path": str(csv_path),
                    "current_portfolio_source": "csv",
                    "allow_fractional_shares": False,
                },
                active_tickers=["SGOV", "XLK"],
                latest_prices=pd.Series({"SGOV": 100.0, "XLK": 200.0}, dtype=float),
                cash_proxy_ticker="SGOV",
                nav=0.0,
            )
            write_current_portfolio_report(
                state,
                report_path,
                latest_prices=pd.Series({"SGOV": 100.0, "XLK": 200.0}, dtype=float),
                latest_price_date="2026-05-05",
                price_basis="adjusted_close_proxy",
                data_source="yfinance",
                data_freshness_ok=True,
            )
            report_text = report_path.read_text(encoding="utf-8")
            rows.append(
                _test_row(
                    "current_portfolio_report_includes_price_metadata_and_consistency_checks",
                    "PASS"
                    if "latest_price=100.0000" in report_text
                    and "data_source=yfinance" in report_text
                    and "stale_price_warning=False" in report_text
                    and "weights_including_cash_close_to_1: True" in report_text
                    else "FAIL",
                    report_text.splitlines()[0] if report_text else "empty",
                )
            )
    except Exception as exc:
        rows.append(_test_row("current_portfolio_report_includes_price_metadata_and_consistency_checks", "FAIL", str(exc)))

    try:
        candidate_map = {
            "HOLD_CURRENT": type("CandidateLike", (), {"weights_proxy": pd.Series(dtype=float), "weights_actual": pd.Series(dtype=float), "shares": pd.Series(dtype=float), "cash_left": 1000.0})(),
            "OPTIMIZER_TARGET::FLOOR_BASE_0": type("CandidateLike", (), {"weights_proxy": pd.Series(dtype=float), "weights_actual": pd.Series(dtype=float), "shares": pd.Series(dtype=float), "cash_left": 50.0})(),
        }
        scores_frame = pd.DataFrame(
            [
                {
                    "discrete_candidate": "OPTIMIZER_TARGET::FLOOR_BASE_0",
                    "net_robust_score": 0.0009,
                    "cvar_5": -0.02,
                    "turnover_vs_current": 0.20,
                    "max_abs_weight_drift": 0.01,
                    "number_of_positions": 4,
                    "cash_left": 50.0,
                    "valid_constraints": True,
                    "delta_vs_cash": 0.0009,
                    "probability_beats_hold": 0.80,
                    "probability_beats_cash": 0.80,
                },
                {
                    "discrete_candidate": "HOLD_CURRENT",
                    "net_robust_score": 0.0002,
                    "cvar_5": -0.01,
                    "turnover_vs_current": 0.00,
                    "max_abs_weight_drift": 0.00,
                    "number_of_positions": 0,
                    "cash_left": 1000.0,
                    "valid_constraints": True,
                    "delta_vs_cash": 0.0002,
                    "probability_beats_hold": 1.00,
                    "probability_beats_cash": 1.00,
                },
            ]
        )
        selection = select_best_discrete_portfolio(
            {
                "scores_frame": scores_frame,
                "candidate_map": candidate_map,
                "selection_config": {
                    "hurdle": 0.0010,
                    "risk_premium_hurdle": 0.0005,
                    "p_hold_min": 0.55,
                    "p_cash_min": 0.52,
                },
            }
        )
        rows.append(
            _test_row(
                "select_best_discrete_portfolio_does_not_trade_without_hurdle",
                "PASS" if selection["best_discrete_candidate_name"] == "HOLD_CURRENT" else "FAIL",
                selection["best_discrete_candidate_name"],
            )
        )
    except Exception as exc:
        rows.append(_test_row("select_best_discrete_portfolio_does_not_trade_without_hurdle", "FAIL", str(exc)))

    try:
        candidate_map = {
            "HOLD_CURRENT": type("CandidateLike", (), {"weights_proxy": pd.Series(dtype=float), "weights_actual": pd.Series(dtype=float), "shares": pd.Series(dtype=float), "cash_left": 1000.0})(),
            "OPTIMIZER_TARGET::ROUND_NEAREST_0": type("CandidateLike", (), {"weights_proxy": pd.Series(dtype=float), "weights_actual": pd.Series(dtype=float), "shares": pd.Series(dtype=float), "cash_left": 100.0})(),
        }
        scores_frame = pd.DataFrame(
            [
                {
                    "discrete_candidate": "OPTIMIZER_TARGET::ROUND_NEAREST_0",
                    "net_robust_score": 0.0002,
                    "cvar_5": -0.02,
                    "turnover_vs_current": 0.10,
                    "max_abs_weight_drift": 0.01,
                    "number_of_positions": 2,
                    "cash_left": 100.0,
                    "valid_constraints": True,
                    "delta_vs_cash": 0.0008,
                    "probability_beats_hold": 0.90,
                    "probability_beats_cash": 0.90,
                },
                {
                    "discrete_candidate": "HOLD_CURRENT",
                    "net_robust_score": 0.0002,
                    "cvar_5": -0.01,
                    "turnover_vs_current": 0.00,
                    "max_abs_weight_drift": 0.00,
                    "number_of_positions": 0,
                    "cash_left": 1000.0,
                    "valid_constraints": True,
                    "delta_vs_cash": 0.0002,
                    "probability_beats_hold": 1.00,
                    "probability_beats_cash": 1.00,
                },
            ]
        )
        selection = select_best_discrete_portfolio(
            {
                "scores_frame": scores_frame,
                "candidate_map": candidate_map,
                "selection_config": {
                    "hurdle": 0.0,
                    "risk_premium_hurdle": 0.0005,
                    "p_hold_min": 0.55,
                    "p_cash_min": 0.52,
                },
            }
        )
        rows.append(
            _test_row(
                "select_best_discrete_portfolio_prefers_hold_on_equal_score",
                "PASS" if selection["best_discrete_candidate_name"] == "HOLD_CURRENT" else "FAIL",
                selection["best_discrete_candidate_name"],
            )
        )
    except Exception as exc:
        rows.append(_test_row("select_best_discrete_portfolio_prefers_hold_on_equal_score", "FAIL", str(exc)))

    try:
        candidate_map = {
            "HOLD_CURRENT": type("CandidateLike", (), {"weights_proxy": pd.Series(dtype=float), "weights_actual": pd.Series(dtype=float), "shares": pd.Series(dtype=float), "cash_left": 1000.0})(),
            "OPTIMIZER_TARGET::FLOOR_BASE_0": type("CandidateLike", (), {"weights_proxy": pd.Series(dtype=float), "weights_actual": pd.Series(dtype=float), "shares": pd.Series(dtype=float), "cash_left": 50.0})(),
        }
        scores_frame = pd.DataFrame(
            [
                {
                    "discrete_candidate": "OPTIMIZER_TARGET::FLOOR_BASE_0",
                    "net_robust_score": 0.0020,
                    "cvar_5": -0.02,
                    "turnover_vs_current": 0.20,
                    "max_abs_weight_drift": 0.01,
                    "number_of_positions": 4,
                    "cash_left": 50.0,
                    "valid_constraints": True,
                    "delta_vs_cash": 0.0015,
                    "probability_beats_hold": 0.40,
                    "probability_beats_cash": 0.40,
                },
                {
                    "discrete_candidate": "HOLD_CURRENT",
                    "net_robust_score": 0.0002,
                    "cvar_5": -0.01,
                    "turnover_vs_current": 0.00,
                    "max_abs_weight_drift": 0.00,
                    "number_of_positions": 0,
                    "cash_left": 1000.0,
                    "valid_constraints": True,
                    "delta_vs_cash": 0.0002,
                    "probability_beats_hold": 1.00,
                    "probability_beats_cash": 1.00,
                },
            ]
        )
        selection = select_best_discrete_portfolio(
            {
                "scores_frame": scores_frame,
                "candidate_map": candidate_map,
                "selection_config": {
                    "hurdle": 0.0010,
                    "risk_premium_hurdle": 0.0005,
                    "p_hold_min": 0.55,
                    "p_cash_min": 0.52,
                },
            }
        )
        failed_reason = selection["scores_frame"].loc[
            selection["scores_frame"]["discrete_candidate"] == "OPTIMIZER_TARGET::FLOOR_BASE_0",
            "selection_failed_reason",
        ].iloc[0]
        rows.append(
            _test_row(
                "select_best_discrete_portfolio_reports_failed_reason",
                "PASS" if failed_reason == "failed_probability" else "FAIL",
                str(failed_reason),
            )
        )
    except Exception as exc:
        rows.append(_test_row("select_best_discrete_portfolio_reports_failed_reason", "FAIL", str(exc)))

    try:
        mu = pd.Series({"A": 0.10, "B": 0.02}, dtype=float)
        sigma = pd.DataFrame(0.0, index=mu.index, columns=mu.index)
        current = pd.Series({"A": 0.5, "B": 0.5}, dtype=float)
        params_linear = _make_test_optimizer_params(
            ["A", "B"],
            cash_ticker=None,
            risk_aversion=0.0,
            turnover_penalty=0.0,
            concentration_penalty=0.0,
            max_turnover=2.0,
            group_map={"A": "risk", "B": "risk"},
            group_limits={"risk": 1.0},
            max_equity_like_total=1.0,
        )
        result = optimize_allocation(mu=mu, Sigma=sigma, w_current=current, params=params_linear)
        rows.append(
            _test_row(
                "optimizer_two_asset_analytic_boundary_optimum",
                "PASS"
                if float(result.target_weights["A"]) > 0.999 and float(result.target_weights["B"]) < 1e-6
                else "FAIL",
                f"{result.solver_name}: {result.target_weights.to_dict()}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("optimizer_two_asset_analytic_boundary_optimum", "FAIL", str(exc)))

    try:
        mu = pd.Series({"CASHX": 0.00, "A": 0.10, "B": 0.05}, dtype=float)
        sigma = pd.DataFrame(0.0, index=mu.index, columns=mu.index)
        current = pd.Series({"CASHX": 1 / 3, "A": 1 / 3, "B": 1 / 3}, dtype=float)
        params_cash = _make_test_optimizer_params(
            ["CASHX", "A", "B"],
            cash_ticker="CASHX",
            risk_aversion=0.0,
            turnover_penalty=0.0,
            concentration_penalty=0.0,
            min_cash_weight=0.2,
            max_turnover=2.0,
            group_map={"CASHX": "cash", "A": "risk", "B": "risk"},
            group_limits={"cash": 1.0, "risk": 1.0},
            max_equity_like_total=1.0,
            min_defensive_weight=0.0,
        )
        result = optimize_allocation(mu=mu, Sigma=sigma, w_current=current, params=params_cash)
        rows.append(
            _test_row(
                "optimizer_three_asset_cash_constraint",
                "PASS"
                if abs(float(result.target_weights["CASHX"]) - 0.2) <= 1e-4
                and abs(float(result.target_weights["A"]) - 0.8) <= 1e-4
                and abs(float(result.target_weights["B"])) <= 1e-4
                else "FAIL",
                f"{result.solver_name}: {result.target_weights.to_dict()}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("optimizer_three_asset_cash_constraint", "FAIL", str(exc)))

    try:
        mu = pd.Series({"CASHX": 0.00, "A": 0.10, "B": 0.05}, dtype=float)
        sigma = pd.DataFrame(0.0, index=mu.index, columns=mu.index)
        current = pd.Series({"CASHX": 1 / 3, "A": 1 / 3, "B": 1 / 3}, dtype=float)
        params_cap = _make_test_optimizer_params(
            ["CASHX", "A", "B"],
            cash_ticker="CASHX",
            asset_max_weights={"CASHX": 1.0, "A": 0.4, "B": 1.0},
            risk_aversion=0.0,
            turnover_penalty=0.0,
            concentration_penalty=0.0,
            max_turnover=2.0,
            group_map={"CASHX": "cash", "A": "risk", "B": "risk"},
            group_limits={"cash": 1.0, "risk": 1.0},
            max_equity_like_total=1.0,
        )
        result = optimize_allocation(mu=mu, Sigma=sigma, w_current=current, params=params_cap)
        rows.append(
            _test_row(
                "optimizer_asset_max_forbids_unconstrained_optimum",
                "PASS"
                if abs(float(result.target_weights["A"]) - 0.4) <= 1e-4
                and abs(float(result.target_weights["B"]) - 0.6) <= 1e-4
                else "FAIL",
                f"{result.solver_name}: {result.target_weights.to_dict()}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("optimizer_asset_max_forbids_unconstrained_optimum", "FAIL", str(exc)))

    try:
        tickers = ["CASHX", "A"]
        prices = pd.Series({"CASHX": 1.0, "A": 1.0}, dtype=float)
        current_shares = pd.Series({"CASHX": 0.0, "A": 0.0}, dtype=float)
        target_weights = pd.Series({"CASHX": 0.0, "A": 1.0}, dtype=float)
        nav = 4.0
        current_cash = 4.0
        scenario_set = _make_test_scenario_set(tickers, [[0.0, -0.10], [0.0, -0.05]])
        params_hold = _make_test_optimizer_params(
            tickers,
            cash_ticker="CASHX",
            group_map={"CASHX": "cash", "A": "risk"},
            group_limits={"cash": 1.0, "risk": 1.0},
            hurdle=0.0,
            risk_premium_hurdle=0.0,
            p_hold_min=0.0,
            p_cash_min=0.0,
            min_order_value_usd=0.0,
        )
        heuristic_candidates = generate_discrete_candidates(
            target_weights=target_weights,
            latest_prices=prices,
            nav=nav,
            current_positions=current_shares,
            min_order_value=0.0,
            cash_buffer=0.0,
            max_candidates=100,
            allow_fractional_shares=False,
            marginal_priority=target_weights,
            cash_proxy_ticker="CASHX",
        )
        brute_candidates = _bruteforce_discrete_candidates(
            prices=prices,
            nav=nav,
            current_shares=current_shares,
            min_order_value=0.0,
            cash_proxy_ticker="CASHX",
        )
        heuristic = _score_and_select_discrete_fixture(
            candidates=heuristic_candidates,
            scenario_set=scenario_set,
            params=params_hold,
            current_shares=current_shares,
            latest_prices=prices,
            nav=nav,
            current_cash=current_cash,
            target_weights=target_weights,
            cash_proxy_ticker="CASHX",
        )
        brute = _score_and_select_discrete_fixture(
            candidates=brute_candidates,
            scenario_set=scenario_set,
            params=params_hold,
            current_shares=current_shares,
            latest_prices=prices,
            nav=nav,
            current_cash=current_cash,
            target_weights=target_weights,
            cash_proxy_ticker="CASHX",
        )
        same_shares = np.allclose(
            heuristic["selected"]["candidate"].shares.reindex(tickers).to_numpy(dtype=float),
            brute["selected"]["candidate"].shares.reindex(tickers).to_numpy(dtype=float),
        )
        rows.append(
            _test_row(
                "discrete_bruteforce_hold_optimal_matches_heuristic",
                "PASS"
                if heuristic["selected"]["best_discrete_candidate_name"] == "HOLD_CURRENT"
                and same_shares
                else "FAIL",
                f"heuristic={heuristic['selected']['best_discrete_candidate_name']}; brute_shares={brute['selected']['candidate'].shares.to_dict()}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("discrete_bruteforce_hold_optimal_matches_heuristic", "FAIL", str(exc)))

    try:
        tickers = ["CASHX", "A"]
        prices = pd.Series({"CASHX": 1.0, "A": 1.0}, dtype=float)
        current_shares = pd.Series({"CASHX": 0.0, "A": 0.0}, dtype=float)
        target_weights = pd.Series({"CASHX": 0.0, "A": 1.0}, dtype=float)
        nav = 4.0
        scenario_set = _make_test_scenario_set(tickers, [[0.0, 0.40], [0.0, 0.20]])
        params_rebalance = _make_test_optimizer_params(
            tickers,
            cash_ticker="CASHX",
            group_map={"CASHX": "cash", "A": "risk"},
            group_limits={"cash": 1.0, "risk": 1.0},
            hurdle=0.0,
            risk_premium_hurdle=0.0,
            p_hold_min=0.0,
            p_cash_min=0.0,
            min_order_value_usd=0.0,
        )
        heuristic_candidates = generate_discrete_candidates(
            target_weights=target_weights,
            latest_prices=prices,
            nav=nav,
            current_positions=current_shares,
            min_order_value=0.0,
            cash_buffer=0.0,
            max_candidates=100,
            allow_fractional_shares=False,
            marginal_priority=target_weights,
            cash_proxy_ticker="CASHX",
        )
        brute_candidates = _bruteforce_discrete_candidates(
            prices=prices,
            nav=nav,
            current_shares=current_shares,
            min_order_value=0.0,
            cash_proxy_ticker="CASHX",
        )
        heuristic = _score_and_select_discrete_fixture(
            candidates=heuristic_candidates,
            scenario_set=scenario_set,
            params=params_rebalance,
            current_shares=current_shares,
            latest_prices=prices,
            nav=nav,
            current_cash=4.0,
            target_weights=target_weights,
            cash_proxy_ticker="CASHX",
        )
        brute = _score_and_select_discrete_fixture(
            candidates=brute_candidates,
            scenario_set=scenario_set,
            params=params_rebalance,
            current_shares=current_shares,
            latest_prices=prices,
            nav=nav,
            current_cash=4.0,
            target_weights=target_weights,
            cash_proxy_ticker="CASHX",
        )
        heuristic_shares = heuristic["selected"]["candidate"].shares.reindex(tickers).to_numpy(dtype=float)
        brute_shares = brute["selected"]["candidate"].shares.reindex(tickers).to_numpy(dtype=float)
        rows.append(
            _test_row(
                "discrete_bruteforce_rebalance_optimal_matches_heuristic",
                "PASS"
                if np.allclose(heuristic_shares, np.array([0.0, 4.0]))
                and np.allclose(brute_shares, np.array([0.0, 4.0]))
                else "FAIL",
                f"heuristic={heuristic_shares.tolist()} brute={brute_shares.tolist()}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("discrete_bruteforce_rebalance_optimal_matches_heuristic", "FAIL", str(exc)))

    try:
        tickers = ["A", "B"]
        prices = pd.Series({"A": 1.0, "B": 1.0}, dtype=float)
        current_shares = pd.Series({"A": 4.0, "B": 0.0}, dtype=float)
        target_weights = pd.Series({"A": 0.0, "B": 1.0}, dtype=float)
        nav = 4.0
        scenario_set = _make_test_scenario_set(tickers, [[-0.05, 0.30], [-0.02, 0.20]])
        params_sell_buy = _make_test_optimizer_params(
            tickers,
            cash_ticker=None,
            group_map={"A": "risk", "B": "risk"},
            group_limits={"risk": 1.0},
            hurdle=0.0,
            risk_premium_hurdle=0.0,
            p_hold_min=0.0,
            p_cash_min=0.0,
            min_order_value_usd=0.0,
        )
        heuristic_candidates = generate_discrete_candidates(
            target_weights=target_weights,
            latest_prices=prices,
            nav=nav,
            current_positions=current_shares,
            min_order_value=0.0,
            cash_buffer=0.0,
            max_candidates=100,
            allow_fractional_shares=False,
            marginal_priority=target_weights,
            cash_proxy_ticker=None,
        )
        brute_candidates = _bruteforce_discrete_candidates(
            prices=prices,
            nav=nav,
            current_shares=current_shares,
            min_order_value=0.0,
            cash_proxy_ticker=None,
        )
        heuristic = _score_and_select_discrete_fixture(
            candidates=heuristic_candidates,
            scenario_set=scenario_set,
            params=params_sell_buy,
            current_shares=current_shares,
            latest_prices=prices,
            nav=nav,
            current_cash=0.0,
            target_weights=target_weights,
            cash_proxy_ticker=None,
        )
        brute = _score_and_select_discrete_fixture(
            candidates=brute_candidates,
            scenario_set=scenario_set,
            params=params_sell_buy,
            current_shares=current_shares,
            latest_prices=prices,
            nav=nav,
            current_cash=0.0,
            target_weights=target_weights,
            cash_proxy_ticker=None,
        )
        heuristic_shares = heuristic["selected"]["candidate"].shares.reindex(tickers).to_numpy(dtype=float)
        brute_shares = brute["selected"]["candidate"].shares.reindex(tickers).to_numpy(dtype=float)
        rows.append(
            _test_row(
                "discrete_bruteforce_sell_buy_optimal_matches_heuristic",
                "PASS"
                if np.allclose(heuristic_shares, np.array([0.0, 4.0]))
                and np.allclose(brute_shares, np.array([0.0, 4.0]))
                else "FAIL",
                f"heuristic={heuristic_shares.tolist()} brute={brute_shares.tolist()}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("discrete_bruteforce_sell_buy_optimal_matches_heuristic", "FAIL", str(exc)))

    try:
        tickers = ["CASHX", "A"]
        prices = pd.Series({"CASHX": 1.0, "A": 1.0}, dtype=float)
        current_shares = pd.Series({"CASHX": 0.0, "A": 0.0}, dtype=float)
        target_weights = pd.Series({"CASHX": 0.0, "A": 1.0}, dtype=float)
        nav = 4.0
        scenario_set = _make_test_scenario_set(tickers, [[0.0, 0.04], [0.0, 0.02]])
        params_costs = _make_test_optimizer_params(
            tickers,
            cash_ticker="CASHX",
            group_map={"CASHX": "cash", "A": "risk"},
            group_limits={"cash": 1.0, "risk": 1.0},
            hurdle=0.0,
            risk_premium_hurdle=0.0,
            p_hold_min=0.0,
            p_cash_min=0.0,
            min_order_value_usd=0.0,
            default_spread_bps=300.0,
            default_slippage_bps=300.0,
        )
        heuristic_candidates = generate_discrete_candidates(
            target_weights=target_weights,
            latest_prices=prices,
            nav=nav,
            current_positions=current_shares,
            min_order_value=0.0,
            cash_buffer=0.0,
            max_candidates=100,
            allow_fractional_shares=False,
            marginal_priority=target_weights,
            cash_proxy_ticker="CASHX",
        )
        brute_candidates = _bruteforce_discrete_candidates(
            prices=prices,
            nav=nav,
            current_shares=current_shares,
            min_order_value=0.0,
            cash_proxy_ticker="CASHX",
        )
        heuristic = _score_and_select_discrete_fixture(
            candidates=heuristic_candidates,
            scenario_set=scenario_set,
            params=params_costs,
            current_shares=current_shares,
            latest_prices=prices,
            nav=nav,
            current_cash=4.0,
            target_weights=target_weights,
            cash_proxy_ticker="CASHX",
        )
        brute = _score_and_select_discrete_fixture(
            candidates=brute_candidates,
            scenario_set=scenario_set,
            params=params_costs,
            current_shares=current_shares,
            latest_prices=prices,
            nav=nav,
            current_cash=4.0,
            target_weights=target_weights,
            cash_proxy_ticker="CASHX",
        )
        rows.append(
            _test_row(
                "discrete_costs_can_make_rebalance_unattractive",
                "PASS"
                if heuristic["selected"]["best_discrete_candidate_name"] == "HOLD_CURRENT"
                and np.allclose(
                    brute["selected"]["candidate"].shares.reindex(tickers).to_numpy(dtype=float),
                    np.array([0.0, 0.0]),
                )
                else "FAIL",
                f"heuristic={heuristic['selected']['best_discrete_candidate_name']}; brute_shares={brute['selected']['candidate'].shares.to_dict()}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("discrete_costs_can_make_rebalance_unattractive", "FAIL", str(exc)))

    try:
        tickers = ["CASHX", "A"]
        prices = pd.Series({"CASHX": 1.0, "A": 4.0}, dtype=float)
        current_shares = pd.Series({"CASHX": 0.0, "A": 0.0}, dtype=float)
        target_weights = pd.Series({"CASHX": 0.4, "A": 0.6}, dtype=float)
        nav = 5.0
        scenario_set = _make_test_scenario_set(tickers, [[0.0, 0.25], [0.0, 0.15]])
        params_rounding = _make_test_optimizer_params(
            tickers,
            cash_ticker="CASHX",
            group_map={"CASHX": "cash", "A": "risk"},
            group_limits={"cash": 1.0, "risk": 1.0},
            hurdle=0.0,
            risk_premium_hurdle=0.0,
            p_hold_min=0.0,
            p_cash_min=0.0,
            min_order_value_usd=0.0,
        )
        heuristic_candidates = generate_discrete_candidates(
            target_weights=target_weights,
            latest_prices=prices,
            nav=nav,
            current_positions=current_shares,
            min_order_value=0.0,
            cash_buffer=0.0,
            max_candidates=100,
            allow_fractional_shares=False,
            marginal_priority=target_weights,
            cash_proxy_ticker="CASHX",
        )
        brute_candidates = _bruteforce_discrete_candidates(
            prices=prices,
            nav=nav,
            current_shares=current_shares,
            min_order_value=0.0,
            cash_proxy_ticker="CASHX",
            max_shares_by_ticker={"CASHX": 5, "A": 1},
        )
        heuristic = _score_and_select_discrete_fixture(
            candidates=heuristic_candidates,
            scenario_set=scenario_set,
            params=params_rounding,
            current_shares=current_shares,
            latest_prices=prices,
            nav=nav,
            current_cash=5.0,
            target_weights=target_weights,
            cash_proxy_ticker="CASHX",
        )
        brute = _score_and_select_discrete_fixture(
            candidates=brute_candidates,
            scenario_set=scenario_set,
            params=params_rounding,
            current_shares=current_shares,
            latest_prices=prices,
            nav=nav,
            current_cash=5.0,
            target_weights=target_weights,
            cash_proxy_ticker="CASHX",
        )
        heuristic_shares = heuristic["selected"]["candidate"].shares.reindex(tickers).to_numpy(dtype=float)
        brute_shares = brute["selected"]["candidate"].shares.reindex(tickers).to_numpy(dtype=float)
        rows.append(
            _test_row(
                "discrete_rounding_case_exposes_heuristic_gap_vs_bruteforce",
                "PASS"
                if not np.allclose(heuristic_shares, brute_shares)
                and np.allclose(brute_shares, np.array([0.0, 1.0]))
                else "FAIL",
                f"heuristic={heuristic_shares.tolist()} brute={brute_shares.tolist()}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("discrete_rounding_case_exposes_heuristic_gap_vs_bruteforce", "FAIL", str(exc)))

    try:
        tickers = ["CASHX", "A", "B"]
        prices = pd.Series({"CASHX": 1.0, "A": 1.0, "B": 1.0}, dtype=float)
        current_shares = pd.Series({"CASHX": 4.0, "A": 0.0, "B": 0.0}, dtype=float)
        target_weights = pd.Series({"CASHX": 0.0, "A": 1.0, "B": 0.0}, dtype=float)
        nav = 4.0
        scenario_set = _make_test_scenario_set(tickers, [[0.0, 0.20, 0.19], [0.0, 0.18, 0.17]])
        params_local = _make_test_optimizer_params(
            tickers,
            cash_ticker="CASHX",
            asset_max_weights={"CASHX": 1.0, "A": 0.5, "B": 1.0},
            group_map={"CASHX": "cash", "A": "risk", "B": "risk"},
            group_limits={"cash": 1.0, "risk": 1.0},
            hurdle=0.0,
            risk_premium_hurdle=0.0,
            p_hold_min=0.0,
            p_cash_min=0.0,
            min_order_value_usd=0.0,
        )
        heuristic_candidates = generate_discrete_candidates(
            target_weights=target_weights,
            latest_prices=prices,
            nav=nav,
            current_positions=current_shares,
            min_order_value=0.0,
            cash_buffer=0.0,
            max_candidates=100,
            allow_fractional_shares=False,
            marginal_priority=target_weights,
            cash_proxy_ticker="CASHX",
        )
        brute_candidates = _bruteforce_discrete_candidates(
            prices=prices,
            nav=nav,
            current_shares=current_shares,
            min_order_value=0.0,
            cash_proxy_ticker="CASHX",
            max_shares_by_ticker={"CASHX": 4, "A": 4, "B": 4},
        )
        heuristic = _score_and_select_discrete_fixture(
            candidates=heuristic_candidates,
            scenario_set=scenario_set,
            params=params_local,
            current_shares=current_shares,
            latest_prices=prices,
            nav=nav,
            current_cash=0.0,
            target_weights=target_weights,
            cash_proxy_ticker="CASHX",
        )
        brute = _score_and_select_discrete_fixture(
            candidates=brute_candidates,
            scenario_set=scenario_set,
            params=params_local,
            current_shares=current_shares,
            latest_prices=prices,
            nav=nav,
            current_cash=0.0,
            target_weights=target_weights,
            cash_proxy_ticker="CASHX",
        )
        heuristic_score = float(heuristic["selected"]["best_discrete_score"])
        brute_score = float(brute["selected"]["best_discrete_score"])
        rows.append(
            _test_row(
                "discrete_target_local_search_not_global_under_asset_cap",
                "PASS"
                if brute_score > heuristic_score + 1e-9
                else "FAIL",
                f"heuristic_score={heuristic_score:.6f}; brute_score={brute_score:.6f}; heuristic={heuristic['selected']['candidate'].shares.to_dict()}; brute={brute['selected']['candidate'].shares.to_dict()}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("discrete_target_local_search_not_global_under_asset_cap", "FAIL", str(exc)))

    try:
        unknown_open_orders = detect_open_orders(object())
        rows.append(
            _test_row(
                "unknown_open_orders_blocks_execution",
                "PASS" if bool(unknown_open_orders["open_orders_exist"]) and str(unknown_open_orders["status"]) == "BLOCK" else "FAIL",
                str(unknown_open_orders["message"]),
            )
        )
    except Exception as exc:
        rows.append(_test_row("unknown_open_orders_blocks_execution", "FAIL", str(exc)))

    try:
        reconciliation_result = reconcile_before_execution(
            model_weights=pd.Series({"SGOV": 1.0}, dtype=float),
            latest_prices=pd.Series({"SGOV": 100.0}, dtype=float),
            execution_mode="investopedia",
            broker_positions=pd.DataFrame([{"ticker": "SGOV", "shares": 10.0, "last_price": 100.0, "market_value": 1000.0}]),
            broker_cash=0.0,
            adapter_or_stub=object(),
        )
        rows.append(
            _test_row(
                "reconciliation_failure_on_unknown_open_orders",
                "PASS" if reconciliation_result["status"] == "FAIL" and bool(reconciliation_result["open_orders_exist"]) else "FAIL",
                str(reconciliation_result["message"]),
            )
        )
    except Exception as exc:
        rows.append(_test_row("reconciliation_failure_on_unknown_open_orders", "FAIL", str(exc)))

    try:
        state = {
            "current_date": "2026-04-28",
            "current_iso_week": "2026-W18",
            "current_month": "2026-04",
            "orders_today": 3,
            "turnover_today": 0.4,
            "turnover_week": 0.7,
            "turnover_month": 0.9,
        }
        reset = reset_state_periods_if_needed(state, pd.Timestamp("2026-04-29"))
        rows.append(
            _test_row(
                "daily_bot_state_resets_new_day",
                "PASS" if int(reset["orders_today"]) == 0 and float(reset["turnover_today"]) == 0.0 else "FAIL",
                str(reset),
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_bot_state_resets_new_day", "FAIL", str(exc)))

    try:
        state = {
            "current_date": "2026-05-03",
            "current_iso_week": "2026-W18",
            "current_month": "2026-05",
            "orders_today": 1,
            "turnover_today": 0.1,
            "turnover_week": 0.8,
            "turnover_month": 1.1,
        }
        reset = reset_state_periods_if_needed(state, pd.Timestamp("2026-05-04"))
        rows.append(
            _test_row(
                "daily_bot_state_resets_new_week",
                "PASS" if float(reset["turnover_week"]) == 0.0 else "FAIL",
                str(reset),
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_bot_state_resets_new_week", "FAIL", str(exc)))

    try:
        preview_orders = pd.DataFrame(
            [
                {"ticker": "SGOV", "side": "BUY", "current_shares": 0.0, "target_shares": 10.0, "order_shares": 10.0},
                {"ticker": "XLK", "side": "HOLD", "current_shares": 0.0, "target_shares": 0.0, "order_shares": 0.0},
            ]
        )
        signature = compute_order_signature(preview_orders)
        decision_id = compute_decision_id(pd.Timestamp("2026-04-29"), "TEST", signature)
        updated = update_state_after_execution(
            {
                "orders_today": 0,
                "turnover_today": 0.0,
                "turnover_week": 0.0,
                "turnover_month": 0.0,
                "last_order_timestamp": None,
                "last_order_signature": "",
                "last_selected_candidate": "",
                "last_trade_now_score": 0.0,
                "last_decision_id": "",
                "last_execution_status": "",
                "executed_order_ids": [],
            },
            executed_orders=0,
            turnover=0.6,
            timestamp=None,
            decision_id=decision_id,
            order_signature=signature,
            selected_candidate="TEST",
            trade_now_score=0.1,
            execution_status="order_preview_only",
            execution_mode="order_preview_only",
        )
        rows.append(
            _test_row(
                "preview_only_does_not_increment_executed_orders",
                "PASS"
                if int(updated["orders_today"]) == 0
                and float(updated["turnover_today"]) == 0.0
                and not updated["last_order_timestamp"]
                else "FAIL",
                str(updated),
            )
        )
    except Exception as exc:
        rows.append(_test_row("preview_only_does_not_increment_executed_orders", "FAIL", str(exc)))

    bad_prices = pd.DataFrame({"A": [1.0, 1.0, 1.0, None, None], "B": [None, None, None, None, None]})
    bad_returns = bad_prices.pct_change(fill_method=None).dropna(how="all")
    dq = compute_data_quality_report(bad_prices, bad_returns, ["A", "B"], {})
    rows.append(_test_row("bad_data_quality_pause_signal", "PASS" if dq["global_data_quality_score"] < 0.50 else "FAIL", f"{dq['global_data_quality_score']:.3f}"))

    try:
        stale_prices = pd.DataFrame(
            {
                "SGOV": [100.0] * 320,
                "IEF": [95.0] * 320,
            },
            index=pd.bdate_range(end="2024-01-31", periods=320),
        )
        freshness = check_data_freshness(stale_prices, max_staleness_days=5)
        rows.append(
            _test_row(
                "stale_cache_detected",
                "PASS" if not bool(freshness["data_freshness_ok"]) else "FAIL",
                str(freshness),
            )
        )
    except Exception as exc:
        rows.append(_test_row("stale_cache_detected", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "prices_cache.csv"
            fresh_index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=320)
            cache_prices = pd.DataFrame(
                {ticker: 100.0 + i for i, ticker in enumerate(params["tickers"])},
                index=fresh_index,
            )
            cache_prices.to_csv(cache_path)
            original_yf = data_module.yf
            data_module.yf = None
            try:
                loaded = load_price_data(
                    tickers=list(params["tickers"]),
                    start_date=str(fresh_index.min().date()),
                    end_date=None,
                    cache_path=cache_path,
                    use_cache=True,
                    prefer_live=True,
                    allow_cache_fallback=True,
                    force_refresh=True,
                )
            finally:
                data_module.yf = original_yf
        rows.append(
            _test_row(
                "yfinance_failure_cache_fallback",
                "PASS" if loaded.attrs.get("data_source") == "cache_fallback" and bool(loaded.attrs.get("used_cache_fallback", False)) else "FAIL",
                f"data_source={loaded.attrs.get('data_source')} used_cache_fallback={loaded.attrs.get('used_cache_fallback')}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("yfinance_failure_cache_fallback", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_yf = data_module.yf
            data_module.yf = None
            try:
                try:
                    load_price_data(
                        tickers=list(params["tickers"]),
                        start_date="2025-01-01",
                        end_date=None,
                        cache_path=Path(tmp_dir) / "missing_cache.csv",
                        use_cache=True,
                        prefer_live=True,
                        allow_cache_fallback=True,
                        force_refresh=True,
                    )
                    rows.append(_test_row("yfinance_failure_no_cache", "FAIL", "load_price_data unexpectedly succeeded"))
                except Exception:
                    rows.append(_test_row("yfinance_failure_no_cache", "PASS", "clean error raised"))
            finally:
                data_module.yf = original_yf
    except Exception as exc:
        rows.append(_test_row("yfinance_failure_no_cache", "FAIL", str(exc)))

    model_confidence = {"model_confidence_score": 0.40}
    sizing = compute_trade_fraction(dummy_selection, evaluate_execution_gate(dummy_selection, trade_now_hurdle=-1.0), model_confidence, {"global_data_quality_score": 0.8}, {"turnover_budget_remaining": 1.0})
    rows.append(_test_row("low_model_confidence_caps_trade", "PASS" if float(sizing["trade_fraction"]) <= 0.25 else "FAIL", f"{sizing['trade_fraction']:.2f}"))

    infeasible_params = {**params, "asset_max_weights": {ticker: 0.01 for ticker in params["tickers"]}, "max_equity_like_total": 0.05, "min_defensive_weight": 0.80}
    feas = check_portfolio_feasibility(list(params["tickers"]), infeasible_params)
    rows.append(_test_row("infeasible_config_pause_before_optimizer", "PASS" if not feas["feasible"] else "FAIL", "; ".join(feas["errors"]) or "unexpectedly feasible"))

    try:
        adapter = InvestopediaSimulatorAdapter.from_env()
        try:
            adapter.login()
            rows.append(_test_row("investopedia_disabled_no_login", "FAIL", "login unexpectedly proceeded"))
        except Exception:
            rows.append(_test_row("investopedia_disabled_no_login", "PASS", "safe failure when disabled"))
    except Exception as exc:
        rows.append(_test_row("investopedia_disabled_no_login", "PASS", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            initialize_paper_account(Path(tmp_dir) / "paper.sqlite", initial_cash=10000.0)
        rows.append(_test_row("paper_broker_init", "PASS", "local paper account initialized"))
    except Exception as exc:
        rows.append(_test_row("paper_broker_init", "FAIL", str(exc)))

    try:
        compute_model_confidence(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), type("DummyOptimizer", (), {"success": False})(), {"global_data_quality_score": 0.5})
        rows.append(_test_row("model_governance_smoke", "PASS", "computed"))
    except Exception as exc:
        rows.append(_test_row("model_governance_smoke", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnostics = create_run_diagnostics(mode="daily_bot")
            diagnostics.final_orders_summary = {"order_count": 0, "turnover": 0.0, "estimated_cost": 0.0}
            log_final_action(diagnostics, "HOLD", selected_candidate="HOLD_CURRENT", reason="Safe default.")
            write_run_diagnostics(diagnostics, output_dir=tmp_dir)
            json_path = Path(tmp_dir) / "run_diagnostics.json"
            exists = json_path.exists()
        rows.append(
            _test_row(
                "diagnostic_report_created_on_success",
                "PASS" if exists else "FAIL",
                str(json_path),
            )
        )
    except Exception as exc:
        rows.append(_test_row("diagnostic_report_created_on_success", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnostics = create_run_diagnostics(mode="daily_bot")
            try:
                raise RuntimeError("forced failure for diagnostics")
            except Exception as exc:
                log_error(diagnostics, "robustness_tests", "forced_failure", exc, stage="test")
            write_run_diagnostics(diagnostics, output_dir=tmp_dir)
            json_path = Path(tmp_dir) / "run_diagnostics.json"
            error_path = Path(tmp_dir) / "error_log.csv"
            exists = json_path.exists() and error_path.exists()
        rows.append(
            _test_row(
                "diagnostic_report_created_on_failure",
                "PASS" if exists else "FAIL",
                f"{exists}",
            )
        )
    except Exception as exc:
        rows.append(_test_row("diagnostic_report_created_on_failure", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnostics = create_run_diagnostics(mode="daily_bot")
            try:
                raise ValueError("debug failure")
            except Exception as exc:
                log_error(diagnostics, "robustness_tests", "debug_report", exc, stage="test")
            path = write_codex_debug_report(diagnostics, output_path=Path(tmp_dir) / "codex_daily_debug_report.md")
            exists = path.exists()
        rows.append(
            _test_row(
                "codex_debug_report_created_on_failure",
                "PASS" if exists else "FAIL",
                str(path),
            )
        )
    except Exception as exc:
        rows.append(_test_row("codex_debug_report_created_on_failure", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnostics = create_run_diagnostics(mode="daily_bot")
            try:
                raise ValueError("prompt failure")
            except Exception as exc:
                log_error(diagnostics, "robustness_tests", "codex_next_prompt", exc, stage="test")
            path = write_codex_next_prompt(diagnostics, output_path=Path(tmp_dir) / "codex_next_prompt.md")
            exists = path.exists()
        rows.append(
            _test_row(
                "codex_next_prompt_created_on_failure",
                "PASS" if exists else "FAIL",
                str(path),
            )
        )
    except Exception as exc:
        rows.append(_test_row("codex_next_prompt_created_on_failure", "FAIL", str(exc)))

    try:
        diagnostics = create_run_diagnostics(mode="daily_bot")
        log_rejected_order(diagnostics, "SGOV", "BUY", "too_small", extra={"shares": 0.1})
        report = build_codex_debug_report(diagnostics)
        rows.append(
            _test_row(
                "rejected_orders_appear_in_codex_report",
                "PASS" if "SGOV" in report and "too_small" in report else "FAIL",
                "checked",
            )
        )
    except Exception as exc:
        rows.append(_test_row("rejected_orders_appear_in_codex_report", "FAIL", str(exc)))

    try:
        diagnostics = create_run_diagnostics(mode="daily_bot")
        log_warning(diagnostics, "daily_bot", "cache fallback active", stage="data_loading")
        report = build_codex_debug_report(diagnostics)
        rows.append(
            _test_row(
                "warnings_appear_in_codex_report",
                "PASS" if "cache fallback active" in report else "FAIL",
                "checked",
            )
        )
    except Exception as exc:
        rows.append(_test_row("warnings_appear_in_codex_report", "FAIL", str(exc)))

    try:
        diagnostics = create_run_diagnostics(mode="daily_bot")
        try:
            raise ValueError("SMTP_PASSWORD=topsecret")
        except Exception as exc:
            log_error(diagnostics, "daily_bot", "failure_path", exc, stage="test")
        error_entry = diagnostics.errors[0]
        traceback_text = str(error_entry.get("traceback", ""))
        rows.append(
            _test_row(
                "errors_include_traceback",
                "PASS" if "ValueError" in traceback_text and "topsecret" not in traceback_text else "FAIL",
                traceback_text.splitlines()[-1] if traceback_text else "missing",
            )
        )
    except Exception as exc:
        rows.append(_test_row("errors_include_traceback", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnostics = create_run_diagnostics(mode="daily_bot")
            log_final_action(diagnostics, "HOLD", selected_candidate="HOLD_CURRENT", reason="No edge after costs.")
            path = write_daily_analysis_report(diagnostics, output_path=Path(tmp_dir) / "daily_analysis_report.md")
            exists = path.exists()
        rows.append(
            _test_row(
                "daily_analysis_report_created",
                "PASS" if exists else "FAIL",
                str(path),
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_analysis_report_created", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnostics = create_run_diagnostics(mode="daily_bot")
            diagnostics.data_context.update(
                {
                    "data_source": "yfinance",
                    "cache_status": "refreshed",
                    "latest_price_date": "2026-05-05",
                    "data_freshness_ok": True,
                    "expected_latest_trading_day": "2026-05-05",
                    "within_allowed_window": True,
                    "execution_allowed_by_calendar": True,
                    "current_date_berlin": "2026-05-05",
                    "current_time_berlin": "17:00:00",
                    "is_project_trading_day": True,
                    "price_basis": "adjusted_close_proxy",
                }
            )
            diagnostics.model_context["daily_review_payload"] = {
                "run_status": {
                    "review_date": "2026-05-05",
                    "review_time_berlin": "17:00:00",
                    "current_date_berlin": "2026-05-05",
                    "current_time_berlin": "17:00:00",
                    "is_project_trading_day": True,
                    "within_allowed_window": True,
                    "execution_allowed_by_calendar": True,
                    "final_action": "HOLD",
                    "execution_mode": "order_preview_only",
                    "gate_reason": "No edge after costs.",
                },
                "data_status": {
                    "data_source": "yfinance",
                    "cache_status": "refreshed",
                    "synthetic_data": False,
                    "used_cache_fallback": False,
                    "latest_price_date": "2026-05-05",
                    "staleness_days": 0,
                    "data_freshness_ok": True,
                    "live_data_error": "",
                    "missing_prices": [],
                    "price_basis": "adjusted_close_proxy",
                },
                "current_portfolio": {
                    "current_portfolio_source": "csv",
                    "positions_count": 2,
                    "cash_usd": 3.73,
                    "nav_usd": 1000.0,
                    "current_portfolio_100pct_cash": False,
                    "current_weights_sum_including_cash": 1.0,
                    "current_weights_sum_without_cash": 0.99627,
                    "parser_warnings": [],
                    "parser_errors": [],
                },
                "current_positions": [
                    {"ticker": "SGOV", "current_shares": 10.0, "latest_price": 100.0, "latest_price_date": "2026-05-05", "market_value_usd": 1000.0, "current_weight": 1.0, "price_basis": "adjusted_close_proxy", "data_warning": ""},
                ],
                "target_allocation": [
                    {"ticker": "SGOV", "target_weight": 1.0, "target_shares": 10.0, "target_market_value_usd": 1000.0, "continuous_target_weight": 1.0, "abs_weight_drift": 0.0, "latest_price": 100.0},
                ],
                "delta_transactions": [],
                "cost_edge": {
                    "simulator_fee_usd": 0.0,
                    "total_simulator_fees_usd": 0.0,
                    "modeled_transaction_costs_usd": 0.0,
                    "modeled_transaction_costs_pct_nav": 0.0,
                    "current_portfolio_score": 0.1,
                    "target_score_before_costs": 0.1,
                    "target_score_after_costs": 0.1,
                    "delta_score_vs_current": 0.0,
                    "execution_buffer": 0.001,
                    "model_uncertainty_buffer": 0.001,
                    "trade_now_edge": -0.001,
                    "cost_model_used": "no_orders",
                },
                "decision_context": {
                    "why_this_target": "Hold stayed best.",
                    "why_not_hold": "n/a",
                    "why_not_cash": "cash not needed",
                    "trade_decision_reason": "No edge after costs.",
                    "positive_drivers": ["No target drift"],
                    "negative_drivers": ["trade_now_edge negative"],
                    "rejected_candidates": [],
                    "main_blocker_category": "costs/edge",
                },
                "pre_trade_validation_status": "PASS",
                "preview_only": True,
                "manual_orders_preview_ready": False,
                "cash_after_orders": 3.73,
                "main_daily_scope_differs": True,
                "exception_message": "",
            }
            paths = write_daily_portfolio_review_outputs(diagnostics, output_dir=tmp_dir)
            exists = all(
                Path(paths[key]).exists()
                for key in [
                    "daily_portfolio_review_txt",
                    "daily_portfolio_review_csv",
                    "daily_email_subject",
                    "daily_email_briefing",
                    "daily_portfolio_briefing_md",
                    "daily_portfolio_briefing_html",
                    "hold_dominance_analysis",
                    "hold_sensitivity_report",
                    "decision_history",
                    "latest_email_notification",
                    "email_safety_report",
                    "email_delivery_diagnosis_report",
                    "daily_review_validation_report",
                    "email_final_acceptance_report",
                    "last_email_state",
                ]
            )
            review_text = (Path(tmp_dir) / "daily_portfolio_review.txt").read_text(encoding="utf-8")
            email_safety = (Path(tmp_dir) / "email_safety_report.txt").read_text(encoding="utf-8")
            delivery_diagnosis = (Path(tmp_dir) / "email_delivery_diagnosis_report.txt").read_text(encoding="utf-8")
            email_briefing = (Path(tmp_dir) / "daily_email_briefing.txt").read_text(encoding="utf-8")
            portfolio_briefing_md = (Path(tmp_dir) / "daily_portfolio_briefing.md").read_text(encoding="utf-8")
            portfolio_briefing_html = (Path(tmp_dir) / "daily_portfolio_briefing.html").read_text(encoding="utf-8")
            latest_notification = (Path(tmp_dir) / "latest_email_notification.txt").read_text(encoding="utf-8")
            acceptance_report = (Path(tmp_dir) / "email_final_acceptance_report.txt").read_text(encoding="utf-8")
            hold_dominance = (Path(tmp_dir) / "hold_dominance_analysis.txt").read_text(encoding="utf-8")
            hold_sensitivity = (Path(tmp_dir) / "hold_sensitivity_report.txt").read_text(encoding="utf-8")
            last_email_state = json.loads((Path(tmp_dir) / "last_email_state.json").read_text(encoding="utf-8"))
        rows.append(
            _test_row(
                "daily_portfolio_review_files_created",
                "PASS"
                if exists
                and "real_email_send_allowed: false" in email_safety.lower()
                and "hard_fail_count:" in email_safety
                and "soft_warning_count:" in email_safety
                and "info_count:" in email_safety
                and "all_blockers:" in email_safety
                and "issue_table:" in email_safety
                and "Subject:" in latest_notification
                and "TODAY'S ACTION" in email_briefing
                and "DATA STATUS" in email_briefing
                and "SAFETY STATUS" in email_briefing
                and "DECISION SUMMARY" in email_briefing
                and "Warum HOLD?" in email_briefing
                and "Daily Portfolio Briefing" in portfolio_briefing_md
                and "Safe Orders" in portfolio_briefing_md
                and "Active Preview Orders" in portfolio_briefing_md
                and "<html" in portfolio_briefing_html.lower()
                and "Daily Portfolio Briefing" in portfolio_briefing_html
                and "8b. Warum HOLD?" in review_text
                and "Dies ist das kompakte Daily Portfolio Briefing." in latest_notification
                and "Daily Portfolio Briefing Markdown: outputs/daily_portfolio_briefing.md" in latest_notification
                and "cron_environment_diagnostics:" in delivery_diagnosis
                and "Hold Dominance Analysis" in hold_dominance
                and "Hold Sensitivity Report" in hold_sensitivity
                and "CONFIG" in acceptance_report
                and "DAILY BOT" in acceptance_report
                and "MAIL CONTENT" in acceptance_report
                and "SEND RESULT" in acceptance_report
                and "DEDUPE" in acceptance_report
                and "SECURITY" in acceptance_report
                and "READY_FOR_DAILY_EMAIL_PREVIEW" in acceptance_report
                and bool(last_email_state.get("dedupe_key"))
                and bool(last_email_state.get("current_decision_fingerprint"))
                and last_email_state.get("real_email_send_allowed") is False
                else "FAIL",
                str(tmp_dir),
            )
        )
        rows.append(
            _test_row(
                "daily_portfolio_briefing_md_and_html_exist",
                "PASS"
                if "daily_portfolio_briefing.md" in str(paths["daily_portfolio_briefing_md"])
                and "daily_portfolio_briefing.html" in str(paths["daily_portfolio_briefing_html"])
                and "Daily Portfolio Briefing" in portfolio_briefing_md
                and "<html" in portfolio_briefing_html.lower()
                else "FAIL",
                str(paths),
            )
        )
        rows.append(
            _test_row(
                "briefing_contains_safe_and_active_sections",
                "PASS"
                if (
                    "## Safe Orders" in portfolio_briefing_md
                    and "## Active Preview Orders" in portfolio_briefing_md
                    and "## Safety Gates" in portfolio_briefing_md
                    and "Active Preview Orders" in portfolio_briefing_html
                )
                else "FAIL",
                portfolio_briefing_md,
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_portfolio_review_files_created", "FAIL", str(exc)))

    try:
        review_payload = {
            "run_status": {
                "review_date": "2026-05-05",
                "final_action": "HOLD",
                "execution_mode": "order_preview_only",
            },
            "data_status": {
                "data_source": "yfinance",
                "latest_price_date": "2026-05-05",
                "used_cache_fallback": False,
                "synthetic_data": False,
                "data_freshness_ok": True,
                "live_data_error": "",
            },
            "current_portfolio": {"nav_usd": 1000.0, "cash_usd": 3.73},
            "current_positions": [],
            "target_allocation": [],
            "delta_transactions": [],
            "order_summary": {"manual_eligible_order_count": 0, "order_count": 0},
            "cost_edge": {"trade_now_edge": -0.001},
            "decision_context": {
                "continuous_candidate": "MOMENTUM_TILT_SIMPLE",
                "final_discrete_candidate": "HOLD_CURRENT",
                "reasons_against_trade": [],
            },
            "preview_only": True,
        }
        issues = {
            "review_status": "WAIT",
            "first_blocker": "trade_now_edge negative",
            "all_blockers": ["trade_now_edge negative"],
            "hard_fail_count": 0,
            "soft_warning_count": 1,
            "issue_table": [{"severity": "SOFT_WARNING", "message": "trade_now_edge negative"}],
        }
        body = build_daily_email_briefing(review_payload, issues)
        rows.append(
            _test_row(
                "daily_email_briefing_hold_and_zero_orders_are_explicit",
                "PASS"
                if (
                    "TODAY'S ACTION" in body
                    and "DATA STATUS" in body
                    and "SAFETY STATUS" in body
                    and "DECISION SUMMARY" in body
                    and "Warum HOLD?" in body
                    and "Heute keine Orders eingeben. Beste Aktion laut Bot: HOLD." in body
                    and "Keine Simulator-Orders eingeben." in body
                    and "- manual_order_count: 0" in body
                )
                else "FAIL",
                body,
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_email_briefing_hold_and_zero_orders_are_explicit", "FAIL", str(exc)))

    try:
        review_payload = {
            "run_status": {
                "review_date": "2026-05-05",
                "final_action": "WAIT_OUTSIDE_WINDOW",
                "execution_mode": "blocked",
            },
            "data_status": {
                "data_source": "cache_fallback",
                "latest_price_date": "2026-05-05",
                "used_cache_fallback": True,
                "synthetic_data": False,
                "data_freshness_ok": True,
                "live_data_error": "yfinance timeout",
            },
            "current_portfolio": {"nav_usd": 1000.0, "cash_usd": 3.73},
            "current_positions": [],
            "target_allocation": [],
            "delta_transactions": [],
            "order_summary": {"manual_eligible_order_count": 0, "order_count": 0},
            "cost_edge": {"trade_now_edge": -0.001},
            "decision_context": {
                "continuous_candidate": "MOMENTUM_TILT_SIMPLE",
                "final_discrete_candidate": "HOLD_CURRENT",
                "reasons_against_trade": [],
            },
            "preview_only": True,
        }
        issues = {
            "review_status": "WAIT",
            "first_blocker": "outside allowed trading window",
            "all_blockers": ["outside allowed trading window", "trade_now_edge negative"],
            "hard_fail_count": 0,
            "soft_warning_count": 2,
            "issue_table": [{"severity": "SOFT_WARNING", "message": "cache_fallback used"}],
        }
        body = build_daily_email_briefing(review_payload, issues)
        rows.append(
            _test_row(
                "daily_email_briefing_cache_fallback_warning_is_explicit",
                "PASS"
                if (
                    "Warnung: Live-Daten nicht genutzt; Bericht nur vorsichtig verwenden." in body
                    and "Keine Orders eingeben." in body
                    and "- used_cache_fallback: true" in body
                )
                else "FAIL",
                body,
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_email_briefing_cache_fallback_warning_is_explicit", "FAIL", str(exc)))

    try:
        review_payload = {
            "run_status": {
                "review_date": "2026-05-05",
                "final_action": "BLOCK",
                "execution_mode": "blocked",
            },
            "data_status": {
                "data_source": "synthetic",
                "latest_price_date": "2026-05-05",
                "used_cache_fallback": False,
                "synthetic_data": True,
                "data_freshness_ok": False,
                "live_data_error": "synthetic fallback",
            },
            "current_portfolio": {"nav_usd": 1000.0, "cash_usd": 3.73},
            "current_positions": [],
            "target_allocation": [],
            "delta_transactions": [],
            "order_summary": {"manual_eligible_order_count": 0, "order_count": 0},
            "cost_edge": {"trade_now_edge": -0.001},
            "decision_context": {
                "continuous_candidate": "MOMENTUM_TILT_SIMPLE",
                "final_discrete_candidate": "HOLD_CURRENT",
                "reasons_against_trade": [],
            },
            "preview_only": True,
        }
        issues = {
            "review_status": "BLOCK",
            "first_blocker": "synthetic_data=true",
            "all_blockers": ["synthetic_data=true"],
            "hard_fail_count": 1,
            "soft_warning_count": 0,
            "issue_table": [{"severity": "HARD_FAIL", "message": "synthetic_data=true"}],
        }
        body = build_daily_email_briefing(review_payload, issues)
        rows.append(
            _test_row(
                "daily_email_briefing_synthetic_data_blocks_orders",
                "PASS"
                if (
                    "Blockiert: synthetische Daten; keine Orders." in body
                    and "- synthetic_data: true" in body
                    and "- hard_fail_count: 1" in body
                )
                else "FAIL",
                body,
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_email_briefing_synthetic_data_blocks_orders", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            review_payload = {
                "run_status": {
                    "review_date": "2026-05-05",
                    "review_time_berlin": "16:00:00",
                    "current_date_berlin": "2026-05-05",
                    "current_time_berlin": "16:00:00",
                    "is_project_trading_day": True,
                    "within_allowed_window": True,
                    "execution_allowed_by_calendar": True,
                    "final_action": "BUY",
                    "execution_mode": "paper",
                },
                "data_status": {
                    "data_source": "yfinance",
                    "cache_status": "live",
                    "synthetic_data": False,
                    "used_cache_fallback": False,
                    "latest_price_date": "2026-05-05",
                    "staleness_days": 0,
                    "data_freshness_ok": True,
                    "live_data_error": "",
                    "missing_prices": ["XLK"],
                    "price_basis": "adjusted_close_proxy",
                },
                "current_portfolio": {
                    "current_portfolio_source": "csv",
                    "positions_count": 1,
                    "cash_usd": 0.0,
                    "invested_market_value_usd": -10.0,
                    "nav_usd": 100.0,
                    "current_portfolio_100pct_cash": True,
                    "current_weights_sum_including_cash": 0.80,
                    "current_weights_sum_without_cash": 0.80,
                    "parser_warnings": [],
                    "parser_errors": [],
                },
                "current_positions": [
                    {
                        "ticker": "XLK",
                        "current_shares": 1.0,
                        "latest_price": 0.0,
                        "latest_price_date": "2026-05-05",
                        "market_value_usd": -10.0,
                        "current_weight": 0.80,
                        "price_basis": "adjusted_close_proxy",
                        "data_source": "yfinance",
                        "stale_price_warning": False,
                        "data_warning": "missing_latest_price",
                    }
                ],
                "target_allocation": [],
                "delta_transactions": [],
                "cost_edge": {
                    "simulator_fee_usd": 0.0,
                    "total_simulator_fees_usd": 0.0,
                    "modeled_transaction_costs_usd": 0.0,
                    "modeled_transaction_costs_pct_nav": 0.0,
                    "current_portfolio_score": 0.1,
                    "target_score_before_costs": 0.1,
                    "target_score_after_costs": 0.1,
                    "delta_score_vs_current": 0.0,
                    "execution_buffer": 0.001,
                    "model_uncertainty_buffer": 0.001,
                    "trade_now_edge": -0.001,
                    "cost_model_used": "no_orders",
                },
                "decision_context": {
                    "why_this_target": "test",
                    "why_not_hold": "test",
                    "why_not_cash": "test",
                    "trade_decision_reason": "test",
                    "positive_drivers": [],
                    "negative_drivers": ["test"],
                    "rejected_candidates": [],
                    "main_blocker_category": "data",
                },
                "pre_trade_validation_status": "PASS",
                "preview_only": False,
                "manual_orders_preview_ready": True,
                "cash_after_orders": 0.0,
                "main_daily_scope_differs": False,
                "exception_message": "",
            }
            write_daily_portfolio_review_outputs(review_payload, output_dir=tmp_dir)
            validation_text = (tmp_dir / "daily_review_validation_report.txt").read_text(encoding="utf-8")
            review_text = (tmp_dir / "daily_portfolio_review.txt").read_text(encoding="utf-8")
            subject_text = (tmp_dir / "daily_email_subject.txt").read_text(encoding="utf-8")
            rows.append(
                _test_row(
                    "daily_review_validation_reports_nav_price_consistency_and_blocks",
                    "PASS"
                    if "negative_market_value_count: 1" in validation_text
                    and ("final_action: BLOCK" in review_text or "- final_action: BLOCK" in review_text)
                    and "BLOCK" in subject_text
                    else "FAIL",
                    subject_text.strip(),
                )
            )
    except Exception as exc:
        rows.append(_test_row("daily_review_validation_reports_nav_price_consistency_and_blocks", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            review_payload = {
                "run_status": {
                    "review_date": "2026-05-05",
                    "review_time_berlin": "21:58:13",
                    "current_date_berlin": "2026-05-05",
                    "current_time_berlin": "21:58",
                    "is_project_trading_day": True,
                    "within_allowed_window": True,
                    "execution_allowed_by_calendar": True,
                    "final_action": "HOLD",
                    "execution_mode": "order_preview_only",
                },
                "data_status": {
                    "data_source": "yfinance",
                    "cache_status": "refreshed",
                    "synthetic_data": False,
                    "used_cache_fallback": False,
                    "latest_price_date": "2026-05-05",
                    "staleness_days": 0,
                    "data_freshness_ok": True,
                    "live_data_error": "none",
                    "missing_prices": [],
                    "price_basis": "adjusted_close_proxy",
                },
                "current_portfolio": {
                    "current_portfolio_source": "csv",
                    "positions_count": 1,
                    "cash_usd": 100.0,
                    "invested_market_value_usd": 900.0,
                    "nav_usd": 1000.0,
                    "current_portfolio_100pct_cash": False,
                    "current_weights_sum_including_cash": 1.0,
                    "current_weights_sum_without_cash": 0.9,
                    "read_from_current_portfolio_csv": True,
                    "parser_errors": [],
                    "parser_warnings": [],
                },
                "current_positions": [
                    {
                        "ticker": "SGOV",
                        "current_shares": 9.0,
                        "latest_price": 100.0,
                        "latest_price_date": "2026-05-05",
                        "market_value_usd": 900.0,
                        "current_weight": 0.9,
                        "price_basis": "adjusted_close_proxy",
                        "data_source": "yfinance",
                        "stale_price_warning": False,
                        "data_warning": "adjusted_close_proxy",
                    }
                ],
                "target_portfolio": [
                    {
                        "ticker": "SGOV",
                        "target_weight": 0.9,
                        "target_shares": 9.0,
                        "target_market_value_usd": 900.0,
                        "continuous_target_weight": 0.8,
                        "abs_weight_drift": 0.0,
                        "latest_price": 100.0,
                    }
                ],
                "delta_transactions": [],
                "order_summary": {
                    "cash_before_orders": 100.0,
                    "cash_after_orders": 100.0,
                    "estimated_sell_value": 0.0,
                    "estimated_buy_value": 0.0,
                    "total_simulator_fees_usd": 0.0,
                    "modeled_transaction_costs_usd": 0.0,
                    "buy_count": 0,
                    "sell_count": 0,
                    "hold_count": 1,
                    "order_count": 0,
                    "manual_eligible_order_count": 0,
                    "negative_cash_check": True,
                    "leverage_check": True,
                    "short_check": True,
                    "manual_orders_usable": False,
                    "preview_only": True,
                },
                "cost_edge": {
                    "simulator_fee_usd_per_order": 0.0,
                    "total_simulator_fees_usd": 0.0,
                    "modeled_transaction_costs_usd": 0.0,
                    "modeled_transaction_costs_pct_nav": 0.0,
                    "current_portfolio_score": 0.1,
                    "target_score_before_costs": 0.1,
                    "target_score_after_costs": 0.1,
                    "delta_score_vs_current": 0.0,
                    "execution_buffer": 0.001,
                    "model_uncertainty_buffer": 0.001,
                    "trade_now_edge": -0.001,
                    "cost_model_used": "no_orders",
                },
                "decision_context": {
                    "why_this_target": "test",
                    "why_not_hold": "test",
                    "why_not_cash": "test",
                    "trade_decision_reason": "test",
                    "positive_drivers": [],
                    "negative_drivers": ["test"],
                    "rejected_candidates": [],
                    "main_blocker_category": "costs/edge",
                },
                "pre_trade_validation_status": "PASS",
                "preview_only": True,
                "manual_orders_preview_ready": False,
                "cash_after_orders": 100.0,
                "main_daily_scope_differs": False,
                "exception_message": "",
            }
            write_daily_portfolio_review_outputs(review_payload, output_dir=tmp_dir)
            review_text = (tmp_dir / "daily_portfolio_review.txt").read_text(encoding="utf-8")
            rows.append(
                _test_row(
                    "daily_review_no_delta_orders_still_preview_only",
                    "PASS"
                    if "Preview only. Heute nicht manuell handeln." in review_text
                    and "Keine echten Orders wurden gesendet." in review_text
                    else "FAIL",
                    "preview-only no-delta guidance checked",
                )
            )
    except Exception as exc:
        rows.append(_test_row("daily_review_no_delta_orders_still_preview_only", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            (tmp_dir / "current_data_freshness_report.txt").write_text(
                "latest_price_date: 2026-05-04\n",
                encoding="utf-8",
            )
            review_payload = {
                "run_status": {
                    "review_date": "2026-05-05",
                    "review_time_berlin": "17:30:00",
                    "current_date_berlin": "2026-05-05",
                    "current_time_berlin": "17:30",
                    "is_project_trading_day": True,
                    "within_allowed_window": True,
                    "execution_allowed_by_calendar": True,
                    "final_action": "HOLD",
                    "execution_mode": "order_preview_only",
                },
                "data_status": {
                    "data_source": "yfinance",
                    "cache_status": "refreshed",
                    "synthetic_data": False,
                    "used_cache_fallback": False,
                    "latest_price_date": "2026-05-05",
                    "staleness_days": 0,
                    "data_freshness_ok": True,
                    "live_data_error": "none",
                    "missing_prices": [],
                    "low_history_assets": ["IBIT"],
                    "price_basis": "mid_quote_proxy",
                },
                "current_portfolio": {
                    "current_portfolio_source": "csv",
                    "positions_count": 1,
                    "cash_usd": 100.0,
                    "invested_market_value_usd": 900.0,
                    "nav_usd": 1000.0,
                    "current_portfolio_100pct_cash": False,
                    "current_weights_sum_including_cash": 1.0,
                    "current_weights_sum_without_cash": 0.9,
                    "parser_errors": [],
                    "parser_warnings": [],
                },
                "current_positions": [
                    {
                        "ticker": "SGOV",
                        "current_shares": 9.0,
                        "latest_price": 100.0,
                        "latest_price_date": "2026-05-05",
                        "market_value_usd": 900.0,
                        "current_weight": 0.9,
                        "price_basis": "mid_quote_proxy",
                        "data_source": "yfinance",
                        "stale_price_warning": False,
                        "data_warning": "",
                    }
                ],
                "target_allocation": [],
                "delta_transactions": [],
                "order_summary": {
                    "cash_before_orders": 100.0,
                    "cash_after_orders": 100.0,
                    "estimated_sell_value": 0.0,
                    "estimated_buy_value": 0.0,
                    "total_simulator_fees_usd": 0.0,
                    "modeled_transaction_costs_usd": 0.0,
                    "buy_count": 0,
                    "sell_count": 0,
                    "hold_count": 1,
                    "order_count": 0,
                    "manual_eligible_order_count": 0,
                    "negative_cash_check": True,
                    "leverage_check": True,
                    "short_check": True,
                    "manual_orders_usable": False,
                    "preview_only": True,
                },
                "cost_edge": {
                    "simulator_fee_usd": 0.0,
                    "total_simulator_fees_usd": 0.0,
                    "modeled_transaction_costs_usd": 0.0,
                    "modeled_transaction_costs_pct_nav": 0.0,
                    "current_portfolio_score": 0.1,
                    "target_score_before_costs": 0.1,
                    "target_score_after_costs": 0.1,
                    "delta_score_vs_current": 0.0,
                    "execution_buffer": 0.001,
                    "model_uncertainty_buffer": 0.001,
                    "trade_now_edge": 0.001,
                    "cost_model_used": "no_orders",
                },
                "decision_context": {
                    "why_this_target": "test",
                    "why_not_hold": "test",
                    "why_not_cash": "test",
                    "trade_decision_reason": "test",
                    "positive_drivers": [],
                    "negative_drivers": [],
                    "rejected_candidates": [],
                    "main_blocker_category": "data",
                },
                "pre_trade_validation_status": "PASS",
                "preview_only": True,
                "manual_orders_preview_ready": False,
                "cash_after_orders": 100.0,
                "main_daily_scope_differs": False,
                "exception_message": "",
            }
            write_daily_portfolio_review_outputs(review_payload, output_dir=tmp_dir)
            review_text = (tmp_dir / "daily_portfolio_review.txt").read_text(encoding="utf-8")
            validation_text = (tmp_dir / "daily_review_validation_report.txt").read_text(encoding="utf-8")
            email_safety = (tmp_dir / "email_safety_report.txt").read_text(encoding="utf-8")
            rows.append(
                _test_row(
                    "daily_review_detects_report_mismatch_and_low_history_asset",
                    "PASS"
                    if "latest_price_date mismatch between reports" in review_text
                    and "low history asset: IBIT" in review_text
                    and "latest_price_date_mismatch_between_reports: true" in validation_text.lower()
                    and "issue_table:" in email_safety
                    else "FAIL",
                    "mismatch and low-history warnings checked",
                )
            )
    except Exception as exc:
        rows.append(_test_row("daily_review_detects_report_mismatch_and_low_history_asset", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            review_payload = {
                "run_status": {
                    "review_date": "2026-05-05",
                    "review_time_berlin": "21:58:13",
                    "current_date_berlin": "2026-05-05",
                    "current_time_berlin": "21:58",
                    "is_project_trading_day": True,
                    "within_allowed_window": True,
                    "execution_allowed_by_calendar": True,
                    "final_action": "HOLD",
                    "execution_mode": "order_preview_only",
                },
                "data_status": {
                    "data_source": "yfinance",
                    "cache_status": "refreshed",
                    "synthetic_data": False,
                    "used_cache_fallback": False,
                    "latest_price_date": "2026-05-05",
                    "staleness_days": 0,
                    "data_freshness_ok": True,
                    "live_data_error": "none",
                    "missing_prices": [],
                    "low_history_assets": [],
                    "price_basis": "adjusted_close_proxy",
                },
                "current_portfolio": {
                    "current_portfolio_source": "csv",
                    "positions_count": 1,
                    "cash_usd": 100.0,
                    "invested_market_value_usd": 900.0,
                    "nav_usd": 1000.0,
                    "current_portfolio_100pct_cash": False,
                    "current_weights_sum_including_cash": 1.0,
                    "current_weights_sum_without_cash": 0.9,
                    "parser_errors": [],
                    "parser_warnings": [],
                },
                "current_positions": [
                    {
                        "ticker": "SGOV",
                        "current_shares": 9.0,
                        "latest_price": 100.0,
                        "latest_price_date": "2026-05-05",
                        "market_value_usd": 900.0,
                        "current_weight": 0.9,
                        "price_basis": "adjusted_close_proxy",
                        "data_source": "yfinance",
                        "stale_price_warning": False,
                        "data_warning": "adjusted_close_proxy",
                    }
                ],
                "target_allocation": [],
                "delta_transactions": [],
                "order_summary": {
                    "cash_before_orders": 100.0,
                    "cash_after_orders": 100.0,
                    "estimated_sell_value": 0.0,
                    "estimated_buy_value": 0.0,
                    "total_simulator_fees_usd": 0.0,
                    "modeled_transaction_costs_usd": 0.0,
                    "buy_count": 0,
                    "sell_count": 0,
                    "hold_count": 1,
                    "order_count": 0,
                    "manual_eligible_order_count": 0,
                    "negative_cash_check": True,
                    "leverage_check": True,
                    "short_check": True,
                    "manual_orders_usable": False,
                    "preview_only": True,
                },
                "cost_edge": {
                    "simulator_fee_usd": 0.0,
                    "total_simulator_fees_usd": 0.0,
                    "modeled_transaction_costs_usd": 0.0,
                    "modeled_transaction_costs_pct_nav": 0.0,
                    "current_portfolio_score": 0.1,
                    "target_score_before_costs": 0.1,
                    "target_score_after_costs": 0.1,
                    "delta_score_vs_current": 0.0,
                    "execution_buffer": 0.001,
                    "model_uncertainty_buffer": 0.001,
                    "trade_now_edge": -0.001,
                    "cost_model_used": "no_orders",
                },
                "decision_context": {
                    "why_this_target": "test",
                    "why_not_hold": "test",
                    "why_not_cash": "test",
                    "trade_decision_reason": "test",
                    "positive_drivers": [],
                    "negative_drivers": ["test"],
                    "rejected_candidates": [],
                    "main_blocker_category": "costs/edge",
                },
                "pre_trade_validation_status": "PASS",
                "preview_only": True,
                "manual_orders_preview_ready": False,
                "cash_after_orders": 100.0,
                "main_daily_scope_differs": False,
                "exception_message": "",
            }
            write_daily_portfolio_review_outputs(review_payload, output_dir=tmp_dir)
            write_daily_portfolio_review_outputs(review_payload, output_dir=tmp_dir)
            last_email_state = json.loads((tmp_dir / "last_email_state.json").read_text(encoding="utf-8"))
            rows.append(
                _test_row(
                    "daily_review_email_state_preview_does_not_mark_accepted_duplicate",
                    "PASS"
                    if (
                        not bool(last_email_state.get("duplicate_today"))
                        and not bool(last_email_state.get("would_block_duplicate_real_send_today"))
                        and bool(last_email_state.get("current_decision_fingerprint"))
                    )
                    else "FAIL",
                    str(last_email_state),
                )
            )
    except Exception as exc:
        rows.append(_test_row("daily_review_email_state_preview_does_not_mark_accepted_duplicate", "FAIL", str(exc)))

    try:
        diagnostics = create_run_diagnostics(mode="daily_bot")
        log_final_action(diagnostics, "HOLD", selected_candidate="HOLD_CURRENT", reason="Disabled email path.")
        result = send_daily_analysis_email_if_needed(
            diagnostics,
            settings={
                "ENV_FILE_PRESENT": True,
                "ENABLE_EMAIL_NOTIFICATIONS": False,
                "ENABLE_DAILY_ANALYSIS_EMAIL": True,
                "EMAIL_TO": "test@example.com",
                "EMAIL_FROM": "sender@example.com",
                "SMTP_HOST": "smtp.example.com",
                "SMTP_PORT": 587,
            },
        )
        rows.append(
            _test_row(
                "email_not_sent_when_disabled",
                "PASS" if not result["sent"] and result["reason"] == "email_notifications_disabled" else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("email_not_sent_when_disabled", "FAIL", str(exc)))

    try:
        diagnostics = create_run_diagnostics(mode="daily_bot")
        log_final_action(diagnostics, "HOLD", selected_candidate="HOLD_CURRENT", reason="Daily analysis email disabled.")
        result = send_daily_analysis_email_if_needed(
            diagnostics,
            settings={
                "ENV_FILE_PRESENT": True,
                "ENABLE_EMAIL_NOTIFICATIONS": True,
                "ENABLE_DAILY_ANALYSIS_EMAIL": False,
                "EMAIL_TO": "test@example.com",
                "EMAIL_FROM": "sender@example.com",
                "SMTP_HOST": "smtp.example.com",
                "SMTP_PORT": 587,
            },
        )
        rows.append(
            _test_row(
                "email_not_sent_when_daily_analysis_disabled",
                "PASS" if not result["sent"] and result["reason"] == "daily_analysis_email_disabled" else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("email_not_sent_when_daily_analysis_disabled", "FAIL", str(exc)))

    try:
        diagnostics = create_run_diagnostics(mode="daily_bot")
        log_final_action(diagnostics, "HOLD", selected_candidate="HOLD_CURRENT", reason="SMTP incomplete.")
        result = send_daily_analysis_email_if_needed(
            diagnostics,
            settings={
                "ENV_FILE_PRESENT": True,
                "ENABLE_EMAIL_NOTIFICATIONS": True,
                "ENABLE_DAILY_ANALYSIS_EMAIL": True,
                "EMAIL_TO": "",
                "EMAIL_FROM": "sender@example.com",
                "SMTP_HOST": "",
                "SMTP_PORT": 587,
            },
        )
        rows.append(
            _test_row(
                "email_not_sent_when_smtp_incomplete",
                "PASS" if not result["sent"] and result["reason"] == "smtp_incomplete" else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("email_not_sent_when_smtp_incomplete", "FAIL", str(exc)))

    try:
        diagnostics = create_run_diagnostics(mode="daily_bot")
        log_final_action(diagnostics, "HOLD", selected_candidate="HOLD_CURRENT", reason="Preview-only phase gate.")
        result = send_daily_analysis_email_if_needed(
            diagnostics,
            settings={
                "ENV_FILE_PRESENT": True,
                "ENABLE_EMAIL_NOTIFICATIONS": True,
                "ENABLE_DAILY_ANALYSIS_EMAIL": True,
                "EMAIL_TO": "test@example.com",
                "EMAIL_FROM": "sender@example.com",
                "SMTP_HOST": "smtp.example.com",
                "SMTP_PORT": 587,
                "EMAIL_SEND_ENABLED": False,
                "EMAIL_DRY_RUN": True,
                "EMAIL_RECIPIENT": "",
                "USER_CONFIRMED_EMAIL_PHASE": False,
                "PHASE": "DAILY_REVIEW_PREVIEW",
                "ENABLE_EXTERNAL_BROKER": False,
                "ENABLE_INVESTOPEDIA_SIMULATOR": False,
                "SEND_ANALYSIS_EMAIL_ONLY_ON_TRADING_DAYS": False,
                "DAILY_ANALYSIS_EMAIL_TIME_LOCAL": "00:00",
            },
        )
        rows.append(
            _test_row(
                "daily_analysis_email_phase_gate_blocks_real_send",
                "PASS"
                if (
                    not result["sent"]
                    and result["reason"] == "preview_only"
                    and "EMAIL_SEND_ENABLED=false" in list(result.get("blocked_reasons", []))
                )
                else "FAIL",
                str(result),
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_analysis_email_phase_gate_blocks_real_send", "FAIL", str(exc)))

    try:
        diagnostics = create_run_diagnostics(mode="daily_bot")
        log_final_action(diagnostics, "HOLD", selected_candidate="HOLD_CURRENT", reason="Summary body.")
        body = build_daily_analysis_email_body(diagnostics, include_codex_prompt=False)
        rows.append(
            _test_row(
                "email_body_contains_summary",
                "PASS" if "Final action: HOLD" in body and "Daily analysis summary" in body else "FAIL",
                "checked",
            )
        )
    except Exception as exc:
        rows.append(_test_row("email_body_contains_summary", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnostics = create_run_diagnostics(mode="daily_bot")
            log_warning(
                diagnostics,
                "daily_bot",
                "secret check",
                extra={"SMTP_PASSWORD": "hunter2", "API_KEY": "abc123"},
            )
            write_run_diagnostics(diagnostics, output_dir=tmp_dir)
            payload = (Path(tmp_dir) / "run_diagnostics.json").read_text(encoding="utf-8")
        rows.append(
            _test_row(
                "no_secrets_logged",
                "PASS" if "hunter2" not in payload and "abc123" not in payload and "[REDACTED]" in payload else "FAIL",
                "checked",
            )
        )
    except Exception as exc:
        rows.append(_test_row("no_secrets_logged", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnostics = create_run_diagnostics(mode="daily_bot")
            log_warning(diagnostics, "daily_bot", "non serializable extra", extra={"callback": lambda x: x})
            write_run_diagnostics(diagnostics, output_dir=tmp_dir)
            path = Path(tmp_dir) / "run_diagnostics.json"
            exists = path.exists()
        rows.append(
            _test_row(
                "diagnostics_do_not_crash_on_nonserializable_extra",
                "PASS" if exists else "FAIL",
                str(path),
            )
        )
    except Exception as exc:
        rows.append(_test_row("diagnostics_do_not_crash_on_nonserializable_extra", "FAIL", str(exc)))

    try:
        preview = pd.DataFrame(
            [
                {"ticker": "SGOV", "side": "BUY", "not_executable": False, "reason": ""},
                {"ticker": "IEF", "side": "HOLD", "not_executable": True, "reason": "research"},
            ]
        )
        marked = mark_research_preview(preview)
        rows.append(
            _test_row(
                "main_order_preview_marked_research_only",
                "PASS"
                if bool((marked["preview_context"] == "research_backtest_preview").all())
                and bool((marked["executable"] == False).all())  # noqa: E712
                and bool((marked["not_executable_reason"] == "research_preview_only").all())
                else "FAIL",
                ",".join(marked.columns),
            )
        )
    except Exception as exc:
        rows.append(_test_row("main_order_preview_marked_research_only", "FAIL", str(exc)))

    try:
        preview = pd.DataFrame(
            [
                {"ticker": "SGOV", "side": "BUY", "not_executable": False, "reason": ""},
                {"ticker": "IEF", "side": "BUY", "not_executable": True, "reason": "calendar_blocked"},
            ]
        )
        marked = mark_daily_simulator_preview(preview)
        rows.append(
            _test_row(
                "daily_bot_discrete_preview_marked_simulator_context",
                "PASS"
                if bool((marked["preview_context"] == "daily_bot_discrete_simulator").all())
                and bool(marked.loc[0, "executable"])
                and not bool(marked.loc[1, "executable"])
                and str(marked.loc[1, "not_executable_reason"]) == "calendar_blocked"
                else "FAIL",
                marked[["ticker", "executable", "not_executable_reason"]].to_json(orient="records"),
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_bot_discrete_preview_marked_simulator_context", "FAIL", str(exc)))

    try:
        preview = build_discrete_order_preview(
            current_shares=pd.Series({"SGOV": 10.0, "IEF": 5.0}, dtype=float),
            target_shares=pd.Series({"SGOV": 7.0, "IEF": 9.0}, dtype=float),
            latest_prices=pd.Series({"SGOV": 100.0, "IEF": 95.0}, dtype=float),
            nav=2000.0,
            min_order_value=1.0,
        )
        sgov_row = preview.loc[preview["ticker"] == "SGOV"].iloc[0]
        ief_row = preview.loc[preview["ticker"] == "IEF"].iloc[0]
        rows.append(
            _test_row(
                "delta_order_preview_uses_absolute_shares_and_action",
                "PASS"
                if float(sgov_row["order_shares"]) == 3.0
                and str(sgov_row["action"]) == "SELL"
                and float(ief_row["order_shares"]) == 4.0
                and str(ief_row["action"]) == "BUY"
                and float(sgov_row["estimated_order_value"]) == 300.0
                and float(ief_row["estimated_order_value"]) == 380.0
                else "FAIL",
                preview[["ticker", "action", "order_shares", "estimated_order_value"]].to_json(orient="records"),
            )
        )
    except Exception as exc:
        rows.append(_test_row("delta_order_preview_uses_absolute_shares_and_action", "FAIL", str(exc)))

    try:
        positions = pd.DataFrame(
            [
                {"asset": "AAA", "shares": 1.0},
                {"asset": "BBB", "shares": 0.0},
            ]
        )
        orders = convert_weights_to_orders(
            current_positions=positions,
            target_weights=pd.Series({"AAA": 0.35, "BBB": 0.65}, dtype=float),
            latest_prices=pd.Series({"AAA": 100.0, "BBB": 50.0}, dtype=float),
            total_portfolio_value_usd=1000.0,
            min_order_value_usd=10.0,
            fractional_shares=False,
        )
        integer_delta = bool((orders["share_delta"].round(0) == orders["share_delta"]).all())
        integer_target = bool((orders["target_shares"].round(0) == orders["target_shares"]).all())
        required_columns = [
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
        rows.append(
            _test_row(
                "order_sizing_whole_shares_integer_deltas_and_columns",
                "PASS"
                if integer_delta
                and integer_target
                and abs(float(orders.attrs.get("rounding_cash_drift_usd", 0.0)) - 50.0) < 1e-9
                and orders.columns.tolist() == required_columns
                and str(orders.loc[orders["asset"] == "AAA", "trade_side"].iloc[0]) == "BUY"
                and str(orders.loc[orders["asset"] == "BBB", "trade_side"].iloc[0]) == "BUY"
                else "FAIL",
                orders.to_json(orient="records"),
            )
        )
    except Exception as exc:
        rows.append(_test_row("order_sizing_whole_shares_integer_deltas_and_columns", "FAIL", str(exc)))

    try:
        orders = convert_weights_to_orders(
            current_positions=pd.DataFrame([{"asset": "AAA", "shares": 1.0}]),
            target_weights=pd.Series({"AAA": 0.155}, dtype=float),
            latest_prices=pd.Series({"AAA": 100.0}, dtype=float),
            total_portfolio_value_usd=1000.0,
            min_order_value_usd=10.0,
            fractional_shares=True,
        )
        row = orders.iloc[0]
        rows.append(
            _test_row(
                "order_sizing_fractional_shares_allows_decimal_delta",
                "PASS"
                if abs(float(row["target_shares"]) - 1.55) < 1e-9
                and abs(float(row["share_delta"]) - 0.55) < 1e-9
                and str(row["trade_side"]) == "BUY"
                else "FAIL",
                orders.to_json(orient="records"),
            )
        )
    except Exception as exc:
        rows.append(_test_row("order_sizing_fractional_shares_allows_decimal_delta", "FAIL", str(exc)))

    try:
        orders = convert_weights_to_orders(
            current_positions=pd.DataFrame([{"asset": "AAA", "shares": 0.0}]),
            target_weights=pd.Series({"AAA": 0.10}, dtype=float),
            latest_prices=pd.Series(dtype=float),
            total_portfolio_value_usd=1000.0,
            min_order_value_usd=10.0,
            fractional_shares=False,
        )
        row = orders.iloc[0]
        rows.append(
            _test_row(
                "order_sizing_missing_price_blocks_row_as_hold",
                "PASS"
                if str(row["trade_side"]) == "HOLD"
                and str(row["skipped_reason"]) == "missing_price"
                and float(row["share_delta"]) == 0.0
                else "FAIL",
                orders.to_json(orient="records"),
            )
        )
    except Exception as exc:
        rows.append(_test_row("order_sizing_missing_price_blocks_row_as_hold", "FAIL", str(exc)))

    try:
        orders = convert_weights_to_orders(
            current_positions=pd.DataFrame([{"asset": "AAA", "shares": 1.0}]),
            target_weights=pd.Series({"AAA": 0.105}, dtype=float),
            latest_prices=pd.Series({"AAA": 100.0}, dtype=float),
            total_portfolio_value_usd=1000.0,
            min_order_value_usd=10.0,
            fractional_shares=True,
        )
        row = orders.iloc[0]
        rows.append(
            _test_row(
                "order_sizing_below_min_order_value_holds",
                "PASS"
                if str(row["trade_side"]) == "HOLD"
                and str(row["skipped_reason"]) == "below_min_order_value"
                and abs(float(row["share_delta"])) < 1e-9
                else "FAIL",
                orders.to_json(orient="records"),
            )
        )
    except Exception as exc:
        rows.append(_test_row("order_sizing_below_min_order_value_holds", "FAIL", str(exc)))

    try:
        preview = pd.DataFrame(
            [
                {
                    "ticker": "SGOV",
                    "side": "BUY",
                    "current_shares": 10.0,
                    "target_shares": 7.0,
                    "order_value": -300.0,
                    "latest_price": 100.0,
                    "not_executable": False,
                    "reason": "",
                },
                {
                    "ticker": "IEF",
                    "side": "SELL",
                    "current_shares": 5.0,
                    "target_shares": 9.0,
                    "order_value": 380.0,
                    "latest_price": 95.0,
                    "not_executable": False,
                    "reason": "",
                },
            ]
        )
        annotated = _annotate_final_daily_preview(
            preview_df=preview,
            cash_before_orders=1000.0,
            cash_after_orders=1075.0,
            preview_only=True,
            preview_only_reason="dry_run_preview_only",
        )
        rows.append(
            _test_row(
                "dry_run_preview_keeps_delta_direction_without_marking_not_executable",
                "PASS"
                if str(annotated.loc[0, "action"]) == "SELL"
                and str(annotated.loc[1, "action"]) == "BUY"
                and abs(float(annotated.loc[0, "order_shares"]) - 3.0) < 1e-9
                and abs(float(annotated.loc[1, "order_shares"]) - 4.0) < 1e-9
                and not bool(annotated.loc[0, "not_executable"])
                and not bool(annotated.loc[1, "not_executable"])
                else "FAIL",
                annotated[["ticker", "action", "order_shares", "preview_only", "not_executable", "execution_block_reason"]].to_json(orient="records"),
            )
        )
    except Exception as exc:
        rows.append(_test_row("dry_run_preview_keeps_delta_direction_without_marking_not_executable", "FAIL", str(exc)))

    try:
        preview = pd.DataFrame(
            [
                {
                    "ticker": "SGOV",
                    "side": "SELL",
                    "action": "SELL",
                    "current_shares": 10.0,
                    "target_shares": 7.0,
                    "order_shares": 3.0,
                    "estimated_price": 100.0,
                    "estimated_order_value": 300.0,
                    "preview_only": True,
                    "not_executable": False,
                    "execution_block_reason": "dry_run_preview_only",
                },
                {
                    "ticker": "XLK",
                    "side": "BUY",
                    "action": "BUY",
                    "current_shares": 2.0,
                    "target_shares": 5.0,
                    "order_shares": 3.0,
                    "estimated_price": 200.0,
                    "estimated_order_value": 600.0,
                    "preview_only": True,
                    "not_executable": True,
                    "execution_block_reason": "calendar_blocked",
                    "estimated_total_order_cost": 3.0,
                },
                {
                    "ticker": "IEF",
                    "side": "HOLD",
                    "action": "HOLD",
                    "current_shares": 5.0,
                    "target_shares": 5.0,
                    "order_shares": 0.0,
                    "estimated_price": 95.0,
                    "estimated_order_value": 0.0,
                    "preview_only": True,
                    "not_executable": True,
                    "execution_block_reason": "dry_run_preview_only",
                },
            ]
        )
        manual_frame, _ = _build_manual_simulator_order_outputs(
            order_preview=preview,
            latest_price_date="2026-04-29",
            rest_cash_usd=25.0,
            cash_before_orders=1000.0,
            cash_after_orders=1299.0,
        )
        rows.append(
            _test_row(
                "manual_simulator_orders_only_delta_buy_sell",
                "PASS"
                if len(manual_frame) == 1
                and str(manual_frame.iloc[0]["action"]) == "SELL"
                and float(manual_frame.iloc[0]["shares"]) > 0.0
                and "HOLD" not in manual_frame["action"].astype(str).tolist()
                and bool(manual_frame.iloc[0]["preview_only"])
                and not bool(manual_frame.iloc[0]["not_executable"])
                else "FAIL",
                manual_frame.to_json(orient="records"),
            )
        )
    except Exception as exc:
        rows.append(_test_row("manual_simulator_orders_only_delta_buy_sell", "FAIL", str(exc)))

    try:
        preview = pd.DataFrame(
            [
                {"ticker": "BUY1", "side": "BUY", "order_shares": 10.0, "latest_price": 100.0, "order_value": 1000.0},
                {"ticker": "SELL1", "side": "SELL", "order_shares": 10.0, "latest_price": 100.0, "order_value": -1000.0},
            ]
        )
        _, summary = estimate_order_list_costs(
            order_preview_df=preview,
            latest_prices=pd.Series({"BUY1": 100.0, "SELL1": 100.0}, dtype=float),
            config={
                "nav": 1000.0,
                "current_cash": 0.0,
                "min_order_value_usd": 1.0,
                "default_commission_per_trade_usd": 0.0,
                "default_spread_bps": 0.0,
                "default_slippage_bps": 0.0,
                "default_market_impact_bps": 0.0,
            },
        )
        rows.append(
            _test_row(
                "sell_before_buy_cash_sequencing",
                "PASS"
                if bool(summary["no_negative_cash"]) and abs(float(summary["cash_after_orders"])) < 1e-9
                else "FAIL",
                str(summary),
            )
        )
    except Exception as exc:
        rows.append(_test_row("sell_before_buy_cash_sequencing", "FAIL", str(exc)))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            guide_path = Path(tmp_dir) / "output_file_guide.txt"
            write_output_file_guide(guide_path)
            guide_text = guide_path.read_text(encoding="utf-8")
        rows.append(
            _test_row(
                "research_outputs_marked_research",
                "PASS"
                if "outputs/research_order_preview.csv" in guide_text
                and "legacy research alias" in guide_text.lower()
                and "best_discrete_order_preview.csv" in guide_text
                else "FAIL",
                "guide contains research and simulator preview distinctions",
            )
        )
    except Exception as exc:
        rows.append(_test_row("research_outputs_marked_research", "FAIL", str(exc)))

    try:
        prices = pd.DataFrame(
            {"SGOV": [100.0, 100.1], "IEF": [95.0, 95.3]},
            index=pd.to_datetime(["2026-04-28", "2026-04-29"]),
        )
        prices.attrs.update(
            {
                "data_source": "yfinance",
                "cache_status": "refreshed",
                "synthetic_data": False,
                "used_cache_fallback": False,
                "tickers_loaded": ["SGOV", "IEF"],
                "tickers_failed": [],
                "price_basis": "adjusted_close_proxy",
                "yfinance_available": True,
                "live_data_error": "",
            }
        )
        freshness = {
            "latest_price_date": "2026-04-29",
            "staleness_days": 1,
            "data_freshness_ok": True,
            "warning": "",
        }
        market_gate = {
            "date": "2026-04-30",
            "current_time_berlin": "01:25",
            "is_trading_day": True,
            "allowed_start_berlin": "16:00",
            "allowed_end_berlin": "22:00",
            "within_allowed_window": False,
            "execution_allowed": False,
            "reason": "outside_allowed_window",
        }
        context = build_run_data_context(
            prices=prices,
            freshness=freshness,
            market_gate=market_gate,
            run_context="daily_bot_discrete_simulator",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "current_data_freshness_report.txt"
            write_data_freshness_report(
                prices=prices,
                freshness=freshness,
                output_path=report_path,
                market_gate=market_gate,
                data_context=context,
            )
            report_text = report_path.read_text(encoding="utf-8")
        diagnostics = create_run_diagnostics(mode="daily_bot")
        log_data_context(diagnostics, attrs=context.as_dict())
        consistent = (
            "latest_price_date: 2026-04-29" in report_text
            and "expected_latest_trading_day: 2026-04-29" in report_text
            and str(diagnostics.data_context.get("latest_price_date")) == "2026-04-29"
            and str(diagnostics.data_context.get("expected_latest_trading_day")) == "2026-04-29"
        )
        rows.append(
            _test_row(
                "reports_use_consistent_latest_price_date",
                "PASS" if consistent else "FAIL",
                "data freshness report and diagnostics share latest/expected trading day",
            )
        )
    except Exception as exc:
        rows.append(_test_row("reports_use_consistent_latest_price_date", "FAIL", str(exc)))

    try:
        prices = pd.DataFrame(
            {"SGOV": [100.0, 100.1], "IEF": [95.0, 95.3]},
            index=pd.to_datetime(["2026-04-28", "2026-04-29"]),
        )
        prices.attrs.update(
            {
                "data_source": "yfinance",
                "cache_status": "refreshed",
                "synthetic_data": False,
                "used_cache_fallback": False,
                "tickers_loaded": ["SGOV", "IEF"],
                "tickers_failed": [],
                "price_basis": "adjusted_close_proxy",
                "yfinance_available": True,
                "live_data_error": "",
            }
        )
        context = build_run_data_context(
            prices=prices,
            freshness={
                "latest_price_date": "2026-04-29",
                "staleness_days": 1,
                "data_freshness_ok": True,
                "warning": "",
            },
            market_gate={
                "date": "2026-04-30",
                "current_time_berlin": "01:25",
                "is_trading_day": True,
                "allowed_start_berlin": "16:00",
                "allowed_end_berlin": "22:00",
                "within_allowed_window": False,
                "execution_allowed": False,
                "reason": "outside_allowed_window",
            },
            run_context="daily_bot_discrete_simulator",
        )
        diagnostics = create_run_diagnostics(mode="daily_bot")
        log_data_context(diagnostics, attrs=context.as_dict())
        log_final_action(diagnostics, "HOLD", selected_candidate="HOLD_CURRENT", reason="Context consistency test.")
        daily_report = build_daily_analysis_email_body(diagnostics, include_codex_prompt=False)
        codex_report = build_codex_debug_report(diagnostics)
        consistent = (
            "2026-04-29" in daily_report
            and "Erwarteter letzter Handelstag: 2026-04-29" in build_daily_analysis_report(diagnostics)
            and "latest_price_date: 2026-04-29" in codex_report
            and "expected_latest_trading_day: 2026-04-29" in codex_report
        )
        rows.append(
            _test_row(
                "daily_bot_reports_share_same_data_context",
                "PASS" if consistent else "FAIL",
                "daily analysis and codex report show the same latest/expected trading day",
            )
        )
    except Exception as exc:
        rows.append(_test_row("daily_bot_reports_share_same_data_context", "FAIL", str(exc)))

    try:
        daily = pd.DataFrame(
            [
                {
                    "date": "2026-04-28",
                    "next_date": "2026-04-29",
                    "decision": "HOLD",
                    "risk_state": "normal",
                }
            ]
        )
        weights = pd.DataFrame([{"SGOV": 1.0}], index=pd.to_datetime(["2026-04-28"]))
        target_weights = pd.DataFrame([{"SGOV": 1.0}], index=pd.to_datetime(["2026-04-28"]))
        result = {
            "daily": daily,
            "weights": weights,
            "target_weights": target_weights,
            "safety_context": {
                "run_context": "research_backtest",
                "data_source": "yfinance",
                "cache_status": "refreshed",
                "synthetic_data": False,
                "latest_price_date": "2026-04-28",
                "expected_latest_trading_day": "2026-04-28",
                "data_freshness_ok": True,
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "latest_decision_report.txt"
            write_latest_decision_report(result, report_path)
            report_text = report_path.read_text(encoding="utf-8")
        marked = "Run Context: research_backtest" in report_text and "Expected Latest Trading Day: 2026-04-28" in report_text
        rows.append(
            _test_row(
                "main_report_marked_research_context_if_different",
                "PASS" if marked else "FAIL",
                "research latest decision report includes run context and expected latest trading day",
            )
        )
    except Exception as exc:
        rows.append(_test_row("main_report_marked_research_context_if_different", "FAIL", str(exc)))

    try:
        diagnostics = create_run_diagnostics(mode="daily_bot")
        allowed = should_send_daily_analysis_email(
            diagnostics,
            settings={
                "ENV_FILE_PRESENT": True,
                "ENABLE_EMAIL_NOTIFICATIONS": True,
                "ENABLE_DAILY_ANALYSIS_EMAIL": True,
                "EMAIL_SEND_ENABLED": True,
                "EMAIL_DRY_RUN": False,
                "EMAIL_PROVIDER": "brevo",
                "BREVO_API_KEY": "test-api-key",
                "EMAIL_SENDER": "sender@example.com",
                "EMAIL_RECIPIENT": "test@example.com",
                "EMAIL_TO": "test@example.com",
                "EMAIL_FROM": "sender@example.com",
                "SMTP_HOST": "smtp.example.com",
                "SMTP_PORT": 587,
                "USER_CONFIRMED_EMAIL_PHASE": True,
                "PHASE": "DAILY_REVIEW_SEND_READY",
                "ENABLE_EXTERNAL_BROKER": False,
                "ENABLE_INVESTOPEDIA_SIMULATOR": False,
                "SEND_ANALYSIS_EMAIL_ONLY_ON_TRADING_DAYS": True,
                "SEND_ANALYSIS_EMAIL_ON_ERRORS_ONLY": False,
                "DAILY_ANALYSIS_EMAIL_TIME_LOCAL": "18:00",
                "DAILY_ANALYSIS_TIMEZONE": "Europe/Berlin",
            },
            now=datetime(2026, 5, 2, 19, 0, tzinfo=pd.Timestamp("2026-05-02 19:00", tz="Europe/Berlin").tz),
        )
        rows.append(
            _test_row(
                "trading_day_email_gate_weekend_blocks_if_configured",
                "PASS" if not allowed else "FAIL",
                str(allowed),
            )
        )
    except Exception as exc:
        rows.append(_test_row("trading_day_email_gate_weekend_blocks_if_configured", "FAIL", str(exc)))

    try:
        diagnostics = create_run_diagnostics(mode="daily_bot")
        allowed = should_send_daily_analysis_email(
            diagnostics,
            settings={
                "ENV_FILE_PRESENT": True,
                "ENABLE_EMAIL_NOTIFICATIONS": True,
                "ENABLE_DAILY_ANALYSIS_EMAIL": True,
                "EMAIL_SEND_ENABLED": True,
                "EMAIL_DRY_RUN": False,
                "EMAIL_PROVIDER": "brevo",
                "BREVO_API_KEY": "test-api-key",
                "EMAIL_SENDER": "sender@example.com",
                "EMAIL_RECIPIENT": "test@example.com",
                "EMAIL_TO": "test@example.com",
                "EMAIL_FROM": "sender@example.com",
                "SMTP_HOST": "smtp.example.com",
                "SMTP_PORT": 587,
                "USER_CONFIRMED_EMAIL_PHASE": True,
                "PHASE": "DAILY_REVIEW_SEND_READY",
                "ENABLE_EXTERNAL_BROKER": False,
                "ENABLE_INVESTOPEDIA_SIMULATOR": False,
                "SEND_ANALYSIS_EMAIL_ONLY_ON_TRADING_DAYS": True,
                "SEND_ANALYSIS_EMAIL_ON_ERRORS_ONLY": False,
                "DAILY_ANALYSIS_EMAIL_TIME_LOCAL": "18:00",
                "DAILY_ANALYSIS_TIMEZONE": "Europe/Berlin",
            },
            now=datetime(2026, 5, 4, 18, 10, tzinfo=pd.Timestamp("2026-05-04 18:10", tz="Europe/Berlin").tz),
        )
        rows.append(
            _test_row(
                "trading_day_email_gate_after_time_allows_if_configured",
                "PASS" if allowed and should_send_after_local_time("18:00", "Europe/Berlin", datetime(2026, 5, 4, 18, 10, tzinfo=pd.Timestamp("2026-05-04 18:10", tz="Europe/Berlin").tz)) else "FAIL",
                str(allowed),
            )
        )
    except Exception as exc:
        rows.append(_test_row("trading_day_email_gate_after_time_allows_if_configured", "FAIL", str(exc)))

    try:
        active_tickers = ["AAA", "BBB"]
        prices = pd.Series({"AAA": 10.0, "BBB": 20.0})
        current_shares = pd.Series({"AAA": 50.0, "BBB": 0.0})
        active_shares = pd.Series({"AAA": 30.0, "BBB": 25.0})
        hold_candidate = _finalize_candidate(
            name="HOLD_CURRENT",
            shares=current_shares,
            latest_prices=prices,
            nav=1000.0,
            cash_proxy_ticker=None,
            current_positions=current_shares,
            min_order_value=0.0,
            metadata={"continuous_source": "HOLD", "number_of_orders": 0},
        )
        active_candidate = _finalize_candidate(
            name="ACTIVE_REPAIR::ROUND_NEAREST_REPAIR_0c",
            shares=active_shares,
            latest_prices=prices,
            nav=1000.0,
            cash_proxy_ticker=None,
            current_positions=current_shares,
            min_order_value=0.0,
            metadata={"continuous_source": "ACTIVE_REPAIR", "number_of_orders": 2},
        )
        if hold_candidate is None or active_candidate is None:
            raise AssertionError("fixture candidates were not created")
        fixture_scores = pd.DataFrame(
            [
                {
                    "discrete_candidate": "HOLD_CURRENT",
                    "continuous_source": "HOLD",
                    "net_robust_score": 0.0010,
                    "delta_vs_hold": 0.0,
                    "delta_vs_cash": 0.0010,
                    "probability_beats_hold": 0.50,
                    "probability_beats_cash": 0.55,
                    "valid_constraints": True,
                    "validation_errors": "",
                    "number_of_orders": 0,
                    "turnover_vs_current": 0.0,
                    "selected": True,
                },
                {
                    "discrete_candidate": "ACTIVE_REPAIR::ROUND_NEAREST_REPAIR_0c",
                    "continuous_source": "ACTIVE_REPAIR",
                    "net_robust_score": 0.0040,
                    "delta_vs_hold": 0.0030,
                    "delta_vs_cash": 0.0030,
                    "probability_beats_hold": 0.70,
                    "probability_beats_cash": 0.68,
                    "valid_constraints": True,
                    "validation_errors": "",
                    "number_of_orders": 2,
                    "turnover_vs_current": 0.20,
                    "selected": False,
                },
            ]
        )
        active_params = {
            **_make_test_optimizer_params(active_tickers, min_order_value_usd=0.0),
            "enable_active_preview": True,
            "active_preview_trade_now_hurdle": 0.00075,
            "active_preview_execution_buffer": 0.00035,
            "active_preview_model_uncertainty_multiplier": 0.50,
            "active_preview_delta_vs_cash_min": 0.00025,
            "active_preview_p_current_min": 0.52,
            "active_preview_p_cash_min": 0.51,
            "active_preview_max_turnover": 0.25,
            "active_preview_min_order_value_usd": 0.0,
            "active_preview_allow_execution": True,
        }
        candidate_map = {
            "HOLD_CURRENT": hold_candidate,
            "ACTIVE_REPAIR::ROUND_NEAREST_REPAIR_0c": active_candidate,
        }
        current_state = type("State", (), {})()
        current_state.current_shares = current_shares
        current_state.current_weights_actual = hold_candidate.weights_actual
        current_state.nav = 1000.0
        current_state.current_cash = 500.0
        summary = _select_active_preview_candidate(
            scores_frame=fixture_scores,
            current_portfolio_score=0.0010,
            params=active_params,
            safe_model_uncertainty_buffer=0.0009,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            written_summary = _build_active_preview_files(
                active_preview_summary=dict(summary),
                candidate_map=candidate_map,
                current_state=current_state,
                latest_prices=prices,
                active_tickers=active_tickers,
                params=active_params,
                output_dir=tmp_dir,
            )
            expected_paths = [
                tmp_dir / "active_preview_decision_report.txt",
                tmp_dir / "active_preview_orders.csv",
                tmp_dir / "active_preview_allocation.csv",
                tmp_dir / "active_preview_gate_report.csv",
            ]
            rows.append(
                _test_row(
                    "active_preview_outputs_exist",
                    "PASS" if all(path.exists() for path in expected_paths) else "FAIL",
                    ", ".join(path.name for path in expected_paths if path.exists()),
                )
            )
            orders = pd.read_csv(tmp_dir / "active_preview_orders.csv")
            rows.append(
                _test_row(
                    "active_preview_never_executes_orders",
                    "PASS"
                    if (
                        not bool(written_summary.get("active_preview_executable", True))
                        and not bool(written_summary.get("active_preview_order_submission_allowed", True))
                    )
                    else "FAIL",
                    str({k: written_summary.get(k) for k in ["active_preview_executable", "active_preview_order_submission_allowed", "configured_active_preview_allow_execution"]}),
                )
            )
            rows.append(
                _test_row(
                    "active_preview_uses_lower_hurdles_than_safe_mode",
                    "PASS"
                    if (
                        float(written_summary["active_preview_hurdle"]) < float(written_summary["active_preview_safe_mode_trade_now_hurdle"])
                        and float(written_summary["active_preview_execution_buffer"]) < float(written_summary["active_preview_safe_mode_execution_buffer"])
                        and float(written_summary["active_preview_model_uncertainty_buffer"]) < float(written_summary["active_preview_safe_mode_model_uncertainty_buffer"])
                    )
                    else "FAIL",
                    str({k: written_summary.get(k) for k in ["active_preview_hurdle", "active_preview_safe_mode_trade_now_hurdle", "active_preview_execution_buffer", "active_preview_safe_mode_execution_buffer"]}),
                )
            )
            rows.append(
                _test_row(
                    "active_preview_orders_marked_not_executable",
                    "PASS"
                    if (
                        not orders.empty
                        and orders["preview_context"].astype(str).eq("active_preview").all()
                        and orders["executable"].astype(str).str.lower().isin(["false", "0"]).all()
                        and orders["not_executable"].astype(str).str.lower().isin(["true", "1"]).all()
                        and orders["execution_block_reason"].astype(str).eq("active_preview_never_executes").all()
                    )
                    else "FAIL",
                    orders.to_string(index=False),
                )
            )
            rows.append(
                _test_row(
                    "safe_mode_unchanged_by_active_preview",
                    "PASS" if fixture_scores.loc[fixture_scores["selected"], "discrete_candidate"].tolist() == ["HOLD_CURRENT"] else "FAIL",
                    fixture_scores.loc[fixture_scores["selected"], "discrete_candidate"].to_string(index=False),
                )
            )
            matrix = _write_rebalance_sensitivity_matrix(
                scores_frame=fixture_scores,
                candidate_map=candidate_map,
                current_portfolio_score=0.0010,
                params=active_params,
                safe_model_uncertainty_buffer=0.0009,
                current_state=current_state,
                latest_prices=prices,
                active_tickers=active_tickers,
                output_dir=tmp_dir,
            )
            rows.append(
                _test_row(
                    "sensitivity_matrix_exists_and_has_parameter_grid",
                    "PASS"
                    if (
                        (tmp_dir / "rebalance_sensitivity_matrix.csv").exists()
                        and (tmp_dir / "rebalance_sensitivity_matrix.txt").exists()
                        and len(matrix) == 80
                        and {"trade_now_hurdle", "execution_buffer", "model_uncertainty_multiplier", "final_action_under_params"}.issubset(matrix.columns)
                    )
                    else "FAIL",
                    f"rows={len(matrix)} columns={list(matrix.columns)}",
                )
            )
            sensitivity_text = (tmp_dir / "rebalance_sensitivity_matrix.txt").read_text(encoding="utf-8")
            has_trade = bool(matrix["final_action_under_params"].astype(str).eq("BUY_SELL_PREVIEW").any())
            rows.append(
                _test_row(
                    "sensitivity_matrix_identifies_first_trade_threshold_if_any",
                    "PASS" if ("First trade threshold:" in sensitivity_text and (not has_trade or "candidate=ACTIVE_REPAIR::ROUND_NEAREST_REPAIR_0c" in sensitivity_text)) else "FAIL",
                    sensitivity_text,
                )
            )
    except Exception as exc:
        for name in [
            "active_preview_outputs_exist",
            "active_preview_never_executes_orders",
            "active_preview_uses_lower_hurdles_than_safe_mode",
            "active_preview_orders_marked_not_executable",
            "safe_mode_unchanged_by_active_preview",
            "sensitivity_matrix_exists_and_has_parameter_grid",
            "sensitivity_matrix_identifies_first_trade_threshold_if_any",
        ]:
            rows.append(_test_row(name, "FAIL", str(exc)))

    return pd.DataFrame(rows)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_robustness_tests()
    results.to_csv(output_dir / "robustness_tests_report.csv", index=False)
    text = "\n".join(
        f"{row.status:>4} | {row.test_name} | {row.message}"
        for row in results.itertuples(index=False)
    )
    (output_dir / "robustness_tests_report.txt").write_text(text + "\n", encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
