"""Daily dry-run bot for scenario-weighted RF-adjusted allocation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from audit import create_run_metadata, write_audit_metadata
from conditional_scenario_model import build_conditional_scenarios
from config import (
    DATA_DIR,
    OUTPUT_DIR,
    PORTFOLIO_NAV_USD,
    PRICE_CACHE_PATH,
    build_params,
)
from codex_report import write_codex_debug_report, write_codex_next_prompt
from config_validation import validate_config
from data import build_run_data_context, check_data_freshness, load_price_data, write_data_freshness_report
from data_quality import compute_data_quality_report, save_data_quality_report
from daily_analysis_report import write_daily_analysis_report
from daily_portfolio_review import (
    build_review_issues,
    load_daily_review_settings,
    review_email_send_allowed,
    send_daily_review_email_if_needed,
    write_daily_portfolio_review_outputs,
)
from database import create_run, init_db, save_data_quality_to_db, save_execution_result
from diagnostics import (
    create_run_diagnostics,
    detect_performance_flags,
    log_candidate_selection,
    log_data_context,
    log_data_quality,
    log_error,
    log_execution_gate,
    log_final_action,
    log_optimizer_result,
    log_rejected_order,
    log_stage,
    log_warning,
    write_run_diagnostics,
)
from discrete_portfolio_optimizer import (
    build_discrete_order_preview,
    load_current_portfolio_state,
    validate_portfolio_constraints,
    write_current_portfolio_report,
)
from ensemble_model import build_model_ensemble_outputs, save_model_ensemble_report
from execution_gate import evaluate_execution_gate
from explainability import (
    explain_asset_changes,
    explain_factor_drivers,
    explain_selected_decision,
    save_explainability_reports,
)
from factor_data import build_factor_data
from factor_forecast import build_factor_forecast
from feasibility import check_portfolio_feasibility
from forecast_3m import build_forecast_3m
from macro_data import DEFAULT_PROXY_TICKERS, load_macro_proxy_data
from model_governance import compute_model_confidence, save_model_governance_report
from order_sizing import convert_weights_to_orders
from order_preview import mark_daily_simulator_preview
from paper_broker_stub import PaperBrokerStub
from pre_trade_validation import run_pre_trade_validation, save_pre_trade_validation_report
from reconciliation import build_reconciliation_report, reconcile_before_execution
from regime_engine import detect_regime, save_regime_report
from report import write_output_file_guide
from scenario_model import ScenarioSet, build_3m_scenarios
from scenario_risk_model import (
    build_candidate_risk_return_frame,
    build_scenario_attribution_frame,
    build_scenario_risk_distribution,
    write_candidate_risk_return_reports,
    write_scenario_risk_reports,
)
from scenario_daily_pipeline import run_scenario_weighted_daily_solve
from scenario_weighted_solver import (
    SolverConfig as ScenarioWeightedSolverConfig,
    SolverResult as ScenarioWeightedSolverResult,
    evaluate_weights as evaluate_scenario_weighted_weights,
)
from simulator_orchestrator import run_execution_layer
from system_init import run_system_initialization
from transaction_costs import (
    build_transaction_cost_review_summary,
    compute_trade_now_edge,
    estimate_order_list_costs,
    format_cost_assumptions_summary,
)
from tradability import (
    apply_tradability_filter,
    build_tradability_report,
    save_tradability_report,
    save_tradability_to_db,
    select_cash_proxy,
)
from features import compute_returns
from risk import estimate_robust_covariance_at_date
from asset_exposure_model import estimate_asset_factor_exposures
from calendar_utils import DEFAULT_PROJECT_CALENDAR_PATH, is_within_project_trading_window


LOGGER = logging.getLogger(__name__)
DAILY_BOT_LOCK_PATH = DATA_DIR / "daily_bot.lock"
DAILY_BOT_STATE_PATH = DATA_DIR / "daily_bot_state.json"
LOCK_STALE_AFTER_SECONDS = 2 * 60 * 60
BERLIN_TZ = ZoneInfo("Europe/Berlin")
FINAL_TARGET_SOURCE_SCENARIO = "SCENARIO_WEIGHTED_RF_SHARPE_OPTIMAL"
FINAL_TARGET_SOURCE_SOLVER_FAILED = "HOLD_SOLVER_FAILED"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _normalize_weight_series(
    weights: pd.Series,
    index: pd.Index,
    *,
    fallback_ticker: str | None = None,
) -> pd.Series:
    """Return finite long-only weights on ``index`` with full-investment sum."""

    aligned = (
        weights.copy()
        .rename(index=str)
        .reindex(index)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    total = float(aligned.sum())
    if total > 0.0:
        return aligned / total
    fallback = str(fallback_ticker or "")
    if fallback and fallback in aligned.index:
        aligned.loc[fallback] = 1.0
    elif len(aligned.index) > 0:
        aligned.iloc[0] = 1.0
    return aligned


def _scenario_weighted_solver_config(params: dict[str, object]) -> ScenarioWeightedSolverConfig:
    solver_block = params.get("solver", {})
    solver_params = dict(solver_block) if isinstance(solver_block, dict) else {}
    merged = {**solver_params, **params}
    return ScenarioWeightedSolverConfig(
        lambda_turnover=float(merged.get("lambda_turnover", merged.get("direct_scenario_lambda_turnover", 0.03))),
        lambda_concentration=float(
            merged.get("lambda_concentration", merged.get("direct_scenario_lambda_concentration", 0.01))
        ),
        lambda_downside=float(merged.get("lambda_downside", merged.get("direct_scenario_lambda_downside", 0.15))),
        eps_variance=float(merged.get("eps_variance", merged.get("sharpe_epsilon", 1.0e-10))),
        max_turnover=float(merged.get("max_turnover", 0.75)),
        ftol=float(merged.get("optimizer_ftol", merged.get("direct_scenario_optimizer_ftol", 1.0e-9))),
        maxiter=int(merged.get("optimizer_maxiter", merged.get("direct_scenario_optimizer_maxiter", 1000))),
    )


def _solver_failure_result(message: str, weights: pd.Series) -> ScenarioWeightedSolverResult:
    weights = weights.astype(float).copy()
    return ScenarioWeightedSolverResult(
        success=False,
        status=-1,
        message=str(message),
        weights=weights,
        objective_value=0.0,
        weighted_sharpe=0.0,
        turnover=0.0,
        concentration=float(np.square(weights.to_numpy(dtype=float)).sum()),
        downside_penalty=0.0,
        per_scenario_metrics=pd.DataFrame(
            columns=[
                "scenario",
                "probability",
                "portfolio_return",
                "risk_free_return",
                "excess_return",
                "portfolio_volatility",
                "rf_adjusted_sharpe",
                "downside_shortfall",
            ]
        ),
        constraint_diagnostics={"solver_failed": True, "failure_reason": str(message), "warnings": [str(message)]},
    )


def _optimizer_adapter_from_solver(
    result: ScenarioWeightedSolverResult,
    *,
    target_weights: pd.Series,
) -> SimpleNamespace:
    return SimpleNamespace(
        target_weights=target_weights.copy(),
        solver_name="scenario_weighted_rf_sharpe_slsqp",
        success=bool(result.success),
        objective_value=float(result.objective_value),
        status=str(result.status),
        message=str(result.message),
        diagnostics={
            **dict(result.constraint_diagnostics),
            "weighted_rf_adjusted_sharpe": float(result.weighted_sharpe),
            "turnover": float(result.turnover),
            "concentration": float(result.concentration),
            "downside_penalty": float(result.downside_penalty),
        },
    )


def _execution_fraction(params: dict[str, object]) -> float:
    value = _safe_float(
        params.get("scenario_execution_fraction", params.get("execution_fraction", 1.0)),
        1.0,
    )
    return float(min(max(value, 0.0), 1.0))


def _apply_execution_fraction(
    *,
    current_weights: pd.Series,
    optimal_weights: pd.Series,
    execution_fraction: float,
    fallback_ticker: str | None,
) -> pd.Series:
    index = pd.Index(
        list(dict.fromkeys([*current_weights.index.astype(str), *optimal_weights.index.astype(str)])),
        name="ticker",
    )
    current = _normalize_weight_series(current_weights, index, fallback_ticker=fallback_ticker)
    optimal = _normalize_weight_series(optimal_weights, index, fallback_ticker=fallback_ticker)
    executable = current + float(execution_fraction) * (optimal - current)
    return _normalize_weight_series(executable, index, fallback_ticker=fallback_ticker)


def _correlation_label(correlation: float) -> str:
    if correlation >= 0.75:
        return "strong_positive"
    if correlation >= 0.40:
        return "moderate_positive"
    if correlation > -0.20:
        return "low_or_neutral"
    if correlation > -0.50:
        return "moderate_negative"
    return "strong_negative"


def _diversification_label(
    correlation: float,
    covariance: float | None = None,
    vol_i: float | None = None,
    vol_j: float | None = None,
) -> str:
    try:
        correlation_f = float(correlation)
    except (TypeError, ValueError):
        return "low"
    if not np.isfinite(correlation_f):
        return "low"

    # Correlation drives the label, while covariance/volatility are used as a
    # sanity check so high-volatility, positively co-moving pairs are never
    # accidentally presented as strong diversifiers.
    if covariance is not None and vol_i is not None and vol_j is not None:
        try:
            covariance_f = float(covariance)
            vol_i_f = float(vol_i)
            vol_j_f = float(vol_j)
            if (
                np.isfinite(covariance_f)
                and np.isfinite(vol_i_f)
                and np.isfinite(vol_j_f)
                and vol_i_f > 0.0
                and vol_j_f > 0.0
            ):
                normalized_covariance = covariance_f / (vol_i_f * vol_j_f)
                if normalized_covariance >= 0.50:
                    return "low"
                if normalized_covariance >= 0.20 and correlation_f < 0.20:
                    return "medium"
        except (TypeError, ValueError, FloatingPointError):
            pass

    if correlation_f < 0.20:
        return "high"
    if correlation_f < 0.50:
        return "medium"
    return "low"


def _write_scenario_weighted_solver_reports(
    *,
    scenarios: list,
    result: ScenarioWeightedSolverResult,
    current_weights: pd.Series,
    target_weights: pd.Series,
    executable_weights: pd.Series,
    execution_fraction: float,
    output_dir: Path,
    final_target_source: str,
) -> None:
    probability_rows = [
        {
            "scenario": scenario.name,
            "probability": float(scenario.probability),
            "risk_free_return": float(scenario.risk_free_return),
            "probability_source": str(getattr(scenario, "metadata", {}).get("probability_source", "unknown")),
            "assumption_type": str(getattr(scenario, "metadata", {}).get("assumption_type", "unknown")),
            "diagnostic_warnings": str(getattr(scenario, "metadata", {}).get("diagnostic_warnings", "none")),
        }
        for scenario in scenarios
    ]
    _write_csv(output_dir / "scenario_probabilities.csv", pd.DataFrame(probability_rows), index=False)

    expected_return_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for asset, value in scenario.expected_returns.items():
            expected_return_rows.append(
                {
                    "scenario": scenario.name,
                    "asset": str(asset),
                    "expected_return": float(value),
                    "expected_return_source": str(getattr(scenario, "metadata", {}).get("expected_return_source", "unknown")),
                    "adjustment_source": str(getattr(scenario, "metadata", {}).get("adjustment_source", "unknown")),
                }
            )
        scenario.covariance.to_csv(output_dir / f"scenario_covariance_{scenario.name}.csv")
        vol = np.sqrt(np.maximum(np.diag(scenario.covariance.to_numpy(dtype=float)), 0.0))
        denom = np.outer(vol, vol)
        corr_values = np.divide(
            scenario.covariance.to_numpy(dtype=float),
            denom,
            out=np.zeros_like(scenario.covariance.to_numpy(dtype=float)),
            where=denom > 0.0,
        )
        np.fill_diagonal(corr_values, 1.0)
        corr = pd.DataFrame(corr_values, index=scenario.covariance.index, columns=scenario.covariance.columns)
        corr.to_csv(output_dir / f"scenario_correlation_{scenario.name}.csv")
        assets = list(scenario.covariance.index.astype(str))
        covariance_values = scenario.covariance.to_numpy(dtype=float)
        for i, asset_i in enumerate(assets):
            for j in range(i + 1, len(assets)):
                asset_j = assets[j]
                correlation = float(corr_values[i, j])
                covariance = float(covariance_values[i, j])
                vol_i = float(vol[i])
                vol_j = float(vol[j])
                pairwise_rows.append(
                    {
                        "scenario": scenario.name,
                        "asset_i": asset_i,
                        "asset_j": asset_j,
                        "correlation": correlation,
                        "covariance": covariance,
                        "vol_i": vol_i,
                        "vol_j": vol_j,
                        "relationship_label": _correlation_label(correlation),
                        "diversification_label": _diversification_label(
                            correlation,
                            covariance=covariance,
                            vol_i=vol_i,
                            vol_j=vol_j,
                        ),
                    }
                )
    _write_csv(output_dir / "scenario_expected_returns.csv", pd.DataFrame(expected_return_rows), index=False)
    _write_csv(output_dir / "pairwise_asset_relationships.csv", pd.DataFrame(pairwise_rows), index=False)

    per_scenario = result.per_scenario_metrics.copy()
    if "volatility" in per_scenario.columns and "portfolio_volatility" not in per_scenario.columns:
        per_scenario["portfolio_volatility"] = per_scenario["volatility"]
    required_metric_columns = [
        "scenario",
        "probability",
        "portfolio_return",
        "risk_free_return",
        "excess_return",
        "portfolio_volatility",
        "rf_adjusted_sharpe",
        "downside_shortfall",
    ]
    for column in required_metric_columns:
        if column not in per_scenario.columns:
            per_scenario[column] = np.nan
    _write_csv(
        output_dir / "scenario_solver_per_scenario_metrics.csv",
        per_scenario[required_metric_columns],
        index=False,
    )

    result_frame = pd.DataFrame(
        {
            "asset": target_weights.index.astype(str),
            "current_weight": current_weights.reindex(target_weights.index).fillna(0.0).to_numpy(dtype=float),
            "optimal_weight": target_weights.to_numpy(dtype=float),
            "executable_weight": executable_weights.reindex(target_weights.index).fillna(0.0).to_numpy(dtype=float),
            "delta_optimal": (
                target_weights - current_weights.reindex(target_weights.index).fillna(0.0)
            ).to_numpy(dtype=float),
            "delta_executable": (
                executable_weights.reindex(target_weights.index).fillna(0.0)
                - current_weights.reindex(target_weights.index).fillna(0.0)
            ).to_numpy(dtype=float),
        }
    )
    _write_csv(output_dir / "scenario_solver_result.csv", result_frame, index=False)
    turnover_executable = float(
        (
            executable_weights.reindex(target_weights.index).fillna(0.0)
            - current_weights.reindex(target_weights.index).fillna(0.0)
        )
        .abs()
        .sum()
    )
    result_diagnostics = dict(result.constraint_diagnostics or {})
    solver_failed = bool(result_diagnostics.get("solver_failed", not bool(result.success)))
    failure_reason = str(result_diagnostics.get("failure_reason", "" if result.success else result.message))
    metrics = pd.DataFrame(
        [
            {"metric": "final_target_source", "value": final_target_source},
            {"metric": "solver_failed", "value": solver_failed},
            {"metric": "failure_reason", "value": failure_reason or "none"},
            {"metric": "execution_fraction", "value": float(execution_fraction)},
            {"metric": "execution_damping_applied", "value": bool(float(execution_fraction) < 1.0 - 1.0e-12)},
            {"metric": "objective_value", "value": float(result.objective_value)},
            {"metric": "weighted_rf_adjusted_sharpe", "value": float(result.weighted_sharpe)},
            {"metric": "turnover_optimal", "value": float(result.turnover)},
            {"metric": "turnover_executable", "value": turnover_executable},
            {"metric": "concentration", "value": float(result.concentration)},
            {"metric": "downside_penalty", "value": float(result.downside_penalty)},
            {"metric": "success", "value": bool(result.success)},
            {"metric": "status", "value": int(result.status)},
            {"metric": "message", "value": str(result.message)},
        ]
    )
    _write_csv(output_dir / "scenario_solver_metrics.csv", metrics, index=False)
    top_weights = target_weights.sort_values(ascending=False).head(8)
    top_executable_weights = executable_weights.reindex(target_weights.index).fillna(0.0).sort_values(ascending=False).head(8)
    per_scenario_rank = per_scenario.sort_values("rf_adjusted_sharpe", ascending=False) if "rf_adjusted_sharpe" in per_scenario else pd.DataFrame()
    high_diversifiers = (
        pd.DataFrame(pairwise_rows)
        .sort_values(["diversification_label", "correlation"], ascending=[True, True])
        .head(8)
        if pairwise_rows
        else pd.DataFrame()
    )
    decision_lines = [
        "Scenario-Weighted RF-Adjusted Sharpe Solver Decision",
        "",
        f"Final Target Source: {final_target_source}",
        f"solver_failed: {solver_failed}",
        f"failure_reason: {failure_reason or 'none'}",
        f"solver_success: {bool(result.success)}",
        f"solver_status: {result.status}",
        f"solver_message: {result.message}",
        f"objective_value: {float(result.objective_value):.8f}",
        f"weighted_rf_adjusted_sharpe: {float(result.weighted_sharpe):.8f}",
        f"turnover_optimal: {float(result.turnover):.6f}",
        f"turnover_executable: {turnover_executable:.6f}",
        f"execution_fraction: {float(execution_fraction):.4f}",
        f"execution_damping_applied: {bool(float(execution_fraction) < 1.0 - 1.0e-12)}",
        f"concentration: {float(result.concentration):.6f}",
        f"downside_penalty: {float(result.downside_penalty):.6f}",
        "",
        "Optimal allocation top weights:",
        *[f"- {asset}: {float(weight):.2%}" for asset, weight in top_weights.items()],
        "",
        "Executable allocation top weights:",
        *[f"- {asset}: {float(weight):.2%}" for asset, weight in top_executable_weights.items()],
        "",
        "Scenario drivers:",
        *(
            [
                f"- {row.scenario}: sharpe={float(row.rf_adjusted_sharpe):.4f}, return={float(row.portfolio_return):.4f}, vol={float(row.portfolio_volatility):.4f}"
                for row in per_scenario_rank.head(6).itertuples(index=False)
            ]
            if not per_scenario_rank.empty
            else ["- no per-scenario metrics available"]
        ),
        "",
        "Diversification evidence:",
        *(
            [
                f"- {row.scenario}: {row.asset_i}/{row.asset_j} corr={float(row.correlation):.2f}, covariance={float(row.covariance):.6f}, diversification={row.diversification_label}"
                for row in high_diversifiers.itertuples(index=False)
            ]
            if not high_diversifiers.empty
            else ["- no pairwise covariance relationships available"]
        ),
        "",
        "Decision logic:",
        "- Final allocation is solved directly over weights w.",
        "- Legacy candidates are diagnostic benchmarks only and do not determine final target weights.",
        "- Scenario Sharpe is RF-adjusted and uses each scenario's own covariance matrix.",
        "- Assets are limited by hard caps, group caps, turnover and covariance-aware scenario volatility.",
        "- The result is not only momentum: momentum can shape scenario expected returns, but covariance, risk-free excess return, downside, concentration and turnover all enter the objective.",
        "- Execution damping is applied only after mathematical optimization as a trading-layer adjustment.",
    ]
    _write_text(output_dir / "scenario_solver_decision.md", "\n".join(decision_lines) + "\n")


def _force_solver_discrete_selection(
    scored_discrete_candidates: dict[str, object],
    *,
    solver_source_name: str,
    final_target_source: str,
    solver_failed: bool,
) -> dict[str, object]:
    """Choose the final whole-share target from the solver source, not legacy candidate ranking."""

    scores_frame = scored_discrete_candidates["scores_frame"].copy()
    candidate_map = scored_discrete_candidates["candidate_map"]
    if scores_frame.empty:
        raise ValueError("No discrete candidates were generated for solver execution target.")
    scores_frame["selected"] = False
    scores_frame["final_target_source"] = final_target_source
    scores_frame["manual_candidate_selection_for_final_target"] = False
    scores_frame["final_allocation_method"] = "scenario_weighted_rf_sharpe_solver"

    if solver_failed:
        candidate_rows = scores_frame[scores_frame["discrete_candidate"].astype(str) == "HOLD_CURRENT"].copy()
        selected_reason = "solver_failed_hold_fallback"
        reason = "Scenario-weighted solver failed; final target is current portfolio and execution is fail-closed."
    else:
        candidate_rows = scores_frame[
            (scores_frame["continuous_source"].astype(str) == str(solver_source_name))
            & (scores_frame["discrete_candidate"].astype(str) != "HOLD_CURRENT")
            & (scores_frame["valid_constraints"] == True)  # noqa: E712
        ].copy()
        selected_reason = "scenario_weighted_solver_execution_target"
        reason = (
            "Final target comes from SCENARIO_WEIGHTED_RF_SHARPE_OPTIMAL; "
            "whole-share variant selected by closest feasible rounding, not legacy candidate ranking."
        )
    if candidate_rows.empty:
        candidate_rows = scores_frame[
            (scores_frame["continuous_source"].astype(str) == str(solver_source_name))
            & (scores_frame["discrete_candidate"].astype(str) != "HOLD_CURRENT")
        ].copy()
        selected_reason = "solver_target_rounding_constraint_fallback"
        reason = "No valid rounded solver target was available; closest solver rounding is reported and gates must block if invalid."
    if candidate_rows.empty:
        candidate_rows = scores_frame[scores_frame["discrete_candidate"].astype(str) == "HOLD_CURRENT"].copy()
        selected_reason = "solver_target_no_discrete_candidate_hold_fallback"
        reason = "No rounded solver target was available; HOLD_CURRENT is used as fail-closed fallback."
    if candidate_rows.empty:
        candidate_rows = scores_frame.copy()

    sort_columns = [
        "total_abs_weight_drift",
        "max_abs_weight_drift",
        "number_of_orders",
        "cash_left",
        "net_robust_score",
    ]
    for column in sort_columns:
        if column not in candidate_rows.columns:
            candidate_rows[column] = 0.0
    candidate_rows = candidate_rows.sort_values(
        sort_columns,
        ascending=[True, True, True, True, False],
        kind="mergesort",
    )
    chosen_row = candidate_rows.iloc[0]
    best_name = str(chosen_row["discrete_candidate"])
    scores_frame.loc[scores_frame["discrete_candidate"].astype(str) == best_name, "selected"] = True
    scores_frame["selected_reason"] = selected_reason
    scores_frame["candidate_role"] = np.where(
        scores_frame["discrete_candidate"].astype(str) == best_name,
        "final_solver_whole_share_target",
        "diagnostic_rounding_or_legacy_benchmark",
    )

    hold_rows = scores_frame[scores_frame["discrete_candidate"].astype(str) == "HOLD_CURRENT"]
    hold_row = hold_rows.iloc[0] if not hold_rows.empty else None
    hold_current_constraint_valid = bool(hold_row["valid_constraints"]) if hold_row is not None else False
    current_constraint_errors = str(hold_row.get("validation_errors", "") if hold_row is not None else "").strip()
    non_hold_rows = scores_frame[scores_frame["discrete_candidate"].astype(str) != "HOLD_CURRENT"].copy()
    best_non_hold_row = (
        non_hold_rows.sort_values("net_robust_score", ascending=False, kind="mergesort").iloc[0]
        if not non_hold_rows.empty
        else None
    )
    best_non_hold_candidate = str(best_non_hold_row["discrete_candidate"]) if best_non_hold_row is not None else ""
    best_non_hold_score = float(best_non_hold_row["net_robust_score"]) if best_non_hold_row is not None else np.nan
    best_non_hold_valid_constraints = bool(best_non_hold_row["valid_constraints"]) if best_non_hold_row is not None else False
    best_non_hold_failed_reason = (
        str(best_non_hold_row.get("selection_failed_reason", "") or "")
        if best_non_hold_row is not None
        else ""
    )
    if best_non_hold_row is not None and not best_non_hold_valid_constraints and not best_non_hold_failed_reason:
        best_non_hold_failed_reason = "failed_constraints"
    for column, value in {
        "hold_current_constraint_valid": hold_current_constraint_valid,
        "current_portfolio_constraint_violation": not hold_current_constraint_valid,
        "current_constraint_errors": current_constraint_errors,
        "best_non_hold_candidate": best_non_hold_candidate,
        "best_non_hold_score": best_non_hold_score,
        "best_non_hold_valid_constraints": best_non_hold_valid_constraints,
        "best_non_hold_failed_reason": best_non_hold_failed_reason,
        "best_model_candidate": str(solver_source_name),
        "best_model_candidate_valid_constraints": bool(chosen_row.get("valid_constraints", False)),
        "final_selection_is_safe_fallback": bool(best_name == "HOLD_CURRENT" or solver_failed),
    }.items():
        scores_frame[column] = value

    candidate = candidate_map[best_name]
    objective_score = float(chosen_row.get("_selection_objective_score", chosen_row.get("net_robust_score", 0.0)))
    return {
        "best_discrete_candidate_name": best_name,
        "best_discrete_weights": candidate.weights_proxy.copy(),
        "best_discrete_shares": candidate.shares.copy(),
        "best_discrete_score": float(chosen_row.get("net_robust_score", 0.0)),
        "best_discrete_objective_score": objective_score,
        "objective_used": "scenario_weighted_rf_sharpe",
        "objective_score_column": "solver_target_closest_feasible_rounding",
        "hold_objective_score": float(hold_row["net_robust_score"]) if hold_row is not None else float("-inf"),
        "hold_net_robust_score": float(hold_row["net_robust_score"]) if hold_row is not None else float("-inf"),
        "best_discrete_allocation": candidate.weights_actual.copy(),
        "best_discrete_orders": candidate.shares.copy(),
        "reason": reason,
        "selected_reason": selected_reason,
        "hold_current_constraint_valid": hold_current_constraint_valid,
        "current_portfolio_constraint_violation": not hold_current_constraint_valid,
        "current_constraint_errors": current_constraint_errors,
        "best_non_hold_candidate": best_non_hold_candidate,
        "best_non_hold_score": best_non_hold_score,
        "best_non_hold_objective_score": best_non_hold_score,
        "best_non_hold_valid_constraints": best_non_hold_valid_constraints,
        "best_non_hold_failed_reason": best_non_hold_failed_reason,
        "best_model_candidate": str(solver_source_name),
        "best_model_candidate_valid_constraints": bool(chosen_row.get("valid_constraints", False)),
        "final_selection_is_safe_fallback": bool(best_name == "HOLD_CURRENT" or solver_failed),
        "scores_frame": scores_frame,
        "candidate": candidate,
    }


def _write_scenario_weighted_allocation_csv(
    *,
    current_state,
    optimal_weights: pd.Series,
    executable_weights: pd.Series,
    latest_prices: pd.Series,
    params: dict[str, object],
    output_dir: Path,
) -> None:
    assets = pd.Index(
        list(
            dict.fromkeys(
                [
                    *current_state.current_weights_proxy.index.astype(str).tolist(),
                    *optimal_weights.index.astype(str).tolist(),
                    *executable_weights.index.astype(str).tolist(),
                ]
            )
        ),
        name="ticker",
    )
    current_weights = current_state.current_weights_proxy.reindex(assets).fillna(0.0).astype(float)
    optimal = optimal_weights.rename(index=str).reindex(assets).fillna(0.0).astype(float)
    executable = executable_weights.rename(index=str).reindex(assets).fillna(0.0).astype(float)
    nav = float(current_state.nav)
    current_values = current_state.current_values.reindex(assets).fillna(0.0).astype(float)
    max_weights = pd.Series(params.get("asset_max_weights", {}), dtype=float).rename(index=str).reindex(assets).fillna(1.0)
    group_map = pd.Series(params.get("group_map", {}), dtype=object).rename(index=str).reindex(assets).fillna("unknown")
    executable_values = executable * nav
    trade_values = executable_values - current_values
    min_order_value = float(params.get("min_order_value_usd", 10.0))
    trade_side = pd.Series("HOLD", index=assets, dtype=object)
    trade_side = trade_side.where(~trade_values.gt(min_order_value), "BUY")
    trade_side = trade_side.where(~trade_values.lt(-min_order_value), "SELL")
    allocation = pd.DataFrame(
        {
            "asset": assets.astype(str),
            "current_weight": current_weights.to_numpy(dtype=float),
            "optimal_weight": optimal.to_numpy(dtype=float),
            "executable_weight": executable.to_numpy(dtype=float),
            "delta_optimal": (optimal - current_weights).to_numpy(dtype=float),
            "delta_executable": (executable - current_weights).to_numpy(dtype=float),
            "max_weight": max_weights.to_numpy(dtype=float),
            "group": group_map.astype(str).to_numpy(),
            "current_value_usd": current_values.to_numpy(dtype=float),
            "optimal_value_usd": (optimal * nav).to_numpy(dtype=float),
            "executable_value_usd": executable_values.to_numpy(dtype=float),
            "trade_value_usd": trade_values.to_numpy(dtype=float),
            "trade_side": trade_side.to_numpy(),
        }
    )
    _write_csv(output_dir / "scenario_weighted_optimal_allocation.csv", allocation, index=False)


def _build_cost_preview_from_sized_orders(sized_orders: pd.DataFrame) -> pd.DataFrame:
    """Adapt slim order-sizing rows to the existing transaction-cost schema."""

    preview = sized_orders.copy()
    side = preview["trade_side"].astype(str).str.upper()
    signed_value = pd.to_numeric(preview["estimated_order_value_usd"], errors="coerce").fillna(0.0)
    signed_value = signed_value.where(~side.eq("SELL"), -signed_value)
    signed_value = signed_value.where(side.isin(["BUY", "SELL"]), 0.0)
    return pd.DataFrame(
        {
            "ticker": preview["asset"].astype(str),
            "side": side,
            "current_shares": pd.to_numeric(preview["current_shares"], errors="coerce").fillna(0.0),
            "target_shares": pd.to_numeric(preview["target_shares"], errors="coerce").fillna(0.0),
            "shares_delta": pd.to_numeric(preview["share_delta"], errors="coerce").fillna(0.0),
            "order_shares": pd.to_numeric(preview["share_delta"], errors="coerce").fillna(0.0).abs(),
            "estimated_shares": pd.to_numeric(preview["share_delta"], errors="coerce").fillna(0.0).abs(),
            "latest_price": pd.to_numeric(preview["latest_price"], errors="coerce"),
            "order_value": signed_value,
            "estimated_order_value": signed_value.abs(),
            "reason": preview["skipped_reason"].astype(str),
            "not_executable": False,
        }
    )


def _write_disabled_legacy_daily_candidate_artifacts(*, output_dir: Path, final_order_file: Path) -> None:
    """Overwrite stale legacy candidate artifacts with explicit disabled notices."""

    note = (
        "Legacy candidate/discrete ranking is disabled in the active Daily run. "
        "Use scenario_weighted_optimal_allocation.csv and scenario_weighted_order_preview.csv."
    )
    disabled = pd.DataFrame(
        [
            {
                "report_role": "legacy",
                "status": "disabled_in_active_daily_run",
                "final_target_source": FINAL_TARGET_SOURCE_SCENARIO,
                "final_order_file": str(final_order_file),
                "note": note,
            }
        ]
    )
    for csv_name in [
        "discrete_candidate_scores.csv",
        "best_discrete_allocation.csv",
        "best_discrete_order_preview.csv",
    ]:
        _write_csv(output_dir / csv_name, disabled, index=False)
    _write_text(
        output_dir / "discrete_optimization_report.txt",
        "\n".join(
            [
                "Legacy Discrete Candidate Report",
                "",
                "status: disabled_in_active_daily_run",
                f"final_order_file: {final_order_file}",
                note,
                "",
            ]
        ),
    )


def _write_slim_manual_simulator_orders(
    *,
    scenario_preview: pd.DataFrame,
    gate_action: str,
    gate_status: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Write a safe manual simulator file for the slim scenario-weighted path."""

    columns = [
        "asset",
        "action",
        "shares",
        "estimated_price",
        "estimated_order_value_usd",
        "note",
    ]
    if gate_status == "PASS" and str(gate_action) == "BUY_SELL_PREVIEW":
        actionable = scenario_preview.loc[
            scenario_preview["trade_side"].astype(str).isin(["BUY", "SELL"])
        ].copy()
    else:
        actionable = pd.DataFrame(columns=scenario_preview.columns)

    if actionable.empty:
        manual = pd.DataFrame(columns=columns)
        text = (
            "Manual Simulator Orders\n\n"
            "No manual simulator orders are eligible for this run.\n"
            f"gate_status: {gate_status}\n"
            f"gate_action: {gate_action}\n"
            "final_order_file: outputs/scenario_weighted_order_preview.csv\n"
        )
    else:
        manual = pd.DataFrame(
            {
                "asset": actionable["asset"].astype(str),
                "action": actionable["trade_side"].astype(str),
                "shares": pd.to_numeric(actionable["share_delta"], errors="coerce").fillna(0.0).abs(),
                "estimated_price": pd.to_numeric(actionable["latest_price"], errors="coerce").fillna(0.0),
                "estimated_order_value_usd": pd.to_numeric(
                    actionable["estimated_order_value_usd"], errors="coerce"
                ).fillna(0.0),
                "note": "Preview only. No automatic execution.",
            }
        )
        text = "Manual Simulator Orders\n\n" + manual.to_string(index=False) + "\n"

    _write_csv(output_dir / "manual_simulator_orders.csv", manual, index=False)
    _write_text(output_dir / "manual_simulator_orders.txt", text)
    return manual


def _finalize_slim_scenario_daily_run(
    *,
    args: argparse.Namespace,
    diagnostics,
    params: dict[str, object],
    effective_dry_run: bool,
    as_of: pd.Timestamp,
    active_tickers: list[str],
    latest_prices_at_asof: pd.Series,
    current_state,
    data_context: dict[str, object],
    data_freshness: dict[str, object],
    market_gate: dict[str, object],
    final_target_source: str,
    scenario_inputs: list,
    solver_config: ScenarioWeightedSolverConfig,
    solver_current_weights: pd.Series,
    scenario_solver_result: ScenarioWeightedSolverResult,
    solver_validation: dict[str, object],
    optimal_solver_weights: pd.Series,
    executable_solver_weights: pd.Series,
    execution_fraction_value: float,
) -> dict[str, object]:
    """Finish the active Daily path without legacy candidate ranking."""

    log_stage(diagnostics, "slim scenario finalization", "START")
    output_dir = OUTPUT_DIR
    assets = pd.Index(active_tickers, name="ticker")
    current_positions = pd.DataFrame(
        {
            "asset": assets.astype(str),
            "shares": current_state.current_shares.reindex(assets).fillna(0.0).to_numpy(dtype=float),
        }
    )

    order_sizing_target = executable_solver_weights.reindex(assets).fillna(0.0).astype(float)
    if final_target_source == FINAL_TARGET_SOURCE_SOLVER_FAILED:
        order_sizing_target = current_state.current_weights_proxy.reindex(assets).fillna(0.0).astype(float)

    sized_orders = convert_weights_to_orders(
        current_positions=current_positions,
        target_weights=order_sizing_target,
        latest_prices=latest_prices_at_asof.reindex(assets),
        total_portfolio_value_usd=float(current_state.nav),
        min_order_value_usd=float(params.get("min_order_value_usd", 10.0)),
        fractional_shares=bool(params.get("allow_fractional_shares", False)),
        cash_buffer_usd=float(params.get("cash_buffer_usd", 0.0)),
    )
    scenario_preview = pd.DataFrame(
        {
            "asset": sized_orders["asset"].astype(str),
            "trade_side": sized_orders["trade_side"].astype(str),
            "trade_value_usd": pd.to_numeric(sized_orders["trade_value_usd"], errors="coerce").fillna(0.0),
            "current_weight": pd.to_numeric(sized_orders["current_weight"], errors="coerce").fillna(0.0),
            "executable_weight": pd.to_numeric(sized_orders["target_weight"], errors="coerce").fillna(0.0),
            "current_value_usd": pd.to_numeric(sized_orders["current_value_usd"], errors="coerce").fillna(0.0),
            "executable_value_usd": pd.to_numeric(sized_orders["target_value_usd"], errors="coerce").fillna(0.0),
            "latest_price": pd.to_numeric(sized_orders["latest_price"], errors="coerce"),
            "current_shares": pd.to_numeric(sized_orders["current_shares"], errors="coerce").fillna(0.0),
            "target_shares": pd.to_numeric(sized_orders["target_shares"], errors="coerce").fillna(0.0),
            "share_delta": pd.to_numeric(sized_orders["share_delta"], errors="coerce").fillna(0.0),
            "estimated_order_value_usd": pd.to_numeric(
                sized_orders["estimated_order_value_usd"], errors="coerce"
            ).fillna(0.0),
            "min_order_value_usd": float(params.get("min_order_value_usd", 10.0)),
            "skipped_reason": sized_orders["skipped_reason"].astype(str),
        }
    )
    if final_target_source == FINAL_TARGET_SOURCE_SOLVER_FAILED:
        scenario_preview["skipped_reason"] = scenario_preview["skipped_reason"].where(
            scenario_preview["skipped_reason"].ne(""),
            "solver_failed",
        )
    elif not bool(market_gate.get("execution_allowed", False)):
        actionable = scenario_preview["trade_side"].astype(str).isin(["BUY", "SELL"])
        scenario_preview.loc[actionable, "skipped_reason"] = "outside_trading_window"

    _write_csv(output_dir / "scenario_weighted_order_preview.csv", scenario_preview, index=False)
    _write_scenario_weighted_allocation_csv(
        current_state=current_state,
        optimal_weights=optimal_solver_weights.reindex(assets).fillna(0.0),
        executable_weights=executable_solver_weights.reindex(assets).fillna(0.0),
        latest_prices=latest_prices_at_asof.reindex(assets),
        params=params,
        output_dir=output_dir,
    )

    cost_preview = _build_cost_preview_from_sized_orders(sized_orders)
    cost_preview, order_cost_summary = estimate_order_list_costs(
        order_preview_df=cost_preview,
        latest_prices=latest_prices_at_asof.reindex(assets),
        config={
            **params,
            "nav": float(current_state.nav),
            "current_cash": float(current_state.current_cash),
        },
    )

    current_eval_score = 0.0
    executable_eval_score = 0.0
    if scenario_inputs:
        try:
            current_eval = evaluate_scenario_weighted_weights(
                solver_current_weights,
                solver_current_weights,
                scenario_inputs,
                solver_config,
            )
            executable_eval = evaluate_scenario_weighted_weights(
                executable_solver_weights.reindex(solver_current_weights.index).fillna(0.0),
                solver_current_weights,
                scenario_inputs,
                solver_config,
            )
            current_eval_score = float(current_eval.objective_value)
            executable_eval_score = float(executable_eval.objective_value)
        except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            log_warning(
                diagnostics,
                "daily_bot",
                f"Slim scenario score evaluation failed; execution edge will be conservative: {exc}",
                severity="WARNING",
                stage="slim scenario finalization",
            )

    target_score_after_costs = executable_eval_score - float(order_cost_summary["total_order_cost_pct_nav"])
    execution_buffer_value = float(params.get("effective_execution_buffer", params.get("execution_buffer", 0.001)))
    model_uncertainty_buffer_value = float(
        params.get("effective_model_uncertainty_buffer", params.get("model_uncertainty_buffer", 0.001))
    )
    trade_edge_summary = compute_trade_now_edge(
        current_score=current_eval_score,
        target_score_after_costs=target_score_after_costs,
        total_order_cost=float(order_cost_summary["total_estimated_transaction_cost"]),
        execution_buffer=execution_buffer_value,
        model_uncertainty_buffer=model_uncertainty_buffer_value,
        other_penalties=0.0,
    )
    selected_name = final_target_source
    selected_score = SimpleNamespace(
        delta_vs_hold=float(target_score_after_costs - current_eval_score),
        estimated_cost=float(order_cost_summary["total_order_cost_pct_nav"]),
    )
    gate = evaluate_execution_gate(
        selection_result=SimpleNamespace(
            selected_candidate=SimpleNamespace(name=selected_name),
            selected_score=selected_score,
        ),
        synthetic_data=bool(data_context.get("synthetic_data", False)),
        data_freshness_ok=bool(data_freshness.get("data_freshness_ok", False)),
        broker_state_reconciled=True,
        open_orders_exist=False,
        estimated_spread_cost=float(order_cost_summary["total_estimated_spread_cost"]) / max(float(current_state.nav), 1e-12),
        estimated_slippage=float(order_cost_summary["total_estimated_slippage_cost"]) / max(float(current_state.nav), 1e-12),
        estimated_transaction_cost=float(order_cost_summary["total_order_cost_pct_nav"]),
        delta_vs_hold_is_net=True,
        costs_include_spread_slippage=True,
        execution_uncertainty_buffer=execution_buffer_value,
        model_uncertainty_buffer=model_uncertainty_buffer_value,
        trade_now_hurdle=float(params.get("effective_trade_now_hurdle", params.get("trade_now_hurdle", 0.0025))),
    )

    preview_order_count = int(scenario_preview["trade_side"].astype(str).isin(["BUY", "SELL"]).sum())
    if final_target_source == FINAL_TARGET_SOURCE_SOLVER_FAILED:
        gate.gate_status = "BLOCK"
        gate.action = FINAL_TARGET_SOURCE_SOLVER_FAILED
        gate.reason = (
            "Scenario-weighted solver failed or failed validation; current portfolio is retained."
        )
    elif not bool(market_gate.get("execution_allowed", False)):
        gate.gate_status = "BLOCK"
        gate.action = (
            "WAIT_MARKET_CLOSED"
            if not bool(market_gate.get("is_trading_day", False))
            else "WAIT_OUTSIDE_WINDOW"
        )
        gate.reason = f"Project calendar blocked execution: {market_gate.get('reason', 'calendar_blocked')}."
    elif preview_order_count == 0:
        gate.gate_status = "BLOCK"
        gate.action = "HOLD"
        gate.reason = "Scenario-weighted executable target creates no order above the minimum order value."
    elif gate.gate_status == "PASS":
        gate.action = "BUY_SELL_PREVIEW"
        gate.reason = "Execution gate passed; DRY_RUN keeps this as a preview only."
    elif gate.action == "WAIT" and "execution hurdle" in str(gate.reason).lower():
        gate.action = "HOLD"

    manual_orders = _write_slim_manual_simulator_orders(
        scenario_preview=scenario_preview,
        gate_action=gate.action,
        gate_status=gate.gate_status,
        output_dir=output_dir,
    )
    _write_disabled_legacy_daily_candidate_artifacts(
        output_dir=output_dir,
        final_order_file=output_dir / "scenario_weighted_order_preview.csv",
    )

    log_execution_gate(
        diagnostics,
        {
            "gate_status": gate.gate_status,
            "action": gate.action,
            "reason": gate.reason,
            "trade_now_score": gate.trade_now_score,
            "spread_cost": gate.spread_cost,
            "slippage": gate.slippage,
            "buffers": gate.buffers,
        },
    )
    diagnostics.execution_mode = "dry_run_preview_only" if effective_dry_run else "preview_only_no_auto_execution"
    diagnostics.final_orders_summary = {
        "order_count": preview_order_count,
        "manual_eligible_order_count": int(len(manual_orders)),
        "estimated_cost": float(order_cost_summary["total_order_cost_pct_nav"]),
        "estimated_cost_usd": float(order_cost_summary["total_estimated_transaction_cost"]),
        "turnover": float(
            (
                executable_solver_weights.reindex(assets).fillna(0.0)
                - current_state.current_weights_proxy.reindex(assets).fillna(0.0)
            )
            .abs()
            .sum()
        ),
    }
    diagnostics.model_context["active_daily_path"] = {
        "path": "scenario_weighted_solver_slim",
        "legacy_candidate_selection_active": False,
        "legacy_discrete_candidate_selection_active": False,
        "final_order_file": str(output_dir / "scenario_weighted_order_preview.csv"),
    }
    log_final_action(diagnostics, gate.action, selected_candidate=final_target_source, reason=gate.reason)

    report_lines = [
        "Daily Bot Decision Report",
        "",
        f"Run Status: {'BLOCKED' if gate.gate_status != 'PASS' else 'PREVIEW_ONLY'}",
        "Active Daily Path: scenario_weighted_solver_slim",
        f"Final Target Source: {final_target_source}",
        "Legacy Candidate Selection Active: false",
        "Legacy Discrete Candidate Selection Active: false",
        "Legacy Candidate Ranking Reports Active Daily: false",
        "Execution: no automatic live/broker/simulator submission from this path",
        "",
        "Files:",
        f"- Final allocation CSV: {output_dir / 'scenario_weighted_optimal_allocation.csv'}",
        f"- Final order preview CSV: {output_dir / 'scenario_weighted_order_preview.csv'}",
        f"- Solver decision report: {output_dir / 'scenario_solver_decision.md'}",
        f"- Manual simulator CSV: {output_dir / 'manual_simulator_orders.csv'}",
        "",
        "State And Data:",
        f"- as_of: {as_of.date()}",
        f"- state_source: {current_state.source}",
        f"- data_source: {data_context.get('data_source', 'unknown')}",
        f"- used_cache_fallback: {data_context.get('used_cache_fallback', False)}",
        f"- synthetic_data: {data_context.get('synthetic_data', False)}",
        f"- latest_price_date: {data_context.get('latest_price_date', 'n/a')}",
        f"- expected_latest_trading_day: {data_context.get('expected_latest_trading_day', 'n/a')}",
        f"- data_freshness_ok: {data_freshness.get('data_freshness_ok', False)}",
        "",
        "Calendar / Gate:",
        f"- is_project_trading_day: {market_gate.get('is_trading_day', 'n/a')}",
        f"- within_allowed_window: {market_gate.get('within_allowed_window', 'n/a')}",
        f"- execution_allowed_by_calendar: {market_gate.get('execution_allowed', 'n/a')}",
        f"- calendar_reason: {market_gate.get('reason', 'n/a')}",
        f"- gate_status: {gate.gate_status}",
        f"- final_action: {gate.action}",
        f"- gate_reason: {gate.reason}",
        f"- trade_now_edge: {float(trade_edge_summary['trade_now_edge']):.8f}",
        f"- trade_now_hurdle: {float(params.get('effective_trade_now_hurdle', params.get('trade_now_hurdle', 0.0025))):.8f}",
        "",
        "Solver:",
        f"- solver_success: {bool(scenario_solver_result.success)}",
        f"- solver_failed: {final_target_source == FINAL_TARGET_SOURCE_SOLVER_FAILED}",
        f"- failure_reason: {scenario_solver_result.constraint_diagnostics.get('failure_reason') or 'none'}",
        f"- post_solver_validation_ok: {bool(solver_validation.get('ok', False))}",
        f"- weighted_rf_adjusted_sharpe: {float(scenario_solver_result.weighted_sharpe):.8f}",
        f"- execution_fraction: {float(execution_fraction_value):.4f}",
        "",
        "Orders:",
        f"- preview_order_count: {preview_order_count}",
        f"- manual_eligible_order_count: {len(manual_orders)}",
        f"- buy_count: {int((scenario_preview['trade_side'].astype(str) == 'BUY').sum())}",
        f"- sell_count: {int((scenario_preview['trade_side'].astype(str) == 'SELL').sum())}",
        f"- total_estimated_transaction_cost_usd: {float(order_cost_summary['total_estimated_transaction_cost']):.2f}",
        f"- cash_before_orders: {float(order_cost_summary['cash_before_orders']):.2f}",
        f"- cash_after_orders: {float(order_cost_summary['cash_after_orders']):.2f}",
        "",
        "Legacy Artifacts:",
        "- discrete_candidate_scores.csv, best_discrete_allocation.csv and best_discrete_order_preview.csv are legacy-disabled notices in the active Daily run.",
        "- Old candidate modules remain available for audits/tests but do not determine the Daily final target.",
        "",
    ]
    report_text = "\n".join(report_lines)
    for name in [
        "daily_bot_decision_report.txt",
        "latest_decision_report.txt",
        "today_decision_summary.txt",
        "rebalance_decision_report.txt",
        "final_acceptance_report.txt",
    ]:
        _write_text(output_dir / name, report_text)
    write_output_file_guide(output_dir / "output_file_guide.txt")
    log_stage(diagnostics, "slim scenario finalization", "DONE", extra=diagnostics.model_context["active_daily_path"])
    return {
        "as_of": as_of,
        "factor_mode": "scenario_weighted_solver_slim",
        "continuous_candidate": final_target_source,
        "final_allocation_method": "scenario_weighted_rf_sharpe_solver",
        "final_target_source": final_target_source,
        "selected_candidate": final_target_source,
        "gate_action": gate.action,
        "execution_mode": diagnostics.execution_mode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily 3M forward-looking dry-run bot.")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run mode.")
    parser.add_argument("--broker", default="none", help="Optional future broker selector.")
    parser.add_argument("--portfolio-value", type=float, default=float(PORTFOLIO_NAV_USD))
    parser.add_argument("--skip-submit", action="store_true", help="Skip any execution submission step.")
    parser.add_argument("--mode", choices=["single", "continuous"], default="single")
    parser.add_argument("--check-interval-minutes", type=int, default=15)
    parser.add_argument("--full-recompute-interval-minutes", type=int, default=60)
    parser.add_argument("--force-refresh", action="store_true", help="Force a live data refresh before decision-making.")
    return parser.parse_args()


def _ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (Path(__file__).resolve().parent / "notebooks").mkdir(parents=True, exist_ok=True)


def _data_tickers(base_tickers: list[str]) -> list[str]:
    return list(dict.fromkeys([*base_tickers, *DEFAULT_PROXY_TICKERS]))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str))
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _default_daily_bot_state() -> dict[str, object]:
    return {
        "current_date": "",
        "current_iso_week": "",
        "current_month": "",
        "last_order_timestamp": None,
        "last_decision_id": "",
        "last_order_signature": "",
        "last_selected_candidate": "",
        "last_trade_now_score": 0.0,
        "orders_today": 0,
        "turnover_today": 0.0,
        "turnover_week": 0.0,
        "turnover_month": 0.0,
        "executed_order_ids": [],
        "last_execution_status": "",
    }


def load_daily_bot_state(path: Path = DAILY_BOT_STATE_PATH) -> dict[str, object]:
    state = _default_daily_bot_state()
    if not path.exists():
        return state
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return state
    if not isinstance(payload, dict):
        return state
    state.update(payload)
    if not isinstance(state.get("executed_order_ids"), list):
        state["executed_order_ids"] = []
    return state


def save_daily_bot_state(path: Path, state: dict[str, object]) -> None:
    _write_state(path, state)


def reset_state_periods_if_needed(state: dict[str, object], current_date: pd.Timestamp) -> dict[str, object]:
    updated = dict(state)
    current_date = pd.Timestamp(current_date).normalize()
    iso_week = current_date.isocalendar()
    week_key = f"{int(iso_week.year)}-W{int(iso_week.week):02d}"
    month_key = current_date.strftime("%Y-%m")
    date_key = str(current_date.date())

    if str(updated.get("current_date", "")) != date_key:
        updated["orders_today"] = 0
        updated["turnover_today"] = 0.0
    if str(updated.get("current_iso_week", "")) != week_key:
        updated["turnover_week"] = 0.0
    if str(updated.get("current_month", "")) != month_key:
        updated["turnover_month"] = 0.0

    updated["current_date"] = date_key
    updated["current_iso_week"] = week_key
    updated["current_month"] = month_key
    return updated


def compute_turnover_budget_remaining(state: dict[str, object], params: dict[str, object]) -> dict[str, float]:
    daily_limit = float(params.get("max_daily_turnover", 0.0))
    weekly_limit = float(params.get("max_weekly_turnover", 0.0))
    monthly_limit = float(params.get("max_monthly_turnover", 0.0))
    daily_remaining = max(daily_limit - float(state.get("turnover_today", 0.0)), 0.0)
    weekly_remaining = max(weekly_limit - float(state.get("turnover_week", 0.0)), 0.0)
    monthly_remaining = max(monthly_limit - float(state.get("turnover_month", 0.0)), 0.0)
    return {
        "daily": daily_remaining,
        "weekly": weekly_remaining,
        "monthly": monthly_remaining,
        "binding_remaining": min(daily_remaining, weekly_remaining, monthly_remaining),
    }


def compute_order_signature(final_orders: pd.DataFrame) -> str:
    if final_orders.empty:
        return "no_orders"
    actionable = final_orders.loc[
        final_orders["side"].astype(str).isin(["BUY", "SELL"])
        & final_orders["order_shares"].astype(float).abs().gt(1e-9)
    ].copy()
    if actionable.empty:
        return "no_orders"
    normalized = actionable.loc[
        :,
        ["ticker", "side", "current_shares", "target_shares", "order_shares"],
    ].copy()
    normalized["ticker"] = normalized["ticker"].astype(str)
    normalized["side"] = normalized["side"].astype(str)
    for column in ["current_shares", "target_shares", "order_shares"]:
        normalized[column] = normalized[column].astype(float).round(8)
    normalized = normalized.sort_values(["ticker", "side", "order_shares"]).reset_index(drop=True)
    return normalized.to_json(orient="records")


def compute_decision_id(signal_date: object, selected_candidate: str, order_signature: str) -> str:
    signal_text = str(pd.Timestamp(signal_date).date()) if signal_date is not None else "na"
    return f"{signal_text}|{selected_candidate}|{order_signature}"


def update_state_after_execution(
    state: dict[str, object],
    *,
    executed_orders: int,
    turnover: float,
    timestamp: str | None,
    decision_id: str,
    order_signature: str,
    selected_candidate: str,
    trade_now_score: float,
    execution_status: str,
    execution_mode: str,
) -> dict[str, object]:
    updated = dict(state)
    updated["last_decision_id"] = decision_id
    updated["last_selected_candidate"] = selected_candidate
    updated["last_trade_now_score"] = float(trade_now_score)
    updated["last_execution_status"] = str(execution_status)
    if execution_mode != "order_preview_only" and executed_orders > 0:
        updated["orders_today"] = int(updated.get("orders_today", 0)) + int(executed_orders)
        updated["turnover_today"] = float(updated.get("turnover_today", 0.0)) + float(turnover)
        updated["turnover_week"] = float(updated.get("turnover_week", 0.0)) + float(turnover)
        updated["turnover_month"] = float(updated.get("turnover_month", 0.0)) + float(turnover)
        updated["last_order_timestamp"] = timestamp
        updated["last_order_signature"] = order_signature
    return updated


def _write_csv(path: Path, frame: pd.DataFrame, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        frame.to_csv(temp_path, **kwargs)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _manual_order_share(value: float) -> int | float:
    rounded = round(float(value))
    if abs(float(value) - rounded) <= 1e-9:
        return int(rounded)
    return float(value)


def _annotate_final_daily_preview(
    *,
    preview_df: pd.DataFrame,
    cash_before_orders: float,
    cash_after_orders: float,
    preview_only: bool,
    preview_only_reason: str = "",
) -> pd.DataFrame:
    preview = preview_df.copy()
    current_shares = pd.to_numeric(
        preview.get("current_shares", pd.Series(0.0, index=preview.index)),
        errors="coerce",
    ).fillna(0.0)
    target_shares = pd.to_numeric(
        preview.get("target_shares", pd.Series(0.0, index=preview.index)),
        errors="coerce",
    ).fillna(0.0)
    absolute_delta_shares = (target_shares - current_shares).abs()
    shares_delta = target_shares - current_shares
    existing_reason = (
        preview.get("reason", pd.Series("", index=preview.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    too_small_or_no_change = existing_reason.eq("too_small_or_no_change") | absolute_delta_shares.le(1e-9)
    inferred_action = pd.Series("HOLD", index=preview.index, dtype=object)
    inferred_action = inferred_action.where(~shares_delta.gt(1e-9), "BUY")
    inferred_action = inferred_action.where(~shares_delta.lt(-1e-9), "SELL")
    side = inferred_action.where(~too_small_or_no_change, "HOLD").astype(str).str.upper()
    actionable = side.isin(["BUY", "SELL"])

    preview["current_shares"] = current_shares
    preview["target_shares"] = target_shares
    preview["shares_delta"] = shares_delta
    preview["order_shares"] = absolute_delta_shares
    preview["estimated_shares"] = absolute_delta_shares
    preview["action"] = side
    preview["estimated_price"] = pd.to_numeric(
        preview.get("latest_price", pd.Series(0.0, index=preview.index)),
        errors="coerce",
    ).fillna(0.0)
    signed_order_value = pd.to_numeric(
        preview.get("order_value", preview.get("estimated_order_value", pd.Series(0.0, index=preview.index))),
        errors="coerce",
    ).fillna(0.0)
    preview["order_value"] = signed_order_value
    preview["estimated_order_value"] = signed_order_value.abs()
    preview["cash_before_orders"] = float(cash_before_orders)
    preview["cash_after_orders"] = float(cash_after_orders)
    preview["preview_only"] = bool(preview_only)

    existing_block_reason = (
        preview.get("execution_block_reason", pd.Series("", index=preview.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    force_block_all_rows = bool(preview_only_reason and preview_only_reason != "dry_run_preview_only")
    if force_block_all_rows:
        existing_block_reason = existing_block_reason.where(existing_block_reason.ne(""), preview_only_reason)
    elif preview_only_reason:
        existing_block_reason = existing_block_reason.where(existing_block_reason.ne("") | ~actionable, preview_only_reason)
    preview["execution_block_reason"] = existing_block_reason

    not_executable = preview.get("not_executable", pd.Series(False, index=preview.index)).fillna(False).astype(bool)
    if force_block_all_rows:
        preview["not_executable"] = True
    else:
        preview["not_executable"] = not_executable

    preview["side"] = side
    preview["reason"] = existing_reason.where(existing_reason.ne("") | ~too_small_or_no_change, "too_small_or_no_change")

    return mark_daily_simulator_preview(preview)


def _summarize_delta_order_preview(
    *,
    order_preview: pd.DataFrame,
    cash_before_orders: float,
    cash_after_orders: float,
    nav: float,
) -> dict[str, object]:
    preview = order_preview.copy()
    if preview.empty:
        return {
            "order_count": 0,
            "manual_eligible_order_count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
            "estimated_buy_value": 0.0,
            "estimated_sell_value": 0.0,
            "cash_before_orders": float(cash_before_orders),
            "cash_after_orders": float(cash_after_orders),
            "total_simulator_fees_usd": 0.0,
            "modeled_transaction_costs_usd": 0.0,
            "negative_cash_check": True,
            "leverage_check": True,
            "short_check": True,
            "manual_orders_usable": False,
            "preview_only": bool(preview.get("preview_only", pd.Series([True])).iloc[0]) if not preview.empty else True,
            "manual_eligible_mask": pd.Series(False, index=preview.index, dtype=bool),
        }

    current_shares = pd.to_numeric(preview.get("current_shares", pd.Series(0.0, index=preview.index)), errors="coerce").fillna(0.0)
    target_shares = pd.to_numeric(preview.get("target_shares", pd.Series(0.0, index=preview.index)), errors="coerce").fillna(0.0)
    order_shares = pd.to_numeric(preview.get("order_shares", pd.Series(0.0, index=preview.index)), errors="coerce").fillna(0.0).abs()
    action = preview.get("action", preview.get("side", pd.Series("HOLD", index=preview.index))).fillna("HOLD").astype(str).str.upper()
    estimated_order_value = pd.to_numeric(
        preview.get("estimated_order_value", preview.get("order_value", pd.Series(0.0, index=preview.index))),
        errors="coerce",
    ).fillna(0.0).abs()
    modeled_costs = pd.to_numeric(preview.get("estimated_total_order_cost", pd.Series(0.0, index=preview.index)), errors="coerce").fillna(0.0)
    simulator_fees = pd.to_numeric(preview.get("simulator_fee_usd", pd.Series(0.0, index=preview.index)), errors="coerce").fillna(0.0)
    not_executable = preview.get("not_executable", pd.Series(False, index=preview.index)).fillna(False).astype(bool)
    execution_block_reason = preview.get("execution_block_reason", pd.Series("", index=preview.index)).fillna("").astype(str).str.strip()
    reason = preview.get("reason", pd.Series("", index=preview.index)).fillna("").astype(str).str.strip()
    preview_only = preview.get("preview_only", pd.Series(False, index=preview.index)).fillna(False).astype(bool)

    expected_order_shares = (target_shares - current_shares).abs()
    direction_buy = target_shares > current_shares + 1e-9
    direction_sell = target_shares < current_shares - 1e-9
    direction_hold = ~(direction_buy | direction_sell)
    micro_or_none = reason.eq("too_small_or_no_change") | order_shares.le(1e-9)
    hold_mask = action.eq("HOLD") | direction_hold | micro_or_none
    buy_mask = action.eq("BUY") & direction_buy & ~micro_or_none
    sell_mask = action.eq("SELL") & direction_sell & ~micro_or_none
    actionable_mask = (buy_mask | sell_mask) & order_shares.gt(1e-9)
    allowed_block_reason = execution_block_reason.isin(["", "dry_run_preview_only"])
    manual_eligible_mask = actionable_mask & ~not_executable & allowed_block_reason

    sequencing_frame = pd.DataFrame(
        {
            "action": action,
            "estimated_order_value": estimated_order_value,
            "modeled_costs": modeled_costs,
        },
        index=preview.index,
    ).loc[actionable_mask].copy()
    sequencing_frame["__side_order__"] = sequencing_frame["action"].map({"SELL": 0, "BUY": 1}).fillna(2)
    sequencing_frame = sequencing_frame.sort_values(["__side_order__"], kind="stable")

    cash_running = float(cash_before_orders)
    min_cash_seen = cash_running
    for row in sequencing_frame.itertuples(index=False):
        if str(row.action) == "SELL":
            cash_running += float(row.estimated_order_value) - float(row.modeled_costs)
        elif str(row.action) == "BUY":
            cash_running -= float(row.estimated_order_value) + float(row.modeled_costs)
        min_cash_seen = min(min_cash_seen, cash_running)

    negative_cash_check = bool(min_cash_seen >= -1e-9 and cash_running >= -1e-9 and float(cash_after_orders) >= -1e-9)
    target_market_value = float(
        pd.to_numeric(preview.get("target_value", pd.Series(0.0, index=preview.index)), errors="coerce").fillna(0.0).sum()
    )
    leverage_check = bool(target_market_value <= float(nav) + 1e-6 and cash_running >= -1e-9)
    short_check = bool((target_shares >= -1e-9).all() and (~sell_mask | (order_shares <= current_shares + 1e-9)).all())

    return {
        "order_count": int(actionable_mask.sum()),
        "manual_eligible_order_count": int(manual_eligible_mask.sum()),
        "buy_count": int(buy_mask.sum()),
        "sell_count": int(sell_mask.sum()),
        "hold_count": int(hold_mask.sum()),
        "estimated_buy_value": float(estimated_order_value.loc[buy_mask].sum()),
        "estimated_sell_value": float(estimated_order_value.loc[sell_mask].sum()),
        "cash_before_orders": float(cash_before_orders),
        "cash_after_orders": float(cash_running if actionable_mask.any() else cash_after_orders),
        "total_simulator_fees_usd": float(simulator_fees.loc[actionable_mask].sum()),
        "modeled_transaction_costs_usd": float(modeled_costs.loc[actionable_mask].sum()),
        "negative_cash_check": bool(negative_cash_check),
        "leverage_check": bool(leverage_check),
        "short_check": bool(short_check),
        "manual_orders_usable": bool(manual_eligible_mask.any() and negative_cash_check and leverage_check and short_check),
        "preview_only": bool(preview_only.any()),
        "manual_eligible_mask": manual_eligible_mask,
        "order_shares_match_abs_delta": bool((order_shares.sub(expected_order_shares).abs() <= 1e-9).all()),
        "buy_direction_ok": bool((~buy_mask | direction_buy).all()),
        "sell_direction_ok": bool((~sell_mask | direction_sell).all()),
    }


def _build_manual_simulator_order_outputs(
    *,
    order_preview: pd.DataFrame,
    latest_price_date: object,
    rest_cash_usd: float,
    cash_before_orders: float,
    cash_after_orders: float,
) -> tuple[pd.DataFrame, str]:
    preview = order_preview.copy()
    summary = _summarize_delta_order_preview(
        order_preview=preview,
        cash_before_orders=float(cash_before_orders),
        cash_after_orders=float(cash_after_orders),
        nav=float(
            pd.to_numeric(preview.get("target_value", pd.Series(0.0, index=preview.index)), errors="coerce").fillna(0.0).sum()
            + float(cash_after_orders)
        ),
    )
    actionable = preview.loc[summary["manual_eligible_mask"]].copy()

    manual_rows: list[dict[str, object]] = []
    txt_lines = [
        "Manual simulator entry only",
        f"latest_price_date: {latest_price_date}",
        f"rest_cash_usd: {float(rest_cash_usd):.2f}",
        f"cash_before_orders: {float(cash_before_orders):.2f}",
        f"cash_after_orders: {float(summary['cash_after_orders']):.2f}",
        f"estimated_sell_value: {float(summary['estimated_sell_value']):.2f}",
        f"estimated_buy_value: {float(summary['estimated_buy_value']):.2f}",
        f"total_simulator_fees_usd: {float(summary['total_simulator_fees_usd']):.2f}",
        f"modeled_transaction_costs_usd: {float(summary['modeled_transaction_costs_usd']):.2f}",
        f"buy_count: {int(summary['buy_count'])}",
        f"sell_count: {int(summary['sell_count'])}",
        f"hold_count: {int(summary['hold_count'])}",
        f"order_count: {int(summary['order_count'])}",
        f"preview_only: {bool(summary['preview_only'])}",
        f"negative_cash_check: {bool(summary['negative_cash_check'])}",
        f"leverage_check: {bool(summary['leverage_check'])}",
        f"short_check: {bool(summary['short_check'])}",
        f"manual_orders_usable: {bool(summary['manual_orders_usable'])}",
        "price_note: latest adjusted close is a proxy, not an executable quote",
        "",
    ]

    if actionable.empty:
        txt_lines.append("Preview only. No BUY/SELL orders.")
        manual_frame = pd.DataFrame(
            columns=[
                "ticker",
                "action",
                "current_shares",
                "target_shares",
                "order_shares",
                "shares",
                "estimated_price",
                "estimated_order_value",
                "simulator_fee_usd",
                "modeled_transaction_costs_usd",
                "preview_only",
                "not_executable",
                "execution_block_reason",
                "cash_before_orders",
                "cash_after_orders",
                "note",
            ]
        )
        return manual_frame, "\n".join(txt_lines) + "\n"

    for row in actionable.itertuples(index=False):
        action = str(row.side)
        shares = _manual_order_share(float(row.order_shares))
        latest_price_value = getattr(row, "estimated_price", None)
        if latest_price_value is None:
            latest_price_value = getattr(row, "latest_price", 0.0)
        latest_price = float(latest_price_value)
        estimated_order_value = abs(float(getattr(row, "estimated_order_value", getattr(row, "order_value", 0.0))))
        note_parts = [
            "Manual simulator entry only",
            f"latest_price_date={latest_price_date}",
            f"rest_cash_usd={float(rest_cash_usd):.2f}",
            f"cash_before_orders={float(cash_before_orders):.2f}",
            f"cash_after_orders={float(cash_after_orders):.2f}",
            "latest adjusted close proxy",
        ]
        reason = str(getattr(row, "execution_block_reason", "") or getattr(row, "reason", "") or "").strip()
        preview_only = bool(getattr(row, "preview_only", False))
        not_executable = bool(getattr(row, "not_executable", False))
        if preview_only or not_executable or reason:
            note_parts.append("preview_only")
        if reason:
            note_parts.append(f"reason={reason}")
        manual_rows.append(
            {
                "ticker": str(row.ticker),
                "action": action,
                "current_shares": float(getattr(row, "current_shares", 0.0)),
                "target_shares": float(getattr(row, "target_shares", 0.0)),
                "order_shares": float(getattr(row, "order_shares", 0.0)),
                "shares": shares,
                "estimated_price": latest_price,
                "estimated_order_value": estimated_order_value,
                "simulator_fee_usd": 0.0,
                "modeled_transaction_costs_usd": float(getattr(row, "estimated_total_order_cost", 0.0)),
                "preview_only": preview_only,
                "not_executable": not_executable,
                "execution_block_reason": reason,
                "cash_before_orders": float(cash_before_orders),
                "cash_after_orders": float(summary["cash_after_orders"]),
                "note": "; ".join(note_parts),
            }
        )
        txt_lines.append(f"{action} {shares} {row.ticker}")

    manual_frame = pd.DataFrame(
        manual_rows,
        columns=[
            "ticker",
            "action",
            "current_shares",
            "target_shares",
            "order_shares",
            "shares",
            "estimated_price",
            "estimated_order_value",
            "simulator_fee_usd",
            "modeled_transaction_costs_usd",
            "preview_only",
            "not_executable",
            "execution_block_reason",
            "cash_before_orders",
            "cash_after_orders",
            "note",
        ],
    )
    return manual_frame, "\n".join(txt_lines) + "\n"


def _format_top_weights(weights: pd.Series, top_n: int = 6) -> str:
    non_zero = weights.astype(float)
    non_zero = non_zero[non_zero > 1e-6].sort_values(ascending=False).head(top_n)
    if non_zero.empty:
        return "none"
    return ", ".join(f"{ticker} {weight:.2%}" for ticker, weight in non_zero.items())


def _ja_nein(value: object) -> str:
    return "Ja" if bool(value) else "Nein"


def _factor_driver_names(factor_forecast_df: pd.DataFrame, top_n: int = 5) -> list[str]:
    if not isinstance(factor_forecast_df, pd.DataFrame) or factor_forecast_df.empty or "factor" not in factor_forecast_df.columns:
        return []
    ranked = factor_forecast_df.copy()
    if "confidence" in ranked.columns:
        ranked = ranked.sort_values("confidence", ascending=False)
    elif "forecast" in ranked.columns:
        ranked = ranked.reindex(ranked["forecast"].abs().sort_values(ascending=False).index)
    return ranked["factor"].astype(str).head(top_n).tolist()


def _top_weight_change_lines(
    current_weights: pd.Series,
    target_weights: pd.Series,
    *,
    top_n: int = 4,
    min_delta: float = 0.0025,
) -> list[str]:
    current = current_weights.astype(float).copy()
    target = target_weights.astype(float).copy()
    index = pd.Index(sorted(set(current.index).union(target.index)), name="ticker")
    current = current.reindex(index).fillna(0.0)
    target = target.reindex(index).fillna(0.0)
    delta = (target - current).sort_values(ascending=False)
    lines: list[str] = []
    for ticker, delta_weight in delta.items():
        if float(delta_weight) <= min_delta:
            continue
        lines.append(
            f"{ticker} +{float(delta_weight):.2%} (target {float(target.loc[ticker]):.2%} vs current {float(current.loc[ticker]):.2%})"
        )
        if len(lines) >= top_n:
            break
    return lines


def _candidate_failure_reason(
    row: pd.Series,
    *,
    risk_premium_hurdle: float,
    p_hold_min: float,
    p_cash_min: float,
) -> str:
    validation_errors = str(row.get("validation_errors", "") or "").strip()
    if validation_errors and validation_errors.lower() != "none":
        return validation_errors
    if not bool(row.get("valid_constraints", True)):
        return "constraints failed after rounding"
    if float(row.get("delta_vs_cash", 0.0)) <= risk_premium_hurdle:
        return f"delta_vs_cash {float(row.get('delta_vs_cash', 0.0)):.6f} <= hurdle {risk_premium_hurdle:.6f}"
    if float(row.get("probability_beats_hold", 0.0)) < p_hold_min:
        return f"probability_beats_current {float(row.get('probability_beats_hold', 0.0)):.2%} < threshold {p_hold_min:.2%}"
    if float(row.get("probability_beats_cash", 0.0)) < p_cash_min:
        return f"probability_beats_cash {float(row.get('probability_beats_cash', 0.0)):.2%} < threshold {p_cash_min:.2%}"
    if float(row.get("delta_vs_hold", 0.0)) <= 0.0:
        return "no positive edge versus current portfolio after costs"
    return "did not clear all post-cost execution hurdles"


def _candidate_family_name(candidate_name: str) -> str:
    return str(candidate_name).split("::", 1)[0]


def _parse_violation_items(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _find_repaired_candidate_name(candidate_name: str, all_names: set[str]) -> str:
    family = _candidate_family_name(candidate_name)
    repair_map = {
        "MOMENTUM_TILT_SIMPLE": "MOMENTUM_TILT_REPAIRED",
        "CONDITIONAL_FACTOR_TARGET": "FACTOR_TARGET_REPAIRED",
        "HOLD_CURRENT": "CURRENT_COMPLIANCE_REPAIR",
        "HOLD": "CURRENT_COMPLIANCE_REPAIR",
    }
    repaired_family = repair_map.get(family)
    if not repaired_family:
        return ""
    if "::" in str(candidate_name):
        repaired_name = str(candidate_name).replace(family, repaired_family, 1)
        if repaired_name in all_names:
            return repaired_name
    matches = sorted(name for name in all_names if name.startswith(f"{repaired_family}::") or name == repaired_family)
    return matches[0] if matches else ""


def _build_constraint_pressure_reports(
    *,
    scores_frame: pd.DataFrame,
    candidate_map: dict[str, object],
    params: dict[str, object],
    selected_reason: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Write reports showing which attractive candidates are blocked by constraints."""

    required_columns = [
        "candidate",
        "candidate_family",
        "score",
        "net_score_after_order_costs",
        "delta_vs_hold",
        "delta_vs_current",
        "valid_constraints",
        "failed_constraints",
        "failed_constraint_type",
        "asset_or_group",
        "actual_weight",
        "limit",
        "excess",
        "repair_possible",
        "repaired_candidate_name",
        "score_after_repair",
        "score_lost_due_to_repair",
        "order_count",
        "turnover_vs_current",
        "selected",
        "selected_reason",
    ]
    if scores_frame.empty:
        empty = pd.DataFrame(columns=required_columns)
        _write_csv(output_dir / "constraint_pressure_report.csv", empty, index=False)
        _write_text(output_dir / "constraint_pressure_report.txt", "Constraint Pressure Report\n\nNo discrete candidates available.\n")
        return empty

    all_names = set(scores_frame["discrete_candidate"].astype(str).tolist())
    score_by_name = scores_frame.set_index(scores_frame["discrete_candidate"].astype(str))
    rows: list[dict[str, object]] = []
    for _, row in scores_frame.iterrows():
        candidate_name = str(row.get("discrete_candidate", ""))
        candidate = candidate_map.get(candidate_name)
        family = _candidate_family_name(candidate_name)
        repaired_name = _find_repaired_candidate_name(candidate_name, all_names)
        score_after_repair = (
            float(score_by_name.loc[repaired_name, "net_robust_score"])
            if repaired_name and repaired_name in score_by_name.index
            else pd.NA
        )
        score_lost_due_to_repair = (
            float(row.get("net_robust_score", 0.0)) - float(score_after_repair)
            if repaired_name and repaired_name in score_by_name.index
            else pd.NA
        )
        repair_possible = bool(getattr(candidate, "metadata", {}).get("repair_possible", False)) if candidate is not None else False
        if repaired_name:
            repair_possible = True
        asset_items = _parse_violation_items(row.get("asset_limit_violations", ""))
        group_items = _parse_violation_items(row.get("group_limit_violations", ""))
        violation_rows: list[tuple[str, dict[str, object]]] = []
        violation_rows.extend(("asset", item) for item in asset_items)
        violation_rows.extend(("group", item) for item in group_items)
        if not violation_rows and not bool(row.get("valid_constraints", True)):
            violation_rows.append(("other", {"asset_or_group": str(row.get("validation_errors", "unknown_constraint"))}))
        if not violation_rows:
            violation_rows.append(("", {}))
        for violation_type, item in violation_rows:
            asset_or_group = (
                item.get("ticker")
                or item.get("group")
                or item.get("asset_or_group")
                or ""
            )
            rows.append(
                {
                    "candidate": candidate_name,
                    "candidate_family": family,
                    "score": float(row.get("robust_score", row.get("net_robust_score", 0.0))),
                    "net_score_after_order_costs": float(row.get("net_robust_score", 0.0)),
                    "delta_vs_hold": float(row.get("delta_vs_hold", 0.0)),
                    "delta_vs_current": float(row.get("delta_vs_hold", 0.0)),
                    "valid_constraints": bool(row.get("valid_constraints", True)),
                    "failed_constraints": str(row.get("validation_errors", "")),
                    "failed_constraint_type": violation_type,
                    "asset_or_group": str(asset_or_group),
                    "actual_weight": item.get("actual_weight", pd.NA),
                    "limit": item.get("limit", pd.NA),
                    "excess": item.get("excess", pd.NA),
                    "repair_possible": bool(repair_possible),
                    "repaired_candidate_name": repaired_name,
                    "score_after_repair": score_after_repair,
                    "score_lost_due_to_repair": score_lost_due_to_repair,
                    "order_count": int(row.get("number_of_orders", 0)),
                    "turnover_vs_current": float(row.get("turnover_vs_current", row.get("turnover", 0.0))),
                    "selected": bool(row.get("selected", False)),
                    "selected_reason": selected_reason,
                }
            )
    pressure = pd.DataFrame(rows, columns=required_columns)
    _write_csv(output_dir / "constraint_pressure_report.csv", pressure, index=False)

    top_score = scores_frame.sort_values("net_robust_score", ascending=False).head(5)
    invalid = scores_frame.loc[scores_frame["valid_constraints"] != True].copy()  # noqa: E712
    invalid_top = invalid.sort_values("net_robust_score", ascending=False).head(5) if not invalid.empty else invalid
    asset_pressure = pressure.loc[pressure["failed_constraint_type"].eq("asset")].copy()
    group_pressure = pressure.loc[pressure["failed_constraint_type"].eq("group")].copy()
    better_than_hold = pressure.loc[
        (pressure["candidate"].astype(str) != "HOLD_CURRENT")
        & (pd.to_numeric(pressure["delta_vs_hold"], errors="coerce").fillna(0.0) > 0.0)
        & (~pressure["valid_constraints"].astype(bool))
    ].copy()
    hold_invalid = bool(
        not scores_frame.loc[
            scores_frame["discrete_candidate"].astype(str).eq("HOLD_CURRENT"),
            "valid_constraints",
        ].fillna(False).astype(bool).all()
    ) if (scores_frame["discrete_candidate"].astype(str).eq("HOLD_CURRENT")).any() else False
    lines = [
        "Constraint Pressure Report",
        "",
        "Top 5 Kandidaten nach Net-Score:",
        *[
            f"- {item.discrete_candidate}: net={float(item.net_robust_score):.6f}, valid_constraints={bool(item.valid_constraints)}, failed={str(getattr(item, 'validation_errors', '') or 'none')}"
            for item in top_score.itertuples(index=False)
        ],
        "",
        "Top 5 invalid Kandidaten:",
    ]
    if invalid_top.empty:
        lines.append("- none")
    else:
        lines.extend(
            f"- {item.discrete_candidate}: net={float(item.net_robust_score):.6f}, failed={str(getattr(item, 'validation_errors', '') or 'unknown')}"
            for item in invalid_top.itertuples(index=False)
        )
    lines.extend(["", "Wichtigste bindende Asset-Caps:"])
    if asset_pressure.empty:
        lines.append("- none")
    else:
        lines.extend(
            f"- {row.asset_or_group}: actual={float(row.actual_weight):.2%}, limit={float(row.limit):.2%}, excess={float(row.excess):.2%}, candidate={row.candidate}"
            for row in asset_pressure.sort_values("excess", ascending=False).head(8).itertuples(index=False)
        )
    lines.extend(["", "Wichtigste bindende Gruppen-Caps:"])
    if group_pressure.empty:
        lines.append("- none")
    else:
        lines.extend(
            f"- {row.asset_or_group}: actual={float(row.actual_weight):.2%}, limit={float(row.limit):.2%}, excess={float(row.excess):.2%}, candidate={row.candidate}"
            for row in group_pressure.sort_values("excess", ascending=False).head(8).itertuples(index=False)
        )
    lines.extend(["", "Kandidaten besser als HOLD, aber constraint-invalid:"])
    if better_than_hold.empty:
        lines.append("- none")
    else:
        lines.extend(
            f"- {row.candidate}: delta_vs_hold={float(row.delta_vs_hold):.6f}, failed={row.failed_constraint_type}:{row.asset_or_group}"
            for row in better_than_hold.head(10).itertuples(index=False)
        )
    lines.extend(
        [
            "",
            f"HOLD_CURRENT constraint-invalid: {hold_invalid}",
            f"selected_reason: {selected_reason}",
        ]
    )
    _write_text(output_dir / "constraint_pressure_report.txt", "\n".join(lines) + "\n")
    return pressure


def _truthy(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _active_preview_settings(
    params: dict[str, object],
    *,
    safe_model_uncertainty_buffer: float,
    trade_now_hurdle: float | None = None,
    execution_buffer: float | None = None,
    model_uncertainty_multiplier: float | None = None,
) -> dict[str, object]:
    """Build the non-executable Active Preview threshold set."""

    multiplier = (
        float(model_uncertainty_multiplier)
        if model_uncertainty_multiplier is not None
        else float(params.get("active_preview_model_uncertainty_multiplier", 0.50))
    )
    active_execution_buffer = (
        float(execution_buffer)
        if execution_buffer is not None
        else float(params.get("active_preview_execution_buffer", 0.00035))
    )
    active_trade_now_hurdle = (
        float(trade_now_hurdle)
        if trade_now_hurdle is not None
        else float(params.get("active_preview_trade_now_hurdle", 0.00075))
    )
    return {
        "enabled": bool(params.get("enable_active_preview", True)),
        "safe_trade_now_hurdle": 0.0025,
        "safe_execution_buffer": 0.0010,
        "safe_model_uncertainty_buffer": float(safe_model_uncertainty_buffer),
        "trade_now_hurdle": active_trade_now_hurdle,
        "execution_buffer": active_execution_buffer,
        "model_uncertainty_multiplier": multiplier,
        "model_uncertainty_buffer": float(safe_model_uncertainty_buffer) * multiplier,
        "delta_vs_cash_min": float(params.get("active_preview_delta_vs_cash_min", 0.00025)),
        "p_current_min": float(params.get("active_preview_p_current_min", 0.52)),
        "p_cash_min": float(params.get("active_preview_p_cash_min", 0.51)),
        "max_turnover": float(params.get("active_preview_max_turnover", 0.20)),
        "min_order_value_usd": float(params.get("active_preview_min_order_value_usd", params.get("min_order_value_usd", 10.0))),
        "configured_allow_execution": bool(params.get("active_preview_allow_execution", False)),
        "active_preview_executable": False,
        "active_preview_order_submission_allowed": False,
    }


def _constraint_text_from_row(row: pd.Series) -> str:
    parts: list[str] = []
    for key, prefix in (("asset_limit_violations", "asset"), ("group_limit_violations", "group")):
        for item in _parse_violation_items(row.get(key, "")):
            name = str(item.get("ticker") or item.get("group") or item.get("asset_or_group") or "unknown")
            actual = _safe_float(item.get("actual_weight"), 0.0)
            limit = _safe_float(item.get("limit"), 0.0)
            excess = _safe_float(item.get("excess"), max(actual - limit, 0.0))
            parts.append(f"{prefix}:{name} actual={actual:.4f} limit={limit:.4f} excess={excess:.4f}")
    validation_errors = str(row.get("validation_errors", "") or "").strip()
    if validation_errors and validation_errors.lower() != "none":
        parts.append(validation_errors)
    return " | ".join(parts)


def _active_preview_row_metrics(
    row: pd.Series,
    *,
    current_portfolio_score: float,
    settings: dict[str, object],
) -> dict[str, float]:
    net_score = _safe_float(row.get("net_robust_score"), 0.0)
    delta_vs_current = net_score - float(current_portfolio_score)
    execution_buffer = float(settings["execution_buffer"])
    model_buffer = float(settings["model_uncertainty_buffer"])
    trade_now_edge = delta_vs_current - execution_buffer - model_buffer
    return {
        "net_score": net_score,
        "delta_vs_current": delta_vs_current,
        "trade_now_edge": trade_now_edge,
        "delta_vs_cash": _safe_float(row.get("delta_vs_cash"), 0.0),
        "probability_beats_current": _safe_float(row.get("probability_beats_hold"), 0.0),
        "probability_beats_cash": _safe_float(row.get("probability_beats_cash"), 0.0),
        "turnover": _safe_float(row.get("turnover_vs_current", row.get("turnover")), 0.0),
        "order_count": _safe_float(row.get("number_of_orders"), 0.0),
    }


def _select_active_preview_candidate(
    *,
    scores_frame: pd.DataFrame,
    current_portfolio_score: float,
    params: dict[str, object],
    safe_model_uncertainty_buffer: float,
    trade_now_hurdle: float | None = None,
    execution_buffer: float | None = None,
    model_uncertainty_multiplier: float | None = None,
) -> dict[str, object]:
    """Select a non-executable lower-hurdle non-HOLD candidate, if any clears."""

    settings = _active_preview_settings(
        params,
        safe_model_uncertainty_buffer=float(safe_model_uncertainty_buffer),
        trade_now_hurdle=trade_now_hurdle,
        execution_buffer=execution_buffer,
        model_uncertainty_multiplier=model_uncertainty_multiplier,
    )
    base_summary: dict[str, object] = {
        "enabled": bool(settings["enabled"]),
        "active_preview_action": "HOLD",
        "active_preview_candidate": "HOLD_CURRENT",
        "active_preview_best_candidate": "none",
        "active_preview_best_candidate_valid": False,
        "active_preview_trade_now_edge": 0.0,
        "active_preview_hurdle": float(settings["trade_now_hurdle"]),
        "active_preview_execution_buffer": float(settings["execution_buffer"]),
        "active_preview_model_uncertainty_buffer": float(settings["model_uncertainty_buffer"]),
        "active_preview_model_uncertainty_multiplier": float(settings["model_uncertainty_multiplier"]),
        "active_preview_delta_vs_cash_min": float(settings["delta_vs_cash_min"]),
        "active_preview_p_current_min": float(settings["p_current_min"]),
        "active_preview_p_cash_min": float(settings["p_cash_min"]),
        "active_preview_max_turnover": float(settings["max_turnover"]),
        "active_preview_min_order_value_usd": float(settings["min_order_value_usd"]),
        "active_preview_reason": "active_preview_disabled" if not bool(settings["enabled"]) else "no_candidate_evaluated",
        "active_preview_blocking_reason": "active_preview_disabled" if not bool(settings["enabled"]) else "no_candidate_evaluated",
        "active_preview_executable": False,
        "active_preview_order_submission_allowed": False,
        "configured_active_preview_allow_execution": bool(settings["configured_allow_execution"]),
        "active_preview_order_count": 0,
        "active_preview_buy_count": 0,
        "active_preview_sell_count": 0,
        "active_preview_turnover": 0.0,
        "active_preview_delta_vs_current": 0.0,
        "active_preview_delta_vs_cash": 0.0,
        "active_preview_probability_beats_current": 0.0,
        "active_preview_probability_beats_cash": 0.0,
        "active_preview_binding_constraints": "none",
        "active_preview_safe_mode_trade_now_hurdle": float(settings["safe_trade_now_hurdle"]),
        "active_preview_safe_mode_execution_buffer": float(settings["safe_execution_buffer"]),
        "active_preview_safe_mode_model_uncertainty_buffer": float(settings["safe_model_uncertainty_buffer"]),
    }
    if not bool(settings["enabled"]) or not isinstance(scores_frame, pd.DataFrame) or scores_frame.empty:
        return base_summary

    non_hold = scores_frame.loc[scores_frame["discrete_candidate"].astype(str) != "HOLD_CURRENT"].copy()
    if non_hold.empty:
        base_summary["active_preview_reason"] = "no_non_hold_candidate_available"
        base_summary["active_preview_blocking_reason"] = "no_non_hold_candidate_available"
        return base_summary

    non_hold = non_hold.sort_values("net_robust_score", ascending=False, kind="mergesort").reset_index(drop=True)
    best_row = non_hold.iloc[0]
    best_metrics = _active_preview_row_metrics(best_row, current_portfolio_score=current_portfolio_score, settings=settings)
    base_summary.update(
        {
            "active_preview_best_candidate": str(best_row.get("discrete_candidate", "none")),
            "active_preview_best_candidate_valid": _truthy(best_row.get("valid_constraints", False)),
            "active_preview_trade_now_edge": float(best_metrics["trade_now_edge"]),
            "active_preview_delta_vs_current": float(best_metrics["delta_vs_current"]),
            "active_preview_delta_vs_cash": float(best_metrics["delta_vs_cash"]),
            "active_preview_probability_beats_current": float(best_metrics["probability_beats_current"]),
            "active_preview_probability_beats_cash": float(best_metrics["probability_beats_cash"]),
            "active_preview_turnover": float(best_metrics["turnover"]),
            "active_preview_order_count": int(best_metrics["order_count"]),
            "active_preview_binding_constraints": _constraint_text_from_row(best_row) or "none",
        }
    )

    first_failure = "no_valid_non_hold_candidate"
    for _, row in non_hold.iterrows():
        candidate_name = str(row.get("discrete_candidate", ""))
        metrics = _active_preview_row_metrics(row, current_portfolio_score=current_portfolio_score, settings=settings)
        valid_constraints = _truthy(row.get("valid_constraints", False))
        failures: list[str] = []
        if not valid_constraints:
            failures.append("failed_constraints")
        else:
            if metrics["delta_vs_current"] <= 0.0:
                failures.append("failed_no_positive_edge_vs_current")
            if metrics["delta_vs_cash"] <= float(settings["delta_vs_cash_min"]):
                failures.append("failed_delta_vs_cash_min")
            if metrics["probability_beats_current"] < float(settings["p_current_min"]):
                failures.append("failed_probability_beats_current")
            if metrics["probability_beats_cash"] < float(settings["p_cash_min"]):
                failures.append("failed_probability_beats_cash")
            if metrics["turnover"] > float(settings["max_turnover"]):
                failures.append("failed_active_preview_max_turnover")
            if metrics["trade_now_edge"] <= float(settings["trade_now_hurdle"]):
                failures.append("failed_active_preview_trade_now_hurdle")
            if int(metrics["order_count"]) <= 0:
                failures.append("failed_no_delta_orders")
        failure = "|".join(failures)

        if failure:
            if first_failure == "no_valid_non_hold_candidate" and valid_constraints:
                first_failure = failure
            continue

        base_summary.update(
            {
                "active_preview_action": "BUY_SELL_PREVIEW",
                "active_preview_candidate": candidate_name,
                "active_preview_best_candidate": candidate_name,
                "active_preview_best_candidate_valid": True,
                "active_preview_trade_now_edge": float(metrics["trade_now_edge"]),
                "active_preview_delta_vs_current": float(metrics["delta_vs_current"]),
                "active_preview_delta_vs_cash": float(metrics["delta_vs_cash"]),
                "active_preview_probability_beats_current": float(metrics["probability_beats_current"]),
                "active_preview_probability_beats_cash": float(metrics["probability_beats_cash"]),
                "active_preview_turnover": float(metrics["turnover"]),
                "active_preview_order_count": int(metrics["order_count"]),
                "active_preview_reason": "active_preview_gate_passed_never_executable",
                "active_preview_blocking_reason": "active_preview_never_executes",
                "active_preview_binding_constraints": "none",
            }
        )
        return base_summary

    base_summary["active_preview_reason"] = first_failure
    base_summary["active_preview_blocking_reason"] = first_failure
    return base_summary


def _active_preview_order_columns() -> list[str]:
    return [
        "preview_context",
        "ticker",
        "action",
        "side",
        "current_shares",
        "target_shares",
        "order_shares",
        "shares_delta",
        "estimated_price",
        "estimated_order_value",
        "estimated_total_order_cost",
        "preview_only",
        "executable",
        "not_executable",
        "execution_block_reason",
        "note",
    ]


def _build_active_preview_files(
    *,
    active_preview_summary: dict[str, object],
    candidate_map: dict[str, object],
    current_state,
    latest_prices: pd.Series,
    active_tickers: list[str],
    params: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    """Write strictly non-executable Active Preview order/allocation/gate files."""

    selected_name = str(active_preview_summary.get("active_preview_candidate", "HOLD_CURRENT"))
    selected_candidate = candidate_map.get(selected_name)
    if selected_candidate is None or selected_name == "HOLD_CURRENT":
        target_shares = current_state.current_shares.reindex(active_tickers).fillna(0.0)
        target_weights = current_state.current_weights_actual.reindex(active_tickers).fillna(0.0)
        cash_left = float(current_state.current_cash)
        preview = pd.DataFrame(columns=_active_preview_order_columns())
        cost_summary = {
            "cash_before_orders": float(current_state.current_cash),
            "cash_after_orders": float(current_state.current_cash),
            "total_estimated_transaction_cost": 0.0,
        }
    else:
        target_shares = selected_candidate.shares.reindex(active_tickers).fillna(0.0)
        target_weights = selected_candidate.weights_actual.reindex(active_tickers).fillna(0.0)
        cash_left = float(selected_candidate.cash_left)
        raw_preview = build_discrete_order_preview(
            current_shares=current_state.current_shares.reindex(active_tickers).fillna(0.0),
            target_shares=target_shares,
            latest_prices=latest_prices.reindex(active_tickers).fillna(0.0),
            nav=float(current_state.nav),
            min_order_value=float(active_preview_summary.get("active_preview_min_order_value_usd", params.get("min_order_value_usd", 10.0))),
            not_executable=True,
            reason="active_preview_never_executes",
        )
        raw_preview, cost_summary = estimate_order_list_costs(
            order_preview_df=raw_preview,
            latest_prices=latest_prices.reindex(active_tickers).fillna(0.0),
            config={
                **params,
                "nav": float(current_state.nav),
                "current_cash": float(current_state.current_cash),
            },
        )
        preview = _annotate_final_daily_preview(
            preview_df=raw_preview,
            cash_before_orders=float(cost_summary["cash_before_orders"]),
            cash_after_orders=float(cost_summary["cash_after_orders"]),
            preview_only=True,
            preview_only_reason="active_preview_never_executes",
        )
        preview["preview_context"] = "active_preview"
        preview["executable"] = False
        preview["not_executable"] = True
        preview["execution_block_reason"] = "active_preview_never_executes"
        preview["note"] = "For analysis only. Do not submit automatically."
        action_mask = preview["action"].astype(str).isin(["BUY", "SELL"]) & pd.to_numeric(preview["order_shares"], errors="coerce").fillna(0.0).gt(1e-9)
        preview = preview.loc[action_mask].copy()

    if preview.empty:
        preview = pd.DataFrame(columns=_active_preview_order_columns())
    else:
        for column in _active_preview_order_columns():
            if column not in preview.columns:
                preview[column] = "" if column in {"preview_context", "ticker", "action", "side", "execution_block_reason", "note"} else False
        preview = preview[_active_preview_order_columns()]

    _write_csv(output_dir / "active_preview_orders.csv", preview, index=False)

    allocation = pd.DataFrame(
        {
            "ticker": active_tickers,
            "current_shares": current_state.current_shares.reindex(active_tickers).fillna(0.0).values,
            "target_shares": target_shares.reindex(active_tickers).fillna(0.0).values,
            "current_weight": current_state.current_weights_actual.reindex(active_tickers).fillna(0.0).values,
            "target_weight": target_weights.reindex(active_tickers).fillna(0.0).values,
            "delta_weight": (
                target_weights.reindex(active_tickers).fillna(0.0)
                - current_state.current_weights_actual.reindex(active_tickers).fillna(0.0)
            ).values,
            "latest_price": latest_prices.reindex(active_tickers).fillna(0.0).values,
            "active_preview_candidate": selected_name,
            "preview_context": "active_preview",
            "executable": False,
        }
    )
    _write_csv(output_dir / "active_preview_allocation.csv", allocation, index=False)

    order_summary = _summarize_delta_order_preview(
        order_preview=preview,
        cash_before_orders=float(cost_summary["cash_before_orders"]),
        cash_after_orders=float(cost_summary["cash_after_orders"]),
        nav=float(current_state.nav),
    )
    active_preview_summary.update(
        {
            "active_preview_order_count": int(order_summary["order_count"]),
            "active_preview_buy_count": int(order_summary["buy_count"]),
            "active_preview_sell_count": int(order_summary["sell_count"]),
            "active_preview_total_order_cost": float(cost_summary.get("total_estimated_transaction_cost", 0.0)),
            "active_preview_cash_before_orders": float(cost_summary["cash_before_orders"]),
            "active_preview_cash_after_orders": float(cost_summary["cash_after_orders"]),
            "active_preview_cash_left": cash_left,
            "active_preview_executable": False,
            "active_preview_order_submission_allowed": False,
        }
    )

    gate_frame = pd.DataFrame([active_preview_summary])
    _write_csv(output_dir / "active_preview_gate_report.csv", gate_frame, index=False)

    if selected_name != "HOLD_CURRENT" and int(order_summary["order_count"]) > 0:
        verdict = "Safe Mode remains HOLD, but Active Preview finds a lower-hurdle rebalance candidate."
    else:
        verdict = "Even Active Preview did not find a valid non-HOLD candidate."
    lines = [
        "Active Preview Decision Report",
        "",
        "Active Preview is analysis only and never executable.",
        f"active_preview_executable: {bool(active_preview_summary['active_preview_executable'])}",
        f"active_preview_order_submission_allowed: {bool(active_preview_summary['active_preview_order_submission_allowed'])}",
        f"active_preview_action: {active_preview_summary.get('active_preview_action', 'HOLD')}",
        f"active_preview_candidate: {active_preview_summary.get('active_preview_candidate', 'HOLD_CURRENT')}",
        f"active_preview_best_candidate: {active_preview_summary.get('active_preview_best_candidate', 'none')}",
        f"active_preview_trade_now_edge: {_safe_float(active_preview_summary.get('active_preview_trade_now_edge')):.6f}",
        f"active_preview_hurdle: {_safe_float(active_preview_summary.get('active_preview_hurdle')):.6f}",
        f"active_preview_order_count: {int(active_preview_summary.get('active_preview_order_count', 0))}",
        f"active_preview_buy_count: {int(active_preview_summary.get('active_preview_buy_count', 0))}",
        f"active_preview_sell_count: {int(active_preview_summary.get('active_preview_sell_count', 0))}",
        f"active_preview_turnover: {_safe_float(active_preview_summary.get('active_preview_turnover')):.6f}",
        f"active_preview_reason: {active_preview_summary.get('active_preview_reason', 'unknown')}",
        f"active_preview_blocking_reason: {active_preview_summary.get('active_preview_blocking_reason', 'unknown')}",
        f"configured_active_preview_allow_execution: {bool(active_preview_summary.get('configured_active_preview_allow_execution', False))}",
        "",
        verdict,
        "",
        "Files:",
        "- outputs/active_preview_orders.csv",
        "- outputs/active_preview_allocation.csv",
        "- outputs/active_preview_gate_report.csv",
        "",
        "Order note: For analysis only. Do not submit automatically.",
    ]
    _write_text(output_dir / "active_preview_decision_report.txt", "\n".join(lines) + "\n")
    return active_preview_summary


def _write_rebalance_sensitivity_matrix(
    *,
    scores_frame: pd.DataFrame,
    candidate_map: dict[str, object],
    current_portfolio_score: float,
    params: dict[str, object],
    safe_model_uncertainty_buffer: float,
    current_state,
    latest_prices: pd.Series,
    active_tickers: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Write a lower-hurdle sensitivity grid without changing production defaults."""

    rows: list[dict[str, object]] = []
    first_trade: dict[str, object] | None = None
    hurdles = [0.0025, 0.0015, 0.0010, 0.00075, 0.0005]
    execution_buffers = [0.0010, 0.0005, 0.00035, 0.0000]
    model_multipliers = [1.0, 0.75, 0.5, 0.25]
    for hurdle, execution_buffer, multiplier in product(hurdles, execution_buffers, model_multipliers):
        summary = _select_active_preview_candidate(
            scores_frame=scores_frame,
            current_portfolio_score=float(current_portfolio_score),
            params=params,
            safe_model_uncertainty_buffer=float(safe_model_uncertainty_buffer),
            trade_now_hurdle=float(hurdle),
            execution_buffer=float(execution_buffer),
            model_uncertainty_multiplier=float(multiplier),
        )
        action = str(summary.get("active_preview_action", "HOLD"))
        candidate_name = str(summary.get("active_preview_candidate", "HOLD_CURRENT"))
        best_name = str(summary.get("active_preview_best_candidate", candidate_name))
        clears_hurdle = bool(action == "BUY_SELL_PREVIEW")
        row = {
            "trade_now_hurdle": float(hurdle),
            "execution_buffer": float(execution_buffer),
            "model_uncertainty_multiplier": float(multiplier),
            "best_candidate": best_name,
            "best_candidate_valid": bool(summary.get("active_preview_best_candidate_valid", False)),
            "final_action_under_params": action,
            "trade_now_edge": float(summary.get("active_preview_trade_now_edge", 0.0)),
            "hurdle": float(summary.get("active_preview_hurdle", hurdle)),
            "clears_hurdle": clears_hurdle,
            "order_count": int(summary.get("active_preview_order_count", 0)) if clears_hurdle else 0,
            "turnover": float(summary.get("active_preview_turnover", 0.0)),
            "delta_vs_current": float(summary.get("active_preview_delta_vs_current", 0.0)),
            "delta_vs_cash": float(summary.get("active_preview_delta_vs_cash", 0.0)),
            "probability_beats_current": float(summary.get("active_preview_probability_beats_current", 0.0)),
            "probability_beats_cash": float(summary.get("active_preview_probability_beats_cash", 0.0)),
            "blocking_reason": str(summary.get("active_preview_blocking_reason", "")),
            "binding_constraints": str(summary.get("active_preview_binding_constraints", "none")),
            "selected_candidate_if_trade": candidate_name if clears_hurdle else "",
        }
        rows.append(row)
        if clears_hurdle and first_trade is None:
            first_trade = {**summary, **row}

    matrix = pd.DataFrame(rows)
    _write_csv(output_dir / "rebalance_sensitivity_matrix.csv", matrix, index=False)

    first_trade_lines: list[str]
    if first_trade is None:
        first_trade_lines = [
            "First trade threshold: none in tested grid",
            "Even the least conservative tested grid did not create a valid non-HOLD Active Preview trade.",
        ]
    else:
        first_candidate = str(first_trade.get("selected_candidate_if_trade") or first_trade.get("active_preview_candidate", ""))
        order_lines: list[str] = []
        candidate = candidate_map.get(first_candidate)
        if candidate is not None:
            preview = build_discrete_order_preview(
                current_shares=current_state.current_shares.reindex(active_tickers).fillna(0.0),
                target_shares=candidate.shares.reindex(active_tickers).fillna(0.0),
                latest_prices=latest_prices.reindex(active_tickers).fillna(0.0),
                nav=float(current_state.nav),
                min_order_value=float(params.get("active_preview_min_order_value_usd", params.get("min_order_value_usd", 10.0))),
                not_executable=True,
                reason="active_preview_never_executes",
            )
            actionable = preview.loc[preview["side"].astype(str).isin(["BUY", "SELL"])].copy()
            for order in actionable.head(8).itertuples(index=False):
                order_lines.append(
                    f"- {order.side} {float(order.order_shares):.0f} {order.ticker} value={abs(float(order.order_value)):.2f}"
                )
        first_trade_lines = [
            "First trade threshold:",
            f"- trade_now_hurdle={float(first_trade['trade_now_hurdle']):.6f}",
            f"- execution_buffer={float(first_trade['execution_buffer']):.6f}",
            f"- model_uncertainty_multiplier={float(first_trade['model_uncertainty_multiplier']):.2f}",
            f"- candidate={first_candidate}",
            f"- trade_now_edge={float(first_trade['trade_now_edge']):.6f}",
            "Orders that would appear in Active Preview only:",
            *(order_lines or ["- none"]),
        ]
    lines = [
        "Rebalance Sensitivity Matrix",
        "",
        "This grid is analysis-only and never changes Safe Mode execution.",
        f"rows: {len(matrix)}",
        "",
        *first_trade_lines,
        "",
        "Sensitivity conclusion:",
        "- If final_action_under_params remains HOLD, the tested hurdle/buffer combination did not clear Active Preview.",
        "- If BUY_SELL_PREVIEW appears, those orders are still non-executable and live only in outputs/active_preview_orders.csv.",
    ]
    _write_text(output_dir / "rebalance_sensitivity_matrix.txt", "\n".join(lines) + "\n")
    return matrix


def _select_discrete_expansion_sources(
    scores_frame: pd.DataFrame,
    continuous_model_optimal_candidate: str,
) -> list[str]:
    """Return all non-HOLD continuous candidates to expand into discrete variants.

    The candidate set is intentionally small. Expanding all non-HOLD/non-cash
    continuous candidates avoids silently dropping a lower-ranked continuous
    portfolio that may discretize better than the top few rows.
    """

    if not isinstance(scores_frame, pd.DataFrame) or "candidate" not in scores_frame.columns:
        return [str(continuous_model_optimal_candidate)]

    expansion_sources: list[str] = []
    for candidate_name in scores_frame["candidate"].astype(str).tolist():
        if candidate_name in {"HOLD", "DEFENSIVE_CASH"}:
            continue
        if candidate_name not in expansion_sources:
            expansion_sources.append(candidate_name)
    if str(continuous_model_optimal_candidate) not in expansion_sources:
        expansion_sources.insert(0, str(continuous_model_optimal_candidate))
    return expansion_sources


def _top_rejected_discrete_candidates(
    scores_frame: pd.DataFrame,
    *,
    best_candidate_name: str,
    risk_premium_hurdle: float,
    p_hold_min: float,
    p_cash_min: float,
    top_n: int = 3,
) -> list[str]:
    if not isinstance(scores_frame, pd.DataFrame) or scores_frame.empty:
        return []
    others = scores_frame.loc[
        scores_frame["discrete_candidate"].astype(str) != str(best_candidate_name)
    ].copy()
    if others.empty:
        return []
    others = others.sort_values("net_robust_score", ascending=False)
    lines: list[str] = []
    for _, row in others.head(top_n).iterrows():
        reason = _candidate_failure_reason(
            row,
            risk_premium_hurdle=risk_premium_hurdle,
            p_hold_min=p_hold_min,
            p_cash_min=p_cash_min,
        )
        lines.append(
            f"{row['discrete_candidate']}: net={float(row['net_robust_score']):.6f}, "
            f"delta_vs_current={float(row.get('delta_vs_hold', 0.0)):.6f}, reason={reason}"
        )
    return lines


def _build_hold_vs_target_analysis(
    *,
    as_of: object,
    current_portfolio_score: float,
    target_score_before_costs: float,
    target_score_after_costs: float,
    delta_score_vs_current: float,
    total_order_cost: float,
    execution_buffer: float,
    model_uncertainty_buffer: float,
    trade_now_edge: float,
    trade_now_hurdle: float,
    probability_beats_current: float,
    probability_beats_cash: float,
    tail_risk_current: float,
    tail_risk_target: float,
    current_weights: pd.Series,
    continuous_target_weights: pd.Series,
    final_discrete_weights: pd.Series,
    continuous_model_optimal_candidate: str,
    best_discrete_candidate_name: str,
    factor_forecast_df: pd.DataFrame,
    discrete_scores_frame: pd.DataFrame,
    gate_reason: str,
    data_context: dict[str, object],
    risk_premium_hurdle: float,
    p_hold_min: float,
    p_cash_min: float,
) -> tuple[str, dict[str, object]]:
    factor_driver_names = _factor_driver_names(factor_forecast_df)
    continuous_target_drivers = _top_weight_change_lines(current_weights, continuous_target_weights, top_n=4)
    discrete_target_drivers = _top_weight_change_lines(current_weights, final_discrete_weights, top_n=4)
    rejected_candidates = _top_rejected_discrete_candidates(
        discrete_scores_frame,
        best_candidate_name=best_discrete_candidate_name,
        risk_premium_hurdle=risk_premium_hurdle,
        p_hold_min=p_hold_min,
        p_cash_min=p_cash_min,
        top_n=3,
    )

    target_driver_lines: list[str] = []
    if continuous_target_drivers:
        target_driver_lines.append(
            "Kontinuierliches Modell wollte vor allem diese Uebergewichte relativ zum aktuellen Portfolio: "
            + "; ".join(continuous_target_drivers)
        )
    if factor_driver_names:
        target_driver_lines.append("Top Faktor-/Makrotreiber: " + ", ".join(factor_driver_names))
    if discrete_target_drivers:
        target_driver_lines.append(
            "Das finale diskrete Ziel haette diese tatsaechlichen Uebergewichte: " + "; ".join(discrete_target_drivers)
        )
    if best_discrete_candidate_name == "HOLD_CURRENT":
        target_driver_lines.append(
            "Die kaufbare diskrete Umsetzung blieb bei HOLD_CURRENT, weil keine Alternative nach Kosten, Rundung und Huerden robust besser war."
        )
    if not target_driver_lines:
        target_driver_lines.append("Es gab keine materialen Zielgewichtsdifferenzen gegenueber dem aktuellen Portfolio.")

    reasons_against_trade: list[str] = []
    if trade_now_edge <= 0.0:
        reasons_against_trade.append(
            f"Die Net-Trade-Now-Edge ist negativ ({trade_now_edge:.6f}) und liegt damit unter der Execution-Huerde."
        )
    reasons_against_trade.append(gate_reason)
    if best_discrete_candidate_name == "HOLD_CURRENT":
        reasons_against_trade.append(
            "Das beste diskrete Zielportfolio ist identisch mit dem aktuellen Portfolio; es entstehen keine BUY/SELL-Deltas."
        )
    if bool(data_context.get("used_cache_fallback", False)):
        reasons_against_trade.append(
            "Der Lauf nutzte Cache-Fallback statt eines erfolgreichen Live-Refreshs; fuer echte Freigaben sollte der Datenpfad weiterhin konservativ behandelt werden."
        )
    reasons_against_trade.extend(rejected_candidates)

    edge_gap_to_hurdle = max(float(trade_now_hurdle) - float(trade_now_edge), 0.0)
    release_conditions = [
        f"Eine nicht-HOLD-Variante muss die Trade-Now-Huerde klar schlagen; aktuell fehlt etwa {edge_gap_to_hurdle:.6f} Net-Edge bis zur Huerde {trade_now_hurdle:.6f}.",
        f"Eine diskrete Alternative muss delta_vs_current > 0 sowie delta_vs_cash > {risk_premium_hurdle:.6f} erreichen.",
        f"Die Wahrscheinlichkeiten muessen mindestens p_current >= {p_hold_min:.2%} und p_cash >= {p_cash_min:.2%} erreichen.",
        "Die kaufbare Whole-Share-Umsetzung muss nach Rounding und Constraints valide bleiben und nicht auf HOLD_CURRENT zurueckfallen.",
    ]
    if bool(data_context.get("used_cache_fallback", False)):
        release_conditions.append("Ein erfolgreicher Live-Daten-Refresh ohne Cache-Fallback waere fuer spaetere nicht-nur-Preview-Freigaben die robustere Basis.")

    probability_note = ""
    if best_discrete_candidate_name == "HOLD_CURRENT" and abs(delta_score_vs_current) <= 1e-12:
        probability_note = "Hinweis: probability_beats_current ist hier eine strikte Outperformance-Metrik; ein identisches HOLD_CURRENT-Portfolio schlaegt sich selbst daher nicht."

    lines = [
        "Hold-vs-Rebalance-Vermessung",
        "",
        f"Date: {getattr(as_of, 'date', lambda: as_of)() if hasattr(as_of, 'date') else as_of}",
        f"Continuous model-optimal candidate: {continuous_model_optimal_candidate}",
        f"Final discrete target candidate: {best_discrete_candidate_name}",
        "",
        "1. Score-Vergleich",
        f"- current_portfolio_score: {current_portfolio_score:.6f}",
        f"- target_score_before_costs: {target_score_before_costs:.6f}",
        f"- target_score_after_costs: {target_score_after_costs:.6f}",
        f"- delta_score_vs_current: {delta_score_vs_current:.6f}",
        f"- total_order_cost: {total_order_cost:.2f} USD",
        f"- execution_buffer: {execution_buffer:.6f}",
        f"- model_uncertainty_buffer: {model_uncertainty_buffer:.6f}",
        f"- trade_now_edge: {trade_now_edge:.6f}",
        "",
        "2. Wahrscheinlichkeiten und Tail Risk",
        f"- probability_beats_current: {probability_beats_current:.2%}",
        f"- probability_beats_cash: {probability_beats_cash:.2%}",
        f"- tail_risk_current: {tail_risk_current:.2%}",
        f"- tail_risk_target: {tail_risk_target:.2%}",
    ]
    if probability_note:
        lines.append(f"- {probability_note}")
    lines.extend(
        [
            "",
            "3. Wichtigste Treiber fuer das Zielportfolio",
            *[f"- {line}" for line in target_driver_lines],
            "",
            "4. Wichtigste Gruende gegen sofortiges Handeln",
            *[f"- {line}" for line in reasons_against_trade[:6]],
            "",
            "5. Was muesste sich aendern, damit BUY/SELL freigegeben wird?",
            *[f"- {line}" for line in release_conditions],
        ]
    )

    summary = {
        "current_portfolio_score": float(current_portfolio_score),
        "target_score_before_costs": float(target_score_before_costs),
        "target_score_after_costs": float(target_score_after_costs),
        "delta_score_vs_current": float(delta_score_vs_current),
        "total_order_cost": float(total_order_cost),
        "execution_buffer": float(execution_buffer),
        "model_uncertainty_buffer": float(model_uncertainty_buffer),
        "trade_now_edge": float(trade_now_edge),
        "probability_beats_current": float(probability_beats_current),
        "probability_beats_cash": float(probability_beats_cash),
        "tail_risk_current": float(tail_risk_current),
        "tail_risk_target": float(tail_risk_target),
        "target_drivers": target_driver_lines,
        "reasons_against_trade": reasons_against_trade,
        "release_conditions": release_conditions,
        "probability_beats_current_note": probability_note,
    }
    return "\n".join(lines) + "\n", summary


def _build_today_decision_summary(
    *,
    as_of: object,
    data_freshness: dict[str, object],
    price_attrs: dict[str, object],
    current_state,
    continuous_model_optimal_candidate: str,
    continuous_target_weights: pd.Series,
    best_discrete_candidate_name: str,
    best_discrete_candidate,
    manual_simulator_orders: pd.DataFrame,
    market_gate: dict[str, object],
    gate,
    execution_result: dict[str, object],
    delta_score: float,
    current_portfolio_score: float,
    best_discrete_score: float,
    cash_before_orders: float,
    cash_after_orders: float,
    cost_review: dict[str, object],
    hold_vs_target_summary: dict[str, object],
    review_issues: dict[str, object],
    active_preview_summary: dict[str, object] | None = None,
) -> str:
    current_portfolio_text = (
        "100 % Cash"
        if current_state.current_shares.abs().sum() <= 1e-9
        else "bestehenden ETF-Positionen plus Cash"
    )
    manual_lines: list[str] = []
    for row in manual_simulator_orders.itertuples(index=False):
        manual_lines.append(f"- {row.action} {row.shares} {row.ticker}")
    if not manual_lines:
        manual_lines.append("- Keine manuellen BUY/SELL-Orders.")

    risks: list[str] = [
        "- Der letzte adjusted close ist nur ein Preis-Proxy und kein handelbarer Live-Quote.",
    ]
    if bool(price_attrs.get("used_cache_fallback", False)):
        risks.append("- Dieser Lauf hat nach einem Live-Datenproblem auf den Cache zurueckgegriffen. Analyse ist weiter moeglich, Execution sollte aber konservativ bleiben.")
    if not bool(market_gate.get("execution_allowed", False)):
        risks.append(
            f"- Das Projekt-Handelsfenster ist aktuell geschlossen: {market_gate.get('reason', 'calendar_blocked')}."
        )
    crypto_weight = float(
        best_discrete_candidate.weights_actual.reindex(["IBIT", "ETHA"]).fillna(0.0).sum()
    )
    if crypto_weight > 1e-6:
        risks.append("- Crypto-ETFs handeln nur zu Marktzeiten, waehrend die Underlyings 24/7 laufen. Weekend- und Gap-Risiko bleiben also bestehen.")
    inverse_weight = float(
        best_discrete_candidate.weights_actual.reindex(["SH", "PSQ"]).fillna(0.0).sum()
    )
    if inverse_weight > 1e-6:
        risks.append("- Inverse ETFs koennen sich ueber die Zeit anders verhalten als eine einfache Short-Position und brauchen besondere Vorsicht.")

    if int(review_issues.get("hard_fail_count", 0)) > 0:
        recommendation = "Blockiert. Mindestens ein Hard-Fail liegt vor. Heute nichts manuell im Simulator eingeben und erst den ersten Blocker beheben."
    elif gate.action == "HOLD":
        recommendation = "Halten. Heute nichts manuell im Simulator eingeben, weil das ausfuehrbare Zielportfolio nach Kosten und Buffern nicht stark genug besser ist als das aktuelle Portfolio."
    elif gate.action.startswith("WAIT"):
        recommendation = "Warten. Das Modell sieht zwar eine Idee fuer Umschichtung, aber das Execution-Gate blockiert aktuell. Heute keine manuelle Simulatororder eingeben."
    elif gate.gate_status == "PASS":
        recommendation = "Handelbar in der Logik, aber dieser Lauf bleibt trotzdem ein Dry-Run ohne echte Orders."
    else:
        recommendation = "Konservativ blockiert. Im aktuellen Zustand wird keine Ausfuehrung freigegeben und heute soll nichts manuell eingegeben werden."

    executable_now = bool(gate.gate_status == "PASS" and execution_result.get("execution_mode") != "blocked")
    hold_vs_reasons = list(hold_vs_target_summary.get("reasons_against_trade", []))
    hold_vs_release = list(hold_vs_target_summary.get("release_conditions", []))
    selected_reason = str(hold_vs_target_summary.get("selected_reason", "unknown"))
    active_preview = dict(active_preview_summary or {})
    active_action = str(active_preview.get("active_preview_action", "HOLD"))
    active_candidate = str(active_preview.get("active_preview_candidate", "HOLD_CURRENT"))
    if gate.action == "HOLD" and active_action == "BUY_SELL_PREVIEW":
        active_preview_note = "Safe Mode remains HOLD, but Active Preview finds a lower-hurdle rebalance candidate."
    elif active_action == "BUY_SELL_PREVIEW":
        active_preview_note = "Active Preview finds a lower-hurdle rebalance candidate, but it is not executable."
    else:
        active_preview_note = "Even Active Preview did not find a valid non-HOLD candidate."
    current_constraint_valid = bool(hold_vs_target_summary.get("current_portfolio_constraint_valid", True))
    current_constraint_errors = str(hold_vs_target_summary.get("current_constraint_errors", "") or "none")
    current_asset_violation_count = int(hold_vs_target_summary.get("current_portfolio_asset_limit_violations", 0) or 0)
    current_group_violation_count = int(hold_vs_target_summary.get("current_portfolio_group_limit_violations", 0) or 0)
    lines = [
        "Heute: Entscheidungs-Zusammenfassung",
        "",
        f"Date: {getattr(as_of, 'date', lambda: as_of)() if hasattr(as_of, 'date') else as_of}",
        "",
        "1. Kurzentscheidung",
        recommendation,
        (
            f"Final action: {gate.action}. "
            f"Execution mode: {execution_result.get('execution_mode', 'unknown')}. "
            f"Erster Blocker: {review_issues.get('first_blocker', gate.reason)}."
        ),
        (
            f"Review-Status: {review_issues.get('review_status', 'REVIEW')}. "
            f"hard_fail_count={int(review_issues.get('hard_fail_count', 0))}, "
            f"soft_warning_count={int(review_issues.get('soft_warning_count', 0))}, "
            f"info_count={int(review_issues.get('info_count', 0))}."
        ),
        (
            "Sofortentscheidung: Heute keine Orders im Simulator eingeben."
            if manual_simulator_orders.empty or gate.gate_status != "PASS"
            else "Sofortentscheidung: Nur die freigegebenen Delta-Orders aus `outputs/manual_simulator_orders.csv` verwenden."
        ),
        "",
        "1b. Safe Mode vs Active Preview",
        "SAFE MODE:",
        f"- final_action: {gate.action}",
        f"- final_discrete_candidate: {best_discrete_candidate_name}",
        f"- selected_reason: {selected_reason}",
        f"- trade_now_edge: {float(hold_vs_target_summary.get('trade_now_edge', 0.0)):.6f}",
        "- trade_now_hurdle: 0.002500",
        f"- order_count: {int(manual_simulator_orders.shape[0])}",
        "ACTIVE PREVIEW:",
        f"- active_preview_action: {active_action}",
        f"- active_preview_candidate: {active_candidate}",
        f"- active_preview_trade_now_edge: {_safe_float(active_preview.get('active_preview_trade_now_edge')):.6f}",
        f"- active_preview_hurdle: {_safe_float(active_preview.get('active_preview_hurdle')):.6f}",
        f"- active_preview_order_count: {int(active_preview.get('active_preview_order_count', 0) or 0)}",
        f"- active_preview_buy_count: {int(active_preview.get('active_preview_buy_count', 0) or 0)}",
        f"- active_preview_sell_count: {int(active_preview.get('active_preview_sell_count', 0) or 0)}",
        f"- active_preview_turnover: {_safe_float(active_preview.get('active_preview_turnover')):.6f}",
        f"- active_preview_reason: {active_preview.get('active_preview_reason', 'unknown')}",
        "- active_preview_executable: false",
        active_preview_note,
        "",
        "2. Aktuelles Portfolio",
        (
            f"Der Bot geht aktuell von {current_portfolio_text} aus. "
            f"Das effektive Cash-Gewicht liegt bei {float(current_state.actual_cash_weight):.2%} aus Quelle {current_state.source}. "
            f"Current score: {float(hold_vs_target_summary.get('current_portfolio_score', current_portfolio_score)):.6f}."
        ),
        (
            f"Current portfolio constraint-valid: {_ja_nein(current_constraint_valid)}. "
            f"selected_reason={selected_reason}. "
            f"current_constraint_errors={current_constraint_errors}. "
            f"asset_limit_violations={current_asset_violation_count}, group_limit_violations={current_group_violation_count}."
        ),
        "",
        "3. Zielportfolio",
        (
            f"Der theoretische Modell-Sieger ist {continuous_model_optimal_candidate}. "
            f"Die groessten Zielgewichte sind: {_format_top_weights(continuous_target_weights)}."
        ),
        (
            f"Das finale diskrete Ziel ist {best_discrete_candidate_name}. "
            f"Die groessten kaufbaren Gewichte sind: {_format_top_weights(best_discrete_candidate.weights_actual)}. "
            f"Target score nach Kosten: {float(hold_vs_target_summary.get('target_score_after_costs', best_discrete_score)):.6f}."
        ),
        (
            "HOLD_CURRENT ist hier ein Safe-Fallback mit Constraint-Verletzungen, nicht die perfekte/optimale Zielallokation."
            if selected_reason == "constraint_invalid_hold_fallback"
            else f"Final selection reason: {selected_reason}."
        ),
        "",
        "4. Delta-Orders",
        *manual_lines,
        "Diese Orders sind Delta-Orders relativ zum aktuellen Portfolio.",
        f"Cash vor Orders: {float(cash_before_orders):.2f} USD. Cash nach Orders: {float(cash_after_orders):.2f} USD.",
        (
            "Wenn `outputs/manual_simulator_orders.csv` leer ist, dann gibt es fuer heute keine manuelle BUY/SELL-Eingabe."
            if manual_simulator_orders.empty
            else "Wenn du manuell im Simulator arbeiten willst, nutze nur die BUY/SELL-Zeilen aus `outputs/manual_simulator_orders.csv`."
        ),
        "",
        "5. Warum handeln / warum nicht handeln",
        (
            f"Delta score vs current: {float(hold_vs_target_summary.get('delta_score_vs_current', delta_score)):.6f}. "
            f"P(beats current): {float(hold_vs_target_summary.get('probability_beats_current', 0.0)):.2%}. "
            f"P(beats cash): {float(hold_vs_target_summary.get('probability_beats_cash', 0.0)):.2%}."
        ),
        *[f"- {line}" for line in hold_vs_reasons[:4]],
        f"Was sich fuer BUY/SELL aendern muesste: {hold_vs_release[0] if hold_vs_release else 'Eine diskrete Alternative muss klar besser als HOLD_CURRENT werden.'}",
        "",
        "6. Datenstatus",
        (
            f"latest_price_date={data_freshness.get('latest_price_date', 'n/a')}, "
            f"data_freshness_ok={_ja_nein(data_freshness.get('data_freshness_ok', False))}, "
            f"data_source={price_attrs.get('data_source', 'unknown')}, "
            f"synthetic_data={_ja_nein(price_attrs.get('synthetic_data', False))}, "
            f"used_cache_fallback={_ja_nein(price_attrs.get('used_cache_fallback', False))}."
        ),
        (
            f"expected_latest_trading_day={data_freshness.get('expected_latest_trading_day', 'n/a')}, "
            f"staleness_days={data_freshness.get('staleness_days', 'n/a')}."
        ),
        "",
        "7. Kostenstatus",
        (
            f"Direkte Simulatorgebuehren: {float(cost_review.get('simulator_order_fee_usd', 0.0)):.2f} USD je Order, "
            f"gesamt {float(cost_review.get('total_simulator_order_fees_usd', 0.0)):.2f} USD."
        ),
        (
            f"Konservative Modellkosten: {float(cost_review.get('modeled_transaction_costs_usd', 0.0)):.2f} USD "
            f"({float(cost_review.get('modeled_transaction_costs_pct_nav', 0.0)):.6f} NAV) "
            f"mit Spread {float(cost_review.get('modeled_spread_bps', 0.0)):.2f} bps, "
            f"Slippage {float(cost_review.get('modeled_slippage_bps', 0.0)):.2f} bps und "
            f"Turnover {float(cost_review.get('modeled_bps_per_turnover', 0.0)):.2f} bps."
        ),
        f"Trade-now-Edge nach Modellkosten/Buffern: {float(cost_review.get('trade_now_edge_after_modeled_costs', 0.0)):.6f}.",
        "",
        "8. Kalender-/Gate-Status",
        (
            f"Handelsfenster offen: {_ja_nein(market_gate.get('execution_allowed', False))}. "
            f"Berlin-Zeitfenster {market_gate.get('allowed_start_berlin', 'n/a')} bis {market_gate.get('allowed_end_berlin', 'n/a')}. "
            f"Aktueller Kalenderstatus: {market_gate.get('reason', 'n/a')}."
        ),
        (
            f"is_project_trading_day={_ja_nein(market_gate.get('is_trading_day', False))}, "
            f"within_allowed_window={_ja_nein(market_gate.get('execution_allowed', False))}."
        ),
        (
            "Jetzt ausfuehrbar."
            if executable_now
            else f"Nur Preview. Final action ist {gate.action}, execution mode ist {execution_result.get('execution_mode', 'unknown')} und der Gate-Grund lautet: {gate.reason}"
        ),
        "",
        "9. Risiken",
        *risks,
        f"first_blocker: {review_issues.get('first_blocker', 'none')}",
        f"all_blockers: {' | '.join(map(str, review_issues.get('all_blockers', ['none'])))}",
        "issue_table:",
        *[
            f"- {item.get('severity')}: {item.get('message')}"
            for item in list(review_issues.get("issue_table", []))
        ],
        "",
        "10. Welche Datei soll ich fuer den Simulator verwenden?",
        "Diese Datei verwenden: outputs/manual_simulator_orders.csv",
        "Nicht verwenden für manuelle Simulatororders: outputs/order_preview.csv",
        "Diese Orders sind Delta-Orders relativ zum aktuellen Portfolio",
        "Keine echten Orders wurden gesendet",
        (
            "Heute keine Orders im Simulator eingeben, solange `outputs/manual_simulator_orders.csv` leer ist."
            if manual_simulator_orders.empty
            else "Pruefe vor manueller Eingabe noch einmal, dass jede Zeile eine echte BUY/SELL-Delta-Order ist."
        ),
        "Manual simulator entry only.",
        f"Absolute Pfade: {OUTPUT_DIR / 'manual_simulator_orders.csv'} | {OUTPUT_DIR / 'manual_simulator_orders.txt'} | {OUTPUT_DIR / 'best_discrete_order_preview.csv'}",
    ]
    return "\n".join(lines) + "\n"


def _main_blocker_category(
    *,
    data_context: dict[str, object],
    validation_result: dict[str, object],
    gate,
    hold_vs_target_summary: dict[str, object],
) -> str:
    if not bool(data_context.get("execution_allowed_by_calendar", False)):
        return "calendar"
    if bool(data_context.get("synthetic_data", False)) or not bool(data_context.get("data_freshness_ok", False)):
        return "data"
    if not bool(validation_result.get("ok", False)):
        return "validation"
    gate_reason = str(getattr(gate, "reason", "") or "").lower()
    if "constraint" in gate_reason or any("constraint" in str(item).lower() for item in hold_vs_target_summary.get("reasons_against_trade", [])):
        return "constraints"
    if "edge" in gate_reason or "cost" in gate_reason or _safe_float(hold_vs_target_summary.get("trade_now_edge", 0.0)) < 0.0:
        return "costs/edge"
    return "execution_gate"


def _build_daily_review_payload(
    *,
    as_of: pd.Timestamp,
    data_context: dict[str, object],
    current_state,
    latest_prices_at_asof: pd.Series,
    active_tickers: list[str],
    continuous_model_optimal_candidate: str,
    continuous_target_weights: pd.Series,
    best_discrete_candidate_name: str,
    best_discrete_candidate,
    final_target_shares: pd.Series,
    annotated_adjusted_order_preview: pd.DataFrame,
    cost_review: dict[str, object],
    hold_vs_target_summary: dict[str, object],
    gate,
    execution_result: dict[str, object],
    trade_edge_summary: dict[str, object],
    data_quality_report: dict[str, object],
    validation_result: dict[str, object],
    discrete_selected_score,
    order_summary: dict[str, object],
    active_preview_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    now_berlin = datetime.now(BERLIN_TZ)
    data_quality_df = data_quality_report.get("report_df", pd.DataFrame())
    if isinstance(data_quality_df, pd.DataFrame) and not data_quality_df.empty and {"ticker", "history_length"}.issubset(data_quality_df.columns):
        low_history_assets = [
            str(ticker)
            for ticker in data_quality_df.loc[pd.to_numeric(data_quality_df["history_length"], errors="coerce").fillna(0).lt(252), "ticker"].astype(str).tolist()
        ]
    else:
        low_history_assets = []
    current_positions: list[dict[str, object]] = []
    missing_prices: list[str] = []
    for ticker in active_tickers:
        shares = float(current_state.current_shares.get(ticker, 0.0))
        if abs(shares) <= 1e-9:
            continue
        latest_price = _safe_float(latest_prices_at_asof.get(ticker, 0.0))
        if latest_price <= 0.0:
            missing_prices.append(str(ticker))
        current_positions.append(
            {
                "ticker": str(ticker),
                "current_shares": shares,
                "latest_price": latest_price,
                "latest_price_date": data_context.get("latest_price_date", "n/a"),
                "market_value_usd": float(current_state.current_values.get(ticker, 0.0)),
                "current_weight": float(current_state.current_weights_actual.get(ticker, 0.0)),
                "price_basis": data_context.get("price_basis", "adjusted_close_proxy"),
                "data_source": data_context.get("data_source", "unknown"),
                "stale_price_warning": not bool(data_context.get("data_freshness_ok", False)),
                "data_warning": "missing_latest_price"
                if latest_price <= 0.0
                else (
                    "adjusted_close_proxy"
                    if str(data_context.get("price_basis", "")) == "adjusted_close_proxy"
                    else ""
                ),
            }
        )

    target_allocation: list[dict[str, object]] = []
    for ticker in active_tickers:
        target_weight = float(best_discrete_candidate.weights_actual.get(ticker, 0.0))
        target_shares = float(final_target_shares.get(ticker, 0.0))
        continuous_weight = float(continuous_target_weights.get(ticker, 0.0))
        latest_price = _safe_float(latest_prices_at_asof.get(ticker, 0.0))
        if target_weight <= 1e-9 and target_shares <= 1e-9 and continuous_weight <= 1e-9:
            continue
        current_weight = float(current_state.current_weights_actual.get(ticker, 0.0))
        target_allocation.append(
            {
                "ticker": str(ticker),
                "target_weight": target_weight,
                "target_shares": target_shares,
                "target_market_value_usd": float(best_discrete_candidate.values.get(ticker, 0.0)),
                "continuous_target_weight": continuous_weight,
                "abs_weight_drift": abs(target_weight - current_weight),
                "latest_price": latest_price,
            }
        )

    actionable_preview = annotated_adjusted_order_preview.loc[
        annotated_adjusted_order_preview["action"].astype(str).isin(["BUY", "SELL"])
        & pd.to_numeric(annotated_adjusted_order_preview["order_shares"], errors="coerce").fillna(0.0).gt(1e-9)
    ].copy()
    delta_transactions: list[dict[str, object]] = []
    for row in actionable_preview.itertuples(index=False):
        delta_transactions.append(
            {
                "ticker": str(getattr(row, "ticker", "")),
                "action": str(getattr(row, "action", getattr(row, "side", "HOLD"))),
                "current_shares": float(getattr(row, "current_shares", 0.0)),
                "target_shares": float(getattr(row, "target_shares", 0.0)),
                "order_shares": float(getattr(row, "order_shares", 0.0)),
                "estimated_price": float(getattr(row, "estimated_price", 0.0)),
                "estimated_order_value": abs(float(getattr(row, "estimated_order_value", 0.0))),
                "simulator_fee_usd": 0.0,
                "modeled_transaction_cost_usd": float(getattr(row, "estimated_total_order_cost", 0.0)),
                "preview_only": bool(getattr(row, "preview_only", True)),
                "not_executable": bool(getattr(row, "not_executable", True)),
                "execution_block_reason": str(
                    getattr(row, "execution_block_reason", "") or getattr(row, "not_executable_reason", "") or getattr(row, "reason", "")
                ).strip(),
            }
        )

    negative_drivers = list(hold_vs_target_summary.get("reasons_against_trade", []))
    rejected_candidates = [item for item in negative_drivers if "::" in str(item)]
    positive_drivers = list(hold_vs_target_summary.get("target_drivers", []))
    why_not_cash = (
        f"Das aktuelle/finale Portfolio bleibt gegen Cash ueberlegen mit delta_vs_cash={float(discrete_selected_score.delta_vs_cash):.6f} "
        f"und probability_beats_cash={float(discrete_selected_score.probability_beats_cash):.2%}."
        if float(discrete_selected_score.probability_beats_cash) > 0.5
        else "Cash bleibt eine valide Referenz, aber aktuell ist kein klarer Cash-Switch noetig."
    )
    why_not_hold = (
        "HOLD_CURRENT blieb das beste kaufbare diskrete Portfolio."
        if best_discrete_candidate_name == "HOLD_CURRENT"
        else "Eine nicht-HOLD-Variante waere nur zulaessig, wenn sie HOLD_CURRENT nach Kosten und Buffern klar schlaegt."
    )
    active_preview = dict(active_preview_summary or {})

    return {
        "run_status": {
            "review_date": str(now_berlin.date()),
            "review_time_berlin": now_berlin.strftime("%H:%M:%S"),
            "current_date_berlin": data_context.get("current_date_berlin", str(now_berlin.date())),
            "current_time_berlin": data_context.get("current_time_berlin", now_berlin.strftime("%H:%M:%S")),
            "is_project_trading_day": bool(data_context.get("is_project_trading_day", False)),
            "within_allowed_window": bool(data_context.get("within_allowed_window", False)),
            "execution_allowed_by_calendar": bool(data_context.get("execution_allowed_by_calendar", False)),
            "final_action": getattr(gate, "action", "PAUSE"),
            "execution_mode": str(execution_result.get("execution_mode", "blocked")),
            "gate_reason": str(getattr(gate, "reason", "")),
        },
        "data_status": {
            "data_source": data_context.get("data_source", "unknown"),
            "cache_status": data_context.get("cache_status", "unknown"),
            "synthetic_data": bool(data_context.get("synthetic_data", False)),
            "used_cache_fallback": bool(data_context.get("used_cache_fallback", False)),
            "latest_price_date": data_context.get("latest_price_date", "n/a"),
            "staleness_days": data_context.get("staleness_days", "n/a"),
            "data_freshness_ok": bool(data_context.get("data_freshness_ok", False)),
            "live_data_error": data_context.get("live_data_error", ""),
            "missing_prices": missing_prices,
            "low_history_assets": low_history_assets,
            "price_basis": data_context.get("price_basis", "adjusted_close_proxy"),
        },
        "current_portfolio": {
            "current_portfolio_source": current_state.source,
            "positions_count": int(current_state.current_shares.abs().gt(1e-9).sum()),
            "cash_usd": float(current_state.current_cash),
            "invested_market_value_usd": float(current_state.current_values.sum()),
            "nav_usd": float(current_state.nav),
            "current_portfolio_100pct_cash": bool(current_state.current_shares.abs().sum() <= 1e-9 and current_state.current_cash > 0.0),
            "current_weights_sum_including_cash": float(current_state.current_weights_actual.sum()) + float(current_state.actual_cash_weight),
            "current_weights_sum_without_cash": float(current_state.current_weights_actual.sum()),
            "current_portfolio_constraint_valid": bool(hold_vs_target_summary.get("current_portfolio_constraint_valid", True)),
            "current_portfolio_constraint_violation": bool(hold_vs_target_summary.get("current_portfolio_constraint_violation", False)),
            "current_constraint_errors": str(hold_vs_target_summary.get("current_constraint_errors", "") or "none"),
            "parser_warnings": list(getattr(current_state, "parser_warnings", [])),
            "parser_errors": list(getattr(current_state, "parser_errors", [])),
        },
        "current_positions": current_positions,
        "target_allocation": target_allocation,
        "delta_transactions": delta_transactions,
        "cost_edge": {
            "simulator_fee_usd": float(cost_review.get("simulator_order_fee_usd", 0.0)),
            "total_simulator_fees_usd": float(cost_review.get("total_simulator_order_fees_usd", 0.0)),
            "modeled_transaction_costs_usd": float(cost_review.get("modeled_transaction_costs_usd", 0.0)),
            "modeled_transaction_costs_pct_nav": float(cost_review.get("modeled_transaction_costs_pct_nav", 0.0)),
            "current_portfolio_score": float(hold_vs_target_summary.get("current_portfolio_score", 0.0)),
            "target_score_before_costs": float(hold_vs_target_summary.get("target_score_before_costs", 0.0)),
            "target_score_after_costs": float(hold_vs_target_summary.get("target_score_after_costs", 0.0)),
            "delta_score_vs_current": float(hold_vs_target_summary.get("delta_score_vs_current", 0.0)),
            "execution_buffer": float(trade_edge_summary.get("execution_buffer", 0.0)),
            "model_uncertainty_buffer": float(trade_edge_summary.get("model_uncertainty_buffer", 0.0)),
            "trade_now_edge": float(hold_vs_target_summary.get("trade_now_edge", trade_edge_summary.get("trade_now_edge", 0.0))),
            "cost_model_used": str(cost_review.get("cost_model_used", "unknown")),
        },
        "order_summary": {
            "cash_before_orders": float(order_summary.get("cash_before_orders", 0.0)),
            "cash_after_orders": float(order_summary.get("cash_after_orders", 0.0)),
            "estimated_sell_value": float(order_summary.get("estimated_sell_value", 0.0)),
            "estimated_buy_value": float(order_summary.get("estimated_buy_value", 0.0)),
            "total_simulator_fees_usd": float(order_summary.get("total_simulator_fees_usd", 0.0)),
            "modeled_transaction_costs_usd": float(order_summary.get("modeled_transaction_costs_usd", 0.0)),
            "buy_count": int(order_summary.get("buy_count", 0)),
            "sell_count": int(order_summary.get("sell_count", 0)),
            "hold_count": int(order_summary.get("hold_count", 0)),
            "order_count": int(order_summary.get("order_count", 0)),
            "manual_eligible_order_count": int(order_summary.get("manual_eligible_order_count", 0)),
            "negative_cash_check": bool(order_summary.get("negative_cash_check", True)),
            "leverage_check": bool(order_summary.get("leverage_check", True)),
            "short_check": bool(order_summary.get("short_check", True)),
            "manual_orders_usable": bool(order_summary.get("manual_orders_usable", False)),
        },
        "decision_context": {
            "continuous_candidate": str(continuous_model_optimal_candidate),
            "final_discrete_candidate": str(best_discrete_candidate_name),
            "selected_reason": str(hold_vs_target_summary.get("selected_reason", "unknown")),
            "final_selection_is_safe_fallback": bool(hold_vs_target_summary.get("final_selection_is_safe_fallback", False)),
            "best_non_hold_candidate": str(hold_vs_target_summary.get("best_non_hold_candidate", "") or "none"),
            "best_non_hold_failed_reason": str(hold_vs_target_summary.get("best_non_hold_failed_reason", "") or "none"),
            "why_this_target": (
                f"Das kontinuierliche Modell bevorzugte {continuous_model_optimal_candidate}, "
                f"aber das finale kaufbare diskrete Ziel wurde zu {best_discrete_candidate_name}."
            ),
            "why_not_hold": why_not_hold,
            "why_not_cash": why_not_cash,
            "trade_decision_reason": str(getattr(gate, "reason", "")),
            "positive_drivers": positive_drivers,
            "negative_drivers": negative_drivers[:6],
            "rejected_candidates": rejected_candidates[:5],
            "main_blocker_category": _main_blocker_category(
                data_context=data_context,
                validation_result=validation_result,
                gate=gate,
                hold_vs_target_summary=hold_vs_target_summary,
            ),
        },
        "active_preview": active_preview,
        "pre_trade_validation_status": "PASS" if validation_result.get("ok", False) else "FAIL",
        "preview_only": bool(execution_result.get("execution_mode", "order_preview_only") in {"blocked", "order_preview_only", "preview_only"}),
        "manual_orders_preview_ready": bool(
            delta_transactions
            and execution_result.get("execution_mode", "order_preview_only") == "order_preview_only"
            and getattr(gate, "action", "HOLD") not in {"HOLD", "PAUSE"}
        ),
        "cash_after_orders": float(cost_review.get("cash_after_orders", 0.0)),
        "main_daily_scope_differs": True,
        "exception_message": "",
    }


def _pid_looks_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_daily_bot_lock(lock_path: Path = DAILY_BOT_LOCK_PATH) -> tuple[bool, str]:
    """Acquire a simple lockfile to prevent overlapping cron-style runs."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        created_at_text = str(payload.get("created_at_utc", ""))
        pid_value = payload.get("pid")
        pid = int(pid_value) if isinstance(pid_value, int | float) else None
        created_at = None
        if created_at_text:
            try:
                created_at = datetime.fromisoformat(created_at_text)
            except Exception:
                created_at = None
        age_seconds = (
            max((now - created_at).total_seconds(), 0.0)
            if created_at is not None
            else float("inf")
        )
        if _pid_looks_alive(pid) and age_seconds < LOCK_STALE_AFTER_SECONDS:
            return False, f"Daily bot lock is active for pid={pid}; skipping overlapping run."
        LOGGER.warning(
            "Removing stale or invalid daily bot lock at %s (pid=%s, age_seconds=%s).",
            lock_path,
            pid,
            f"{age_seconds:.0f}" if age_seconds != float("inf") else "unknown",
        )
        lock_path.unlink(missing_ok=True)

    _write_state(
        lock_path,
        {
            "pid": os.getpid(),
            "created_at_utc": now.isoformat(),
            "cwd": str(Path.cwd()),
        },
    )
    return True, "lock_acquired"


def _release_daily_bot_lock(lock_path: Path = DAILY_BOT_LOCK_PATH) -> None:
    if lock_path.exists():
        lock_path.unlink(missing_ok=True)


def _expected_execution_mode(params: dict[str, object]) -> str:
    if bool(params.get("dry_run", True)):
        return "order_preview_only"
    if bool(params.get("enable_investopedia_simulator", False)):
        return "investopedia"
    if bool(params.get("enable_local_paper_trading", False)):
        return "local_paper"
    return "order_preview_only"


def _write_minimal_daily_bot_reports(diagnostics) -> None:
    lines = [
        f"Run ID: {diagnostics.run_id}",
        f"Mode: {diagnostics.mode}",
        f"Dry Run: {diagnostics.dry_run}",
        f"Signal Date: {diagnostics.signal_date or 'n/a'}",
        f"Execution Date: {diagnostics.execution_date or 'n/a'}",
        f"Final Action: {diagnostics.final_action}",
        f"Selected Candidate: {diagnostics.selected_candidate}",
        f"Execution Mode: {diagnostics.execution_mode}",
        f"Reason: {diagnostics.final_reason or diagnostics.execution_gate_context.get('reason', 'n/a')}",
        f"Data Source: {diagnostics.data_context.get('data_source', 'n/a')}",
        f"Cache Status: {diagnostics.data_context.get('cache_status', 'n/a')}",
        f"Synthetic Data: {diagnostics.data_context.get('synthetic_data', 'n/a')}",
        f"Used Cache Fallback: {diagnostics.data_context.get('used_cache_fallback', 'n/a')}",
        f"Latest Price Date: {diagnostics.data_context.get('latest_price_date', 'n/a')}",
        f"Expected Latest Trading Day: {diagnostics.data_context.get('expected_latest_trading_day', 'n/a')}",
        f"Tickers Loaded: {', '.join(map(str, diagnostics.data_context.get('tickers_loaded', []))) if diagnostics.data_context.get('tickers_loaded') else 'none'}",
        f"Tickers Failed: {', '.join(map(str, diagnostics.data_context.get('tickers_failed', []))) if diagnostics.data_context.get('tickers_failed') else 'none'}",
        f"Warnings: {len(diagnostics.warnings)}",
        f"Errors: {len(diagnostics.errors)}",
    ]
    text = "\n".join(lines) + "\n"
    _write_text(OUTPUT_DIR / "daily_bot_decision_report.txt", text)
    _write_text(OUTPUT_DIR / "latest_decision_report.txt", text)


def _finalize_daily_bot_diagnostics(diagnostics) -> None:
    try:
        detect_performance_flags(diagnostics)
    except Exception as exc:  # pragma: no cover - defensive
        log_warning(diagnostics, "daily_bot", f"detect_performance_flags failed: {exc}", stage="report_writing")

    try:
        if not (OUTPUT_DIR / "daily_bot_decision_report.txt").exists():
            _write_minimal_daily_bot_reports(diagnostics)
        if not (OUTPUT_DIR / "latest_decision_report.txt").exists():
            _write_minimal_daily_bot_reports(diagnostics)
    except Exception as exc:  # pragma: no cover - defensive
        log_warning(diagnostics, "daily_bot", f"minimal decision report fallback failed: {exc}", stage="report_writing")

    try:
        write_run_diagnostics(diagnostics, output_dir=OUTPUT_DIR)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Daily bot diagnostics JSON/CSV write failed: %s", exc)
    try:
        write_codex_debug_report(diagnostics, output_path=OUTPUT_DIR / "codex_daily_debug_report.md")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Daily bot Codex debug report write failed: %s", exc)
    try:
        write_codex_next_prompt(diagnostics, output_path=OUTPUT_DIR / "codex_next_prompt.md")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Daily bot Codex next prompt write failed: %s", exc)
    try:
        write_daily_analysis_report(diagnostics, output_path=OUTPUT_DIR / "daily_analysis_report.md")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Daily bot daily analysis report write failed: %s", exc)
    try:
        write_daily_portfolio_review_outputs(diagnostics, output_dir=OUTPUT_DIR)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Daily bot daily portfolio review write failed: %s", exc)

    email_result = {"sent": False, "reason": "not_attempted", "error": None}
    try:
        log_stage(diagnostics, "email sending", "START")
        review_email_settings = load_daily_review_settings()
        real_send_allowed, blocked_reasons = review_email_send_allowed(review_email_settings)
        if real_send_allowed:
            email_result = send_daily_review_email_if_needed(diagnostics, output_dir=OUTPUT_DIR, settings=review_email_settings)
        else:
            email_result = {
                "attempted": False,
                "sent": False,
                "reason": "preview_only",
                "error": None,
                "blocked_reasons": blocked_reasons,
            }
        log_stage(diagnostics, "email sending", "DONE", extra=email_result)
        if email_result.get("error"):
            log_warning(
                diagnostics,
                "daily_analysis_report",
                f"Daily analysis email send failed: {email_result.get('reason')}",
                stage="email_sending",
                extra=email_result,
            )
    except Exception as exc:  # pragma: no cover - defensive
        log_error(diagnostics, "daily_portfolio_review", "send_daily_review_email_if_needed", exc, stage="email_sending")

    try:
        write_run_diagnostics(diagnostics, output_dir=OUTPUT_DIR)
        write_codex_debug_report(diagnostics, output_path=OUTPUT_DIR / "codex_daily_debug_report.md")
        write_codex_next_prompt(diagnostics, output_path=OUTPUT_DIR / "codex_next_prompt.md")
        write_daily_analysis_report(diagnostics, output_path=OUTPUT_DIR / "daily_analysis_report.md")
        write_daily_portfolio_review_outputs(diagnostics, output_dir=OUTPUT_DIR, email_result=email_result)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Daily bot diagnostics rewrite after email step failed: %s", exc)


def _run_single(args: argparse.Namespace) -> dict[str, object]:
    diagnostics = create_run_diagnostics(mode="daily_bot")
    diagnostics.execution_mode = "order_preview_only"
    diagnostics.model_context["requested_mode"] = args.mode
    diagnostics.model_context["force_refresh"] = bool(args.force_refresh)
    result = {
        "as_of": None,
        "factor_mode": "fallback",
        "selected_candidate": "HOLD",
        "gate_action": "PAUSE",
        "execution_mode": "blocked",
        "message": "Run not completed.",
    }
    try:
        result = _run_single_impl(args, diagnostics)
    except Exception as exc:
        log_error(diagnostics, "daily_bot", "_run_single", exc, stage="run_single", extra={"args": vars(args)})
        log_final_action(
            diagnostics,
            "PAUSE",
            selected_candidate=diagnostics.selected_candidate or "HOLD",
            reason="Unhandled daily bot exception. Reports written fail-closed.",
        )
        diagnostics.final_orders_summary.setdefault("order_count", 0)
        diagnostics.final_orders_summary.setdefault("turnover", 0.0)
        diagnostics.final_orders_summary.setdefault("estimated_cost", 0.0)
        result = {
            "as_of": None,
            "factor_mode": diagnostics.model_context.get("factor_mode", "fallback"),
            "selected_candidate": diagnostics.selected_candidate or "HOLD",
            "gate_action": diagnostics.final_action or "PAUSE",
            "execution_mode": diagnostics.execution_mode or "blocked",
            "message": diagnostics.final_reason or str(exc),
        }
    finally:
        _finalize_daily_bot_diagnostics(diagnostics)
    return result


def _run_single_impl(args: argparse.Namespace, diagnostics) -> dict[str, object]:
    params = build_params()
    effective_dry_run = bool(args.dry_run or args.skip_submit or params.get("dry_run", True))
    diagnostics.dry_run = effective_dry_run
    params["dry_run"] = effective_dry_run
    log_stage(diagnostics, "system initialization", "START")
    if effective_dry_run:
        params["enable_investopedia_simulator"] = False
        params["enable_local_paper_trading"] = False
        params["enable_external_broker"] = False
    log_stage(diagnostics, "config validation", "START")
    config_check = validate_config(params)
    for warning in config_check["warnings"]:
        LOGGER.warning("Config validation warning: %s", warning)
        log_warning(diagnostics, "daily_bot", warning, stage="config_validation")
    if not config_check["ok"]:
        message = "Configuration validation failed: " + "; ".join(config_check["errors"])
        _write_text(OUTPUT_DIR / "daily_bot_decision_report.txt", message + "\n")
        for error in config_check["errors"]:
            log_warning(diagnostics, "daily_bot", error, severity="ERROR", stage="config_validation")
        log_final_action(diagnostics, "PAUSE", selected_candidate="HOLD", reason=message)
        diagnostics.execution_mode = "blocked"
        diagnostics.final_orders_summary = {"order_count": 0, "turnover": 0.0, "estimated_cost": 0.0}
        return {
            "as_of": None,
            "factor_mode": "fallback",
            "selected_candidate": "HOLD",
            "gate_action": "PAUSE",
            "execution_mode": "blocked",
            "message": message,
        }
    log_stage(diagnostics, "config validation", "DONE")
    init_result = run_system_initialization(params)
    for warning in init_result["warnings"]:
        LOGGER.warning("System initialization warning: %s", warning)
        log_warning(diagnostics, "daily_bot", warning, stage="system_initialization")
    for error in init_result["errors"]:
        LOGGER.warning("System initialization error: %s", error)
        log_warning(diagnostics, "daily_bot", error, severity="ERROR", stage="system_initialization")
    log_stage(diagnostics, "system initialization", "DONE", extra={"warnings": init_result["warnings"], "errors": init_result["errors"]})

    requested_tickers = list(params["tickers"])
    state_date = pd.Timestamp(datetime.now(timezone.utc).date())
    state_payload = reset_state_periods_if_needed(load_daily_bot_state(), state_date)
    turnover_budget_remaining = compute_turnover_budget_remaining(state_payload, params)
    diagnostics.model_context["state_context"] = {
        "current_date": state_payload.get("current_date"),
        "orders_today": state_payload.get("orders_today"),
        "turnover_today": state_payload.get("turnover_today"),
        "turnover_week": state_payload.get("turnover_week"),
        "turnover_month": state_payload.get("turnover_month"),
        "turnover_budget_remaining": turnover_budget_remaining,
    }
    log_stage(diagnostics, "data loading", "START")
    prices = load_price_data(
        tickers=_data_tickers(requested_tickers),
        start_date=str(params["start_date"]),
        end_date=params["end_date"],
        cache_path=PRICE_CACHE_PATH,
        use_cache=True,
        prefer_live=True,
        allow_cache_fallback=True,
        force_refresh=bool(args.force_refresh),
    )
    log_stage(diagnostics, "data loading", "DONE", extra={"rows": len(prices), "columns": len(prices.columns)})
    log_stage(diagnostics, "data freshness", "START")
    data_freshness = check_data_freshness(prices)
    market_gate = is_within_project_trading_window(calendar_path=DEFAULT_PROJECT_CALENDAR_PATH)
    data_context = build_run_data_context(
        prices=prices,
        freshness=data_freshness,
        market_gate=market_gate,
        calendar_path=DEFAULT_PROJECT_CALENDAR_PATH,
        run_context="daily_bot_discrete_simulator",
    ).as_dict()
    log_data_context(diagnostics, attrs=data_context)
    diagnostics.signal_date = str(data_context.get("latest_price_date") or "")
    diagnostics.execution_date = str(data_context.get("current_date_berlin") or "")
    write_data_freshness_report(
        prices=prices,
        freshness=data_freshness,
        output_path=OUTPUT_DIR / "current_data_freshness_report.txt",
        market_gate=market_gate,
        data_context=data_context,
    )
    if data_freshness.get("warning"):
        LOGGER.warning("%s", data_freshness["warning"])
        log_warning(diagnostics, "daily_bot", str(data_freshness["warning"]), stage="data_freshness")
    log_stage(diagnostics, "data freshness", "DONE", extra=data_freshness)
    log_stage(diagnostics, "tradability", "START")
    tradability_df = build_tradability_report(
        tickers=requested_tickers,
        prices=prices,
        enable_local_paper=bool(params.get("enable_local_paper_trading", False)),
        enable_investopedia=bool(params.get("enable_investopedia_simulator", False)),
        dry_run=effective_dry_run,
    )
    save_tradability_report(
        tradability_df=tradability_df,
        output_path=OUTPUT_DIR / "tradability_report.csv",
    )
    active_tickers = apply_tradability_filter(
        tickers=requested_tickers,
        tradability_df=tradability_df,
        min_assets=10,
    )
    log_stage(diagnostics, "tradability", "DONE", extra={"active_tickers": active_tickers})
    removed_tickers = [ticker for ticker in requested_tickers if ticker not in active_tickers]
    params = build_params(tickers=active_tickers)
    effective_cash_ticker = select_cash_proxy(active_tickers, tradability_df)
    if effective_cash_ticker is not None:
        params["cash_ticker"] = effective_cash_ticker
        params["effective_cash_ticker"] = effective_cash_ticker
    latest_prices_for_run = prices.reindex(columns=active_tickers).iloc[-1].astype(float)
    log_stage(diagnostics, "current portfolio loading", "START")
    current_state = load_current_portfolio_state(
        params=params,
        active_tickers=active_tickers,
        latest_prices=latest_prices_for_run,
        cash_proxy_ticker=effective_cash_ticker,
        nav=float(args.portfolio_value),
    )
    current_constraint_validation = validate_portfolio_constraints(
        weights_actual=current_state.current_weights_actual.reindex(active_tickers).fillna(0.0),
        cash_weight=float(current_state.actual_cash_weight),
        params=params,
        index=pd.Index(active_tickers, name="ticker"),
        label="Current portfolio",
    )
    write_current_portfolio_report(
        current_state,
        OUTPUT_DIR / "current_portfolio_report.txt",
        latest_prices=latest_prices_for_run,
        latest_price_date=data_context.get("latest_price_date", "n/a"),
        price_basis=str(data_context.get("price_basis", "adjusted_close_proxy")),
        data_source=str(data_context.get("data_source", "unknown")),
        data_freshness_ok=bool(data_context.get("data_freshness_ok", False)),
        constraint_validation=current_constraint_validation,
    )
    diagnostics.model_context["current_portfolio"] = {
        "source": current_state.source,
        "actual_cash_weight": float(current_state.actual_cash_weight),
        "actual_cash_value": float(current_state.current_cash),
        "cash_proxy_ticker": effective_cash_ticker,
        "nav": float(current_state.nav),
        "current_portfolio_constraint_valid": bool(current_constraint_validation["ok"]),
        "current_portfolio_constraint_errors": list(current_constraint_validation["errors"]),
    }
    log_stage(diagnostics, "current portfolio loading", "DONE", extra=diagnostics.model_context["current_portfolio"])
    quality_prices = prices.reindex(columns=active_tickers).copy()
    quality_prices.attrs.update(prices.attrs)
    log_stage(diagnostics, "data quality", "START")
    data_quality_report = compute_data_quality_report(
        prices=quality_prices,
        returns=compute_returns(quality_prices),
        active_tickers=active_tickers,
        params=params,
    )
    save_data_quality_report(
        report=data_quality_report,
        output_path=OUTPUT_DIR / "data_quality_report.csv",
    )
    log_data_quality(diagnostics, data_quality_report)
    log_stage(diagnostics, "data quality", "DONE", extra={"score": data_quality_report["global_data_quality_score"]})
    log_stage(diagnostics, "feasibility", "START")
    feasibility_result = check_portfolio_feasibility(active_tickers, params)
    log_stage(diagnostics, "feasibility", "DONE", extra=feasibility_result)
    log_stage(diagnostics, "regime detection", "START")
    regime_result = detect_regime(prices=prices, date=prices.index[-1])
    save_regime_report(
        regime_result=regime_result,
        csv_path=OUTPUT_DIR / "regime_report.csv",
        txt_path=OUTPUT_DIR / "regime_report.txt",
    )
    diagnostics.model_context["risk_state"] = str(regime_result.get("risk_state", "normal"))
    log_stage(diagnostics, "regime detection", "DONE", extra={"risk_state": diagnostics.model_context["risk_state"]})
    for warning in [*data_quality_report["warnings"], *feasibility_result["warnings"]]:
        LOGGER.warning("%s", warning)
        log_warning(diagnostics, "daily_bot", str(warning), stage="pre_optimizer_checks")
    if data_quality_report["errors"] or not feasibility_result["feasible"]:
        message = "; ".join([*data_quality_report["errors"], *feasibility_result["errors"]]) or "Daily bot paused by pre-optimizer safety checks."
        _write_text(OUTPUT_DIR / "daily_bot_decision_report.txt", message + "\n")
        for error in [*data_quality_report["errors"], *feasibility_result["errors"]]:
            log_warning(diagnostics, "daily_bot", str(error), severity="ERROR", stage="pre_optimizer_checks")
        log_final_action(diagnostics, "PAUSE", selected_candidate="HOLD", reason=message)
        diagnostics.execution_mode = "blocked"
        diagnostics.final_orders_summary = {"order_count": 0, "turnover": 0.0, "estimated_cost": 0.0}
        return {
            "as_of": None,
            "factor_mode": "fallback",
            "selected_candidate": "HOLD",
            "gate_action": "PAUSE",
            "execution_mode": "blocked",
            "message": message,
        }

    returns = compute_returns(prices.reindex(columns=active_tickers))
    as_of = pd.Timestamp(returns.index[-1])
    latest_prices_at_asof = prices.reindex(columns=active_tickers).loc[as_of]
    risk_state = str(regime_result.get("risk_state", "normal"))
    diagnostics.signal_date = str(as_of.date())
    diagnostics.model_context["risk_state"] = risk_state
    optimizer_constraint_params = {
        **params,
        "max_equity_like_total": params["max_equity_like_total_risk_off"] if risk_state == "risk_off" else params["max_equity_like_total_normal"],
        "min_defensive_weight": params["min_defensive_weight_risk_off"] if risk_state == "risk_off" else params["min_defensive_weight_normal"],
    }
    log_stage(diagnostics, "forecast", "START")
    forecast = build_forecast_3m(prices=prices, returns=returns, date=as_of, params=params, tickers=active_tickers)
    diagnostics.model_context["forecast_summary"] = {
        "asset_count": int(len(forecast.table)),
        "signal_confidence_mean": float(forecast.table["signal_confidence"].mean()),
    }
    log_stage(diagnostics, "forecast", "DONE", extra=diagnostics.model_context["forecast_summary"])
    log_stage(diagnostics, "covariance/risk", "START")
    sigma = estimate_robust_covariance_at_date(
        returns=returns,
        date=as_of,
        window=int(params["cov_window"]),
        horizon_days=int(params["horizon_days"]),
        alpha=float(params["cov_shrink_alpha"]),
        jitter=float(params["cov_jitter"]),
    )
    log_stage(diagnostics, "covariance/risk", "DONE", extra={"shape": list(sigma.shape)})

    w_current = current_state.current_weights_proxy.reindex(active_tickers).fillna(0.0).astype(float)
    log_stage(
        diagnostics,
        "legacy optimizer",
        "SKIPPED",
        extra={
            "reason": "Daily final allocation now uses scenario_weighted_rf_sharpe_solver only.",
            "used_for_final_target": False,
        },
    )
    legacy_optimizer_result = SimpleNamespace(
        success=False,
        solver_name="disabled_legacy_optimizer",
        message="Legacy optimizer skipped in active daily path.",
        objective_value=0.0,
    )
    optimizer_result = legacy_optimizer_result
    w_target = w_current.copy()

    log_stage(diagnostics, "scenario model", "START")
    direct_scenarios = build_3m_scenarios(
        forecast_table=forecast.table,
        covariance_3m=sigma,
        risk_state=risk_state,
        as_of=as_of,
        prices=prices,
        market_ticker="SPY",
    )
    log_stage(diagnostics, "scenario model", "DONE", extra={"scenario_count": len(direct_scenarios.scenario_names)})

    factor_mode = "direct_only"
    factor_diagnostics: list[str] = []
    factor_forecasts_df = pd.DataFrame()
    factor_data_df = pd.DataFrame()
    exposure_matrix = pd.DataFrame()
    conditional_summary = direct_scenarios.summary.copy()
    conditional_scenarios: ScenarioSet | None = None
    try:
        log_stage(diagnostics, "factor model", "START")
        macro_bundle = load_macro_proxy_data(prices=prices, date=as_of)
        factor_data_result = build_factor_data(macro_bundle.proxy_prices)
        factor_data_df = factor_data_result.factor_data.copy()
        factor_forecasts_df = build_factor_forecast(factor_data_df, date=as_of, risk_state=risk_state, horizon_days=int(params["horizon_days"]))
        exposure_result = estimate_asset_factor_exposures(
            asset_returns=returns.reindex(columns=active_tickers),
            factor_returns=factor_data_df,
            date=as_of,
        )
        exposure_matrix = exposure_result.exposure_matrix.copy()
        conditional_result = build_conditional_scenarios(
            direct_scenarios=direct_scenarios,
            factor_forecast_df=factor_forecasts_df,
            exposure_matrix=exposure_matrix,
            residual_volatility=exposure_result.residual_volatility,
        )
        factor_mode = conditional_result.mode
        factor_diagnostics = [*macro_bundle.diagnostics, *factor_data_result.diagnostics, *exposure_result.diagnostics, *conditional_result.diagnostics]
        conditional_summary = conditional_result.summary.copy()
        conditional_scenarios = ScenarioSet(
            as_of=as_of,
            scenario_returns_matrix=conditional_result.scenario_returns_matrix.copy(),
            scenario_names=list(conditional_result.scenario_probabilities.index),
            scenario_probabilities=conditional_result.scenario_probabilities.copy(),
            summary=conditional_result.summary.copy(),
            risk_state=risk_state,
        )
        diagnostics.model_context["factor_mode"] = factor_mode
        log_stage(diagnostics, "factor model", "DONE", extra={"factor_mode": factor_mode, "diagnostics": factor_diagnostics})
    except Exception as exc:
        factor_mode = "direct_only"
        LOGGER.warning(
            "Conditional factor mode unavailable; using direct-only fallback. Reason: %s",
            exc,
        )
        factor_diagnostics = [f"Conditional factor mode unavailable; using direct-only fallback. Reason: {exc}"]
        diagnostics.model_context["factor_mode"] = factor_mode
        log_warning(diagnostics, "daily_bot", factor_diagnostics[0], stage="factor_model")

    fallback_note = "Conditional factor mode unavailable; using direct-only fallback."
    if factor_mode == "direct_only" and not any(
        fallback_note in str(note) for note in factor_diagnostics
    ):
        factor_diagnostics = [fallback_note, *factor_diagnostics]

    scenario_set = conditional_scenarios if conditional_scenarios is not None else direct_scenarios
    scenario_risk_distribution = build_scenario_risk_distribution(
        forecast_table=forecast.table,
        returns=returns.reindex(columns=active_tickers),
        as_of=as_of,
        params=params,
        effective_horizon_days=int(params.get("effective_horizon_days", params.get("horizon_days", 63))),
    )
    reporting_config = params.get("reporting", {})
    if not isinstance(reporting_config, dict):
        reporting_config = {}
    write_scenario_risk_reports(
        distribution=scenario_risk_distribution,
        output_dir=OUTPUT_DIR,
        write_pairwise_relationships=bool(reporting_config.get("write_pairwise_relationships", True)),
        write_scenario_matrices=bool(reporting_config.get("write_scenario_matrices", True)),
    )
    execution_fraction_value = _execution_fraction(params)
    if not bool(params.get("direct_scenario_optimizer_enabled", True)):
        log_warning(
            diagnostics,
            "daily_bot",
            "direct_scenario_optimizer_enabled=false is deprecated for daily final allocation; "
            "the scenario-weighted RF Sharpe solver remains the mandatory target source.",
            stage="scenario_weighted_solver",
        )
    log_stage(diagnostics, "scenario-weighted RF Sharpe optimizer", "START")
    daily_solve = run_scenario_weighted_daily_solve(
        forecast_table=forecast.table,
        returns=returns.reindex(columns=active_tickers),
        current_weights=w_current,
        active_tickers=active_tickers,
        params=params,
        optimizer_constraint_params=optimizer_constraint_params,
        effective_cash_ticker=effective_cash_ticker,
        execution_fraction=execution_fraction_value,
        prices=prices.reindex(columns=active_tickers),
        market_ticker=str(params.get("market_ticker") or "SPY"),
        success_target_source=FINAL_TARGET_SOURCE_SCENARIO,
        failure_target_source=FINAL_TARGET_SOURCE_SOLVER_FAILED,
    )
    for warning in daily_solve.warnings:
        log_warning(
            diagnostics,
            "daily_bot",
            str(warning),
            severity="ERROR" if "failed" in str(warning).lower() else "WARNING",
            stage="scenario_weighted_solver",
        )
    final_target_source = daily_solve.final_target_source
    scenario_inputs = daily_solve.scenario_inputs
    scenario_solver_result = daily_solve.solver_result
    solver_validation = daily_solve.solver_validation
    solver_current_weights = daily_solve.solver_current_weights
    solver_assets = daily_solve.solver_assets
    solver_config = daily_solve.solver_config
    optimal_solver_weights = daily_solve.optimal_weights
    executable_solver_weights = daily_solve.executable_weights
    w_target = daily_solve.target_weights
    optimizer_result = _optimizer_adapter_from_solver(
        scenario_solver_result,
        target_weights=w_target,
    )
    direct_optimizer_context = {
        "final_allocation_method": "scenario_weighted_rf_sharpe_solver",
        "final_target_source": final_target_source,
        "optimizer_success": bool(getattr(optimizer_result, "success", False)),
        "optimizer_solver": str(getattr(optimizer_result, "solver_name", "")),
        "optimizer_status": str(getattr(optimizer_result, "status", "")),
        "objective_value": float(getattr(optimizer_result, "objective_value", 0.0) or 0.0),
        "solver_failed": final_target_source == FINAL_TARGET_SOURCE_SOLVER_FAILED,
        "failure_reason": str(scenario_solver_result.constraint_diagnostics.get("failure_reason") or "none"),
        "post_solver_validation_ok": bool(solver_validation.get("ok", False)),
        "post_solver_validation_errors": list(solver_validation.get("errors", [])),
        "post_solver_validation_warnings": list(solver_validation.get("warnings", [])),
        "weighted_rf_adjusted_sharpe": float(scenario_solver_result.weighted_sharpe),
        "solver_turnover": float(scenario_solver_result.turnover),
        "execution_fraction": float(execution_fraction_value),
        "executable_turnover": float((executable_solver_weights - w_current.reindex(executable_solver_weights.index).fillna(0.0)).abs().sum()),
        "solver_concentration": float(scenario_solver_result.concentration),
        "solver_downside_penalty": float(scenario_solver_result.downside_penalty),
        "legacy_optimizer_solver": str(getattr(legacy_optimizer_result, "solver_name", "")),
        "legacy_optimizer_objective_value": float(getattr(legacy_optimizer_result, "objective_value", 0.0) or 0.0),
    }
    diagnostics.model_context["direct_scenario_optimizer"] = direct_optimizer_context
    log_optimizer_result(diagnostics, direct_optimizer_context)
    if not bool(getattr(optimizer_result, "success", False)):
        log_warning(
            diagnostics,
            "daily_bot",
            "Scenario-weighted RF Sharpe optimizer returned a fallback target; execution gates remain fail-closed.",
            severity="WARNING",
            stage="scenario_weighted_solver",
        )
    allocation_report = pd.DataFrame(
        {
            "ticker": w_target.index.astype(str),
            "scenario_weighted_rf_sharpe_optimal_weight": optimal_solver_weights.reindex(w_target.index).fillna(0.0).to_numpy(dtype=float),
            "scenario_weighted_rf_sharpe_executable_weight": w_target.to_numpy(dtype=float),
            "current_weight": w_current.reindex(w_target.index).fillna(0.0).to_numpy(dtype=float),
            "delta_optimal_weight": (
                optimal_solver_weights.reindex(w_target.index).fillna(0.0)
                - w_current.reindex(w_target.index).fillna(0.0)
            ).to_numpy(dtype=float),
            "delta_executable_weight": (
                w_target - w_current.reindex(w_target.index).fillna(0.0)
            ).to_numpy(dtype=float),
            "execution_fraction": float(execution_fraction_value),
            "final_target_source": final_target_source,
        }
    )
    _write_csv(OUTPUT_DIR / "direct_scenario_optimizer_allocation.csv", allocation_report, index=False)
    _write_scenario_weighted_solver_reports(
        scenarios=scenario_inputs,
        result=scenario_solver_result,
        current_weights=solver_current_weights,
        target_weights=optimal_solver_weights.reindex(solver_assets).fillna(0.0)
        if scenario_solver_result.success
        else solver_current_weights,
        executable_weights=executable_solver_weights.reindex(solver_assets).fillna(0.0)
        if scenario_solver_result.success
        else solver_current_weights,
        execution_fraction=execution_fraction_value,
        output_dir=OUTPUT_DIR,
        final_target_source=final_target_source,
    )
    _write_text(
        OUTPUT_DIR / "direct_scenario_optimizer_report.txt",
        "\n".join(
            [
                "Scenario-Weighted RF-Adjusted Sharpe Optimizer Report",
                "",
                "final_allocation_method: scenario_weighted_rf_sharpe_solver",
                f"final_target_source: {final_target_source}",
                "manual_candidate_selection_for_final_target: false",
                "hold_current_role: benchmark_and_execution_fallback_only",
                f"solver_objective: {params.get('scenario_solver_objective', 'scenario_weighted_rf_sharpe')}",
                f"solver_horizon_days: {params.get('horizon_days')}",
                f"risk_free_rate_annual: {params.get('risk_free_rate_annual')}",
                f"eps_variance: {params.get('eps_variance')}",
                f"lambda_turnover: {params.get('direct_scenario_lambda_turnover')}",
                f"lambda_concentration: {params.get('direct_scenario_lambda_concentration')}",
                f"lambda_downside: {params.get('direct_scenario_lambda_downside')}",
                f"max_turnover: {params.get('max_turnover')}",
                f"optimizer_method: {params.get('direct_scenario_optimizer_method')}",
                f"optimizer_ftol: {params.get('direct_scenario_optimizer_ftol')}",
                f"optimizer_maxiter: {params.get('direct_scenario_optimizer_maxiter')}",
                f"use_scenario_covariance: {params.get('use_scenario_covariance')}",
                f"use_scenario_probabilities: {params.get('use_scenario_probabilities')}",
                f"use_rf_adjusted_sharpe: {params.get('use_rf_adjusted_sharpe')}",
                f"solver: {getattr(optimizer_result, 'solver_name', 'unknown')}",
                f"success: {bool(getattr(optimizer_result, 'success', False))}",
                f"solver_failed: {final_target_source == FINAL_TARGET_SOURCE_SOLVER_FAILED}",
                f"failure_reason: {scenario_solver_result.constraint_diagnostics.get('failure_reason') or 'none'}",
                f"post_solver_validation_ok: {bool(solver_validation.get('ok', False))}",
                f"post_solver_validation_errors: {'; '.join(map(str, solver_validation.get('errors', []))) if solver_validation.get('errors') else 'none'}",
                f"status: {getattr(optimizer_result, 'status', 'unknown')}",
                f"message: {getattr(optimizer_result, 'message', '')}",
                f"objective_value: {_safe_float(getattr(optimizer_result, 'objective_value', float('nan')), float('nan')):.8f}",
                f"weighted_rf_adjusted_sharpe: {float(scenario_solver_result.weighted_sharpe):.8f}",
                f"turnover_vs_current: {float(scenario_solver_result.turnover):.6f}",
                f"execution_fraction: {float(execution_fraction_value):.4f}",
                f"executable_turnover_vs_current: {float((executable_solver_weights - w_current.reindex(executable_solver_weights.index).fillna(0.0)).abs().sum()):.6f}",
                f"scenario_count: {len(scenario_inputs) if scenario_inputs else len(scenario_risk_distribution.scenario_names)}",
                f"scenario_names: {', '.join([scenario.name for scenario in scenario_inputs]) if scenario_inputs else ', '.join(scenario_risk_distribution.scenario_names)}",
                "",
                "Objective:",
                "sum_s p_s * ((w^T mu_s - rf_s) / sqrt(w^T Sigma_s w + eps))",
                "- lambda_turnover * ||w - w_current||_1",
                "- lambda_concentration * sum_i w_i^2",
                "- lambda_downside * sum_s p_s * max(0, rf_s - w^T mu_s)",
                "",
                "Constraints are enforced before Whole-Share discretization:",
                "sum weights = 1; long-only; asset caps; group caps; max turnover; defensive/cash limits.",
            ]
        )
        + "\n",
    )
    log_stage(diagnostics, "scenario-weighted RF Sharpe optimizer", "DONE", extra=direct_optimizer_context)
    diagnostics.model_context["scenario_risk_model"] = {
        "scenario_count": len(scenario_risk_distribution.scenario_names),
        "scenario_names": scenario_risk_distribution.scenario_names,
        "uses_scenario_dependent_covariance": True,
        "optimization_objective": str(params.get("optimization_objective", "robust_score")),
        "warnings": list(scenario_risk_distribution.warnings),
    }
    return _finalize_slim_scenario_daily_run(
        args=args,
        diagnostics=diagnostics,
        params=params,
        effective_dry_run=effective_dry_run,
        as_of=as_of,
        active_tickers=active_tickers,
        latest_prices_at_asof=latest_prices_at_asof,
        current_state=current_state,
        data_context=data_context,
        data_freshness=data_freshness,
        market_gate=market_gate,
        final_target_source=final_target_source,
        scenario_inputs=scenario_inputs,
        solver_config=solver_config,
        solver_current_weights=solver_current_weights,
        scenario_solver_result=scenario_solver_result,
        solver_validation=solver_validation,
        optimal_solver_weights=optimal_solver_weights,
        executable_solver_weights=executable_solver_weights,
        execution_fraction_value=execution_fraction_value,
    )

    raise RuntimeError(
        "Legacy candidate/discrete selection is disabled for the final daily allocation. "
        "Final target source must remain SCENARIO_WEIGHTED_RF_SHARPE_OPTIMAL or HOLD_SOLVER_FAILED."
    )

    conditional_factor_target = w_target if factor_mode == "conditional_factor" else None
    log_stage(diagnostics, "candidate construction", "START")
    candidates = build_candidate_portfolios(
        w_current=w_current,
        w_target=w_target,
        forecast_table=forecast.table,
        params=params,
        conditional_factor_target=conditional_factor_target,
    )
    log_stage(diagnostics, "candidate construction", "DONE", extra={"candidates": list(candidates.keys())})
    ensemble_result = build_model_ensemble_outputs(
        optimizer_target=candidates["OPTIMIZER_TARGET"].weights,
        defensive_cash=candidates["DEFENSIVE_CASH"].weights,
        momentum_tilt=candidates["MOMENTUM_TILT_SIMPLE"].weights,
        conditional_factor_target=candidates.get("CONDITIONAL_FACTOR_TARGET").weights if "CONDITIONAL_FACTOR_TARGET" in candidates else None,
    )
    save_model_ensemble_report(
        ensemble_result,
        csv_path=OUTPUT_DIR / "model_ensemble_report.csv",
        txt_path=OUTPUT_DIR / "model_ensemble_report.txt",
    )
    log_stage(diagnostics, "robust scoring", "START")
    selection = select_robust_candidate(
        candidates=candidates,
        scenario_set=scenario_set,
        w_current=w_current,
        params=params,
        mode=factor_mode,
    )
    direct_final_allocation = True
    diagnostics.candidate_context["continuous_selected_candidate"] = selection.selected_candidate.name
    diagnostics.candidate_context["continuous_selected_score"] = float(selection.selected_score.net_robust_score)
    log_stage(diagnostics, "robust scoring", "DONE", extra=diagnostics.candidate_context)
    if direct_final_allocation and FINAL_TARGET_SOURCE_SCENARIO in candidates:
        continuous_model_optimal_candidate = FINAL_TARGET_SOURCE_SCENARIO
        direct_score_rows = selection.scores_frame.loc[
            selection.scores_frame["candidate"].astype(str) == continuous_model_optimal_candidate
        ]
        continuous_model_optimal_score = (
            float(direct_score_rows.iloc[0]["net_robust_score"])
            if not direct_score_rows.empty
            else float("nan")
        )
        selection.scores_frame["final_allocation_candidate"] = (
            selection.scores_frame["candidate"].astype(str) == continuous_model_optimal_candidate
        )
        selection.scores_frame["manual_candidate_selection_for_final_target"] = False
        selection.scores_frame["final_allocation_method"] = "scenario_weighted_rf_sharpe_solver"
        selection.scores_frame["final_target_source"] = final_target_source
        selection.scores_frame["candidate_role"] = np.where(
            selection.scores_frame["candidate"].astype(str) == continuous_model_optimal_candidate,
            "final_direct_solver_target",
            "diagnostic_benchmark_only",
        )
    else:
        continuous_model_optimal_candidate = str(selection.scores_frame.iloc[0]["candidate"])
        continuous_model_optimal_score = float(selection.scores_frame.iloc[0]["net_robust_score"])
        selection.scores_frame["final_allocation_candidate"] = (
            selection.scores_frame["candidate"].astype(str) == continuous_model_optimal_candidate
        )
        selection.scores_frame["manual_candidate_selection_for_final_target"] = True
        selection.scores_frame["final_allocation_method"] = "legacy_candidate_selection"
        selection.scores_frame["candidate_role"] = np.where(
            selection.scores_frame["candidate"].astype(str) == continuous_model_optimal_candidate,
            "legacy_selected_candidate",
            "diagnostic_candidate",
        )
    continuous_target_weights = candidates[continuous_model_optimal_candidate].weights.reindex(active_tickers).fillna(0.0)
    expansion_sources = (
        [continuous_model_optimal_candidate]
        if direct_final_allocation
        else _select_discrete_expansion_sources(
            selection.scores_frame,
            continuous_model_optimal_candidate,
        )
    )

    discrete_candidates = []
    hold_current_added = False
    log_stage(diagnostics, "discrete candidate selection", "START")
    for source_name in expansion_sources:
        source_target_weights = candidates[source_name].weights.reindex(active_tickers).fillna(0.0)
        strict_discrete_constraints = (
            params
            if str(source_name)
            in {
                FINAL_TARGET_SOURCE_SCENARIO,
                "DIRECT_SCENARIO_OPTIMIZER",
                "MOMENTUM_TILT_REPAIRED",
                "MOMENTUM_TILT_CAP_AWARE",
                "FACTOR_TARGET_REPAIRED",
                "MIN_TURNOVER_ACTIVE_REPAIR",
                "CURRENT_COMPLIANCE_REPAIR",
            }
            else None
        )
        generated = generate_discrete_candidates(
            target_weights=source_target_weights,
            latest_prices=latest_prices_at_asof,
            nav=float(current_state.nav),
            current_positions=current_state.current_shares.reindex(active_tickers).fillna(0.0),
            current_cash=float(current_state.current_cash),
            min_order_value=float(params.get("min_order_value_usd", 10.0)),
            cash_buffer=float(params.get("cash_buffer_usd", 0.0)),
            max_candidates=25,
            allow_fractional_shares=bool(params.get("allow_fractional_shares", False)),
            marginal_priority=forecast.table["expected_return_3m"] * forecast.table["signal_confidence"],
            cash_proxy_ticker=effective_cash_ticker,
            constraint_params=strict_discrete_constraints,
        )
        for candidate in generated:
            if candidate.name == "HOLD_CURRENT":
                if hold_current_added:
                    continue
                hold_current_added = True
                candidate.metadata["continuous_source"] = "HOLD"
                candidate.metadata["continuous_target_weights"] = candidates["HOLD"].weights.reindex(active_tickers).fillna(0.0).copy()
            else:
                candidate.name = f"{source_name}::{candidate.name}"
                candidate.metadata["continuous_source"] = source_name
                candidate.metadata["continuous_target_weights"] = source_target_weights.copy()
            discrete_candidates.append(candidate)
    discrete_scored = score_discrete_candidates(
        discrete_candidates=discrete_candidates,
        scenario_returns=scenario_set,
        scorer_config={
            "params": params,
            "hold_weights": candidates["HOLD"].weights,
            "cash_weights": candidates["DEFENSIVE_CASH"].weights,
            "continuous_target": continuous_target_weights,
        },
        current_weights=w_current,
        current_shares=current_state.current_shares.reindex(active_tickers).fillna(0.0),
        current_cash=float(current_state.current_cash),
        latest_prices=latest_prices_at_asof,
        nav=float(current_state.nav),
    )
    discrete_candidate_weights = {
        name: candidate.weights_proxy.reindex(active_tickers).fillna(0.0)
        for name, candidate in discrete_scored["candidate_map"].items()
    }
    pre_selection_risk_return = build_candidate_risk_return_frame(
        candidate_weights=discrete_candidate_weights,
        distribution=scenario_risk_distribution,
        current_weights=w_current,
        defensive_cash_weights=candidates["DEFENSIVE_CASH"].weights.reindex(active_tickers).fillna(0.0),
        hold_weights=candidates["HOLD"].weights.reindex(active_tickers).fillna(0.0),
        params=params,
        scores_frame=discrete_scored["scores_frame"],
    )
    scenario_metric_columns = [
        "candidate",
        "mixture_expected_return",
        "defensive_cash_mixture_return",
        "excess_return_vs_defensive_cash",
        "excess_return_vs_current",
        "within_scenario_variance",
        "between_scenario_variance",
        "mixture_variance",
        "mixture_volatility",
        "scenario_mixture_sharpe",
        "sortino_like_ratio",
        "probability_weighted_var",
        "probability_weighted_cvar",
        "concentration_hhi",
        "average_pairwise_correlation_weighted",
        "diversification_ratio",
        "tail_penalty",
        "turnover_penalty",
        "cost_penalty",
        "concentration_penalty",
        "robust_scenario_sharpe_objective",
    ]
    discrete_scored["scores_frame"] = discrete_scored["scores_frame"].merge(
        pre_selection_risk_return[scenario_metric_columns].rename(columns={"candidate": "discrete_candidate"}),
        on="discrete_candidate",
        how="left",
    )
    diagnostic_discrete_selection = select_best_discrete_portfolio(discrete_scored)
    if direct_final_allocation:
        discrete_selection = _force_solver_discrete_selection(
            discrete_scored,
            solver_source_name=continuous_model_optimal_candidate,
            final_target_source=final_target_source,
            solver_failed=final_target_source == FINAL_TARGET_SOURCE_SOLVER_FAILED,
        )
        diagnostics.model_context["legacy_discrete_selection_diagnostic"] = {
            "candidate": str(diagnostic_discrete_selection.get("best_discrete_candidate_name", "")),
            "reason": str(diagnostic_discrete_selection.get("reason", "")),
            "used_for_final_target": False,
        }
    else:
        discrete_selection = diagnostic_discrete_selection
    selected_reason = str(discrete_selection.get("selected_reason", ""))
    current_constraint_errors_text = " | ".join(map(str, current_constraint_validation.get("errors", [])))
    if not bool(current_constraint_validation.get("ok", True)) and discrete_selection["best_discrete_candidate_name"] == "HOLD_CURRENT":
        selected_reason = "constraint_invalid_hold_fallback"
        discrete_selection["selected_reason"] = selected_reason
        discrete_selection["hold_current_constraint_valid"] = False
        discrete_selection["current_portfolio_constraint_violation"] = True
        discrete_selection["current_constraint_errors"] = current_constraint_errors_text
        discrete_selection["final_selection_is_safe_fallback"] = True
        discrete_selection["scores_frame"]["selected_reason"] = selected_reason
        discrete_selection["scores_frame"]["hold_current_constraint_valid"] = False
        discrete_selection["scores_frame"]["current_portfolio_constraint_violation"] = True
        discrete_selection["scores_frame"]["current_constraint_errors"] = current_constraint_errors_text
        discrete_selection["scores_frame"]["final_selection_is_safe_fallback"] = True
    candidate_risk_return_frame = build_candidate_risk_return_frame(
        candidate_weights=discrete_candidate_weights,
        distribution=scenario_risk_distribution,
        current_weights=w_current,
        defensive_cash_weights=candidates["DEFENSIVE_CASH"].weights.reindex(active_tickers).fillna(0.0),
        hold_weights=candidates["HOLD"].weights.reindex(active_tickers).fillna(0.0),
        params=params,
        scores_frame=discrete_selection["scores_frame"],
        selected_name=str(discrete_selection["best_discrete_candidate_name"]),
        selected_reason=selected_reason,
    )
    write_candidate_risk_return_reports(
        risk_return_frame=candidate_risk_return_frame,
        selected_candidate=str(discrete_selection["best_discrete_candidate_name"]),
        objective_used=str(discrete_selection.get("objective_used", params.get("optimization_objective", "robust_score"))),
        output_dir=OUTPUT_DIR,
    )
    scenario_attribution = build_scenario_attribution_frame(
        candidate_weights=discrete_candidate_weights,
        distribution=scenario_risk_distribution,
        hold_weights=candidates["HOLD"].weights.reindex(active_tickers).fillna(0.0),
        defensive_cash_weights=candidates["DEFENSIVE_CASH"].weights.reindex(active_tickers).fillna(0.0),
    )
    _write_csv(OUTPUT_DIR / "scenario_attribution_by_candidate.csv", scenario_attribution, index=False)
    diagnostics.model_context["scenario_mixture_objective"] = {
        "objective_used": str(discrete_selection.get("objective_used", params.get("optimization_objective", "robust_score"))),
        "objective_score_column": str(discrete_selection.get("objective_score_column", "net_robust_score")),
        "highest_objective_candidate": (
            str(candidate_risk_return_frame.iloc[0]["candidate"]) if not candidate_risk_return_frame.empty else ""
        ),
        "selected_candidate": str(discrete_selection["best_discrete_candidate_name"]),
    }
    log_candidate_selection(
        diagnostics,
        {
            "best_discrete_candidate_name": discrete_selection["best_discrete_candidate_name"],
            "best_discrete_score": float(discrete_selection["best_discrete_score"]),
            "best_discrete_objective_score": float(discrete_selection.get("best_discrete_objective_score", discrete_selection["best_discrete_score"])),
            "objective_used": str(discrete_selection.get("objective_used", params.get("optimization_objective", "robust_score"))),
            "reason": discrete_selection["reason"],
            "selected_reason": selected_reason,
            "scores_frame_head": discrete_selection["scores_frame"].head(10),
        },
    )
    log_stage(diagnostics, "discrete candidate selection", "DONE", extra={"selected": diagnostics.selected_candidate})
    best_discrete_candidate = discrete_selection["candidate"]
    best_discrete_candidate_name = str(discrete_selection["best_discrete_candidate_name"])
    best_discrete_source = str(best_discrete_candidate.metadata.get("continuous_source", continuous_model_optimal_candidate))
    best_continuous_for_discrete = candidates.get(best_discrete_source, candidates[continuous_model_optimal_candidate]).weights.reindex(active_tickers).fillna(0.0)
    final_weights = discrete_selection["best_discrete_weights"].reindex(active_tickers).fillna(0.0)
    final_target_shares = best_discrete_candidate.shares.reindex(active_tickers).fillna(0.0)
    discrete_selected_score = discrete_scored["score_objects"][best_discrete_candidate_name]
    hold_current_row = discrete_selection["scores_frame"].loc[
        discrete_selection["scores_frame"]["discrete_candidate"].astype(str) == "HOLD_CURRENT"
    ]
    current_portfolio_score = (
        float(hold_current_row.iloc[0]["net_robust_score"])
        if not hold_current_row.empty
        else 0.0
    )
    current_tail_risk = (
        float(hold_current_row.iloc[0]["cvar_5"])
        if not hold_current_row.empty
        else float(getattr(discrete_selected_score, "cvar_5", 0.0))
    )
    delta_score = float(discrete_selection["best_discrete_score"]) - current_portfolio_score
    final_selection = type(
        "SelectionLike",
        (),
        {
            "selected_candidate": type(
                "DiscreteCandidateLike",
                (),
                {"name": best_discrete_candidate_name, "weights": final_weights},
            )(),
            "selected_score": discrete_selected_score,
        },
    )()
    model_governance = compute_model_confidence(
        forecast_report=forecast,
        factor_report=factor_forecasts_df,
        scenario_report=scenario_set.summary,
        optimizer_result=optimizer_result,
        data_quality_report=data_quality_report,
    )
    save_model_governance_report(
        model_governance,
        csv_path=OUTPUT_DIR / "model_governance_report.csv",
        txt_path=OUTPUT_DIR / "model_governance_report.txt",
    )
    diagnostics.model_context["model_confidence_score"] = float(model_governance.get("model_confidence_score", 0.0))
    adapter_or_stub = None
    account_summary = None
    broker_positions = None
    execution_mode_hint = _expected_execution_mode(params)
    if execution_mode_hint == "local_paper":
        try:
            adapter_or_stub = PaperBrokerStub(
                db_path=params["db_path"],
                initial_cash=float(args.portfolio_value),
            )
            account_summary = adapter_or_stub.get_account_summary()
            broker_positions = adapter_or_stub.get_positions()
        except Exception as exc:
            LOGGER.warning("Local paper state could not be loaded for daily-bot validation: %s", exc)
            log_warning(diagnostics, "daily_bot", f"Local paper state could not be loaded: {exc}", stage="current portfolio loading")
    log_stage(diagnostics, "reconciliation", "START")
    reconciliation_result = reconcile_before_execution(
        model_weights=final_weights,
        latest_prices=prices.reindex(columns=active_tickers).loc[as_of],
        execution_mode=execution_mode_hint,
        broker_positions=broker_positions,
        broker_cash=float(account_summary["cash"]) if account_summary is not None else None,
        adapter_or_stub=adapter_or_stub,
    )
    build_reconciliation_report(
        reconciliation_result=reconciliation_result,
        output_path=OUTPUT_DIR / "reconciliation_report.csv",
    )
    diagnostics.model_context["reconciliation_status"] = reconciliation_result.get("status", "SKIP")
    if reconciliation_result.get("errors"):
        for error in reconciliation_result["errors"]:
            log_warning(diagnostics, "daily_bot", str(error), severity="ERROR", stage="reconciliation")
    log_stage(diagnostics, "reconciliation", "DONE", extra={"status": reconciliation_result.get("status", "SKIP")})

    log_stage(diagnostics, "final order generation", "START")
    order_preview = build_discrete_order_preview(
        current_shares=current_state.current_shares.reindex(active_tickers).fillna(0.0),
        target_shares=final_target_shares,
        latest_prices=latest_prices_at_asof,
        nav=float(current_state.nav),
        min_order_value=float(params.get("min_order_value_usd", 10.0)),
        not_executable=not bool(market_gate.get("execution_allowed", False)),
        reason=(
            f"calendar:{market_gate.get('reason', 'calendar_blocked')}"
            if not bool(market_gate.get("execution_allowed", False))
            else ""
        ),
    )
    order_preview, order_cost_summary = estimate_order_list_costs(
        order_preview_df=order_preview,
        latest_prices=latest_prices_at_asof,
        config={
            **params,
            "nav": float(current_state.nav),
            "current_cash": float(current_state.current_cash),
        },
    )
    _write_csv(OUTPUT_DIR / "best_discrete_order_preview.csv", mark_daily_simulator_preview(order_preview), index=False)
    log_stage(diagnostics, "pre-trade validation", "START")
    no_trade_hold_current = best_discrete_candidate_name == "HOLD_CURRENT" and bool((order_preview["side"] == "HOLD").all())
    if no_trade_hold_current:
        validation_result = {
            "ok": True,
            "warnings": ["Current portfolio already matches the selected HOLD_CURRENT discrete candidate; no rebalance orders were generated."],
            "errors": [],
            "adjusted_order_preview": order_preview.copy(),
            "blocked_orders": pd.DataFrame(columns=order_preview.columns.tolist()),
            "validation_report": pd.DataFrame(
                [
                    {"check": "weights", "status": "PASS", "message": "No-trade HOLD_CURRENT path."},
                    {"check": "prices", "status": "PASS", "message": "No-trade HOLD_CURRENT path."},
                    {"check": "order_preview", "status": "PASS", "message": "No-trade HOLD_CURRENT path."},
                    {"check": "cash_positions", "status": "PASS", "message": "No-trade HOLD_CURRENT path."},
                ]
            ),
        }
    else:
        validation_result = run_pre_trade_validation(
            w_current=current_state.current_weights_actual.reindex(active_tickers).fillna(0.0),
            w_target=best_discrete_candidate.weights_actual.reindex(active_tickers).fillna(0.0),
            latest_prices=latest_prices_at_asof,
            order_preview_df=order_preview,
            params={
                **params,
                "dry_run": effective_dry_run,
                "__current_cash_weight__": float(current_state.actual_cash_weight),
                "__target_cash_weight__": float(best_discrete_candidate.cash_weight),
                "blocked_tickers": tradability_df.loc[~tradability_df["final_allowed"], "ticker"].astype(str).tolist(),
                "max_equity_like_total": params["max_equity_like_total_risk_off"] if risk_state == "risk_off" else params["max_equity_like_total_normal"],
                "min_defensive_weight": params["min_defensive_weight_risk_off"] if risk_state == "risk_off" else params["min_defensive_weight_normal"],
            },
            account_summary=account_summary,
            positions=broker_positions,
        )
    save_pre_trade_validation_report(
        validation_report=validation_result["validation_report"],
        output_path=OUTPUT_DIR / "pre_trade_validation_report.csv",
    )
    if validation_result["warnings"]:
        for warning in validation_result["warnings"]:
            log_warning(diagnostics, "daily_bot", str(warning), stage="pre_trade_validation")
    if validation_result["errors"]:
        for error in validation_result["errors"]:
            log_warning(diagnostics, "daily_bot", str(error), severity="ERROR", stage="pre_trade_validation")
    log_stage(diagnostics, "pre-trade validation", "DONE", extra={"ok": validation_result["ok"]})
    adjusted_order_preview = validation_result["adjusted_order_preview"].copy()
    adjusted_order_preview, final_order_cost_summary = estimate_order_list_costs(
        order_preview_df=adjusted_order_preview,
        latest_prices=latest_prices_at_asof,
        config={
            **params,
            "nav": float(current_state.nav),
            "current_cash": float(current_state.current_cash),
        },
    )
    validation_result["adjusted_order_preview"] = adjusted_order_preview
    discrete_selected_score.estimated_cost = float(final_order_cost_summary["total_order_cost_pct_nav"])
    discrete_selected_score.estimated_commission = float(final_order_cost_summary["total_estimated_commission"])
    discrete_selected_score.estimated_spread_cost = float(final_order_cost_summary["total_estimated_spread_cost"])
    discrete_selected_score.estimated_slippage_cost = float(final_order_cost_summary["total_estimated_slippage_cost"])
    discrete_selected_score.estimated_market_impact_cost = float(final_order_cost_summary["total_estimated_market_impact_cost"])
    discrete_selected_score.estimated_total_order_cost = float(final_order_cost_summary["total_estimated_transaction_cost"])
    discrete_selected_score.cost_bps_used = float(final_order_cost_summary["weighted_average_cost_bps"])
    discrete_selected_score.cost_model_used = str(final_order_cost_summary["cost_model_used"])
    discrete_selected_score.live_costs_available = bool(final_order_cost_summary["live_costs_available"])
    target_score_before_costs = float(discrete_selected_score.robust_score - discrete_selected_score.dynamic_buffer)
    target_score_after_costs = float(target_score_before_costs - discrete_selected_score.estimated_cost)
    discrete_selected_score.net_robust_score = (
        discrete_selected_score.robust_score
        - discrete_selected_score.estimated_cost
        - discrete_selected_score.dynamic_buffer
    )
    target_score_after_costs = float(discrete_selected_score.net_robust_score)
    delta_score = float(discrete_selected_score.net_robust_score) - float(current_portfolio_score)
    discrete_selected_score.delta_vs_hold = float(delta_score)
    discrete_selection["best_discrete_score"] = float(discrete_selected_score.net_robust_score)
    discrete_scored["score_objects"][best_discrete_candidate_name] = discrete_selected_score
    selected_mask = discrete_selection["scores_frame"]["discrete_candidate"].astype(str) == best_discrete_candidate_name
    discrete_selection["scores_frame"].loc[selected_mask, "estimated_cost"] = float(discrete_selected_score.estimated_cost)
    discrete_selection["scores_frame"].loc[selected_mask, "estimated_transaction_cost"] = float(discrete_selected_score.estimated_cost)
    discrete_selection["scores_frame"].loc[selected_mask, "estimated_commission"] = float(discrete_selected_score.estimated_commission)
    discrete_selection["scores_frame"].loc[selected_mask, "estimated_spread_cost"] = float(discrete_selected_score.estimated_spread_cost)
    discrete_selection["scores_frame"].loc[selected_mask, "estimated_slippage_cost"] = float(discrete_selected_score.estimated_slippage_cost)
    discrete_selection["scores_frame"].loc[selected_mask, "estimated_market_impact_cost"] = float(discrete_selected_score.estimated_market_impact_cost)
    discrete_selection["scores_frame"].loc[selected_mask, "total_estimated_transaction_cost"] = float(discrete_selected_score.estimated_total_order_cost)
    discrete_selection["scores_frame"].loc[selected_mask, "cost_bps_used"] = float(discrete_selected_score.cost_bps_used)
    discrete_selection["scores_frame"].loc[selected_mask, "cost_model_used"] = str(discrete_selected_score.cost_model_used)
    discrete_selection["scores_frame"].loc[selected_mask, "live_costs_available"] = bool(discrete_selected_score.live_costs_available)
    discrete_selection["scores_frame"].loc[selected_mask, "gross_robust_score"] = float(discrete_selected_score.robust_score)
    discrete_selection["scores_frame"].loc[selected_mask, "net_robust_score"] = float(discrete_selected_score.net_robust_score)
    discrete_selection["scores_frame"].loc[selected_mask, "net_score_after_order_costs"] = float(discrete_selected_score.net_robust_score)
    discrete_selection["scores_frame"].loc[selected_mask, "total_order_cost"] = float(discrete_selected_score.estimated_total_order_cost)
    discrete_selection["scores_frame"].loc[selected_mask, "total_order_cost_pct_nav"] = float(discrete_selected_score.estimated_cost)
    discrete_selection["scores_frame"].loc[selected_mask, "delta_vs_hold"] = float(discrete_selected_score.delta_vs_hold)
    candidate_risk_return_frame = build_candidate_risk_return_frame(
        candidate_weights=discrete_candidate_weights,
        distribution=scenario_risk_distribution,
        current_weights=w_current,
        defensive_cash_weights=candidates["DEFENSIVE_CASH"].weights.reindex(active_tickers).fillna(0.0),
        hold_weights=candidates["HOLD"].weights.reindex(active_tickers).fillna(0.0),
        params=params,
        scores_frame=discrete_selection["scores_frame"],
        selected_name=best_discrete_candidate_name,
        selected_reason=selected_reason,
    )
    write_candidate_risk_return_reports(
        risk_return_frame=candidate_risk_return_frame,
        selected_candidate=best_discrete_candidate_name,
        objective_used=str(discrete_selection.get("objective_used", params.get("optimization_objective", "robust_score"))),
        output_dir=OUTPUT_DIR,
    )
    final_selection.selected_score = discrete_selected_score
    execution_buffer_value = float(0.001)
    model_uncertainty_buffer_value = float(model_governance.get("model_uncertainty_buffer", 0.001))
    trade_edge_summary = compute_trade_now_edge(
        current_score=float(current_portfolio_score),
        target_score_after_costs=target_score_after_costs,
        total_order_cost=float(discrete_selected_score.estimated_total_order_cost),
        execution_buffer=execution_buffer_value,
        model_uncertainty_buffer=model_uncertainty_buffer_value,
        other_penalties=0.0,
    )
    cost_review = build_transaction_cost_review_summary(
        final_order_cost_summary,
        nav=float(current_state.nav),
        config={
            **params,
            "nav": float(current_state.nav),
            "portfolio_nav_usd": float(current_state.nav),
        },
        trade_edge_summary=trade_edge_summary,
    )
    log_stage(diagnostics, "execution gate", "START")
    gate = evaluate_execution_gate(
        selection_result=final_selection,
        synthetic_data=bool(prices.attrs.get("synthetic_data", False)),
        data_freshness_ok=bool(data_freshness.get("data_freshness_ok", False)),
        broker_state_reconciled=bool(reconciliation_result["broker_state_reconciled"]),
        open_orders_exist=bool(reconciliation_result["open_orders_exist"]),
        estimated_spread_cost=float(final_order_cost_summary["total_estimated_spread_cost"]) / max(float(current_state.nav), 1e-12),
        estimated_slippage=float(final_order_cost_summary["total_estimated_slippage_cost"]) / max(float(current_state.nav), 1e-12),
        estimated_transaction_cost=float(discrete_selected_score.estimated_cost),
        delta_vs_hold_is_net=True,
        costs_include_spread_slippage=True,
        execution_uncertainty_buffer=execution_buffer_value,
        model_uncertainty_buffer=model_uncertainty_buffer_value,
    )
    if final_target_source == FINAL_TARGET_SOURCE_SOLVER_FAILED:
        gate.gate_status = "BLOCK"
        gate.action = FINAL_TARGET_SOURCE_SOLVER_FAILED
        gate.reason = (
            "Scenario-weighted solver failed post-validation; executable target is the current portfolio. "
            f"failure_reason={scenario_solver_result.constraint_diagnostics.get('failure_reason', scenario_solver_result.message)}"
        )
    elif not bool(market_gate.get("execution_allowed", False)):
        gate.gate_status = "BLOCK"
        gate.action = (
            "WAIT_MARKET_CLOSED"
            if not bool(market_gate.get("is_trading_day", False))
            else "WAIT_OUTSIDE_WINDOW"
        )
        gate.reason = f"Project calendar blocked execution: {market_gate.get('reason', 'calendar_blocked')}."
    log_execution_gate(
        diagnostics,
        {
            "gate_status": gate.gate_status,
            "action": gate.action,
            "reason": gate.reason,
            "trade_now_score": gate.trade_now_score,
            "spread_cost": gate.spread_cost,
            "slippage": gate.slippage,
            "buffers": gate.buffers,
        },
    )
    hold_vs_target_text, hold_vs_target_summary = _build_hold_vs_target_analysis(
        as_of=as_of,
        current_portfolio_score=current_portfolio_score,
        target_score_before_costs=target_score_before_costs,
        target_score_after_costs=target_score_after_costs,
        delta_score_vs_current=delta_score,
        total_order_cost=float(final_order_cost_summary["total_estimated_transaction_cost"]),
        execution_buffer=float(trade_edge_summary["execution_buffer"]),
        model_uncertainty_buffer=float(trade_edge_summary["model_uncertainty_buffer"]),
        trade_now_edge=float(trade_edge_summary["trade_now_edge"]),
        trade_now_hurdle=0.0025,
        probability_beats_current=float(discrete_selected_score.probability_beats_hold),
        probability_beats_cash=float(discrete_selected_score.probability_beats_cash),
        tail_risk_current=current_tail_risk,
        tail_risk_target=float(discrete_selected_score.cvar_5),
        current_weights=current_state.current_weights_actual.reindex(active_tickers).fillna(0.0),
        continuous_target_weights=continuous_target_weights.reindex(active_tickers).fillna(0.0),
        final_discrete_weights=final_weights.reindex(active_tickers).fillna(0.0),
        continuous_model_optimal_candidate=continuous_model_optimal_candidate,
        best_discrete_candidate_name=best_discrete_candidate_name,
        factor_forecast_df=factor_forecasts_df,
        discrete_scores_frame=discrete_selection["scores_frame"],
        gate_reason=str(gate.reason),
        data_context=data_context,
        risk_premium_hurdle=float(params.get("risk_premium_hurdle", 0.0005)),
        p_hold_min=float(params.get("p_hold_min", 0.55)),
        p_cash_min=float(params.get("p_cash_min", 0.52)),
    )
    hold_vs_target_summary.update(
        {
            "selected_reason": selected_reason,
            "hold_current_constraint_valid": bool(discrete_selection.get("hold_current_constraint_valid", True)),
            "current_portfolio_constraint_valid": bool(current_constraint_validation.get("ok", True)),
            "current_portfolio_constraint_violation": not bool(current_constraint_validation.get("ok", True)),
            "current_constraint_errors": current_constraint_errors_text,
            "current_portfolio_asset_limit_violations": len(current_constraint_validation.get("asset_limit_violations", [])),
            "current_portfolio_group_limit_violations": len(current_constraint_validation.get("group_limit_violations", [])),
            "best_non_hold_candidate": discrete_selection.get("best_non_hold_candidate", ""),
            "best_non_hold_score": discrete_selection.get("best_non_hold_score", float("nan")),
            "best_non_hold_valid_constraints": bool(discrete_selection.get("best_non_hold_valid_constraints", False)),
            "best_non_hold_failed_reason": discrete_selection.get("best_non_hold_failed_reason", ""),
            "best_model_candidate": discrete_selection.get("best_model_candidate", ""),
            "best_model_candidate_valid_constraints": bool(discrete_selection.get("best_model_candidate_valid_constraints", False)),
            "final_selection_is_safe_fallback": bool(discrete_selection.get("final_selection_is_safe_fallback", False)),
        }
    )
    log_stage(diagnostics, "execution gate", "DONE", extra=diagnostics.execution_gate_context)
    initial_preview_reason = "dry_run_preview_only" if effective_dry_run else (gate.reason if gate.gate_status != "PASS" else "")
    annotated_adjusted_order_preview = _annotate_final_daily_preview(
        preview_df=adjusted_order_preview,
        cash_before_orders=float(final_order_cost_summary["cash_before_orders"]),
        cash_after_orders=float(final_order_cost_summary["cash_after_orders"]),
        preview_only=bool(effective_dry_run or gate.gate_status != "PASS"),
        preview_only_reason=initial_preview_reason,
    )
    _write_csv(OUTPUT_DIR / "daily_bot_order_preview.csv", annotated_adjusted_order_preview, index=False)
    _write_csv(OUTPUT_DIR / "best_discrete_order_preview.csv", annotated_adjusted_order_preview, index=False)
    actionable_order_count = int((adjusted_order_preview["side"].astype(str) != "HOLD").sum())
    order_signature = compute_order_signature(adjusted_order_preview)
    decision_id = compute_decision_id(as_of, best_discrete_candidate_name, order_signature)
    duplicate_order_signature = (
        execution_mode_hint != "order_preview_only"
        and str(state_payload.get("current_date", "")) == str(state_date.date())
        and str(state_payload.get("last_order_signature", "")) == order_signature
        and order_signature != "no_orders"
    )

    execution_data_block_reason = ""
    if final_target_source == FINAL_TARGET_SOURCE_SOLVER_FAILED:
        execution_data_block_reason = (
            "Scenario-weighted solver failed; no execution path may be used. "
            f"failure_reason={scenario_solver_result.constraint_diagnostics.get('failure_reason', scenario_solver_result.message)}"
        )
    elif not bool(market_gate.get("execution_allowed", False)):
        execution_data_block_reason = f"Project calendar blocked execution: {market_gate.get('reason', 'calendar_blocked')}."
    elif not bool(data_freshness.get("data_freshness_ok", False)):
        execution_data_block_reason = str(
            data_freshness.get("warning")
            or "Current market data is too stale for any execution path."
        )
    elif bool(prices.attrs.get("synthetic_data", False)):
        execution_data_block_reason = "Synthetic data active; no execution path may be used."
    elif duplicate_order_signature:
        execution_data_block_reason = "Duplicate order signature for the same trading day detected; repeat execution is blocked."

    if execution_data_block_reason:
        annotated_adjusted_order_preview = _annotate_final_daily_preview(
            preview_df=adjusted_order_preview,
            cash_before_orders=float(final_order_cost_summary["cash_before_orders"]),
            cash_after_orders=float(final_order_cost_summary["cash_after_orders"]),
            preview_only=True,
            preview_only_reason=execution_data_block_reason,
        )
        _write_csv(OUTPUT_DIR / "daily_bot_order_preview.csv", annotated_adjusted_order_preview, index=False)
        _write_csv(OUTPUT_DIR / "best_discrete_order_preview.csv", annotated_adjusted_order_preview, index=False)
        log_warning(diagnostics, "daily_bot", execution_data_block_reason, severity="ERROR", stage="execution_gate")
        execution_result = {
            "execution_mode": "blocked",
            "orders_submitted": 0,
            "orders_failed": int((order_preview["side"] != "HOLD").sum()),
            "message": execution_data_block_reason,
            "errors": [execution_data_block_reason],
        }
    elif not validation_result["ok"]:
        execution_result = {
            "execution_mode": "blocked",
            "orders_submitted": 0,
            "orders_failed": int((order_preview["side"] != "HOLD").sum()),
            "message": "Pre-trade validation failed; no execution was attempted.",
            "errors": list(validation_result["errors"]),
        }
        for error in validation_result["errors"]:
            log_rejected_order(diagnostics, "MULTI", "MULTI", str(error), extra={"stage": "pre_trade_validation"})
    else:
        log_stage(diagnostics, "execution layer", "START")
        execution_result = run_execution_layer(
            order_preview_df=adjusted_order_preview,
            latest_prices=latest_prices_at_asof,
            params={
                **params,
                "dry_run": effective_dry_run,
                "enable_investopedia_simulator": False if effective_dry_run else bool(params.get("enable_investopedia_simulator", False)),
                "enable_local_paper_trading": False if effective_dry_run else bool(params.get("enable_local_paper_trading", False)),
                "enable_external_broker": False if effective_dry_run else bool(params.get("enable_external_broker", False)),
            },
            db_path=params["db_path"],
        )
        log_stage(diagnostics, "execution layer", "DONE", extra=execution_result)

    _write_scenario_weighted_allocation_csv(
        current_state=current_state,
        optimal_weights=optimal_solver_weights,
        executable_weights=executable_solver_weights.reindex(active_tickers).fillna(0.0),
        latest_prices=latest_prices_at_asof,
        params=params,
        output_dir=OUTPUT_DIR,
    )
    _write_csv(OUTPUT_DIR / "scenario_weighted_order_preview.csv", annotated_adjusted_order_preview, index=False)

    order_summary = _summarize_delta_order_preview(
        order_preview=annotated_adjusted_order_preview,
        cash_before_orders=float(final_order_cost_summary["cash_before_orders"]),
        cash_after_orders=float(final_order_cost_summary["cash_after_orders"]),
        nav=float(current_state.nav),
    )

    manual_simulator_orders, manual_simulator_text = _build_manual_simulator_order_outputs(
        order_preview=annotated_adjusted_order_preview,
        latest_price_date=data_freshness.get("latest_price_date", "n/a"),
        rest_cash_usd=float(best_discrete_candidate.cash_left),
        cash_before_orders=float(final_order_cost_summary["cash_before_orders"]),
        cash_after_orders=float(final_order_cost_summary["cash_after_orders"]),
    )
    _write_csv(OUTPUT_DIR / "manual_simulator_orders.csv", manual_simulator_orders, index=False)
    _write_text(OUTPUT_DIR / "manual_simulator_orders.txt", manual_simulator_text)

    log_stage(diagnostics, "trade sizing", "START")
    trade_sizing = compute_trade_fraction(
        selection_result=final_selection,
        execution_gate_result=gate,
        model_confidence=model_governance,
        data_quality=data_quality_report,
        turnover_budget={"turnover_budget_remaining": 1.0},
    )
    log_stage(diagnostics, "trade sizing", "DONE", extra=trade_sizing)

    continuous_model_target_weights = pd.DataFrame(
        {
            "ticker": active_tickers,
            "continuous_weight": continuous_target_weights.reindex(active_tickers).fillna(0.0).values,
            "source_candidate": continuous_model_optimal_candidate,
            "as_of_date": str(as_of.date()),
            "latest_price_date": str(data_freshness.get("latest_price_date", "n/a")),
        }
    )
    _write_csv(OUTPUT_DIR / "continuous_model_target_weights.csv", continuous_model_target_weights, index=False)

    compliance_discrete_rows = discrete_selection["scores_frame"].loc[
        discrete_selection["scores_frame"]["discrete_candidate"].astype(str).str.startswith("CURRENT_COMPLIANCE_REPAIR")
    ].copy()
    compliance_repair_available = bool(not compliance_discrete_rows.empty)
    compliance_best_row = (
        compliance_discrete_rows.sort_values("net_robust_score", ascending=False).iloc[0]
        if compliance_repair_available
        else None
    )
    compliance_repair_order_count = int(compliance_best_row.get("number_of_orders", 0)) if compliance_best_row is not None else 0
    compliance_repair_turnover = float(compliance_best_row.get("turnover_vs_current", 0.0)) if compliance_best_row is not None else 0.0
    compliance_repair_score = float(compliance_best_row.get("net_robust_score", 0.0)) if compliance_best_row is not None else float("nan")
    compliance_repair_delta_vs_hold = float(compliance_best_row.get("delta_vs_hold", 0.0)) if compliance_best_row is not None else float("nan")
    compliance_repair_fixes_current_constraints = bool(
        compliance_repair_available
        and not bool(current_constraint_validation.get("ok", True))
        and bool(compliance_best_row.get("valid_constraints", False))
    )
    selection.scores_frame["current_portfolio_constraint_valid"] = bool(current_constraint_validation.get("ok", True))
    selection.scores_frame["current_portfolio_constraint_violation"] = not bool(current_constraint_validation.get("ok", True))
    selection.scores_frame["current_portfolio_constraint_errors"] = current_constraint_errors_text
    selection.scores_frame["compliance_repair_available"] = compliance_repair_available
    selection.scores_frame["compliance_repair_order_count"] = compliance_repair_order_count
    selection.scores_frame["compliance_repair_turnover"] = compliance_repair_turnover
    selection.scores_frame["compliance_repair_score"] = compliance_repair_score
    selection.scores_frame["compliance_repair_delta_vs_hold"] = compliance_repair_delta_vs_hold
    selection.scores_frame["compliance_repair_fixes_current_constraints"] = compliance_repair_fixes_current_constraints
    for column, value in {
        "selected_reason": selected_reason,
        "current_portfolio_constraint_valid": bool(current_constraint_validation.get("ok", True)),
        "current_portfolio_constraint_violation": not bool(current_constraint_validation.get("ok", True)),
        "current_constraint_errors": current_constraint_errors_text,
        "compliance_repair_available": compliance_repair_available,
        "compliance_repair_order_count": compliance_repair_order_count,
        "compliance_repair_turnover": compliance_repair_turnover,
        "compliance_repair_score": compliance_repair_score,
        "compliance_repair_delta_vs_hold": compliance_repair_delta_vs_hold,
        "compliance_repair_fixes_current_constraints": compliance_repair_fixes_current_constraints,
    }.items():
        discrete_selection["scores_frame"][column] = value
    constraint_pressure = _build_constraint_pressure_reports(
        scores_frame=discrete_selection["scores_frame"],
        candidate_map=discrete_scored["candidate_map"],
        params=params,
        selected_reason=selected_reason,
        output_dir=OUTPUT_DIR,
    )
    active_preview_summary = _select_active_preview_candidate(
        scores_frame=discrete_selection["scores_frame"],
        current_portfolio_score=float(current_portfolio_score),
        params=params,
        safe_model_uncertainty_buffer=float(model_uncertainty_buffer_value),
    )
    active_preview_summary.update(
        {
            "safe_final_action": gate.action,
            "safe_final_discrete_candidate": best_discrete_candidate_name,
            "safe_selected_reason": selected_reason,
            "safe_trade_now_edge": float(trade_edge_summary["trade_now_edge"]),
            "safe_trade_now_hurdle": 0.0025,
            "safe_order_count": int(order_summary["order_count"]),
        }
    )
    active_preview_summary = _build_active_preview_files(
        active_preview_summary=active_preview_summary,
        candidate_map=discrete_scored["candidate_map"],
        current_state=current_state,
        latest_prices=latest_prices_at_asof,
        active_tickers=active_tickers,
        params=params,
        output_dir=OUTPUT_DIR,
    )
    rebalance_sensitivity_matrix = _write_rebalance_sensitivity_matrix(
        scores_frame=discrete_selection["scores_frame"],
        candidate_map=discrete_scored["candidate_map"],
        current_portfolio_score=float(current_portfolio_score),
        params=params,
        safe_model_uncertainty_buffer=float(model_uncertainty_buffer_value),
        current_state=current_state,
        latest_prices=latest_prices_at_asof,
        active_tickers=active_tickers,
        output_dir=OUTPUT_DIR,
    )
    diagnostics.model_context["active_preview"] = dict(active_preview_summary)
    diagnostics.model_context["rebalance_sensitivity_rows"] = int(len(rebalance_sensitivity_matrix))
    if gate.action == "HOLD" and str(active_preview_summary.get("active_preview_action", "HOLD")) == "BUY_SELL_PREVIEW":
        safe_active_note = "Safe Mode remains HOLD, but Active Preview finds a lower-hurdle rebalance candidate."
    elif str(active_preview_summary.get("active_preview_action", "HOLD")) == "BUY_SELL_PREVIEW":
        safe_active_note = "Active Preview finds a lower-hurdle rebalance candidate, but it is not executable."
    else:
        safe_active_note = "Even Active Preview did not find a valid non-HOLD candidate."
    safe_active_report_lines = [
        "",
        "SAFE MODE:",
        f"- final_action: {gate.action}",
        f"- final_discrete_candidate: {best_discrete_candidate_name}",
        f"- selected_reason: {selected_reason}",
        f"- trade_now_edge: {float(trade_edge_summary['trade_now_edge']):.6f}",
        "- trade_now_hurdle: 0.002500",
        f"- order_count: {int(order_summary['order_count'])}",
        "",
        "ACTIVE PREVIEW:",
        f"- active_preview_action: {active_preview_summary.get('active_preview_action', 'HOLD')}",
        f"- active_preview_candidate: {active_preview_summary.get('active_preview_candidate', 'HOLD_CURRENT')}",
        f"- active_preview_trade_now_edge: {_safe_float(active_preview_summary.get('active_preview_trade_now_edge')):.6f}",
        f"- active_preview_hurdle: {_safe_float(active_preview_summary.get('active_preview_hurdle')):.6f}",
        f"- active_preview_order_count: {int(active_preview_summary.get('active_preview_order_count', 0) or 0)}",
        f"- active_preview_buy_count: {int(active_preview_summary.get('active_preview_buy_count', 0) or 0)}",
        f"- active_preview_sell_count: {int(active_preview_summary.get('active_preview_sell_count', 0) or 0)}",
        f"- active_preview_turnover: {_safe_float(active_preview_summary.get('active_preview_turnover')):.6f}",
        f"- active_preview_reason: {active_preview_summary.get('active_preview_reason', 'unknown')}",
        "- active_preview_executable: false",
        f"- active_preview_order_submission_allowed: {bool(active_preview_summary.get('active_preview_order_submission_allowed', False))}",
        safe_active_note,
    ]
    safe_active_report_text = "\n".join(safe_active_report_lines) + "\n"

    _write_csv(OUTPUT_DIR / "forecast_3m_diagnostics.csv", forecast.diagnostics)
    _write_csv(OUTPUT_DIR / "scenario_summary.csv", scenario_set.summary, index=False)
    _write_csv(OUTPUT_DIR / "candidate_scores.csv", selection.scores_frame, index=False)
    _write_csv(OUTPUT_DIR / "discrete_candidate_scores.csv", discrete_selection["scores_frame"], index=False)
    best_discrete_allocation = pd.DataFrame(
        {
            "ticker": active_tickers,
            "continuous_target_weight": best_continuous_for_discrete.reindex(active_tickers).fillna(0.0).values,
            "target_weight_continuous": best_continuous_for_discrete.reindex(active_tickers).fillna(0.0).values,
            "discrete_weight": best_discrete_candidate.weights_actual.reindex(active_tickers).fillna(0.0).values,
            "latest_price": latest_prices_at_asof.reindex(active_tickers).values,
            "shares": final_target_shares.reindex(active_tickers).fillna(0.0).values,
            "value": best_discrete_candidate.values.reindex(active_tickers).fillna(0.0).values,
            "weight_drift": (
                best_discrete_candidate.weights_actual.reindex(active_tickers).fillna(0.0)
                - best_continuous_for_discrete.reindex(active_tickers).fillna(0.0)
            ).values,
            "abs_weight_drift": (
                best_discrete_candidate.weights_actual.reindex(active_tickers).fillna(0.0)
                - best_continuous_for_discrete.reindex(active_tickers).fillna(0.0)
            ).abs().values,
        }
    )
    _write_csv(OUTPUT_DIR / "best_discrete_allocation.csv", best_discrete_allocation, index=False)
    _write_csv(
        OUTPUT_DIR / "selected_candidate_weights.csv",
        final_weights.rename("weight").to_frame(),
        header=True,
    )
    _write_csv(
        OUTPUT_DIR / "execution_gate_report.csv",
        pd.DataFrame(
        [
            {
                "gate_status": gate.gate_status,
                "action": gate.action,
                "reason": gate.reason,
                "run_context": data_context["run_context"],
                "latest_price_date": data_context["latest_price_date"],
                "expected_latest_trading_day": data_context["expected_latest_trading_day"],
                "data_source": data_context["data_source"],
                "cache_status": data_context["cache_status"],
                "synthetic_data": data_context["synthetic_data"],
                "used_cache_fallback": data_context["used_cache_fallback"],
                "within_allowed_window": data_context["within_allowed_window"],
                "execution_allowed_by_calendar": data_context["execution_allowed_by_calendar"],
                "trade_now_score": gate.trade_now_score,
                "estimated_transaction_cost": float(discrete_selected_score.estimated_cost),
                "total_order_cost_usd": float(discrete_selected_score.estimated_total_order_cost),
                "spread_cost": gate.spread_cost,
                "slippage": gate.slippage,
                "execution_uncertainty_buffer": gate.buffers["execution_uncertainty_buffer"],
                "model_uncertainty_buffer": gate.buffers["model_uncertainty_buffer"],
            }
        ]
        ),
        index=False,
    )
    discrete_report_lines = [
        f"current_portfolio_source: {current_state.source}",
        "current_portfolio_label: CURRENT_PORTFOLIO",
        f"final_target_source: {final_target_source}",
        "legacy_candidate_ranking_used_for_final_target: false",
        "final_target_rule: scenario-weighted RF-adjusted Sharpe optimizer, then whole-share sizing and execution gates",
        f"continuous_model_optimal_candidate: {continuous_model_optimal_candidate}",
        "continuous_model_target_label: CONTINUOUS_MODEL_TARGET",
        f"continuous_model_optimal_score: {continuous_model_optimal_score:.6f}",
        f"best_discrete_candidate: {best_discrete_candidate_name}",
        "discrete_model_target_label: DISCRETE_MODEL_TARGET",
        f"best_discrete_source_candidate: {best_discrete_source}",
        f"best_discrete_score: {float(discrete_selection['best_discrete_score']):.6f}",
        f"objective_used: {discrete_selection.get('objective_used', params.get('optimization_objective', 'robust_score'))}",
        f"objective_score_column: {discrete_selection.get('objective_score_column', 'net_robust_score')}",
        f"best_discrete_objective_score: {_safe_float(discrete_selection.get('best_discrete_objective_score'), float('nan')):.6f}",
        f"highest_scenario_objective_candidate: {str(candidate_risk_return_frame.iloc[0]['candidate']) if not candidate_risk_return_frame.empty else 'none'}",
        f"highest_scenario_objective_score: {_safe_float(candidate_risk_return_frame.iloc[0]['robust_scenario_sharpe_objective'] if not candidate_risk_return_frame.empty else float('nan'), float('nan')):.6f}",
        "candidate_risk_return_report: outputs/candidate_risk_return_report.csv",
        "scenario_covariance_summary: outputs/scenario_covariance_summary.csv",
        f"selected_reason: {selected_reason}",
        f"hold_current_constraint_valid: {bool(discrete_selection.get('hold_current_constraint_valid', True))}",
        f"current_portfolio_constraint_valid: {bool(current_constraint_validation.get('ok', True))}",
        f"current_portfolio_constraint_violation: {not bool(current_constraint_validation.get('ok', True))}",
        "current_constraint_errors: "
        + (current_constraint_errors_text if current_constraint_errors_text else "none"),
        f"current_portfolio_asset_limit_violations: {len(current_constraint_validation.get('asset_limit_violations', []))}",
        f"current_portfolio_group_limit_violations: {len(current_constraint_validation.get('group_limit_violations', []))}",
        f"best_non_hold_candidate: {discrete_selection.get('best_non_hold_candidate', '') or 'none'}",
        f"best_non_hold_score: {_safe_float(discrete_selection.get('best_non_hold_score'), float('nan')):.6f}",
        f"best_non_hold_valid_constraints: {bool(discrete_selection.get('best_non_hold_valid_constraints', False))}",
        f"best_non_hold_failed_reason: {discrete_selection.get('best_non_hold_failed_reason', '') or 'none'}",
        f"best_model_candidate: {discrete_selection.get('best_model_candidate', '') or 'none'}",
        f"best_model_candidate_valid_constraints: {bool(discrete_selection.get('best_model_candidate_valid_constraints', False))}",
        f"final_selection_is_safe_fallback: {bool(discrete_selection.get('final_selection_is_safe_fallback', False))}",
        f"compliance_repair_available: {compliance_repair_available}",
        f"compliance_repair_order_count: {compliance_repair_order_count}",
        f"compliance_repair_turnover: {compliance_repair_turnover:.6f}",
        f"compliance_repair_score: {compliance_repair_score:.6f}",
        f"compliance_repair_delta_vs_hold: {compliance_repair_delta_vs_hold:.6f}",
        f"compliance_repair_fixes_current_constraints: {compliance_repair_fixes_current_constraints}",
        f"reason: {discrete_selection['reason']}",
        f"cash_left: {best_discrete_candidate.cash_left:.2f}",
        f"cash_weight: {best_discrete_candidate.cash_weight:.6f}",
        f"total_estimated_transaction_cost: {float(final_order_cost_summary['total_estimated_transaction_cost']):.2f}",
        f"estimated_transaction_cost_pct_nav: {float(final_order_cost_summary['total_order_cost_pct_nav']):.6f}",
        f"cost_model_used: {final_order_cost_summary['cost_model_used']}",
        f"live_costs_available: {bool(final_order_cost_summary['live_costs_available'])}",
        f"cost_assumptions: {format_cost_assumptions_summary(params)}",
    ]
    selected_discrete_row = discrete_selection["scores_frame"].loc[
        discrete_selection["scores_frame"]["discrete_candidate"].astype(str) == best_discrete_candidate_name
    ].iloc[0]
    discrete_report_lines.extend(
        [
            f"current_portfolio_score: {float(hold_vs_target_summary['current_portfolio_score']):.6f}",
            f"target_score_before_costs: {float(hold_vs_target_summary['target_score_before_costs']):.6f}",
            f"target_score_after_costs: {float(hold_vs_target_summary['target_score_after_costs']):.6f}",
            f"delta_score_vs_current: {float(hold_vs_target_summary['delta_score_vs_current']):.6f}",
            f"probability_beats_current: {float(hold_vs_target_summary['probability_beats_current']):.4f}",
            f"probability_beats_cash: {float(hold_vs_target_summary['probability_beats_cash']):.4f}",
            f"tail_risk_current: {float(hold_vs_target_summary['tail_risk_current']):.6f}",
            f"tail_risk_target: {float(hold_vs_target_summary['tail_risk_target']):.6f}",
            f"selected_total_abs_weight_drift: {float(selected_discrete_row['total_abs_weight_drift']):.6f}",
            f"selected_max_abs_weight_drift: {float(selected_discrete_row['max_abs_weight_drift']):.6f}",
        ]
    )
    closest_rounding = discrete_selection["scores_frame"].loc[
        (
            discrete_selection["scores_frame"]["continuous_source"].astype(str) == best_discrete_source
        )
        & (
            discrete_selection["scores_frame"]["discrete_candidate"].astype(str).str.endswith("ROUND_NEAREST_REPAIR_0")
            | discrete_selection["scores_frame"]["discrete_candidate"].astype(str).str.contains("ROUND_NEAREST_REPAIR")
        )
    ]
    if not closest_rounding.empty:
        closest = closest_rounding.iloc[0]
        discrete_report_lines.extend(
            [
                f"closest_rounding_candidate: {closest['discrete_candidate']}",
                f"closest_rounding_score: {float(closest['net_robust_score']):.6f}",
                f"score_difference_vs_closest_rounding: {float(discrete_selection['best_discrete_score']) - float(closest['net_robust_score']):.6f}",
            ]
        )
    hold_current_rows = discrete_selection["scores_frame"].loc[
        discrete_selection["scores_frame"]["discrete_candidate"].astype(str) == "HOLD_CURRENT"
    ]
    if not hold_current_rows.empty:
        hold_current_row = hold_current_rows.iloc[0]
        discrete_report_lines.append(
            f"score_difference_vs_hold_current: {float(discrete_selection['best_discrete_score']) - float(hold_current_row['net_robust_score']):.6f}"
        )
    discrete_report_lines.extend(
        [
            f"final_action: {gate.action}",
            f"executable: {bool(gate.gate_status == 'PASS' and validation_result['ok'] and market_gate.get('execution_allowed', False))}",
            f"reason_after_gate: {gate.reason}",
        ]
    )
    discrete_report_lines.append("target_drivers:")
    discrete_report_lines.extend(
        [f"- {line}" for line in list(hold_vs_target_summary.get("target_drivers", []))]
    )
    discrete_report_lines.append("reasons_against_immediate_trade:")
    discrete_report_lines.extend(
        [f"- {line}" for line in list(hold_vs_target_summary.get("reasons_against_trade", []))]
    )
    discrete_report_lines.append("what_must_change_for_buy_sell:")
    discrete_report_lines.extend(
        [f"- {line}" for line in list(hold_vs_target_summary.get("release_conditions", []))]
    )
    discrete_report_lines.append("Selection is score-based, not only weight-drift-based.")
    discrete_report_lines.extend(safe_active_report_lines)
    _write_text(OUTPUT_DIR / "discrete_optimization_report.txt", "\n".join(discrete_report_lines) + "\n")

    daily_review_payload = _build_daily_review_payload(
        as_of=as_of,
        data_context=data_context,
        current_state=current_state,
        latest_prices_at_asof=latest_prices_at_asof,
        active_tickers=active_tickers,
        continuous_model_optimal_candidate=continuous_model_optimal_candidate,
        continuous_target_weights=continuous_target_weights.reindex(active_tickers).fillna(0.0),
        best_discrete_candidate_name=best_discrete_candidate_name,
        best_discrete_candidate=best_discrete_candidate,
        final_target_shares=final_target_shares.reindex(active_tickers).fillna(0.0),
        annotated_adjusted_order_preview=annotated_adjusted_order_preview,
        cost_review=cost_review,
        hold_vs_target_summary=hold_vs_target_summary,
        gate=gate,
        execution_result=execution_result,
        trade_edge_summary=trade_edge_summary,
        data_quality_report=data_quality_report,
        validation_result=validation_result,
        discrete_selected_score=discrete_selected_score,
        order_summary=order_summary,
        active_preview_summary=active_preview_summary,
    )
    review_issues = build_review_issues(daily_review_payload, output_dir=OUTPUT_DIR)

    rebalance_buffer = float(gate.buffers["execution_uncertainty_buffer"]) + float(gate.buffers["model_uncertainty_buffer"])
    rebalance_report = [
        f"current_score: {current_portfolio_score:.6f}",
        f"target_score_before_costs: {target_score_before_costs:.6f}",
        f"target_score_after_costs: {target_score_after_costs:.6f}",
        f"best_discrete_score: {float(discrete_selection['best_discrete_score']):.6f}",
        f"objective_used: {discrete_selection.get('objective_used', params.get('optimization_objective', 'robust_score'))}",
        f"objective_score_column: {discrete_selection.get('objective_score_column', 'net_robust_score')}",
        f"best_discrete_objective_score: {_safe_float(discrete_selection.get('best_discrete_objective_score'), float('nan')):.6f}",
        f"highest_scenario_objective_candidate: {str(candidate_risk_return_frame.iloc[0]['candidate']) if not candidate_risk_return_frame.empty else 'none'}",
        f"highest_scenario_objective_score: {_safe_float(candidate_risk_return_frame.iloc[0]['robust_scenario_sharpe_objective'] if not candidate_risk_return_frame.empty else float('nan'), float('nan')):.6f}",
        f"selected_reason: {selected_reason}",
        f"hold_current_constraint_valid: {bool(discrete_selection.get('hold_current_constraint_valid', True))}",
        f"current_portfolio_constraint_valid: {bool(current_constraint_validation.get('ok', True))}",
        f"current_portfolio_constraint_violation: {not bool(current_constraint_validation.get('ok', True))}",
        "current_constraint_errors: "
        + (current_constraint_errors_text if current_constraint_errors_text else "none"),
        f"current_portfolio_asset_limit_violations: {len(current_constraint_validation.get('asset_limit_violations', []))}",
        f"current_portfolio_group_limit_violations: {len(current_constraint_validation.get('group_limit_violations', []))}",
        f"best_non_hold_candidate: {discrete_selection.get('best_non_hold_candidate', '') or 'none'}",
        f"best_non_hold_score: {_safe_float(discrete_selection.get('best_non_hold_score'), float('nan')):.6f}",
        f"best_non_hold_valid_constraints: {bool(discrete_selection.get('best_non_hold_valid_constraints', False))}",
        f"best_non_hold_failed_reason: {discrete_selection.get('best_non_hold_failed_reason', '') or 'none'}",
        f"best_model_candidate: {discrete_selection.get('best_model_candidate', '') or 'none'}",
        f"best_model_candidate_valid_constraints: {bool(discrete_selection.get('best_model_candidate_valid_constraints', False))}",
        f"final_selection_is_safe_fallback: {bool(discrete_selection.get('final_selection_is_safe_fallback', False))}",
        f"compliance_repair_available: {compliance_repair_available}",
        f"compliance_repair_order_count: {compliance_repair_order_count}",
        f"compliance_repair_turnover: {compliance_repair_turnover:.6f}",
        f"compliance_repair_score: {compliance_repair_score:.6f}",
        f"compliance_repair_delta_vs_hold: {compliance_repair_delta_vs_hold:.6f}",
        f"compliance_repair_fixes_current_constraints: {compliance_repair_fixes_current_constraints}",
        f"delta_score: {delta_score:.6f}",
        f"estimated_cost: {float(discrete_selected_score.estimated_cost):.6f}",
        f"estimated_cost_usd: {float(final_order_cost_summary['total_estimated_transaction_cost']):.2f}",
        f"total_order_cost: {float(final_order_cost_summary['total_estimated_transaction_cost']):.2f}",
        f"total_order_cost_pct_nav: {float(final_order_cost_summary['total_order_cost_pct_nav']):.6f}",
        f"commission_per_trade_usd: {float(cost_review['commission_per_trade_usd']):.2f}",
        f"simulator_order_fee_usd: {float(cost_review['simulator_order_fee_usd']):.2f}",
        f"total_simulator_order_fees_usd: {float(cost_review['total_simulator_order_fees_usd']):.2f}",
        f"modeled_spread_bps: {float(cost_review['modeled_spread_bps']):.2f}",
        f"modeled_slippage_bps: {float(cost_review['modeled_slippage_bps']):.2f}",
        f"modeled_bps_per_turnover: {float(cost_review['modeled_bps_per_turnover']):.2f}",
        f"modeled_transaction_costs_usd: {float(cost_review['modeled_transaction_costs_usd']):.2f}",
        f"modeled_transaction_costs_pct_nav: {float(cost_review['modeled_transaction_costs_pct_nav']):.6f}",
        f"cost_model_used: {final_order_cost_summary['cost_model_used']}",
        f"cash_before_orders: {float(final_order_cost_summary['cash_before_orders']):.2f}",
        f"cash_after_orders: {float(order_summary['cash_after_orders']):.2f}",
        f"estimated_sell_value: {float(order_summary['estimated_sell_value']):.2f}",
        f"estimated_buy_value: {float(order_summary['estimated_buy_value']):.2f}",
        f"total_simulator_fees_usd: {float(order_summary['total_simulator_fees_usd']):.2f}",
        f"modeled_transaction_costs_usd: {float(order_summary['modeled_transaction_costs_usd']):.2f}",
        f"buy_count: {int(order_summary['buy_count'])}",
        f"sell_count: {int(order_summary['sell_count'])}",
        f"hold_count: {int(order_summary['hold_count'])}",
        f"order_count: {int(order_summary['order_count'])}",
        f"negative_cash_check: {bool(order_summary['negative_cash_check'])}",
        f"leverage_check: {bool(order_summary['leverage_check'])}",
        f"short_check: {bool(order_summary['short_check'])}",
        f"manual_orders_usable: {bool(order_summary['manual_orders_usable'])}",
        f"execution_buffer: {trade_edge_summary['execution_buffer']:.6f}",
        f"model_uncertainty_buffer: {trade_edge_summary['model_uncertainty_buffer']:.6f}",
        f"buffer: {rebalance_buffer:.6f}",
        f"trade_now_edge: {trade_edge_summary['trade_now_edge']:.6f}",
        f"trade_now_edge_after_modeled_costs: {float(cost_review['trade_now_edge_after_modeled_costs']):.6f}",
        f"trade_now_edge_without_direct_simulator_fees: {float(cost_review['trade_now_edge_without_direct_simulator_fees']):.6f}",
        f"probability_beats_current: {float(hold_vs_target_summary['probability_beats_current']):.4f}",
        f"probability_beats_cash: {float(hold_vs_target_summary['probability_beats_cash']):.4f}",
        f"tail_risk_current: {float(hold_vs_target_summary['tail_risk_current']):.6f}",
        f"tail_risk_target: {float(hold_vs_target_summary['tail_risk_target']):.6f}",
        f"order_count: {int(order_summary['order_count'])}",
        f"manual_eligible_order_count: {int(order_summary['manual_eligible_order_count'])}",
        f"buy_count: {int(order_summary['buy_count'])}",
        f"sell_count: {int(order_summary['sell_count'])}",
        f"hold_count: {int(order_summary['hold_count'])}",
        f"estimated_sell_value: {float(order_summary['estimated_sell_value']):.2f}",
        f"estimated_buy_value: {float(order_summary['estimated_buy_value']):.2f}",
        f"manual_orders_usable: {bool(order_summary['manual_orders_usable'])}",
        f"preview_only: {bool(effective_dry_run or gate.gate_status != 'PASS')}",
        f"review_status: {review_issues.get('review_status', 'REVIEW')}",
        f"first_blocker: {review_issues.get('first_blocker', 'none')}",
        f"all_blockers: {' | '.join(map(str, review_issues.get('all_blockers', ['none'])))}",
        f"hard_fail_count: {int(review_issues.get('hard_fail_count', 0))}",
        f"soft_warning_count: {int(review_issues.get('soft_warning_count', 0))}",
        f"info_count: {int(review_issues.get('info_count', 0))}",
        f"final_action: {gate.action}",
        f"reason: {gate.reason}",
    ]
    rebalance_report.append("target_drivers: " + " | ".join(list(hold_vs_target_summary.get("target_drivers", [])) or ["none"]))
    rebalance_report.append("reasons_against_trade: " + " | ".join(list(hold_vs_target_summary.get("reasons_against_trade", [])) or ["none"]))
    rebalance_report.append("what_must_change: " + " | ".join(list(hold_vs_target_summary.get("release_conditions", [])) or ["none"]))
    rebalance_report.append(
        "issue_table: "
        + " | ".join(
            f"{item.get('severity')}:{item.get('message')}" for item in list(review_issues.get("issue_table", []))
        )
    )
    rebalance_report.extend(safe_active_report_lines)
    _write_text(OUTPUT_DIR / "rebalance_decision_report.txt", "\n".join(rebalance_report) + "\n")

    optimizer_price_usage_lines = [
        "Optimizer price-usage audit",
        "",
        "1. Main optimizer:",
        "- uses forecast returns and covariance inputs, not raw price levels, for allocation scoring.",
        "- forecast inputs come from return-derived signals up to the current as_of date.",
        "",
        "2. Discrete execution layer:",
        "- uses latest_prices only for share counts, target_value/current_value, order_value, NAV and cash feasibility.",
        "- does not re-optimize on raw price levels.",
        "",
        "3. Current run:",
        f"- latest_price_date: {data_freshness.get('latest_price_date', 'n/a')}",
        f"- continuous_model_target_candidate: {continuous_model_optimal_candidate}",
        f"- best_discrete_candidate: {best_discrete_candidate_name}",
        "",
        "4. Conclusion:",
        "- Historical data are input for today's forecasts, not the optimization target.",
        "- No direct use of initial or stale price levels was found in the final discrete order construction path.",
    ]
    _write_text(OUTPUT_DIR / "optimizer_price_usage_audit.txt", "\n".join(optimizer_price_usage_lines) + "\n")

    transaction_cost_audit_lines = [
        "Transaction cost and slippage audit",
        "",
        "The system uses modeled transaction cost assumptions, not live broker fees.",
        "",
        "1. Cost calculation locations",
        "- robust_scorer.py: continuous / generic candidate evaluation still uses a turnover proxy via estimated_cost = turnover * cost_rate.",
        "- discrete_portfolio_optimizer.py: final discrete candidates are re-scored with modeled costs on the actual whole-share order preview.",
        "- transaction_costs.py: central cost model for spread, slippage, market impact, commission and order-list aggregation.",
        "- pre_trade_validation.py: cash and position validation uses estimated_total_order_cost per row when available, else falls back to cost_rate.",
        "- execution_gate.py: final gate uses the discrete candidate edge after modeled order costs and only subtracts uncertainty buffers again; spread/slippage are still checked as execution blockers.",
        "- paper_broker_stub.py: local paper execution uses estimated_total_order_cost per row when available, else falls back to cost_rate.",
        "- simulator_orchestrator.py: passes the final order preview through the paper/simulator layer without enabling real orders.",
        "",
        "2. Current formulas",
        "- continuous proxy: estimated_cost_pct = turnover * cost_rate",
        "- discrete per-order: cost_i = commission_i + abs(order_value_i) * (spread_bps + slippage_bps + market_impact_bps) / 10000",
        "- order-list total: total_trade_cost = sum(cost_i)",
        "- net discrete score: net_robust_score = robust_score - total_trade_cost / NAV - dynamic_buffer",
        "",
        "3. Current parameters",
        f"- commission_per_trade_usd: {float(params.get('default_commission_per_trade_usd', 0.0)):.2f}",
        f"- default_bps_per_turnover: {float(params.get('default_bps_per_turnover', 5.0)):.2f}",
        f"- default_spread_bps: {float(params.get('default_spread_bps', 2.0)):.2f}",
        f"- default_slippage_bps: {float(params.get('default_slippage_bps', 3.0)):.2f}",
        f"- default_market_impact_bps: {float(params.get('default_market_impact_bps', 0.0)):.2f}",
        f"- min_order_value_usd: {float(params.get('min_order_value_usd', 10.0)):.2f}",
        f"- cash_buffer_usd: {float(params.get('cash_buffer_usd', 0.0)):.2f}",
        "",
        "4. Asset-specific differentiation",
        "- lower-cost bucket: SGOV, SHY, IEF, AGG, LQD, TIP, TLT",
        "- normal-liquidity bucket: XLC, XLY, XLE, XLF, XLRE, XLB, XLI, XLK, XLP, XLU, XLV, VEA, VWO, RPV, SIZE, VBR, SPHQ, SPLV, SPMO and similar broad ETFs",
        "- normal-liquidity credit bucket: HYG, EMB",
        "- higher-cost commodity bucket: PDBC, GLD, SLV plus configured commodity overrides",
        "- higher-cost inverse bucket: SH (and PSQ if enabled later)",
        "- higher-cost crypto bucket: IBIT (and ETHA if enabled later)",
        "",
        "5. Live quotes and broker fees",
        f"- live_costs_available_current_run: {bool(final_order_cost_summary['live_costs_available'])}",
        "- live bid/ask spreads are not available in the current daily-bot path, so modeled spread assumptions were used.",
        f"- simulator_order_fee_usd: {float(cost_review['simulator_order_fee_usd']):.2f}",
        f"- total_simulator_order_fees_usd: {float(cost_review['total_simulator_order_fees_usd']):.2f}",
        "- the simulator/game direct order fee is separated from conservative modeled spread/slippage/market-impact costs.",
        "",
        "6. Small orders and minimum order handling",
        "- min_order_value_usd is enforced in discrete candidate construction and order preview generation.",
        "- too-small rows are converted to HOLD / not_executable rather than executed.",
        f"- skipped_small_orders_current_run: {int(final_order_cost_summary['skipped_small_orders'])}",
        "",
        "7. Current run summary",
        f"- best_discrete_candidate: {best_discrete_candidate_name}",
        f"- commission_per_trade_usd: {float(cost_review['commission_per_trade_usd']):.2f}",
        f"- modeled_spread_bps: {float(cost_review['modeled_spread_bps']):.2f}",
        f"- modeled_slippage_bps: {float(cost_review['modeled_slippage_bps']):.2f}",
        f"- modeled_bps_per_turnover: {float(cost_review['modeled_bps_per_turnover']):.2f}",
        f"- total_simulator_order_fees_usd: {float(cost_review['total_simulator_order_fees_usd']):.2f}",
        f"- modeled_transaction_costs_usd: {float(cost_review['modeled_transaction_costs_usd']):.2f}",
        f"- modeled_transaction_costs_pct_nav: {float(cost_review['modeled_transaction_costs_pct_nav']):.6f}",
        f"- total_estimated_transaction_cost_usd: {float(cost_review['total_transaction_costs_usd']):.2f}",
        f"- total_estimated_transaction_cost_pct_nav: {float(cost_review['total_transaction_costs_pct_nav']):.6f}",
        f"- trade_now_edge_after_modeled_costs: {float(cost_review['trade_now_edge_after_modeled_costs']):.6f}",
        f"- trade_now_edge_without_direct_simulator_fees: {float(cost_review['trade_now_edge_without_direct_simulator_fees']):.6f}",
        f"- weighted_average_cost_bps: {float(final_order_cost_summary['weighted_average_cost_bps']):.2f}",
        f"- cost_model_used: {final_order_cost_summary['cost_model_used']}",
        f"- cash_before_orders: {float(final_order_cost_summary['cash_before_orders']):.2f}",
        f"- cash_after_orders: {float(final_order_cost_summary['cash_after_orders']):.2f}",
        f"- no_negative_cash_after_orders: {bool(final_order_cost_summary['no_negative_cash'])}",
        "",
        "8. Final assessment",
        "- Final rebalance yes/no logic now uses the discrete whole-share candidate with modeled order-list costs, not only a theoretical percent-turnover proxy.",
        "- No real broker fees or live current transaction costs are claimed by the system.",
    ]
    _write_text(OUTPUT_DIR / "transaction_cost_audit.txt", "\n".join(transaction_cost_audit_lines) + "\n")
    cost_by_group = (
        adjusted_order_preview.loc[adjusted_order_preview["side"].astype(str).isin(["BUY", "SELL"])]
        .groupby("assumption_bucket", dropna=False)
        .agg(
            orders=("ticker", "count"),
            total_order_value=("order_value", lambda s: float(pd.Series(s, dtype=float).abs().sum())),
            total_cost=("estimated_total_order_cost", "sum"),
            max_cost_bps=("cost_bps_used", "max"),
        )
        .reset_index()
    )
    most_expensive_order_text = "none"
    highest_bps_order_text = "none"
    actionable_orders = adjusted_order_preview.loc[adjusted_order_preview["side"].astype(str).isin(["BUY", "SELL"])].copy()
    if not actionable_orders.empty:
        most_expensive_row = actionable_orders.loc[actionable_orders["estimated_total_order_cost"].astype(float).idxmax()]
        most_expensive_order_text = (
            f"{most_expensive_row['side']} {abs(float(most_expensive_row['order_shares'])):.0f} {most_expensive_row['ticker']} "
            f"cost={float(most_expensive_row['estimated_total_order_cost']):.2f}"
        )
        highest_bps_row = actionable_orders.loc[actionable_orders["cost_bps_used"].astype(float).idxmax()]
        highest_bps_order_text = (
            f"{highest_bps_row['ticker']} {float(highest_bps_row['cost_bps_used']):.2f}bps"
        )
    transaction_cost_report_lines = [
        "Transaction cost report",
        "",
        f"Uses live current transaction costs: {'yes' if bool(final_order_cost_summary['live_costs_available']) else 'no'}",
        "If no: the system uses modeled transaction cost assumptions, not live broker fees.",
        f"cost_model_used: {cost_review['cost_model_used']}",
        f"Cost assumptions: {format_cost_assumptions_summary(params)}",
        f"commission_per_trade_usd: {float(cost_review['commission_per_trade_usd']):.2f}",
        f"simulator_order_fee_usd: {float(cost_review['simulator_order_fee_usd']):.2f}",
        f"total_simulator_order_fees_usd: {float(cost_review['total_simulator_order_fees_usd']):.2f}",
        f"modeled_spread_bps: {float(cost_review['modeled_spread_bps']):.2f}",
        f"modeled_slippage_bps: {float(cost_review['modeled_slippage_bps']):.2f}",
        f"modeled_bps_per_turnover: {float(cost_review['modeled_bps_per_turnover']):.2f}",
        "",
        "Costs by asset group:",
    ]
    if cost_by_group.empty:
        transaction_cost_report_lines.append("- none")
    else:
        for row in cost_by_group.itertuples(index=False):
            transaction_cost_report_lines.append(
                f"- {row.assumption_bucket}: orders={int(row.orders)}, order_value={float(row.total_order_value):.2f}, "
                f"total_cost={float(row.total_cost):.2f}, max_cost_bps={float(row.max_cost_bps):.2f}"
            )
    transaction_cost_report_lines.extend(
        [
            "",
            f"modeled_transaction_costs_usd: {float(cost_review['modeled_transaction_costs_usd']):.2f}",
            f"modeled_transaction_costs_pct_nav: {float(cost_review['modeled_transaction_costs_pct_nav']):.6f}",
            f"total_estimated_order_cost: {float(cost_review['total_transaction_costs_usd']):.2f}",
            f"total_estimated_order_cost_bps_of_nav: {float(cost_review['total_transaction_costs_pct_nav']) * 10000.0:.2f}",
            f"trade_now_edge_after_modeled_costs: {float(cost_review['trade_now_edge_after_modeled_costs']):.6f}",
            f"trade_now_edge_without_direct_simulator_fees: {float(cost_review['trade_now_edge_without_direct_simulator_fees']):.6f}",
            f"teuerste_order_nach_kosten: {most_expensive_order_text}",
            f"hoechste_cost_bps_used: {highest_bps_order_text}",
            f"orders_skipped_wegen_min_order_value_usd: {int(final_order_cost_summary['skipped_small_orders'])}",
            f"cash_before_orders: {float(final_order_cost_summary['cash_before_orders']):.2f}",
            f"cash_after_orders: {float(final_order_cost_summary['cash_after_orders']):.2f}",
            f"cash_buffer: {float(final_order_cost_summary['cash_buffer_usd']):.2f}",
            f"no_negative_cash: {bool(final_order_cost_summary['no_negative_cash'])}",
        ]
    )
    _write_text(OUTPUT_DIR / "transaction_cost_report.txt", "\n".join(transaction_cost_report_lines) + "\n")

    analysis_lines = [hold_vs_target_text.rstrip(), "", "Appendix: continuous_candidate_scores"]
    for row in selection.scores_frame.itertuples(index=False):
        analysis_lines.append(
            f"- {row.candidate}: mean={row.mean_return:.6f}, median={row.median_return:.6f}, vol={row.volatility:.6f}, "
            f"cvar={row.cvar_5:.6f}, prob_loss={row.probability_loss:.4f}, robust={row.robust_score:.6f}, "
            f"cost={row.estimated_cost:.6f}, buffer={row.dynamic_buffer:.6f}, net={row.net_robust_score:.6f}, "
            f"delta_vs_hold={row.delta_vs_hold:.6f}, delta_vs_cash={row.delta_vs_cash:.6f}, "
            f"p_hold={row.probability_beats_hold:.4f}, p_cash={row.probability_beats_cash:.4f}, turnover={row.turnover:.6f}"
        )
    analysis_lines.extend(["", "Appendix: top_discrete_candidates"])
    for row in discrete_selection["scores_frame"].head(10).itertuples(index=False):
        analysis_lines.append(
            f"- {row.discrete_candidate}: net={row.net_robust_score:.6f}, robust={row.robust_score:.6f}, "
            f"cost={row.estimated_cost:.6f}, buffer={row.dynamic_buffer:.6f}, cash_left={row.cash_left:.2f}, "
            f"cash_weight={row.cash_weight:.4f}, delta_vs_hold={row.delta_vs_hold:.6f}, "
            f"valid_constraints={bool(row.valid_constraints)}, errors={row.validation_errors or 'none'}"
        )
    if selection.selected_candidate.name == "HOLD":
        analysis_lines.append("")
        analysis_lines.append("why_hold_plausible: HOLD was selected by the robust threshold logic after comparing net_robust_score, cost, dynamic buffer, and probability thresholds against HOLD and DEFENSIVE_CASH.")
    if best_discrete_candidate_name != continuous_model_optimal_candidate:
        analysis_lines.append("")
        analysis_lines.append(
            "continuous_vs_discrete_note: The continuous model-optimal candidate differed from the final discrete candidate. "
            "This can happen when whole-share constraints, cash left, or post-discretization validation change the ranking."
        )
    _write_text(OUTPUT_DIR / "hold_vs_target_analysis.txt", "\n".join(analysis_lines) + "\n")

    _write_csv(OUTPUT_DIR / "factor_forecasts.csv", factor_forecasts_df, index=False)
    _write_csv(OUTPUT_DIR / "factor_data.csv", factor_data_df)
    _write_csv(OUTPUT_DIR / "asset_factor_exposures.csv", exposure_matrix)
    _write_csv(OUTPUT_DIR / "conditional_scenario_summary.csv", conditional_summary, index=False)
    _write_text(OUTPUT_DIR / "factor_model_diagnostics.txt", "\n".join(factor_diagnostics or ["Conditional factor mode unavailable; using direct-only fallback."]))
    explainability_text = explain_selected_decision(
        final_selection,
        gate,
        data_quality_report=data_quality_report,
        model_confidence=model_governance,
        reconciliation_result=reconciliation_result,
        validation_result=validation_result,
    )
    explainability_text += "\n" + explain_factor_drivers(exposure_matrix, factor_forecasts_df)
    asset_change_explanations = explain_asset_changes(
        w_current=w_current,
        w_target=final_weights,
        forecast_table=forecast.table,
        exposure_matrix=exposure_matrix,
    )
    save_explainability_reports(
        explainability_text,
        asset_change_explanations,
        text_path=OUTPUT_DIR / "explainability_report.txt",
        csv_path=OUTPUT_DIR / "asset_change_explanations.csv",
    )
    write_audit_metadata(
        create_run_metadata(
            params=params,
            active_tickers=active_tickers,
            mode="daily_bot_single",
            removed_tickers=removed_tickers,
            data_start=str(prices.index.min().date()) if not prices.empty else None,
            data_end=str(prices.index.max().date()) if not prices.empty else None,
            execution_mode=execution_result.get("execution_mode", "order_preview_only"),
        ),
        output_path=OUTPUT_DIR / "audit_metadata.json",
    )

    decision_report = (
        f"Date: {as_of.date()}\n"
        f"Risk State: {risk_state}\n"
        f"Primary Regime: {regime_result.get('primary_regime', 'neutral')}\n"
        f"Mode: {factor_mode}\n"
        f"Current Portfolio Source: {current_state.source}\n"
        f"Current Portfolio Label: CURRENT_PORTFOLIO\n"
        f"Actual Cash Weight: {current_state.actual_cash_weight:.2%}\n"
        f"Run Context: {data_context.get('run_context', 'daily_bot_discrete_simulator')}\n"
        f"Synthetic Data: {data_context.get('synthetic_data', 'n/a')}\n"
        f"Data Source: {data_context.get('data_source', 'unknown')}\n"
        f"Cache Status: {data_context.get('cache_status', 'unknown')}\n"
        f"Latest Price Date: {data_context.get('latest_price_date', 'n/a')}\n"
        f"Expected Latest Trading Day: {data_context.get('expected_latest_trading_day', 'n/a')}\n"
        f"Staleness Days: {data_context.get('staleness_days', 'n/a')}\n"
        f"Data Freshness OK: {data_context.get('data_freshness_ok', 'n/a')}\n"
        f"yfinance Available: {data_context.get('yfinance_available', 'n/a')}\n"
        f"Tickers Loaded: {', '.join(map(str, data_context.get('tickers_loaded', []))) if data_context.get('tickers_loaded') else 'none'}\n"
        f"Tickers Failed: {', '.join(map(str, data_context.get('tickers_failed', []))) if data_context.get('tickers_failed') else 'none'}\n"
        f"Used Cache Fallback: {data_context.get('used_cache_fallback', 'n/a')}\n"
        f"Live Data Error: {data_context.get('live_data_error', '') or 'none'}\n"
        f"Price Basis: {data_context.get('price_basis', 'adjusted_close_proxy')}\n"
        f"Project Calendar Path: {data_context.get('project_calendar_path', DEFAULT_PROJECT_CALENDAR_PATH)}\n"
        f"Current Date Berlin: {data_context.get('current_date_berlin', 'n/a')}\n"
        f"Current Time Berlin: {data_context.get('current_time_berlin', 'n/a')}\n"
        f"Is Project Trading Day: {data_context.get('is_project_trading_day', 'n/a')}\n"
        f"Allowed Start Berlin: {data_context.get('allowed_start_berlin', 'n/a')}\n"
        f"Allowed End Berlin: {data_context.get('allowed_end_berlin', 'n/a')}\n"
        f"Within Allowed Window: {data_context.get('within_allowed_window', 'n/a')}\n"
        f"Execution Allowed By Calendar: {data_context.get('execution_allowed_by_calendar', 'n/a')}\n"
        f"Calendar Reason: {data_context.get('calendar_status', 'n/a')}\n"
        f"Factor Diagnostics: {'; '.join(factor_diagnostics) if factor_diagnostics else 'none'}\n"
        f"Active Tickers Count: {len(active_tickers)}\n"
        f"Removed Tickers: {', '.join(removed_tickers) if removed_tickers else 'none'}\n"
        f"Tradability Warnings: {'; '.join(tradability_df.loc[tradability_df['reason'] != 'ok', 'reason'].astype(str).tolist()) if (tradability_df['reason'] != 'ok').any() else 'none'}\n"
        f"Data Quality Score: {data_quality_report['global_data_quality_score']:.3f}\n"
        f"Model Confidence Score: {model_governance['model_confidence_score']:.3f}\n"
        "Final Allocation Method: scenario_weighted_rf_sharpe_solver\n"
        f"Final Target Source: {final_target_source}\n"
        f"Solver Failed: {final_target_source == FINAL_TARGET_SOURCE_SOLVER_FAILED}\n"
        f"Solver Failure Reason: {scenario_solver_result.constraint_diagnostics.get('failure_reason') or 'none'}\n"
        f"Post Solver Validation OK: {bool(solver_validation.get('ok', False))}\n"
        "Manual Candidate Selection For Final Target: False\n"
        "Legacy Candidate Ranking Used For Final Target: False\n"
        f"Direct Scenario Optimizer Report: {OUTPUT_DIR / 'direct_scenario_optimizer_report.txt'}\n"
        f"Direct Scenario Optimizer Allocation: {OUTPUT_DIR / 'direct_scenario_optimizer_allocation.csv'}\n"
        f"Continuous Selected Candidate: {selection.selected_candidate.name}\n"
        f"Continuous Model-Optimal Candidate: {continuous_model_optimal_candidate}\n"
        f"Continuous Model Target Label: CONTINUOUS_MODEL_TARGET\n"
        f"Best Discrete Candidate: {best_discrete_candidate_name}\n"
        f"Discrete Model Target Label: DISCRETE_MODEL_TARGET\n"
        f"Optimization Objective Used: {discrete_selection.get('objective_used', params.get('optimization_objective', 'robust_score'))}\n"
        f"Objective Score Column: {discrete_selection.get('objective_score_column', 'net_robust_score')}\n"
        f"Best Discrete Objective Score: {_safe_float(discrete_selection.get('best_discrete_objective_score'), float('nan')):.6f}\n"
        f"Highest Scenario Objective Candidate: {str(candidate_risk_return_frame.iloc[0]['candidate']) if not candidate_risk_return_frame.empty else 'none'}\n"
        f"Highest Scenario Objective Score: {_safe_float(candidate_risk_return_frame.iloc[0]['robust_scenario_sharpe_objective'] if not candidate_risk_return_frame.empty else float('nan'), float('nan')):.6f}\n"
        f"Scenario Risk/Return Report: {OUTPUT_DIR / 'candidate_risk_return_report.csv'}\n"
        f"Selected Reason: {selected_reason}\n"
        f"Hold Current Constraint Valid: {bool(discrete_selection.get('hold_current_constraint_valid', True))}\n"
        f"Current Portfolio Constraint Valid: {bool(current_constraint_validation.get('ok', True))}\n"
        f"Current Portfolio Constraint Violation: {not bool(current_constraint_validation.get('ok', True))}\n"
        f"Current Constraint Errors: {current_constraint_errors_text if current_constraint_errors_text else 'none'}\n"
        f"Current Portfolio Asset Limit Violations: {len(current_constraint_validation.get('asset_limit_violations', []))}\n"
        f"Current Portfolio Group Limit Violations: {len(current_constraint_validation.get('group_limit_violations', []))}\n"
        f"Best Non-HOLD Candidate: {discrete_selection.get('best_non_hold_candidate', '') or 'none'}\n"
        f"Best Non-HOLD Score: {_safe_float(discrete_selection.get('best_non_hold_score'), float('nan')):.6f}\n"
        f"Best Non-HOLD Valid Constraints: {bool(discrete_selection.get('best_non_hold_valid_constraints', False))}\n"
        f"Best Non-HOLD Failed Reason: {discrete_selection.get('best_non_hold_failed_reason', '') or 'none'}\n"
        f"Best Model Candidate: {discrete_selection.get('best_model_candidate', '') or 'none'}\n"
        f"Best Model Candidate Valid Constraints: {bool(discrete_selection.get('best_model_candidate_valid_constraints', False))}\n"
        f"Final Selection Is Safe Fallback: {bool(discrete_selection.get('final_selection_is_safe_fallback', False))}\n"
        f"Compliance Repair Available: {compliance_repair_available}\n"
        f"Compliance Repair Order Count: {compliance_repair_order_count}\n"
        f"Compliance Repair Turnover: {compliance_repair_turnover:.6f}\n"
        f"Compliance Repair Score: {compliance_repair_score:.6f}\n"
        f"Compliance Repair Delta vs HOLD: {compliance_repair_delta_vs_hold:.6f}\n"
        f"Compliance Repair Fixes Current Constraints: {compliance_repair_fixes_current_constraints}\n"
        + (
            "HOLD_CURRENT Note: current portfolio selected as safe HOLD fallback, but it has constraint violations.\n"
            if selected_reason == "constraint_invalid_hold_fallback"
            else ""
        )
        + f"Selection Utility (Net Robust Score): {discrete_selected_score.net_robust_score:.6f}\n"
        f"Robust Score (Pre-Cost/Buffer): {discrete_selected_score.robust_score:.6f}\n"
        f"Current Portfolio Score: {current_portfolio_score:.6f}\n"
        f"Delta Score vs Current Portfolio: {delta_score:.6f}\n"
        f"Gate Status: {gate.gate_status}\n"
        f"Action: {gate.action}\n"
        f"Final Action Label: FINAL_ACTION\n"
        f"Reason: {gate.reason}\n"
        f"Delta vs HOLD: {discrete_selected_score.delta_vs_hold:.6f}\n"
        f"Delta vs CASH: {discrete_selected_score.delta_vs_cash:.6f}\n"
        f"P(beats HOLD): {discrete_selected_score.probability_beats_hold:.2%}\n"
        f"P(beats CASH): {discrete_selected_score.probability_beats_cash:.2%}\n"
        f"CVaR 5: {discrete_selected_score.cvar_5:.2%}\n"
        f"Worst Scenario: {discrete_selected_score.worst_scenario}\n"
        f"Turnover: {discrete_selected_score.turnover:.2%}\n"
        f"Estimated Cost: {discrete_selected_score.estimated_cost:.6f}\n"
        f"Estimated Total Order Cost USD: {discrete_selected_score.estimated_total_order_cost:.2f}\n"
        f"Estimated Spread Cost USD: {discrete_selected_score.estimated_spread_cost:.2f}\n"
        f"Estimated Slippage Cost USD: {discrete_selected_score.estimated_slippage_cost:.2f}\n"
        f"Cost Model Used: {discrete_selected_score.cost_model_used}\n"
        f"Live Costs Available: {discrete_selected_score.live_costs_available}\n"
        f"TradeNowScore: {gate.trade_now_score:.6f}\n"
        f"Suggested Trade Fraction: {trade_sizing['trade_fraction']:.2f}\n"
        f"Suggested Action: {trade_sizing['suggested_action']}\n"
        f"Order Count: {actionable_order_count}\n"
        f"Discrete Cash Left: {best_discrete_candidate.cash_left:.2f}\n"
        f"Pre-Trade Validation Status: {'PASS' if validation_result['ok'] else 'FAIL'}\n"
        f"Pre-Trade Validation Warnings: {'; '.join(validation_result['warnings']) if validation_result['warnings'] else 'none'}\n"
        f"Pre-Trade Validation Errors: {'; '.join(validation_result['errors']) if validation_result['errors'] else 'none'}\n"
        f"Reconciliation Status: {reconciliation_result.get('status', 'SKIP')}\n"
        f"Execution Mode: {execution_result['execution_mode']}\n"
        f"Execution Message: {execution_result['message']}\n"
        f"Blocked Orders: {len(validation_result['blocked_orders'])}\n"
        f"Final Discrete Order Preview File: {OUTPUT_DIR / 'best_discrete_order_preview.csv'}\n"
        f"Manual Simulator Orders CSV: {OUTPUT_DIR / 'manual_simulator_orders.csv'}\n"
        f"Manual Simulator Orders TXT: {OUTPUT_DIR / 'manual_simulator_orders.txt'}\n"
        f"Active Preview Orders CSV: {OUTPUT_DIR / 'active_preview_orders.csv'}\n"
        f"Active Preview Decision Report: {OUTPUT_DIR / 'active_preview_decision_report.txt'}\n"
        "Research Preview Warning: outputs/research_order_preview.csv belongs to the main.py research/backtest path only; outputs/order_preview.csv is only a legacy alias and neither file is the final simulator order file.\n"
        "\n"
        f"{safe_active_report_text}\n"
        f"{explainability_text}\n"
    )
    _write_text(OUTPUT_DIR / "daily_bot_decision_report.txt", decision_report)
    _write_text(OUTPUT_DIR / "latest_decision_report.txt", decision_report)
    final_acceptance_snapshot = (
        "Daily Bot Final Acceptance Snapshot\n"
        "\n"
        f"final_action: {gate.action}\n"
        f"execution_mode: {execution_result.get('execution_mode', 'unknown')}\n"
        f"final_target_source: {final_target_source}\n"
        f"solver_failed: {final_target_source == FINAL_TARGET_SOURCE_SOLVER_FAILED}\n"
        f"failure_reason: {scenario_solver_result.constraint_diagnostics.get('failure_reason') or 'none'}\n"
        f"post_solver_validation_ok: {bool(solver_validation.get('ok', False))}\n"
        "legacy_candidate_ranking_used_for_final_target: false\n"
        f"final_discrete_candidate: {best_discrete_candidate_name}\n"
        f"objective_used: {discrete_selection.get('objective_used', params.get('optimization_objective', 'robust_score'))}\n"
        f"best_discrete_objective_score: {_safe_float(discrete_selection.get('best_discrete_objective_score'), float('nan')):.6f}\n"
        f"highest_scenario_objective_candidate: {str(candidate_risk_return_frame.iloc[0]['candidate']) if not candidate_risk_return_frame.empty else 'none'}\n"
        f"candidate_risk_return_report_csv: {OUTPUT_DIR / 'candidate_risk_return_report.csv'}\n"
        f"selected_reason: {selected_reason}\n"
        f"hold_current_constraint_valid: {bool(discrete_selection.get('hold_current_constraint_valid', True))}\n"
        f"current_portfolio_constraint_valid: {bool(current_constraint_validation.get('ok', True))}\n"
        f"current_portfolio_constraint_violation: {not bool(current_constraint_validation.get('ok', True))}\n"
        f"current_constraint_errors: {current_constraint_errors_text if current_constraint_errors_text else 'none'}\n"
        f"best_non_hold_candidate: {discrete_selection.get('best_non_hold_candidate', '') or 'none'}\n"
        f"best_non_hold_score: {_safe_float(discrete_selection.get('best_non_hold_score'), float('nan')):.6f}\n"
        f"best_non_hold_valid_constraints: {bool(discrete_selection.get('best_non_hold_valid_constraints', False))}\n"
        f"best_non_hold_failed_reason: {discrete_selection.get('best_non_hold_failed_reason', '') or 'none'}\n"
        f"best_model_candidate: {discrete_selection.get('best_model_candidate', '') or 'none'}\n"
        f"best_model_candidate_valid_constraints: {bool(discrete_selection.get('best_model_candidate_valid_constraints', False))}\n"
        f"final_selection_is_safe_fallback: {bool(discrete_selection.get('final_selection_is_safe_fallback', False))}\n"
        f"compliance_repair_available: {compliance_repair_available}\n"
        f"compliance_repair_order_count: {compliance_repair_order_count}\n"
        f"compliance_repair_turnover: {compliance_repair_turnover:.6f}\n"
        f"compliance_repair_score: {compliance_repair_score:.6f}\n"
        f"compliance_repair_delta_vs_hold: {compliance_repair_delta_vs_hold:.6f}\n"
        f"compliance_repair_fixes_current_constraints: {compliance_repair_fixes_current_constraints}\n"
        f"manual_simulator_orders_csv: {OUTPUT_DIR / 'manual_simulator_orders.csv'}\n"
        f"constraint_pressure_report_csv: {OUTPUT_DIR / 'constraint_pressure_report.csv'}\n"
        f"active_preview_decision_report_txt: {OUTPUT_DIR / 'active_preview_decision_report.txt'}\n"
        f"active_preview_orders_csv: {OUTPUT_DIR / 'active_preview_orders.csv'}\n"
        f"active_preview_action: {active_preview_summary.get('active_preview_action', 'HOLD')}\n"
        f"active_preview_candidate: {active_preview_summary.get('active_preview_candidate', 'HOLD_CURRENT')}\n"
        f"active_preview_executable: {bool(active_preview_summary.get('active_preview_executable', False))}\n"
    )
    _write_text(OUTPUT_DIR / "final_acceptance_report.txt", final_acceptance_snapshot)
    today_decision_summary = _build_today_decision_summary(
        as_of=as_of,
        data_freshness=data_freshness,
        price_attrs=dict(prices.attrs),
        current_state=current_state,
        continuous_model_optimal_candidate=continuous_model_optimal_candidate,
        continuous_target_weights=continuous_target_weights.reindex(active_tickers).fillna(0.0),
        best_discrete_candidate_name=best_discrete_candidate_name,
        best_discrete_candidate=best_discrete_candidate,
        manual_simulator_orders=manual_simulator_orders,
        market_gate=market_gate,
        gate=gate,
        execution_result=execution_result,
        hold_vs_target_summary=hold_vs_target_summary,
        delta_score=delta_score,
        current_portfolio_score=current_portfolio_score,
        best_discrete_score=float(discrete_selection["best_discrete_score"]),
        cash_before_orders=float(final_order_cost_summary["cash_before_orders"]),
        cash_after_orders=float(final_order_cost_summary["cash_after_orders"]),
        cost_review=cost_review,
        review_issues=review_issues,
        active_preview_summary=active_preview_summary,
    )
    _write_text(OUTPUT_DIR / "today_decision_summary.txt", today_decision_summary)
    diagnostics.model_context["daily_review_payload"] = daily_review_payload
    write_daily_portfolio_review_outputs(
        diagnostics,
        output_dir=OUTPUT_DIR,
        email_result={"sent": False, "reason": "preview_only", "error": None},
    )
    write_output_file_guide(OUTPUT_DIR / "output_file_guide.txt")

    rejected_mask = pd.Series(False, index=adjusted_order_preview.index, dtype=bool)
    if "not_executable" in adjusted_order_preview.columns:
        rejected_mask = rejected_mask | adjusted_order_preview["not_executable"].fillna(False).astype(bool)
    if "execution_block_reason" in adjusted_order_preview.columns:
        rejected_mask = rejected_mask | adjusted_order_preview["execution_block_reason"].astype(str).str.strip().ne("")
    if "reason" in adjusted_order_preview.columns:
        rejected_mask = rejected_mask | adjusted_order_preview["reason"].astype(str).str.strip().ne("")

    for row in adjusted_order_preview.loc[rejected_mask].itertuples(index=False):
        reason = str(getattr(row, "execution_block_reason", "") or getattr(row, "reason", "") or "").strip()
        if reason:
            log_rejected_order(
                diagnostics,
                str(getattr(row, "ticker", "UNKNOWN")),
                str(getattr(row, "side", "HOLD")),
                reason,
                extra={"order_shares": getattr(row, "order_shares", None)},
            )

    diagnostics.final_orders_summary = {
        "order_count": actionable_order_count,
        "turnover": float(discrete_selected_score.turnover),
        "estimated_cost": float(discrete_selected_score.estimated_cost),
        "estimated_cost_usd": float(final_order_cost_summary["total_estimated_transaction_cost"]),
        "estimated_cost_pct_nav": float(final_order_cost_summary["total_order_cost_pct_nav"]),
        "cash_before": float(final_order_cost_summary["cash_before_orders"]),
        "cash_after": float(final_order_cost_summary["cash_after_orders"]),
    }
    diagnostics.execution_mode = str(execution_result.get("execution_mode", "unknown"))
    log_final_action(diagnostics, gate.action, selected_candidate=best_discrete_candidate_name, reason=gate.reason)
    diagnostics.candidate_context.update(
        {
            "net_robust_score": float(discrete_selected_score.net_robust_score),
            "delta_vs_hold": float(discrete_selected_score.delta_vs_hold),
            "delta_vs_cash": float(discrete_selected_score.delta_vs_cash),
            "probability_beats_hold": float(discrete_selected_score.probability_beats_hold),
            "probability_beats_cash": float(discrete_selected_score.probability_beats_cash),
            "worst_scenario": discrete_selected_score.worst_scenario,
            "cvar_5": float(discrete_selected_score.cvar_5),
            "trade_now_edge": float(trade_edge_summary["trade_now_edge"]),
        }
    )
    log_stage(diagnostics, "report writing", "DONE", extra={"execution_mode": diagnostics.execution_mode})

    executed_orders = int(execution_result.get("orders_submitted", 0))
    state_payload = update_state_after_execution(
        state_payload,
        executed_orders=executed_orders,
        turnover=float(discrete_selected_score.turnover),
        timestamp=datetime.now(timezone.utc).isoformat() if executed_orders > 0 else None,
        decision_id=decision_id,
        order_signature=order_signature,
        selected_candidate=best_discrete_candidate_name,
        trade_now_score=float(gate.trade_now_score),
        execution_status=str(execution_result.get("execution_mode", "unknown")),
        execution_mode=str(execution_result.get("execution_mode", "unknown")),
    )
    state_payload["turnover_budget_remaining"] = compute_turnover_budget_remaining(state_payload, params)
    save_daily_bot_state(DAILY_BOT_STATE_PATH, state_payload)

    try:
        init_db(params["db_path"])
        run_id = create_run(params["db_path"], params=params, tickers=active_tickers)
        save_tradability_to_db(params["db_path"], run_id, tradability_df)
        save_data_quality_to_db(params["db_path"], run_id, data_quality_report)
        save_execution_result(params["db_path"], run_id, execution_result)
    except Exception as exc:
        LOGGER.warning("Daily bot SQLite persistence warning: %s", exc)

    LOGGER.info(
        "Daily bot single-run completed with mode=%s, final_target_source=%s, continuous_candidate=%s, final_candidate=%s.",
        factor_mode,
        final_target_source,
        continuous_model_optimal_candidate,
        best_discrete_candidate_name,
    )
    return {
        "as_of": as_of,
        "factor_mode": factor_mode,
        "continuous_candidate": continuous_model_optimal_candidate,
        "final_allocation_method": "scenario_weighted_rf_sharpe_solver",
        "final_target_source": final_target_source,
        "selected_candidate": best_discrete_candidate_name,
        "gate_action": gate.action,
        "execution_mode": execution_result["execution_mode"],
    }


def main() -> None:
    setup_logging()
    _ensure_output_dirs()
    args = parse_args()
    acquired, message = _acquire_daily_bot_lock()
    if not acquired:
        LOGGER.warning("%s", message)
        _write_text(OUTPUT_DIR / "daily_bot_decision_report.txt", message + "\n")
        print("Daily bot completed successfully.")
        print(f"message: {message}")
        return

    try:
        if args.mode == "continuous":
            LOGGER.info("Starting daily bot in continuous dry-run mode.")
            try:
                last_full_recompute = 0.0
                while True:
                    now = time.time()
                    if now - last_full_recompute >= args.full_recompute_interval_minutes * 60:
                        _run_single(args)
                        last_full_recompute = now
                    time.sleep(max(args.check_interval_minutes, 1) * 60)
            except KeyboardInterrupt:
                LOGGER.info("Daily bot continuous mode stopped by user.")
                return

        result = _run_single(args)
        print("Daily bot completed successfully.")
        for key, value in result.items():
            print(f"{key}: {value}")
    finally:
        _release_daily_bot_lock()


if __name__ == "__main__":
    main()
