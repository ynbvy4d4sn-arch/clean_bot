"""Command-line entry point for the robust 3M active allocation optimizer."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from asset_universe import validate_asset_universe
from audit import create_run_metadata, write_audit_metadata
from backtest import run_backtest
from calendar_utils import DEFAULT_PROJECT_CALENDAR_PATH, is_within_project_trading_window
from codex_report import write_codex_debug_report, write_codex_next_prompt
from config import DATA_DIR, DEFAULT_PORTFOLIO_VALUE, OUTPUT_DIR, PRICE_CACHE_PATH, build_params
from config_validation import validate_config
from data import build_run_data_context, check_data_freshness, load_price_data, write_data_freshness_report
from data_quality import compute_data_quality_report, save_data_quality_report
from database import create_run, init_db, save_data_quality_to_db, save_full_run
from daily_analysis_report import send_daily_analysis_email_if_needed, write_daily_analysis_report
from diagnostics import (
    create_run_diagnostics,
    detect_performance_flags,
    log_data_context,
    log_data_quality,
    log_error,
    log_final_action,
    log_stage,
    log_warning,
    write_run_diagnostics,
)
from feasibility import check_portfolio_feasibility
from features import compute_returns
from metrics import number_of_trades, performance_summary
from notifications import build_email_body, write_latest_notification
from order_preview import generate_order_preview, mark_research_preview
from paper_broker_stub import PaperBrokerStub
from pre_trade_validation import run_pre_trade_validation, save_pre_trade_validation_report
from reconciliation import build_reconciliation_report, reconcile_before_execution
from report import (
    create_decision_summary,
    create_report_artifacts,
    write_latest_decision_report,
    write_output_file_guide,
)
from simulator_orchestrator import run_execution_layer
from system_init import run_system_initialization
from tradability import (
    apply_tradability_filter,
    build_tradability_report,
    save_tradability_report,
    save_tradability_to_db,
    select_cash_proxy,
)


LOGGER = logging.getLogger(__name__)
AUXILIARY_MARKET_TICKERS = ("SPY",)
MIN_REQUIRED_ASSETS = 10


def setup_logging() -> None:
    """Configure a simple console logger for the run."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the robust 3M active allocation optimizer research pipeline."
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=float(DEFAULT_PORTFOLIO_VALUE),
        help="Portfolio value used for the order preview CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory where reports and previews should be written.",
    )
    parser.add_argument(
        "--skip-email",
        action="store_true",
        help="Skip SMTP sending even if triggers are met. The latest email text is still written.",
    )
    parser.add_argument(
        "--send-analysis-email",
        action="store_true",
        help="Allow the new research analysis email path. Default remains local-report only.",
    )
    return parser.parse_args()


def ensure_directories(output_dir: str | Path) -> None:
    """Ensure the required output and data directories exist."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _write_text_atomic(path: Path, text: str) -> None:
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


def _write_csv_atomic(path: Path, frame: pd.DataFrame, **kwargs: object) -> None:
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


def build_data_tickers(investable_tickers: list[str]) -> list[str]:
    """Return investable plus auxiliary market tickers for data loading."""

    return list(dict.fromkeys([*investable_tickers, *AUXILIARY_MARKET_TICKERS]))


def infer_available_investable_tickers(prices: pd.DataFrame, requested_tickers: list[str]) -> list[str]:
    """Return investable tickers that are actually present in the cleaned price matrix."""

    available = [ticker for ticker in requested_tickers if ticker in prices.columns]
    missing = [ticker for ticker in requested_tickers if ticker not in prices.columns]
    if missing:
        LOGGER.warning("Proceeding without these unavailable tickers: %s", ", ".join(missing))
    if len(available) < MIN_REQUIRED_ASSETS:
        raise ValueError(
            f"Too few investable tickers remain after data loading: {len(available)} available, "
            f"at least {MIN_REQUIRED_ASSETS} required."
        )
    return available


def build_performance_frame(result: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the performance summary table used for reporting and SQLite."""

    daily = result["daily"].copy()
    benchmark_returns = result["benchmark_returns"].copy()
    returns_dict: dict[str, pd.Series] = {
        "strategy": daily.set_index("date")["portfolio_return_net"]
    }
    for column in benchmark_returns.columns:
        returns_dict[column] = benchmark_returns[column]
    daily_records_dict = {"strategy": daily}
    return performance_summary(returns_dict, daily_records_dict=daily_records_dict)


def expected_execution_mode(params: dict[str, object]) -> str:
    """Infer the intended execution mode before the orchestrator runs."""

    if bool(params.get("dry_run", True)):
        return "order_preview_only"
    if bool(params.get("enable_investopedia_simulator", False)):
        return "investopedia"
    if bool(params.get("enable_local_paper_trading", False)):
        return "local_paper"
    return "order_preview_only"


def print_console_summary(result: dict[str, pd.DataFrame], perf_summary: pd.DataFrame) -> None:
    """Print performance, decision and latest-decision summaries to the console."""

    daily = result["daily"].copy()
    decision_summary = create_decision_summary(daily)
    latest_record = daily.iloc[-1]

    print("\nPerformance Summary")
    print(perf_summary.to_string(index=False))

    print("\nDecision Summary")
    print(decision_summary.to_string(index=False))

    print("\nLatest Decision")
    latest_fields = [
        ("date", latest_record["date"]),
        ("decision", latest_record["decision"]),
        ("risk_state", latest_record["risk_state"]),
        ("net_benefit", latest_record["net_benefit"]),
        ("delta_score", latest_record["delta_score"]),
        ("realized_turnover", latest_record["realized_turnover"]),
        ("estimated_cost", latest_record["estimated_cost"]),
        ("buffer", latest_record["buffer"]),
        ("target_vol", latest_record["target_vol"]),
        ("cvar_95", latest_record["cvar_95"]),
        ("max_drawdown", latest_record["max_drawdown"]),
        ("weekly_rebalance_day", latest_record["weekly_rebalance_day"]),
        ("emergency_condition", latest_record["emergency_condition"]),
        ("risk_gate_failed", latest_record["risk_gate_failed"]),
    ]
    for label, value in latest_fields:
        print(f"{label}: {value}")
    execution_result = result.get("execution_result", {})
    if execution_result:
        print(f"execution_mode: {execution_result.get('execution_mode', 'unknown')}")
        print(f"execution_message: {execution_result.get('message', '')}")


def write_pause_outputs(
    *,
    output_dir: Path,
    reason: str,
    active_tickers: list[str],
    removed_tickers: list[str],
    notes: list[str] | None = None,
    data_context: dict[str, object] | None = None,
) -> None:
    """Write minimal pause artifacts when the run must stop before optimization."""

    notes = list(notes or [])
    report_lines = [
        f"Date: {pd.Timestamp.now().date()}",
        "Decision: PAUSE",
        "Risk State: unknown",
        f"Reason: {reason}",
        "Execution Mode: order_preview_only",
        f"Active Tickers Count: {len(active_tickers)}",
        f"Removed Tickers: {', '.join(removed_tickers) if removed_tickers else 'none'}",
    ]
    if notes:
        report_lines.append("Notes:")
        report_lines.extend(f"- {note}" for note in notes)
    if data_context:
        report_lines.extend(
            [
                f"Run Context: {data_context.get('run_context', 'research_backtest')}",
                f"Data Source: {data_context.get('data_source', 'n/a')}",
                f"Cache Status: {data_context.get('cache_status', 'n/a')}",
                f"Synthetic Data: {data_context.get('synthetic_data', 'n/a')}",
                f"Latest Price Date: {data_context.get('latest_price_date', 'n/a')}",
                f"Expected Latest Trading Day: {data_context.get('expected_latest_trading_day', 'n/a')}",
                f"Staleness Days: {data_context.get('staleness_days', 'n/a')}",
                f"Data Freshness OK: {data_context.get('data_freshness_ok', 'n/a')}",
                f"yfinance Available: {data_context.get('yfinance_available', 'n/a')}",
                f"Tickers Loaded: {', '.join(data_context.get('tickers_loaded', [])) if data_context.get('tickers_loaded') else 'none'}",
                f"Tickers Failed: {', '.join(data_context.get('tickers_failed', [])) if data_context.get('tickers_failed') else 'none'}",
                f"Used Cache Fallback: {data_context.get('used_cache_fallback', 'n/a')}",
                f"Live Data Error: {data_context.get('live_data_error', 'n/a') or 'none'}",
                f"Price Basis: {data_context.get('price_basis', 'n/a')}",
                f"Project Calendar Path: {data_context.get('project_calendar_path', 'n/a')}",
                f"Current Date Berlin: {data_context.get('current_date_berlin', 'n/a')}",
                f"Current Time Berlin: {data_context.get('current_time_berlin', 'n/a')}",
                f"Is Project Trading Day: {data_context.get('is_project_trading_day', 'n/a')}",
                f"Allowed Start Berlin: {data_context.get('allowed_start_berlin', 'n/a')}",
                f"Allowed End Berlin: {data_context.get('allowed_end_berlin', 'n/a')}",
                f"Within Allowed Window: {data_context.get('within_allowed_window', 'n/a')}",
                f"Execution Allowed By Calendar: {data_context.get('execution_allowed_by_calendar', 'n/a')}",
                f"Calendar Reason: {data_context.get('calendar_reason', 'n/a')}",
            ]
        )
    report_lines.append("Next Step: Review optimizer constraints, tradability, and feasibility inputs.")

    body = "\n".join(report_lines) + "\n"
    _write_text_atomic(output_dir / "research_latest_decision_report.txt", body)
    write_latest_notification(body=body, output_dir=output_dir)


def _write_minimal_research_reports(diagnostics, output_dir: Path) -> None:
    lines = [
        f"Run ID: {diagnostics.run_id}",
        f"Mode: {diagnostics.mode}",
        f"Dry Run: {diagnostics.dry_run}",
        f"Signal Date: {diagnostics.signal_date or 'n/a'}",
        f"Execution Date: {diagnostics.execution_date or 'n/a'}",
        f"Final Action: {diagnostics.final_action}",
        f"Selected Candidate: {diagnostics.selected_candidate}",
        f"Execution Mode: {diagnostics.execution_mode}",
        f"Reason: {diagnostics.final_reason or 'n/a'}",
        f"Run Context: {diagnostics.data_context.get('run_context', 'research_backtest')}",
        f"Data Source: {diagnostics.data_context.get('data_source', 'n/a')}",
        f"Latest Price Date: {diagnostics.data_context.get('latest_price_date', 'n/a')}",
        f"Expected Latest Trading Day: {diagnostics.data_context.get('expected_latest_trading_day', 'n/a')}",
        f"Warnings: {len(diagnostics.warnings)}",
        f"Errors: {len(diagnostics.errors)}",
    ]
    text = "\n".join(lines) + "\n"
    _write_text_atomic(output_dir / "research_latest_decision_report.txt", text)
    write_latest_notification(body=text, output_dir=output_dir)


def _finalize_research_diagnostics(diagnostics, output_dir: Path, allow_email: bool) -> None:
    try:
        detect_performance_flags(diagnostics)
    except Exception as exc:  # pragma: no cover - defensive
        log_warning(diagnostics, "main", f"detect_performance_flags failed: {exc}", stage="report_writing")

    try:
        if not (output_dir / "research_latest_decision_report.txt").exists():
            _write_minimal_research_reports(diagnostics, output_dir)
    except Exception as exc:  # pragma: no cover - defensive
        log_warning(diagnostics, "main", f"minimal research report fallback failed: {exc}", stage="report_writing")

    try:
        write_run_diagnostics(diagnostics, output_dir=output_dir)
        write_codex_debug_report(diagnostics, output_path=output_dir / "research_codex_debug_report.md")
        write_codex_next_prompt(diagnostics, output_path=output_dir / "research_codex_next_prompt.md")
        write_daily_analysis_report(diagnostics, output_path=output_dir / "research_analysis_report.md")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Research diagnostics write failed: %s", exc)

    if allow_email:
        try:
            email_result = send_daily_analysis_email_if_needed(diagnostics)
            log_stage(diagnostics, "email sending", "DONE", extra=email_result)
            if email_result.get("error"):
                log_warning(diagnostics, "main", f"Research analysis email send failed: {email_result.get('reason')}", stage="email_sending", extra=email_result)
        except Exception as exc:  # pragma: no cover - defensive
            log_error(diagnostics, "main", "send_daily_analysis_email_if_needed", exc, stage="email_sending")

    try:
        write_run_diagnostics(diagnostics, output_dir=output_dir)
        write_codex_debug_report(diagnostics, output_path=output_dir / "research_codex_debug_report.md")
        write_codex_next_prompt(diagnostics, output_path=output_dir / "research_codex_next_prompt.md")
        write_daily_analysis_report(diagnostics, output_path=output_dir / "research_analysis_report.md")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Research diagnostics rewrite failed: %s", exc)


def _main_impl(args: argparse.Namespace, diagnostics) -> None:
    """Run the complete research, reporting, order-preview and notification workflow."""

    output_dir = Path(args.output_dir)
    ensure_directories(output_dir)
    LOGGER.info("Starting optimizer run.")
    diagnostics.dry_run = True
    diagnostics.execution_mode = "order_preview_only"
    log_stage(diagnostics, "system initialization", "START")

    validate_asset_universe()

    initial_params = build_params()
    log_stage(diagnostics, "config validation", "START")
    config_check = validate_config(initial_params)
    for warning in config_check["warnings"]:
        LOGGER.warning("Config validation warning: %s", warning)
        log_warning(diagnostics, "main", warning, stage="config_validation")
    if not config_check["ok"]:
        for error in config_check["errors"]:
            LOGGER.error("Config validation error: %s", error)
            log_warning(diagnostics, "main", error, severity="ERROR", stage="config_validation")
        log_final_action(diagnostics, "PAUSE", selected_candidate="HOLD", reason="Configuration validation failed; aborting before optimizer run.")
        raise SystemExit("Configuration validation failed; aborting before optimizer run.")
    log_stage(diagnostics, "config validation", "DONE")
    init_result = run_system_initialization(initial_params)
    for warning in init_result["warnings"]:
        LOGGER.warning("System initialization warning: %s", warning)
        log_warning(diagnostics, "main", warning, stage="system_initialization")
    for error in init_result["errors"]:
        LOGGER.warning("System initialization error: %s", error)
        log_warning(diagnostics, "main", error, severity="ERROR", stage="system_initialization")
    LOGGER.info(
        "Run mode: DRY_RUN=%s | ENABLE_LOCAL_PAPER_TRADING=%s | ENABLE_INVESTOPEDIA_SIMULATOR=%s | ENABLE_EXTERNAL_BROKER=%s",
        bool(initial_params.get("dry_run", True)),
        bool(initial_params.get("enable_local_paper_trading", False)),
        bool(initial_params.get("enable_investopedia_simulator", False)),
        bool(initial_params.get("enable_external_broker", False)),
    )
    log_stage(diagnostics, "system initialization", "DONE", extra={"dry_run": bool(initial_params.get("dry_run", True))})
    requested_tickers = list(initial_params["tickers"])
    data_tickers = build_data_tickers(requested_tickers)
    LOGGER.info("Loading prices for %s requested symbols.", len(data_tickers))

    log_stage(diagnostics, "data loading", "START")
    prices = load_price_data(
        tickers=data_tickers,
        start_date=str(initial_params["start_date"]),
        end_date=initial_params["end_date"],
        cache_path=PRICE_CACHE_PATH,
        use_cache=True,
        prefer_live=True,
        allow_cache_fallback=True,
        force_refresh=False,
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
        run_context="research_backtest",
    ).as_dict()
    log_data_context(diagnostics, attrs=data_context)
    diagnostics.signal_date = str(data_context.get("latest_price_date") or "")
    diagnostics.execution_date = None
    write_data_freshness_report(
        prices=prices,
        freshness=data_freshness,
        output_path=output_dir / "research_current_data_freshness_report.txt",
        market_gate=market_gate,
        data_context=data_context,
    )
    data_notes = list(prices.attrs.get("data_notes", []))
    data_removed_tickers = list(prices.attrs.get("removed_tickers", []))
    if data_freshness.get("warning"):
        LOGGER.warning("%s", data_freshness["warning"])
        log_warning(diagnostics, "main", str(data_freshness["warning"]), stage="data_freshness")
    if data_notes:
        for note in data_notes:
            LOGGER.warning("%s", note)
            log_warning(diagnostics, "main", str(note), stage="data_loading")
    log_stage(diagnostics, "data freshness", "DONE", extra=data_context)

    infer_available_investable_tickers(prices, requested_tickers)
    log_stage(diagnostics, "tradability", "START")
    tradability_df = build_tradability_report(
        tickers=requested_tickers,
        prices=prices,
        enable_local_paper=bool(initial_params.get("enable_local_paper_trading", False)),
        enable_investopedia=bool(initial_params.get("enable_investopedia_simulator", False)),
        dry_run=bool(initial_params.get("dry_run", True)),
    )
    save_tradability_report(
        tradability_df=tradability_df,
        output_path=output_dir / "tradability_report.csv",
    )
    available_investable_tickers = apply_tradability_filter(
        tickers=requested_tickers,
        tradability_df=tradability_df,
        min_assets=MIN_REQUIRED_ASSETS,
    )
    log_stage(diagnostics, "tradability", "DONE", extra={"tradable_count": len(available_investable_tickers)})
    removed_tickers = [
        ticker for ticker in requested_tickers if ticker not in available_investable_tickers
    ]
    if removed_tickers:
        LOGGER.warning(
            "These investable tickers were removed for this run due to missing price data: %s",
            ", ".join(removed_tickers),
        )
    auxiliary_removed_tickers = [
        ticker for ticker in data_removed_tickers if ticker not in requested_tickers
    ]
    if auxiliary_removed_tickers:
        LOGGER.warning(
            "These auxiliary market tickers were unavailable in this run: %s",
            ", ".join(auxiliary_removed_tickers),
        )
    params = build_params(tickers=available_investable_tickers)
    effective_cash_ticker = select_cash_proxy(available_investable_tickers, tradability_df)
    if effective_cash_ticker is not None:
        params["cash_ticker"] = effective_cash_ticker
        params["effective_cash_ticker"] = effective_cash_ticker
    quality_prices = prices.reindex(columns=available_investable_tickers).copy()
    quality_prices.attrs.update(prices.attrs)
    log_stage(diagnostics, "data quality", "START")
    data_quality_report = compute_data_quality_report(
        prices=quality_prices,
        returns=compute_returns(quality_prices),
        active_tickers=available_investable_tickers,
        params=params,
    )
    save_data_quality_report(
        report=data_quality_report,
        output_path=output_dir / "data_quality_report.csv",
    )
    log_data_quality(diagnostics, data_quality_report)
    for warning in data_quality_report["warnings"]:
        LOGGER.warning("Data quality warning: %s", warning)
        log_warning(diagnostics, "main", str(warning), stage="data_quality")
    if data_quality_report["errors"]:
        for error in data_quality_report["errors"]:
            log_warning(diagnostics, "main", str(error), severity="ERROR", stage="data_quality")
        log_final_action(diagnostics, "PAUSE", selected_candidate="HOLD", reason="Data quality failed before optimizer run.")
        raise SystemExit(
            "Data quality failed before optimizer run: " + "; ".join(data_quality_report["errors"])
        )
    log_stage(diagnostics, "data quality", "DONE", extra={"score": data_quality_report["global_data_quality_score"]})
    log_stage(diagnostics, "feasibility", "START")
    feasibility_result = check_portfolio_feasibility(available_investable_tickers, params)
    for warning in feasibility_result["warnings"]:
        LOGGER.warning("Feasibility warning: %s", warning)
        log_warning(diagnostics, "main", str(warning), stage="feasibility")
    if not feasibility_result["feasible"]:
        failure_reason = (
            "Portfolio feasibility check failed before optimizer run: "
            + "; ".join(feasibility_result["errors"])
        )
        LOGGER.error("%s", failure_reason)
        for error in feasibility_result["errors"]:
            log_warning(diagnostics, "main", str(error), severity="ERROR", stage="feasibility")
        log_final_action(diagnostics, "PAUSE", selected_candidate="HOLD", reason=failure_reason)
        write_pause_outputs(
            output_dir=output_dir,
            reason=failure_reason,
            active_tickers=available_investable_tickers,
            removed_tickers=removed_tickers,
            notes=data_notes,
            data_context=data_context,
        )
        LOGGER.info("Run paused before optimizer execution due to infeasible constraints.")
        print("\nRun completed successfully.")
        return
    log_stage(diagnostics, "feasibility", "DONE", extra=feasibility_result)
    LOGGER.info("Running backtest with %s investable tickers.", len(available_investable_tickers))

    log_stage(diagnostics, "backtest", "START")
    result = run_backtest(prices=prices, params=params)
    run_notes: list[str] = []
    run_notes.extend(data_notes)
    if removed_tickers:
        run_notes.append(
            "Removed tickers in this run because no usable price history was available: "
            + ", ".join(removed_tickers)
        )
    if run_notes:
        result["run_notes"] = run_notes
    create_report_artifacts(
        result=result,
        output_dir=output_dir,
        latest_decision_filename="research_latest_decision_report.txt",
    )
    LOGGER.info(
        "Backtest completed from %s to %s across %s evaluation days.",
        pd.Timestamp(result["daily"].iloc[0]["date"]).date(),
        pd.Timestamp(result["daily"].iloc[-1]["next_date"]).date(),
        len(result["daily"]),
    )
    LOGGER.info("Number of trades during backtest: %s", number_of_trades(result["daily"]))
    log_stage(diagnostics, "backtest", "DONE", extra={"evaluation_days": len(result["daily"])})

    latest_record = result["daily"].iloc[-1]
    latest_weights = result["weights"].iloc[-1]
    latest_target_weights = result["target_weights"].iloc[-1]
    # Align the order preview with the actual latest decision date from the
    # backtest, rather than the already-realized t+1 return date.
    latest_prices = prices.loc[pd.Timestamp(latest_record["date"])]

    order_preview_df = generate_order_preview(
        w_current=latest_weights,
        w_target=latest_target_weights,
        latest_prices=latest_prices,
        portfolio_value=float(args.portfolio_value),
        output_path=None,
        min_order_value=float(params.get("min_order_value_usd", 10.0)),
        price_basis=str(prices.attrs.get("price_basis", "adjusted_close_proxy")),
    )

    adapter_or_stub = None
    account_summary = None
    broker_positions = None
    execution_mode_hint = expected_execution_mode(params)
    if execution_mode_hint == "local_paper":
        try:
            adapter_or_stub = PaperBrokerStub(
                db_path=params["db_path"],
                initial_cash=float(args.portfolio_value),
            )
            account_summary = adapter_or_stub.get_account_summary()
            broker_positions = adapter_or_stub.get_positions()
        except Exception as exc:
            LOGGER.warning("Local paper state could not be loaded for validation: %s", exc)

    reconciliation_result = reconcile_before_execution(
        model_weights=latest_target_weights,
        latest_prices=latest_prices,
        execution_mode=execution_mode_hint,
        broker_positions=broker_positions,
        broker_cash=float(account_summary["cash"]) if account_summary is not None else None,
        adapter_or_stub=adapter_or_stub,
    )
    build_reconciliation_report(
        reconciliation_result=reconciliation_result,
        output_path=output_dir / "reconciliation_report.csv",
    )
    validation_result = run_pre_trade_validation(
        w_current=latest_weights,
        w_target=latest_target_weights,
        latest_prices=latest_prices,
        order_preview_df=order_preview_df,
        params={
            **params,
            "blocked_tickers": tradability_df.loc[~tradability_df["final_allowed"], "ticker"].astype(str).tolist(),
            "max_equity_like_total": params["max_equity_like_total_normal"],
            "min_defensive_weight": params["min_defensive_weight_normal"],
        },
        account_summary=account_summary,
        positions=broker_positions,
    )
    save_pre_trade_validation_report(
        validation_report=validation_result["validation_report"],
        output_path=output_dir / "pre_trade_validation_report.csv",
    )

    executable_order_preview = mark_research_preview(validation_result["adjusted_order_preview"])
    _write_csv_atomic(output_dir / "research_order_preview.csv", executable_order_preview, index=False)
    _write_csv_atomic(output_dir / "order_preview.csv", executable_order_preview, index=False)
    execution_data_block_reason = ""
    if not bool(market_gate.get("execution_allowed", False)):
        execution_data_block_reason = f"Project calendar blocked execution: {market_gate.get('reason', 'calendar_blocked')}."
    elif not bool(data_freshness.get("data_freshness_ok", False)):
        execution_data_block_reason = str(
            data_freshness.get("warning")
            or "Current market data is too stale for any execution path."
        )
    elif bool(prices.attrs.get("synthetic_data", False)):
        execution_data_block_reason = "Synthetic data active; no execution path may be used."

    if execution_data_block_reason:
        executable_order_preview["executable"] = False
        executable_order_preview["execution_block_reason"] = execution_data_block_reason
        executable_order_preview["not_executable_reason"] = execution_data_block_reason
        _write_csv_atomic(output_dir / "order_preview.csv", executable_order_preview, index=False)
        _write_csv_atomic(output_dir / "research_order_preview.csv", executable_order_preview, index=False)
        execution_result = {
            "execution_mode": "blocked",
            "orders_submitted": 0,
            "orders_failed": int((order_preview_df["side"] != "HOLD").sum()),
            "message": execution_data_block_reason,
            "errors": [execution_data_block_reason],
        }
    elif not validation_result["ok"]:
        execution_result = {
            "execution_mode": "blocked",
            "orders_submitted": 0,
            "orders_failed": int((order_preview_df["side"] != "HOLD").sum()),
            "message": "Pre-trade validation failed; no execution was attempted.",
            "errors": list(validation_result["errors"]),
        }
    elif not reconciliation_result["broker_state_reconciled"] and execution_mode_hint != "order_preview_only":
        execution_result = {
            "execution_mode": "blocked",
            "orders_submitted": 0,
            "orders_failed": int((order_preview_df["side"] != "HOLD").sum()),
            "message": "Broker reconciliation failed; no execution was attempted.",
            "errors": list(reconciliation_result["errors"] or ["Broker reconciliation failed."]),
        }
    else:
        execution_result = run_execution_layer(
            order_preview_df=executable_order_preview,
            latest_prices=latest_prices,
            params={**params, "paper_initial_cash": float(args.portfolio_value)},
            db_path=params["db_path"],
        )
    result["execution_result"] = execution_result
    diagnostics.final_orders_summary = {
        "order_count": int((executable_order_preview["side"].astype(str) != "HOLD").sum()),
        "turnover": float(latest_record.get("realized_turnover", latest_record.get("turnover", 0.0)) or 0.0),
        "estimated_cost": float(latest_record.get("estimated_cost", 0.0) or 0.0),
    }
    result["safety_context"] = {
        "active_tickers_count": len(available_investable_tickers),
        "removed_tickers": removed_tickers,
        "tradability_warnings": tradability_df.loc[tradability_df["reason"] != "ok", "reason"].astype(str).tolist(),
        "data_quality_score": f"{float(data_quality_report['global_data_quality_score']):.3f}",
        "model_confidence_score": "n/a",
        "synthetic_data": data_context["synthetic_data"],
        "data_source": data_context["data_source"],
        "cache_status": data_context["cache_status"],
        "latest_price_date": data_context["latest_price_date"],
        "staleness_days": data_context["staleness_days"],
        "data_freshness_ok": data_context["data_freshness_ok"],
        "yfinance_available": data_context["yfinance_available"],
        "tickers_loaded": data_context["tickers_loaded"],
        "tickers_failed": data_context["tickers_failed"],
        "used_cache_fallback": data_context["used_cache_fallback"],
        "live_data_error": data_context["live_data_error"],
        "price_basis": data_context["price_basis"],
        "expected_latest_trading_day": data_context["expected_latest_trading_day"],
        "run_context": data_context["run_context"],
        "project_calendar_path": data_context["project_calendar_path"],
        "current_date_berlin": data_context["current_date_berlin"],
        "current_time_berlin": data_context["current_time_berlin"],
        "is_project_trading_day": data_context["is_project_trading_day"],
        "allowed_start_berlin": data_context["allowed_start_berlin"],
        "allowed_end_berlin": data_context["allowed_end_berlin"],
        "within_allowed_window": data_context["within_allowed_window"],
        "execution_allowed_by_calendar": data_context["execution_allowed_by_calendar"],
        "calendar_reason": data_context["calendar_status"],
        "pre_trade_validation_status": "PASS" if validation_result["ok"] else "FAIL",
        "pre_trade_warnings": validation_result["warnings"],
        "pre_trade_errors": validation_result["errors"],
        "reconciliation_status": reconciliation_result.get("status", "SKIP"),
        "execution_mode": execution_result.get("execution_mode", "unknown"),
        "blocked_orders_count": int(len(validation_result["blocked_orders"])),
        "research_preview_file": str(output_dir / "research_order_preview.csv"),
        "research_preview_context": "research_backtest_preview",
        "research_preview_note": (
            "outputs/research_order_preview.csv is the canonical main.py research/backtest preview. "
            "outputs/order_preview.csv is kept only as a legacy compatibility alias and is not the final manual simulator order file."
        ),
    }
    LOGGER.info(
        "Execution result: mode=%s submitted=%s failed=%s message=%s",
        execution_result.get("execution_mode", "unknown"),
        execution_result.get("orders_submitted", 0),
        execution_result.get("orders_failed", 0),
        execution_result.get("message", ""),
    )
    diagnostics.selected_candidate = str(latest_record.get("decision", "HOLD"))
    diagnostics.execution_mode = str(execution_result.get("execution_mode", "unknown"))
    log_final_action(diagnostics, str(latest_record.get("decision", "HOLD")), selected_candidate=str(latest_record.get("decision", "HOLD")), reason=str(execution_result.get("message", "")))
    write_latest_decision_report(
        result=result,
        output_path=output_dir / "research_latest_decision_report.txt",
    )
    write_output_file_guide(output_dir / "output_file_guide.txt")

    perf_summary = build_performance_frame(result)

    try:
        db_path = params["db_path"]
        init_db(db_path)
        run_id = create_run(db_path, params=params, tickers=available_investable_tickers)
        save_tradability_to_db(
            db_path=db_path,
            run_id=run_id,
            tradability_df=tradability_df,
        )
        save_data_quality_to_db(
            db_path=db_path,
            run_id=run_id,
            data_quality_report=data_quality_report,
        )
        save_full_run(
            db_path=db_path,
            run_id=run_id,
            result=result,
            order_preview_df=executable_order_preview,
            perf_df=perf_summary,
        )
        write_audit_metadata(
            create_run_metadata(
                params=params,
                active_tickers=available_investable_tickers,
                mode="main_backtest",
                removed_tickers=removed_tickers,
                data_start=str(prices.index.min().date()) if not prices.empty else None,
                data_end=str(prices.index.max().date()) if not prices.empty else None,
                execution_mode=execution_result.get("execution_mode", "order_preview_only"),
            ),
            output_path=output_dir / "audit_metadata.json",
        )
        LOGGER.info("SQLite persistence completed: %s", db_path)
    except Exception as exc:
        LOGGER.exception("SQLite persistence failed but the run will continue: %s", exc)

    latest_record_payload = latest_record.copy()
    latest_record_payload["execution_mode"] = execution_result.get("execution_mode", "unknown")
    latest_record_payload["execution_message"] = execution_result.get("message", "")
    try:
        body = build_email_body(
            latest_record=latest_record_payload,
            latest_weights=latest_weights,
            latest_target_weights=latest_target_weights,
        )
    except Exception as exc:
        LOGGER.warning(
            "Could not build latest email notification body normally; writing fallback text: %s",
            exc,
        )
        body = (
            "Email notification body could not be built.\n"
            f"Reason: {exc}\n"
            "The run still completed and no email was sent.\n"
        )
    write_latest_notification(body=body, output_dir=output_dir)
    LOGGER.info("Research run wrote latest_email_notification.txt. SMTP send remains disabled here unless the explicit diagnostics email path is enabled.")

    LOGGER.info(
        "Latest decision: %s | risk_state=%s | solver=%s",
        latest_record["decision"],
        latest_record["risk_state"],
        latest_record.get("solver", "n/a"),
    )
    LOGGER.info("Core outputs written to %s", output_dir)
    print_console_summary(result=result, perf_summary=perf_summary)
    print("\nRun completed successfully.")


def main() -> None:
    """Wrapper that guarantees research diagnostics outputs on success and failure."""

    setup_logging()
    args = parse_args()
    diagnostics = create_run_diagnostics(mode="research_main")
    diagnostics.model_context["requested_mode"] = "main_backtest"
    diagnostics.model_context["send_analysis_email"] = bool(args.send_analysis_email)
    output_dir = Path(args.output_dir)
    failure: BaseException | None = None
    try:
        _main_impl(args, diagnostics)
    except BaseException as exc:
        failure = exc
        log_error(diagnostics, "main", "_main_impl", exc, stage="research_run")
        log_final_action(diagnostics, "PAUSE", selected_candidate="HOLD", reason="Research pipeline failed closed. Local diagnostics were still written.")
        _write_minimal_research_reports(diagnostics, output_dir)
    finally:
        _finalize_research_diagnostics(diagnostics, output_dir=output_dir, allow_email=bool(args.send_analysis_email))
    if failure is not None:
        if isinstance(failure, KeyboardInterrupt):
            raise failure
        print("Research run paused after an error. See outputs/research_run_diagnostics.json for details.")


if __name__ == "__main__":
    main()
