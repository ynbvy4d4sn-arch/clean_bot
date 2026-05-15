"""Simple smoke tests for broker-style interfaces and execution orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

from broker_interface import BrokerInterface
import daily_bot as daily_bot_module
from investopedia_adapter import InvestopediaSimulatorAdapter
from order_preview import generate_order_preview
from paper_broker_stub import (
    PaperBrokerStub,
    execute_order_preview_locally,
    get_paper_account_summary,
    initialize_paper_account,
)
from simulator_orchestrator import run_execution_layer


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _announce(message: str) -> None:
    print(f"[interface_tests] {message}", flush=True)


def _record(messages: list[str], message: str) -> None:
    messages.append(message)
    _announce(message)


def run_interface_smoke_tests() -> list[str]:
    """Run a lightweight interface smoke test suite."""

    messages: list[str] = []
    _assert(isinstance(BrokerInterface, type), "BrokerInterface import failed.")
    _record(messages, "BrokerInterface import: OK")

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "paper.sqlite"
        initialize_paper_account(db_path, initial_cash=10000)
        stub = PaperBrokerStub(db_path=db_path, initial_cash=10000)
        summary = stub.get_account_summary()
        _assert(float(summary["cash"]) == 10000.0, "PaperBrokerStub initial cash mismatch.")
        _record(messages, "PaperBrokerStub init/account summary: OK")

        current = pd.Series({"SGOV": 1.0, "IEF": 0.0}, dtype=float)
        target = pd.Series({"SGOV": 0.5, "IEF": 0.5}, dtype=float)
        prices = pd.Series({"SGOV": 100.0, "IEF": 95.0}, dtype=float)
        preview = generate_order_preview(
            w_current=current,
            w_target=target,
            latest_prices=prices,
            portfolio_value=10000.0,
            output_path=None,
        )
        validated = stub.preview_orders(preview)
        _assert(not validated.empty, "PaperBrokerStub preview validation failed.")
        _record(messages, "PaperBrokerStub preview validation: OK")

        trades = execute_order_preview_locally(
            db_path=db_path,
            order_preview_df=validated,
            latest_prices=prices,
            cost_rate=0.001,
        )
        account_after = get_paper_account_summary(db_path)
        _assert(len(trades) >= 1, "PaperBrokerStub local execution produced no trades.")
        _assert(float(account_after["total_equity"]) > 0.0, "PaperBrokerStub equity invalid after execution.")
        _record(messages, "PaperBrokerStub local BUY/SELL simulation: OK")

        adapter = InvestopediaSimulatorAdapter.from_env()
        try:
            adapter.submit_orders(validated)
        except RuntimeError as exc:
            _assert("disabled" in str(exc).lower() or "incomplete" in str(exc).lower(), "Unexpected Investopedia disabled error.")
            _record(messages, "Investopedia adapter disabled/incomplete path: OK")
        except NotImplementedError:
            _record(messages, "Investopedia adapter experimental stub path: OK")

        execution_result = run_execution_layer(
            order_preview_df=validated,
            latest_prices=prices,
            params={
                "dry_run": True,
                "enable_local_paper_trading": False,
                "enable_investopedia_simulator": False,
                "cost_rate": 0.001,
            },
            db_path=db_path,
        )
        _assert(
            execution_result["execution_mode"] == "order_preview_only",
            "Execution orchestrator did not stay in preview-only mode.",
        )
        _record(messages, "Simulator orchestrator preview-only default: OK")

    _announce("Starting main.py --skip-email smoke run...")
    run = subprocess.run(
        [sys.executable, "main.py", "--skip-email"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    _assert(run.returncode == 0, f"main.py smoke run failed with code {run.returncode}.")
    _assert(
        "Run completed successfully." in run.stdout or "Run completed successfully." in run.stderr,
        "main.py smoke run did not report successful completion.",
    )
    _record(messages, "main.py --skip-email smoke run: OK")

    _announce("Starting daily_bot.py --dry-run --mode single --force-refresh smoke run...")
    daily_run = subprocess.run(
        [sys.executable, "daily_bot.py", "--dry-run", "--mode", "single", "--force-refresh"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    _assert(daily_run.returncode == 0, f"daily_bot.py smoke run failed with code {daily_run.returncode}.")
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    for required in [
        "run_diagnostics.json",
        "codex_daily_debug_report.md",
        "codex_next_prompt.md",
        "daily_analysis_report.md",
        "daily_portfolio_review.txt",
        "daily_portfolio_review.csv",
        "daily_review_email.html",
        "daily_review_report.tex",
        "daily_email_briefing.txt",
        "daily_email_subject.txt",
        "latest_email_notification.txt",
        "email_safety_report.txt",
        "daily_review_validation_report.txt",
        "email_final_acceptance_report.txt",
        "last_email_state.json",
    ]:
        _assert((outputs_dir / required).exists(), f"Missing daily bot diagnostic output: {required}")
    for required_chart in [
        "charts/current_portfolio_allocation.png",
        "charts/current_vs_target_weights.png",
    ]:
        _assert((outputs_dir / required_chart).exists(), f"Missing daily bot chart output: {required_chart}")
    if shutil.which("pdflatex"):
        _assert((outputs_dir / "daily_review_report.pdf").exists(), "Missing daily bot PDF output: daily_review_report.pdf")
    _record(messages, "daily_bot.py diagnostic reports on success: OK")

    _announce("Starting forced daily_bot.py failure-path diagnostics smoke run...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_output_dir = Path(tmp_dir) / "outputs"
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        original_output_dir = daily_bot_module.OUTPUT_DIR
        original_loader = daily_bot_module.load_price_data
        try:
            daily_bot_module.OUTPUT_DIR = temp_output_dir

            def _raise_loader(*args: object, **kwargs: object) -> pd.DataFrame:
                raise RuntimeError("forced diagnostics failure")

            daily_bot_module.load_price_data = _raise_loader
            failure_result = daily_bot_module._run_single(
                argparse.Namespace(
                    dry_run=True,
                    broker="none",
                    portfolio_value=10000.0,
                    skip_submit=True,
                    mode="single",
                    check_interval_minutes=15,
                    full_recompute_interval_minutes=60,
                    force_refresh=True,
                )
            )
        finally:
            daily_bot_module.OUTPUT_DIR = original_output_dir
            daily_bot_module.load_price_data = original_loader
        _assert(failure_result["gate_action"] == "PAUSE", "Forced failure path did not fail-closed to PAUSE.")
        for required in [
            "run_diagnostics.json",
            "error_log.csv",
            "codex_daily_debug_report.md",
            "codex_next_prompt.md",
            "daily_analysis_report.md",
            "daily_portfolio_review.txt",
            "daily_portfolio_review.csv",
            "daily_review_email.html",
            "daily_review_report.tex",
            "daily_email_briefing.txt",
            "daily_email_subject.txt",
            "latest_email_notification.txt",
            "email_safety_report.txt",
            "daily_review_validation_report.txt",
            "email_final_acceptance_report.txt",
            "last_email_state.json",
        ]:
            _assert((temp_output_dir / required).exists(), f"Missing forced-failure diagnostic output: {required}")
        for required_chart in [
            "charts/current_portfolio_allocation.png",
            "charts/current_vs_target_weights.png",
        ]:
            _assert((temp_output_dir / required_chart).exists(), f"Missing forced-failure chart output: {required_chart}")
    _record(messages, "daily_bot failure path writes pause diagnostics: OK")

    _announce("Starting SMTP-failure fail-closed smoke run...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_output_dir = Path(tmp_dir) / "outputs"
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        original_output_dir = daily_bot_module.OUTPUT_DIR
        original_loader = daily_bot_module.load_price_data
        original_sender = daily_bot_module.send_daily_review_email_if_needed
        try:
            daily_bot_module.OUTPUT_DIR = temp_output_dir

            def _raise_loader_again(*args: object, **kwargs: object) -> pd.DataFrame:
                raise RuntimeError("forced diagnostics failure for smtp path")

            def _raise_sender(*args: object, **kwargs: object) -> dict[str, object]:
                raise RuntimeError("smtp send failed")

            daily_bot_module.load_price_data = _raise_loader_again
            daily_bot_module.send_daily_review_email_if_needed = _raise_sender
            smtp_failure_result = daily_bot_module._run_single(
                argparse.Namespace(
                    dry_run=True,
                    broker="none",
                    portfolio_value=10000.0,
                    skip_submit=True,
                    mode="single",
                    check_interval_minutes=15,
                    full_recompute_interval_minutes=60,
                    force_refresh=True,
                )
            )
        finally:
            daily_bot_module.OUTPUT_DIR = original_output_dir
            daily_bot_module.load_price_data = original_loader
            daily_bot_module.send_daily_review_email_if_needed = original_sender
        _assert(smtp_failure_result["gate_action"] == "PAUSE", "SMTP failure path did not stay fail-closed.")
        _assert((temp_output_dir / "daily_analysis_report.md").exists(), "SMTP failure path did not preserve daily analysis report.")
        _assert((temp_output_dir / "daily_portfolio_review.txt").exists(), "SMTP failure path did not preserve daily portfolio review.")
        _assert((temp_output_dir / "latest_email_notification.txt").exists(), "SMTP failure path did not preserve latest email notification preview.")
        _assert((temp_output_dir / "email_final_acceptance_report.txt").exists(), "SMTP failure path did not preserve final acceptance report.")
        _assert((temp_output_dir / "last_email_state.json").exists(), "SMTP failure path did not preserve last email state.")
    _record(messages, "SMTP error does not crash daily bot: OK")
    _record(messages, "No external orders are sent under default dry-run settings: OK")
    return messages


def main() -> None:
    messages = run_interface_smoke_tests()
    print("Interface smoke tests passed:")
    for message in messages:
        print(f"- {message}")


if __name__ == "__main__":
    main()
