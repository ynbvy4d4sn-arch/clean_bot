"""Minimal smoke test for the core dry-run optimizer stack."""

from __future__ import annotations

import importlib
import logging
import sys
from contextlib import contextmanager

import pandas as pd

from asset_universe import validate_asset_universe
from config import build_params
from execution_gate import evaluate_execution_gate
from feasibility import check_portfolio_feasibility
from health_check import run_health_check
from investopedia_adapter import InvestopediaSimulatorAdapter
from optimizer import build_feasible_initial_weights, optimize_allocation
from order_preview import generate_order_preview


@contextmanager
def _suppress_expected_notification_failure_logs():
    """Suppress expected notification failure logs emitted by health-check test probes."""

    logger = logging.getLogger("notifications")

    class ExpectedNotificationFailureFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = str(record.getMessage())
            return not message.startswith("Email send failed but the run will continue:")

    filter_obj = ExpectedNotificationFailureFilter()
    logger.addFilter(filter_obj)
    try:
        yield
    finally:
        logger.removeFilter(filter_obj)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_smoke_test() -> list[str]:
    """Run a minimal project smoke test without external execution."""

    messages: list[str] = []

    core_modules = [
        "config",
        "asset_universe",
        "data",
        "optimizer",
        "main",
        "daily_bot",
        "health_check",
    ]
    for module_name in core_modules:
        importlib.import_module(module_name)
    messages.append("Core module imports: OK")

    validate_asset_universe()
    messages.append("Asset registry validation: OK")

    params = build_params()
    _assert(isinstance(params, dict), "build_params() did not return a dict.")
    _assert(bool(params.get("tickers")), "build_params() returned no tickers.")
    messages.append("Config load/build_params: OK")

    test_tickers = ["SGOV", "IEF", "XLK", "XLI", "TIP", "PDBC", "SPHQ", "SPLV", "AGG", "LQD"]
    test_params = build_params(tickers=test_tickers)
    feasibility = check_portfolio_feasibility(test_tickers, test_params)
    _assert(feasibility["feasible"], "Feasibility check failed on dummy setup.")
    messages.append("Feasibility check: OK")

    with _suppress_expected_notification_failure_logs():
        health_df = run_health_check(quick=True, full=False)
    _assert(not health_df.empty, "Health check quick returned no rows.")
    _assert(
        "python_version" in set(health_df["check_name"].astype(str)),
        "Health check quick did not include python_version.",
    )
    messages.append("Health check quick: OK")

    mu = pd.Series(0.01, index=test_tickers, dtype=float)
    sigma = pd.DataFrame(0.0, index=test_tickers, columns=test_tickers, dtype=float)
    for ticker in sigma.index:
        sigma.loc[ticker, ticker] = 0.02
    w_current = build_feasible_initial_weights(test_tickers, test_params)
    optimizer_result = optimize_allocation(
        mu=mu,
        Sigma=sigma,
        w_current=w_current,
        params={
            **test_params,
            "max_equity_like_total": test_params["max_equity_like_total_normal"],
            "min_defensive_weight": test_params["min_defensive_weight_normal"],
        },
    )
    weights = optimizer_result.target_weights
    _assert(isinstance(weights, pd.Series), "Optimizer did not return a pandas Series.")
    _assert(list(weights.index) == test_tickers, "Optimizer weights index does not match active tickers.")
    _assert(abs(float(weights.sum()) - 1.0) < 1e-6, "Optimizer weights do not sum to 1.")
    _assert(bool((weights >= -1e-10).all()), "Optimizer returned negative weights.")
    messages.append(f"Optimizer dummy setup: OK ({optimizer_result.solver_name})")

    preview = generate_order_preview(
        w_current=pd.Series({"SGOV": 1.0, "IEF": 0.0}, dtype=float),
        w_target=pd.Series({"SGOV": 0.5, "IEF": 0.5}, dtype=float),
        latest_prices=pd.Series({"SGOV": 100.0, "IEF": 95.0}, dtype=float),
        portfolio_value=10000.0,
        output_path=None,
    )
    required_preview_columns = {
        "ticker",
        "current_weight",
        "target_weight",
        "delta_weight",
        "side",
        "estimated_order_value",
        "estimated_shares",
    }
    _assert(required_preview_columns.issubset(set(preview.columns)), "Order preview columns are incomplete.")
    messages.append("Order preview dummy generation: OK")

    dummy_score = type(
        "DummyScore",
        (),
        {
            "delta_vs_hold": 0.01,
            "estimated_cost": 0.0,
            "delta_vs_cash": 0.01,
            "probability_beats_hold": 0.70,
            "probability_beats_cash": 0.70,
        },
    )()
    dummy_candidate = type("DummyCandidate", (), {"name": "OPTIMIZER_TARGET"})()
    dummy_selection = type(
        "DummySelection",
        (),
        {"selected_score": dummy_score, "selected_candidate": dummy_candidate},
    )()
    gate = evaluate_execution_gate(dummy_selection, synthetic_data=True)
    _assert(gate.action == "PAUSE", "Execution gate did not block synthetic_data.")
    messages.append("Execution gate synthetic-data block: OK")

    _assert(
        bool(params.get("enable_investopedia_simulator", False)) is False,
        "Investopedia should be disabled by default.",
    )
    adapter = InvestopediaSimulatorAdapter.from_env()
    _assert(adapter.settings.enabled is False, "Investopedia adapter settings should be disabled by default.")
    try:
        adapter.login()
        raise AssertionError("Investopedia login unexpectedly proceeded while disabled.")
    except RuntimeError as exc:
        _assert("disabled" in str(exc).lower(), "Investopedia disabled path did not fail safely.")
    messages.append("Investopedia disabled-by-default path: OK")

    return messages


def main() -> None:
    messages = run_smoke_test()
    print("Smoke test passed:")
    for message in messages:
        print(f"- {message}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        raise
