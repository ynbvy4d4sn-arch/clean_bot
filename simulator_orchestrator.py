"""Execution-layer orchestration for preview-only, local paper and optional simulator modes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from investopedia_adapter import InvestopediaSimulatorAdapter
from paper_broker_stub import (
    PaperBrokerStub,
    execute_order_preview_locally,
    get_paper_account_summary,
    initialize_paper_account,
    save_paper_account_state,
)


LOGGER = logging.getLogger(__name__)


def _as_execution_result(
    *,
    execution_mode: str,
    orders_submitted: int = 0,
    orders_failed: int = 0,
    message: str = "",
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Build a normalized execution result dictionary."""

    return {
        "execution_mode": execution_mode,
        "orders_submitted": int(orders_submitted),
        "orders_failed": int(orders_failed),
        "message": str(message),
        "errors": list(errors or []),
    }


def run_execution_layer(
    order_preview_df: pd.DataFrame,
    latest_prices: pd.Series,
    params: dict[str, object],
    db_path: str | Path,
) -> dict[str, Any]:
    """Run the optional execution layer after order preview creation.

    The core optimizer remains preview-first and dry-run by default. External
    execution is never required for a successful run.
    """

    dry_run = bool(params.get("dry_run", True))
    enable_investopedia = bool(params.get("enable_investopedia_simulator", False))
    enable_local_paper = bool(params.get("enable_local_paper_trading", False))
    initial_cash = float(params.get("paper_initial_cash", 10000.0))

    if dry_run:
        return _as_execution_result(
            execution_mode="order_preview_only",
            message=(
                "DRY_RUN=true, so no external simulator orders or local paper trades were executed. "
                "Only order preview output was generated."
            ),
        )

    if enable_investopedia:
        try:
            adapter = InvestopediaSimulatorAdapter.from_env(logger=LOGGER)
            preview = adapter.preview_orders(order_preview_df)
            response = adapter.submit_orders(preview)
            order_count = int(response.get("order_count", len(preview)))
            if response.get("mode") == "investopedia_dry_run":
                return _as_execution_result(
                    execution_mode="order_preview_only",
                    orders_submitted=0,
                    orders_failed=0,
                    message=str(response.get("message", "Investopedia dry-run preview only.")),
                )
            return _as_execution_result(
                execution_mode="investopedia",
                orders_submitted=order_count,
                orders_failed=0,
                message="Investopedia simulator adapter reported a submission path.",
            )
        except Exception as exc:
            LOGGER.warning("Investopedia execution path failed safely: %s", exc)
            if not enable_local_paper:
                return _as_execution_result(
                    execution_mode="failed",
                    orders_failed=int(len(order_preview_df)),
                    message="Investopedia simulator path failed; no execution was performed.",
                    errors=[str(exc)],
                )
            LOGGER.info("Falling back from Investopedia execution to local paper simulation.")

    if enable_local_paper:
        try:
            adapter = PaperBrokerStub(db_path=db_path, initial_cash=initial_cash)
            preview = adapter.preview_orders(order_preview_df)
            initialize_paper_account(db_path=db_path, initial_cash=initial_cash)
            trades = execute_order_preview_locally(
                db_path=db_path,
                order_preview_df=preview,
                latest_prices=latest_prices,
                cost_rate=float(params.get("cost_rate", 0.0)),
            )
            save_paper_account_state(
                db_path=db_path,
                date=pd.Timestamp.utcnow().date().isoformat(),
            )
            summary = get_paper_account_summary(db_path=db_path)
            return _as_execution_result(
                execution_mode="local_paper",
                orders_submitted=int(len(trades)),
                orders_failed=max(int(len(preview[preview["side"] != "HOLD"])) - int(len(trades)), 0),
                message=(
                    "Local paper trading simulation completed. "
                    f"Paper equity={float(summary.get('total_equity', 0.0)):.2f}"
                ),
            )
        except Exception as exc:
            LOGGER.warning("Local paper execution failed safely: %s", exc)
            return _as_execution_result(
                execution_mode="failed",
                orders_failed=int(len(order_preview_df)),
                message="Local paper execution failed; only the order preview remains available.",
                errors=[str(exc)],
            )

    return _as_execution_result(
        execution_mode="order_preview_only",
        message="Execution layer disabled. Only order preview output was generated.",
    )
