"""Model-vs-broker reconciliation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def compute_weights_from_positions(
    positions_df: pd.DataFrame,
    cash: float,
    latest_prices: pd.Series,
) -> pd.Series:
    """Compute current weights from positions plus cash."""

    if positions_df.empty:
        return pd.Series(dtype=float)

    positions = positions_df.copy()
    positions["ticker"] = positions["ticker"].astype(str)
    prices = latest_prices.astype(float).copy()
    prices.index = pd.Index([str(t) for t in prices.index], name="ticker")
    positions["price_used"] = positions["ticker"].map(prices).fillna(positions.get("last_price", 0.0))
    positions["market_value_used"] = positions["shares"].astype(float) * positions["price_used"].astype(float)
    positions_value = float(positions["market_value_used"].sum())
    total_equity = float(cash) + positions_value
    if total_equity <= 0.0:
        return pd.Series(dtype=float)
    weights = positions.set_index("ticker")["market_value_used"] / total_equity
    return weights.astype(float).sort_index()


def reconcile_model_vs_broker_state(
    model_weights: pd.Series,
    broker_positions: pd.DataFrame,
    broker_cash: float,
    latest_prices: pd.Series,
) -> dict[str, Any]:
    """Compare model weights with broker or paper-account state."""

    model = model_weights.astype(float).copy()
    model.index = pd.Index([str(t) for t in model.index], name="ticker")
    broker_weights = compute_weights_from_positions(
        positions_df=broker_positions,
        cash=broker_cash,
        latest_prices=latest_prices,
    )
    all_tickers = pd.Index(sorted(set(model.index).union(broker_weights.index)), name="ticker")
    comparison = pd.DataFrame(
        {
            "model_weight": model.reindex(all_tickers).fillna(0.0),
            "broker_weight": broker_weights.reindex(all_tickers).fillna(0.0),
        }
    )
    comparison["abs_diff"] = (comparison["model_weight"] - comparison["broker_weight"]).abs()
    reconciled = bool(comparison["abs_diff"].max() <= 0.05) if not comparison.empty else True
    return {
        "broker_weights": broker_weights,
        "position_mismatches": comparison.reset_index(),
        "broker_state_reconciled": reconciled,
    }


def detect_open_orders(adapter_or_stub: object | None) -> dict[str, Any]:
    """Detect whether open orders exist.

    For the local paper stub this is always false because fills are immediate.
    """

    if adapter_or_stub is None:
        return {"open_orders_exist": False, "status": "SKIP", "message": "No adapter or stub was supplied."}
    name = adapter_or_stub.__class__.__name__
    if name == "PaperBrokerStub":
        return {"open_orders_exist": False, "status": "PASS", "message": "Local paper stub uses immediate fills only."}
    return {
        "open_orders_exist": True,
        "status": "BLOCK",
        "message": "Open-order detection is not implemented for this adapter. Unknown open orders are blocked fail-closed.",
    }


def build_reconciliation_report(reconciliation_result: dict[str, Any], output_path: str | Path) -> Path:
    """Persist reconciliation diagnostics."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mismatches = reconciliation_result.get("position_mismatches")
    if isinstance(mismatches, pd.DataFrame) and not mismatches.empty:
        report_df = mismatches.copy()
    else:
        report_df = pd.DataFrame(
            [
                {
                    "status": reconciliation_result.get("status", "SKIP"),
                    "message": "; ".join(
                        [
                            *reconciliation_result.get("warnings", []),
                            *reconciliation_result.get("errors", []),
                        ]
                    )
                    or reconciliation_result.get("message", "No reconciliation data."),
                }
            ]
        )
    report_df.to_csv(path, index=False)
    return path


def reconcile_before_execution(
    *,
    model_weights: pd.Series,
    latest_prices: pd.Series,
    execution_mode: str,
    broker_positions: pd.DataFrame | None = None,
    broker_cash: float | None = None,
    adapter_or_stub: object | None = None,
) -> dict[str, Any]:
    """Run reconciliation before any optional execution path."""

    if execution_mode == "order_preview_only":
        return {
            "broker_state_reconciled": True,
            "open_orders_exist": False,
            "position_mismatches": pd.DataFrame(),
            "warnings": ["Reconciliation skipped in preview-only mode."],
            "errors": [],
            "status": "SKIP",
            "message": "Preview-only mode; no broker reconciliation required.",
        }

    if broker_positions is None or broker_cash is None:
        return {
            "broker_state_reconciled": False,
            "open_orders_exist": False,
            "position_mismatches": pd.DataFrame(),
            "warnings": [],
            "errors": ["Broker positions or cash were not provided for reconciliation."],
            "status": "FAIL",
            "message": "Missing broker state.",
        }

    comparison = reconcile_model_vs_broker_state(
        model_weights=model_weights,
        broker_positions=broker_positions,
        broker_cash=float(broker_cash),
        latest_prices=latest_prices,
    )
    open_orders = detect_open_orders(adapter_or_stub)
    warnings = []
    errors = []
    if open_orders.get("status") == "WARN":
        warnings.append(str(open_orders.get("message", "")))
    if open_orders.get("status") == "BLOCK":
        errors.append(str(open_orders.get("message", "")))
    if not comparison["broker_state_reconciled"]:
        warnings.append("Model and broker weights differ materially.")
    status = "PASS"
    if errors:
        status = "FAIL"
    elif not comparison["broker_state_reconciled"]:
        status = "WARN"
    return {
        "broker_state_reconciled": bool(comparison["broker_state_reconciled"]),
        "open_orders_exist": bool(open_orders["open_orders_exist"]),
        "position_mismatches": comparison["position_mismatches"],
        "warnings": warnings,
        "errors": errors,
        "status": status,
        "message": open_orders.get("message", "ok"),
    }
