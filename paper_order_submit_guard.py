"""Paper order submit guard.

Safety layer before any future simulator submission.

Default behavior:
- BLOCKED
- Does not submit orders
- Requires BOTH environment variables to be explicitly true:
  PAPER_TRADING_ENABLED=true
  SUBMIT_ORDERS=true

Even when both are true, this file still does not implement actual submission.
It only changes status from blocked to armed_no_submit_client.
"""

from __future__ import annotations

from pathlib import Path
import json
import os


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

DRY_RUN_ORDERS_PATH = OUTPUT_DIR / "paper_simulator_dry_run_orders.json"
GUARD_JSON_PATH = OUTPUT_DIR / "paper_order_submit_guard.json"
GUARD_REPORT_PATH = OUTPUT_DIR / "paper_order_submit_guard_report.txt"


def env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y"}


def load_dry_run_payload() -> dict:
    if not DRY_RUN_ORDERS_PATH.exists():
        return {
            "status": "missing_dry_run_payload",
            "orders": [],
        }

    return json.loads(DRY_RUN_ORDERS_PATH.read_text(encoding="utf-8"))


def evaluate_guard() -> dict:
    dry_run = load_dry_run_payload()

    paper_trading_enabled = env_true("PAPER_TRADING_ENABLED")
    submit_orders = env_true("SUBMIT_ORDERS")

    checks = [
        {
            "check": "dry_run_payload_exists",
            "ok": DRY_RUN_ORDERS_PATH.exists(),
            "detail": str(DRY_RUN_ORDERS_PATH),
        },
        {
            "check": "dry_run_payload_ready",
            "ok": dry_run.get("status") == "dry_run_ready",
            "detail": str(dry_run.get("status")),
        },
        {
            "check": "dry_run_only_payload",
            "ok": dry_run.get("dry_run_only") is True,
            "detail": str(dry_run.get("dry_run_only")),
        },
        {
            "check": "paper_trading_enabled_env",
            "ok": paper_trading_enabled,
            "detail": f"PAPER_TRADING_ENABLED={os.environ.get('PAPER_TRADING_ENABLED', '')!r}",
            "required_for_submit": True,
        },
        {
            "check": "submit_orders_env",
            "ok": submit_orders,
            "detail": f"SUBMIT_ORDERS={os.environ.get('SUBMIT_ORDERS', '')!r}",
            "required_for_submit": True,
        },
    ]

    hard_ready = all(
        c["ok"]
        for c in checks
        if c["check"] in {
            "dry_run_payload_exists",
            "dry_run_payload_ready",
            "dry_run_only_payload",
        }
    )

    armed = hard_ready and paper_trading_enabled and submit_orders

    if not hard_ready:
        status = "blocked_missing_or_invalid_dry_run"
    elif not paper_trading_enabled or not submit_orders:
        status = "blocked_by_kill_switch"
    else:
        status = "armed_no_submit_client"

    return {
        "status": status,
        "orders_were_submitted": False,
        "submit_client_implemented": False,
        "paper_trading_enabled": paper_trading_enabled,
        "submit_orders": submit_orders,
        "order_count": int(dry_run.get("order_count", 0) or 0),
        "orders": dry_run.get("orders", []),
        "checks": checks,
    }


def main() -> None:
    result = evaluate_guard()

    GUARD_JSON_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "Paper Order Submit Guard Report",
        "",
        f"status: {result['status']}",
        f"orders_were_submitted: {result['orders_were_submitted']}",
        f"submit_client_implemented: {result['submit_client_implemented']}",
        f"paper_trading_enabled: {result['paper_trading_enabled']}",
        f"submit_orders: {result['submit_orders']}",
        f"order_count: {result['order_count']}",
        "",
        "important:",
        "- This script does not submit simulator or broker orders.",
        "- Default behavior is blocked_by_kill_switch.",
        "- Future real submission must require PAPER_TRADING_ENABLED=true and SUBMIT_ORDERS=true.",
        "- No submit client is implemented here.",
        "",
        "checks:",
    ]

    for check in result["checks"]:
        marker = "OK" if check["ok"] else "BLOCK"
        lines.append(f"- {marker}: {check['check']} :: {check['detail']}")

    lines.extend(["", "orders_visible_to_guard:"])
    if not result["orders"]:
        lines.append("- none")
    else:
        for order in result["orders"]:
            lines.append(
                f"- {str(order.get('side', '')).upper()} {order.get('ticker')} "
                f"qty={order.get('quantity')} "
                f"est_notional={float(order.get('estimated_notional', 0.0)):.2f}"
            )

    GUARD_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(GUARD_REPORT_PATH.read_text())


if __name__ == "__main__":
    main()
