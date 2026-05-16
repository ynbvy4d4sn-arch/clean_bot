"""Paper simulator adapter skeleton.

Dry-run only. Does not submit orders.

Purpose:
- Read validated paper_order_preview.csv
- Convert BUY/SELL rows into a simulator-style order payload
- Write outputs/paper_simulator_dry_run_orders.json
- Never send orders unless future code explicitly implements a submit client
"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
ORDER_PREVIEW_PATH = OUTPUT_DIR / "paper_order_preview.csv"
VALIDATION_PATH = OUTPUT_DIR / "paper_order_preview_validation.json"
DRY_RUN_ORDERS_PATH = OUTPUT_DIR / "paper_simulator_dry_run_orders.json"
DRY_RUN_REPORT_PATH = OUTPUT_DIR / "paper_simulator_dry_run_report.txt"


def load_validation() -> dict:
    if not VALIDATION_PATH.exists():
        return {"status": "missing_validation"}
    return json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))


def build_dry_run_orders() -> dict:
    validation = load_validation()
    if validation.get("status") != "ready_for_manual_review":
        return {
            "status": "blocked",
            "reason": f"validation_status={validation.get('status')}",
            "orders": [],
        }

    if not ORDER_PREVIEW_PATH.exists():
        return {
            "status": "blocked",
            "reason": f"missing_order_preview={ORDER_PREVIEW_PATH}",
            "orders": [],
        }

    df = pd.read_csv(ORDER_PREVIEW_PATH)
    actionable = df[df["action"].isin(["BUY", "SELL"])].copy()

    orders = []
    for row in actionable.itertuples(index=False):
        quantity = abs(int(row.estimated_shares))
        if quantity <= 0:
            continue

        orders.append(
            {
                "ticker": str(row.ticker),
                "side": str(row.action).lower(),
                "quantity": quantity,
                "order_type": "market",
                "time_in_force": "day",
                "estimated_price": float(row.latest_price),
                "estimated_notional": abs(float(row.rounded_dollar_delta)),
                "source": "paper_order_preview",
                "dry_run_only": True,
            }
        )

    return {
        "status": "dry_run_ready",
        "dry_run_only": True,
        "order_count": len(orders),
        "orders": orders,
    }


def main() -> None:
    payload = build_dry_run_orders()
    DRY_RUN_ORDERS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "Paper Simulator Dry-Run Adapter Report",
        "",
        f"status: {payload['status']}",
        "dry_run_only: true",
        f"order_count: {payload.get('order_count', 0)}",
        "",
        "important:",
        "- No simulator or broker orders were submitted.",
        "- This adapter only writes a dry-run payload.",
        "- Submission code is intentionally not implemented.",
        "",
        "orders:",
    ]

    if not payload.get("orders"):
        lines.append("- none")
    else:
        for order in payload["orders"]:
            lines.append(
                f"- {order['side'].upper()} {order['ticker']} "
                f"qty={order['quantity']} "
                f"est_price={order['estimated_price']:.4f} "
                f"est_notional={order['estimated_notional']:.2f}"
            )

    DRY_RUN_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DRY_RUN_REPORT_PATH.read_text())


if __name__ == "__main__":
    main()
