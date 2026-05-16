"""Validate paper order preview output.

Safety-only. Does not send orders.
"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
ORDER_PATH = OUTPUT_DIR / "paper_order_preview.csv"
REPORT_PATH = OUTPUT_DIR / "paper_order_preview_report.txt"
VALIDATION_JSON = OUTPUT_DIR / "paper_order_preview_validation.json"
VALIDATION_REPORT = OUTPUT_DIR / "paper_order_preview_validation_report.txt"

MAX_ORDER_USD = 25_000.0
MIN_CASH_AFTER_ORDERS = 0.0
VALID_ACTIONS = {"BUY", "SELL", "HOLD", "REJECT"}


def check_orders() -> dict:
    checks = []

    if not ORDER_PATH.exists():
        return {
            "status": "not_ready",
            "checks": [
                {
                    "check": "order_preview_exists",
                    "ok": False,
                    "detail": f"missing {ORDER_PATH}",
                }
            ],
        }

    df = pd.read_csv(ORDER_PATH)

    checks.append(
        {
            "check": "order_preview_exists",
            "ok": True,
            "detail": str(ORDER_PATH),
        }
    )

    unknown_actions = sorted(set(df["action"].astype(str)) - VALID_ACTIONS)
    checks.append(
        {
            "check": "valid_actions",
            "ok": not unknown_actions,
            "detail": ",".join(unknown_actions) if unknown_actions else "ok",
        }
    )

    actionable = df[df["action"].isin(["BUY", "SELL"])].copy()

    missing_prices = actionable[actionable["latest_price"].isna() | (actionable["latest_price"] <= 0)]
    checks.append(
        {
            "check": "actionable_orders_have_prices",
            "ok": missing_prices.empty,
            "detail": ",".join(missing_prices["ticker"].astype(str).tolist()) if not missing_prices.empty else "ok",
        }
    )

    oversized = actionable[actionable["rounded_dollar_delta"].abs() > MAX_ORDER_USD + 1e-6]
    checks.append(
        {
            "check": "no_order_exceeds_max_order_usd",
            "ok": oversized.empty,
            "detail": ",".join(oversized["ticker"].astype(str).tolist()) if not oversized.empty else "ok",
        }
    )

    fractional = actionable[
        (actionable["estimated_shares"].round(0) - actionable["estimated_shares"]).abs() > 1e-9
    ]
    checks.append(
        {
            "check": "whole_share_orders",
            "ok": fractional.empty,
            "detail": ",".join(fractional["ticker"].astype(str).tolist()) if not fractional.empty else "ok",
        }
    )

    rejected = df[df["action"].eq("REJECT")]
    checks.append(
        {
            "check": "rejected_orders_present",
            "ok": rejected.empty,
            "detail": ",".join(
                f"{row.ticker}:{row.reject_reason}" for row in rejected.itertuples(index=False)
            ) if not rejected.empty else "none",
            "warning_only": True,
        }
    )

    cash_after = None
    if REPORT_PATH.exists():
        for line in REPORT_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("preview_cash_after_orders:"):
                try:
                    cash_after = float(line.split(":", 1)[1].strip())
                except ValueError:
                    cash_after = None
                break

    checks.append(
        {
            "check": "preview_cash_after_orders_non_negative",
            "ok": cash_after is not None and cash_after >= MIN_CASH_AFTER_ORDERS,
            "detail": str(cash_after),
        }
    )

    hard_checks = [c for c in checks if not c.get("warning_only")]
    status = "ready_for_manual_review" if all(c["ok"] for c in hard_checks) else "not_ready"

    return {
        "status": status,
        "order_count": int(len(actionable)),
        "rejected_count": int(len(rejected)),
        "checks": checks,
    }


def main() -> None:
    result = check_orders()
    VALIDATION_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "Paper Order Preview Validation Report",
        "",
        f"status: {result['status']}",
        f"order_count: {result.get('order_count', 0)}",
        f"rejected_count: {result.get('rejected_count', 0)}",
        "",
        "checks:",
    ]

    for check in result["checks"]:
        marker = "OK" if check["ok"] else ("WARN" if check.get("warning_only") else "FAIL")
        lines.append(f"- {marker}: {check['check']} :: {check['detail']}")

    VALIDATION_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(VALIDATION_REPORT.read_text())


if __name__ == "__main__":
    main()
