"""Send TUD Investopedia dry-run trading report by email.

Uses existing project notification infrastructure.
Does not submit simulator or broker orders.

Inputs:
- outputs/paper_dry_run_pipeline_report.txt
- outputs/paper_order_preview_report.txt
- outputs/paper_order_preview_validation_report.txt
- outputs/paper_simulator_dry_run_report.txt
- outputs/paper_order_submit_guard_report.txt

Outputs:
- outputs/tud_dry_run_email_subject.txt
- outputs/tud_dry_run_email_body.txt
- outputs/tud_dry_run_email_result.json
"""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from notifications import load_email_settings, send_email_notification
from config import get_email_gate_status


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

PIPELINE_REPORT = OUTPUT_DIR / "paper_dry_run_pipeline_report.txt"
ORDER_PREVIEW_REPORT = OUTPUT_DIR / "paper_order_preview_report.txt"
VALIDATION_REPORT = OUTPUT_DIR / "paper_order_preview_validation_report.txt"
DRY_RUN_REPORT = OUTPUT_DIR / "paper_simulator_dry_run_report.txt"
SUBMIT_GUARD_REPORT = OUTPUT_DIR / "paper_order_submit_guard_report.txt"

SUBJECT_OUT = OUTPUT_DIR / "tud_dry_run_email_subject.txt"
BODY_OUT = OUTPUT_DIR / "tud_dry_run_email_body.txt"
RESULT_OUT = OUTPUT_DIR / "tud_dry_run_email_result.json"
STATE_OUT = OUTPUT_DIR / "tud_dry_run_email_state.json"


BERLIN = ZoneInfo("Europe/Berlin")


def read_text(path: Path) -> str:
    if not path.exists():
        return f"[missing: {path}]"
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def extract_line(text: str, prefix: str, default: str = "unknown") -> str:
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return default


def extract_orders(pipeline_text: str) -> list[str]:
    lines = pipeline_text.splitlines()
    orders = []
    in_orders = False
    for line in lines:
        stripped = line.strip()
        if stripped == "orders:":
            in_orders = True
            continue
        if in_orders:
            if stripped.startswith("- "):
                orders.append(stripped)
            elif stripped:
                break
    return orders


def build_subject_and_body() -> tuple[str, str, dict]:
    now = datetime.now(BERLIN).strftime("%Y-%m-%d %H:%M:%S %Z")

    pipeline = read_text(PIPELINE_REPORT)
    preview = read_text(ORDER_PREVIEW_REPORT)
    validation = read_text(VALIDATION_REPORT)
    dry_run = read_text(DRY_RUN_REPORT)
    guard = read_text(SUBMIT_GUARD_REPORT)

    pipeline_status = extract_line(pipeline, "status")
    dry_run_status = extract_line(pipeline, "dry_run_status")
    order_count = extract_line(pipeline, "order_count")
    validation_status = extract_line(validation, "status")
    guard_status = extract_line(guard, "status")

    preview_cash_after = extract_line(preview, "preview_cash_after_orders")
    current_cash = extract_line(preview, "current_cash_usd")
    nav = extract_line(preview, "preview_nav_usd")
    target_date = extract_line(preview, "target_allocation_date")

    orders = extract_orders(pipeline)

    subject = (
        f"TUD Dry-Run Trading Report: {dry_run_status}, "
        f"{order_count} orders, guard={guard_status}"
    )

    body_lines = [
        "TUD Investopedia Dry-Run Trading Report",
        "",
        f"generated_at: {now}",
        "",
        "SUMMARY",
        f"- pipeline_status: {pipeline_status}",
        f"- dry_run_status: {dry_run_status}",
        f"- validation_status: {validation_status}",
        f"- submit_guard_status: {guard_status}",
        f"- order_count: {order_count}",
        f"- target_allocation_date: {target_date}",
        f"- preview_nav_usd: {nav}",
        f"- current_cash_usd: {current_cash}",
        f"- preview_cash_after_orders: {preview_cash_after}",
        "",
        "IMPORTANT",
        "- No simulator or broker orders were submitted.",
        "- This is a dry-run recommendation email only.",
        "- Submit guard remains the final safety barrier.",
        "",
        "DRY-RUN ORDERS",
    ]

    if orders:
        body_lines.extend(orders)
    else:
        body_lines.append("- none")

    body_lines.extend(
        [
            "",
            "ORDER PREVIEW DETAILS",
            preview,
            "",
            "VALIDATION REPORT",
            validation,
            "",
            "SUBMIT GUARD REPORT",
            guard,
            "",
            "PIPELINE REPORT",
            pipeline,
            "",
            "DRY-RUN ADAPTER REPORT",
            dry_run,
        ]
    )

    meta = {
        "pipeline_status": pipeline_status,
        "dry_run_status": dry_run_status,
        "validation_status": validation_status,
        "guard_status": guard_status,
        "order_count": order_count,
        "preview_cash_after_orders": preview_cash_after,
    }

    return subject, "\n".join(body_lines).strip() + "\n", meta


def load_state() -> dict:
    if not STATE_OUT.exists():
        return {}
    try:
        return json.loads(STATE_OUT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Send even if same body hash was already mailed.")
    parser.add_argument("--preview-only", action="store_true", help="Only write subject/body/result, do not call mail sender.")
    args = parser.parse_args()

    subject, body, meta = build_subject_and_body()
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

    SUBJECT_OUT.write_text(subject + "\n", encoding="utf-8")
    BODY_OUT.write_text(body, encoding="utf-8")

    state = load_state()
    duplicate = state.get("last_body_hash") == body_hash

    settings = load_email_settings()
    gate = get_email_gate_status(settings)

    result = {
        "attempted": False,
        "sent": False,
        "reason": "not_attempted",
        "duplicate": duplicate,
        "body_hash": body_hash,
        "gate": gate,
        "meta": meta,
    }

    if args.preview_only:
        result["reason"] = "preview_only"
    elif duplicate and not args.force:
        result["reason"] = "duplicate_body_hash"
    else:
        send_result = send_email_notification(
            subject=subject,
            body=body,
            recipient=str(settings.get("EMAIL_RECIPIENT", "") or settings.get("EMAIL_TO", "")).strip(),
            settings=settings,
        )
        result.update(send_result)
        result["attempted"] = bool(send_result.get("attempted", False))

        if bool(send_result.get("sent", False)) or bool(send_result.get("provider_accepted", False)):
            STATE_OUT.write_text(
                json.dumps(
                    {
                        "last_body_hash": body_hash,
                        "last_subject": subject,
                        "last_sent_at": datetime.now(BERLIN).isoformat(),
                        "last_result": send_result,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

    RESULT_OUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print("TUD dry-run email prepared.")
    print(f"subject: {subject}")
    print(f"body: {BODY_OUT}")
    print(f"result: {RESULT_OUT}")
    print(f"attempted: {result.get('attempted')}")
    print(f"sent: {result.get('sent')}")
    print(f"reason: {result.get('reason')}")
    print(f"gate_real_email_send_allowed: {gate.get('real_email_send_allowed')}")
    print(f"gate_blockers: {gate.get('blockers')}")


if __name__ == "__main__":
    main()
