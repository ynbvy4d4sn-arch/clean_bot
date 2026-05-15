#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

export MPLCONFIGDIR="$SCRIPT_DIR/outputs/.mplconfig"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$SCRIPT_DIR/outputs"

# Safety defaults for the email-review entry point: no broker, no simulator automation, no real orders.
export DRY_RUN=true
export ENABLE_EXTERNAL_BROKER=false
export ENABLE_INVESTOPEDIA_SIMULATOR=false
export ENABLE_LOCAL_PAPER_TRADING=false
export EMAIL_DRY_RUN=true
export EMAIL_SEND_ENABLED=false
export ENABLE_EMAIL_NOTIFICATIONS=false

echo "Running config validation..."
"$PYTHON_BIN" config_validation.py

echo
echo "Running health check..."
"$PYTHON_BIN" health_check.py --quick

echo
echo "Running daily review dry-run..."
"$PYTHON_BIN" daily_bot.py --dry-run --mode single --force-refresh

echo
echo "Regenerating Daily Review outputs from run_diagnostics.json..."
"$PYTHON_BIN" daily_portfolio_review.py --input-file outputs/run_diagnostics.json --output-dir outputs

echo
echo "Quick summary:"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import re
import shutil
import sys

root = Path("outputs")

def read(path: str) -> str:
    return (root / path).read_text(encoding="utf-8")

def ensure_non_empty(path: str) -> None:
    file_path = root / path
    if not file_path.exists():
        raise SystemExit(f"Missing required output: {file_path}")
    if file_path.stat().st_size <= 0:
        raise SystemExit(f"Empty required output: {file_path}")

def find_key(text: str, key: str, prefix: str = "") -> str:
    pattern = rf"^{re.escape(prefix + key)}: (.+)$"
    match = re.search(pattern, text, re.M)
    return match.group(1).strip() if match else "n/a"

required_non_empty = [
    "daily_portfolio_review.txt",
    "daily_review_email.html",
    "daily_review_report.tex",
    "hold_dominance_analysis.txt",
    "hold_sensitivity_report.txt",
    "decision_history.csv",
    "daily_email_subject.txt",
    "daily_email_briefing.txt",
    "daily_portfolio_briefing.md",
    "daily_portfolio_briefing.html",
    "latest_email_notification.txt",
    "email_safety_report.txt",
    "email_delivery_diagnosis_report.txt",
    "daily_review_validation_report.txt",
    "email_final_acceptance_report.txt",
    "last_email_state.json",
    "charts/current_portfolio_allocation.png",
    "charts/current_vs_target_weights.png",
]
for relative_path in required_non_empty:
    ensure_non_empty(relative_path)
if shutil.which("pdflatex"):
    ensure_non_empty("daily_review_report.pdf")

review = read("daily_portfolio_review.txt")
freshness = read("current_data_freshness_report.txt")
validation = read("daily_review_validation_report.txt")
email_safety = read("email_safety_report.txt")
acceptance = read("email_final_acceptance_report.txt")
subject = read("daily_email_subject.txt").strip()
mail = read("daily_email_briefing.txt").strip()
state = json.loads(read("last_email_state.json"))

print(f"subject: {subject or 'n/a'}")
print(f"final_action: {find_key(review, 'final_action', '- ')}")
print(f"data_source: {find_key(freshness, 'data_source')}")
print(f"used_cache_fallback: {find_key(freshness, 'used_cache_fallback')}")
print(f"latest_price_date: {find_key(freshness, 'latest_price_date')}")
print(f"manual_order_count: {find_key(validation, 'manual_eligible_order_count')}")
print(f"email_preview_ready: {'yes' if subject and mail else 'no'}")
print(f"email_send_attempted: {state.get('email_send_attempted', False)}")
print(f"provider_accepted: {state.get('provider_accepted', state.get('email_result_sent', False))}")
print(f"delivery_confirmed: {state.get('delivery_confirmed', False)}")
print(f"email_result_reason: {state.get('email_result_reason', 'n/a')}")
print(f"acceptance_status: {acceptance.strip().splitlines()[-1] if acceptance.strip() else 'n/a'}")
print("output_paths:")
for path in [
    "outputs/daily_portfolio_review.txt",
    "outputs/daily_review_email.html",
    "outputs/daily_review_report.tex",
    "outputs/daily_review_report.pdf",
    "outputs/hold_dominance_analysis.txt",
    "outputs/hold_sensitivity_report.txt",
    "outputs/decision_history.csv",
    "outputs/daily_email_subject.txt",
    "outputs/daily_email_briefing.txt",
    "outputs/daily_portfolio_briefing.md",
    "outputs/daily_portfolio_briefing.html",
    "outputs/latest_email_notification.txt",
    "outputs/email_safety_report.txt",
    "outputs/email_delivery_diagnosis_report.txt",
    "outputs/daily_review_validation_report.txt",
    "outputs/email_final_acceptance_report.txt",
    "outputs/manual_simulator_orders.csv",
    "outputs/charts/current_portfolio_allocation.png",
    "outputs/charts/current_vs_target_weights.png",
]:
    print(f"- {path}")

gate_open = find_key(email_safety, "real_email_send_allowed") == "true"
send_attempted = bool(state.get("email_send_attempted", False))
send_success = bool(state.get("provider_accepted", state.get("email_result_sent", False)))
reason = str(state.get("email_result_reason", ""))
if gate_open and send_attempted and not send_success:
    sys.exit(2)
PY

echo
echo "No real orders were sent."
