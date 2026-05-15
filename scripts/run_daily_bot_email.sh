#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/uni_trading_bot_project/clean_bot"
LOG_DIR="$PROJECT_DIR/logs"
RUN_LOG="$LOG_DIR/daily_bot_email_wrapper.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

log() {
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "$RUN_LOG"
}

log "START daily bot email wrapper"

if [ ! -f ".env" ]; then
  log "FAIL .env missing"
  exit 20
fi

if [ ! -f "daily_bot.py" ]; then
  log "FAIL daily_bot.py missing"
  exit 21
fi

if [ ! -f "data/trading_calendar_2026.csv" ]; then
  log "FAIL trading calendar missing"
  exit 22
fi

if ! /usr/bin/curl -Is --max-time 10 https://query1.finance.yahoo.com >/dev/null 2>&1; then
  log "FAIL internet/yahoo preflight failed"
  exit 30
fi

if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/opt/miniconda3/etc/profile.d/conda.sh"
  conda activate base
fi

PYTHON_BIN="$(command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
  log "FAIL python not found"
  exit 31
fi
log "python=$PYTHON_BIN"

python - <<'PY'
from config import build_params, get_email_gate_status
params = build_params()
gate = get_email_gate_status(params)
print("preflight_email_allowed=", gate.get("real_email_send_allowed"))
print("preflight_email_blockers=", gate.get("blockers"))
print("preflight_external_broker=", params.get("enable_external_broker"))
print("preflight_investopedia=", params.get("enable_investopedia_simulator"))
print("preflight_local_paper=", params.get("enable_local_paper_trading"))
print("preflight_dry_run=", params.get("dry_run"))
if params.get("enable_external_broker"):
    raise SystemExit("External broker must remain disabled.")
if params.get("enable_investopedia_simulator"):
    raise SystemExit("Investopedia simulator must remain disabled.")
PY

log "RUN daily_bot.py --dry-run --skip-submit"
python daily_bot.py --dry-run --skip-submit

python - <<'PY'
import json
from pathlib import Path

diag_path = Path("outputs/run_diagnostics.json")
if not diag_path.exists():
    raise SystemExit("outputs/run_diagnostics.json missing")

data = json.loads(diag_path.read_text())
payload = data.get("model_context", {}).get("daily_review_payload") or {}
run_status = payload.get("run_status", {})
data_status = payload.get("data_status", {})
current = payload.get("current_portfolio", {})
orders = payload.get("order_summary", {})

print("postcheck_final_action=", run_status.get("final_action"))
print("postcheck_execution_mode=", run_status.get("execution_mode"))
print("postcheck_data_freshness_ok=", data_status.get("data_freshness_ok"))
print("postcheck_latest_price_date=", data_status.get("latest_price_date"))
print("postcheck_synthetic_data=", data_status.get("synthetic_data"))
print("postcheck_used_cache_fallback=", data_status.get("used_cache_fallback"))
print("postcheck_nav_usd=", current.get("nav_usd"))
print("postcheck_positions_count=", current.get("positions_count"))
print("postcheck_order_count=", orders.get("order_count"))

if data_status.get("synthetic_data"):
    raise SystemExit("Synthetic data detected after run.")
if not data_status.get("data_freshness_ok"):
    raise SystemExit("Data freshness failed after run.")
if not current.get("nav_usd") or float(current.get("nav_usd")) <= 0:
    raise SystemExit("NAV invalid after run.")
PY

if [ -f "outputs/email_safety_report.txt" ]; then
  log "EMAIL STATUS"
  grep -E "real_email_send_allowed|email_send_attempted|provider_accepted|email_result_reason|delivery_status" outputs/email_safety_report.txt | tee -a "$RUN_LOG" || true
else
  log "WARN outputs/email_safety_report.txt missing"
fi

log "DONE daily bot email wrapper"
