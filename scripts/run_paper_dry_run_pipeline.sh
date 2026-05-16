#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_DIR="$HOME/uni_trading_bot_project/clean_bot"
cd "$PROJECT_DIR" || exit 1

mkdir -p logs outputs

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/paper_dry_run_pipeline_${STAMP}.log"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S %z') | $*" | tee -a "$LOG_FILE"
}

run_step() {
  name="$1"
  shift
  log "START STEP: $name"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  status="${PIPESTATUS[0]}"
  if [ "$status" -ne 0 ]; then
    log "FAILED STEP: $name status=$status"
    exit "$status"
  fi
  log "DONE STEP: $name"
}

log "START paper dry-run pipeline"
log "NO ORDERS WILL BE SENT"
log "branch=$(git branch --show-current)"
log "head=$(git log --oneline -1)"
log "git_status_start=$(git status --short | tr '\n' ';')"

run_step "compile pipeline files" python -m compileall -q \
  backtest_tactical_gurobi_replay.py \
  build_paper_order_preview.py \
  validate_paper_order_preview.py \
  paper_simulator_adapter.py

run_step "build paper preview allocation" python backtest_tactical_gurobi_replay.py \
  --score-name v3 \
  --rebalance-every 1 \
  --top-n-gate 6 \
  --max-weight 0.25 \
  --max-rebalance-turnover 0.20 \
  --lambda-variance 10.0 \
  --lambda-turnover 0.02 \
  --lambda-concentration 0.02 \
  --signal-scale 0.012 \
  --output-prefix paper_preview_candidate_v3_lower_turnover

run_step "build paper order preview" python build_paper_order_preview.py
run_step "validate paper order preview" python validate_paper_order_preview.py
run_step "build dry-run simulator payload" python paper_simulator_adapter.py

log "Write final pipeline report"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
from pathlib import Path
import json

outputs = {
    "paper_preview_report": Path("outputs/paper_preview_candidate_v3_lower_turnover_report.txt"),
    "order_preview_report": Path("outputs/paper_order_preview_report.txt"),
    "order_validation_report": Path("outputs/paper_order_preview_validation_report.txt"),
    "dry_run_report": Path("outputs/paper_simulator_dry_run_report.txt"),
    "dry_run_orders": Path("outputs/paper_simulator_dry_run_orders.json"),
}

lines = [
    "Paper Dry-Run Pipeline Report",
    "",
    "status: completed_no_orders_sent",
    "candidate: v3_lower_turnover",
    "",
    "outputs:",
]

for name, path in outputs.items():
    lines.append(f"- {name}: {path} exists={path.exists()}")

orders_path = outputs["dry_run_orders"]
if orders_path.exists():
    payload = json.loads(orders_path.read_text())
    lines.extend([
        "",
        f"dry_run_status: {payload.get('status')}",
        f"dry_run_only: {payload.get('dry_run_only')}",
        f"order_count: {payload.get('order_count')}",
        "",
        "orders:",
    ])
    for order in payload.get("orders", []):
        lines.append(
            f"- {order['side'].upper()} {order['ticker']} "
            f"qty={order['quantity']} "
            f"estimated_notional={order['estimated_notional']:.2f}"
        )

Path("outputs/paper_dry_run_pipeline_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(Path("outputs/paper_dry_run_pipeline_report.txt").read_text())
PY

log "git_status_end=$(git status --short | tr '\n' ';')"
log "DONE paper dry-run pipeline"
log "log_file=$LOG_FILE"
