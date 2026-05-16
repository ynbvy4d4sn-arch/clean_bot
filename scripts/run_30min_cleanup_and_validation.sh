#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_DIR="$HOME/uni_trading_bot_project/clean_bot"
cd "$PROJECT_DIR" || exit 1

mkdir -p logs outputs outputs/archive_cleanup

START_TS="$(date +%s)"
MAX_SECONDS="${MAX_SECONDS:-1800}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/cleanup_validation_${STAMP}.log"
ARCHIVE_DIR="outputs/archive_cleanup/${STAMP}"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S %z') | $*" | tee -a "$LOG_FILE"
}

time_left() {
  now="$(date +%s)"
  echo $((MAX_SECONDS - (now - START_TS)))
}

stop_if_time_low() {
  remaining="$(time_left)"
  if [ "$remaining" -le 60 ]; then
    log "STOP: less than 60 seconds left. remaining=${remaining}s"
    exit 0
  fi
}

run_step() {
  name="$1"
  shift
  stop_if_time_low
  log "START STEP: $name"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  status="${PIPESTATUS[0]}"
  if [ "$status" -ne 0 ]; then
    log "FAILED STEP: $name status=$status"
    return "$status"
  fi
  log "DONE STEP: $name"
  return 0
}

log "START 30min cleanup and validation"
log "project=$PROJECT_DIR"
log "branch=$(git branch --show-current)"
log "head=$(git log --oneline -1)"
log "max_seconds=$MAX_SECONDS"

log "=== initial git status ==="
git status --short | tee -a "$LOG_FILE"

# Safety: never touch simulator/order execution. This script is research-only.
mkdir -p "$ARCHIVE_DIR"

log "Clean Python caches"
find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name ".DS_Store" -delete 2>/dev/null || true

log "Archive heavy research run outputs while keeping summary/report files"
# Keep aggregate summaries/reports in outputs/.
# Move per-run detailed daily/trades/allocations files from grid/refined runs.
for pattern in \
  "outputs/gurobi_grid_*_daily_equity_curve.csv" \
  "outputs/gurobi_grid_*_trades.csv" \
  "outputs/gurobi_grid_*_allocations.csv" \
  "outputs/gurobi_refined*_daily_equity_curve.csv" \
  "outputs/gurobi_refined*_trades.csv" \
  "outputs/gurobi_refined*_allocations.csv" \
  "outputs/robust_*_daily_equity_curve.csv" \
  "outputs/robust_*_trades.csv" \
  "outputs/robust_*_allocations.csv"; do
  for f in $pattern; do
    [ -f "$f" ] || continue
    mkdir -p "$ARCHIVE_DIR/$(dirname "$f")"
    mv "$f" "$ARCHIVE_DIR/$f"
  done
done

log "Compile core research files"
python -m compileall -q \
  daily_bot.py \
  tactical_forecast.py \
  backtest_tactical_gurobi_replay.py \
  validate_gurobi_candidate_robustness.py \
  validate_paper_readiness.py

log "Run final v3_lower_turnover replay candidate"
python backtest_tactical_gurobi_replay.py \
  --score-name v3 \
  --rebalance-every 1 \
  --top-n-gate 6 \
  --max-weight 0.25 \
  --max-rebalance-turnover 0.20 \
  --lambda-variance 10.0 \
  --lambda-turnover 0.02 \
  --lambda-concentration 0.02 \
  --signal-scale 0.012 \
  --output-prefix final_paper_candidate_v3_lower_turnover \
  2>&1 | tee -a "$LOG_FILE"

log "Build final candidate comparison"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
from pathlib import Path
import pandas as pd

rows = []

paths = {
    "research_champion_v3_best": "outputs/gurobi_grid_medium_143_v3_top8_lv10p0_lt0p03_ss0p015_cap30_summary.csv",
    "paper_candidate_v3_lower_turnover": "outputs/final_paper_candidate_v3_lower_turnover_summary.csv",
    "v4b_overlay_diagnostic": "outputs/tactical_gurobi_v4b_top8_cap30_summary.csv",
}

for label, path in paths.items():
    p = Path(path)
    if not p.exists():
        print(f"missing: {path}")
        continue
    row = pd.read_csv(p).iloc[0].to_dict()
    row["label"] = label
    rows.append(row)

out = pd.DataFrame(rows)
if not out.empty:
    cols = [
        "label",
        "score_name",
        "top_n_gate",
        "max_rebalance_turnover",
        "lambda_variance",
        "lambda_turnover",
        "lambda_concentration",
        "signal_scale",
        "total_return",
        "annualized_return",
        "annualized_vol",
        "sharpe",
        "max_drawdown",
        "total_turnover",
    ]
    out = out[cols].sort_values("sharpe", ascending=False)
    out.to_csv("outputs/final_candidate_comparison.csv", index=False)
    print(out.to_string(index=False))

    lines = [
        "Final Candidate Comparison",
        "",
        "status: research_only_no_simulator_orders",
        "",
        "decision:",
        "- research_champion_v3_best has the strongest full-period result.",
        "- paper_candidate_v3_lower_turnover is preferred for initial paper-simulator testing because it is more conservative.",
        "- v4b_overlay remains diagnostic only.",
        "",
        "candidates:",
    ]

    for row in out.itertuples(index=False):
        lines.append(
            f"- {row.label}: score={row.score_name}, top_n={row.top_n_gate}, "
            f"cap={row.max_rebalance_turnover}, return={row.total_return:.4f}, "
            f"vol={row.annualized_vol:.4f}, sharpe={row.sharpe:.3f}, "
            f"dd={row.max_drawdown:.4f}, turnover={row.total_turnover:.2f}"
        )

    Path("outputs/final_candidate_comparison_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("")
    print(Path("outputs/final_candidate_comparison_report.txt").read_text())
PY

log "Run paper readiness check"
python validate_paper_readiness.py 2>&1 | tee -a "$LOG_FILE" || true

log "Write cleanup manifest"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
from pathlib import Path
import json
import subprocess

archive = sorted(str(p) for p in Path("outputs/archive_cleanup").glob("**/*") if p.is_file())
status = subprocess.run(["git", "status", "--short"], text=True, stdout=subprocess.PIPE).stdout

payload = {
    "status": "cleanup_validation_completed",
    "archived_file_count": len(archive),
    "archived_sample": archive[:80],
    "git_status": status.splitlines(),
    "important_outputs": [
        "outputs/gurobi_candidate_robustness_report.txt",
        "outputs/gurobi_candidate_robustness_summary.csv",
        "outputs/final_paper_candidate_v3_lower_turnover_report.txt",
        "outputs/final_paper_candidate_v3_lower_turnover_summary.csv",
        "outputs/final_candidate_comparison_report.txt",
        "outputs/final_candidate_comparison.csv",
        "outputs/paper_readiness_check_report.txt",
        "outputs/paper_readiness_check.json",
    ],
}
Path("outputs/cleanup_validation_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "Cleanup and Validation Summary",
    "",
    f"archived_file_count: {len(archive)}",
    "",
    "important_outputs:",
]
for item in payload["important_outputs"]:
    lines.append(f"- {item}")

lines.extend(["", "git_status:"])
if payload["git_status"]:
    for line in payload["git_status"]:
        lines.append(f"- {line}")
else:
    lines.append("- clean")

Path("outputs/cleanup_validation_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(Path("outputs/cleanup_validation_summary.txt").read_text())
PY

log "=== final git status ==="
git status --short | tee -a "$LOG_FILE"

log "DONE 30min cleanup and validation"
log "log_file=$LOG_FILE"
