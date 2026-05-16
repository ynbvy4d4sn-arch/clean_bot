#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_DIR="$HOME/uni_trading_bot_project/clean_bot"
cd "$PROJECT_DIR" || exit 1

mkdir -p logs outputs

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/paper_preview_only_${STAMP}.log"

echo "$(date '+%Y-%m-%d %H:%M:%S %z') | START paper preview only" | tee -a "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S %z') | NO simulator orders will be sent" | tee -a "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S %z') | candidate=v3_lower_turnover" | tee -a "$LOG_FILE"

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
  --output-prefix paper_preview_candidate_v3_lower_turnover \
  2>&1 | tee -a "$LOG_FILE"

python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
from pathlib import Path
import pandas as pd

alloc_path = Path("outputs/paper_preview_candidate_v3_lower_turnover_allocations.csv")
summary_path = Path("outputs/paper_preview_candidate_v3_lower_turnover_summary.csv")
report_path = Path("outputs/paper_preview_only_report.txt")

lines = [
    "Paper Preview Only Report",
    "",
    "status: preview_only_no_orders_sent",
    "candidate: v3_lower_turnover",
    "",
]

if summary_path.exists():
    s = pd.read_csv(summary_path).iloc[0]
    lines.extend([
        "summary:",
        f"- total_return: {s.total_return:.4f}",
        f"- sharpe: {s.sharpe:.3f}",
        f"- max_drawdown: {s.max_drawdown:.4f}",
        f"- total_turnover: {s.total_turnover:.2f}",
        "",
    ])

if alloc_path.exists():
    alloc = pd.read_csv(alloc_path)
    latest_date = alloc["date"].max()
    latest = alloc[alloc["date"].eq(latest_date)].sort_values("weight", ascending=False)
    lines.append(f"latest_preview_allocation_date: {latest_date}")
    lines.append("latest_preview_allocation:")
    for row in latest.itertuples(index=False):
        lines.append(f"- {row.ticker}: {row.weight:.4f}")

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report_path.read_text())
PY

echo "$(date '+%Y-%m-%d %H:%M:%S %z') | DONE paper preview only" | tee -a "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S %z') | log_file=$LOG_FILE" | tee -a "$LOG_FILE"
