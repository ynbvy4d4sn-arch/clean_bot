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

echo "Running health check..."
"$PYTHON_BIN" health_check.py --quick

echo
echo "Running daily dry-run..."
"$PYTHON_BIN" daily_bot.py --dry-run --mode single --force-refresh

echo
for report in \
  "outputs/current_data_freshness_report.txt" \
  "outputs/current_portfolio_report.txt" \
  "outputs/discrete_optimization_report.txt" \
  "outputs/rebalance_decision_report.txt"
do
  echo "=== ${report##*/} ==="
  if [[ -f "$report" ]]; then
    sed -n '1,40p' "$report"
  else
    echo "Missing: $report"
  fi
  echo
done

echo "best_discrete_order_preview.csv: $SCRIPT_DIR/outputs/best_discrete_order_preview.csv"
echo "Dry-run only. No real orders were sent."
