#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="logs/tud_daily_report_${TS}.log"

exec > >(tee -a "$LOG") 2>&1

echo "=== TUD Daily Report Pipeline ==="
echo "timestamp: $TS"
echo "cwd: $(pwd)"
echo ""

echo "=== 1) Auto-refresh Investopedia TUD portfolio ==="

if [ -f "investopedia_auto_refresh.py" ]; then
  echo "running investopedia_auto_refresh.py"
  python investopedia_auto_refresh.py || {
    echo "ERROR: automatic Investopedia refresh failed"
    exit 1
  }
else
  echo "ERROR: investopedia_auto_refresh.py not found"
  exit 1
fi

if [ -f "investopedia_tud_portfolio_parser.py" ]; then
  echo "running investopedia_tud_portfolio_parser.py"
  python investopedia_tud_portfolio_parser.py || {
    echo "ERROR: portfolio parser failed"
    exit 1
  }
else
  echo "ERROR: investopedia_tud_portfolio_parser.py not found"
  exit 1
fi


echo ""
echo "=== 1b) Guard parsed Investopedia positions ==="

if [ ! -f "config/paper_positions.csv" ]; then
  echo "ERROR: config/paper_positions.csv missing after parser"
  exit 1
fi

if ! head -1 config/paper_positions.csv | grep -q "ticker"; then
  echo "ERROR: config/paper_positions.csv has wrong header"
  head -5 config/paper_positions.csv
  exit 1
fi

position_rows=$(tail -n +2 config/paper_positions.csv | wc -l | tr -d ' ')
if [ "$position_rows" -lt 5 ]; then
  echo "ERROR: too few parsed positions: $position_rows"
  cat config/paper_positions.csv
  exit 1
fi

echo "parsed position rows: $position_rows"
head -12 config/paper_positions.csv

echo ""
echo "=== 2) Pull newest trade-history export if present ==="

mkdir -p data/investopedia

LATEST_TRADE_HISTORY="$(find "$HOME/Downloads" data/investopedia . \
  -maxdepth 2 \
  \( -iname 'trade-history-*.xls' -o -iname 'trade-history-*.html' -o -iname 'trade-history-*.csv' \) \
  -type f \
  -print 2>/dev/null \
  | sort \
  | tail -n 1 || true)"

if [ -n "${LATEST_TRADE_HISTORY}" ]; then
  echo "latest trade history: ${LATEST_TRADE_HISTORY}"
  cp "${LATEST_TRADE_HISTORY}" "data/investopedia/$(basename "${LATEST_TRADE_HISTORY}")" || true
else
  echo "WARNING: no trade-history export found"
fi

echo ""
echo "=== 3) Build paper order preview ==="
python build_paper_order_preview.py


echo ""
echo "=== 3b) Guard order preview source ==="

if grep -q "current_position_source: empty_all_cash" outputs/paper_order_preview_report.txt 2>/dev/null || \
   grep -q "current_position_source: empty_all_cash" outputs/paper_order_preview.txt 2>/dev/null; then
  echo "ERROR: order preview is using empty_all_cash. Refusing to continue."
  exit 1
fi

if ! grep -q "current_position_source: positions_file:" outputs/paper_order_preview_report.txt 2>/dev/null && \
   ! grep -q "current_position_source: positions_file:" outputs/paper_order_preview.txt 2>/dev/null; then
  echo "ERROR: order preview did not confirm positions_file source."
  exit 1
fi

echo "order preview uses parsed positions file."

echo ""
echo "=== 4) Validate order preview ==="
python validate_paper_order_preview.py

echo ""
echo "=== 5) Build dry-run simulator payload ==="
python paper_simulator_adapter.py

echo ""
echo "=== 6) Render Butter Brezel report card ==="
python render_tud_report_card.py

echo ""
echo "=== 7) Send email report ==="

export ENABLE_EMAIL_NOTIFICATIONS="${ENABLE_EMAIL_NOTIFICATIONS:-true}"
export EMAIL_SEND_ENABLED="${EMAIL_SEND_ENABLED:-true}"
export EMAIL_DRY_RUN="${EMAIL_DRY_RUN:-false}"
export USER_CONFIRMED_EMAIL_PHASE="${USER_CONFIRMED_EMAIL_PHASE:-true}"
export PHASE="${PHASE:-DAILY_REVIEW_SEND_READY}"

python send_tud_dry_run_email.py --force

echo ""
echo "=== 8) Result ==="
cat outputs/tud_dry_run_email_result.json || true

echo ""
echo "=== 9) Key outputs ==="
ls -lh \
  outputs/tud_daily_report_card.png \
  outputs/tud_daily_report_card.html \
  outputs/tud_report_data_audit.txt \
  outputs/paper_order_preview.csv \
  outputs/paper_simulator_dry_run_orders.json \
  outputs/tud_dry_run_email_result.json 2>/dev/null || true

echo ""
echo "=== done ==="
