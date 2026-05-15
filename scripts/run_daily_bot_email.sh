#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/uni_trading_bot_project/clean_bot"

# Load conda shell support if available.
if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/opt/miniconda3/etc/profile.d/conda.sh"
  conda activate base
fi

python daily_bot.py --dry-run --skip-submit
