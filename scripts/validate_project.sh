#!/usr/bin/env bash
set -euo pipefail

python -m pytest tests_new -q
python -m compileall -q .
python smoke_test.py
python daily_bot.py --dry-run --skip-submit

git checkout -- data/prices_cache.csv data/prices_cache.csv.meta.json 2>/dev/null || true

echo "Validation passed."
