# Diagnostics Layer

## Purpose

The diagnostics layer collects structured run context for the research path in `main.py`
and the daily simulator path in `daily_bot.py`.

It does not:

- change code automatically
- start Codex automatically
- deploy anything automatically
- enable real broker trading

If the bot is uncertain, it should prefer `HOLD` or `PAUSE` and still write reports.

## Main Outputs

Daily-bot runs write these files to `outputs/`:

- `run_diagnostics.json`
- `error_log.csv`
- `warnings_log.csv`
- `rejected_orders_report.csv` when rejected orders exist
- `performance_flags.csv` when flags exist
- `codex_daily_debug_report.md`
- `codex_next_prompt.md`
- `daily_analysis_report.md`

Research runs from `main.py` write these local diagnostics files:

- `research_run_diagnostics.json`
- `research_codex_debug_report.md`
- `research_codex_next_prompt.md`
- `research_analysis_report.md`

## How To Read The Reports

### `codex_daily_debug_report.md`

This is the technical report for manual debugging work. It summarizes:

- run metadata
- data source and freshness context
- optimizer and model status
- candidate selection and execution gate state
- rejected orders
- errors and warnings
- suspected root causes
- suggested Codex follow-up tasks

### `codex_next_prompt.md`

This is a copy/paste-ready prompt for a manual Codex debugging session.
It includes:

- safety rules
- the current debug report
- likely affected files
- suggested tests
- acceptance criteria

### `daily_analysis_report.md`

This is the human-readable summary for the end of the day. It explains:

- what the bot decided
- whether anything was traded or only previewed
- the data state
- the main risk and cost context
- errors, warnings and next checks

## Email Activation

Email is disabled by default.

Required environment settings:

- `ENABLE_EMAIL_NOTIFICATIONS=true`
- `ENABLE_DAILY_ANALYSIS_EMAIL=true`
- complete SMTP settings
- at least one recipient via `DAILY_ANALYSIS_EMAIL_TO` or `EMAIL_TO`

Additional daily-analysis settings:

- `DAILY_ANALYSIS_EMAIL_TIME_LOCAL=18:00`
- `DAILY_ANALYSIS_TIMEZONE=Europe/Berlin`
- `SEND_ANALYSIS_EMAIL_ONLY_ON_TRADING_DAYS=true`
- `SEND_ANALYSIS_EMAIL_ON_ERRORS_ONLY=false`
- `SEND_ANALYSIS_EMAIL_INCLUDE_CODEX_PROMPT=true`
- `DAILY_ANALYSIS_EMAIL_SUBJECT_PREFIX=[Portfolio Bot Daily Analysis]`

If email is disabled or SMTP is incomplete, the bot still writes all local reports.

## Trading-Day And Time Gate

The email gate is scheduler-friendly:

- by default, emails are only sent on trading days
- if no robust calendar library is available, the fallback is Monday to Friday
- if the bot runs before the configured local send time, it only writes local reports
- if `SEND_ANALYSIS_EMAIL_ON_ERRORS_ONLY=true`, an early run may still send mail when critical errors exist

## Example Cronjob

Run the daily bot on trading days after market close in Berlin time:

```cron
10 18 * * 1-5 cd /path/to/project && . .venv/bin/activate && python daily_bot.py --dry-run --mode single >> logs/daily_bot.log 2>&1
```

Optional paper-style simulator run, only if you intentionally enable that mode:

```cron
10 18 * * 1-5 cd /path/to/project && . .venv/bin/activate && python daily_bot.py --mode single >> logs/daily_bot.log 2>&1
```

Cron often runs with a minimal environment and a different working directory.
Use an explicit `cd` into the project and prefer absolute paths.

## Safety Notes

- The bot does not modify code automatically.
- The bot does not start Codex automatically.
- The bot does not deploy automatically.
- Real broker execution stays disabled by default.
- `DRY_RUN=true` remains the default.
- Diagnostics must never be the reason why an error report is missing.
