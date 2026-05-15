"""Codex-oriented markdown reporting for diagnostics output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from diagnostics import RunDiagnostics, diagnostics_to_dict


def _rows_to_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_none_"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(str(value).replace("\n", " ").strip())
        body_lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body_lines])


def _bullet(label: str, value: Any) -> str:
    return f"- {label}: {value if value not in (None, '', []) else 'n/a'}"


def _collect_likely_files(payload: dict[str, Any]) -> list[str]:
    files: list[str] = []
    for item in payload.get("suspected_root_causes", []):
        for file_name in item.get("files", []) or []:
            text = str(file_name)
            if text and text not in files:
                files.append(text)
    for item in payload.get("suggested_codex_tasks", []):
        for file_name in item.get("files", []) or []:
            text = str(file_name)
            if text and text not in files:
                files.append(text)
    return files


def _collect_test_suggestions(payload: dict[str, Any]) -> list[str]:
    tests: list[str] = []
    for item in payload.get("suggested_codex_tasks", []):
        suggestion = item.get("test_suggestion") or item.get("test") or item.get("tests")
        if isinstance(suggestion, list):
            for value in suggestion:
                text = str(value)
                if text and text not in tests:
                    tests.append(text)
        elif suggestion:
            text = str(suggestion)
            if text and text not in tests:
                tests.append(text)
    return tests


def build_codex_debug_report(diagnostics: RunDiagnostics | dict[str, Any]) -> str:
    """Build a Codex-friendly markdown debug report."""

    payload = diagnostics_to_dict(diagnostics)
    data_context = payload.get("data_context", {})
    data_quality = payload.get("data_quality", {})
    model_context = payload.get("model_context", {})
    optimizer_context = payload.get("optimizer_context", {})
    candidate_context = payload.get("candidate_context", {})
    gate_context = payload.get("execution_gate_context", {})
    final_orders_summary = payload.get("final_orders_summary", {})

    sections = [
        "# Codex Daily Debug Report",
        "",
        "## 1. Run Summary",
        _bullet("Run ID", payload.get("run_id")),
        _bullet("Timestamp", payload.get("run_timestamp_utc")),
        _bullet("Local Date", payload.get("local_date")),
        _bullet("Signal Date", payload.get("signal_date")),
        _bullet("Execution Date", payload.get("execution_date")),
        _bullet("Mode", payload.get("mode")),
        _bullet("Dry Run", payload.get("dry_run")),
        _bullet("Selected Candidate", payload.get("selected_candidate")),
        _bullet("Final Action", payload.get("final_action")),
        _bullet("Execution Gate Status", gate_context.get("gate_status")),
        _bullet("Execution Mode", payload.get("execution_mode")),
        "",
        "## 2. Data Context",
        _bullet("data_source", data_context.get("data_source")),
        _bullet("cache_status", data_context.get("cache_status")),
        _bullet("synthetic_data", data_context.get("synthetic_data")),
        _bullet("used_cache_fallback", data_context.get("used_cache_fallback")),
        _bullet("latest_price_date", data_context.get("latest_price_date")),
        _bullet("expected_latest_trading_day", data_context.get("expected_latest_trading_day")),
        _bullet("tickers_loaded", data_context.get("tickers_loaded")),
        _bullet("tickers_failed", data_context.get("tickers_failed")),
        _bullet("data_freshness_ok", data_context.get("data_freshness_ok")),
        _bullet("staleness_days", data_context.get("staleness_days")),
        "",
        "## 3. Data Quality",
        _bullet("global_data_quality_score", data_quality.get("global_data_quality_score")),
        _bullet("removed_tickers", data_quality.get("removed_tickers")),
        _bullet("warnings", data_quality.get("warnings")),
        _bullet("errors", data_quality.get("errors")),
        "",
        "## 4. Model / Optimizer Context",
        _bullet("optimizer_success", optimizer_context.get("optimizer_success", optimizer_context.get("success"))),
        _bullet("optimizer_solver", optimizer_context.get("optimizer_solver", optimizer_context.get("solver_name"))),
        _bullet("optimizer_status", optimizer_context.get("optimizer_status", optimizer_context.get("status"))),
        _bullet("objective_value", optimizer_context.get("objective_value")),
        _bullet("model_confidence_score", model_context.get("model_confidence_score")),
        _bullet("risk_state", model_context.get("risk_state")),
        _bullet("factor_mode", model_context.get("factor_mode")),
        "",
        "## 5. Candidate Selection",
        _bullet("selected_candidate", payload.get("selected_candidate")),
        _bullet("net_robust_score", candidate_context.get("net_robust_score", candidate_context.get("best_discrete_score"))),
        _bullet("delta_vs_hold", candidate_context.get("delta_vs_hold")),
        _bullet("delta_vs_cash", candidate_context.get("delta_vs_cash")),
        _bullet("probability_beats_hold", candidate_context.get("probability_beats_hold")),
        _bullet("probability_beats_cash", candidate_context.get("probability_beats_cash")),
        _bullet("worst_scenario", candidate_context.get("worst_scenario")),
        _bullet("cvar_5", candidate_context.get("cvar_5")),
        "",
        "## 6. Execution Gate",
        _bullet("gate_status", gate_context.get("gate_status")),
        _bullet("action", gate_context.get("action")),
        _bullet("reason", gate_context.get("reason")),
        _bullet("trade_now_score", gate_context.get("trade_now_score")),
        _bullet("spread_cost", gate_context.get("spread_cost")),
        _bullet("slippage", gate_context.get("slippage")),
        _bullet("buffers", gate_context.get("buffers")),
        "",
        "## 7. Final Action",
        _bullet("Action", payload.get("final_action")),
        _bullet("Reason", payload.get("final_reason")),
        _bullet("Final Orders", final_orders_summary.get("order_count")),
        _bullet("Turnover", final_orders_summary.get("turnover")),
        _bullet("estimated_cost", final_orders_summary.get("estimated_cost")),
        "",
        "## 8. Rejected Orders",
        _rows_to_markdown_table(
            payload.get("rejected_orders", []),
            ["ticker", "side", "reason", "extra_json"],
        ),
        "",
        "## 9. Errors",
        _rows_to_markdown_table(
            payload.get("errors", []),
            ["severity", "stage", "module", "function", "exception_type", "exception_message"],
        ),
        "",
        "## 10. Warnings",
        _rows_to_markdown_table(
            payload.get("warnings", []),
            ["severity", "stage", "module", "message"],
        ),
        "",
        "## 11. Performance Flags",
        _rows_to_markdown_table(
            payload.get("performance_flags", []),
            ["flag_name", "severity", "message", "suggested_action", "files_likely_involved"],
        ),
        "",
        "## 12. Suspected Root Causes",
    ]
    root_causes = payload.get("suspected_root_causes", [])
    if root_causes:
        sections.extend([f"- {item.get('message')} ({item.get('severity', 'INFO')})" for item in root_causes])
    else:
        sections.append("- none recorded")
    sections.extend(["", "## 13. Suggested Codex Tasks"])
    tasks = payload.get("suggested_codex_tasks", [])
    if tasks:
        for task in tasks:
            sections.append(f"- Task: {task.get('task', 'n/a')}")
            sections.append(f"  File: {task.get('files', [])}")
            sections.append(f"  Problem: {task.get('problem', 'n/a')}")
            sections.append(f"  Zielverhalten: {task.get('expected_behavior', 'n/a')}")
            sections.append(f"  Testvorschlag: {task.get('test_suggestion', task.get('tests', 'n/a'))}")
    else:
        sections.append("- none recorded")
    sections.extend(
        [
            "",
            "## 14. Safety Notes",
            "- No real broker execution.",
            "- Preserve DRY_RUN safety defaults.",
            "- Do not log secrets.",
            "- Add tests before changing execution behavior.",
            "- Fail closed if uncertain.",
        ]
    )
    return "\n".join(sections).strip() + "\n"


def write_codex_debug_report(
    diagnostics: RunDiagnostics | dict[str, Any],
    output_path: str | Path = "outputs/codex_daily_debug_report.md",
) -> Path:
    """Write the markdown debug report to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_codex_debug_report(diagnostics), encoding="utf-8")
    return path


def build_codex_next_prompt(diagnostics: RunDiagnostics | dict[str, Any]) -> str:
    """Build a copy/paste-ready prompt for a manual Codex debugging pass."""

    payload = diagnostics_to_dict(diagnostics)
    debug_report = build_codex_debug_report(diagnostics)
    affected_files = _collect_likely_files(payload)
    suggested_tests = _collect_test_suggestions(payload)
    lines = [
        "Analyze the following daily bot diagnostics and fix the root causes. Do not enable real broker trading. Preserve DRY_RUN defaults. Do not log secrets. Add or update tests. Run python -m py_compile *.py, python robustness_tests.py, python interface_tests.py and python health_check.py --quick. Log changes in outputs/codex_worklog.md.",
        "",
        debug_report.strip(),
        "",
        "## Likely Affected Files",
    ]
    if affected_files:
        lines.extend([f"- {file_name}" for file_name in affected_files])
    else:
        lines.append("- Review daily_bot.py, diagnostics.py, daily_analysis_report.py, codex_report.py and notifications.py.")
    lines.extend(["", "## Suggested Tests"])
    if suggested_tests:
        lines.extend([f"- {name}" for name in suggested_tests])
    else:
        lines.extend(
            [
                "- python -m py_compile *.py",
                "- python robustness_tests.py",
                "- python interface_tests.py",
                "- python health_check.py --quick",
            ]
        )
    lines.extend(
        [
            "",
            "## Acceptance Criteria",
            "- python -m py_compile *.py passes",
            "- python robustness_tests.py passes",
            "- python interface_tests.py passes",
            "- python health_check.py --quick passes",
            "- daily_bot.py writes all diagnostic reports",
            "- no final orders when final_action is HOLD/PAUSE",
            "- error path still writes codex_next_prompt.md",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def write_codex_next_prompt(
    diagnostics: RunDiagnostics | dict[str, Any],
    output_path: str | Path = "outputs/codex_next_prompt.md",
) -> Path:
    """Write the manual Codex next-step prompt to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_codex_next_prompt(diagnostics), encoding="utf-8")
    return path
