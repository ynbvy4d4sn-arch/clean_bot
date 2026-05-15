"""Human-readable daily analysis reporting and optional email delivery."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from config import get_email_gate_status
from codex_report import build_codex_next_prompt
from diagnostics import RunDiagnostics, diagnostics_to_dict
from notifications import load_email_settings, send_email_with_result, smtp_settings_complete


def _parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _status_from_diagnostics(payload: dict[str, Any]) -> str:
    if payload.get("errors"):
        return "ERROR"
    if str(payload.get("final_action", "")).upper() == "PAUSE":
        return "PAUSE"
    if str(payload.get("final_action", "")).upper() == "HOLD":
        return "HOLD"
    gate_status = str(payload.get("execution_gate_context", {}).get("gate_status", "")).upper()
    if payload.get("warnings") or gate_status == "BLOCK":
        return "WARN"
    order_count = int(payload.get("final_orders_summary", {}).get("order_count", 0) or 0)
    execution_mode = str(payload.get("execution_mode", "")).lower()
    if order_count > 0 and execution_mode not in {"blocked", "order_preview_only", "preview_only"}:
        return "TRADED"
    return "OK"


def is_trading_day_for_report(
    date: object,
    prices: pd.DataFrame | pd.Series | None = None,
    calendar: Any = None,
) -> bool:
    """Return whether the given local date should count as a trading day for reporting."""

    try:
        local_date = pd.Timestamp(date).normalize()
    except Exception:
        return False

    if calendar is not None:
        try:
            if callable(getattr(calendar, "is_trading_day", None)):
                return bool(calendar.is_trading_day(local_date))
            if isinstance(calendar, dict) and "is_trading_day" in calendar:
                value = calendar["is_trading_day"]
                return bool(value(local_date) if callable(value) else value)
        except Exception:
            pass

    if prices is not None and hasattr(prices, "index") and len(prices.index) > 0:
        try:
            available_days = {pd.Timestamp(item).normalize() for item in prices.index}
            if local_date in available_days:
                return True
        except Exception:
            pass

    return int(local_date.weekday()) < 5


def should_send_after_local_time(
    configured_time: str = "18:00",
    timezone: str = "Europe/Berlin",
    now: datetime | None = None,
) -> bool:
    """Check whether the current local time is after the configured report-send threshold."""

    try:
        hour_text, minute_text = str(configured_time).split(":", 1)
        cutoff_hour = int(hour_text)
        cutoff_minute = int(minute_text)
    except Exception:
        cutoff_hour = 18
        cutoff_minute = 0
    tz = ZoneInfo(str(timezone))
    local_now = (now or datetime.now(tz)).astimezone(tz)
    return (local_now.hour, local_now.minute) >= (cutoff_hour, cutoff_minute)


def load_daily_analysis_email_settings() -> dict[str, Any]:
    """Load daily-analysis-specific email settings on top of the existing SMTP configuration."""

    base_settings = load_email_settings()
    explicit_review_recipient = str(os.getenv("EMAIL_RECIPIENT", "")).strip()
    recipient = (
        explicit_review_recipient
        or str(os.getenv("DAILY_ANALYSIS_EMAIL_TO", "")).strip()
        or str(base_settings.get("EMAIL_TO", "")).strip()
    )
    settings = {
        **base_settings,
        "ENABLE_DAILY_ANALYSIS_EMAIL": _parse_bool(os.getenv("ENABLE_DAILY_ANALYSIS_EMAIL"), default=False),
        "DAILY_ANALYSIS_EMAIL_TIME_LOCAL": os.getenv("DAILY_ANALYSIS_EMAIL_TIME_LOCAL", "18:00"),
        "DAILY_ANALYSIS_TIMEZONE": os.getenv("DAILY_ANALYSIS_TIMEZONE", "Europe/Berlin"),
        "SEND_ANALYSIS_EMAIL_ONLY_ON_TRADING_DAYS": _parse_bool(
            os.getenv("SEND_ANALYSIS_EMAIL_ONLY_ON_TRADING_DAYS"),
            default=True,
        ),
        "SEND_ANALYSIS_EMAIL_ON_ERRORS_ONLY": _parse_bool(
            os.getenv("SEND_ANALYSIS_EMAIL_ON_ERRORS_ONLY"),
            default=False,
        ),
        "SEND_ANALYSIS_EMAIL_INCLUDE_CODEX_PROMPT": _parse_bool(
            os.getenv("SEND_ANALYSIS_EMAIL_INCLUDE_CODEX_PROMPT"),
            default=True,
        ),
        "DAILY_ANALYSIS_EMAIL_TO": recipient,
        "DAILY_ANALYSIS_EMAIL_SUBJECT_PREFIX": os.getenv(
            "DAILY_ANALYSIS_EMAIL_SUBJECT_PREFIX",
            "[Portfolio Bot Daily Analysis]",
        ),
        "EMAIL_DRY_RUN": _parse_bool(os.getenv("EMAIL_DRY_RUN"), default=True),
        "EMAIL_SEND_ENABLED": _parse_bool(os.getenv("EMAIL_SEND_ENABLED"), default=False),
        "EMAIL_RECIPIENT": explicit_review_recipient,
        "USER_CONFIRMED_EMAIL_PHASE": _parse_bool(os.getenv("USER_CONFIRMED_EMAIL_PHASE"), default=False),
        "PHASE": os.getenv("PHASE", "DAILY_REVIEW_PREVIEW").strip() or "DAILY_REVIEW_PREVIEW",
        "ENABLE_EXTERNAL_BROKER": _parse_bool(os.getenv("ENABLE_EXTERNAL_BROKER"), default=False),
        "ENABLE_INVESTOPEDIA_SIMULATOR": _parse_bool(os.getenv("ENABLE_INVESTOPEDIA_SIMULATOR"), default=False),
    }
    settings["EMAIL_TO"] = recipient
    return settings


def build_daily_analysis_report(diagnostics: RunDiagnostics | dict[str, Any]) -> str:
    """Build a human-readable daily report."""

    payload = diagnostics_to_dict(diagnostics)
    data_context = payload.get("data_context", {})
    candidate_context = payload.get("candidate_context", {})
    gate_context = payload.get("execution_gate_context", {})
    final_orders_summary = payload.get("final_orders_summary", {})
    rejected_orders = payload.get("rejected_orders", [])
    warnings = payload.get("warnings", [])
    errors = payload.get("errors", [])
    execution_mode = str(payload.get("execution_mode", "")).lower()
    traded = int(final_orders_summary.get("order_count", 0) or 0) > 0 and execution_mode not in {"blocked", "order_preview_only", "preview_only"}

    lines = [
        "# Daily Bot Analysis Report",
        "",
        "## Zusammenfassung",
        f"- Datum: {payload.get('local_date', 'n/a')}",
        f"- Final Action: {payload.get('final_action', 'n/a')}",
        f"- Selected Candidate: {payload.get('selected_candidate', 'n/a')}",
        f"- Gehandelt: {'ja' if traded else 'nein'}",
        f"- Wichtigste Begruendung: {payload.get('final_reason') or gate_context.get('reason') or 'n/a'}",
        "",
        "## Datenstatus",
        f"- Run Context: {data_context.get('run_context', 'daily_bot_discrete_simulator')}",
        f"- Quelle: {data_context.get('data_source', 'n/a')}",
        f"- Letzter Preisstand: {data_context.get('latest_price_date', 'n/a')}",
        f"- Erwarteter letzter Handelstag: {data_context.get('expected_latest_trading_day', 'n/a')}",
        f"- Daten frisch?: {data_context.get('data_freshness_ok', 'n/a')}",
        f"- Synthetische Daten?: {data_context.get('synthetic_data', 'n/a')}",
        f"- Cache-Fallback?: {data_context.get('used_cache_fallback', 'n/a')}",
        f"- Ticker geladen/fehlgeschlagen: {data_context.get('tickers_loaded', [])} / {data_context.get('tickers_failed', [])}",
        "",
        "## Entscheidung",
        f"- Warum gewaehlt: {candidate_context.get('reason') or payload.get('final_reason') or gate_context.get('reason') or 'n/a'}",
        f"- Warum HOLD/PAUSE, falls nicht gehandelt: {payload.get('final_reason') if not traded else 'n/a'}",
        f"- Warum nicht HOLD, falls gehandelt: {candidate_context.get('delta_vs_hold', 'n/a') if traded else 'n/a'}",
        "",
        "## Risiko und Kosten",
        f"- trade_now_score: {gate_context.get('trade_now_score', 'n/a')}",
        f"- estimated costs: {final_orders_summary.get('estimated_cost', final_orders_summary.get('estimated_cost_pct_nav', 'n/a'))}",
        f"- spread/slippage: {gate_context.get('spread_cost', 'n/a')} / {gate_context.get('slippage', 'n/a')}",
        f"- worst scenario: {candidate_context.get('worst_scenario', 'n/a')}",
        f"- CVaR / tail risk: {candidate_context.get('cvar_5', 'n/a')}",
        "",
        "## Orders",
    ]
    if traded:
        lines.append(f"- Finale Orders: {int(final_orders_summary.get('order_count', 0) or 0)}")
    else:
        lines.append("- Finale Orders: keine Orders")
    if rejected_orders:
        lines.extend([f"- Rejected {item.get('ticker')} {item.get('side')}: {item.get('reason')}" for item in rejected_orders])
    else:
        lines.append("- Rejected Orders: keine")
    lines.extend(["", "## Fehler und Warnungen"])
    if errors:
        lines.extend([f"- Kritischer Fehler in {item.get('module')}: {item.get('exception_message')}" for item in errors[:5]])
    else:
        lines.append("- Kritische Fehler: keine")
    if warnings:
        lines.extend([f"- Warnung in {item.get('module')}: {item.get('message')}" for item in warnings[:5]])
    else:
        lines.append("- Warnungen: keine")
    next_checks = []
    if errors:
        next_checks.append("Fehlerpfad im Codex Debug Report pruefen.")
    if warnings:
        next_checks.append("Warnungen und Gate-Blockgruende gegen die Daten- und Kostenannahmen pruefen.")
    if not next_checks:
        next_checks.append("Keine akuten Auffaelligkeiten; Reports fuer Drift und wiederholte HOLD/PAUSE-Lagen beobachten.")
    lines.extend([f"- Naechste Pruefung: {item}" for item in next_checks])
    lines.extend(
        [
            "",
            "## Codex-Hinweise",
            "- Pfad zum Codex Debug Report: outputs/codex_daily_debug_report.md",
            "- Pfad zum Codex Next Prompt: outputs/codex_next_prompt.md",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def write_daily_analysis_report(
    diagnostics: RunDiagnostics | dict[str, Any],
    output_path: str | Path = "outputs/daily_analysis_report.md",
) -> Path:
    """Write the human-readable daily report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_daily_analysis_report(diagnostics), encoding="utf-8")
    return path


def build_daily_analysis_email_subject(
    diagnostics: RunDiagnostics | dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> str:
    """Build the daily analysis subject line."""

    payload = diagnostics_to_dict(diagnostics)
    settings = settings or load_daily_analysis_email_settings()
    prefix = str(settings.get("DAILY_ANALYSIS_EMAIL_SUBJECT_PREFIX", "[Portfolio Bot Daily Analysis]")).strip()
    return f"{prefix} - {payload.get('local_date', 'n/a')} - {payload.get('final_action', 'n/a')} - {_status_from_diagnostics(payload)}"


def build_daily_analysis_email_body(
    diagnostics: RunDiagnostics | dict[str, Any],
    include_codex_prompt: bool = True,
) -> str:
    """Build the email body with a short summary and optional Codex follow-up prompt."""

    payload = diagnostics_to_dict(diagnostics)
    report = build_daily_analysis_report(diagnostics).strip()
    errors = payload.get("errors", [])
    warnings = payload.get("warnings", [])
    lines = [
        f"Daily analysis summary for {payload.get('local_date', 'n/a')}",
        f"Final action: {payload.get('final_action', 'n/a')}",
        f"Selected candidate: {payload.get('selected_candidate', 'n/a')}",
        f"Status: {_status_from_diagnostics(payload)}",
        "",
        "Top issues:",
    ]
    if errors:
        lines.extend([f"- ERROR {item.get('module')}: {item.get('exception_message')}" for item in errors[:3]])
    elif warnings:
        lines.extend([f"- WARN {item.get('module')}: {item.get('message')}" for item in warnings[:3]])
    else:
        lines.append("- No critical errors or warnings recorded.")
    lines.extend(
        [
            "",
            "Local reports:",
            "- outputs/run_diagnostics.json",
            "- outputs/codex_daily_debug_report.md",
            "- outputs/codex_next_prompt.md",
            "- outputs/daily_analysis_report.md",
            "",
            report,
        ]
    )
    if include_codex_prompt:
        lines.extend(["", "---", "", build_codex_next_prompt(diagnostics).strip()])
    return "\n".join(lines).strip() + "\n"


def _should_send_with_reason(
    diagnostics: RunDiagnostics | dict[str, Any],
    settings: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> tuple[bool, str]:
    payload = diagnostics_to_dict(diagnostics)
    settings = settings or load_daily_analysis_email_settings()
    if not bool(settings.get("ENV_FILE_PRESENT", False)):
        return False, "env_missing"
    if not bool(settings.get("ENABLE_EMAIL_NOTIFICATIONS", False)):
        return False, "email_notifications_disabled"
    if not bool(settings.get("ENABLE_DAILY_ANALYSIS_EMAIL", False)):
        return False, "daily_analysis_email_disabled"
    if not smtp_settings_complete(settings):
        return False, "smtp_incomplete"
    if not str(settings.get("EMAIL_TO", "")).strip():
        return False, "recipient_missing"
    gate_status = get_email_gate_status(settings)
    if not bool(gate_status.get("real_email_send_allowed", False)):
        return False, str(gate_status.get("reason", "preview_only"))

    timezone_name = str(settings.get("DAILY_ANALYSIS_TIMEZONE", "Europe/Berlin"))
    local_now = (now or datetime.now(ZoneInfo(timezone_name))).astimezone(ZoneInfo(timezone_name))
    only_on_trading_days = bool(settings.get("SEND_ANALYSIS_EMAIL_ONLY_ON_TRADING_DAYS", True))
    on_errors_only = bool(settings.get("SEND_ANALYSIS_EMAIL_ON_ERRORS_ONLY", False))
    has_critical_issue = bool(payload.get("errors")) or any(
        str(item.get("severity", "")).upper() in {"ERROR", "CRITICAL"} for item in payload.get("warnings", [])
    )

    if only_on_trading_days and not is_trading_day_for_report(local_now.date()):
        return False, "non_trading_day"

    after_cutoff = should_send_after_local_time(
        configured_time=str(settings.get("DAILY_ANALYSIS_EMAIL_TIME_LOCAL", "18:00")),
        timezone=timezone_name,
        now=local_now,
    )
    if not after_cutoff:
        if on_errors_only and has_critical_issue:
            return True, "critical_error_before_cutoff"
        return False, "before_cutoff"

    if on_errors_only and not has_critical_issue:
        return False, "no_critical_errors"
    return True, "eligible"


def should_send_daily_analysis_email(
    diagnostics: RunDiagnostics | dict[str, Any],
    settings: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> bool:
    """Return whether the daily-analysis email should be sent."""

    allowed, _reason = _should_send_with_reason(diagnostics, settings=settings, now=now)
    return allowed


def send_daily_analysis_email_if_needed(
    diagnostics: RunDiagnostics | dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attempt to send the daily analysis email without crashing the caller."""

    settings = settings or load_daily_analysis_email_settings()
    allowed, reason = _should_send_with_reason(diagnostics, settings=settings)
    if not allowed:
        result = {"sent": False, "reason": reason, "error": None}
        if reason in {"preview_only", "blocked_by_gate"}:
            gate_status = get_email_gate_status(settings)
            result["blocked_reasons"] = list(gate_status.get("blockers", []))
        return result

    include_codex_prompt = bool(settings.get("SEND_ANALYSIS_EMAIL_INCLUDE_CODEX_PROMPT", True))
    subject = build_daily_analysis_email_subject(diagnostics, settings=settings)
    body = build_daily_analysis_email_body(diagnostics, include_codex_prompt=include_codex_prompt)
    result = send_email_with_result(subject=subject, body=body, settings=settings)
    result.setdefault("reason", "sent" if result.get("sent") else "smtp_failed")
    return result
