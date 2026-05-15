"""Daily portfolio review and email-preview helpers for the daily bot."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from config import get_email_gate_status, review_email_send_allowed as config_review_email_send_allowed
from daily_review_rendering import build_daily_review_render_bundle
from diagnostics import RunDiagnostics, diagnostics_to_dict
from hold_analysis import build_hold_analysis_bundle
from notifications import load_email_settings, sanitize_for_output, send_email_notification


BERLIN_TZ = ZoneInfo("Europe/Berlin")


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


def _parse_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _safe_bool(value: object) -> bool:
    return _parse_bool(value, default=False)


def _timestamp_text(value: object) -> str:
    if value in (None, ""):
        return "n/a"
    try:
        return str(pd.Timestamp(value))
    except Exception:
        return str(value)


def _bool_text(value: object) -> str:
    return "true" if bool(value) else "false"


def _first_non_empty(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _extract_report_value(path: Path, *keys: str) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""
    normalized_keys = [str(key).strip().lower() for key in keys if str(key).strip()]
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().lower() in normalized_keys:
            return value.strip()
    return ""


def _collect_report_consistency(review: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    expected_latest_price_date = str(review.get("data_status", {}).get("latest_price_date", "") or "").strip()
    if output_dir is None:
        return {
            "expected_latest_price_date": expected_latest_price_date,
            "latest_price_date_mismatch": False,
            "latest_price_dates_by_report": {},
        }

    output_path = Path(output_dir)
    report_paths = {
        "current_data_freshness_report.txt": output_path / "current_data_freshness_report.txt",
        "daily_bot_decision_report.txt": output_path / "daily_bot_decision_report.txt",
        "latest_decision_report.txt": output_path / "latest_decision_report.txt",
    }
    latest_price_dates_by_report: dict[str, str] = {}
    for report_name, report_path in report_paths.items():
        value = _extract_report_value(report_path, "latest_price_date", "Latest Price Date")
        if value:
            latest_price_dates_by_report[report_name] = value

    latest_price_date_mismatch = False
    if expected_latest_price_date:
        latest_price_date_mismatch = any(
            value != expected_latest_price_date for value in latest_price_dates_by_report.values()
        )
    elif latest_price_dates_by_report:
        latest_price_date_mismatch = len(set(latest_price_dates_by_report.values())) > 1

    return {
        "expected_latest_price_date": expected_latest_price_date,
        "latest_price_date_mismatch": bool(latest_price_date_mismatch),
        "latest_price_dates_by_report": latest_price_dates_by_report,
    }


def load_daily_review_settings() -> dict[str, Any]:
    """Load safe review/email-preview settings."""

    email_settings = load_email_settings()
    return {
        "phase": os.getenv("PHASE", "DAILY_REVIEW_PREVIEW").strip() or "DAILY_REVIEW_PREVIEW",
        "enable_email_notifications": _parse_bool(os.getenv("ENABLE_EMAIL_NOTIFICATIONS"), default=False),
        "email_dry_run": _parse_bool(os.getenv("EMAIL_DRY_RUN"), default=True),
        "email_send_enabled": _parse_bool(os.getenv("EMAIL_SEND_ENABLED"), default=False),
        "daily_briefing_only": _parse_bool(os.getenv("DAILY_BRIEFING_ONLY"), default=True),
        "max_emails_per_day": _parse_int(os.getenv("MAX_EMAILS_PER_DAY"), default=1),
        "email_recipient": str(os.getenv("EMAIL_RECIPIENT", "")).strip(),
        "user_confirmed_email_phase": _parse_bool(os.getenv("USER_CONFIRMED_EMAIL_PHASE"), default=False),
        "enable_external_broker": _parse_bool(os.getenv("ENABLE_EXTERNAL_BROKER"), default=False),
        "enable_investopedia_simulator": _parse_bool(os.getenv("ENABLE_INVESTOPEDIA_SIMULATOR"), default=False),
        "enable_local_paper_trading": _parse_bool(os.getenv("ENABLE_LOCAL_PAPER_TRADING"), default=False),
        "dry_run": _parse_bool(os.getenv("DRY_RUN"), default=True),
        "email_provider": str(email_settings.get("EMAIL_PROVIDER", "brevo")),
        "smtp_host": str(email_settings.get("SMTP_HOST", "")),
        "smtp_port": _parse_int(email_settings.get("SMTP_PORT"), default=587),
        "email_sender_present": bool(str(email_settings.get("EMAIL_FROM", "")).strip()),
    }


def review_email_send_allowed(settings: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return whether a real email send is allowed for the daily review stage."""

    return config_review_email_send_allowed(settings)


def _review_status_from_payload(review: dict[str, Any], issues: dict[str, Any]) -> str:
    if int(issues.get("hard_fail_count", 0)) > 0:
        return "BLOCK"
    final_action = str(review.get("run_status", {}).get("final_action", "")).upper()
    if final_action in {"WAIT", "WAIT_OUTSIDE_WINDOW", "WAIT_MARKET_CLOSED", "PAUSE", "HOLD"}:
        return "WAIT"
    delta_transactions = list(review.get("delta_transactions", []))
    return "REVIEW" if delta_transactions else "WAIT"


def _build_issue_table(review: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    hard_fails: list[dict[str, str]] = []
    soft_warnings: list[dict[str, str]] = []
    infos: list[dict[str, str]] = []

    data_status = review.get("data_status", {})
    current_portfolio = review.get("current_portfolio", {})
    cost_edge = review.get("cost_edge", {})
    order_summary = review.get("order_summary", {})
    run_status = review.get("run_status", {})
    delta_transactions = list(review.get("delta_transactions", []))
    positions = list(review.get("current_positions", []))
    validation_status = str(review.get("pre_trade_validation_status", "PASS")).upper()
    exception_message = str(review.get("exception_message", "")).strip()
    price_basis = str(data_status.get("price_basis", "")).strip()
    missing_prices = list(data_status.get("missing_prices", []) or [])
    parser_errors = list(current_portfolio.get("parser_errors", []) or [])
    parser_warnings = list(current_portfolio.get("parser_warnings", []) or [])
    low_history_assets = [str(item) for item in list(data_status.get("low_history_assets", []) or []) if str(item).strip()]
    report_consistency = _collect_report_consistency(review, output_dir=output_dir)
    all_blockers: list[str] = []
    weights_sum_including_cash = _safe_float(current_portfolio.get("current_weights_sum_including_cash", 0.0))
    negative_market_values = [
        str(row.get("ticker"))
        for row in positions
        if _safe_float(row.get("market_value_usd", 0.0)) < -1e-9
    ]
    cash_usd_value = current_portfolio.get("cash_usd", 0.0)
    cash_is_nan = False
    try:
        cash_is_nan = pd.isna(float(cash_usd_value))
    except (TypeError, ValueError):
        cash_is_nan = True

    def add_issue(target: list[dict[str, str]], severity: str, code: str, message: str) -> None:
        target.append({"severity": severity, "code": code, "message": message})
        all_blockers.append(message)

    if _safe_bool(data_status.get("synthetic_data", False)):
        add_issue(hard_fails, "HARD_FAIL", "synthetic_data", "synthetic_data=true")
    if not _safe_bool(data_status.get("data_freshness_ok", False)):
        add_issue(hard_fails, "HARD_FAIL", "data_freshness", "data_freshness_ok=false")
    if parser_errors:
        add_issue(hard_fails, "HARD_FAIL", "current_portfolio_invalid", "current_portfolio invalid: " + "; ".join(parser_errors))
    if _safe_float(current_portfolio.get("nav_usd", 0.0)) <= 0.0:
        add_issue(hard_fails, "HARD_FAIL", "nav_non_positive", "NAV <= 0")
    if cash_is_nan:
        add_issue(hard_fails, "HARD_FAIL", "cash_nan", "cash_usd is NaN or invalid")
    if missing_prices:
        add_issue(hard_fails, "HARD_FAIL", "missing_latest_price", "missing latest price for current holding: " + ", ".join(map(str, missing_prices)))
    negative_shares = [row["ticker"] for row in positions if _safe_float(row.get("current_shares", 0.0)) < 0.0]
    if negative_shares:
        add_issue(hard_fails, "HARD_FAIL", "negative_shares", "negative shares detected: " + ", ".join(negative_shares))
    if negative_market_values:
        add_issue(hard_fails, "HARD_FAIL", "negative_market_value", "negative market value detected: " + ", ".join(negative_market_values))
    if _safe_float(current_portfolio.get("nav_usd", 0.0)) > 0.0 and abs(weights_sum_including_cash - 1.0) > 0.01:
        add_issue(hard_fails, "HARD_FAIL", "weights_sum_mismatch", f"current_weights_sum_including_cash not close to 1.0: {weights_sum_including_cash:.6f}")
    if int(current_portfolio.get("positions_count", 0) or 0) > 0 and _safe_bool(current_portfolio.get("current_portfolio_100pct_cash", False)):
        add_issue(hard_fails, "HARD_FAIL", "portfolio_cash_misclassified", "current_portfolio_100pct_cash=true despite existing positions")
    if _safe_float(review.get("cash_after_orders", current_portfolio.get("cash_usd", 0.0))) < -1e-9 or not _safe_bool(order_summary.get("negative_cash_check", True)):
        add_issue(hard_fails, "HARD_FAIL", "negative_projected_cash", "negative projected cash")
    if validation_status != "PASS":
        add_issue(hard_fails, "HARD_FAIL", "pre_trade_validation_failed", "pre_trade_validation failed")
    if exception_message:
        add_issue(hard_fails, "HARD_FAIL", "exception_during_run", f"exception during daily_bot run: {exception_message}")

    if _safe_bool(data_status.get("used_cache_fallback", False)):
        add_issue(soft_warnings, "SOFT_WARNING", "cache_fallback", "cache_fallback used")
    if not _safe_bool(run_status.get("within_allowed_window", False)):
        add_issue(soft_warnings, "SOFT_WARNING", "outside_allowed_window", "outside allowed trading window")
    if _safe_float(cost_edge.get("trade_now_edge", 0.0)) < 0.0:
        add_issue(soft_warnings, "SOFT_WARNING", "negative_trade_edge", "trade_now_edge negative")
    if _safe_float(cost_edge.get("modeled_transaction_costs_pct_nav", 0.0)) >= 0.0020:
        add_issue(soft_warnings, "SOFT_WARNING", "modeled_costs_high", "modeled transaction costs high")
    if bool(report_consistency.get("latest_price_date_mismatch", False)):
        details = ", ".join(
            f"{name}={value}" for name, value in sorted(report_consistency.get("latest_price_dates_by_report", {}).items())
        ) or "inconsistent report snapshots"
        add_issue(soft_warnings, "SOFT_WARNING", "latest_price_date_mismatch", f"latest_price_date mismatch between reports: {details}")
    if parser_warnings:
        for warning in parser_warnings:
            add_issue(soft_warnings, "SOFT_WARNING", "current_portfolio_warning", warning)
    if any(str(row.get("ticker")) in {"SH", "PSQ"} for row in positions):
        add_issue(soft_warnings, "SOFT_WARNING", "inverse_exposure", "short ETF / inverse ETF exposure exists")
    if any(str(row.get("ticker")) in {"IBIT", "ETHA"} for row in positions):
        add_issue(soft_warnings, "SOFT_WARNING", "crypto_exposure", "crypto ETF exposure exists")
    if price_basis == "adjusted_close_proxy":
        add_issue(soft_warnings, "SOFT_WARNING", "adjusted_close_proxy", "adjusted_close_proxy used")
    if low_history_assets:
        add_issue(soft_warnings, "SOFT_WARNING", "low_history_asset", "low history asset: " + ", ".join(low_history_assets[:8]))
    if review.get("main_daily_scope_differs"):
        add_issue(soft_warnings, "SOFT_WARNING", "scope_differs", "main.py and daily_bot.py scope differs")

    add_issue(infos, "INFO", "dry_run_active", "dry_run active")
    add_issue(infos, "INFO", "no_real_orders", "no real orders sent")
    if abs(_safe_float(cost_edge.get("total_simulator_fees_usd", 0.0))) < 1e-12:
        add_issue(infos, "INFO", "simulator_fees_zero", "simulator fees 0.00")
    if delta_transactions:
        add_issue(infos, "INFO", "manual_orders_generated", "manual simulator orders generated")
    else:
        add_issue(infos, "INFO", "no_manual_orders", "no manual orders generated")

    first_blocker = ""
    if hard_fails:
        first_blocker = hard_fails[0]["message"]
    elif soft_warnings:
        first_blocker = soft_warnings[0]["message"]
    else:
        first_blocker = _first_non_empty(run_status.get("gate_reason"), run_status.get("first_blocker"), "none")

    issue_table = hard_fails + soft_warnings + infos
    issues = {
        "hard_fails": hard_fails,
        "soft_warnings": soft_warnings,
        "infos": infos,
        "hard_fail_count": len(hard_fails),
        "soft_warning_count": len(soft_warnings),
        "info_count": len(infos),
        "first_blocker": first_blocker or "none",
        "all_blockers": all_blockers or ["none"],
        "issue_table": issue_table,
        "report_consistency": report_consistency,
    }
    issues["review_status"] = _review_status_from_payload(review, issues)
    return issues


def build_review_issues(review: dict[str, Any], output_dir: str | Path = "outputs") -> dict[str, Any]:
    return _build_issue_table(review, output_dir=output_dir)


def _build_manual_next_action(review: dict[str, Any], issues: dict[str, Any]) -> list[str]:
    delta_transactions = list(review.get("delta_transactions", []))
    order_summary = review.get("order_summary", {})
    manual_orders_ready = bool(order_summary.get("manual_orders_usable", False)) and int(issues.get("hard_fail_count", 0)) == 0
    lines = [
        "Fuer Simulatororders verwenden: outputs/manual_simulator_orders.csv",
        "Nicht verwenden: outputs/order_preview.csv",
    ]
    if int(issues.get("hard_fail_count", 0)) > 0:
        lines.append("Nicht manuell im Simulator handeln. Erst den ersten Hard-Fail beheben.")
    elif not delta_transactions:
        if _safe_bool(review.get("preview_only", True)):
            lines.append("Preview only. Heute nicht manuell handeln. Beobachten ist empfohlen, weil keine BUY/SELL-Delta-Orders vorliegen.")
        else:
            lines.append("Heute nicht manuell handeln. Beobachten ist empfohlen, weil keine BUY/SELL-Delta-Orders vorliegen.")
    elif not bool(order_summary.get("manual_orders_usable", False)):
        lines.append("Preview only. Die Delta-Orders sind sichtbar, aber die manuelle Simulatorliste ist wegen des aktuellen Blockers nicht verwendbar.")
    elif _safe_bool(review.get("preview_only", True)):
        lines.append("Preview only. Die Delta-Orders sind sichtbar und koennen fuer manuelle Simulator-Eingabe verwendet werden, aber es wurden keine echten Orders gesendet.")
    elif manual_orders_ready:
        lines.append("Manuelle Pruefung empfohlen. Danach nur die Delta-Orders aus outputs/manual_simulator_orders.csv verwenden.")
    else:
        lines.append("Keine manuelle Aktion empfohlen.")
    lines.append("Keine echten Orders wurden gesendet.")
    return lines


def _manual_order_count(review: dict[str, Any]) -> int:
    order_summary = review.get("order_summary", {})
    return int(
        order_summary.get(
            "manual_eligible_order_count",
            order_summary.get("order_count", 0),
        )
        or 0
    )


def _build_operator_instruction(review: dict[str, Any], issues: dict[str, Any]) -> str:
    run_status = review.get("run_status", {})
    data_status = review.get("data_status", {})
    final_action = str(run_status.get("final_action", "") or "").upper()
    manual_order_count = _manual_order_count(review)

    instructions: list[str] = []
    if _safe_bool(data_status.get("synthetic_data", False)):
        instructions.append("Blockiert: synthetische Daten; keine Orders.")
    elif not _safe_bool(data_status.get("data_freshness_ok", False)):
        instructions.append("Blockiert: Daten nicht frisch.")
    elif "WAIT" in final_action or "BLOCK" in final_action or int(issues.get("hard_fail_count", 0)) > 0:
        instructions.append("Keine Orders eingeben.")
    elif final_action == "HOLD":
        instructions.append("Heute keine Orders eingeben. Beste Aktion laut Bot: HOLD.")

    if manual_order_count == 0:
        instructions.append("Keine Simulator-Orders eingeben.")
    if _safe_bool(data_status.get("used_cache_fallback", False)):
        instructions.append("Warnung: Live-Daten nicht genutzt; Bericht nur vorsichtig verwenden.")

    if not instructions:
        if _safe_bool(review.get("preview_only", True)):
            instructions.append("Preview only. Nur die freigegebenen Delta-Orders aus outputs/manual_simulator_orders.csv manuell pruefen.")
        else:
            instructions.append("Nur die freigegebenen Delta-Orders aus outputs/manual_simulator_orders.csv verwenden.")
    return " ".join(dict.fromkeys(instructions))


def _stable_json_value(value: Any) -> Any:
    """Return a deterministic JSON-safe representation for fingerprints."""

    if isinstance(value, dict):
        return {str(key): _stable_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return round(float(value), 10)
    if isinstance(value, (int, bool)) or value is None:
        return value
    return str(value)


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(_stable_json_value(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _email_recipient_from_settings(settings: dict[str, Any] | None = None) -> str:
    effective_settings = {**load_email_settings(), **dict(settings or {})}
    return str(
        effective_settings.get("email_recipient")
        or effective_settings.get("EMAIL_RECIPIENT")
        or effective_settings.get("EMAIL_TO")
        or ""
    ).strip()


def _selected_candidate(review: dict[str, Any]) -> str:
    decision_context = review.get("decision_context", {})
    return str(
        decision_context.get("final_discrete_candidate")
        or decision_context.get("selected_candidate")
        or review.get("run_status", {}).get("selected_candidate")
        or "n/a"
    )


def _selected_reason(review: dict[str, Any]) -> str:
    decision_context = review.get("decision_context", {})
    return str(
        decision_context.get("selected_reason")
        or review.get("run_status", {}).get("selected_reason")
        or decision_context.get("trade_decision_reason")
        or "n/a"
    )


def _order_hash_from_rows(rows: list[dict[str, Any]]) -> str:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_rows.append(
            {
                "ticker": str(row.get("ticker", "")).strip(),
                "action": str(row.get("action", row.get("side", ""))).strip().upper(),
                "order_shares": _safe_float(row.get("order_shares", row.get("quantity", 0.0))),
                "estimated_price": _safe_float(row.get("estimated_price", row.get("price", 0.0))),
                "estimated_order_value": _safe_float(row.get("estimated_order_value", row.get("notional", 0.0))),
            }
        )
    normalized_rows = sorted(normalized_rows, key=lambda item: (item["ticker"], item["action"], item["order_shares"]))
    return _hash_payload(normalized_rows)


def _active_preview_order_hash(review: dict[str, Any]) -> str:
    active_preview = dict(review.get("active_preview", {}) or {})
    explicit_orders = list(active_preview.get("orders", []) or review.get("active_preview_orders", []) or [])
    if explicit_orders:
        return _order_hash_from_rows([dict(row) for row in explicit_orders])
    return _hash_payload(
        {
            "candidate": active_preview.get("active_preview_candidate", "HOLD_CURRENT"),
            "order_count": int(active_preview.get("active_preview_order_count", 0) or 0),
            "buy_count": int(active_preview.get("active_preview_buy_count", 0) or 0),
            "sell_count": int(active_preview.get("active_preview_sell_count", 0) or 0),
            "turnover": _safe_float(active_preview.get("active_preview_turnover", 0.0)),
        }
    )


def _portfolio_snapshot_hash(review: dict[str, Any]) -> str:
    current_portfolio = review.get("current_portfolio", {})
    positions = [
        {
            "ticker": str(row.get("ticker", "")).strip(),
            "shares": _safe_float(row.get("current_shares", 0.0)),
            "weight": _safe_float(row.get("current_weight", 0.0)),
        }
        for row in list(review.get("current_positions", []) or [])
    ]
    return _hash_payload(
        {
            "nav_usd": _safe_float(current_portfolio.get("nav_usd", 0.0)),
            "cash_usd": _safe_float(current_portfolio.get("cash_usd", 0.0)),
            "positions": sorted(positions, key=lambda item: item["ticker"]),
        }
    )


def _effective_horizon_days(review: dict[str, Any]) -> str:
    for container_name in ("horizon", "run_status", "data_status", "decision_context"):
        container = review.get(container_name, {})
        if isinstance(container, dict) and container.get("effective_horizon_days") not in (None, ""):
            return str(container.get("effective_horizon_days"))
    return "n/a"


def build_decision_fingerprint(
    review: dict[str, Any],
    *,
    settings: dict[str, Any] | None = None,
    recipient: str | None = None,
) -> str:
    """Build a stable same-decision fingerprint that excludes volatile body/runtime fields."""

    active_preview = dict(review.get("active_preview", {}) or {})
    run_status = review.get("run_status", {})
    data_status = review.get("data_status", {})
    recipient_text = str(recipient if recipient is not None else _email_recipient_from_settings(settings)).strip()
    payload = {
        "review_date": str(run_status.get("review_date", "")),
        "recipient": recipient_text,
        "safe_final_action": str(run_status.get("final_action", "")),
        "safe_selected_candidate": _selected_candidate(review),
        "safe_selected_reason": _selected_reason(review),
        "active_preview_action": str(active_preview.get("active_preview_action", "HOLD")),
        "active_preview_candidate": str(active_preview.get("active_preview_candidate", "HOLD_CURRENT")),
        "safe_order_list_hash": _order_hash_from_rows([dict(row) for row in list(review.get("delta_transactions", []) or [])]),
        "active_preview_order_list_hash": _active_preview_order_hash(review),
        "first_blocker": str(run_status.get("first_blocker") or review.get("first_blocker") or ""),
        "latest_price_date": str(data_status.get("latest_price_date", "")),
        "current_portfolio_hash": _portfolio_snapshot_hash(review),
        "effective_horizon_days": _effective_horizon_days(review),
    }
    return _hash_payload(payload)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    safe_headers = [str(header) for header in headers]
    lines = ["| " + " | ".join(safe_headers) + " |", "| " + " | ".join(["---"] * len(safe_headers)) + " |"]
    for row in rows:
        cells = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _summarize_orders_for_briefing(rows: list[dict[str, Any]], *, max_rows: int = 8) -> list[list[str]]:
    summarized: list[list[str]] = []
    for row in rows[:max_rows]:
        summarized.append(
            [
                str(row.get("action", row.get("side", "n/a"))),
                str(row.get("ticker", "n/a")),
                f"{_safe_float(row.get('order_shares', row.get('quantity', 0.0))):.4f}",
                f"{_safe_float(row.get('estimated_order_value', row.get('notional', 0.0))):.2f}",
                str(row.get("execution_block_reason", row.get("note", "")) or "none"),
            ]
        )
    return summarized


def _active_preview_summary_sentence(review: dict[str, Any]) -> str:
    active_preview = dict(review.get("active_preview", {}) or {})
    order_count = int(active_preview.get("active_preview_order_count", 0) or 0)
    safe_action = str(review.get("run_status", {}).get("final_action", "n/a"))
    active_action = str(active_preview.get("active_preview_action", "HOLD"))
    if order_count > 0 and safe_action.upper() in {"HOLD", "WAIT", "WAIT_OUTSIDE_WINDOW", "BLOCK"}:
        return "Safe Mode bleibt konservativ, aber Active Preview findet eine nicht-ausfuehrbare Lower-Hurdle-Alternative."
    if order_count > 0:
        return "Active Preview zeigt Analyse-Orders; sie sind strikt nicht ausfuehrbar."
    return f"Active Preview bleibt ebenfalls bei {active_action}; es gibt keine Analyse-Orders."


def build_daily_portfolio_briefing_markdown(
    review: dict[str, Any],
    issues: dict[str, Any],
    *,
    settings: dict[str, Any] | None = None,
    email_result: dict[str, Any] | None = None,
    email_state: dict[str, Any] | None = None,
) -> str:
    """Build the human-oriented Daily Portfolio Briefing as Markdown."""

    effective_settings = {**load_daily_review_settings(), **dict(settings or {})}
    email_result = dict(email_result or {})
    email_state = dict(email_state or {})
    run_status = review.get("run_status", {})
    data_status = review.get("data_status", {})
    current_portfolio = review.get("current_portfolio", {})
    decision_context = review.get("decision_context", {})
    cost_edge = review.get("cost_edge", {})
    active_preview = dict(review.get("active_preview", {}) or {})
    hold_analysis = dict(review.get("hold_analysis", {}) or {})
    current_positions = list(review.get("current_positions", []) or [])
    delta_transactions = list(review.get("delta_transactions", []) or [])
    active_orders = list(active_preview.get("orders", []) or review.get("active_preview_orders", []) or [])
    active_order_count = int(active_preview.get("active_preview_order_count", len(active_orders)) or 0)
    manual_order_count = _manual_order_count(review)
    provider_accepted = _safe_bool(email_result.get("provider_accepted", False))
    delivery_confirmed = _safe_bool(email_result.get("delivery_confirmed", False))
    real_send_allowed = bool(get_email_gate_status(effective_settings).get("real_email_send_allowed", False))
    generated_at = datetime.now(BERLIN_TZ).isoformat(timespec="seconds")
    safe_action = str(run_status.get("final_action", "n/a"))
    active_action = str(active_preview.get("active_preview_action", "HOLD"))
    selected_reason = _selected_reason(review)
    first_blocker = str(issues.get("first_blocker", "none"))
    constraint_valid = current_portfolio.get("current_portfolio_constraint_valid", "n/a")
    executive_summary = (
        f"Safe Mode entscheidet aktuell `{safe_action}` fuer `{_selected_candidate(review)}`. "
        f"Grund: {selected_reason}. {_active_preview_summary_sentence(review)} "
        f"Safe-Orderanzahl: {manual_order_count}. Wichtigster Blocker: {first_blocker}."
    )

    lines: list[str] = [
        "# Daily Portfolio Briefing",
        "",
        "## Header",
        f"- Date: {run_status.get('review_date', 'n/a')}",
        f"- Generated at: {generated_at}",
        f"- Mode: {'DRY_RUN / PREVIEW ONLY' if _safe_bool(review.get('preview_only', True)) else 'LIVE MODE CHECK REQUIRED'}",
        f"- Safe Action: {safe_action}",
        f"- Active Preview Action: {active_action}",
        f"- Data Status: data_freshness_ok={_bool_text(data_status.get('data_freshness_ok', False))}, synthetic_data={_bool_text(data_status.get('synthetic_data', False))}",
        f"- Mail Status: provider_accepted={_bool_text(provider_accepted)}, delivery_confirmed={_bool_text(delivery_confirmed)}",
        f"- Execution Safety Status: DRY_RUN={_bool_text(effective_settings.get('dry_run', True))}, real_email_send_allowed={_bool_text(real_send_allowed)}",
        "",
        "## Executive Summary",
        executive_summary,
        "",
        "## Decision Card",
        *_markdown_table(
            [
                "Safe final_action",
                "Safe selected_candidate",
                "Safe selected_reason",
                "Active preview action",
                "Active preview candidate",
                "order_count",
                "buy_count",
                "sell_count",
                "first_blocker",
            ],
            [
                [
                    safe_action,
                    _selected_candidate(review),
                    selected_reason,
                    active_action,
                    active_preview.get("active_preview_candidate", "HOLD_CURRENT"),
                    manual_order_count,
                    int(review.get("order_summary", {}).get("buy_count", 0) or 0),
                    int(review.get("order_summary", {}).get("sell_count", 0) or 0),
                    first_blocker,
                ]
            ],
        ),
        "",
        "## Portfolio Snapshot",
        f"- NAV: {_safe_float(current_portfolio.get('nav_usd', 0.0)):.2f} USD",
        f"- Cash: {_safe_float(current_portfolio.get('cash_usd', 0.0)):.2f} USD",
        f"- positions_count: {current_portfolio.get('positions_count', 0)}",
        f"- current portfolio 100pct cash: {_bool_text(current_portfolio.get('current_portfolio_100pct_cash', False))}",
        f"- current_portfolio_constraint_valid: {constraint_valid}",
        "- main current exposures:",
    ]
    if current_positions:
        for row in current_positions[:8]:
            lines.append(f"- {row.get('ticker', 'n/a')}: weight={_safe_float(row.get('current_weight', 0.0)):.2%}, shares={_safe_float(row.get('current_shares', 0.0)):.4f}")
    else:
        lines.append("- none")

    lines.extend(["", "## Safe Orders"])
    if not delta_transactions or manual_order_count == 0:
        lines.append("No safe executable/manual delta orders.")
    else:
        lines.extend(_markdown_table(["Action", "Ticker", "Shares", "Value USD", "Block reason"], _summarize_orders_for_briefing(delta_transactions)))
    lines.extend(["", "## Active Preview Orders", "NOT EXECUTABLE / ANALYSIS ONLY"])
    if active_orders:
        lines.extend(_markdown_table(["Action", "Ticker", "Shares", "Value USD", "Block reason"], _summarize_orders_for_briefing([dict(row) for row in active_orders])))
    elif active_order_count > 0:
        lines.append(f"Active Preview reports {active_order_count} non-executable orders in outputs/active_preview_orders.csv.")
    else:
        lines.append("No Active Preview orders.")

    lines.extend(
        [
            "",
            "## Why HOLD / Why Trade",
            f"- selected_reason: {selected_reason}",
            f"- gate_reason: {run_status.get('gate_reason', decision_context.get('trade_decision_reason', 'n/a'))}",
            f"- trade_now_edge: {_safe_float(cost_edge.get('trade_now_edge', 0.0)):.6f}",
            f"- trade_now_hurdle: {_safe_float(cost_edge.get('trade_now_hurdle', cost_edge.get('effective_trade_now_hurdle', 0.0))):.6f}",
            f"- best invalid candidate: {decision_context.get('best_invalid_candidate', 'n/a')}",
            f"- best valid non-HOLD candidate: {decision_context.get('best_non_hold_candidate', 'n/a')}",
            f"- current constraint violation: {current_portfolio.get('current_portfolio_constraint_errors', current_portfolio.get('current_constraint_errors', 'n/a'))}",
        ]
    )
    why_hold_lines = list(hold_analysis.get("why_hold_lines", []) or [])
    for line in why_hold_lines[:6]:
        lines.append(f"- {line}")

    lines.extend(
        [
            "",
            "## Signal Drivers",
            f"- continuous_model_optimal_candidate: {decision_context.get('continuous_candidate', 'n/a')}",
            f"- final_discrete_candidate: {_selected_candidate(review)}",
        ]
    )
    for line in list(decision_context.get("positive_drivers", []))[:5]:
        lines.append(f"- positive: {line}")
    for line in list(decision_context.get("negative_drivers", []))[:5]:
        lines.append(f"- negative: {line}")
    lines.extend(["", "## Constraint Pressure"])
    for line in list(decision_context.get("rejected_candidates", []))[:8]:
        lines.append(f"- {line}")
    if not list(decision_context.get("rejected_candidates", [])):
        lines.append("- See outputs/constraint_pressure_report.txt if available.")
    lines.extend(
        [
            f"- current portfolio violations: {current_portfolio.get('current_portfolio_constraint_errors', 'n/a')}",
            "",
            "## Risk and Data Quality",
            f"- data_source: {data_status.get('data_source', 'n/a')}",
            f"- used_cache_fallback: {_bool_text(data_status.get('used_cache_fallback', False))}",
            f"- latest_price_date: {data_status.get('latest_price_date', 'n/a')}",
            f"- synthetic_data: {_bool_text(data_status.get('synthetic_data', False))}",
            f"- price_basis: {data_status.get('price_basis', 'n/a')}",
            f"- scenario/tail risk: probability_beats_current={decision_context.get('probability_beats_current', 'n/a')}, tail_risk={decision_context.get('tail_risk_target', 'n/a')}",
            "",
            "## Safety Gates",
            f"- DRY_RUN: {_bool_text(effective_settings.get('dry_run', True))}",
            f"- ENABLE_EXTERNAL_BROKER: {_bool_text(effective_settings.get('enable_external_broker', False))}",
            f"- ENABLE_INVESTOPEDIA_SIMULATOR: {_bool_text(effective_settings.get('enable_investopedia_simulator', False))}",
            f"- ENABLE_LOCAL_PAPER_TRADING: {_bool_text(effective_settings.get('enable_local_paper_trading', False))}",
            f"- ENABLE_EMAIL_NOTIFICATIONS: {_bool_text(effective_settings.get('enable_email_notifications', False))}",
            f"- real_email_send_allowed: {_bool_text(real_send_allowed)}",
            "- active_preview_executable: false",
            "",
            "## Technical Appendix",
            f"- current_portfolio_score: {_safe_float(cost_edge.get('current_portfolio_score', 0.0)):.6f}",
            f"- target_score_after_costs: {_safe_float(cost_edge.get('target_score_after_costs', 0.0)):.6f}",
            f"- execution_buffer: {_safe_float(cost_edge.get('execution_buffer', 0.0)):.6f}",
            f"- model_uncertainty_buffer: {_safe_float(cost_edge.get('model_uncertainty_buffer', 0.0)):.6f}",
            f"- effective_horizon_days: {_effective_horizon_days(review)}",
            f"- scenario_count: {decision_context.get('scenario_count', 'n/a')}",
            f"- decision_fingerprint: {email_state.get('current_decision_fingerprint', build_decision_fingerprint(review, settings=effective_settings))}",
            "- file paths: outputs/daily_portfolio_briefing.md, outputs/daily_portfolio_briefing.html, outputs/manual_simulator_orders.csv, outputs/active_preview_orders.csv",
        ]
    )
    return sanitize_for_output("\n".join(lines).strip() + "\n")


def _markdown_to_basic_html(markdown_text: str) -> str:
    body_lines: list[str] = []
    in_ul = False
    in_table = False
    table_header_seen = False

    def close_ul() -> None:
        nonlocal in_ul
        if in_ul:
            body_lines.append("</ul>")
            in_ul = False

    def close_table() -> None:
        nonlocal in_table, table_header_seen
        if in_table:
            body_lines.append("</tbody></table>")
            in_table = False
            table_header_seen = False

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            close_ul()
            close_table()
            continue
        if line.startswith("| "):
            close_ul()
            cells = [html.escape(cell.strip()) for cell in line.strip("|").split("|")]
            if all(set(cell.replace("-", "").strip()) == set() for cell in cells):
                continue
            if not in_table:
                body_lines.append("<table>")
                in_table = True
                table_header_seen = False
            tag = "th" if not table_header_seen else "td"
            body_lines.append("<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>")
            table_header_seen = True
            continue
        close_table()
        escaped = html.escape(line)
        if line.startswith("# "):
            close_ul()
            body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            close_ul()
            body_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_ul:
                body_lines.append("<ul>")
                in_ul = True
            body_lines.append(f"<li>{html.escape(line[2:])}</li>")
        else:
            close_ul()
            body_lines.append(f"<p>{escaped}</p>")
    close_ul()
    close_table()
    return "\n".join(body_lines)


def build_daily_portfolio_briefing_html(markdown_text: str) -> str:
    """Render the Markdown briefing to a self-contained operator-friendly HTML page."""

    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Daily Portfolio Briefing</title>\n"
        "<style>\n"
        "body{margin:0;background:#f5f1e8;color:#1f2a2e;font-family:Georgia,'Times New Roman',serif;}\n"
        "main{max-width:1080px;margin:0 auto;padding:32px 20px 56px;}\n"
        "h1{font-size:42px;margin:0 0 18px;color:#12312f;}\n"
        "h2{margin-top:28px;padding:12px 14px;background:#153f3b;color:#fff;border-radius:14px;font-size:20px;}\n"
        "p,li{font-size:16px;line-height:1.55;}\n"
        "ul{background:#fffaf0;border:1px solid #eadfc9;border-radius:14px;padding:16px 24px;}\n"
        "table{width:100%;border-collapse:collapse;background:white;border-radius:14px;overflow:hidden;box-shadow:0 8px 24px rgba(28,37,40,.08);}\n"
        "th{background:#d7b46a;color:#1f2a2e;text-align:left;padding:10px;font-size:13px;text-transform:uppercase;letter-spacing:.04em;}\n"
        "td{padding:10px;border-top:1px solid #eee4d0;font-size:14px;}\n"
        ".shell{background:linear-gradient(135deg,#fffdf6,#e8f0e8);border:1px solid #ddcfad;border-radius:24px;padding:28px;box-shadow:0 18px 44px rgba(28,37,40,.12);}\n"
        "</style>\n"
        "</head>\n"
        "<body><main><div class=\"shell\">\n"
        f"{_markdown_to_basic_html(markdown_text)}\n"
        "</div></main></body></html>\n"
    )


def _subject_suffix(review: dict[str, Any], issues: dict[str, Any]) -> str:
    if int(issues.get("hard_fail_count", 0)) > 0:
        return "BLOCK - Data Problem"
    if not _safe_bool(review.get("run_status", {}).get("within_allowed_window", False)):
        return "WAIT - Outside Window"
    if list(review.get("delta_transactions", [])) and _safe_bool(review.get("manual_orders_preview_ready", False)):
        return "BUY/SELL Preview - Data OK"
    active_preview = dict(review.get("active_preview", {}) or {})
    if int(active_preview.get("active_preview_order_count", 0) or 0) > 0:
        return "ACTIVE PREVIEW HAS ORDERS - Safe Mode HOLD"
    if str(review.get("run_status", {}).get("final_action", "")).upper() == "HOLD":
        return "HOLD - No Safe Orders"
    return f"{issues.get('review_status', 'REVIEW')} - No Safe Orders"


def build_daily_email_subject(review: dict[str, Any], issues: dict[str, Any]) -> str:
    review_date = str(review.get("run_status", {}).get("review_date", "n/a"))
    return f"Daily Portfolio Briefing {review_date} - {_subject_suffix(review, issues)}"


def build_daily_email_briefing(
    review: dict[str, Any],
    issues: dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> str:
    run_status = review.get("run_status", {})
    data_status = review.get("data_status", {})
    current_portfolio = review.get("current_portfolio", {})
    delta_transactions = list(review.get("delta_transactions", []))
    decision_context = review.get("decision_context", {})
    active_preview = dict(review.get("active_preview", {}) or {})
    hold_analysis = dict(review.get("hold_analysis", {}) or {})
    effective_settings = {**load_daily_review_settings(), **dict(settings or {})}
    manual_order_count = _manual_order_count(review)
    operator_instruction = _build_operator_instruction(review, issues)
    safe_action = str(run_status.get("final_action", "n/a"))
    active_order_count = int(active_preview.get("active_preview_order_count", 0) or 0)
    header_lines = [
        f"Kurzstatus: {issues.get('review_status', 'REVIEW')} / {safe_action} / erster Blocker: {issues.get('first_blocker', 'none')}"
    ]
    if int(issues.get("hard_fail_count", 0)) > 0:
        header_lines.append(f"Hauptproblem zuerst: {issues.get('first_blocker', 'none')}")
    if active_order_count > 0 and safe_action.upper() in {"HOLD", "WAIT", "WAIT_OUTSIDE_WINDOW", "BLOCK"}:
        header_lines.append("Safe Mode bleibt HOLD/WAIT, aber Active Preview hat nicht-ausfuehrbare Analyse-Orders gefunden.")
    lines = [
        *header_lines,
        "",
        "Dies ist das kompakte Daily Portfolio Briefing.",
        "Keine echten Orders wurden gesendet.",
        "Vollstaendige Briefing-Dateien: outputs/daily_portfolio_briefing.md und outputs/daily_portfolio_briefing.html",
        "",
        "EXECUTIVE SUMMARY",
        f"- Safe decision: {safe_action} fuer {_selected_candidate(review)}",
        f"- Safe selected_reason: {_selected_reason(review)}",
        f"- Active Preview: {active_preview.get('active_preview_action', 'HOLD')} / {active_preview.get('active_preview_candidate', 'HOLD_CURRENT')}",
        f"- Orders: safe={manual_order_count}, active_preview={active_order_count}",
        f"- Wichtigster Blocker: {issues.get('first_blocker', 'none')}",
        "",
        "DECISION CARD",
        f"- safe_final_action: {safe_action}",
        f"- safe_selected_candidate: {_selected_candidate(review)}",
        f"- safe_selected_reason: {_selected_reason(review)}",
        f"- active_preview_action: {active_preview.get('active_preview_action', 'HOLD')}",
        f"- active_preview_candidate: {active_preview.get('active_preview_candidate', 'HOLD_CURRENT')}",
        f"- order_count: {manual_order_count}",
        f"- buy_count: {int(review.get('order_summary', {}).get('buy_count', 0) or 0)}",
        f"- sell_count: {int(review.get('order_summary', {}).get('sell_count', 0) or 0)}",
        f"- first_blocker: {issues.get('first_blocker', 'none')}",
        "",
        "TODAY'S ACTION",
        f"- final_action: {safe_action}",
        f"- operator_instruction: {operator_instruction}",
        f"- manual_order_count: {manual_order_count}",
        "- manual_orders_file: outputs/manual_simulator_orders.csv",
        "",
        "DATA STATUS",
        f"- data_source: {data_status.get('data_source', 'n/a')}",
        f"- latest_price_date: {data_status.get('latest_price_date', 'n/a')}",
        f"- used_cache_fallback: {_bool_text(data_status.get('used_cache_fallback', False))}",
        f"- synthetic_data: {_bool_text(data_status.get('synthetic_data', False))}",
        f"- data_freshness_ok: {_bool_text(data_status.get('data_freshness_ok', False))}",
        f"- live_data_error: {data_status.get('live_data_error', '') or 'none'}",
        "",
        "SAFETY STATUS",
        "- real_orders_enabled: false",
        "- external_broker_enabled: false",
        "- investopedia_enabled: false",
        "- active_preview_executable: false",
        f"- email_phase: {effective_settings.get('phase', 'DAILY_REVIEW_PREVIEW')}",
        f"- hard_fail_count: {issues.get('hard_fail_count', 0)}",
        f"- warning_count: {issues.get('soft_warning_count', 0)}",
        "",
        "DECISION SUMMARY",
        f"- continuous_candidate: {decision_context.get('continuous_candidate', 'n/a')}",
        f"- final_discrete_candidate: {decision_context.get('final_discrete_candidate', 'n/a')}",
        f"- trade_now_edge: {_safe_float(review.get('cost_edge', {}).get('trade_now_edge', 0.0)):.6f}",
        f"- first_blocker: {issues.get('first_blocker', 'none')}",
        f"- all_blockers: {' | '.join(map(str, issues.get('all_blockers', ['none'])))}",
        "",
        "ACTIVE PREVIEW",
        f"- active_preview_action: {active_preview.get('active_preview_action', 'HOLD')}",
        f"- active_preview_candidate: {active_preview.get('active_preview_candidate', 'HOLD_CURRENT')}",
        f"- active_preview_trade_now_edge: {_safe_float(active_preview.get('active_preview_trade_now_edge', 0.0)):.6f}",
        f"- active_preview_hurdle: {_safe_float(active_preview.get('active_preview_hurdle', 0.0)):.6f}",
        f"- active_preview_order_count: {int(active_preview.get('active_preview_order_count', 0) or 0)}",
        "- active_preview_executable: false",
        "- active_preview_orders_file: outputs/active_preview_orders.csv",
        "- active_preview_note: NOT EXECUTABLE / ANALYSIS ONLY",
        "",
        "SAFE ORDERS",
    ]
    if not delta_transactions or manual_order_count == 0:
        lines.append("- No safe executable/manual delta orders.")
        lines.append("- Keine Simulator-Orders eingeben.")
    else:
        for row in delta_transactions[:8]:
            lines.append(
                f"- {row.get('action')} {row.get('ticker')}: shares={_safe_float(row.get('order_shares', 0.0)):.4f}, value={_safe_float(row.get('estimated_order_value', 0.0)):.2f} USD"
            )
    lines.extend(
        [
            "",
            "PORTFOLIO SNAPSHOT",
            f"- nav_usd: {_safe_float(current_portfolio.get('nav_usd', 0.0)):.2f}",
            f"- cash_usd: {_safe_float(current_portfolio.get('cash_usd', 0.0)):.2f}",
            f"- positions_count: {current_portfolio.get('positions_count', 0)}",
            f"- current portfolio 100pct cash: {_bool_text(current_portfolio.get('current_portfolio_100pct_cash', False))}",
            f"- current_portfolio_constraint_valid: {current_portfolio.get('current_portfolio_constraint_valid', 'n/a')}",
            "",
            "WHY HOLD / WHY TRADE",
            f"- selected_reason: {_selected_reason(review)}",
            f"- gate_reason: {run_status.get('gate_reason', decision_context.get('trade_decision_reason', 'n/a'))}",
            f"- trade_now_edge: {_safe_float(review.get('cost_edge', {}).get('trade_now_edge', 0.0)):.6f}",
            f"- first_blocker: {issues.get('first_blocker', 'none')}",
            "",
        ]
    )
    if str(safe_action).upper() == "HOLD":
        lines.append("Warum HOLD?")
    else:
        lines.append("Warum handeln / nicht handeln?")
    why_hold_lines = list(hold_analysis.get("why_hold_lines", []) or [])
    if why_hold_lines:
        for line in why_hold_lines[:6]:
            lines.append(f"- {line}")
    else:
        lines.append("- n/a")
    lines.extend(
        [
            "",
            "RELEVANTE DATEIEN",
            "- Fancy Briefing Markdown: outputs/daily_portfolio_briefing.md",
            "- Fancy Briefing HTML: outputs/daily_portfolio_briefing.html",
            "- Fuer Simulatororders verwenden: outputs/manual_simulator_orders.csv",
            "- Active Preview Analyseorders: outputs/active_preview_orders.csv (NOT EXECUTABLE)",
            "- Nicht verwenden: outputs/order_preview.csv",
            "- Review-Text: outputs/daily_portfolio_review.txt",
            "- Entscheidungs-Summary: outputs/today_decision_summary.txt",
            "- Rebalance-Entscheidung: outputs/rebalance_decision_report.txt",
            "- Daily-Bot-Entscheidung: outputs/daily_bot_decision_report.txt",
            "- Datenfrische: outputs/current_data_freshness_report.txt",
        ]
    )
    if _safe_bool(data_status.get("synthetic_data", False)):
        lines.append("Blockiert: synthetische Daten; keine Orders.")
    if _safe_bool(data_status.get("used_cache_fallback", False)):
        lines.append("Warnung: Live-Daten nicht genutzt; Bericht nur vorsichtig verwenden.")
    if str(safe_action).upper() == "HOLD":
        lines.append("Heute keine Orders eingeben. Beste Aktion laut Bot: HOLD.")
    lines.extend(
        [
            "",
            "Probleme / Warnungen:",
        ]
    )
    for item in list(issues.get("issue_table", []))[:8]:
        lines.append(f"- {item.get('severity')}: {item.get('message')}")
    return "\n".join(lines).strip() + "\n"


def _extract_bullet_value(text: str, key: str) -> str:
    match = re.search(rf"^- {re.escape(key)}:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _validation_item(status: str, reason: str) -> dict[str, str]:
    normalized = str(status or "FAIL").strip().upper()
    if normalized not in {"PASS", "WARN", "FAIL"}:
        normalized = "FAIL"
    return {"status": normalized, "reason": sanitize_for_output(reason)}


def build_daily_review_validation_summary(
    review: dict[str, Any],
    issues: dict[str, Any],
    *,
    settings: dict[str, Any] | None = None,
    email_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_settings = {**load_daily_review_settings(), **dict(settings or {})}
    run_status = review.get("run_status", {})
    data_status = review.get("data_status", {})
    current_portfolio = review.get("current_portfolio", {})
    current_positions = list(review.get("current_positions", []))
    order_summary = review.get("order_summary", {})
    subject_text = build_daily_email_subject(review, issues).strip()
    body_text = build_daily_email_briefing(review, issues, settings=effective_settings)
    final_action = str(run_status.get("final_action", "") or "").strip()
    manual_order_count = _manual_order_count(review)
    operator_instruction = _extract_bullet_value(body_text, "operator_instruction")
    used_cache_fallback = _safe_bool(data_status.get("used_cache_fallback", False))
    synthetic_data = _safe_bool(data_status.get("synthetic_data", False))
    data_freshness_ok = _safe_bool(data_status.get("data_freshness_ok", False))
    within_allowed_window = _safe_bool(run_status.get("within_allowed_window", False))
    contains_synthetic_block_warning = "Blockiert: synthetische Daten; keine Orders." in body_text
    contains_stale_data_block_warning = "Blockiert: Daten nicht frisch." in body_text
    contains_cache_warning = "Warnung: Live-Daten nicht genutzt; Bericht nur vorsichtig verwenden." in body_text
    body_sanitized = sanitize_for_output(body_text)
    subject_sanitized = sanitize_for_output(subject_text)
    no_secret_leak_detected = body_sanitized == body_text and subject_sanitized == subject_text
    daily_bot_refs = [
        "outputs/today_decision_summary.txt",
        "outputs/rebalance_decision_report.txt",
        "outputs/daily_bot_decision_report.txt",
        "outputs/current_data_freshness_report.txt",
        "outputs/manual_simulator_orders.csv",
    ]
    missing_daily_bot_refs = [item for item in daily_bot_refs if item not in body_text]
    research_refs_found = [
        item
        for item in ["research_order_preview.csv", "research_latest_decision_report.txt"]
        if item in body_text
    ]
    order_preview_misused = "Fuer Simulatororders verwenden: outputs/order_preview.csv" in body_text
    dedupe_state = dict(email_state or {})
    dedupe_status_present = all(
        key in dedupe_state
        for key in [
            "dedupe_key",
            "duplicate_today",
            "last_attempt_date",
            "last_sent_date",
            "last_send_success",
        ]
    )

    checks: dict[str, dict[str, str]] = {
        "subject_non_empty": _validation_item(
            "PASS" if bool(subject_text) else "FAIL",
            "Subject line present." if subject_text else "Subject line is empty.",
        ),
        "body_non_empty": _validation_item(
            "PASS" if bool(body_text.strip()) else "FAIL",
            "Mail body present." if body_text.strip() else "Mail body is empty.",
        ),
        "contains_today_action": _validation_item(
            "PASS" if "TODAY'S ACTION" in body_text else "FAIL",
            "Mail body contains TODAY'S ACTION."
            if "TODAY'S ACTION" in body_text
            else "Missing TODAY'S ACTION section.",
        ),
        "contains_data_status": _validation_item(
            "PASS" if "DATA STATUS" in body_text else "FAIL",
            "Mail body contains DATA STATUS." if "DATA STATUS" in body_text else "Missing DATA STATUS section.",
        ),
        "contains_safety_status": _validation_item(
            "PASS" if "SAFETY STATUS" in body_text else "FAIL",
            "Mail body contains SAFETY STATUS." if "SAFETY STATUS" in body_text else "Missing SAFETY STATUS section.",
        ),
        "contains_final_action": _validation_item(
            "PASS" if bool(final_action) and "- final_action:" in body_text else "FAIL",
            f"final_action present: {final_action}" if final_action and "- final_action:" in body_text else "final_action missing in mail body.",
        ),
        "contains_manual_order_count": _validation_item(
            "PASS" if "- manual_order_count:" in body_text else "FAIL",
            f"manual_order_count present: {manual_order_count}" if "- manual_order_count:" in body_text else "manual_order_count missing in mail body.",
        ),
        "contains_data_source": _validation_item(
            "PASS" if "- data_source:" in body_text else "FAIL",
            "data_source present in DATA STATUS." if "- data_source:" in body_text else "data_source missing in mail body.",
        ),
        "contains_latest_price_date": _validation_item(
            "PASS" if "- latest_price_date:" in body_text else "FAIL",
            "latest_price_date present in DATA STATUS."
            if "- latest_price_date:" in body_text
            else "latest_price_date missing in mail body.",
        ),
        "contains_cache_fallback_status": _validation_item(
            "PASS" if "- used_cache_fallback:" in body_text else "FAIL",
            "used_cache_fallback present in DATA STATUS."
            if "- used_cache_fallback:" in body_text
            else "used_cache_fallback missing in mail body.",
        ),
        "contains_synthetic_data_status": _validation_item(
            "PASS" if "- synthetic_data:" in body_text else "FAIL",
            "synthetic_data present in DATA STATUS."
            if "- synthetic_data:" in body_text
            else "synthetic_data missing in mail body.",
        ),
        "contains_operator_instruction": _validation_item(
            "PASS" if bool(operator_instruction) else "FAIL",
            f"operator_instruction present: {operator_instruction}" if operator_instruction else "operator_instruction missing in mail body.",
        ),
        "no_secret_leak_detected": _validation_item(
            "PASS" if no_secret_leak_detected else "FAIL",
            "No secret-like content detected in subject/body."
            if no_secret_leak_detected
            else "Secret-like content detected; output differs after sanitization.",
        ),
        "daily_bot_context_used": _validation_item(
            "PASS" if not missing_daily_bot_refs else "FAIL",
            "Daily-bot output references are present."
            if not missing_daily_bot_refs
            else f"Missing daily-bot references: {', '.join(missing_daily_bot_refs)}",
        ),
        "research_context_not_primary": _validation_item(
            "PASS" if not research_refs_found and not order_preview_misused else "FAIL",
            "Research-only files are not primary."
            if not research_refs_found and not order_preview_misused
            else f"Research-only references or misuse found: {', '.join(research_refs_found) or 'outputs/order_preview.csv misuse'}",
        ),
        "email_gate_status_present": _validation_item(
            "PASS" if "- email_phase:" in body_text else "FAIL",
            "email_phase present in SAFETY STATUS."
            if "- email_phase:" in body_text
            else "email_phase missing in mail body.",
        ),
        "dedupe_status_present": _validation_item(
            "PASS" if dedupe_status_present else "FAIL",
            "Dedupe state fields present." if dedupe_status_present else "Dedupe state fields missing.",
        ),
    }

    hard_fail_reasons: list[str] = []
    soft_warning_reasons: list[str] = []
    for required_key in [
        "subject_non_empty",
        "body_non_empty",
        "contains_today_action",
        "contains_data_status",
        "contains_safety_status",
        "contains_final_action",
        "contains_operator_instruction",
        "no_secret_leak_detected",
        "daily_bot_context_used",
        "research_context_not_primary",
        "email_gate_status_present",
        "dedupe_status_present",
    ]:
        if checks[required_key]["status"] == "FAIL":
            hard_fail_reasons.append(f"{required_key}: {checks[required_key]['reason']}")

    if synthetic_data:
        if contains_synthetic_block_warning:
            checks["contains_synthetic_data_status"] = _validation_item(
                "WARN",
                "synthetic_data=true is clearly blocked in operator instruction.",
            )
        else:
            hard_fail_reasons.append("synthetic_data=true without clear block warning.")

    if not data_freshness_ok:
        if contains_stale_data_block_warning:
            checks["contains_latest_price_date"] = _validation_item(
                "WARN",
                "data_freshness_ok=false is clearly blocked in operator instruction.",
            )
        else:
            hard_fail_reasons.append("data_freshness_ok=false without clear block warning.")

    if used_cache_fallback:
        if contains_cache_warning:
            soft_warning_reasons.append("used_cache_fallback=true with explicit caution in operator instruction.")
        else:
            hard_fail_reasons.append("used_cache_fallback=true without explicit caution in operator instruction.")

    if str(final_action).upper() == "HOLD":
        if "Heute keine Orders eingeben. Beste Aktion laut Bot: HOLD." in body_text:
            soft_warning_reasons.append("final_action=HOLD with clear no-order instruction.")
        else:
            hard_fail_reasons.append("final_action=HOLD without clear no-order instruction.")

    if manual_order_count == 0:
        if "Keine Simulator-Orders eingeben." in body_text:
            soft_warning_reasons.append("manual_order_count=0 with clear no-order instruction.")
        else:
            hard_fail_reasons.append("manual_order_count=0 without clear no-order instruction.")

    if not within_allowed_window:
        if "Keine Orders eingeben." in body_text or "Heute keine Orders eingeben." in body_text:
            soft_warning_reasons.append("Handelsfenster geschlossen; no-order instruction present.")
        else:
            hard_fail_reasons.append("Handelsfenster geschlossen without clear no-order instruction.")

    if hard_fail_reasons:
        overall_status = "HARD_FAIL"
    elif soft_warning_reasons:
        overall_status = "SOFT_WARNING"
    else:
        overall_status = "PASS"

    return {
        "overall_status": overall_status,
        "sendable": overall_status != "HARD_FAIL",
        "checks": checks,
        "hard_fail_reasons": hard_fail_reasons,
        "soft_warning_reasons": soft_warning_reasons,
        "summary": {
            "final_action": final_action or "n/a",
            "manual_order_count": manual_order_count,
            "used_cache_fallback": used_cache_fallback,
            "synthetic_data": synthetic_data,
            "data_freshness_ok": data_freshness_ok,
            "within_allowed_window": within_allowed_window,
            "email_phase": str(effective_settings.get("phase", "DAILY_REVIEW_PREVIEW")),
        },
        "subject": subject_text,
        "body": body_text,
    }


def _current_email_provider(settings: dict[str, Any] | None = None) -> str:
    effective_settings = {**load_daily_review_settings(), **dict(settings or {})}
    provider = (
        str(
            effective_settings.get("EMAIL_PROVIDER")
            or effective_settings.get("email_provider")
            or load_email_settings().get("EMAIL_PROVIDER", "brevo")
        )
        .strip()
        .lower()
    )
    return provider or "brevo"


def _state_updated_after_success_only(
    *,
    review: dict[str, Any],
    email_result: dict[str, Any],
    email_state: dict[str, Any],
) -> bool:
    review_date = str(review.get("run_status", {}).get("review_date", "") or "")
    body_hash = str(email_state.get("dedupe_key", "") or email_state.get("current_body_hash", "") or "")
    attempted = bool(email_result.get("attempted", False))
    provider_accepted = bool(email_result.get("provider_accepted", email_result.get("sent", False)))
    last_send_success = bool(email_state.get("last_send_success", False))
    last_sent_date = str(email_state.get("last_sent_date", "") or "")
    last_body_hash = str(email_state.get("last_body_hash", "") or "")

    if provider_accepted:
        return last_send_success and last_sent_date == review_date and last_body_hash == body_hash
    if attempted:
        return not (last_send_success and last_sent_date == review_date and last_body_hash == body_hash)
    return True


def build_email_final_acceptance_report(
    review: dict[str, Any],
    issues: dict[str, Any],
    *,
    settings: dict[str, Any] | None = None,
    email_result: dict[str, Any] | None = None,
    email_state: dict[str, Any] | None = None,
) -> str:
    effective_settings = {**load_daily_review_settings(), **dict(settings or {})}
    email_result_payload = dict(
        email_result
        or {
            "attempted": False,
            "sent": False,
            "reason": "preview_only",
            "error_type": None,
            "sanitized_error": None,
        }
    )
    email_state_payload = dict(email_state or {})
    gate_status = get_email_gate_status(effective_settings)
    validation_summary = build_daily_review_validation_summary(
        review,
        issues,
        settings=effective_settings,
        email_state=email_state_payload,
    )
    run_status = review.get("run_status", {})
    data_status = review.get("data_status", {})
    decision_context = review.get("decision_context", {})
    effective_real_send_allowed = bool(
        gate_status.get("real_email_send_allowed", False) and int(issues.get("hard_fail_count", 0)) == 0
    )
    provider = _current_email_provider(effective_settings)
    state_success_only = _state_updated_after_success_only(
        review=review,
        email_result=email_result_payload,
        email_state=email_state_payload,
    )
    no_secret_leak_detected = (
        validation_summary.get("checks", {})
        .get("no_secret_leak_detected", {})
        .get("status")
        == "PASS"
    )
    no_broker_enabled = not _safe_bool(gate_status.get("external_broker_enabled", False))
    no_investopedia_enabled = not _safe_bool(gate_status.get("investopedia_enabled", False))
    no_real_orders = (
        _safe_bool(review.get("preview_only", True))
        and no_broker_enabled
        and no_investopedia_enabled
        and not _safe_bool(gate_status.get("local_paper_trading_enabled", False))
    )
    checks = validation_summary.get("checks", {})
    subject_non_empty = checks.get("subject_non_empty", {}).get("status") == "PASS"
    body_non_empty = checks.get("body_non_empty", {}).get("status") == "PASS"
    contains_today_action = checks.get("contains_today_action", {}).get("status") == "PASS"
    contains_data_status = checks.get("contains_data_status", {}).get("status") == "PASS"
    contains_safety_status = checks.get("contains_safety_status", {}).get("status") == "PASS"
    contains_operator_instruction = checks.get("contains_operator_instruction", {}).get("status") == "PASS"
    preview_ready = all(
        [
            subject_non_empty,
            body_non_empty,
            contains_today_action,
            contains_data_status,
            contains_safety_status,
            contains_operator_instruction,
            no_secret_leak_detected,
        ]
    )
    send_path_ready = all(
        [
            validation_summary.get("overall_status") != "HARD_FAIL",
            state_success_only,
            no_broker_enabled,
            no_investopedia_enabled,
            no_real_orders,
        ]
    )

    status_lines: list[str] = []
    if preview_ready:
        status_lines.append("- PASS: Mail preview ready")
    else:
        status_lines.append("- FAIL: Mail preview not ready")
    if send_path_ready:
        status_lines.append("- PASS: Send path ready")
    else:
        status_lines.append("- FAIL: Send path not ready")
    if _safe_bool(data_status.get("used_cache_fallback", False)):
        status_lines.append("- WARN: cache_fallback used")
    if str(run_status.get("final_action", "")).upper() == "HOLD" or _manual_order_count(review) == 0:
        status_lines.append("- WARN: HOLD/no orders currently")
    for reason in list(validation_summary.get("hard_fail_reasons", [])):
        status_lines.append(f"- FAIL: {reason}")
    for item in issues.get("hard_fails", []):
        status_lines.append(f"- FAIL: {item.get('message')}")

    if validation_summary.get("overall_status") == "HARD_FAIL" or int(issues.get("hard_fail_count", 0)) > 0:
        readiness = "NOT_READY"
    elif effective_real_send_allowed:
        readiness = "READY_FOR_DAILY_EMAIL_SEND"
    else:
        readiness = "READY_FOR_DAILY_EMAIL_PREVIEW"

    lines = [
        "Email Final Acceptance Report",
        "",
        "CONFIG",
        f"- phase: {gate_status.get('phase', effective_settings.get('phase', 'DAILY_REVIEW_PREVIEW'))}",
        f"- real_email_send_allowed: {_bool_text(effective_real_send_allowed)}",
        f"- email_provider: {provider}",
        f"- email_recipient_configured: {_bool_text(gate_status.get('email_recipient_configured', False))}",
        f"- email_dry_run: {_bool_text(gate_status.get('email_dry_run', True))}",
        f"- external_broker_enabled: {_bool_text(gate_status.get('external_broker_enabled', False))}",
        f"- investopedia_enabled: {_bool_text(gate_status.get('investopedia_enabled', False))}",
        "",
        "DAILY BOT",
        f"- final_action: {run_status.get('final_action', 'n/a')}",
        f"- final_discrete_candidate: {decision_context.get('final_discrete_candidate', 'n/a')}",
        f"- data_source: {data_status.get('data_source', 'n/a')}",
        f"- used_cache_fallback: {_bool_text(data_status.get('used_cache_fallback', False))}",
        f"- synthetic_data: {_bool_text(data_status.get('synthetic_data', False))}",
        f"- data_freshness_ok: {_bool_text(data_status.get('data_freshness_ok', False))}",
        f"- latest_price_date: {data_status.get('latest_price_date', 'n/a')}",
        f"- manual_order_count: {_manual_order_count(review)}",
        "",
        "MAIL CONTENT",
        f"- subject_non_empty: {checks.get('subject_non_empty', {}).get('status', 'FAIL')}",
        f"- body_non_empty: {checks.get('body_non_empty', {}).get('status', 'FAIL')}",
        f"- contains_today_action: {checks.get('contains_today_action', {}).get('status', 'FAIL')}",
        f"- contains_data_status: {checks.get('contains_data_status', {}).get('status', 'FAIL')}",
        f"- contains_safety_status: {checks.get('contains_safety_status', {}).get('status', 'FAIL')}",
        f"- contains_operator_instruction: {checks.get('contains_operator_instruction', {}).get('status', 'FAIL')}",
        "",
        "SEND RESULT",
        f"- attempted: {_bool_text(email_result_payload.get('attempted', False))}",
        f"- provider_accepted: {_bool_text(email_result_payload.get('provider_accepted', email_result_payload.get('sent', False)))}",
        f"- provider_message_id: {sanitize_for_output(email_result_payload.get('provider_message_id') or 'unavailable')}",
        f"- provider_message_id_unavailable: {_bool_text(email_result_payload.get('provider_message_id_unavailable', True))}",
        f"- delivery_confirmed: {_bool_text(email_result_payload.get('delivery_confirmed', False))}",
        f"- delivery_status: {sanitize_for_output(email_result_payload.get('delivery_status') or 'delivery_unknown')}",
        f"- recipient_received_confirmed: {_bool_text(email_result_payload.get('recipient_received_confirmed', False))}",
        f"- reason: {sanitize_for_output(email_result_payload.get('reason', 'preview_only'))}",
        f"- error_type: {sanitize_for_output(email_result_payload.get('error_type') or 'none')}",
        f"- error_class: {sanitize_for_output(email_result_payload.get('error_class') or email_result_payload.get('error_type') or 'none')}",
        f"- sanitized_error: {sanitize_for_output(email_result_payload.get('sanitized_error') or 'none')}",
        "",
        "DEDUPE",
        f"- decision_fingerprint: {email_state_payload.get('current_decision_fingerprint', email_state_payload.get('dedupe_key', ''))}",
        f"- already_sent_same_decision_today: {_bool_text(email_state_payload.get('already_sent_today', False))}",
        f"- state_updated_after_success_only: {_bool_text(state_success_only)}",
        "",
        "SECURITY",
        f"- no_secret_leak_detected: {_bool_text(no_secret_leak_detected)}",
        f"- no_broker_enabled: {_bool_text(no_broker_enabled)}",
        f"- no_investopedia_enabled: {_bool_text(no_investopedia_enabled)}",
        f"- no_real_orders: {_bool_text(no_real_orders)}",
        "",
        "PASS/WARN/FAIL",
        *status_lines,
        "",
        readiness,
    ]
    return "\n".join(lines).strip() + "\n"


def build_email_safety_report(review: dict[str, Any], issues: dict[str, Any], settings: dict[str, Any], email_result: dict[str, Any] | None = None) -> str:
    email_result = dict(email_result or {"sent": False, "reason": "preview_only", "error": None})
    gate_status = get_email_gate_status(settings)
    gate_blockers = list(gate_status.get("blockers", []))
    render_artifacts = dict(review.get("render_artifacts", {}) or {})
    render_warnings = list(review.get("render_warnings", []) or [])
    if int(issues.get("hard_fail_count", 0)) > 0:
        gate_blockers.append("DAILY_REVIEW_VALIDATION_HARD_FAIL")
    effective_real_send_allowed = bool(gate_status.get("real_email_send_allowed", False) and int(issues.get("hard_fail_count", 0)) == 0)
    provider_accepted = bool(email_result.get("provider_accepted", email_result.get("sent", False)))
    lines = [
        "Daily Review Email Safety Report",
        "",
        "cron_environment_diagnostics:",
        f"cwd: {Path.cwd()}",
        f"python_executable: {sys.executable}",
        f"phase: {gate_status.get('phase', settings.get('phase', 'DAILY_REVIEW_PREVIEW'))}",
        f"EMAIL_DRY_RUN: {_bool_text(gate_status.get('email_dry_run', True))}",
        f"EMAIL_SEND_ENABLED: {_bool_text(gate_status.get('email_send_enabled', False))}",
        f"ENABLE_EMAIL_NOTIFICATIONS: {_bool_text(gate_status.get('enable_email_notifications', False))}",
        f"USER_CONFIRMED_EMAIL_PHASE: {_bool_text(gate_status.get('user_confirmed_email_phase', False))}",
        f"EMAIL_PROVIDER: {settings.get('email_provider', settings.get('EMAIL_PROVIDER', 'brevo'))}",
        f"EMAIL_RECIPIENT configured: {_bool_text(gate_status.get('email_recipient_configured', False))}",
        f"EMAIL_SENDER configured: {_bool_text(gate_status.get('email_sender_configured', False))}",
        f"BREVO_API_KEY configured: {_bool_text(gate_status.get('brevo_api_key_configured', False))}",
        f"SMTP_USERNAME configured: {_bool_text(bool(str(os.getenv('SMTP_USERNAME', '') or os.getenv('SMTP_USER', '')).strip()))}",
        f"SMTP_PASSWORD configured: {_bool_text(bool(str(os.getenv('SMTP_PASSWORD', '') or os.getenv('EMAIL_PASSWORD', '')).strip()))}",
        "",
        f"phase: {gate_status.get('phase', settings.get('phase', 'DAILY_REVIEW_PREVIEW'))}",
        f"gate_reason: {gate_status.get('reason', 'preview_only')}",
        f"gate_reason_detail: {gate_status.get('reason_detail', '')}",
        f"enable_email_notifications: {_bool_text(gate_status.get('enable_email_notifications', False))}",
        f"email_send_enabled: {_bool_text(gate_status.get('email_send_enabled', False))}",
        f"email_dry_run: {_bool_text(gate_status.get('email_dry_run', True))}",
        f"daily_briefing_only: {_bool_text(settings.get('daily_briefing_only', True))}",
        f"max_emails_per_day: {_parse_int(settings.get('max_emails_per_day'), 1)}",
        f"email_recipient_present: {_bool_text(gate_status.get('email_recipient_configured', False))}",
        f"user_confirmed_email_phase: {_bool_text(gate_status.get('user_confirmed_email_phase', False))}",
        f"external_broker_enabled: {_bool_text(gate_status.get('external_broker_enabled', False))}",
        f"investopedia_enabled: {_bool_text(gate_status.get('investopedia_enabled', False))}",
        f"local_paper_trading_enabled: {_bool_text(gate_status.get('local_paper_trading_enabled', False))}",
        f"real_email_send_allowed: {_bool_text(effective_real_send_allowed)}",
        f"email_provider: {settings.get('email_provider', 'brevo')}",
        f"review_status: {issues.get('review_status', 'REVIEW')}",
        f"hard_fail_count: {issues.get('hard_fail_count', 0)}",
        f"soft_warning_count: {issues.get('soft_warning_count', 0)}",
        f"info_count: {issues.get('info_count', 0)}",
        f"first_blocker: {issues.get('first_blocker', 'none')}",
        f"all_blockers: {' | '.join(map(str, issues.get('all_blockers', ['none'])))}",
        f"email_send_attempted: {_bool_text(email_result.get('attempted', False))}",
        f"provider_accepted: {_bool_text(provider_accepted)}",
        f"provider_message_id: {sanitize_for_output(email_result.get('provider_message_id') or 'unavailable')}",
        f"provider_message_id_unavailable: {_bool_text(email_result.get('provider_message_id_unavailable', True))}",
        f"delivery_confirmed: {_bool_text(email_result.get('delivery_confirmed', False))}",
        f"delivery_status: {sanitize_for_output(email_result.get('delivery_status') or 'delivery_unknown')}",
        f"recipient_received_confirmed: {_bool_text(email_result.get('recipient_received_confirmed', False))}",
        f"email_result_reason: {sanitize_for_output(email_result.get('reason', 'preview_only'))}",
        f"email_result_error_type: {sanitize_for_output(email_result.get('error_type') or 'none')}",
        f"email_result_error_class: {sanitize_for_output(email_result.get('error_class') or email_result.get('error_type') or 'none')}",
        f"email_result_error: {sanitize_for_output(email_result.get('sanitized_error') or email_result.get('error') or 'none')}",
        f"recipient_masked: {email_result.get('recipient_masked', '[REDACTED]')}",
        "",
        "gate_blockers:",
    ]
    if gate_blockers:
        lines.extend([f"- {sanitize_for_output(reason)}" for reason in gate_blockers])
    else:
        lines.append("- none")
    lines.extend(["", "gate_warnings:"])
    if gate_status.get("warnings"):
        lines.extend([f"- {sanitize_for_output(reason)}" for reason in list(gate_status.get("warnings", []))])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "safety_defaults:",
            f"- DRY_RUN={_bool_text(settings.get('dry_run', True))}",
            f"- ENABLE_EXTERNAL_BROKER={_bool_text(settings.get('enable_external_broker', False))}",
            f"- ENABLE_INVESTOPEDIA_SIMULATOR={_bool_text(settings.get('enable_investopedia_simulator', False))}",
            f"- ENABLE_LOCAL_PAPER_TRADING={_bool_text(settings.get('enable_local_paper_trading', False))}",
            f"- ENABLE_EMAIL_NOTIFICATIONS={_bool_text(settings.get('enable_email_notifications', False))}",
            "",
            "render_artifacts:",
            f"- daily_review_email_html: {render_artifacts.get('daily_review_email_html', 'outputs/daily_review_email.html')}",
            f"- daily_review_report_tex: {render_artifacts.get('daily_review_report_tex', 'outputs/daily_review_report.tex')}",
            f"- daily_review_report_pdf: {render_artifacts.get('daily_review_report_pdf', 'not_available')}",
            f"- current_portfolio_allocation_png: {render_artifacts.get('current_portfolio_allocation_png', 'outputs/charts/current_portfolio_allocation.png')}",
            f"- current_vs_target_weights_png: {render_artifacts.get('current_vs_target_weights_png', 'outputs/charts/current_vs_target_weights.png')}",
            "",
            "render_warnings:",
        ]
    )
    if render_warnings:
        lines.extend([f"- {sanitize_for_output(item)}" for item in render_warnings])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "issue_table:",
        ]
    )
    for item in issues.get("issue_table", []):
        lines.append(f"- {item.get('severity')}: {item.get('message')}")
    return "\n".join(lines).strip() + "\n"


def build_email_delivery_diagnosis_report(
    settings: dict[str, Any],
    email_result: dict[str, Any],
    email_state: dict[str, Any],
) -> str:
    """Build a secret-safe delivery/cron diagnosis report."""

    gate_status = get_email_gate_status(settings)
    lines = [
        "Email Delivery Diagnosis Report",
        "",
        "cron_environment_diagnostics:",
        f"cwd: {Path.cwd()}",
        f"python_executable: {sys.executable}",
        f"phase: {gate_status.get('phase', settings.get('phase', 'DAILY_REVIEW_PREVIEW'))}",
        f"EMAIL_DRY_RUN: {_bool_text(gate_status.get('email_dry_run', True))}",
        f"EMAIL_SEND_ENABLED: {_bool_text(gate_status.get('email_send_enabled', False))}",
        f"ENABLE_EMAIL_NOTIFICATIONS: {_bool_text(gate_status.get('enable_email_notifications', False))}",
        f"USER_CONFIRMED_EMAIL_PHASE: {_bool_text(gate_status.get('user_confirmed_email_phase', False))}",
        f"EMAIL_PROVIDER: {settings.get('email_provider', settings.get('EMAIL_PROVIDER', 'brevo'))}",
        f"EMAIL_RECIPIENT configured: {_bool_text(gate_status.get('email_recipient_configured', False))}",
        f"EMAIL_SENDER configured: {_bool_text(gate_status.get('email_sender_configured', False))}",
        f"BREVO_API_KEY configured: {_bool_text(gate_status.get('brevo_api_key_configured', False))}",
        f"SMTP_USERNAME configured: {_bool_text(bool(str(os.getenv('SMTP_USERNAME', '') or os.getenv('SMTP_USER', '')).strip()))}",
        f"SMTP_PASSWORD configured: {_bool_text(bool(str(os.getenv('SMTP_PASSWORD', '') or os.getenv('EMAIL_PASSWORD', '')).strip()))}",
        "",
        "send_status:",
        f"email_send_attempted: {_bool_text(email_result.get('attempted', False))}",
        f"provider_accepted: {_bool_text(email_result.get('provider_accepted', email_result.get('sent', False)))}",
        f"provider_message_id: {sanitize_for_output(email_result.get('provider_message_id') or 'unavailable')}",
        f"provider_message_id_unavailable: {_bool_text(email_result.get('provider_message_id_unavailable', True))}",
        f"delivery_confirmed: {_bool_text(email_result.get('delivery_confirmed', False))}",
        f"delivery_status: {sanitize_for_output(email_result.get('delivery_status') or 'delivery_unknown')}",
        f"recipient_received_confirmed: {_bool_text(email_result.get('recipient_received_confirmed', False))}",
        f"email_result_reason: {sanitize_for_output(email_result.get('reason', 'preview_only'))}",
        f"email_result_error_type: {sanitize_for_output(email_result.get('error_type') or 'none')}",
        f"email_result_error: {sanitize_for_output(email_result.get('sanitized_error') or email_result.get('error') or 'none')}",
        "",
        "dedupe_state:",
        f"last_decision_fingerprint: {email_state.get('last_decision_fingerprint', '')}",
        f"current_decision_fingerprint: {email_state.get('current_decision_fingerprint', '')}",
        f"last_recipient: {'configured' if str(email_state.get('last_recipient', '')).strip() else 'n/a'}",
        f"last_review_date: {email_state.get('last_review_date', 'n/a')}",
        f"last_provider_accept_success: {_bool_text(email_state.get('last_provider_accept_success', False))}",
        f"last_delivery_confirmed_success: {_bool_text(email_state.get('last_delivery_confirmed_success', False))}",
        f"last_provider_message_id: {sanitize_for_output(email_state.get('last_provider_message_id', '') or 'unavailable')}",
        f"last_attempted_at: {email_state.get('last_attempted_at', '') or 'n/a'}",
        f"last_accepted_at: {email_state.get('last_accepted_at', '') or 'n/a'}",
        f"last_delivery_status: {sanitize_for_output(email_state.get('last_delivery_status', 'delivery_unknown'))}",
    ]
    return sanitize_for_output("\n".join(lines).strip() + "\n")


def _read_last_email_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _email_result_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(state, dict) or not state:
        return None
    if not any(
        key in state
        for key in [
            "email_send_attempted",
            "email_result_sent",
            "email_result_reason",
            "last_reason",
        ]
    ):
        return None
    return {
        "attempted": bool(state.get("email_send_attempted", False)),
        "sent": bool(state.get("email_result_sent", False)),
        "provider_accepted": bool(state.get("provider_accepted", state.get("last_provider_accept_success", state.get("email_result_sent", False)))),
        "provider_message_id": str(state.get("provider_message_id", state.get("last_provider_message_id", "")) or "") or None,
        "provider_message_id_unavailable": bool(state.get("provider_message_id_unavailable", not bool(state.get("provider_message_id", "")))),
        "delivery_confirmed": bool(state.get("delivery_confirmed", state.get("last_delivery_confirmed_success", False))),
        "delivery_status": str(state.get("delivery_status", state.get("last_delivery_status", "delivery_unknown")) or "delivery_unknown"),
        "recipient_received_confirmed": bool(state.get("recipient_received_confirmed", False)),
        "dry_run": bool(state.get("email_dry_run", True)),
        "reason": str(state.get("email_result_reason", state.get("last_reason", "preview_only")) or "preview_only"),
        "provider": _current_email_provider(),
        "recipient_masked": "[REDACTED]",
        "subject": str(state.get("email_subject", "") or ""),
        "error_type": str(state.get("last_error_type", "") or "") or None,
        "error_class": str(state.get("last_error_class", state.get("last_error_type", "")) or "") or None,
        "sanitized_error": str(state.get("email_result_error", "") or "") or None,
        "timestamp_berlin": str(state.get("last_timestamp_berlin", "") or datetime.now(BERLIN_TZ).isoformat(timespec="seconds")),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
        temp_name = handle.name
    os.replace(temp_name, path)
    return path


def _build_email_content_hash(review: dict[str, Any], subject: str, body: str) -> str:
    del subject, body
    return build_decision_fingerprint(review)


def _email_delivery_mode(email_result: dict[str, Any]) -> str:
    if bool(email_result.get("provider_accepted", email_result.get("sent", False))):
        return "provider_accepted_delivery_unconfirmed"
    if bool(email_result.get("attempted", False)):
        return "send_failed"
    reason = str(email_result.get("reason", "") or "")
    if reason in {"already_sent_same_decision_today", "already_sent_today", "max_emails_per_day_reached"}:
        return "dedupe_blocked"
    if reason in {"dry_run", "preview_only_phase_gate"}:
        return "dry_run"
    return "preview_only"


def _build_email_dedupe_state(
    *,
    review: dict[str, Any],
    issues: dict[str, Any],
    settings: dict[str, Any],
    subject: str,
    body: str,
    email_result: dict[str, Any],
    previous_state: dict[str, Any],
) -> dict[str, Any]:
    generated_at = datetime.now(BERLIN_TZ)
    review_date = str(review.get("run_status", {}).get("review_date", generated_at.date()))
    recipient_text = _email_recipient_from_settings(settings) or str(email_result.get("recipient_for_dedupe", "") or "").strip()
    decision_fingerprint = build_decision_fingerprint(review, settings=settings, recipient=recipient_text)
    active_preview = dict(review.get("active_preview", {}) or {})
    safe_final_action = str(review.get("run_status", {}).get("final_action", "n/a"))
    active_preview_action = str(
        active_preview.get("active_preview_action", active_preview.get("final_action", "HOLD"))
    )
    current_body_hash = decision_fingerprint
    previous_last_review_date = str(previous_state.get("last_review_date", previous_state.get("last_sent_date", "")) or "")
    previous_last_fingerprint = str(previous_state.get("last_decision_fingerprint", previous_state.get("last_body_hash", "")) or "")
    previous_last_recipient = str(previous_state.get("last_recipient", recipient_text) or "")
    previous_accept_success = bool(previous_state.get("last_provider_accept_success", previous_state.get("last_send_success", False)))
    previous_last_safe_final_action = str(previous_state.get("last_safe_final_action", previous_state.get("final_action", "")) or "")
    previous_last_active_preview_action = str(previous_state.get("last_active_preview_action", "") or "")
    already_sent_today = bool(
        previous_accept_success
        and previous_last_review_date == review_date
        and previous_last_fingerprint == decision_fingerprint
        and previous_last_recipient == recipient_text
    )
    gate_status = get_email_gate_status(settings)
    recipient_count = len([item for item in recipient_text.split(",") if item.strip()])
    attempted = bool(email_result.get("attempted", False))
    provider_accepted = bool(email_result.get("provider_accepted", email_result.get("sent", False)))
    delivery_confirmed = bool(email_result.get("delivery_confirmed", False))
    reason = sanitize_for_output(email_result.get("reason", "preview_only"))
    previous_last_accepted_date = str(previous_state.get("last_accepted_at", "") or "")
    previous_last_sent_date = str(previous_state.get("last_sent_date", "") or "")
    previous_last_body_hash = str(previous_state.get("last_body_hash", "") or "")
    last_sent_date = review_date if provider_accepted else previous_last_sent_date
    last_body_hash = decision_fingerprint if provider_accepted else previous_last_body_hash
    last_subject = sanitize_for_output(subject) if provider_accepted else sanitize_for_output(previous_state.get("last_subject", ""))
    last_send_success = provider_accepted if attempted else bool(previous_state.get("last_send_success", False))
    last_provider_accept_success = provider_accepted or bool(previous_state.get("last_provider_accept_success", False))
    last_delivery_confirmed_success = delivery_confirmed or bool(previous_state.get("last_delivery_confirmed_success", False))
    last_attempted_at = str(email_result.get("timestamp_berlin") or generated_at.isoformat())
    last_accepted_at = last_attempted_at if provider_accepted else previous_last_accepted_date
    delivery_status = str(email_result.get("delivery_status") or reason or "delivery_unknown")
    provider_message_id = str(email_result.get("provider_message_id") or "")
    last_safe_final_action = (
        safe_final_action if provider_accepted or not previous_last_safe_final_action else previous_last_safe_final_action
    )
    last_active_preview_action = (
        active_preview_action if provider_accepted or not previous_last_active_preview_action else previous_last_active_preview_action
    )
    return {
        "review_date": review_date,
        "generated_at_berlin": generated_at.isoformat(),
        "review_status": str(issues.get("review_status", "REVIEW")),
        "final_action": safe_final_action,
        "safe_final_action": safe_final_action,
        "active_preview_action": active_preview_action,
        "first_blocker": str(issues.get("first_blocker", "none")),
        "email_subject": sanitize_for_output(subject),
        "current_decision_fingerprint": decision_fingerprint,
        "current_body_hash": current_body_hash,
        "body_sha256": current_body_hash,
        "dedupe_key": current_body_hash,
        "duplicate_today": already_sent_today,
        "already_sent_today": already_sent_today,
        "would_block_duplicate_real_send_today": already_sent_today,
        "real_email_send_allowed": bool(gate_status.get("real_email_send_allowed", False) and int(issues.get("hard_fail_count", 0)) == 0),
        "preview_only": not provider_accepted,
        "email_send_enabled": bool(settings.get("email_send_enabled", False)),
        "email_dry_run": bool(settings.get("email_dry_run", True)),
        "enable_email_notifications": bool(settings.get("enable_email_notifications", False)),
        "email_recipient_present": bool(recipient_text),
        "recipient_count": int(recipient_count),
        "max_emails_per_day": int(_parse_int(settings.get("max_emails_per_day"), 1)),
        "user_confirmed_email_phase": bool(settings.get("user_confirmed_email_phase", False)),
        "phase": str(settings.get("phase", "DAILY_REVIEW_PREVIEW")),
        "last_attempt_date": review_date if attempted else str(previous_state.get("last_attempt_date", "")),
        "last_review_date": review_date if provider_accepted else previous_last_review_date,
        "last_recipient": recipient_text if provider_accepted else previous_last_recipient,
        "last_decision_fingerprint": decision_fingerprint if provider_accepted else previous_last_fingerprint,
        "last_safe_final_action": last_safe_final_action,
        "last_active_preview_action": last_active_preview_action,
        "last_sent_date": last_sent_date,
        "last_subject": last_subject,
        "last_body_hash": last_body_hash,
        "last_send_success": last_send_success,
        "last_reason": reason,
        "last_error_type": sanitize_for_output(email_result.get("error_type") or ""),
        "last_error_class": sanitize_for_output(email_result.get("error_class") or email_result.get("error_type") or ""),
        "last_timestamp_berlin": str(email_result.get("timestamp_berlin") or generated_at.isoformat()),
        "last_decision_fingerprint_used_for_dedupe": decision_fingerprint,
        "last_provider_accept_success": bool(last_provider_accept_success),
        "last_delivery_confirmed_success": bool(last_delivery_confirmed_success),
        "last_provider_message_id": sanitize_for_output(provider_message_id),
        "last_attempted_at": last_attempted_at if attempted else str(previous_state.get("last_attempted_at", "")),
        "last_accepted_at": last_accepted_at,
        "last_delivery_status": delivery_status,
        "email_send_attempted": attempted,
        "email_result_sent": provider_accepted,
        "provider_accepted": provider_accepted,
        "provider_message_id": sanitize_for_output(provider_message_id),
        "provider_message_id_unavailable": bool(email_result.get("provider_message_id_unavailable", not bool(provider_message_id))),
        "delivery_confirmed": delivery_confirmed,
        "delivery_status": delivery_status,
        "recipient_received_confirmed": bool(email_result.get("recipient_received_confirmed", False)),
        "email_result_reason": reason,
        "email_result_error": sanitize_for_output(email_result.get("sanitized_error") or email_result.get("error") or ""),
        "gating_reasons": list(gate_status.get("blockers", [])),
    }


def send_daily_review_email_if_needed(
    review_or_diagnostics: dict[str, Any] | RunDiagnostics,
    output_dir: str | Path = "outputs",
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attempt to send the Daily Review email using the review subject/body."""

    review = _coerce_review_payload(review_or_diagnostics)
    output_path = Path(output_dir)
    issues = build_review_issues(review, output_dir=output_path)
    review.setdefault("run_status", {})
    review["run_status"]["first_blocker"] = issues.get("first_blocker", "none")
    review["run_status"]["all_blockers"] = issues.get("all_blockers", ["none"])
    review["run_status"]["review_status"] = issues.get("review_status", "REVIEW")
    effective_settings = {**load_daily_review_settings(), **dict(settings or {})}
    subject = build_daily_email_subject(review, issues).strip()
    body = build_daily_email_briefing(review, issues, settings=effective_settings)
    state_path = output_path / "last_email_state.json"
    previous_state = _read_last_email_state(state_path)
    review_date = str(review.get("run_status", {}).get("review_date", datetime.now(BERLIN_TZ).date()))
    max_emails_per_day = int(_parse_int(effective_settings.get("max_emails_per_day"), 1))
    configured_provider = (
        str(
            effective_settings.get("EMAIL_PROVIDER")
            or effective_settings.get("email_provider")
            or load_email_settings().get("EMAIL_PROVIDER", "brevo")
        )
        .strip()
        .lower()
        or "brevo"
    )
    email_settings = load_email_settings()
    recipient = str(
        effective_settings.get("email_recipient")
        or effective_settings.get("EMAIL_RECIPIENT", "")
        or effective_settings.get("EMAIL_TO", "")
        or email_settings.get("EMAIL_RECIPIENT", "")
        or email_settings.get("EMAIL_TO", "")
    ).strip()
    current_decision_fingerprint = build_decision_fingerprint(review, settings=effective_settings, recipient=recipient)
    last_review_date = str(previous_state.get("last_review_date", previous_state.get("last_sent_date", "")) or "")
    last_recipient = str(previous_state.get("last_recipient", recipient) or "")
    last_decision_fingerprint = str(
        previous_state.get("last_decision_fingerprint", previous_state.get("last_body_hash", "")) or ""
    )
    last_provider_accept_success = bool(previous_state.get("last_provider_accept_success", previous_state.get("last_send_success", False)))

    gate_status = get_email_gate_status(effective_settings)
    gate_blockers = list(gate_status.get("blockers", []))
    if int(issues.get("hard_fail_count", 0)) > 0:
        gate_blockers.append("DAILY_REVIEW_VALIDATION_HARD_FAIL")
    real_email_send_allowed = bool(gate_status.get("real_email_send_allowed", False) and int(issues.get("hard_fail_count", 0)) == 0)
    effective_gate_status = {
        **gate_status,
        "real_email_send_allowed": real_email_send_allowed,
        "blockers": gate_blockers,
        "reason": str(gate_status.get("reason", "preview_only")) if not real_email_send_allowed else "send_allowed",
    }

    if (
        last_provider_accept_success
        and last_review_date == review_date
        and last_recipient == recipient
        and last_decision_fingerprint == current_decision_fingerprint
    ):
        return {
            "attempted": False,
            "sent": False,
            "dry_run": bool(effective_settings.get("email_dry_run", True)),
            "reason": "already_sent_same_decision_today",
            "provider": configured_provider,
            "recipient_masked": "[REDACTED]",
            "recipient_for_dedupe": recipient,
            "subject": sanitize_for_output(subject),
            "error_type": None,
            "sanitized_error": None,
            "provider_accepted": False,
            "provider_message_id": None,
            "provider_message_id_unavailable": True,
            "delivery_confirmed": False,
            "delivery_status": "already_sent_same_decision_today",
            "recipient_received_confirmed": False,
            "timestamp_berlin": datetime.now(BERLIN_TZ).isoformat(timespec="seconds"),
        }
    if last_provider_accept_success and last_review_date == review_date and max_emails_per_day <= 1:
        return {
            "attempted": False,
            "sent": False,
            "dry_run": bool(effective_settings.get("email_dry_run", True)),
            "reason": "max_emails_per_day_reached",
            "provider": configured_provider,
            "recipient_masked": "[REDACTED]",
            "recipient_for_dedupe": recipient,
            "subject": sanitize_for_output(subject),
            "error_type": None,
            "sanitized_error": None,
            "provider_accepted": False,
            "provider_message_id": None,
            "provider_message_id_unavailable": True,
            "delivery_confirmed": False,
            "delivery_status": "max_emails_per_day_reached",
            "recipient_received_confirmed": False,
            "timestamp_berlin": datetime.now(BERLIN_TZ).isoformat(timespec="seconds"),
        }

    render_bundle = build_daily_review_render_bundle(
        review,
        issues,
        output_path,
        subject=subject,
        plain_text_body=body,
    )
    merged_email_settings = {
        **email_settings,
        **effective_settings,
        "EMAIL_TO": recipient,
        "EMAIL_RECIPIENT": recipient,
    }
    if "EMAIL_PROVIDER" not in merged_email_settings and "email_provider" in merged_email_settings:
        merged_email_settings["EMAIL_PROVIDER"] = merged_email_settings["email_provider"]
    if "EMAIL_FAKE_SEND_SUCCESS" not in merged_email_settings and "email_fake_send_success" in merged_email_settings:
        merged_email_settings["EMAIL_FAKE_SEND_SUCCESS"] = merged_email_settings["email_fake_send_success"]
    send_result = send_email_notification(
        subject=subject,
        body=body,
        recipient=recipient,
        dry_run=bool(effective_settings.get("email_dry_run", True)),
        gate_status=effective_gate_status,
        settings=merged_email_settings,
        html_body=str(render_bundle.get("html_text", "") or ""),
        attachments=[str(path) for path in list(render_bundle.get("attachment_paths", []) or [])],
    )
    send_result["recipient_for_dedupe"] = recipient

    email_state = _build_email_dedupe_state(
        review=review,
        issues=issues,
        settings=effective_settings,
        subject=subject,
        body=body,
        email_result=send_result,
        previous_state=previous_state,
    )
    _write_json(state_path, email_state)
    send_result["email_state_path"] = str(state_path)

    return send_result


def build_daily_review_validation_report(
    review: dict[str, Any],
    issues: dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> str:
    current_portfolio = review.get("current_portfolio", {})
    data_status = review.get("data_status", {})
    current_positions = list(review.get("current_positions", []))
    order_summary = review.get("order_summary", {})
    subject_text = build_daily_email_subject(review, issues).strip()
    body_text = build_daily_email_briefing(review, issues, settings=settings)
    contains_today_action = "TODAY'S ACTION" in body_text
    contains_data_status = "DATA STATUS" in body_text
    contains_safety_status = "SAFETY STATUS" in body_text
    contains_decision_summary = "DECISION SUMMARY" in body_text
    weights_sum_including_cash = _safe_float(current_portfolio.get("current_weights_sum_including_cash", 0.0))
    negative_market_value_count = sum(1 for row in current_positions if _safe_float(row.get("market_value_usd", 0.0)) < -1e-9)
    negative_shares_count = sum(1 for row in current_positions if _safe_float(row.get("current_shares", 0.0)) < -1e-9)
    cash_usd_value = current_portfolio.get("cash_usd", 0.0)
    try:
        cash_not_nan = not pd.isna(float(cash_usd_value))
    except (TypeError, ValueError):
        cash_not_nan = False
    lines = [
        "Daily Review Validation Report",
        "",
        f"review_status: {issues.get('review_status', 'REVIEW')}",
        f"hard_fail_count: {issues.get('hard_fail_count', 0)}",
        f"soft_warning_count: {issues.get('soft_warning_count', 0)}",
        f"info_count: {issues.get('info_count', 0)}",
        f"first_blocker: {issues.get('first_blocker', 'none')}",
        f"all_blockers: {' | '.join(map(str, issues.get('all_blockers', ['none'])))}",
        f"cash_usd: {_safe_float(current_portfolio.get('cash_usd', 0.0)):.2f}",
        f"invested_market_value_usd: {_safe_float(current_portfolio.get('invested_market_value_usd', 0.0)):.2f}",
        f"nav_usd: {_safe_float(current_portfolio.get('nav_usd', 0.0)):.2f}",
        f"nav_positive: {_bool_text(_safe_float(current_portfolio.get('nav_usd', 0.0)) > 0.0)}",
        f"cash_not_nan: {_bool_text(cash_not_nan)}",
        f"data_freshness_ok: {_bool_text(data_status.get('data_freshness_ok', False))}",
        f"missing_price_count: {len(list(data_status.get('missing_prices', []) or []))}",
        f"low_history_asset_count: {len(list(data_status.get('low_history_assets', []) or []))}",
        f"latest_price_date_mismatch_between_reports: {_bool_text(issues.get('report_consistency', {}).get('latest_price_date_mismatch', False))}",
        f"negative_market_value_count: {negative_market_value_count}",
        f"negative_shares_count: {negative_shares_count}",
        f"current_portfolio_100pct_cash: {_bool_text(current_portfolio.get('current_portfolio_100pct_cash', False))}",
        f"positions_count: {current_portfolio.get('positions_count', 0)}",
        f"current_weights_sum_without_cash: {_safe_float(current_portfolio.get('current_weights_sum_without_cash', 0.0)):.6f}",
        f"current_weights_sum_including_cash: {weights_sum_including_cash:.6f}",
        f"weights_including_cash_close_to_1: {_bool_text(abs(weights_sum_including_cash - 1.0) <= 0.01 if _safe_float(current_portfolio.get('nav_usd', 0.0)) > 0.0 else False)}",
        f"positions_present_not_100pct_cash: {_bool_text(not (int(current_portfolio.get('positions_count', 0) or 0) > 0 and _safe_bool(current_portfolio.get('current_portfolio_100pct_cash', False))))}",
        f"cash_before_orders: {_safe_float(order_summary.get('cash_before_orders', 0.0)):.2f}",
        f"cash_after_orders: {_safe_float(order_summary.get('cash_after_orders', 0.0)):.2f}",
        f"estimated_sell_value: {_safe_float(order_summary.get('estimated_sell_value', 0.0)):.2f}",
        f"estimated_buy_value: {_safe_float(order_summary.get('estimated_buy_value', 0.0)):.2f}",
        f"total_simulator_fees_usd: {_safe_float(order_summary.get('total_simulator_fees_usd', 0.0)):.2f}",
        f"modeled_transaction_costs_usd: {_safe_float(order_summary.get('modeled_transaction_costs_usd', 0.0)):.2f}",
        f"buy_count: {int(order_summary.get('buy_count', 0))}",
        f"sell_count: {int(order_summary.get('sell_count', 0))}",
        f"hold_count: {int(order_summary.get('hold_count', 0))}",
        f"order_count: {int(order_summary.get('order_count', 0))}",
        f"manual_eligible_order_count: {int(order_summary.get('manual_eligible_order_count', 0))}",
        f"cash_after_orders_non_negative: {_bool_text(order_summary.get('negative_cash_check', True))}",
        f"negative_cash_check: {_bool_text(order_summary.get('negative_cash_check', True))}",
        f"leverage_check: {_bool_text(order_summary.get('leverage_check', True))}",
        f"short_check: {_bool_text(order_summary.get('short_check', True))}",
        f"manual_orders_usable: {_bool_text(order_summary.get('manual_orders_usable', False))}",
        f"manual_orders_count: {int(order_summary.get('order_count', 0))}",
        f"preview_only: {_bool_text(review.get('preview_only', True))}",
        f"recommended_manual_file: outputs/manual_simulator_orders.csv",
        f"disallowed_manual_file: outputs/order_preview.csv",
        f"subject_non_empty: {_bool_text(bool(subject_text))}",
        f"body_non_empty: {_bool_text(bool(body_text.strip()))}",
        f"contains_today_action: {_bool_text(contains_today_action)}",
        f"contains_data_status: {_bool_text(contains_data_status)}",
        f"contains_safety_status: {_bool_text(contains_safety_status)}",
        f"contains_decision_summary: {_bool_text(contains_decision_summary)}",
        f"contains_final_action: {_bool_text('- final_action:' in body_text)}",
        f"contains_manual_order_count: {_bool_text('- manual_order_count:' in body_text)}",
        f"contains_data_source: {_bool_text('- data_source:' in body_text)}",
        f"contains_latest_price_date: {_bool_text('- latest_price_date:' in body_text)}",
        f"contains_cache_fallback_status: {_bool_text('- used_cache_fallback:' in body_text)}",
        f"contains_synthetic_data_status: {_bool_text('- synthetic_data:' in body_text)}",
        f"contains_operator_instruction: {_bool_text('- operator_instruction:' in body_text)}",
        "",
        "issue_table:",
    ]
    for item in issues.get("issue_table", []):
        lines.append(f"- {item.get('severity')}: {item.get('message')}")
    return "\n".join(lines).strip() + "\n"


def build_daily_portfolio_review_text(review: dict[str, Any], issues: dict[str, Any]) -> str:
    run_status = review.get("run_status", {})
    data_status = review.get("data_status", {})
    current_portfolio = review.get("current_portfolio", {})
    current_positions = list(review.get("current_positions", []))
    target_allocation = list(review.get("target_allocation", []))
    delta_transactions = list(review.get("delta_transactions", []))
    cost_edge = review.get("cost_edge", {})
    order_summary = review.get("order_summary", {})
    decision_context = review.get("decision_context", {})
    active_preview = dict(review.get("active_preview", {}) or {})
    hold_analysis = dict(review.get("hold_analysis", {}) or {})

    lines = [
        "Daily Portfolio Review",
        "",
        "1. Laufstatus",
        f"- review_date: {run_status.get('review_date', 'n/a')}",
        f"- review_time_berlin: {run_status.get('review_time_berlin', 'n/a')}",
        f"- current_date_berlin: {run_status.get('current_date_berlin', 'n/a')}",
        f"- current_time_berlin: {run_status.get('current_time_berlin', 'n/a')}",
        f"- is_project_trading_day: {_bool_text(run_status.get('is_project_trading_day', False))}",
        f"- within_allowed_window: {_bool_text(run_status.get('within_allowed_window', False))}",
        f"- execution_allowed_by_calendar: {_bool_text(run_status.get('execution_allowed_by_calendar', False))}",
        f"- final_action: {run_status.get('final_action', 'n/a')}",
        f"- execution_mode: {run_status.get('execution_mode', 'n/a')}",
        f"- first_blocker: {issues.get('first_blocker', 'none')}",
        f"- all_blockers: {' | '.join(map(str, issues.get('all_blockers', ['none'])))}",
        "",
        "2. Datenstatus",
        f"- data_source: {data_status.get('data_source', 'n/a')}",
        f"- cache_status: {data_status.get('cache_status', 'n/a')}",
        f"- synthetic_data: {_bool_text(data_status.get('synthetic_data', False))}",
        f"- used_cache_fallback: {_bool_text(data_status.get('used_cache_fallback', False))}",
        f"- latest_price_date: {data_status.get('latest_price_date', 'n/a')}",
        f"- staleness_days: {data_status.get('staleness_days', 'n/a')}",
        f"- data_freshness_ok: {_bool_text(data_status.get('data_freshness_ok', False))}",
        f"- live_data_error: {data_status.get('live_data_error', '') or 'none'}",
        f"- missing_prices: {', '.join(map(str, data_status.get('missing_prices', []))) if data_status.get('missing_prices') else 'none'}",
        f"- low_history_assets: {', '.join(map(str, data_status.get('low_history_assets', []))) if data_status.get('low_history_assets') else 'none'}",
        "",
        "3. Aktuelles Portfolio",
        f"- current_portfolio_source: {current_portfolio.get('current_portfolio_source', 'n/a')}",
        f"- positions_count: {current_portfolio.get('positions_count', 0)}",
        f"- cash_usd: {_safe_float(current_portfolio.get('cash_usd', 0.0)):.2f}",
        f"- invested_market_value_usd: {_safe_float(current_portfolio.get('invested_market_value_usd', 0.0)):.2f}",
        f"- nav_usd: {_safe_float(current_portfolio.get('nav_usd', 0.0)):.2f}",
        f"- current_portfolio_100pct_cash: {_bool_text(current_portfolio.get('current_portfolio_100pct_cash', False))}",
        f"- current_weights_sum_including_cash: {_safe_float(current_portfolio.get('current_weights_sum_including_cash', 0.0)):.6f}",
        f"- current_weights_sum_without_cash: {_safe_float(current_portfolio.get('current_weights_sum_without_cash', 0.0)):.6f}",
        f"- read_from_current_portfolio_csv: {_bool_text(current_portfolio.get('current_portfolio_source', '') == 'csv')}",
        "",
        "4. Aktuelle Kurse und Marktwerte je Position",
    ]
    if not current_positions:
        lines.append("- none")
    else:
        for row in current_positions:
            lines.append(
                f"- {row.get('ticker')}: shares={_safe_float(row.get('current_shares', 0.0)):.6f}, "
                f"latest_price={_safe_float(row.get('latest_price', 0.0)):.4f}, latest_price_date={row.get('latest_price_date', 'n/a')}, "
                f"market_value_usd={_safe_float(row.get('market_value_usd', 0.0)):.2f}, current_weight={_safe_float(row.get('current_weight', 0.0)):.6f}, "
                f"price_basis={row.get('price_basis', 'n/a')}, data_source={row.get('data_source', 'unknown')}, "
                f"stale_price_warning={_bool_text(row.get('stale_price_warning', False))}, data_warning={row.get('data_warning', 'none') or 'none'}"
            )
    lines.extend(["", "5. Modellziel / ideale Allokation"])
    if not target_allocation:
        lines.append("- none")
    else:
        for row in target_allocation:
            lines.append(
                f"- {row.get('ticker')}: target_weight={_safe_float(row.get('target_weight', 0.0)):.6f}, "
                f"target_shares={_safe_float(row.get('target_shares', 0.0)):.6f}, target_market_value_usd={_safe_float(row.get('target_market_value_usd', 0.0)):.2f}, "
                f"continuous_target_weight={_safe_float(row.get('continuous_target_weight', 0.0)):.6f}, abs_weight_drift={_safe_float(row.get('abs_weight_drift', 0.0)):.6f}, "
                f"latest_price={_safe_float(row.get('latest_price', 0.0)):.4f}"
            )
    lines.extend(["", "6. Delta-Transaktionen"])
    if not delta_transactions:
        lines.append("- none")
    else:
        for row in delta_transactions:
            lines.append(
                f"- {row.get('ticker')}: action={row.get('action')}, current_shares={_safe_float(row.get('current_shares', 0.0)):.6f}, "
                f"target_shares={_safe_float(row.get('target_shares', 0.0)):.6f}, order_shares={_safe_float(row.get('order_shares', 0.0)):.6f}, "
                f"estimated_price={_safe_float(row.get('estimated_price', 0.0)):.4f}, estimated_order_value={_safe_float(row.get('estimated_order_value', 0.0)):.2f}, "
                f"simulator_fee_usd={_safe_float(row.get('simulator_fee_usd', 0.0)):.2f}, modeled_transaction_cost_usd={_safe_float(row.get('modeled_transaction_cost_usd', 0.0)):.2f}, "
                f"preview_only={_bool_text(row.get('preview_only', True))}, not_executable={_bool_text(row.get('not_executable', True))}, "
                f"execution_block_reason={row.get('execution_block_reason', 'none')}"
            )
    lines.extend(
        [
            f"- cash_before_orders={_safe_float(order_summary.get('cash_before_orders', 0.0)):.2f}",
            f"- cash_after_orders={_safe_float(order_summary.get('cash_after_orders', 0.0)):.2f}",
            f"- estimated_sell_value={_safe_float(order_summary.get('estimated_sell_value', 0.0)):.2f}",
            f"- estimated_buy_value={_safe_float(order_summary.get('estimated_buy_value', 0.0)):.2f}",
            f"- total_simulator_fees_usd={_safe_float(order_summary.get('total_simulator_fees_usd', 0.0)):.2f}",
            f"- modeled_transaction_costs_usd={_safe_float(order_summary.get('modeled_transaction_costs_usd', 0.0)):.2f}",
            f"- buy_count={int(order_summary.get('buy_count', 0))}",
            f"- sell_count={int(order_summary.get('sell_count', 0))}",
            f"- hold_count={int(order_summary.get('hold_count', 0))}",
            f"- order_count={int(order_summary.get('order_count', 0))}",
            f"- manual_eligible_order_count={int(order_summary.get('manual_eligible_order_count', 0))}",
            f"- negative_cash_check={_bool_text(order_summary.get('negative_cash_check', True))}",
            f"- leverage_check={_bool_text(order_summary.get('leverage_check', True))}",
            f"- short_check={_bool_text(order_summary.get('short_check', True))}",
            f"- manual_orders_usable={_bool_text(order_summary.get('manual_orders_usable', False))}",
        ]
    )
    lines.extend(
        [
            "",
            "6b. Active Preview (nicht ausfuehrbar)",
            f"- active_preview_action: {active_preview.get('active_preview_action', 'HOLD')}",
            f"- active_preview_candidate: {active_preview.get('active_preview_candidate', 'HOLD_CURRENT')}",
            f"- active_preview_trade_now_edge: {_safe_float(active_preview.get('active_preview_trade_now_edge', 0.0)):.6f}",
            f"- active_preview_hurdle: {_safe_float(active_preview.get('active_preview_hurdle', 0.0)):.6f}",
            f"- active_preview_order_count: {int(active_preview.get('active_preview_order_count', 0) or 0)}",
            f"- active_preview_buy_count: {int(active_preview.get('active_preview_buy_count', 0) or 0)}",
            f"- active_preview_sell_count: {int(active_preview.get('active_preview_sell_count', 0) or 0)}",
            f"- active_preview_turnover: {_safe_float(active_preview.get('active_preview_turnover', 0.0)):.6f}",
            f"- active_preview_reason: {active_preview.get('active_preview_reason', 'n/a')}",
            "- active_preview_executable: false",
            "- active_preview_orders_file: outputs/active_preview_orders.csv",
            "- Active Preview Orders sind ANALYSIS ONLY und duerfen nicht automatisch eingereicht werden.",
        ]
    )
    lines.extend(
        [
            "",
            "Wichtig: outputs/manual_simulator_orders.csv ist die Datei fuer manuelle Simulatororders.",
            "Wichtig: outputs/order_preview.csv ist nur Research-/Backtest-Preview und darf nicht als manuelle Simulatorliste verwendet werden.",
            "",
            "7. Kosten / Edge",
            f"- simulator_fee_usd je Order: {_safe_float(cost_edge.get('simulator_fee_usd', 0.0)):.2f}",
            f"- total_simulator_fees_usd: {_safe_float(cost_edge.get('total_simulator_fees_usd', 0.0)):.2f}",
            f"- modeled_transaction_costs_usd: {_safe_float(cost_edge.get('modeled_transaction_costs_usd', 0.0)):.2f}",
            f"- modeled_transaction_costs_pct_nav: {_safe_float(cost_edge.get('modeled_transaction_costs_pct_nav', 0.0)):.6f}",
            f"- current_portfolio_score: {_safe_float(cost_edge.get('current_portfolio_score', 0.0)):.6f}",
            f"- target_score_before_costs: {_safe_float(cost_edge.get('target_score_before_costs', 0.0)):.6f}",
            f"- target_score_after_costs: {_safe_float(cost_edge.get('target_score_after_costs', 0.0)):.6f}",
            f"- delta_score_vs_current: {_safe_float(cost_edge.get('delta_score_vs_current', 0.0)):.6f}",
            f"- execution_buffer: {_safe_float(cost_edge.get('execution_buffer', 0.0)):.6f}",
            f"- model_uncertainty_buffer: {_safe_float(cost_edge.get('model_uncertainty_buffer', 0.0)):.6f}",
            f"- trade_now_edge: {_safe_float(cost_edge.get('trade_now_edge', 0.0)):.6f}",
            f"- cost_model_used: {cost_edge.get('cost_model_used', 'n/a')}",
            "",
            "8. Warum diese Entscheidung?",
            f"- Warum dieses Zielportfolio? {decision_context.get('why_this_target', 'n/a')}",
            f"- Warum nicht HOLD? {decision_context.get('why_not_hold', 'n/a')}",
            f"- Warum nicht CASH? {decision_context.get('why_not_cash', 'n/a')}",
            f"- Warum handeln oder warum nicht handeln? {decision_context.get('trade_decision_reason', 'n/a')}",
        ]
    )
    for line in list(decision_context.get("positive_drivers", [])):
        lines.append(f"- Positiver Treiber: {line}")
    for line in list(decision_context.get("negative_drivers", [])):
        lines.append(f"- Negativer Treiber: {line}")
    for line in list(decision_context.get("rejected_candidates", [])):
        lines.append(f"- Verworfen: {line}")
    lines.append(f"- Hauptblocker-Kategorie: {decision_context.get('main_blocker_category', 'n/a')}")
    lines.extend(["", "8b. Warum HOLD?"])
    why_hold_lines = list(hold_analysis.get("why_hold_lines", []) or [])
    if why_hold_lines:
        lines.extend([f"- {line}" for line in why_hold_lines])
    else:
        lines.append("- n/a")
    blocker_table = list(hold_analysis.get("blocker_table", []) or [])
    if blocker_table:
        lines.append("- Blocker | Typ | Einfluss auf Entscheidung | Erklaerung")
        for row in blocker_table:
            lines.append(
                f"- {row.get('blocker', 'n/a')} | {row.get('type', 'n/a')} | {row.get('impact', 'n/a')} | {row.get('explanation', 'n/a')}"
            )
    lines.extend(["", "9. Probleme / Warnungen / Fehler", "HARD_FAIL:"])
    if issues.get("hard_fails"):
        lines.extend([f"- {item.get('message')}" for item in issues.get("hard_fails", [])])
    else:
        lines.append("- none")
    lines.append("SOFT_WARNING:")
    if issues.get("soft_warnings"):
        lines.extend([f"- {item.get('message')}" for item in issues.get("soft_warnings", [])])
    else:
        lines.append("- none")
    lines.append("INFO:")
    if issues.get("infos"):
        lines.extend([f"- {item.get('message')}" for item in issues.get("infos", [])])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            f"first_blocker: {issues.get('first_blocker', 'none')}",
            f"all_blockers: {' | '.join(map(str, issues.get('all_blockers', ['none'])))}",
            f"hard_fail_count: {issues.get('hard_fail_count', 0)}",
            f"soft_warning_count: {issues.get('soft_warning_count', 0)}",
            f"info_count: {issues.get('info_count', 0)}",
            "issue_table:",
        ]
    )
    for item in issues.get("issue_table", []):
        lines.append(f"- {item.get('severity')}: {item.get('message')}")
    lines.extend(
        [
            "",
            "10. Manuelle naechste Aktion",
        ]
    )
    lines.extend([f"- {line}" for line in _build_manual_next_action(review, issues)])
    return "\n".join(lines).strip() + "\n"


def build_daily_portfolio_review_csv(review: dict[str, Any], issues: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(section: str, row_type: str, **values: Any) -> None:
        rows.append({"section": section, "row_type": row_type, **values})

    add_row("run_status", "summary", **review.get("run_status", {}))
    add_row("data_status", "summary", **review.get("data_status", {}))
    add_row("current_portfolio", "summary", **review.get("current_portfolio", {}))
    add_row("order_summary", "summary", **review.get("order_summary", {}))
    add_row("active_preview", "summary", **dict(review.get("active_preview", {}) or {}))
    for row in review.get("current_positions", []):
        add_row("current_position", "position", **row)
    for row in review.get("target_allocation", []):
        add_row("target_allocation", "target", **row)
    for row in review.get("delta_transactions", []):
        add_row("delta_transaction", "order", **row)
    add_row("cost_edge", "summary", **review.get("cost_edge", {}))
    for key in ("positive_drivers", "negative_drivers", "rejected_candidates"):
        for item in review.get("decision_context", {}).get(key, []):
            add_row("decision_context", key, message=item)
    for item in review.get("hold_analysis", {}).get("why_hold_lines", []):
        add_row("hold_analysis", "why_hold", message=item)
    for item in review.get("hold_analysis", {}).get("blocker_table", []):
        add_row(
            "hold_analysis",
            "blocker",
            blocker=item.get("blocker", ""),
            blocker_type=item.get("type", ""),
            decision_impact=item.get("impact", ""),
            message=item.get("explanation", ""),
        )
    for item in issues.get("issue_table", []):
        add_row("issues", item.get("severity", "INFO"), **item)
    for item in _build_manual_next_action(review, issues):
        add_row("manual_next_action", "instruction", message=item)
    return pd.DataFrame(rows)


def _fallback_review_from_diagnostics(diagnostics: RunDiagnostics | dict[str, Any]) -> dict[str, Any]:
    payload = diagnostics_to_dict(diagnostics)
    data_context = payload.get("data_context", {})
    model_context = payload.get("model_context", {})
    candidate_context = payload.get("candidate_context", {})
    execution_gate_context = payload.get("execution_gate_context", {})
    final_orders_summary = payload.get("final_orders_summary", {})
    review_now = datetime.now(BERLIN_TZ)

    return {
        "run_status": {
            "review_date": str(review_now.date()),
            "review_time_berlin": review_now.strftime("%H:%M:%S"),
            "current_date_berlin": data_context.get("current_date_berlin", str(review_now.date())),
            "current_time_berlin": data_context.get("current_time_berlin", review_now.strftime("%H:%M:%S")),
            "is_project_trading_day": _safe_bool(data_context.get("is_project_trading_day", False)),
            "within_allowed_window": _safe_bool(data_context.get("within_allowed_window", False)),
            "execution_allowed_by_calendar": _safe_bool(data_context.get("execution_allowed_by_calendar", False)),
            "final_action": payload.get("final_action", "PAUSE"),
            "execution_mode": payload.get("execution_mode", "blocked"),
            "gate_reason": payload.get("final_reason") or execution_gate_context.get("reason", ""),
        },
        "data_status": {
            "data_source": data_context.get("data_source", "unknown"),
            "cache_status": data_context.get("cache_status", "unknown"),
            "synthetic_data": _safe_bool(data_context.get("synthetic_data", False)),
            "used_cache_fallback": _safe_bool(data_context.get("used_cache_fallback", False)),
            "latest_price_date": data_context.get("latest_price_date", "n/a"),
            "staleness_days": data_context.get("staleness_days", "n/a"),
            "data_freshness_ok": _safe_bool(data_context.get("data_freshness_ok", False)),
            "live_data_error": data_context.get("live_data_error", ""),
            "missing_prices": [],
            "price_basis": data_context.get("price_basis", "adjusted_close_proxy"),
        },
        "current_portfolio": {
            "current_portfolio_source": model_context.get("current_portfolio", {}).get("source", "unknown"),
            "positions_count": _parse_int(model_context.get("current_portfolio", {}).get("positions_count", 0), 0),
            "cash_usd": _safe_float(model_context.get("current_portfolio", {}).get("actual_cash_value", 0.0)),
            "nav_usd": _safe_float(model_context.get("current_portfolio", {}).get("nav", 0.0)),
            "current_portfolio_100pct_cash": False,
            "current_weights_sum_including_cash": 0.0,
            "current_weights_sum_without_cash": 0.0,
            "parser_warnings": [],
            "parser_errors": [],
        },
        "current_positions": [],
        "target_allocation": [],
        "delta_transactions": [],
        "cost_edge": {
            "simulator_fee_usd": 0.0,
            "total_simulator_fees_usd": 0.0,
            "modeled_transaction_costs_usd": _safe_float(final_orders_summary.get("estimated_cost_usd", 0.0)),
            "modeled_transaction_costs_pct_nav": _safe_float(final_orders_summary.get("estimated_cost_pct_nav", 0.0)),
            "current_portfolio_score": 0.0,
            "target_score_before_costs": 0.0,
            "target_score_after_costs": _safe_float(candidate_context.get("net_robust_score", 0.0)),
            "delta_score_vs_current": _safe_float(candidate_context.get("delta_vs_hold", 0.0)),
            "execution_buffer": _safe_float(execution_gate_context.get("buffers", {}).get("execution_uncertainty_buffer", 0.0)),
            "model_uncertainty_buffer": _safe_float(execution_gate_context.get("buffers", {}).get("model_uncertainty_buffer", 0.0)),
            "trade_now_edge": _safe_float(candidate_context.get("trade_now_edge", execution_gate_context.get("trade_now_score", 0.0))),
            "cost_model_used": candidate_context.get("cost_model_used", "n/a"),
        },
        "order_summary": {
            "cash_before_orders": _safe_float(final_orders_summary.get("cash_before", 0.0)),
            "cash_after_orders": _safe_float(final_orders_summary.get("cash_after", 0.0)),
            "estimated_sell_value": 0.0,
            "estimated_buy_value": 0.0,
            "total_simulator_fees_usd": 0.0,
            "modeled_transaction_costs_usd": _safe_float(final_orders_summary.get("estimated_cost_usd", 0.0)),
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
            "order_count": int(final_orders_summary.get("order_count", 0) or 0),
            "manual_eligible_order_count": 0,
            "negative_cash_check": _safe_float(final_orders_summary.get("cash_after", 0.0)) >= -1e-9,
            "leverage_check": True,
            "short_check": True,
            "manual_orders_usable": False,
        },
        "decision_context": {
            "continuous_candidate": str(candidate_context.get("continuous_candidate", payload.get("continuous_candidate", "n/a"))),
            "final_discrete_candidate": str(candidate_context.get("selected_candidate", payload.get("selected_candidate", "n/a"))),
            "why_this_target": "Fallback review built from diagnostics because the full run did not finish cleanly.",
            "why_not_hold": "n/a",
            "why_not_cash": "n/a",
            "trade_decision_reason": payload.get("final_reason") or execution_gate_context.get("reason", "n/a"),
            "positive_drivers": [],
            "negative_drivers": [payload.get("final_reason") or execution_gate_context.get("reason", "n/a")],
            "rejected_candidates": [],
            "main_blocker_category": "data" if not _safe_bool(data_context.get("data_freshness_ok", False)) else "validation",
        },
        "pre_trade_validation_status": "PASS",
        "preview_only": True,
        "manual_orders_preview_ready": False,
        "cash_after_orders": _safe_float(final_orders_summary.get("cash_after", 0.0)),
        "main_daily_scope_differs": True,
        "exception_message": payload.get("errors", [{}])[0].get("exception_message", "") if payload.get("errors") else "",
    }


def _coerce_review_payload(review_or_diagnostics: dict[str, Any] | RunDiagnostics) -> dict[str, Any]:
    if isinstance(review_or_diagnostics, dict) and {"run_status", "data_status", "current_portfolio"}.issubset(review_or_diagnostics.keys()):
        return review_or_diagnostics
    payload = diagnostics_to_dict(review_or_diagnostics)
    review_payload = payload.get("model_context", {}).get("daily_review_payload")
    if isinstance(review_payload, dict):
        return review_payload
    return _fallback_review_from_diagnostics(review_or_diagnostics)


def write_daily_portfolio_review_outputs(
    review_or_diagnostics: dict[str, Any] | RunDiagnostics,
    output_dir: str | Path = "outputs",
    *,
    email_result: dict[str, Any] | None = None,
) -> dict[str, Path]:
    review = _coerce_review_payload(review_or_diagnostics)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    issues = build_review_issues(review, output_dir=output_path)
    if int(issues.get("hard_fail_count", 0)) > 0:
        review.setdefault("run_status", {})
        review["run_status"]["final_action"] = "BLOCK"
        review["run_status"]["execution_mode"] = "blocked"
        review["preview_only"] = True
        review["manual_orders_preview_ready"] = False
    review["run_status"]["first_blocker"] = issues.get("first_blocker", "none")
    review["run_status"]["all_blockers"] = issues.get("all_blockers", ["none"])
    review["run_status"]["review_status"] = issues.get("review_status", "REVIEW")
    settings = load_daily_review_settings()
    hold_analysis_bundle = build_hold_analysis_bundle(review, issues, output_path)
    review["hold_analysis"] = dict(hold_analysis_bundle.get("hold_analysis", {}) or {})
    manual_next_action = _build_manual_next_action(review, issues)

    txt_path = output_path / "daily_portfolio_review.txt"
    csv_path = output_path / "daily_portfolio_review.csv"
    subject_path = output_path / "daily_email_subject.txt"
    briefing_path = output_path / "daily_email_briefing.txt"
    portfolio_briefing_md_path = output_path / "daily_portfolio_briefing.md"
    portfolio_briefing_html_path = output_path / "daily_portfolio_briefing.html"
    latest_notification_path = output_path / "latest_email_notification.txt"
    safety_path = output_path / "email_safety_report.txt"
    delivery_diagnosis_path = output_path / "email_delivery_diagnosis_report.txt"
    validation_path = output_path / "daily_review_validation_report.txt"
    acceptance_path = output_path / "email_final_acceptance_report.txt"
    last_email_state_path = output_path / "last_email_state.json"
    hold_dominance_path = Path(hold_analysis_bundle.get("hold_dominance_analysis_path", output_path / "hold_dominance_analysis.txt"))
    hold_sensitivity_path = Path(hold_analysis_bundle.get("hold_sensitivity_report_path", output_path / "hold_sensitivity_report.txt"))
    decision_history_path = Path(hold_analysis_bundle.get("decision_history_path", output_path / "decision_history.csv"))

    subject_text = build_daily_email_subject(review, issues).strip()
    briefing_text = build_daily_email_briefing(review, issues, settings=settings)
    render_bundle = build_daily_review_render_bundle(
        review,
        issues,
        output_path,
        subject=subject_text,
        plain_text_body=briefing_text,
    )
    review["render_artifacts"] = {
        "daily_review_email_html": "outputs/daily_review_email.html",
        "daily_review_report_tex": "outputs/daily_review_report.tex",
        "daily_review_report_pdf": "outputs/daily_review_report.pdf" if render_bundle.get("pdf_path") else "not_available",
        "current_portfolio_allocation_png": "outputs/charts/current_portfolio_allocation.png",
        "current_vs_target_weights_png": "outputs/charts/current_vs_target_weights.png",
    }
    review["render_warnings"] = list(render_bundle.get("warnings", []) or [])
    previous_email_state = _read_last_email_state(last_email_state_path)
    email_result_payload = dict(
        email_result
        or {
            "attempted": False,
            "sent": False,
            "reason": "preview_only",
            "error": None,
            "error_type": None,
            "sanitized_error": None,
        }
    )
    email_state = _build_email_dedupe_state(
        review=review,
        issues=issues,
        settings=settings,
        subject=subject_text,
        body=briefing_text,
        email_result=email_result_payload,
        previous_state=previous_email_state,
    )
    portfolio_briefing_md = build_daily_portfolio_briefing_markdown(
        review,
        issues,
        settings=settings,
        email_result=email_result_payload,
        email_state=email_state,
    )
    portfolio_briefing_html = build_daily_portfolio_briefing_html(portfolio_briefing_md)
    latest_notification_text = (
        f"Subject: {subject_text}\n"
        f"Delivery mode: {_email_delivery_mode(email_result_payload)}\n"
        f"Review status: {issues.get('review_status', 'REVIEW')}\n"
        f"First blocker: {issues.get('first_blocker', 'none')}\n"
        f"Email attempted: {_bool_text(email_result_payload.get('attempted', False))}\n"
        f"Provider accepted: {_bool_text(email_result_payload.get('provider_accepted', email_result_payload.get('sent', False)))}\n"
        f"Delivery confirmed: {_bool_text(email_result_payload.get('delivery_confirmed', False))}\n"
        f"Delivery status: {sanitize_for_output(email_result_payload.get('delivery_status') or 'delivery_unknown')}\n"
        f"Email reason: {sanitize_for_output(email_result_payload.get('reason', 'preview_only'))}\n"
        f"Duplicate today: {_bool_text(email_state.get('duplicate_today', False))}\n"
        f"Decision fingerprint: {email_state.get('current_decision_fingerprint', '')}\n"
        f"Daily Portfolio Briefing Markdown: outputs/daily_portfolio_briefing.md\n"
        f"Daily Portfolio Briefing HTML: outputs/daily_portfolio_briefing.html\n"
        f"HTML preview file: outputs/daily_review_email.html\n"
        f"LaTeX report file: outputs/daily_review_report.tex\n"
        f"PDF report file: {'outputs/daily_review_report.pdf' if render_bundle.get('pdf_path') else 'not_available'}\n"
        "\n"
        f"{briefing_text}"
    )
    if review.get("render_warnings"):
        latest_notification_text += "\nRender warnings:\n"
        latest_notification_text += "\n".join(
            f"- {sanitize_for_output(item)}" for item in list(review.get("render_warnings", []))
        )
        latest_notification_text += "\n"

    txt_path.write_text(build_daily_portfolio_review_text(review, issues), encoding="utf-8")
    build_daily_portfolio_review_csv(review, issues).to_csv(csv_path, index=False)
    subject_path.write_text(sanitize_for_output(subject_text) + "\n", encoding="utf-8")
    briefing_path.write_text(sanitize_for_output(briefing_text), encoding="utf-8")
    portfolio_briefing_md_path.write_text(portfolio_briefing_md, encoding="utf-8")
    portfolio_briefing_html_path.write_text(sanitize_for_output(portfolio_briefing_html), encoding="utf-8")
    latest_notification_path.write_text(sanitize_for_output(latest_notification_text), encoding="utf-8")
    safety_text = build_email_safety_report(review, issues, settings, email_result=email_result_payload).rstrip() + "\n\n"
    safety_text += "last_email_state:\n"
    safety_text += f"- review_date: {email_state.get('review_date', 'n/a')}\n"
    safety_text += f"- decision_fingerprint: {email_state.get('current_decision_fingerprint', '')}\n"
    safety_text += f"- dedupe_key: {email_state.get('dedupe_key', '')}\n"
    safety_text += f"- duplicate_today: {_bool_text(email_state.get('duplicate_today', False))}\n"
    safety_text += f"- would_block_duplicate_real_send_today: {_bool_text(email_state.get('would_block_duplicate_real_send_today', False))}\n"
    safety_text += f"- last_attempt_date: {email_state.get('last_attempt_date', '') or 'n/a'}\n"
    safety_text += f"- last_sent_date: {email_state.get('last_sent_date', '') or 'n/a'}\n"
    safety_text += f"- last_send_success: {_bool_text(email_state.get('last_send_success', False))}\n"
    safety_text += f"- last_provider_accept_success: {_bool_text(email_state.get('last_provider_accept_success', False))}\n"
    safety_text += f"- last_delivery_confirmed_success: {_bool_text(email_state.get('last_delivery_confirmed_success', False))}\n"
    safety_text += f"- last_delivery_status: {email_state.get('last_delivery_status', 'delivery_unknown')}\n"
    safety_text += f"- recipient_count: {email_state.get('recipient_count', 0)}\n"
    safety_text += f"- email_result_reason: {email_state.get('email_result_reason', 'preview_only')}\n"
    safety_text += f"- latest_email_notification_file: outputs/latest_email_notification.txt\n"
    safety_text += f"- last_email_state_file: outputs/last_email_state.json\n"
    safety_path.write_text(sanitize_for_output(safety_text), encoding="utf-8")
    delivery_diagnosis_path.write_text(
        build_email_delivery_diagnosis_report(settings, email_result_payload, email_state),
        encoding="utf-8",
    )
    validation_path.write_text(
        sanitize_for_output(build_daily_review_validation_report(review, issues, settings=settings)),
        encoding="utf-8",
    )
    _write_json(last_email_state_path, email_state)
    acceptance_path.write_text(
        sanitize_for_output(
            build_email_final_acceptance_report(
                review,
                issues,
                settings=settings,
                email_result=email_result_payload,
                email_state=email_state,
            )
        ),
        encoding="utf-8",
    )

    return {
        "daily_portfolio_review_txt": txt_path,
        "daily_portfolio_review_csv": csv_path,
        "daily_email_subject": subject_path,
        "daily_email_briefing": briefing_path,
        "daily_portfolio_briefing_md": portfolio_briefing_md_path,
        "daily_portfolio_briefing_html": portfolio_briefing_html_path,
        "daily_review_email_html": output_path / "daily_review_email.html",
        "daily_review_report_tex": output_path / "daily_review_report.tex",
        "daily_review_report_pdf": output_path / "daily_review_report.pdf",
        "current_portfolio_allocation_png": output_path / "charts" / "current_portfolio_allocation.png",
        "current_vs_target_weights_png": output_path / "charts" / "current_vs_target_weights.png",
        "nav_cash_summary_png": output_path / "charts" / "nav_cash_summary.png",
        "risk_and_blockers_png": output_path / "charts" / "risk_and_blockers.png",
        "hold_dominance_analysis": hold_dominance_path,
        "hold_sensitivity_report": hold_sensitivity_path,
        "decision_history": decision_history_path,
        "latest_email_notification": latest_notification_path,
        "email_safety_report": safety_path,
        "email_delivery_diagnosis_report": delivery_diagnosis_path,
        "daily_review_validation_report": validation_path,
        "email_final_acceptance_report": acceptance_path,
        "last_email_state": last_email_state_path,
        "review_status": Path(str(issues.get("review_status", "REVIEW"))),
        "first_blocker": Path(str(issues.get("first_blocker", "none"))),
        "manual_next_action_count": Path(str(len(manual_next_action))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate Daily Review outputs from run diagnostics.")
    parser.add_argument(
        "--input-file",
        default="outputs/run_diagnostics.json",
        help="Path to the daily-bot diagnostics JSON to convert into Daily Review outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where Daily Review outputs should be written.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise SystemExit(f"Missing diagnostics input: {input_path}")
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not read diagnostics input {input_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Diagnostics input must be a JSON object: {input_path}")

    output_dir = Path(args.output_dir)
    preserved_email_state = _read_last_email_state(output_dir / "last_email_state.json")
    preserved_email_result = _email_result_from_state(preserved_email_state)

    paths = write_daily_portfolio_review_outputs(
        payload,
        output_dir=output_dir,
        email_result=preserved_email_result,
    )
    print("Daily portfolio review regenerated.")
    print(f"daily_portfolio_review_txt: {paths['daily_portfolio_review_txt']}")
    print(f"daily_email_subject: {paths['daily_email_subject']}")
    print(f"daily_email_briefing: {paths['daily_email_briefing']}")
    print(f"latest_email_notification: {paths['latest_email_notification']}")
    print(f"email_safety_report: {paths['email_safety_report']}")
    print(f"daily_review_validation_report: {paths['daily_review_validation_report']}")


if __name__ == "__main__":
    main()
