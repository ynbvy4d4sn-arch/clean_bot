"""Email notification utilities for the portfolio optimizer."""

from __future__ import annotations

import base64
from datetime import datetime
from email.message import EmailMessage
import json
import logging
from mimetypes import guess_type
import os
from pathlib import Path
import re
import smtplib
from urllib import error as urllib_error
from urllib import request as urllib_request
from zoneinfo import ZoneInfo

import pandas as pd

from config import STRONG_SIGNAL_THRESHOLD, get_email_gate_status

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional before dependencies are installed
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        """Fallback no-op when python-dotenv is unavailable."""

        return False


LOGGER = logging.getLogger(__name__)
BERLIN_TZ = ZoneInfo("Europe/Berlin")
SECRET_ENV_NAMES = (
    "SMTP_PASSWORD",
    "EMAIL_PASSWORD",
    "BREVO_API_KEY",
    "EMAIL_API_KEY",
    "SENDGRID_API_KEY",
    "MAILGUN_API_KEY",
    "API_KEY",
    "TOKEN",
    "SECRET",
)
SENSITIVE_PATTERN = re.compile(
    r"(?i)\b(password|smtp_password|email_password|api[_-]?key|token|secret|authorization)\b\s*[:=]\s*([^\s,;]+)"
)
BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-+/=]+")
AUTH_HEADER_PATTERN = re.compile(r"(?im)^(authorization\s*:\s*)(.+)$")
LONG_TOKEN_PATTERN = re.compile(r"\b(?=[A-Za-z0-9._\-+/]{24,}\b)(?=[A-Za-z0-9._\-+/]*\d)[A-Za-z0-9._\-+/]+\b")


def _parse_bool(value: object, default: bool = False) -> bool:
    """Parse bool-like values robustly."""

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


def _get_record_value(record: pd.Series | dict[str, object], key: str, default: object = None) -> object:
    """Read a value from a latest-record object with a default."""

    if isinstance(record, pd.Series):
        return record.get(key, default)
    return record.get(key, default)


def _parse_int(value: object, default: int) -> int:
    """Parse integers robustly and fall back to a default."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: object, default: float) -> float:
    """Parse floats robustly and fall back to a default."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _timestamp_berlin() -> str:
    return datetime.now(BERLIN_TZ).isoformat(timespec="seconds")


def _mask_email_address(email: str) -> str:
    text = str(email or "").strip()
    if "@" not in text:
        return "[REDACTED]"
    local_part, domain = text.split("@", 1)
    if len(local_part) <= 2:
        masked_local = local_part[:1] + "*"
    else:
        masked_local = local_part[:2] + "*" * max(len(local_part) - 2, 1)
    return f"{masked_local}@{domain}"


def sanitize_for_output(text: object) -> str:
    raw_text = str(text or "")
    sanitized = raw_text
    for env_name in SECRET_ENV_NAMES:
        secret_value = os.getenv(env_name, "").strip()
        if secret_value:
            sanitized = sanitized.replace(secret_value, "[REDACTED]")
    sanitized = SENSITIVE_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", sanitized)
    sanitized = AUTH_HEADER_PATTERN.sub(r"\1[REDACTED]", sanitized)
    sanitized = BEARER_PATTERN.sub("Bearer [REDACTED]", sanitized)
    sanitized = LONG_TOKEN_PATTERN.sub("[REDACTED]", sanitized)
    return sanitized


def load_email_settings() -> dict[str, object]:
    """Load email notification settings from .env with safe defaults."""

    env_path = Path(__file__).resolve().parent / ".env"
    env_loaded = load_dotenv(env_path, override=False) if env_path.exists() else False

    return {
        "ENV_PATH": str(env_path),
        "ENV_FILE_PRESENT": env_path.exists(),
        "ENV_FILE_LOADED": bool(env_loaded),
        "ENABLE_EMAIL_NOTIFICATIONS": _parse_bool(
            os.getenv("ENABLE_EMAIL_NOTIFICATIONS"),
            default=False,
        ),
        "EMAIL_PROVIDER": os.getenv("EMAIL_PROVIDER", "brevo").strip().lower() or "brevo",
        "EMAIL_FAKE_SEND_SUCCESS": _parse_bool(os.getenv("EMAIL_FAKE_SEND_SUCCESS"), default=False),
        "BREVO_API_KEY": os.getenv("BREVO_API_KEY", "") or os.getenv("EMAIL_API_KEY", ""),
        "SMTP_HOST": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "SMTP_PORT": _parse_int(os.getenv("SMTP_PORT"), 587),
        "SMTP_USER": os.getenv("SMTP_USER", "").strip() or os.getenv("SMTP_USERNAME", "").strip(),
        "SMTP_USERNAME": os.getenv("SMTP_USERNAME", "").strip() or os.getenv("SMTP_USER", "").strip(),
        "SMTP_PASSWORD": os.getenv("SMTP_PASSWORD", "") or os.getenv("EMAIL_PASSWORD", ""),
        "EMAIL_USE_SSL": _parse_bool(os.getenv("EMAIL_USE_SSL"), default=False),
        "EMAIL_USE_STARTTLS": _parse_bool(os.getenv("EMAIL_USE_STARTTLS"), default=True),
        "EMAIL_FROM": (
            os.getenv("EMAIL_FROM", "").strip()
            or os.getenv("EMAIL_SENDER", "").strip()
            or os.getenv("SMTP_USER", "").strip()
            or os.getenv("SMTP_USERNAME", "").strip()
        ),
        "EMAIL_SENDER": (
            os.getenv("EMAIL_SENDER", "").strip()
            or os.getenv("EMAIL_FROM", "").strip()
            or os.getenv("SMTP_USER", "").strip()
            or os.getenv("SMTP_USERNAME", "").strip()
        ),
        "EMAIL_TO": os.getenv("EMAIL_TO", "").strip() or os.getenv("EMAIL_RECIPIENT", "").strip(),
        "EMAIL_RECIPIENT": os.getenv("EMAIL_RECIPIENT", "").strip() or os.getenv("EMAIL_TO", "").strip(),
        "EMAIL_SUBJECT_PREFIX": os.getenv("EMAIL_SUBJECT_PREFIX", "[Portfolio Optimizer]"),
        "SEND_WEEKLY_SUMMARY": _parse_bool(
            os.getenv("SEND_WEEKLY_SUMMARY"),
            default=True,
        ),
        "SEND_DAILY_HOLD_WAIT_EMAILS": _parse_bool(
            os.getenv("SEND_DAILY_HOLD_WAIT_EMAILS"),
            default=False,
        ),
        "STRONG_SIGNAL_THRESHOLD": _parse_float(
            os.getenv("STRONG_SIGNAL_THRESHOLD"),
            STRONG_SIGNAL_THRESHOLD,
        ),
    }


def _smtp_settings_complete(settings: dict[str, object]) -> bool:
    """Return whether the minimum SMTP fields required for sending are present."""

    host = str(settings.get("SMTP_HOST", "")).strip()
    sender = str(settings.get("EMAIL_FROM", "") or settings.get("EMAIL_SENDER", "")).strip()
    recipients = str(settings.get("EMAIL_TO", "") or settings.get("EMAIL_RECIPIENT", "")).strip()
    username = str(settings.get("SMTP_USER", "") or settings.get("SMTP_USERNAME", "")).strip()
    password = str(settings.get("SMTP_PASSWORD", ""))

    if not host or not sender or not recipients:
        return False
    if bool(username) and not bool(password):
        return False
    return True


def _brevo_settings_complete(settings: dict[str, object]) -> bool:
    """Return whether the minimum Brevo API fields required for sending are present."""

    api_key = str(settings.get("BREVO_API_KEY", "") or settings.get("EMAIL_API_KEY", "")).strip()
    sender = str(settings.get("EMAIL_FROM", "") or settings.get("EMAIL_SENDER", "")).strip()
    recipients = str(settings.get("EMAIL_TO", "") or settings.get("EMAIL_RECIPIENT", "")).strip()
    return bool(api_key and sender and recipients)


def _email_provider_settings_complete(settings: dict[str, object]) -> bool:
    """Return whether the configured provider has enough data for a send attempt."""

    provider = str(settings.get("EMAIL_PROVIDER", "brevo")).strip().lower() or "brevo"
    if provider == "smtp":
        return _smtp_settings_complete(settings)
    if provider == "brevo":
        return _brevo_settings_complete(settings)
    if provider == "fake":
        recipients = str(settings.get("EMAIL_TO", "") or settings.get("EMAIL_RECIPIENT", "")).strip()
        sender = str(settings.get("EMAIL_FROM", "") or settings.get("EMAIL_SENDER", "")).strip()
        return bool(sender and recipients)
    return False


def smtp_settings_complete(settings: dict[str, object]) -> bool:
    """Public wrapper retained for legacy SMTP-specific checks."""

    return _smtp_settings_complete(settings)


def email_settings_complete(settings: dict[str, object]) -> bool:
    """Public wrapper for provider-aware completeness checks."""

    return _email_provider_settings_complete(settings)


def format_percent(x: object, digits: int = 2) -> str:
    """Format a numeric value as percent, handling missing values gracefully."""

    try:
        value = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(value):
        return "n/a"
    return f"{value:.{digits}%}"


def format_number(x: object, digits: int = 4) -> str:
    """Format a numeric value with fixed digits, handling missing values gracefully."""

    try:
        value = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def format_weights(weights: pd.Series, min_weight: float = 0.005) -> str:
    """Format a weight vector as a readable text block."""

    filtered = weights.astype(float).copy()
    filtered = filtered[filtered.abs() >= min_weight].sort_values(ascending=False)
    if filtered.empty:
        return "  (no weights above threshold)"
    return "\n".join(f"  {ticker}: {weight:.2%}" for ticker, weight in filtered.items())


def should_send_email(latest_record: pd.Series | dict[str, object], settings: dict[str, object]) -> bool:
    """Return whether a notification should be sent for the latest decision."""

    if not bool(settings.get("ENABLE_EMAIL_NOTIFICATIONS", False)):
        return False
    if not bool(settings.get("ENV_FILE_PRESENT", False)):
        return False
    if not _email_provider_settings_complete(settings):
        return False

    decision = str(_get_record_value(latest_record, "decision", ""))
    emergency_condition = bool(_get_record_value(latest_record, "emergency_condition", False))
    weekly_rebalance_day = bool(_get_record_value(latest_record, "weekly_rebalance_day", False))
    net_benefit = float(_get_record_value(latest_record, "net_benefit", 0.0) or 0.0)
    realized_turnover = float(_get_record_value(latest_record, "realized_turnover", 0.0) or 0.0)
    strong_signal_threshold = float(settings.get("STRONG_SIGNAL_THRESHOLD", STRONG_SIGNAL_THRESHOLD))

    if decision in {"DE_RISK", "FULL_REBALANCE", "PAUSE"}:
        return True
    if decision == "PARTIAL_REBALANCE" and emergency_condition:
        return True
    if weekly_rebalance_day and bool(settings.get("SEND_WEEKLY_SUMMARY", True)):
        return True
    if net_benefit > strong_signal_threshold:
        return True
    if realized_turnover > 0.40 and net_benefit > 0.0:
        return True
    if bool(settings.get("SEND_DAILY_HOLD_WAIT_EMAILS", False)):
        return True
    return False


def build_email_subject(latest_record: pd.Series | dict[str, object], settings: dict[str, object]) -> str:
    """Build the subject line for the latest decision email."""

    prefix = str(settings.get("EMAIL_SUBJECT_PREFIX", "[Portfolio Optimizer]")).strip()
    decision = str(_get_record_value(latest_record, "decision", ""))
    weekly_rebalance_day = bool(_get_record_value(latest_record, "weekly_rebalance_day", False))

    if decision == "DE_RISK":
        suffix = "URGENT DE_RISK recommended"
    elif decision == "FULL_REBALANCE":
        suffix = "FULL_REBALANCE recommended"
    elif decision == "PARTIAL_REBALANCE":
        suffix = "PARTIAL_REBALANCE recommended"
    elif decision == "PAUSE":
        suffix = "PAUSE - Data or model issue"
    elif weekly_rebalance_day:
        suffix = "Weekly Portfolio Check"
    else:
        suffix = "Daily Check"

    return f"{prefix} {suffix}"


def build_email_body(
    latest_record: pd.Series | dict[str, object],
    latest_weights: pd.Series,
    latest_target_weights: pd.Series,
) -> str:
    """Build the plain-text email body for the latest optimizer decision."""

    decision = str(_get_record_value(latest_record, "decision", "UNKNOWN"))
    explanation_map = {
        "DE_RISK": "Risk controls were breached. Reduce risk and move toward the defensive allocation.",
        "FULL_REBALANCE": "The target portfolio meaningfully improved the objective and cleared execution hurdles.",
        "PARTIAL_REBALANCE": "The target improved the objective, but a partial step is sufficient today.",
        "WAIT": "The signal was not strong enough outside the preferred rebalance cadence.",
        "HOLD": "The target did not beat the cost and buffer hurdle after penalties.",
        "PAUSE": "Data quality or model diagnostics indicated that no action should be taken.",
    }
    suggested_action_map = {
        "DE_RISK": "Move capital toward the defensive portfolio template.",
        "FULL_REBALANCE": "Move from current weights to target weights.",
        "PARTIAL_REBALANCE": "Move partway from current weights toward target weights.",
        "WAIT": "Do not trade today. Re-evaluate on the next decision day.",
        "HOLD": "Keep current allocation unchanged.",
        "PAUSE": "Hold current allocation and investigate the model/data state.",
    }

    date_value = _get_record_value(latest_record, "date", "")
    try:
        date_text = str(pd.Timestamp(date_value).date())
    except Exception:
        date_text = str(date_value)

    turnover_value = _get_record_value(latest_record, "turnover", None)
    if turnover_value is None or pd.isna(turnover_value):
        turnover_value = _get_record_value(latest_record, "realized_turnover", None)

    body = (
        f"Date: {date_text}\n"
        f"Decision: {decision}\n"
        f"Suggested Action: {suggested_action_map.get(decision, 'Review the latest portfolio state.')}\n"
        f"Net Benefit: {format_number(_get_record_value(latest_record, 'net_benefit'))}\n"
        f"Delta Score: {format_number(_get_record_value(latest_record, 'delta_score'))}\n"
        f"Score Current: {format_number(_get_record_value(latest_record, 'score_current'))}\n"
        f"Score Target: {format_number(_get_record_value(latest_record, 'score_target'))}\n"
        f"Turnover: {format_percent(turnover_value)}\n"
        f"Estimated Cost: {format_number(_get_record_value(latest_record, 'estimated_cost'))}\n"
        f"Buffer: {format_number(_get_record_value(latest_record, 'buffer'))}\n"
        f"Target Vol: {format_percent(_get_record_value(latest_record, 'target_vol'))}\n"
        f"Weekly Rebalance Day: {bool(_get_record_value(latest_record, 'weekly_rebalance_day', False))}\n"
        f"Emergency Condition: {bool(_get_record_value(latest_record, 'emergency_condition', False))}\n"
        f"Risk Gate Failed: {bool(_get_record_value(latest_record, 'risk_gate_failed', False))}\n"
        f"Risk State: {_get_record_value(latest_record, 'risk_state', 'n/a')}\n"
        "\n"
        "Current Weights:\n"
        f"{format_weights(latest_weights)}\n"
        "\n"
        "Target Weights:\n"
        f"{format_weights(latest_target_weights)}\n"
        "\n"
        f"Explanation: {explanation_map.get(decision, 'No explanation available.')}\n"
        f"Execution Mode: {_get_record_value(latest_record, 'execution_mode', 'n/a')}\n"
        f"Execution Message: {_get_record_value(latest_record, 'execution_message', 'n/a')}\n"
    )
    return body


def send_email(subject: str, body: str, settings: dict[str, object]) -> bool:
    """Send an email using SMTP with STARTTLS. Returns True on success."""

    return bool(send_email_with_result(subject=subject, body=body, settings=settings).get("sent", False))


def send_email_notification(
    subject: str,
    body: str,
    recipient: str,
    *,
    dry_run: bool,
    gate_status: dict[str, object],
    settings: dict[str, object] | None = None,
    html_body: str | None = None,
    attachments: list[str | Path] | None = None,
) -> dict[str, object]:
    """Send one email via Brevo API, SMTP, or fake provider with a fail-closed structured result."""

    effective_settings = {**load_email_settings(), **dict(settings or {})}
    provider = str(effective_settings.get("EMAIL_PROVIDER", "brevo")).strip().lower() or "brevo"
    recipient_text = str(recipient or "").strip()
    html_body_text = str(html_body or "").strip() or None
    attachment_paths = [Path(item) for item in list(attachments or []) if str(item)]
    result: dict[str, object] = {
        "attempted": False,
        "sent": False,
        "provider_accepted": False,
        "provider_message_id": None,
        "provider_message_id_unavailable": True,
        "delivery_confirmed": False,
        "delivery_status": "not_attempted",
        "recipient_received_confirmed": False,
        "dry_run": bool(dry_run),
        "reason": "not_attempted",
        "provider": provider,
        "recipient_masked": _mask_email_address(recipient_text),
        "subject": sanitize_for_output(subject),
        "error_type": None,
        "error_class": None,
        "sanitized_error": None,
        "timestamp_berlin": _timestamp_berlin(),
    }

    if not bool(gate_status.get("real_email_send_allowed", False)):
        gate_reason = str(gate_status.get("reason", "preview_only") or "preview_only")
        result["reason"] = "preview_only_phase_gate" if gate_reason == "preview_only" else "blocked_by_phase_gate"
        result["delivery_status"] = result["reason"]
        return result

    if bool(dry_run):
        result["reason"] = "preview_only_phase_gate"
        result["delivery_status"] = "preview_only_phase_gate"
        return result

    if not recipient_text:
        result["reason"] = "provider_rejected"
        result["delivery_status"] = "recipient_missing"
        return result

    if provider == "fake":
        result["attempted"] = True
        if _parse_bool(effective_settings.get("EMAIL_FAKE_SEND_SUCCESS"), default=False):
            result["sent"] = True
            result["provider_accepted"] = True
            result["provider_message_id_unavailable"] = True
            result["delivery_confirmed"] = False
            result["delivery_status"] = "fake_send_success"
            result["reason"] = "fake_send_success"
            return result
        result["reason"] = "fake_send_failure"
        result["delivery_status"] = "fake_send_failure"
        result["error_type"] = "FakeEmailProviderError"
        result["error_class"] = "FakeEmailProviderError"
        result["sanitized_error"] = "Fake email provider simulated a delivery failure."
        return result

    if provider == "brevo":
        sender = str(effective_settings.get("EMAIL_FROM", "") or effective_settings.get("EMAIL_SENDER", "")).strip()
        api_key = str(effective_settings.get("BREVO_API_KEY", "") or effective_settings.get("EMAIL_API_KEY", "")).strip()
        if not sender:
            result["reason"] = "provider_rejected"
            result["delivery_status"] = "sender_missing"
            return result
        if not api_key:
            result["reason"] = "provider_rejected"
            result["delivery_status"] = "brevo_api_key_missing"
            return result

        payload = {
            "sender": {"email": sender},
            "to": [{"email": recipient_text}],
            "subject": subject,
            "textContent": body,
        }
        if html_body_text:
            payload["htmlContent"] = html_body_text
        if attachment_paths:
            payload["attachment"] = []
            for path in attachment_paths:
                if not path.exists() or path.stat().st_size <= 0:
                    continue
                payload["attachment"].append(
                    {
                        "name": sanitize_for_output(path.name),
                        "content": base64.b64encode(path.read_bytes()).decode("ascii"),
                    }
                )
        request = urllib_request.Request(
            "https://api.brevo.com/v3/smtp/email",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "accept": "application/json",
                "api-key": api_key,
                "content-type": "application/json",
            },
            method="POST",
        )

        result["attempted"] = True
        try:
            with urllib_request.urlopen(request, timeout=30) as response:
                status_code = int(getattr(response, "status", response.getcode()))
                response_text = response.read().decode("utf-8", errors="replace").strip()
            if status_code >= 400:
                raise RuntimeError(f"Brevo API returned HTTP {status_code}: {response_text}")
            provider_message_id = None
            if response_text:
                try:
                    response_payload = json.loads(response_text)
                    provider_message_id = response_payload.get("messageId") or response_payload.get("message_id")
                except json.JSONDecodeError:
                    provider_message_id = None
            result["sent"] = True
            result["provider_accepted"] = True
            result["provider_message_id"] = sanitize_for_output(provider_message_id or "")
            result["provider_message_id_unavailable"] = not bool(provider_message_id)
            result["delivery_confirmed"] = False
            result["delivery_status"] = "provider_accepted_delivery_unconfirmed"
            result["reason"] = "provider_accepted_delivery_unconfirmed"
            return result
        except urllib_error.HTTPError as exc:  # pragma: no cover - provider dependent
            response_text = exc.read().decode("utf-8", errors="replace").strip()
            result["sent"] = False
            result["reason"] = "provider_rejected"
            result["delivery_status"] = "provider_rejected"
            result["error_type"] = type(exc).__name__
            result["error_class"] = type(exc).__name__
            result["sanitized_error"] = sanitize_for_output(
                f"Brevo API HTTP {exc.code}: {response_text or exc.reason}"
            )
            LOGGER.warning("Email send failed but the run will continue: %s", result["sanitized_error"])
            return result
        except Exception as exc:  # pragma: no cover - network/provider dependent
            result["sent"] = False
            result["reason"] = "delivery_unknown"
            result["delivery_status"] = "delivery_unknown"
            result["error_type"] = type(exc).__name__
            result["error_class"] = type(exc).__name__
            result["sanitized_error"] = sanitize_for_output(str(exc))
            LOGGER.warning("Email send failed but the run will continue: %s", result["sanitized_error"])
            return result

    if provider != "smtp":
        result["reason"] = "provider_rejected"
        result["delivery_status"] = "unsupported_provider"
        result["error_type"] = "UnsupportedEmailProvider"
        result["error_class"] = "UnsupportedEmailProvider"
        result["sanitized_error"] = sanitize_for_output(f"Unsupported EMAIL_PROVIDER={provider}")
        return result

    smtp_settings = {
        **effective_settings,
        "EMAIL_TO": recipient_text,
    }
    if not _smtp_settings_complete(smtp_settings):
        result["reason"] = "provider_rejected"
        result["delivery_status"] = "smtp_incomplete"
        return result

    host = str(smtp_settings.get("SMTP_HOST", "")).strip()
    port = int(smtp_settings.get("SMTP_PORT", 587))
    sender = str(smtp_settings.get("EMAIL_FROM", "") or smtp_settings.get("EMAIL_SENDER", "")).strip()
    username = str(smtp_settings.get("SMTP_USER", "") or smtp_settings.get("SMTP_USERNAME", "")).strip()
    password = str(smtp_settings.get("SMTP_PASSWORD", ""))
    use_ssl = _parse_bool(smtp_settings.get("EMAIL_USE_SSL"), default=False)
    use_starttls = _parse_bool(smtp_settings.get("EMAIL_USE_STARTTLS"), default=not use_ssl)

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient_text
    message.set_content(body)
    if html_body_text:
        message.add_alternative(html_body_text, subtype="html")
    for path in attachment_paths:
        if not path.exists() or path.stat().st_size <= 0:
            continue
        mime_type, _ = guess_type(path.name)
        if mime_type:
            maintype, subtype = mime_type.split("/", 1)
        else:
            maintype, subtype = "application", "octet-stream"
        message.add_attachment(
            path.read_bytes(),
            maintype=maintype,
            subtype=subtype,
            filename=sanitize_for_output(path.name),
        )

    result["attempted"] = True
    try:
        smtp_factory = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
        with smtp_factory(host, port, timeout=30) as server:
            if not use_ssl and use_starttls:
                if callable(getattr(server, "ehlo", None)):
                    server.ehlo()
                server.starttls()
                if callable(getattr(server, "ehlo", None)):
                    server.ehlo()
            if username:
                server.login(username, password)
            server.send_message(message)
        result["sent"] = True
        result["provider_accepted"] = True
        result["provider_message_id_unavailable"] = True
        result["delivery_confirmed"] = False
        result["delivery_status"] = "provider_accepted_delivery_unconfirmed"
        result["reason"] = "provider_accepted_delivery_unconfirmed"
        return result
    except Exception as exc:  # pragma: no cover - network/provider dependent
        result["sent"] = False
        result["reason"] = "smtp_auth_failed" if isinstance(exc, smtplib.SMTPAuthenticationError) else "delivery_unknown"
        result["delivery_status"] = result["reason"]
        result["error_type"] = type(exc).__name__
        result["error_class"] = type(exc).__name__
        result["sanitized_error"] = sanitize_for_output(str(exc))
        LOGGER.warning("Email send failed but the run will continue: %s", result["sanitized_error"])
        return result


def send_email_with_result(subject: str, body: str, settings: dict[str, object]) -> dict[str, object]:
    """Send an email using SMTP with STARTTLS and return a structured result."""

    effective_settings = {**load_email_settings(), **dict(settings or {})}
    recipients_raw = str(
        effective_settings.get("EMAIL_TO", "") or effective_settings.get("EMAIL_RECIPIENT", "")
    ).strip()
    gate_status = get_email_gate_status(
        {
            **effective_settings,
            "EMAIL_RECIPIENT": str(
                effective_settings.get("EMAIL_RECIPIENT", "") or effective_settings.get("EMAIL_TO", "")
            ).strip(),
        }
    )
    result = send_email_notification(
        subject=subject,
        body=body,
        recipient=recipients_raw,
        dry_run=_parse_bool(effective_settings.get("EMAIL_DRY_RUN"), default=False),
        gate_status=gate_status,
        settings=effective_settings,
    )
    return {
        "sent": bool(result.get("sent", False)),
        "reason": str(result.get("reason", "smtp_failed")),
        "error": result.get("sanitized_error"),
        "attempted": bool(result.get("attempted", False)),
        "provider_accepted": bool(result.get("provider_accepted", result.get("sent", False))),
        "provider_message_id": result.get("provider_message_id"),
        "provider_message_id_unavailable": bool(result.get("provider_message_id_unavailable", True)),
        "delivery_confirmed": bool(result.get("delivery_confirmed", False)),
        "delivery_status": str(result.get("delivery_status", "delivery_unknown")),
        "error_type": result.get("error_type"),
        "error_class": result.get("error_class"),
        "provider": result.get("provider"),
    }


def write_latest_notification(body: str, output_dir: str | Path = "outputs") -> Path:
    """Write the latest email body to outputs/latest_email_notification.txt."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "latest_email_notification.txt"
    file_path.write_text(body, encoding="utf-8")
    return file_path


def send_email_alert_if_needed(
    latest_record: pd.Series | dict[str, object],
    latest_weights: pd.Series,
    latest_target_weights: pd.Series,
    output_dir: str | Path = "outputs",
) -> bool:
    """Write the latest notification text and send an email if conditions are met."""

    settings = load_email_settings()

    try:
        body = build_email_body(
            latest_record=latest_record,
            latest_weights=latest_weights,
            latest_target_weights=latest_target_weights,
        )
    except Exception as exc:
        body = (
            "Email notification body could not be built.\n"
            f"Reason: {exc}\n"
            "The run continued without sending an email."
        )
        LOGGER.warning("Email body generation failed; writing fallback notification text: %s", exc)

    write_latest_notification(body=body, output_dir=output_dir)

    if not bool(settings.get("ENV_FILE_PRESENT", False)):
        LOGGER.info("Email not sent because no .env file is present.")
        return False

    if not bool(settings.get("ENABLE_EMAIL_NOTIFICATIONS", False)):
        LOGGER.info("Email not sent because ENABLE_EMAIL_NOTIFICATIONS is false.")
        return False

    if not _email_provider_settings_complete(settings):
        LOGGER.warning("Email not sent because provider settings are incomplete.")
        return False

    if not should_send_email(latest_record=latest_record, settings=settings):
        LOGGER.info("Email notification not sent because trigger conditions were not met.")
        return False

    subject = build_email_subject(latest_record=latest_record, settings=settings)
    return send_email(subject=subject, body=body, settings=settings)
