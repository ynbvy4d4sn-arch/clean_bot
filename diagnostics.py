"""Central diagnostics helpers for daily-bot and research runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path
import re
import traceback as traceback_module
from typing import Any
from uuid import uuid4

import pandas as pd


LOGGER = logging.getLogger(__name__)
SENSITIVE_KEY_PATTERN = re.compile(
    r"(password|secret|token|api[_-]?key|credential|smtp_password|investopedia_password)",
    re.IGNORECASE,
)
SENSITIVE_TEXT_PATTERN = re.compile(
    r"((?:smtp_password|password|secret|token|api[_-]?key)\s*[:=]\s*)([^,\s]+)",
    re.IGNORECASE,
)


@dataclass
class RunDiagnostics:
    """Structured diagnostics payload for one bot or research run."""

    run_id: str
    run_timestamp_utc: str
    local_date: str
    signal_date: str | None = None
    execution_date: str | None = None
    mode: str = "daily_bot"
    dry_run: bool = True
    data_context: dict[str, Any] = field(default_factory=dict)
    data_quality: dict[str, Any] = field(default_factory=dict)
    model_context: dict[str, Any] = field(default_factory=dict)
    optimizer_context: dict[str, Any] = field(default_factory=dict)
    candidate_context: dict[str, Any] = field(default_factory=dict)
    execution_gate_context: dict[str, Any] = field(default_factory=dict)
    execution_mode: str = "order_preview_only"
    final_action: str = "PAUSE"
    selected_candidate: str = "HOLD"
    final_reason: str = ""
    stage_log: list[dict[str, Any]] = field(default_factory=list)
    final_orders_summary: dict[str, Any] = field(default_factory=dict)
    rejected_orders: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    performance_flags: list[dict[str, Any]] = field(default_factory=list)
    suspected_root_causes: list[dict[str, Any]] = field(default_factory=list)
    suggested_codex_tasks: list[dict[str, Any]] = field(default_factory=list)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_timestamp_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    try:
        return str(pd.Timestamp(value).date())
    except Exception:
        return str(value)


def _redact_string(text: str) -> str:
    return SENSITIVE_TEXT_PATTERN.sub(r"\1[REDACTED]", text)


def _safe_jsonable(value: Any, *, key_name: str | None = None) -> Any:
    if key_name and SENSITIVE_KEY_PATTERN.search(str(key_name)):
        return "[REDACTED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        try:
            return pd.Timestamp(value).isoformat()
        except Exception:
            return str(value)
    if isinstance(value, pd.Series):
        return {str(k): _safe_jsonable(v, key_name=str(k)) for k, v in value.to_dict().items()}
    if isinstance(value, pd.Index):
        return [_safe_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        try:
            return [_safe_jsonable(item) for item in value.to_dict(orient="records")]
        except Exception:
            return _redact_string(value.to_json(orient="records", default_handler=str))
    if is_dataclass(value):
        return _safe_jsonable(asdict(value))
    if isinstance(value, dict):
        return {
            str(k): _safe_jsonable(v, key_name=str(k))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_jsonable(item) for item in value]
    if hasattr(value, "_asdict"):
        try:
            return _safe_jsonable(value._asdict())
        except Exception:
            return _redact_string(str(value))
    if hasattr(value, "__dict__") and not isinstance(value, type):
        try:
            return _safe_jsonable(vars(value))
        except Exception:
            return _redact_string(str(value))
    return _redact_string(str(value))


def diagnostics_to_dict(diagnostics: RunDiagnostics | dict[str, Any]) -> dict[str, Any]:
    """Convert diagnostics objects into a sanitized dict representation."""

    if isinstance(diagnostics, RunDiagnostics):
        payload = asdict(diagnostics)
    else:
        payload = dict(diagnostics)
    return _safe_jsonable(payload)


def _default_prefix(mode: str) -> str:
    lowered = str(mode).strip().lower()
    return "research_" if "research" in lowered or "backtest" in lowered or lowered == "main" else ""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    frame = pd.DataFrame(rows, columns=fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _append_log_entry(container: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    try:
        container.append(_safe_jsonable(entry))
    except Exception as exc:  # pragma: no cover - fail-closed fallback
        container.append(
            {
                "timestamp_utc": _utc_now().isoformat(),
                "severity": "ERROR",
                "message": f"Diagnostics logging fallback engaged: {exc}",
            }
        )


def create_run_diagnostics(run_id: str | None = None, mode: str = "daily_bot") -> RunDiagnostics:
    """Create a fresh diagnostics container with safe defaults."""

    now = _utc_now()
    return RunDiagnostics(
        run_id=run_id or f"{mode}-{uuid4().hex[:12]}",
        run_timestamp_utc=now.isoformat(),
        local_date=str(now.astimezone().date()),
        mode=mode,
        dry_run=True,
    )


def log_error(
    diagnostics: RunDiagnostics,
    module: str,
    function: str,
    exception: Exception,
    severity: str = "ERROR",
    stage: str | None = None,
    extra: Any = None,
) -> RunDiagnostics:
    """Append a structured error record without crashing diagnostics handling."""

    try:
        entry = {
            "timestamp_utc": _utc_now().isoformat(),
            "severity": severity,
            "stage": stage or "",
            "module": module,
            "function": function,
            "exception_type": type(exception).__name__,
            "exception_message": _redact_string(str(exception)),
            "traceback": _redact_string(
                "".join(traceback_module.format_exception(type(exception), exception, exception.__traceback__))
            ),
            "extra_json": _safe_jsonable(extra),
        }
        _append_log_entry(diagnostics.errors, entry)
    except Exception as logging_exc:  # pragma: no cover - last line of defense
        LOGGER.warning("Diagnostics log_error fallback failed: %s", logging_exc)
    return diagnostics


def log_warning(
    diagnostics: RunDiagnostics,
    module: str,
    message: str,
    severity: str = "WARN",
    stage: str | None = None,
    extra: Any = None,
) -> RunDiagnostics:
    """Append a structured warning record without raising."""

    try:
        entry = {
            "timestamp_utc": _utc_now().isoformat(),
            "severity": severity,
            "stage": stage or "",
            "module": module,
            "message": _redact_string(str(message)),
            "extra_json": _safe_jsonable(extra),
        }
        _append_log_entry(diagnostics.warnings, entry)
    except Exception as logging_exc:  # pragma: no cover - last line of defense
        LOGGER.warning("Diagnostics log_warning fallback failed: %s", logging_exc)
    return diagnostics


def log_stage(
    diagnostics: RunDiagnostics,
    stage: str,
    status: str = "INFO",
    extra: Any = None,
) -> RunDiagnostics:
    """Append a lightweight pipeline-stage marker."""

    _append_log_entry(
        diagnostics.stage_log,
        {
            "timestamp_utc": _utc_now().isoformat(),
            "stage": str(stage),
            "status": str(status),
            "extra_json": _safe_jsonable(extra),
        },
    )
    return diagnostics


def log_data_context(
    diagnostics: RunDiagnostics,
    prices: pd.DataFrame | None = None,
    freshness: dict[str, Any] | None = None,
    attrs: dict[str, Any] | None = None,
    extra: Any = None,
) -> RunDiagnostics:
    """Capture sanitized data-source and freshness context."""

    try:
        payload: dict[str, Any] = {}
        attrs = attrs or (dict(getattr(prices, "attrs", {})) if prices is not None else {})
        freshness = freshness or {}
        if prices is not None and not prices.empty:
            payload["tickers_loaded"] = list(map(str, prices.columns.tolist()))
        payload.update(attrs)
        payload.update(freshness)
        if extra is not None:
            payload["extra"] = extra
        diagnostics.data_context.update(_safe_jsonable(payload))
    except Exception as exc:  # pragma: no cover - defensive
        log_warning(diagnostics, "diagnostics", f"log_data_context failed: {exc}", stage="data_context")
    return diagnostics


def log_data_quality(diagnostics: RunDiagnostics, data_quality_report: Any) -> RunDiagnostics:
    """Capture sanitized data-quality information."""

    try:
        payload = _safe_jsonable(data_quality_report)
        if isinstance(payload, dict):
            diagnostics.data_quality.update(payload)
        else:
            diagnostics.data_quality["report"] = payload
    except Exception as exc:  # pragma: no cover - defensive
        log_warning(diagnostics, "diagnostics", f"log_data_quality failed: {exc}", stage="data_quality")
    return diagnostics


def log_optimizer_result(diagnostics: RunDiagnostics, optimizer_result: Any) -> RunDiagnostics:
    """Capture optimizer and model context."""

    try:
        payload = _safe_jsonable(optimizer_result)
        if isinstance(payload, dict):
            diagnostics.optimizer_context.update(payload)
        else:
            diagnostics.optimizer_context["result"] = payload
    except Exception as exc:  # pragma: no cover - defensive
        log_warning(diagnostics, "diagnostics", f"log_optimizer_result failed: {exc}", stage="optimizer")
    return diagnostics


def log_candidate_selection(diagnostics: RunDiagnostics, selection_result: Any) -> RunDiagnostics:
    """Capture final candidate selection context."""

    try:
        payload = _safe_jsonable(selection_result)
        if isinstance(payload, dict):
            diagnostics.candidate_context.update(payload)
        else:
            diagnostics.candidate_context["selection"] = payload
        selected_candidate = diagnostics.candidate_context.get("best_discrete_candidate_name") or diagnostics.candidate_context.get("selected_candidate")
        if selected_candidate:
            diagnostics.selected_candidate = str(selected_candidate)
    except Exception as exc:  # pragma: no cover - defensive
        log_warning(diagnostics, "diagnostics", f"log_candidate_selection failed: {exc}", stage="candidate_selection")
    return diagnostics


def log_execution_gate(diagnostics: RunDiagnostics, execution_gate_result: Any) -> RunDiagnostics:
    """Capture gate output and derived execution-mode hints."""

    try:
        payload = _safe_jsonable(execution_gate_result)
        if isinstance(payload, dict):
            diagnostics.execution_gate_context.update(payload)
            if "execution_mode" in payload:
                diagnostics.execution_mode = str(payload["execution_mode"])
        else:
            diagnostics.execution_gate_context["gate"] = payload
    except Exception as exc:  # pragma: no cover - defensive
        log_warning(diagnostics, "diagnostics", f"log_execution_gate failed: {exc}", stage="execution_gate")
    return diagnostics


def log_final_action(
    diagnostics: RunDiagnostics,
    action: str,
    selected_candidate: str | None = None,
    reason: str | None = None,
) -> RunDiagnostics:
    """Persist final action level fields."""

    diagnostics.final_action = str(action)
    if selected_candidate:
        diagnostics.selected_candidate = str(selected_candidate)
    if reason is not None:
        diagnostics.final_reason = _redact_string(str(reason))
    return diagnostics


def log_rejected_order(
    diagnostics: RunDiagnostics,
    ticker: str,
    side: str,
    reason: str,
    extra: Any = None,
) -> RunDiagnostics:
    """Append a rejected-order row."""

    entry = {
        "timestamp_utc": _utc_now().isoformat(),
        "ticker": str(ticker),
        "side": str(side),
        "reason": _redact_string(str(reason)),
        "extra_json": _safe_jsonable(extra),
    }
    _append_log_entry(diagnostics.rejected_orders, entry)
    return diagnostics


def log_performance_flag(
    diagnostics: RunDiagnostics,
    flag_name: str,
    severity: str,
    message: str,
    extra: Any = None,
) -> RunDiagnostics:
    """Append a performance or behavior flag."""

    entry = {
        "timestamp_utc": _utc_now().isoformat(),
        "flag_name": str(flag_name),
        "severity": str(severity),
        "message": _redact_string(str(message)),
        "suggested_action": "",
        "files_likely_involved": [],
        "extra_json": _safe_jsonable(extra),
    }
    _append_log_entry(diagnostics.performance_flags, entry)
    return diagnostics


def add_suspected_root_cause(
    diagnostics: RunDiagnostics,
    message: str,
    files: list[str] | None = None,
    severity: str = "INFO",
) -> RunDiagnostics:
    """Record a suspected root cause for later Codex triage."""

    _append_log_entry(
        diagnostics.suspected_root_causes,
        {
            "severity": str(severity),
            "message": _redact_string(str(message)),
            "files": [str(item) for item in (files or [])],
        },
    )
    return diagnostics


def add_suggested_codex_task(
    diagnostics: RunDiagnostics,
    task: Any,
    files: list[str] | None = None,
    priority: str = "MEDIUM",
) -> RunDiagnostics:
    """Record a concrete follow-up task for manual Codex use."""

    payload: dict[str, Any]
    if isinstance(task, dict):
        payload = dict(task)
    else:
        payload = {"task": str(task)}
    payload.setdefault("task", str(task))
    payload.setdefault("files", [str(item) for item in (files or [])])
    payload.setdefault("priority", priority)
    _append_log_entry(diagnostics.suggested_codex_tasks, payload)
    return diagnostics


def detect_performance_flags(
    diagnostics: RunDiagnostics,
    recent_results: list[dict[str, Any]] | None = None,
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Derive lightweight behavioral flags from diagnostics context."""

    thresholds = thresholds or {}
    recent_results = recent_results or []
    flags: list[dict[str, Any]] = []

    def _record(
        flag_name: str,
        severity: str,
        message: str,
        suggested_action: str,
        files_likely_involved: list[str],
    ) -> None:
        entry = {
            "timestamp_utc": _utc_now().isoformat(),
            "flag_name": flag_name,
            "severity": severity,
            "message": message,
            "suggested_action": suggested_action,
            "files_likely_involved": files_likely_involved,
            "extra_json": {},
        }
        flags.append(entry)

    order_count = int(diagnostics.final_orders_summary.get("order_count", 0) or 0)
    turnover = float(diagnostics.final_orders_summary.get("turnover", 0.0) or 0.0)
    estimated_cost = float(
        diagnostics.final_orders_summary.get("estimated_cost", diagnostics.final_orders_summary.get("estimated_cost_pct_nav", 0.0))
        or 0.0
    )
    model_confidence = diagnostics.model_context.get("model_confidence_score")
    data_quality_score = diagnostics.data_quality.get("global_data_quality_score")

    if order_count > int(thresholds.get("too_many_trades", 8)):
        _record(
            "too_many_trades",
            "WARN",
            f"Order count {order_count} exceeds the configured diagnostic threshold.",
            "Review marginal-order filters and max_orders_per_day settings.",
            ["daily_bot.py", "discrete_portfolio_optimizer.py", "trade_sizing.py"],
        )
    if turnover > float(thresholds.get("high_turnover", 0.35)):
        _record(
            "high_turnover",
            "WARN",
            f"Turnover {turnover:.2%} looks elevated for one run.",
            "Inspect turnover gates and buffer logic before allowing execution.",
            ["optimizer.py", "discrete_portfolio_optimizer.py", "execution_gate.py"],
        )
    if estimated_cost > float(thresholds.get("high_cost_drag", 0.005)):
        _record(
            "high_cost_drag",
            "WARN",
            f"Estimated trading cost burden {estimated_cost:.4f} is elevated.",
            "Re-check order costs, min-order filters and candidate selection.",
            ["transaction_costs.py", "discrete_portfolio_optimizer.py", "execution_gate.py"],
        )
    if model_confidence is not None and float(model_confidence) < float(thresholds.get("low_model_confidence", 0.5)):
        _record(
            "low_model_confidence",
            "WARN",
            f"Model confidence is only {float(model_confidence):.3f}.",
            "Prefer HOLD/PAUSE and inspect forecast and governance diagnostics.",
            ["forecast_3m.py", "model_governance.py", "daily_bot.py"],
        )
    if data_quality_score is not None and float(data_quality_score) < float(thresholds.get("low_data_quality", 0.8)):
        _record(
            "low_data_quality",
            "WARN",
            f"Global data quality score is only {float(data_quality_score):.3f}.",
            "Inspect data freshness, missing prices and filtering diagnostics.",
            ["data.py", "data_quality.py", "daily_bot.py"],
        )
    if len(diagnostics.rejected_orders) >= int(thresholds.get("many_rejected_orders", 3)):
        _record(
            "many_rejected_orders",
            "WARN",
            f"{len(diagnostics.rejected_orders)} orders were rejected in the final preview.",
            "Inspect min-order, cash and constraint filters for noisy trade generation.",
            ["pre_trade_validation.py", "discrete_portfolio_optimizer.py"],
        )
    if "duplicate" in diagnostics.final_reason.lower() or any("duplicate" in str(item).lower() for item in diagnostics.warnings):
        _record(
            "duplicate_order_blocked",
            "INFO",
            "A duplicate order signature appears to have been blocked.",
            "Verify daily state accumulation and order-signature generation.",
            ["daily_bot.py"],
        )

    action_history = [str(item.get("final_action", "")).upper() for item in recent_results if isinstance(item, dict)]
    if action_history.count("PAUSE") >= int(thresholds.get("repeated_pause", 3)):
        _record(
            "repeated_pause",
            "WARN",
            "Recent runs paused repeatedly.",
            "Investigate recurring data, validation or gate failures.",
            ["daily_bot.py", "data.py", "execution_gate.py"],
        )
    if action_history.count("HOLD") >= int(thresholds.get("repeated_hold", 5)):
        _record(
            "repeated_hold",
            "INFO",
            "Recent runs selected HOLD repeatedly.",
            "Confirm the strategy still produces differentiated candidates after costs.",
            ["robust_scorer.py", "discrete_portfolio_optimizer.py"],
        )
    if sum(bool(item.get("used_cache_fallback")) for item in recent_results if isinstance(item, dict)) >= int(thresholds.get("cache_fallback_repeated", 2)):
        _record(
            "cache_fallback_repeated",
            "WARN",
            "Recent runs relied on cache fallback repeatedly.",
            "Inspect live data connectivity and freshness gating.",
            ["data.py", "daily_bot.py"],
        )
    if sum(not bool(item.get("data_freshness_ok", True)) for item in recent_results if isinstance(item, dict)) >= int(thresholds.get("stale_data_repeated", 2)):
        _record(
            "stale_data_repeated",
            "WARN",
            "Recent runs saw stale data repeatedly.",
            "Inspect freshness policy and live refresh behavior.",
            ["data.py", "calendar_utils.py", "daily_bot.py"],
        )

    for entry in flags:
        _append_log_entry(diagnostics.performance_flags, entry)
    return diagnostics.performance_flags


def write_run_diagnostics(diagnostics: RunDiagnostics, output_dir: str | Path = "outputs") -> None:
    """Persist diagnostics artifacts without letting logging failures crash the caller."""

    output_path = Path(output_dir)
    prefix = _default_prefix(diagnostics.mode)
    payload = diagnostics_to_dict(diagnostics)

    try:
        _write_json(output_path / f"{prefix}run_diagnostics.json", payload)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Could not write run diagnostics JSON: %s", exc)

    error_fields = [
        "timestamp_utc",
        "severity",
        "stage",
        "module",
        "function",
        "exception_type",
        "exception_message",
        "traceback",
        "extra_json",
    ]
    warning_fields = [
        "timestamp_utc",
        "severity",
        "stage",
        "module",
        "message",
        "extra_json",
    ]
    rejected_fields = [
        "timestamp_utc",
        "ticker",
        "side",
        "reason",
        "extra_json",
    ]
    performance_fields = [
        "timestamp_utc",
        "flag_name",
        "severity",
        "message",
        "suggested_action",
        "files_likely_involved",
        "extra_json",
    ]

    try:
        _write_csv(output_path / f"{prefix}error_log.csv", diagnostics.errors, error_fields)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Could not write error log CSV: %s", exc)
    try:
        _write_csv(output_path / f"{prefix}warnings_log.csv", diagnostics.warnings, warning_fields)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Could not write warnings log CSV: %s", exc)
    if diagnostics.rejected_orders:
        try:
            _write_csv(output_path / f"{prefix}rejected_orders_report.csv", diagnostics.rejected_orders, rejected_fields)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Could not write rejected orders CSV: %s", exc)
    if diagnostics.performance_flags:
        try:
            _write_csv(output_path / f"{prefix}performance_flags.csv", diagnostics.performance_flags, performance_fields)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Could not write performance flags CSV: %s", exc)
