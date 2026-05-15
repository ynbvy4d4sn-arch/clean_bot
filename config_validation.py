"""Central validation for the parameter dictionary built by config.build_params()."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from asset_universe import EXPECTED_CASH_TICKER
from config import build_params, get_email_gate_status, review_email_send_allowed


REQUIRED_PARAM_KEYS = {
    "tickers",
    "cash_ticker",
    "start_date",
    "horizon_days",
    "cov_window",
    "asset_max_weights",
    "group_map",
    "group_limits",
    "equity_like_groups",
    "defensive_groups",
    "max_equity_like_total_normal",
    "max_equity_like_total_risk_off",
    "min_defensive_weight_normal",
    "min_defensive_weight_risk_off",
    "crypto_max_normal",
    "crypto_max_risk_off",
    "max_turnover",
    "cost_rate",
    "dry_run",
    "enable_external_broker",
}


def validate_config(params: dict[str, Any]) -> dict[str, Any]:
    """Validate the central params dictionary and return warnings plus errors."""

    warnings: list[str] = []
    errors: list[str] = []

    missing = sorted(REQUIRED_PARAM_KEYS - set(params))
    if missing:
        errors.append("Missing required config keys: " + ", ".join(missing))
        return {"ok": False, "warnings": warnings, "errors": errors}

    tickers = [str(ticker) for ticker in params.get("tickers", [])]
    asset_caps = pd.Series(params.get("asset_max_weights", {}), dtype=float)
    group_map = pd.Series(params.get("group_map", {}), dtype=object)
    group_limits = {str(key): float(value) for key, value in dict(params.get("group_limits", {})).items()}

    if not tickers:
        errors.append("No active tickers are configured.")

    invalid_caps = asset_caps[(asset_caps <= 0.0) | (asset_caps > 1.0)]
    if not invalid_caps.empty:
        errors.append("Asset max weights must lie in (0, 1].")

    invalid_group_limits = [
        group for group, limit in group_limits.items() if limit <= 0.0 or limit > 1.0
    ]
    if invalid_group_limits:
        errors.append("Group limits must lie in (0, 1].")

    cash_ticker = params.get("cash_ticker")
    if cash_ticker is not None and str(cash_ticker) != EXPECTED_CASH_TICKER:
        warnings.append(
            f"Configured cash ticker is {cash_ticker}; expected default is {EXPECTED_CASH_TICKER}. "
            "This can be valid when a fallback cash proxy is used."
        )

    enabled_cash_tickers = [
        ticker for ticker in tickers if str(group_map.get(ticker, "")) == "cash"
    ]
    if len(enabled_cash_tickers) > 1:
        errors.append("More than one active cash asset is present in the current ticker set.")

    max_turnover = float(params.get("max_turnover", 0.0))
    if not 0.0 <= max_turnover <= 2.0:
        errors.append("max_turnover must be between 0 and 2.")

    if float(params.get("cost_rate", 0.0)) < 0.0:
        errors.append("cost_rate must be non-negative.")
    objective = str(params.get("optimization_objective", "robust_score")).strip().lower()
    if objective not in {
        "robust_score",
        "scenario_mixture_sharpe",
        "robust_scenario_sharpe_objective",
        "direct_scenario_sharpe",
        "scenario_weighted_rf_sharpe",
    }:
        errors.append(
            "optimization_objective must be robust_score, scenario_mixture_sharpe, "
            "robust_scenario_sharpe_objective, direct_scenario_sharpe, or "
            "scenario_weighted_rf_sharpe."
        )
    probabilities = pd.Series(params.get("scenario_probabilities", {}), dtype=float)
    if probabilities.empty or float(probabilities.clip(lower=0.0).sum()) <= 0.0:
        errors.append("scenario_probabilities must contain at least one positive probability.")

    for threshold_name in (
        "base_buffer",
        "vol_buffer_multiplier",
        "min_turnover_to_trade",
        "partial_threshold",
        "partial_fraction",
        "strong_signal_threshold",
        "default_commission_per_trade_usd",
        "default_bps_per_turnover",
        "default_slippage_bps",
        "default_spread_bps",
        "default_market_impact_bps",
        "min_order_value_usd",
        "cash_buffer_usd",
    ):
        if float(params.get(threshold_name, 0.0)) < 0.0:
            errors.append(f"{threshold_name} must be non-negative.")

    if int(params.get("horizon_days", 0)) <= 0:
        errors.append("HORIZON_DAYS must be positive.")
    if int(params.get("cov_window", 0)) < 63:
        errors.append("COV_WINDOW must be at least 63 trading days.")

    try:
        start_date = pd.Timestamp(params.get("start_date")).date()
        if start_date >= date.today():
            errors.append("START_DATE must be before today.")
    except Exception as exc:
        errors.append(f"START_DATE is invalid: {exc}")

    max_equity_like_normal = float(params.get("max_equity_like_total_normal", 0.0))
    max_equity_like_risk_off = float(params.get("max_equity_like_total_risk_off", 0.0))
    min_defensive_normal = float(params.get("min_defensive_weight_normal", 0.0))
    min_defensive_risk_off = float(params.get("min_defensive_weight_risk_off", 0.0))
    if max_equity_like_normal <= 0.0 or max_equity_like_normal > 1.0:
        errors.append("max_equity_like_total_normal must lie in (0, 1].")
    if max_equity_like_risk_off <= 0.0 or max_equity_like_risk_off > 1.0:
        errors.append("max_equity_like_total_risk_off must lie in (0, 1].")
    if min_defensive_normal < 0.0 or min_defensive_normal > 1.0:
        errors.append("min_defensive_weight_normal must lie in [0, 1].")
    if min_defensive_risk_off < 0.0 or min_defensive_risk_off > 1.0:
        errors.append("min_defensive_weight_risk_off must lie in [0, 1].")
    if max_equity_like_normal + min_defensive_normal > 1.000001:
        warnings.append("Normal-state equity-like and defensive constraints leave almost no slack.")
    if max_equity_like_risk_off + min_defensive_risk_off > 1.000001:
        warnings.append("Risk-off equity-like and defensive constraints leave almost no slack.")

    crypto_group_limit = float(group_limits.get("crypto", 1.0))
    if float(params.get("crypto_max_normal", 0.0)) > crypto_group_limit + 1e-12:
        errors.append("crypto_max_normal exceeds the crypto group limit.")
    if float(params.get("crypto_max_risk_off", 0.0)) > crypto_group_limit + 1e-12:
        errors.append("crypto_max_risk_off exceeds the crypto group limit.")

    defensive_groups = set(str(group) for group in params.get("defensive_groups", []))
    active_defensive = [
        ticker for ticker in tickers if str(group_map.get(ticker, "")) in defensive_groups
    ]
    max_defensive_capacity = float(asset_caps.reindex(active_defensive).fillna(0.0).sum())
    if max_defensive_capacity + 1e-12 < min_defensive_normal:
        errors.append("Current defensive assets cannot satisfy min_defensive_weight_normal.")
    if max_defensive_capacity + 1e-12 < min_defensive_risk_off:
        errors.append("Current defensive assets cannot satisfy min_defensive_weight_risk_off.")

    if bool(params.get("enable_external_broker", False)):
        errors.append("ENABLE_EXTERNAL_BROKER must remain false in this project.")
    if bool(params.get("enable_investopedia_simulator", False)):
        errors.append("ENABLE_INVESTOPEDIA_SIMULATOR must remain false in this project.")
    if not bool(params.get("dry_run", True)) and not (
        bool(params.get("enable_local_paper_trading", False))
        or bool(params.get("enable_investopedia_simulator", False))
    ):
        errors.append("DRY_RUN must stay true when no paper/simulator execution layer is enabled.")

    if bool(params.get("email_send_enabled", False)):
        _email_allowed, blocked_reasons = review_email_send_allowed(params)
        for reason in blocked_reasons:
            warnings.append(f"EMAIL_SEND_ENABLED is true but {reason}; real email send remains blocked.")
    if int(params.get("max_emails_per_day", 1)) <= 0:
        errors.append("MAX_EMAILS_PER_DAY must be at least 1.")

    return {"ok": not errors, "warnings": warnings, "errors": errors}


def _bool_text(value: object) -> str:
    return "true" if bool(value) else "false"


def main() -> None:
    params = build_params()
    validation = validate_config(params)
    gate_status = get_email_gate_status(params)

    print("Config Validation")
    print(f"ok: {_bool_text(validation['ok'])}")
    print(f"warning_count: {len(validation['warnings'])}")
    print(f"error_count: {len(validation['errors'])}")
    if validation["warnings"]:
        print("warnings:")
        for warning in validation["warnings"]:
            print(f"- {warning}")
    if validation["errors"]:
        print("errors:")
        for error in validation["errors"]:
            print(f"- {error}")

    print("")
    print("Email Gate Status")
    print(f"real_email_send_allowed: {_bool_text(gate_status['real_email_send_allowed'])}")
    print(f"phase: {gate_status['phase']}")
    print(f"reason: {gate_status['reason']}")
    print(f"reason_detail: {gate_status.get('reason_detail', '')}")
    print(f"enable_email_notifications: {_bool_text(gate_status['enable_email_notifications'])}")
    print(f"email_send_enabled: {_bool_text(gate_status['email_send_enabled'])}")
    print(f"email_dry_run: {_bool_text(gate_status['email_dry_run'])}")
    print(f"email_provider: {gate_status.get('email_provider', 'brevo')}")
    print(f"email_recipient_configured: {_bool_text(gate_status['email_recipient_configured'])}")
    print(f"email_sender_configured: {_bool_text(gate_status.get('email_sender_configured', False))}")
    print(f"brevo_api_key_configured: {_bool_text(gate_status.get('brevo_api_key_configured', False))}")
    print(f"user_confirmed_email_phase: {_bool_text(gate_status['user_confirmed_email_phase'])}")
    print(f"external_broker_enabled: {_bool_text(gate_status['external_broker_enabled'])}")
    print(f"investopedia_enabled: {_bool_text(gate_status['investopedia_enabled'])}")
    print(f"local_paper_trading_enabled: {_bool_text(gate_status['local_paper_trading_enabled'])}")
    print("blockers:")
    if gate_status["blockers"]:
        for blocker in gate_status["blockers"]:
            print(f"- {blocker}")
    else:
        print("- none")
    print("warnings:")
    if gate_status["warnings"]:
        for warning in gate_status["warnings"]:
            print(f"- {warning}")
    else:
        print("- none")

    if not validation["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
