"""Decision rules for translating optimizer targets into execution recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from risk import RiskRegime, RiskSnapshot, portfolio_volatility


class Recommendation(str, Enum):
    """High-level strategy actions returned by the decision layer."""

    HOLD = "HOLD"
    WAIT = "WAIT"
    PARTIAL_REBALANCE = "PARTIAL_REBALANCE"
    FULL_REBALANCE = "FULL_REBALANCE"
    DE_RISK = "DE_RISK"
    PAUSE = "PAUSE"


@dataclass(slots=True)
class DecisionResult:
    """Compatibility payload for reporting and legacy pipeline consumers."""

    action: Recommendation
    execution_weights: pd.Series
    target_weights: pd.Series
    estimated_turnover: float
    rationale: list[str] = field(default_factory=list)


def _align_weight_inputs(
    w_current: pd.Series,
    w_other: pd.Series,
    index_hint: pd.Index | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Align two weight series to a common index."""

    if index_hint is None:
        ordered = list(dict.fromkeys([*w_current.index.tolist(), *w_other.index.tolist()]))
        index = pd.Index([str(ticker) for ticker in ordered], name="ticker")
    else:
        index = pd.Index([str(ticker) for ticker in index_hint.tolist()], name="ticker")

    current = w_current.astype(float).copy()
    current.index = pd.Index([str(ticker) for ticker in current.index], name="ticker")
    other = w_other.astype(float).copy()
    other.index = pd.Index([str(ticker) for ticker in other.index], name="ticker")
    return current.reindex(index).fillna(0.0), other.reindex(index).fillna(0.0)


def _normalize_weights(weights: pd.Series) -> pd.Series:
    """Clip tiny negatives and normalize weights to sum to one."""

    normalized = weights.astype(float).copy()
    normalized[np.abs(normalized) < 1e-10] = 0.0
    normalized = normalized.clip(lower=0.0)
    total = float(normalized.sum())
    if total <= 0.0:
        normalized[:] = 1.0 / len(normalized)
        return normalized
    return normalized / total


def objective_score(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    w: pd.Series,
    w_current: pd.Series,
    params: dict[str, Any],
) -> float:
    """Compute the optimizer-consistent objective score for a weight vector."""

    if mu.empty or Sigma.empty:
        raise ValueError("mu and Sigma must not be empty.")

    tickers = pd.Index([str(ticker) for ticker in mu.index.tolist()], name="ticker")
    mu_aligned = mu.astype(float).copy()
    mu_aligned.index = tickers
    sigma_aligned = Sigma.astype(float).copy()
    sigma_aligned.index = pd.Index([str(ticker) for ticker in sigma_aligned.index], name="ticker")
    sigma_aligned.columns = pd.Index([str(ticker) for ticker in sigma_aligned.columns], name="ticker")
    sigma_aligned = sigma_aligned.reindex(index=tickers, columns=tickers).fillna(0.0)

    current_aligned, weights_aligned = _align_weight_inputs(
        w_current=w_current,
        w_other=w,
        index_hint=tickers,
    )
    mu_values = mu_aligned.to_numpy(dtype=float)
    sigma_values = sigma_aligned.to_numpy(dtype=float)
    weight_values = weights_aligned.to_numpy(dtype=float)
    current_values = current_aligned.to_numpy(dtype=float)

    return float(
        mu_values @ weight_values
        - float(params.get("risk_aversion", 0.0)) * (weight_values @ sigma_values @ weight_values)
        - float(params.get("turnover_penalty", 0.0)) * np.abs(weight_values - current_values).sum()
        - float(params.get("concentration_penalty", 0.0)) * np.square(weight_values).sum()
    )


def compute_turnover(w_current: pd.Series, w_target: pd.Series) -> float:
    """Compute gross turnover as the sum of absolute weight changes."""

    current_aligned, target_aligned = _align_weight_inputs(w_current=w_current, w_other=w_target)
    return float(np.abs(target_aligned - current_aligned).sum())


def is_emergency_condition(inputs: dict[str, Any]) -> bool:
    """Return whether a non-scheduled rebalance is justified by urgency."""

    risk_gate_failed = bool(inputs.get("risk_gate_failed", False))
    net_benefit = float(inputs.get("net_benefit", 0.0))
    strong_signal_threshold = float(inputs.get("strong_signal_threshold", 0.0))
    turnover = float(inputs.get("turnover", 0.0))
    current_drawdown = float(inputs.get("current_drawdown", 0.0))
    drawdown_limit = float(inputs.get("drawdown_limit", float("-inf")))

    return bool(
        risk_gate_failed
        or net_benefit > strong_signal_threshold
        or (turnover > 0.40 and net_benefit > 0.0)
        or current_drawdown < drawdown_limit
    )


def make_rebalance_decision(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    w_current: pd.Series,
    w_target: pd.Series,
    params: dict[str, Any],
    weekly_rebalance_day: bool,
    risk_report: dict[str, float] | None = None,
    data_quality_ok: bool = True,
) -> dict[str, Any]:
    """Translate optimizer output into a practical rebalance decision."""

    aligned_current, aligned_target = _align_weight_inputs(w_current=w_current, w_other=w_target, index_hint=mu.index)

    turnover = compute_turnover(aligned_current, aligned_target)
    estimated_cost = float(params.get("cost_rate", 0.0)) * turnover

    score_current = float("nan")
    score_target = float("nan")
    delta_score = float("nan")
    target_vol = float("nan")
    volatility_buffer = float("nan")
    buffer = float("nan")
    net_benefit = float("nan")

    if data_quality_ok:
        score_current = objective_score(
            mu=mu,
            Sigma=Sigma,
            w=aligned_current,
            w_current=aligned_current,
            params=params,
        )
        score_target = objective_score(
            mu=mu,
            Sigma=Sigma,
            w=aligned_target,
            w_current=aligned_current,
            params=params,
        )
        delta_score = score_target - score_current
        target_vol = portfolio_volatility(Sigma=Sigma, w=aligned_target)
        volatility_buffer = float(params.get("vol_buffer_multiplier", 0.0)) * target_vol
        buffer = float(params.get("base_buffer", 0.0)) + volatility_buffer
        net_benefit = delta_score - estimated_cost - buffer

    risk_report = dict(risk_report or {})
    cvar_95 = float(risk_report.get("cvar_95", np.nan))
    max_drawdown = float(risk_report.get("max_drawdown", np.nan))
    current_drawdown = float(risk_report.get("current_drawdown", 0.0))
    drawdown_limit = float(params.get("drawdown_limit", float("-inf")))
    cvar_limit = float(params.get("cvar_limit", float("-inf")))

    risk_gate_failed = bool(
        (np.isfinite(cvar_95) and cvar_95 < cvar_limit)
        or (np.isfinite(max_drawdown) and max_drawdown < drawdown_limit)
    )

    emergency_condition = is_emergency_condition(
        {
            "risk_gate_failed": risk_gate_failed,
            "net_benefit": 0.0 if np.isnan(net_benefit) else net_benefit,
            "strong_signal_threshold": float(params.get("strong_signal_threshold", 0.0)),
            "turnover": turnover,
            "current_drawdown": current_drawdown,
            "drawdown_limit": drawdown_limit,
        }
    )

    if not data_quality_ok:
        decision = Recommendation.PAUSE.value
    elif risk_gate_failed:
        decision = Recommendation.DE_RISK.value
    elif net_benefit <= 0.0:
        decision = Recommendation.HOLD.value
    elif not weekly_rebalance_day and not emergency_condition:
        decision = Recommendation.WAIT.value
    elif turnover < float(params.get("min_turnover_to_trade", 0.0)):
        decision = Recommendation.WAIT.value
    elif turnover < float(params.get("partial_threshold", np.inf)):
        decision = Recommendation.PARTIAL_REBALANCE.value
    else:
        decision = Recommendation.FULL_REBALANCE.value

    return {
        "decision": decision,
        "score_current": score_current,
        "score_target": score_target,
        "delta_score": delta_score,
        "turnover": turnover,
        "estimated_cost": estimated_cost,
        "buffer": buffer,
        "volatility_buffer": volatility_buffer,
        "net_benefit": net_benefit,
        "weekly_rebalance_day": bool(weekly_rebalance_day),
        "emergency_condition": emergency_condition,
        "risk_gate_failed": risk_gate_failed,
        "target_vol": target_vol,
    }


def apply_decision(
    w_current: pd.Series,
    w_target: pd.Series,
    decision: str | Recommendation,
    params: dict[str, Any],
) -> pd.Series:
    """Apply a decision rule and return normalized execution weights."""

    current_aligned, target_aligned = _align_weight_inputs(w_current=w_current, w_other=w_target)
    decision_value = decision.value if isinstance(decision, Recommendation) else str(decision)

    if decision_value == Recommendation.FULL_REBALANCE.value:
        new_weights = target_aligned.copy()
    elif decision_value == Recommendation.PARTIAL_REBALANCE.value:
        partial_fraction = float(params.get("partial_fraction", 0.50))
        new_weights = current_aligned + partial_fraction * (target_aligned - current_aligned)
    elif decision_value == Recommendation.DE_RISK.value:
        defensive_weights = pd.Series(params.get("defensive_weights", {}), dtype=float)
        defensive_weights.index = pd.Index([str(ticker) for ticker in defensive_weights.index], name="ticker")
        defensive_weights = defensive_weights.reindex(current_aligned.index).fillna(0.0)
        defensive_weights = _normalize_weights(defensive_weights)
        new_weights = 0.5 * current_aligned + 0.5 * defensive_weights
    else:
        new_weights = current_aligned.copy()

    return _normalize_weights(new_weights)


def estimate_turnover(current_weights: pd.Series, target_weights: pd.Series) -> float:
    """Compatibility alias for compute_turnover()."""

    return compute_turnover(current_weights, target_weights)


def blend_weights(
    current_weights: pd.Series,
    target_weights: pd.Series,
    participation: float,
) -> pd.Series:
    """Compatibility helper for a partial rebalance blend."""

    current_aligned, target_aligned = _align_weight_inputs(w_current=current_weights, w_other=target_weights)
    blended = current_aligned + float(participation) * (target_aligned - current_aligned)
    return _normalize_weights(blended)


def decide_rebalance(
    current_weights: pd.Series,
    target_weights: pd.Series,
    risk_snapshot: RiskSnapshot,
    config: Any,
    data_ready: bool = True,
) -> DecisionResult:
    """Compatibility wrapper used by the legacy pipeline."""

    turnover = compute_turnover(current_weights, target_weights)
    rationale = [
        f"Risk regime: {risk_snapshot.regime.value}",
        f"Turnover: {turnover:.2%}",
    ]

    if not data_ready or risk_snapshot.regime == RiskRegime.PAUSE:
        action = Recommendation.PAUSE if not data_ready else Recommendation.PAUSE
        execution_weights = current_weights.copy()
        rationale.append("Data quality or risk regime requires a pause.")
    elif risk_snapshot.regime == RiskRegime.RISK_OFF:
        action = Recommendation.DE_RISK
        execution_weights = target_weights.copy()
        rationale.append("Risk-off regime triggered defensive action.")
    elif turnover < float(getattr(config, "min_rebalance_turnover", 0.0)):
        action = Recommendation.HOLD
        execution_weights = current_weights.copy()
        rationale.append("Turnover is below the minimum rebalance threshold.")
    elif turnover < float(getattr(config, "full_rebalance_turnover", np.inf)):
        action = Recommendation.PARTIAL_REBALANCE
        execution_weights = blend_weights(
            current_weights=current_weights,
            target_weights=target_weights,
            participation=float(getattr(config, "partial_rebalance_ratio", 0.5)),
        )
        rationale.append("Partial rebalance is sufficient.")
    else:
        action = Recommendation.FULL_REBALANCE
        execution_weights = target_weights.copy()
        rationale.append("Full rebalance recommended.")

    return DecisionResult(
        action=action,
        execution_weights=_normalize_weights(execution_weights),
        target_weights=_normalize_weights(target_weights),
        estimated_turnover=turnover,
        rationale=rationale,
    )
