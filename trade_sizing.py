"""Trade-fraction logic built on top of selection, governance and execution gating."""

from __future__ import annotations

from typing import Any


def compute_trade_fraction(
    selection_result: Any,
    execution_gate_result: Any,
    model_confidence: dict[str, Any],
    data_quality: dict[str, Any],
    turnover_budget: dict[str, Any] | float,
) -> dict[str, Any]:
    """Suggest a trade fraction without forcing actual execution."""

    if getattr(execution_gate_result, "gate_status", "BLOCK") != "PASS":
        return {
            "trade_fraction": 0.0,
            "suggested_action": getattr(execution_gate_result, "action", "PAUSE"),
            "reason": getattr(execution_gate_result, "reason", "Execution gate did not pass."),
        }

    score = float(model_confidence.get("model_confidence_score", 0.50))
    data_score = float(data_quality.get("global_data_quality_score", 0.50))
    turnover_remaining = (
        float(turnover_budget.get("turnover_budget_remaining", 1.0))
        if isinstance(turnover_budget, dict)
        else float(turnover_budget)
    )
    if turnover_remaining <= 0.0:
        return {"trade_fraction": 0.0, "suggested_action": "WAIT", "reason": "Turnover budget exhausted."}

    selected_score = selection_result.selected_score
    edge = max(float(selected_score.delta_vs_hold), 0.0)
    beat_hold = float(selected_score.probability_beats_hold)
    beat_cash = float(selected_score.probability_beats_cash)

    max_fraction = 0.25 if min(score, data_score) < 0.60 else 0.50 if min(score, data_score) < 0.80 else 1.0
    if edge > 0.005 and beat_hold > 0.65 and beat_cash > 0.60:
        fraction = max_fraction
    elif edge > 0.001 and beat_hold > 0.55:
        fraction = min(max_fraction, 0.50)
    else:
        fraction = min(max_fraction, 0.25)

    if selection_result.selected_candidate.name == "HOLD":
        return {"trade_fraction": 0.0, "suggested_action": "HOLD", "reason": "Selected candidate is HOLD."}

    action = "FULL_REBALANCE" if fraction >= 0.99 else "PARTIAL_REBALANCE"
    if selection_result.selected_candidate.name == "DEFENSIVE_CASH":
        action = "DE_RISK"
    return {
        "trade_fraction": float(min(max(fraction, 0.0), turnover_remaining, 1.0)),
        "suggested_action": action,
        "reason": "Trade size derived from edge, confidence and turnover budget.",
    }
