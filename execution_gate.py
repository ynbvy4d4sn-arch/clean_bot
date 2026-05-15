"""Execution gating layer for selective dry-run or paper-trading actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExecutionGateResult:
    """Result of the execution gate."""

    gate_status: str
    action: str
    reason: str
    trade_now_score: float
    spread_cost: float
    slippage: float
    buffers: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def evaluate_execution_gate(
    selection_result: Any,
    *,
    synthetic_data: bool = False,
    data_freshness_ok: bool = True,
    broker_state_reconciled: bool = True,
    open_orders_exist: bool = False,
    estimated_spread_cost: float | None = None,
    estimated_slippage: float | None = None,
    estimated_transaction_cost: float | None = None,
    delta_vs_hold_is_net: bool = True,
    costs_include_spread_slippage: bool = False,
    execution_uncertainty_buffer: float = 0.001,
    model_uncertainty_buffer: float = 0.001,
    max_spread_cost: float = 0.001,
    max_slippage: float = 0.0015,
    trade_now_hurdle: float = 0.0025,
    fallback_spread_cost: float = 0.0005,
    fallback_slippage: float = 0.001,
) -> ExecutionGateResult:
    """Block risky or low-edge trades before any execution layer is called."""

    selected = selection_result.selected_score
    candidate_name = selection_result.selected_candidate.name
    spread_cost = float(estimated_spread_cost if estimated_spread_cost is not None else fallback_spread_cost)
    slippage = float(estimated_slippage if estimated_slippage is not None else fallback_slippage)
    transaction_cost = float(
        estimated_transaction_cost
        if estimated_transaction_cost is not None
        else getattr(selected, "estimated_cost", 0.0)
    )
    delta_vs_hold = float(getattr(selected, "delta_vs_hold", 0.0))

    trade_now_score = delta_vs_hold
    if not delta_vs_hold_is_net:
        trade_now_score -= transaction_cost
    if not costs_include_spread_slippage:
        trade_now_score -= spread_cost
        trade_now_score -= slippage
    trade_now_score -= execution_uncertainty_buffer
    trade_now_score -= model_uncertainty_buffer

    if not data_freshness_ok:
        return ExecutionGateResult(
            gate_status="BLOCK",
            action="PAUSE",
            reason="Loaded price data is stale; no orders may be executed.",
            trade_now_score=trade_now_score,
            spread_cost=spread_cost,
            slippage=slippage,
            buffers={
                "execution_uncertainty_buffer": execution_uncertainty_buffer,
                "model_uncertainty_buffer": model_uncertainty_buffer,
            },
            diagnostics={"candidate": candidate_name, "estimated_transaction_cost": transaction_cost},
        )

    if synthetic_data:
        return ExecutionGateResult(
            gate_status="BLOCK",
            action="PAUSE",
            reason="Synthetic fallback data detected; no orders may be executed.",
            trade_now_score=trade_now_score,
            spread_cost=spread_cost,
            slippage=slippage,
            buffers={
                "execution_uncertainty_buffer": execution_uncertainty_buffer,
                "model_uncertainty_buffer": model_uncertainty_buffer,
            },
            diagnostics={"candidate": candidate_name, "estimated_transaction_cost": transaction_cost},
        )

    if not broker_state_reconciled:
        return ExecutionGateResult(
            gate_status="BLOCK",
            action="PAUSE",
            reason="Broker or paper state is not reconciled with the model state.",
            trade_now_score=trade_now_score,
            spread_cost=spread_cost,
            slippage=slippage,
            buffers={"execution_uncertainty_buffer": execution_uncertainty_buffer, "model_uncertainty_buffer": model_uncertainty_buffer},
            diagnostics={"candidate": candidate_name, "estimated_transaction_cost": transaction_cost},
        )

    if open_orders_exist:
        return ExecutionGateResult(
            gate_status="BLOCK",
            action="WAIT",
            reason="Open orders exist and must clear before another action is considered.",
            trade_now_score=trade_now_score,
            spread_cost=spread_cost,
            slippage=slippage,
            buffers={"execution_uncertainty_buffer": execution_uncertainty_buffer, "model_uncertainty_buffer": model_uncertainty_buffer},
            diagnostics={"candidate": candidate_name, "estimated_transaction_cost": transaction_cost},
        )

    if spread_cost > max_spread_cost or slippage > max_slippage:
        return ExecutionGateResult(
            gate_status="BLOCK",
            action="WAIT_EXECUTION",
            reason="Estimated spread or slippage is too high for execution right now.",
            trade_now_score=trade_now_score,
            spread_cost=spread_cost,
            slippage=slippage,
            buffers={"execution_uncertainty_buffer": execution_uncertainty_buffer, "model_uncertainty_buffer": model_uncertainty_buffer},
            diagnostics={"candidate": candidate_name, "estimated_transaction_cost": transaction_cost},
        )

    if trade_now_score <= trade_now_hurdle:
        action = "HOLD" if candidate_name == "HOLD" or candidate_name.startswith("HOLD_") else "WAIT"
        return ExecutionGateResult(
            gate_status="BLOCK",
            action=action,
            reason="Net trade-now edge did not clear the execution hurdle.",
            trade_now_score=trade_now_score,
            spread_cost=spread_cost,
            slippage=slippage,
            buffers={"execution_uncertainty_buffer": execution_uncertainty_buffer, "model_uncertainty_buffer": model_uncertainty_buffer},
            diagnostics={"candidate": candidate_name, "estimated_transaction_cost": transaction_cost},
        )

    action = "FULL_REBALANCE"
    if candidate_name == "HOLD" or candidate_name.startswith("HOLD_"):
        action = "HOLD"
    elif candidate_name == "DEFENSIVE_CASH" or candidate_name.startswith("DEFENSIVE_CASH"):
        action = "DE_RISK"
    elif candidate_name.startswith("PARTIAL"):
        action = "PARTIAL_REBALANCE"

    return ExecutionGateResult(
        gate_status="PASS",
        action=action,
        reason="Execution gate passed.",
        trade_now_score=trade_now_score,
        spread_cost=spread_cost,
        slippage=slippage,
        buffers={"execution_uncertainty_buffer": execution_uncertainty_buffer, "model_uncertainty_buffer": model_uncertainty_buffer},
        diagnostics={"candidate": candidate_name, "estimated_transaction_cost": transaction_cost},
    )
