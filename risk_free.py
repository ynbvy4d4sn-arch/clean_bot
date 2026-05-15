"""Central risk-free return utilities.

Decision-time Sharpe calculations must use horizon-compatible excess returns.
This module is deliberately tiny to avoid circular imports between scenarios,
optimizers, scorers and reporting code.
"""

from __future__ import annotations

from typing import Any


def risk_free_return_for_horizon(
    risk_free_rate_annual: float,
    horizon_days: int,
    trading_days_per_year: int = 252,
) -> float:
    """Convert an annual risk-free rate to a same-horizon simple return."""

    if int(trading_days_per_year) <= 0:
        raise ValueError("trading_days_per_year must be positive.")
    return float(
        (1.0 + float(risk_free_rate_annual))
        ** (float(horizon_days) / float(trading_days_per_year))
        - 1.0
    )


def risk_free_return_from_params(
    params: dict[str, Any] | None,
    *,
    default_rate_annual: float = 0.02,
    default_horizon_days: int = 63,
    default_trading_days_per_year: int = 252,
) -> float:
    """Resolve a horizon risk-free return from the common params dictionary."""

    values = dict(params or {})
    solver_block = values.get("solver")
    if isinstance(solver_block, dict):
        resolved = dict(solver_block)
        resolved.update(values)
        values = resolved
    annual_rate = float(
        values.get(
            "risk_free_rate_annual",
            values.get("risk_free_rate", default_rate_annual),
        )
        or 0.0
    )
    horizon_days = int(values.get("effective_horizon_days", values.get("horizon_days", default_horizon_days)) or 0)
    trading_days = int(values.get("trading_days_per_year", default_trading_days_per_year) or default_trading_days_per_year)
    return risk_free_return_for_horizon(annual_rate, horizon_days, trading_days)


__all__ = ["risk_free_return_for_horizon", "risk_free_return_from_params"]
