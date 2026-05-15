"""Performance metrics for strategy and benchmark return series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_free import risk_free_return_for_horizon

TRADING_DAYS_PER_YEAR = 252.0


@dataclass(slots=True)
class PerformanceMetrics:
    """Compact container for key strategy performance metrics."""

    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    cvar_95: float
    total_turnover: float
    number_of_trades: int


def _as_return_series(daily_returns: pd.Series | list[float] | np.ndarray) -> pd.Series:
    """Return a clean one-dimensional daily return series."""

    if isinstance(daily_returns, pd.Series):
        series = daily_returns.astype(float).dropna()
    else:
        series = pd.Series(np.asarray(daily_returns, dtype=float)).dropna()

    return series.replace([np.inf, -np.inf], np.nan).dropna()


def annualized_return(daily_returns: pd.Series | list[float] | np.ndarray) -> float:
    """Compute geometric annualized return from daily returns."""

    series = _as_return_series(daily_returns)
    if series.empty:
        return float("nan")

    cumulative_return = float((1.0 + series).prod())
    return float(cumulative_return ** (TRADING_DAYS_PER_YEAR / len(series)) - 1.0)


def annualized_volatility(daily_returns: pd.Series | list[float] | np.ndarray) -> float:
    """Compute annualized daily-return volatility."""

    series = _as_return_series(daily_returns)
    if series.empty:
        return float("nan")
    return float(series.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(
    daily_returns: pd.Series | list[float] | np.ndarray,
    risk_free_rate_annual: float = 0.02,
) -> float:
    """Compute annualized Sharpe ratio from daily returns."""

    series = _as_return_series(daily_returns)
    if series.empty:
        return float("nan")

    volatility = annualized_volatility(series)
    if not np.isfinite(volatility) or volatility <= 0.0:
        return 0.0

    daily_rf = risk_free_return_for_horizon(
        risk_free_rate_annual,
        1,
        int(TRADING_DAYS_PER_YEAR),
    )
    excess_return = series - daily_rf
    return float(excess_return.mean() * TRADING_DAYS_PER_YEAR / volatility)


def max_drawdown(daily_returns: pd.Series | list[float] | np.ndarray) -> float:
    """Compute maximum drawdown from daily returns."""

    series = _as_return_series(daily_returns)
    if series.empty:
        return float("nan")

    equity_curve = (1.0 + series).cumprod()
    running_peak = equity_curve.cummax()
    drawdowns = equity_curve / running_peak - 1.0
    return float(drawdowns.min())


def cvar_95(daily_returns: pd.Series | list[float] | np.ndarray) -> float:
    """Compute 95% historical CVaR / expected shortfall from daily returns."""

    series = _as_return_series(daily_returns)
    if series.empty:
        return float("nan")

    threshold = float(series.quantile(0.05))
    tail = series[series <= threshold]
    if tail.empty:
        return threshold
    return float(tail.mean())


def total_turnover(daily_records: pd.DataFrame) -> float:
    """Sum realized turnover across all daily records."""

    if daily_records.empty or "realized_turnover" not in daily_records.columns:
        return 0.0
    return float(pd.to_numeric(daily_records["realized_turnover"], errors="coerce").fillna(0.0).sum())


def number_of_trades(daily_records: pd.DataFrame) -> int:
    """Count rebalance decisions that imply trading activity."""

    if daily_records.empty or "decision" not in daily_records.columns:
        return 0

    trade_decisions = {"PARTIAL_REBALANCE", "FULL_REBALANCE", "DE_RISK"}
    decisions = daily_records["decision"].astype(str)
    return int(decisions.isin(trade_decisions).sum())


def average_turnover_per_trade(daily_records: pd.DataFrame) -> float:
    """Return average realized turnover on trading days only."""

    if daily_records.empty or "decision" not in daily_records.columns or "realized_turnover" not in daily_records.columns:
        return 0.0

    trade_decisions = {"PARTIAL_REBALANCE", "FULL_REBALANCE", "DE_RISK"}
    trade_rows = daily_records.loc[daily_records["decision"].astype(str).isin(trade_decisions)].copy()
    if trade_rows.empty:
        return 0.0

    turnovers = pd.to_numeric(trade_rows["realized_turnover"], errors="coerce").fillna(0.0)
    return float(turnovers.mean())


def performance_summary(
    strategy_returns_dict: dict[str, pd.Series],
    daily_records_dict: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build a performance summary table for multiple strategy return streams."""

    rows: list[dict[str, float | str]] = []
    daily_records_dict = daily_records_dict or {}
    for name, returns in strategy_returns_dict.items():
        series = _as_return_series(returns)
        daily_records = daily_records_dict.get(name)
        if series.empty:
            row: dict[str, float | str] = {
                "name": name,
                "observations": 0,
                "total_return": float("nan"),
                "cagr": float("nan"),
                "volatility": float("nan"),
                "sharpe": float("nan"),
                "annualized_return": float("nan"),
                "annualized_volatility": float("nan"),
                "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"),
                "cvar_95": float("nan"),
            }
            if daily_records is not None:
                row["total_turnover"] = total_turnover(daily_records)
                row["number_of_trades"] = number_of_trades(daily_records)
                row["average_turnover_per_trade"] = average_turnover_per_trade(daily_records)
            rows.append(row)
            continue

        row = {
            "name": name,
            "observations": int(len(series)),
            "total_return": float((1.0 + series).prod() - 1.0),
            "cagr": annualized_return(series),
            "volatility": annualized_volatility(series),
            "sharpe": sharpe_ratio(series),
            "annualized_return": annualized_return(series),
            "annualized_volatility": annualized_volatility(series),
            "sharpe_ratio": sharpe_ratio(series),
            "max_drawdown": max_drawdown(series),
            "cvar_95": cvar_95(series),
        }
        if daily_records is not None:
            row["total_turnover"] = total_turnover(daily_records)
            row["number_of_trades"] = number_of_trades(daily_records)
            row["average_turnover_per_trade"] = average_turnover_per_trade(daily_records)
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("cagr", ascending=False, na_position="last")
    return summary.reset_index(drop=True)


def compute_performance_metrics(history: pd.DataFrame, risk_free_rate: float = 0.02) -> PerformanceMetrics:
    """Compatibility wrapper for the strategy history DataFrame."""

    if history.empty or "net_return" not in history.columns:
        raise ValueError("History must contain a 'net_return' column.")

    returns = _as_return_series(history["net_return"])
    return PerformanceMetrics(
        annualized_return=annualized_return(returns),
        annualized_volatility=annualized_volatility(returns),
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate_annual=risk_free_rate),
        max_drawdown=max_drawdown(returns),
        cvar_95=cvar_95(returns),
        total_turnover=total_turnover(history),
        number_of_trades=number_of_trades(history),
    )


def metrics_as_series(metrics: PerformanceMetrics) -> pd.Series:
    """Convert a performance-metrics object into a Pandas Series."""

    return pd.Series(
        {
            "annualized_return": metrics.annualized_return,
            "annualized_volatility": metrics.annualized_volatility,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "cvar_95": metrics.cvar_95,
            "total_turnover": metrics.total_turnover,
            "number_of_trades": float(metrics.number_of_trades),
        },
        dtype=float,
    )
