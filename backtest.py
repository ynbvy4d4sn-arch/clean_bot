"""Backtest engine for the robust 3M active allocation optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from asset_universe import (
    CRYPTO_MAX_NORMAL,
    CRYPTO_MAX_RISK_OFF,
    MAX_EQUITY_LIKE_TOTAL_NORMAL,
    MAX_EQUITY_LIKE_TOTAL_RISK_OFF,
    MIN_DEFENSIVE_WEIGHT_NORMAL,
    MIN_DEFENSIVE_WEIGHT_RISK_OFF,
    get_asset_max_weights,
    get_cash_ticker,
    get_group_limits,
    get_group_map,
)
from calendar_utils import get_trading_days, is_last_trading_day_of_week
from config import (
    AppConfig,
    BASE_BUFFER,
    CONCENTRATION_PENALTY,
    COST_RATE,
    COV_JITTER,
    COV_SHRINK_ALPHA,
    COV_WINDOW,
    CVAR_LIMIT,
    DEFENSIVE_WEIGHTS,
    DRAWDOWN_LIMIT,
    FORECAST_CLIP_LOWER,
    FORECAST_CLIP_UPPER,
    HORIZON_DAYS,
    KAPPA,
    MAX_TURNOVER,
    MIN_TURNOVER_TO_TRADE,
    MOMENTUM_LONG,
    MOMENTUM_SHORT,
    PARTIAL_FRACTION,
    PARTIAL_THRESHOLD,
    RISK_AVERSION,
    STRONG_SIGNAL_THRESHOLD,
    TURNOVER_PENALTY,
    USE_RISK_FILTER,
    VOL_BUFFER_MULTIPLIER,
)
from decision import DecisionResult, Recommendation, apply_decision, make_rebalance_decision
from features import compute_market_risk_state, compute_momentum_forecast_at_date, compute_returns
from optimizer import OptimizerResult, build_feasible_initial_weights, optimize_allocation
from risk import (
    RiskRegime,
    RiskSnapshot,
    compute_drawdown,
    estimate_portfolio_historical_risk,
    estimate_robust_covariance_at_date,
)


@dataclass(slots=True)
class BacktestResult:
    """Compatibility wrapper for legacy pipeline consumers."""

    history: pd.DataFrame
    weights_history: pd.DataFrame
    latest_decision: DecisionResult
    latest_optimizer_result: OptimizerResult
    latest_risk_snapshot: RiskSnapshot
    daily: pd.DataFrame
    weights: pd.DataFrame
    target_weights: pd.DataFrame
    benchmark_returns: pd.DataFrame
    benchmark_equity: pd.DataFrame


def _coerce_params(
    params: dict[str, Any] | AppConfig | None,
    tickers: list[str],
) -> dict[str, Any]:
    """Normalize the backtest parameter dictionary."""

    if isinstance(params, AppConfig):
        normalized: dict[str, Any] = {
            "tickers": tickers,
            "momentum_short": params.features.momentum_window_3m,
            "momentum_long": params.features.momentum_window_6m,
            "kappa": params.features.kappa,
            "forecast_clip_lower": params.features.forecast_clip_lower,
            "forecast_clip_upper": params.features.forecast_clip_upper,
            "cov_window": params.risk.cov_window,
            "horizon_days": params.risk.horizon_days,
            "cov_shrink_alpha": params.risk.cov_shrink_alpha,
            "cov_jitter": params.risk.cov_jitter,
            "risk_aversion": params.optimization.risk_aversion,
            "turnover_penalty": params.optimization.turnover_penalty,
            "concentration_penalty": params.optimization.concentration_penalty,
            "cost_rate": params.optimization.cost_rate,
            "base_buffer": params.optimization.base_buffer,
            "vol_buffer_multiplier": params.optimization.vol_buffer_multiplier,
            "min_turnover_to_trade": params.decision.min_rebalance_turnover,
            "partial_threshold": params.decision.full_rebalance_turnover,
            "partial_fraction": params.decision.partial_rebalance_ratio,
            "strong_signal_threshold": params.decision.strong_signal_threshold,
            "max_turnover": params.optimization.max_turnover,
            "drawdown_limit": params.risk.drawdown_limit,
            "cvar_limit": params.risk.cvar_limit,
            "use_risk_filter": params.risk.use_risk_filter,
            "defensive_weights": params.risk.defensive_weights.copy(),
            "asset_max_weights": get_asset_max_weights(),
            "group_map": get_group_map(),
            "group_limits": get_group_limits(),
            "cash_ticker": params.cash_ticker,
            "min_cash_weight": params.optimization.min_cash_buffer,
            "max_equity_like_total_normal": MAX_EQUITY_LIKE_TOTAL_NORMAL,
            "max_equity_like_total_risk_off": MAX_EQUITY_LIKE_TOTAL_RISK_OFF,
            "min_defensive_weight_normal": MIN_DEFENSIVE_WEIGHT_NORMAL,
            "min_defensive_weight_risk_off": MIN_DEFENSIVE_WEIGHT_RISK_OFF,
            "crypto_max_normal": CRYPTO_MAX_NORMAL,
            "crypto_max_risk_off": CRYPTO_MAX_RISK_OFF,
        }
        return normalized

    normalized = dict(params or {})
    normalized.setdefault("tickers", tickers)
    normalized.setdefault("momentum_short", MOMENTUM_SHORT)
    normalized.setdefault("momentum_long", MOMENTUM_LONG)
    normalized.setdefault("kappa", KAPPA)
    normalized.setdefault("forecast_clip_lower", FORECAST_CLIP_LOWER)
    normalized.setdefault("forecast_clip_upper", FORECAST_CLIP_UPPER)
    normalized.setdefault("cov_window", COV_WINDOW)
    normalized.setdefault("horizon_days", HORIZON_DAYS)
    normalized.setdefault("cov_shrink_alpha", COV_SHRINK_ALPHA)
    normalized.setdefault("cov_jitter", COV_JITTER)
    normalized.setdefault("risk_aversion", RISK_AVERSION)
    normalized.setdefault("turnover_penalty", TURNOVER_PENALTY)
    normalized.setdefault("concentration_penalty", CONCENTRATION_PENALTY)
    normalized.setdefault("cost_rate", COST_RATE)
    normalized.setdefault("base_buffer", BASE_BUFFER)
    normalized.setdefault("vol_buffer_multiplier", VOL_BUFFER_MULTIPLIER)
    normalized.setdefault("min_turnover_to_trade", MIN_TURNOVER_TO_TRADE)
    normalized.setdefault("partial_threshold", PARTIAL_THRESHOLD)
    normalized.setdefault("partial_fraction", PARTIAL_FRACTION)
    normalized.setdefault("strong_signal_threshold", STRONG_SIGNAL_THRESHOLD)
    normalized.setdefault("max_turnover", MAX_TURNOVER)
    normalized.setdefault("drawdown_limit", DRAWDOWN_LIMIT)
    normalized.setdefault("cvar_limit", CVAR_LIMIT)
    normalized.setdefault("use_risk_filter", USE_RISK_FILTER)
    normalized.setdefault("defensive_weights", DEFENSIVE_WEIGHTS.copy())
    normalized.setdefault("asset_max_weights", get_asset_max_weights())
    normalized.setdefault("group_map", get_group_map())
    normalized.setdefault("group_limits", get_group_limits())
    normalized.setdefault("cash_ticker", get_cash_ticker())
    normalized.setdefault("min_cash_weight", 0.0)
    normalized.setdefault("max_equity_like_total_normal", MAX_EQUITY_LIKE_TOTAL_NORMAL)
    normalized.setdefault("max_equity_like_total_risk_off", MAX_EQUITY_LIKE_TOTAL_RISK_OFF)
    normalized.setdefault("min_defensive_weight_normal", MIN_DEFENSIVE_WEIGHT_NORMAL)
    normalized.setdefault("min_defensive_weight_risk_off", MIN_DEFENSIVE_WEIGHT_RISK_OFF)
    normalized.setdefault("crypto_max_normal", CRYPTO_MAX_NORMAL)
    normalized.setdefault("crypto_max_risk_off", CRYPTO_MAX_RISK_OFF)
    return normalized


def _params_for_day(base_params: dict[str, Any], risk_state: str, tickers: list[str]) -> dict[str, Any]:
    """Return the day-specific parameter set after applying the risk-state overlay."""

    params_today = dict(base_params)
    params_today["tickers"] = tickers
    params_today["group_limits"] = dict(base_params["group_limits"])
    params_today["asset_max_weights"] = dict(base_params["asset_max_weights"])
    params_today["group_map"] = dict(base_params["group_map"])
    params_today["defensive_weights"] = dict(base_params["defensive_weights"])

    if risk_state == "risk_off":
        params_today["max_equity_like_total"] = float(base_params["max_equity_like_total_risk_off"])
        params_today["min_defensive_weight"] = float(base_params["min_defensive_weight_risk_off"])
        params_today["group_limits"]["crypto"] = min(
            float(params_today["group_limits"].get("crypto", base_params["crypto_max_risk_off"])),
            float(base_params["crypto_max_risk_off"]),
        )
    else:
        params_today["max_equity_like_total"] = float(base_params["max_equity_like_total_normal"])
        params_today["min_defensive_weight"] = float(base_params["min_defensive_weight_normal"])
        params_today["group_limits"]["crypto"] = min(
            float(params_today["group_limits"].get("crypto", base_params["crypto_max_normal"])),
            float(base_params["crypto_max_normal"]),
        )

    return params_today


def _compute_current_drawdown(equity_history: list[float]) -> float:
    """Return the latest strategy drawdown from the equity history."""

    if not equity_history:
        return 0.0
    drawdown_series = compute_drawdown(pd.Series(equity_history, dtype=float))
    return float(drawdown_series.iloc[-1])


def _simulate_benchmark(
    returns: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    start_index: int,
    initial_weights: pd.Series | None,
    rebalance_mode: str | None,
) -> pd.Series:
    """Simulate a simple benchmark return stream over the strategy window."""

    evaluation_dates = trading_days[start_index:-1]
    if initial_weights is None or initial_weights.empty:
        return pd.Series(np.nan, index=evaluation_dates, dtype=float)

    weights = initial_weights.astype(float).reindex(returns.columns).fillna(0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return pd.Series(np.nan, index=evaluation_dates, dtype=float)
    weights = weights / total
    target_weights = weights.copy()

    records: list[tuple[pd.Timestamp, float]] = []
    for idx in range(start_index, len(trading_days) - 1):
        date = pd.Timestamp(trading_days[idx])
        next_date = pd.Timestamp(trading_days[idx + 1])

        if rebalance_mode == "weekly" and is_last_trading_day_of_week(date, trading_days):
            weights = target_weights.copy()

        next_returns = returns.loc[next_date].reindex(weights.index).fillna(0.0)
        benchmark_return = float(weights @ next_returns)
        records.append((date, benchmark_return))

        drifted = weights * (1.0 + next_returns)
        if float(drifted.sum()) > 0.0:
            weights = drifted / float(drifted.sum())

    return pd.Series(
        data=[value for _, value in records],
        index=pd.DatetimeIndex([date for date, _ in records]),
        dtype=float,
    )


def _build_benchmark_frames(
    returns: pd.DataFrame,
    investable_tickers: list[str],
    trading_days: pd.DatetimeIndex,
    start_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build benchmark return and equity DataFrames."""

    tickers = [str(ticker) for ticker in investable_tickers if ticker in returns.columns]
    equal_weight = pd.Series(1.0 / len(tickers), index=tickers, dtype=float) if tickers else pd.Series(dtype=float)

    proxy_60_40: pd.Series | None = None
    if {"SPY", "IEF"}.issubset(returns.columns):
        proxy_60_40 = pd.Series({"SPY": 0.60, "IEF": 0.40}, dtype=float)

    sgov_only: pd.Series | None = None
    if "SGOV" in tickers:
        sgov_only = pd.Series({"SGOV": 1.0}, dtype=float)

    benchmark_returns = pd.DataFrame(
        {
            "equal_weight_weekly": _simulate_benchmark(
                returns=returns,
                trading_days=trading_days,
                start_index=start_index,
                initial_weights=equal_weight,
                rebalance_mode="weekly",
            ),
            "proxy_60_40": _simulate_benchmark(
                returns=returns,
                trading_days=trading_days,
                start_index=start_index,
                initial_weights=proxy_60_40,
                rebalance_mode="weekly",
            ),
            "sgov_only": _simulate_benchmark(
                returns=returns,
                trading_days=trading_days,
                start_index=start_index,
                initial_weights=sgov_only,
                rebalance_mode=None,
            ),
            "buy_hold_equal_weight": _simulate_benchmark(
                returns=returns,
                trading_days=trading_days,
                start_index=start_index,
                initial_weights=equal_weight,
                rebalance_mode=None,
            ),
        }
    )

    benchmark_equity = pd.DataFrame(index=benchmark_returns.index)
    for column in benchmark_returns.columns:
        series = benchmark_returns[column]
        if series.notna().any():
            benchmark_equity[column] = (1.0 + series.fillna(0.0)).cumprod()
        else:
            benchmark_equity[column] = np.nan

    return benchmark_returns, benchmark_equity


def _run_backtest_core(prices: pd.DataFrame, params: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Run the requested daily backtest and return raw result frames plus metadata."""

    price_history = prices.copy()
    price_history.index = pd.to_datetime(price_history.index)
    price_history = price_history.sort_index()

    available_columns = [str(column) for column in price_history.columns]
    if isinstance(params, AppConfig):
        requested_tickers = [str(ticker) for ticker in params.tickers]
    else:
        requested_tickers = [
            str(ticker) for ticker in dict.fromkeys((params or {}).get("tickers", available_columns))
        ]
    missing_tickers = [ticker for ticker in requested_tickers if ticker not in available_columns]
    if missing_tickers:
        raise ValueError(
            "Price history is missing investable tickers required by the backtest: "
            + ", ".join(missing_tickers)
        )

    tickers = requested_tickers
    params = _coerce_params(params=params, tickers=tickers)
    all_returns = compute_returns(price_history)
    returns = all_returns.reindex(columns=tickers).fillna(0.0)
    trading_days = get_trading_days(returns)

    # Warm-up ensures that every decision date has enough trailing history for:
    # - the 200-day market filter
    # - the long momentum window
    # - the covariance estimation window
    start_index = max(
        200,
        int(params["momentum_long"]),
        int(params["cov_window"]),
    ) + 1
    if len(trading_days) <= start_index + 1:
        raise ValueError("Not enough trading history for the requested backtest warmup.")

    w_current = build_feasible_initial_weights(tickers=tickers, params=params)
    equity = 1.0
    equity_history = [equity]

    daily_records: list[dict[str, Any]] = []
    weight_records: list[pd.Series] = []
    target_weight_records: list[pd.Series] = []

    latest_optimizer_result: OptimizerResult | None = None
    latest_decision_info: dict[str, Any] | None = None
    latest_risk_state = "normal"

    for idx in range(start_index, len(trading_days) - 1):
        date = pd.Timestamp(trading_days[idx])
        next_date = pd.Timestamp(trading_days[idx + 1])

        # Walk-forward convention:
        # - optimize using information available up to and including `date`
        # - apply the executed allocation to the realized return from `date` to `next_date`
        risk_state = compute_market_risk_state(price_history, date)
        params_today = _params_for_day(base_params=params, risk_state=risk_state, tickers=tickers)

        data_quality_ok = True
        optimizer_result: OptimizerResult | None = None
        try:
            mu = compute_momentum_forecast_at_date(
                prices=price_history,
                date=date,
                tickers=tickers,
                short_window=int(params["momentum_short"]),
                long_window=int(params["momentum_long"]),
                kappa=float(params["kappa"]),
                clip_lower=float(params["forecast_clip_lower"]),
                clip_upper=float(params["forecast_clip_upper"]),
            )
            Sigma = estimate_robust_covariance_at_date(
                returns=returns,
                date=date,
                window=int(params["cov_window"]),
                horizon_days=int(params["horizon_days"]),
                alpha=float(params["cov_shrink_alpha"]),
                jitter=float(params["cov_jitter"]),
            )
            if not (
                np.isfinite(mu.to_numpy(dtype=float)).all()
                and np.isfinite(Sigma.to_numpy(dtype=float)).all()
            ):
                data_quality_ok = False
        except Exception as exc:
            data_quality_ok = False
            mu = pd.Series(0.0, index=tickers, dtype=float)
            Sigma = pd.DataFrame(
                np.eye(len(tickers), dtype=float) * float(params["cov_jitter"]),
                index=tickers,
                columns=tickers,
            )
            optimizer_result = OptimizerResult(
                target_weights=w_current.copy(),
                solver_name="data_quality_fallback",
                success=False,
                objective_value=None,
                status=str(exc),
                diagnostics={"warning": str(exc)},
            )

        if optimizer_result is None:
            optimizer_result = optimize_allocation(
                mu=mu,
                Sigma=Sigma,
                w_current=w_current,
                params=params_today,
            )

        w_target = optimizer_result.target_weights.reindex(tickers).fillna(0.0)
        returns_window = returns.loc[:date].tail(int(params["cov_window"]))
        risk_report = estimate_portfolio_historical_risk(returns_window=returns_window, w=w_target)
        risk_report["current_drawdown"] = _compute_current_drawdown(equity_history)

        weekly_rebalance_day = is_last_trading_day_of_week(date, trading_days)
        decision_info = make_rebalance_decision(
            mu=mu,
            Sigma=Sigma,
            w_current=w_current,
            w_target=w_target,
            params=params_today,
            weekly_rebalance_day=weekly_rebalance_day,
            risk_report=risk_report,
            data_quality_ok=data_quality_ok,
        )
        w_new = apply_decision(
            w_current=w_current,
            w_target=w_target,
            decision=decision_info["decision"],
            params=params_today,
        )

        realized_turnover = float(np.abs(w_new - w_current.reindex(w_new.index).fillna(0.0)).sum())
        realized_cost = float(params_today["cost_rate"]) * realized_turnover
        next_return_vector = returns.loc[next_date].reindex(w_new.index).fillna(0.0)
        portfolio_return_gross = float(w_new @ next_return_vector)
        portfolio_return_net = portfolio_return_gross - realized_cost
        equity *= 1.0 + portfolio_return_net
        equity_history.append(equity)

        daily_records.append(
            {
                "date": date,
                "next_date": next_date,
                "equity": equity,
                "portfolio_return_gross": portfolio_return_gross,
                "portfolio_return_net": portfolio_return_net,
                "decision": decision_info["decision"],
                "risk_state": risk_state,
                "realized_turnover": realized_turnover,
                "realized_cost": realized_cost,
                "estimated_cost": decision_info["estimated_cost"],
                "buffer": decision_info["buffer"],
                "volatility_buffer": decision_info["volatility_buffer"],
                "net_benefit": decision_info["net_benefit"],
                "delta_score": decision_info["delta_score"],
                "score_current": decision_info["score_current"],
                "score_target": decision_info["score_target"],
                "weekly_rebalance_day": decision_info["weekly_rebalance_day"],
                "emergency_condition": decision_info["emergency_condition"],
                "risk_gate_failed": decision_info["risk_gate_failed"],
                "target_vol": decision_info["target_vol"],
                "cvar_95": risk_report["cvar_95"],
                "max_drawdown": risk_report["max_drawdown"],
                "solver": optimizer_result.solver_name,
                "optimizer_status": optimizer_result.status,
            }
        )

        weight_row = w_new.copy()
        weight_row.name = date
        weight_records.append(weight_row)

        target_row = w_target.copy()
        target_row.name = date
        target_weight_records.append(target_row)

        latest_optimizer_result = optimizer_result
        latest_decision_info = decision_info
        latest_risk_state = risk_state
        drifted_weights = w_new * (1.0 + next_return_vector)
        if float(drifted_weights.sum()) > 0.0:
            w_current = drifted_weights / float(drifted_weights.sum())
        else:
            w_current = w_new.copy()

    if latest_optimizer_result is None or latest_decision_info is None:
        raise ValueError("Backtest could not be run because no evaluation dates were processed.")

    daily_df = pd.DataFrame(daily_records)
    weights_df = pd.DataFrame(weight_records).reindex(columns=tickers).fillna(0.0)
    target_weights_df = pd.DataFrame(target_weight_records).reindex(columns=tickers).fillna(0.0)
    weights_df.index.name = "date"
    target_weights_df.index.name = "date"
    benchmark_returns_df, benchmark_equity_df = _build_benchmark_frames(
        returns=all_returns,
        investable_tickers=tickers,
        trading_days=trading_days,
        start_index=start_index,
    )

    return (
        {
            "daily": daily_df,
            "weights": weights_df,
            "target_weights": target_weights_df,
            "benchmark_returns": benchmark_returns_df,
            "benchmark_equity": benchmark_equity_df,
        },
        {
            "latest_optimizer_result": latest_optimizer_result,
            "latest_decision_info": latest_decision_info,
            "latest_weights": weights_df.iloc[-1].copy(),
            "latest_target_weights": target_weights_df.iloc[-1].copy(),
            "latest_risk_state": latest_risk_state,
        },
    )


def _decision_result_from_info(
    decision_info: dict[str, Any],
    execution_weights: pd.Series,
    target_weights: pd.Series,
) -> DecisionResult:
    """Build the compatibility decision object used by reports."""

    decision = Recommendation(decision_info["decision"])
    rationale = [
        f"Net benefit: {float(decision_info['net_benefit']):.6f}"
        if pd.notna(decision_info["net_benefit"])
        else "Net benefit unavailable",
        f"Turnover: {float(decision_info['turnover']):.2%}",
        f"Weekly rebalance day: {bool(decision_info['weekly_rebalance_day'])}",
        f"Emergency condition: {bool(decision_info['emergency_condition'])}",
        f"Risk gate failed: {bool(decision_info['risk_gate_failed'])}",
    ]
    return DecisionResult(
        action=decision,
        execution_weights=execution_weights.copy(),
        target_weights=target_weights.copy(),
        estimated_turnover=float(decision_info["turnover"]),
        rationale=rationale,
    )


def _risk_snapshot_from_daily(
    latest_row: pd.Series,
    latest_date: pd.Timestamp,
    params: dict[str, Any],
) -> RiskSnapshot:
    """Build a compatibility risk snapshot for reporting."""

    if latest_row["decision"] == Recommendation.PAUSE.value:
        regime = RiskRegime.PAUSE
    elif latest_row["risk_state"] == "risk_off" or bool(latest_row["risk_gate_failed"]):
        regime = RiskRegime.RISK_OFF
    else:
        regime = RiskRegime.RISK_ON

    recommended_risky_cap = (
        float(params["max_equity_like_total_risk_off"])
        if latest_row["risk_state"] == "risk_off"
        else float(params["max_equity_like_total_normal"])
    )

    return RiskSnapshot(
        as_of=pd.Timestamp(latest_date),
        regime=regime,
        market_drawdown=float(latest_row["max_drawdown"]),
        realized_volatility=float(latest_row["target_vol"]) if pd.notna(latest_row["target_vol"]) else 0.0,
        positive_breadth=0.0,
        de_risk_scalar=0.0 if regime in {RiskRegime.RISK_OFF, RiskRegime.PAUSE} else 1.0,
        recommended_risky_cap=recommended_risky_cap,
        diagnostics={
            "risk_state": str(latest_row["risk_state"]),
            "cvar_95": float(latest_row["cvar_95"]),
            "max_drawdown": float(latest_row["max_drawdown"]),
        },
    )


def run_backtest(
    prices: pd.DataFrame,
    params: dict[str, Any] | None = None,
    *,
    returns: pd.DataFrame | None = None,
    universe: Any = None,
    config: AppConfig | None = None,
) -> dict[str, pd.DataFrame] | BacktestResult:
    """Run the requested daily backtest or return a compatibility wrapper."""

    del returns, universe  # Kept for legacy call compatibility.

    effective_params = config if config is not None else params
    result_dict, metadata = _run_backtest_core(prices=prices, params=effective_params or {})

    if config is None:
        return result_dict

    daily_df = result_dict["daily"].copy()
    history = daily_df.rename(
        columns={
            "equity": "portfolio_value",
            "portfolio_return_gross": "gross_return",
            "portfolio_return_net": "net_return",
            "realized_turnover": "turnover",
            "decision": "action",
            "risk_state": "risk_regime",
        }
    ).set_index("date")

    latest_row = daily_df.iloc[-1]
    latest_decision = _decision_result_from_info(
        decision_info=metadata["latest_decision_info"],
        execution_weights=metadata["latest_weights"],
        target_weights=metadata["latest_target_weights"],
    )
    latest_risk_snapshot = _risk_snapshot_from_daily(
        latest_row=latest_row,
        latest_date=pd.Timestamp(latest_row["date"]),
        params=_coerce_params(config, list(prices.columns)),
    )

    return BacktestResult(
        history=history,
        weights_history=result_dict["weights"],
        latest_decision=latest_decision,
        latest_optimizer_result=metadata["latest_optimizer_result"],
        latest_risk_snapshot=latest_risk_snapshot,
        daily=result_dict["daily"],
        weights=result_dict["weights"],
        target_weights=result_dict["target_weights"],
        benchmark_returns=result_dict["benchmark_returns"],
        benchmark_equity=result_dict["benchmark_equity"],
    )


def latest_signal_summary(result: BacktestResult | dict[str, pd.DataFrame]) -> pd.Series:
    """Return the latest signal row from a backtest result."""

    if isinstance(result, dict):
        return result["daily"].iloc[-1]
    return result.history.iloc[-1]
