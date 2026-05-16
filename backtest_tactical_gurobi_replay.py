"""Gurobi-style tactical allocation replay backtest.

Research-only. Does not place orders and does not change production allocation.

Purpose:
- Convert tactical v2/v3 scores into expected excess-return proxies.
- Use a constrained optimizer to choose portfolio weights.
- Backtest the resulting allocation walk-forward.
- Sharpe uses 2% annual risk-free rate.

This is closer to the actual allocation problem than Top-N equal weight,
but it is still a research replay, not a full production daily_bot replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd

from config import build_params
from data import load_price_data
from features import compute_returns
from tactical_forecast import (
    build_multi_horizon_forecast,
    _build_tactical_score_v2_table,
    _build_tactical_score_v3_table,
)
from research_tactical_score_v4 import build_v4_feature_scores

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PRICE_CACHE_PATH = BASE_DIR / "data" / "prices_cache.csv"


@dataclass(frozen=True)
class ReplayConfig:
    start_date: str = "2024-01-01"
    end_date: str | None = None
    min_history: int = 180
    rebalance_every: int = 1
    score_name: str = "v3"  # v2, v3, v4 or v4b
    max_weight: float = 0.25
    top_n_gate: int = 8
    risk_free_rate_annual: float = 0.02
    covariance_window: int = 60
    signal_scale: float = 0.015
    lambda_variance: float = 8.0
    lambda_turnover: float = 0.03
    lambda_concentration: float = 0.02
    max_rebalance_turnover: float | None = 0.30
    output_prefix: str = "tactical_gurobi_replay"


def _zscore(values: pd.Series) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan)
    std = float(clean.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=values.index, dtype=float)
    return ((clean - float(clean.mean())) / std).fillna(0.0)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def _performance_stats(
    daily_returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
    risk_free_rate_annual: float,
) -> dict[str, float]:
    clean = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "excess_annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_daily_turnover": 0.0,
            "total_turnover": 0.0,
        }

    final_equity = float(equity.iloc[-1])
    periods = max(len(clean), 1)
    annualized_return = (final_equity ** (252.0 / periods) - 1.0) if final_equity > 0 else -1.0
    annualized_vol = float(clean.std(ddof=0) * math.sqrt(252.0))
    rf_daily = (1.0 + risk_free_rate_annual) ** (1.0 / 252.0) - 1.0
    excess_annualized_return = float((clean - rf_daily).mean() * 252.0)
    sharpe = float(excess_annualized_return / annualized_vol) if annualized_vol > 1e-12 else 0.0

    return {
        "total_return": final_equity - 1.0,
        "annualized_return": float(annualized_return),
        "excess_annualized_return": excess_annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(equity),
        "avg_daily_turnover": float(turnover.mean()) if not turnover.empty else 0.0,
        "total_turnover": float(turnover.sum()) if not turnover.empty else 0.0,
    }


def _load_prices(params: dict[str, object], end_date: str | None) -> pd.DataFrame:
    tickers = list(params["tickers"])
    prices = load_price_data(
        tickers=tickers,
        start_date=str(params["start_date"]),
        end_date=end_date or params.get("end_date"),
        cache_path=PRICE_CACHE_PATH,
        use_cache=True,
        prefer_live=False,
        allow_cache_fallback=True,
        force_refresh=False,
    )
    return prices.reindex(columns=tickers).sort_index().ffill(limit=3)


def _build_expected_excess(
    table: pd.DataFrame,
    cfg: ReplayConfig,
    *,
    v4_today: pd.DataFrame | None = None,
) -> pd.Series:
    """Build expected excess-return proxy for the optimizer.

    v2/v3 use tactical_forecast score tables.
    v4/v4b use predictive-feature research scores as score overlays,
    while still anchoring the expected return proxy to the existing 10d excess-return diagnostic.
    """

    table = table.set_index("ticker", drop=False)
    base_table = _build_tactical_score_v3_table(table.reset_index(drop=True)).set_index("ticker")
    base = base_table.get("excess_expected_return_10d", pd.Series(0.0, index=base_table.index)).astype(float)

    score_name = cfg.score_name.lower()
    if score_name == "v2":
        scored = _build_tactical_score_v2_table(table.reset_index(drop=True)).set_index("ticker")
        score = scored["tactical_score_v2_candidate"]
    elif score_name == "v3":
        scored = base_table
        score = scored["tactical_score_v3_candidate"]
    elif score_name in {"v4", "v4b"}:
        if v4_today is None or v4_today.empty:
            raise ValueError("v4_today is required for score_name v4/v4b")
        scored = v4_today.set_index("ticker")
        col = "tactical_score_v4_candidate" if score_name == "v4" else "tactical_score_v4b_candidate"
        score = scored[col].reindex(base.index).fillna(0.0)
    else:
        raise ValueError("score_name must be v2, v3, v4 or v4b")

    # Cross-sectional expected-return tilt. Base gives the model a return anchor;
    # score_tilt lets v2/v3/v4/v4b change ranking/strength.
    score_tilt = cfg.signal_scale * _zscore(score.reindex(base.index).fillna(0.0))
    expected = (base + score_tilt).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Gate to top-N by score, unless top_n_gate <= 0.
    if cfg.top_n_gate > 0:
        keep = set(score.sort_values(ascending=False).head(cfg.top_n_gate).index)
        expected = expected.where(expected.index.isin(keep), other=-1.0)

    return expected


def _covariance_matrix(
    returns: pd.DataFrame,
    tickers: list[str],
    date_position: int,
    window: int,
) -> pd.DataFrame:
    start = max(0, date_position - window + 1)
    sample = returns.iloc[start : date_position + 1].reindex(columns=tickers).fillna(0.0)
    cov = sample.cov(ddof=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Small diagonal ridge for numerical stability.
    ridge = max(float(np.nanmean(np.diag(cov.to_numpy(dtype=float)))) * 0.05, 1e-8)
    cov = cov + pd.DataFrame(np.eye(len(tickers)) * ridge, index=tickers, columns=tickers)
    return cov


def _optimize_weights_gurobi(
    *,
    expected_excess: pd.Series,
    covariance: pd.DataFrame,
    previous_weights: pd.Series,
    cfg: ReplayConfig,
) -> pd.Series:
    if gp is None:
        raise RuntimeError("gurobipy is not available.")

    tickers = list(expected_excess.index)
    mu = expected_excess.reindex(tickers).astype(float).to_numpy()
    cov = covariance.reindex(index=tickers, columns=tickers).fillna(0.0).to_numpy(dtype=float)
    prev = previous_weights.reindex(tickers).fillna(0.0).astype(float).to_numpy()

    model = gp.Model("tactical_replay")
    model.Params.OutputFlag = 0

    w = model.addMVar(len(tickers), lb=0.0, ub=float(cfg.max_weight), name="w")
    u = model.addMVar(len(tickers), lb=0.0, name="turnover_abs")

    # Fully invested long-only portfolio.
    model.addConstr(w.sum() == 1.0)

    # Absolute turnover linearization.
    model.addConstr(u >= w - prev)
    model.addConstr(u >= prev - w)

    previous_is_feasible = (
        float(np.max(prev)) <= float(cfg.max_weight) + 1e-9
        and abs(float(np.sum(prev)) - 1.0) <= 1e-6
    )
    if cfg.max_rebalance_turnover is not None and previous_is_feasible:
        model.addConstr(0.5 * u.sum() <= float(cfg.max_rebalance_turnover))

    expected_term = mu @ w
    variance_term = w @ cov @ w
    concentration_term = w @ w
    turnover_term = 0.5 * u.sum()

    objective = (
        expected_term
        - float(cfg.lambda_variance) * variance_term
        - float(cfg.lambda_turnover) * turnover_term
        - float(cfg.lambda_concentration) * concentration_term
    )
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        # If constrained solve fails, fall back to a feasible equal-weight top expected-excess allocation
        # rather than silently staying in an infeasible starting portfolio forever.
        fallback = expected_excess.sort_values(ascending=False).head(max(int(1.0 / cfg.max_weight), 1))
        weights = pd.Series(0.0, index=tickers, dtype=float)
        if not fallback.empty:
            weights.loc[fallback.index] = 1.0 / len(fallback)
        else:
            weights[:] = 1.0 / len(tickers)
        return weights.reindex(tickers).fillna(0.0)

    weights = pd.Series(np.asarray(w.X).reshape(-1), index=tickers, dtype=float)
    weights = weights.clip(lower=0.0)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    return weights


def run_tactical_gurobi_replay(cfg: ReplayConfig) -> dict[str, Path]:
    params = build_params()
    prices = _load_prices(params, cfg.end_date)
    tickers = list(prices.columns)
    returns = compute_returns(prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    all_dates = list(prices.index)
    v4_panel = build_v4_feature_scores(prices) if cfg.score_name.lower() in {"v4", "v4b"} else pd.DataFrame()

    start_ts = pd.Timestamp(cfg.start_date)
    eligible_dates = [
        date
        for pos, date in enumerate(all_dates)
        if pos >= cfg.min_history and pd.Timestamp(date) >= start_ts and pos + 1 < len(all_dates)
    ]

    current_weights = pd.Series(0.0, index=tickers, dtype=float)
    # Start in SGOV if available, else equal weight as a neutral fallback.
    if "SGOV" in tickers:
        current_weights.loc["SGOV"] = 1.0
    else:
        current_weights[:] = 1.0 / len(tickers)

    equity = 1.0
    daily_rows: list[dict[str, object]] = []
    alloc_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    turnover_values: list[float] = []

    for i, date in enumerate(eligible_dates):
        date_ts = pd.Timestamp(date)
        pos = all_dates.index(date)
        should_rebalance = i % max(cfg.rebalance_every, 1) == 0

        if should_rebalance:
            hist_prices = prices.iloc[: pos + 1].reindex(columns=tickers)
            hist_returns = returns.iloc[: pos + 1].reindex(columns=tickers)

            tactical = build_multi_horizon_forecast(
                prices=hist_prices,
                returns=hist_returns,
                date=date_ts,
                params=params,
                tickers=tickers,
            )
            v4_today = (
                v4_panel[v4_panel["date"].eq(pd.Timestamp(date_ts))]
                if cfg.score_name.lower() in {"v4", "v4b"} and not v4_panel.empty
                else None
            )
            expected = _build_expected_excess(
                tactical.table,
                cfg,
                v4_today=v4_today,
            ).reindex(tickers).fillna(-1.0)
            cov = _covariance_matrix(returns, tickers, pos, cfg.covariance_window)

            target = _optimize_weights_gurobi(
                expected_excess=expected,
                covariance=cov,
                previous_weights=current_weights,
                cfg=cfg,
            ).reindex(tickers).fillna(0.0)

            turnover = float((target - current_weights).abs().sum() / 2.0)
            turnover_values.append(turnover)

            for ticker in tickers:
                delta = float(target.loc[ticker] - current_weights.loc[ticker])
                if abs(delta) > 1e-6:
                    trade_rows.append(
                        {
                            "date": str(date_ts.date()),
                            "ticker": ticker,
                            "weight_before": float(current_weights.loc[ticker]),
                            "weight_after": float(target.loc[ticker]),
                            "weight_delta": delta,
                            "expected_excess": float(expected.loc[ticker]),
                        }
                    )

            current_weights = target

            nonzero = current_weights[current_weights.abs() > 1e-8].sort_values(ascending=False)
            for ticker, weight in nonzero.items():
                alloc_rows.append(
                    {
                        "date": str(date_ts.date()),
                        "ticker": ticker,
                        "weight": float(weight),
                    }
                )

        next_date = all_dates[pos + 1]
        next_ret = returns.loc[next_date].reindex(tickers).fillna(0.0)
        portfolio_return = float((current_weights * next_ret).sum())
        equity *= 1.0 + portfolio_return

        daily_rows.append(
            {
                "date": str(pd.Timestamp(next_date).date()),
                "portfolio_return": portfolio_return,
                "equity": equity,
            }
        )

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    allocations = pd.DataFrame(alloc_rows)

    stats = _performance_stats(
        daily["portfolio_return"].astype(float),
        daily["equity"].astype(float),
        pd.Series(turnover_values, dtype=float),
        risk_free_rate_annual=cfg.risk_free_rate_annual,
    )
    summary = pd.DataFrame(
        [
            {
                "strategy": f"gurobi_{cfg.score_name}",
                "score_name": cfg.score_name,
                "rebalance_every": cfg.rebalance_every,
                "top_n_gate": cfg.top_n_gate,
                "max_weight": cfg.max_weight,
                "max_rebalance_turnover": cfg.max_rebalance_turnover,
                "lambda_variance": cfg.lambda_variance,
                "lambda_turnover": cfg.lambda_turnover,
                "lambda_concentration": cfg.lambda_concentration,
                "signal_scale": cfg.signal_scale,
                **stats,
            }
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = OUTPUT_DIR / f"{cfg.output_prefix}_daily_equity_curve.csv"
    trades_path = OUTPUT_DIR / f"{cfg.output_prefix}_trades.csv"
    allocations_path = OUTPUT_DIR / f"{cfg.output_prefix}_allocations.csv"
    summary_path = OUTPUT_DIR / f"{cfg.output_prefix}_summary.csv"
    report_path = OUTPUT_DIR / f"{cfg.output_prefix}_report.txt"

    daily.to_csv(daily_path, index=False)
    trades.to_csv(trades_path, index=False)
    allocations.to_csv(allocations_path, index=False)
    summary.to_csv(summary_path, index=False)

    lines = [
        "Tactical Gurobi Allocation Replay Report",
        "",
        "status: research_only_no_order_change",
        f"score_name: {cfg.score_name}",
        f"start_date: {cfg.start_date}",
        f"end_date: {str(pd.Timestamp(eligible_dates[-1]).date()) if eligible_dates else 'n/a'}",
        f"rebalance_every: {cfg.rebalance_every}",
        f"top_n_gate: {cfg.top_n_gate}",
        f"max_weight: {cfg.max_weight:.4f}",
        f"max_rebalance_turnover: {cfg.max_rebalance_turnover if cfg.max_rebalance_turnover is not None else 'none'}",
        f"risk_free_rate_annual: {cfg.risk_free_rate_annual:.4f}",
        f"lambda_variance: {cfg.lambda_variance:.4f}",
        f"lambda_turnover: {cfg.lambda_turnover:.4f}",
        f"lambda_concentration: {cfg.lambda_concentration:.4f}",
        f"signal_scale: {cfg.signal_scale:.4f}",
        "",
        "method:",
        "- Builds tactical v2/v3 scores using only historical data available at each rebalance date.",
        "- Converts score plus excess-return forecast into expected excess-return proxy.",
        "- Gurobi solves long-only constrained mean-variance allocation with turnover and concentration penalties.",
        "- This is still research-only and does not alter production orders.",
        "",
        "summary:",
    ]

    row = summary.iloc[0]
    lines.append(
        f"- total_return={row['total_return']:.4f}, ann_return={row['annualized_return']:.4f}, "
        f"excess_ann_return={row['excess_annualized_return']:.4f}, ann_vol={row['annualized_vol']:.4f}, "
        f"sharpe={row['sharpe']:.3f}, max_dd={row['max_drawdown']:.4f}, turnover={row['total_turnover']:.2f}"
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "daily": daily_path,
        "trades": trades_path,
        "allocations": allocations_path,
        "summary": summary_path,
        "report": report_path,
    }


def parse_args() -> ReplayConfig:
    parser = argparse.ArgumentParser(description="Run tactical Gurobi allocation replay.")
    parser.add_argument("--score-name", choices=["v2", "v3", "v4", "v4b"], default="v3")
    parser.add_argument("--rebalance-every", type=int, default=1)
    parser.add_argument("--top-n-gate", type=int, default=6)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--max-rebalance-turnover", type=float, default=0.30)
    parser.add_argument("--lambda-variance", type=float, default=8.0)
    parser.add_argument("--lambda-turnover", type=float, default=0.03)
    parser.add_argument("--lambda-concentration", type=float, default=0.02)
    parser.add_argument("--signal-scale", type=float, default=0.015)
    parser.add_argument("--output-prefix", default="tactical_gurobi_replay")
    args = parser.parse_args()

    return ReplayConfig(
        score_name=args.score_name,
        rebalance_every=args.rebalance_every,
        top_n_gate=args.top_n_gate,
        max_weight=args.max_weight,
        max_rebalance_turnover=args.max_rebalance_turnover,
        lambda_variance=args.lambda_variance,
        lambda_turnover=args.lambda_turnover,
        lambda_concentration=args.lambda_concentration,
        signal_scale=args.signal_scale,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    cfg = parse_args()
    paths = run_tactical_gurobi_replay(cfg)
    print("Tactical Gurobi replay outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")
