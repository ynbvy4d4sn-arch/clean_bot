"""Backtest tactical score ranking strategies.

This is a research-only backtest. It does not place orders and does not change
Daily Bot allocation logic.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd

from config import build_params
from data import load_price_data
from features import compute_returns
from tactical_forecast import build_multi_horizon_forecast, _build_tactical_score_v2_table


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PRICE_CACHE_PATH = BASE_DIR / "data" / "prices_cache.csv"


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _performance_stats(
    daily_returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
    risk_free_rate_annual: float = 0.02,
) -> dict[str, float]:
    clean = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "excess_annualized_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_daily_turnover": 0.0,
            "total_turnover": 0.0,
        }

    total_return = float(equity.iloc[-1] - 1.0)
    periods = max(len(clean), 1)
    annualized_return = float((equity.iloc[-1] ** (252.0 / periods)) - 1.0) if equity.iloc[-1] > 0 else -1.0
    annualized_vol = float(clean.std(ddof=0) * math.sqrt(252.0))
    risk_free_daily = float((1.0 + risk_free_rate_annual) ** (1.0 / 252.0) - 1.0)
    excess_daily = clean - risk_free_daily
    excess_annualized_return = float(excess_daily.mean() * 252.0)
    sharpe = float((excess_daily.mean() * 252.0) / annualized_vol) if annualized_vol > 1e-12 else 0.0
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "excess_annualized_return": excess_annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(equity),
        "avg_daily_turnover": float(turnover.mean()) if not turnover.empty else 0.0,
        "total_turnover": float(turnover.sum()) if not turnover.empty else 0.0,
    }


def _equal_weight(index: list[str]) -> pd.Series:
    if not index:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(index), index=index, dtype=float)


def _top_n_weights(score: pd.Series, n: int, max_weight: float = 0.25) -> pd.Series:
    score = score.dropna().astype(float)
    if score.empty:
        return pd.Series(dtype=float)
    selected = list(score.sort_values(ascending=False).head(n).index)
    weights = _equal_weight(selected)
    if max_weight > 0:
        weights = weights.clip(upper=max_weight)
        if weights.sum() > 0:
            weights = weights / weights.sum()
    return weights


def run_tactical_score_backtest(
    *,
    start_date: str = "2024-01-01",
    end_date: str | None = None,
    rebalance_every: int = 5,
    top_n: int = 8,
    min_history: int = 180,
    risk_free_rate_annual: float = 0.02,
    max_weight: float = 0.25,
    output_prefix: str = "tactical_backtest",
) -> dict[str, Path]:
    params = build_params()
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
    ).reindex(columns=tickers).sort_index().ffill(limit=3)

    returns = compute_returns(prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bt_prices = prices.loc[pd.Timestamp(start_date):].dropna(how="all")
    if bt_prices.empty:
        raise ValueError(f"No prices available from start_date={start_date}")

    all_dates = list(prices.index)
    bt_dates = [date for date in bt_prices.index if date in returns.index]
    strategies = [
        "sgov_cash",
        "equal_weight",
        "tactical_v1_top_n",
        "tactical_v2_top_n",
    ]

    weights_by_strategy = {name: pd.Series(dtype=float) for name in strategies}
    equity = {name: 1.0 for name in strategies}
    daily_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

    last_weights = {name: pd.Series(0.0, index=tickers, dtype=float) for name in strategies}

    for idx, date in enumerate(bt_dates):
        global_pos = all_dates.index(date)
        if global_pos < min_history:
            continue
        next_pos = global_pos + 1
        if next_pos >= len(prices.index):
            break

        should_rebalance = idx % max(rebalance_every, 1) == 0

        if should_rebalance:
            hist_prices = prices.iloc[: global_pos + 1].reindex(columns=tickers)
            hist_returns = returns.iloc[: global_pos + 1].reindex(columns=tickers)

            tactical = build_multi_horizon_forecast(
                prices=hist_prices,
                returns=hist_returns,
                date=date,
                params=params,
                tickers=tickers,
            )
            table = _build_tactical_score_v2_table(tactical.table).set_index("ticker")

            weights_by_strategy["sgov_cash"] = pd.Series({"SGOV": 1.0}, dtype=float) if "SGOV" in tickers else _equal_weight(tickers)
            weights_by_strategy["equal_weight"] = _equal_weight(tickers)
            weights_by_strategy["tactical_v1_top_n"] = _top_n_weights(table["tactical_score"], n=top_n, max_weight=max_weight)
            weights_by_strategy["tactical_v2_top_n"] = _top_n_weights(table["tactical_score_v2_candidate"], n=top_n, max_weight=max_weight)

            for strategy in strategies:
                current = weights_by_strategy[strategy].reindex(tickers).fillna(0.0)
                previous = last_weights[strategy].reindex(tickers).fillna(0.0)
                turnover = float((current - previous).abs().sum() / 2.0)
                changed = current[current.abs() > 1e-12]
                trade_rows.append(
                    {
                        "date": str(pd.Timestamp(date).date()),
                        "strategy": strategy,
                        "turnover": turnover,
                        "holdings": ",".join(changed.sort_values(ascending=False).index.tolist()),
                    }
                )
                last_weights[strategy] = current.copy()

        next_date = prices.index[next_pos]
        next_ret = returns.loc[next_date].reindex(tickers).fillna(0.0)

        row: dict[str, object] = {
            "date": str(pd.Timestamp(next_date).date()),
        }
        for strategy in strategies:
            w = weights_by_strategy[strategy].reindex(tickers).fillna(0.0)
            portfolio_return = float((w * next_ret).sum())
            equity[strategy] *= 1.0 + portfolio_return
            row[f"{strategy}_return"] = portfolio_return
            row[f"{strategy}_equity"] = equity[strategy]
        daily_rows.append(row)

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)

    summary_rows: list[dict[str, object]] = []
    for strategy in strategies:
        ret = daily[f"{strategy}_return"].astype(float)
        eq = daily[f"{strategy}_equity"].astype(float)
        if trades.empty:
            turnover = pd.Series(dtype=float)
        else:
            turnover = trades.loc[trades["strategy"].eq(strategy), "turnover"].astype(float)
        stats = _performance_stats(ret, eq, turnover, risk_free_rate_annual=risk_free_rate_annual)
        summary_rows.append({"strategy": strategy, **stats})

    summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = OUTPUT_DIR / f"{output_prefix}_daily_equity_curve.csv"
    trades_path = OUTPUT_DIR / f"{output_prefix}_trades.csv"
    summary_path = OUTPUT_DIR / f"{output_prefix}_summary.csv"
    report_path = OUTPUT_DIR / f"{output_prefix}_report.txt"

    daily.to_csv(daily_path, index=False)
    trades.to_csv(trades_path, index=False)
    summary.to_csv(summary_path, index=False)

    lines = [
        "Tactical Score Portfolio Backtest Report",
        "",
        "status: research_only_no_order_change",
        f"start_date: {start_date}",
        f"end_date: {str(pd.Timestamp(bt_dates[-1]).date()) if bt_dates else 'n/a'}",
        f"rebalance_every: {rebalance_every}",
        f"top_n: {top_n}",
        f"max_weight: {max_weight:.4f}",
        f"risk_free_rate_annual: {risk_free_rate_annual:.4f}",
        "",
        "method:",
        "- Rebalances every N trading days using only data available up to that date.",
        "- tactical_v1_top_n holds equal-weight top-N by tactical_score.",
        "- tactical_v2_top_n holds equal-weight top-N by tactical_score_v2_candidate.",
        "- Direct simulator fees are modeled as zero; turnover is still reported.",
        "- This is a simplified strategy backtest, not a full replay of the production optimizer.",
        "",
        "summary:",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.strategy}: total_return={row.total_return:.4f}, "
            f"ann_return={row.annualized_return:.4f}, excess_ann_return={row.excess_annualized_return:.4f}, "
            f"ann_vol={row.annualized_vol:.4f}, sharpe={row.sharpe:.3f}, max_dd={row.max_drawdown:.4f}, "
            f"total_turnover={row.total_turnover:.2f}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "daily": daily_path,
        "trades": trades_path,
        "summary": summary_path,
        "report": report_path,
    }




def run_tactical_score_backtest_grid(
    *,
    start_date: str = "2024-01-01",
    end_date: str | None = None,
    top_n_values: tuple[int, ...] = (4, 6, 8, 10, 12),
    rebalance_values: tuple[int, ...] = (1, 3, 5, 10),
    max_weight_values: tuple[float, ...] = (0.20, 0.25, 0.33),
    risk_free_rate_annual: float = 0.02,
) -> dict[str, Path]:
    """Run a parameter grid for tactical score backtests.

    Research-only. This helps detect whether a score works robustly or only
    under one hand-picked configuration.
    """

    rows: list[dict[str, object]] = []

    for top_n in top_n_values:
        for rebalance_every in rebalance_values:
            for max_weight in max_weight_values:
                result_paths = run_tactical_score_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    rebalance_every=rebalance_every,
                    top_n=top_n,
                    risk_free_rate_annual=risk_free_rate_annual,
                    max_weight=max_weight,
                    output_prefix=f"grid_top{top_n}_reb{rebalance_every}_mw{int(max_weight * 100)}",
                )
                summary = pd.read_csv(result_paths["summary"])
                for row in summary.to_dict(orient="records"):
                    rows.append(
                        {
                            "top_n": top_n,
                            "rebalance_every": rebalance_every,
                            "max_weight": max_weight,
                            **row,
                        }
                    )

    grid = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid_path = OUTPUT_DIR / "tactical_backtest_grid_summary.csv"
    report_path = OUTPUT_DIR / "tactical_backtest_grid_report.txt"
    grid.to_csv(grid_path, index=False)

    lines = [
        "Tactical Backtest Grid Report",
        "",
        "status: research_only_no_order_change",
        f"start_date: {start_date}",
        f"risk_free_rate_annual: {risk_free_rate_annual:.4f}",
        "",
        "method:",
        "- Tests multiple top_n, rebalance frequency and max-weight settings.",
        "- Sharpe uses excess return over 2% annual risk-free rate.",
        "- This is still a simplified tactical ranking backtest, not full production optimizer replay.",
        "",
    ]

    for strategy in sorted(grid["strategy"].unique()):
        subset = grid[grid["strategy"].eq(strategy)].copy()
        best = subset.sort_values("sharpe", ascending=False).head(5)
        lines.append(f"top_settings_{strategy}:")
        for row in best.itertuples(index=False):
            lines.append(
                f"- top_n={row.top_n}, rebalance={row.rebalance_every}, max_weight={row.max_weight:.2f}: "
                f"sharpe={row.sharpe:.3f}, total_return={row.total_return:.4f}, "
                f"ann_vol={row.annualized_vol:.4f}, max_dd={row.max_drawdown:.4f}, "
                f"turnover={row.total_turnover:.2f}"
            )
        lines.append("")

    v2 = grid[grid["strategy"].eq("tactical_v2_top_n")].copy()
    if not v2.empty:
        lines.extend(
            [
                "v2_robustness:",
                f"- median_sharpe: {float(v2['sharpe'].median()):.3f}",
                f"- mean_sharpe: {float(v2['sharpe'].mean()):.3f}",
                f"- min_sharpe: {float(v2['sharpe'].min()):.3f}",
                f"- max_sharpe: {float(v2['sharpe'].max()):.3f}",
                f"- positive_sharpe_share: {float((v2['sharpe'] > 0).mean()):.3f}",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "grid_summary": grid_path,
        "grid_report": report_path,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run tactical score research backtests.")
    parser.add_argument("--grid", action="store_true", help="Run the slower parameter grid.")
    parser.add_argument("--quick-grid", action="store_true", help="Run a small smoke-test grid.")
    args = parser.parse_args()

    paths = run_tactical_score_backtest()
    print("Backtest outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")

    if args.quick_grid:
        grid_paths = run_tactical_score_backtest_grid(
            top_n_values=(6, 8),
            rebalance_values=(3, 5),
            max_weight_values=(0.25,),
        )
        print("Quick grid outputs:")
        for name, path in grid_paths.items():
            print(f"- {name}: {path}")

    if args.grid:
        grid_paths = run_tactical_score_backtest_grid()
        print("Grid outputs:")
        for name, path in grid_paths.items():
            print(f"- {name}: {path}")
