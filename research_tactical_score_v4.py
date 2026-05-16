"""Research-only tactical score v4 candidate.

Purpose:
- Build a robust v4 tactical score from predictive features that worked across
  multiple horizons in the feature research.
- Compare v3 vs v4 using Top-N portfolio backtests.
- Does not alter production Daily Bot orders.

v4 is intentionally simple and interpretable:
- volatility leadership
- 60d trend
- path efficiency
- moving-average trend
- cross-sectional volatility rank
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
from tactical_forecast import build_multi_horizon_forecast, _build_tactical_score_v3_table


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PRICE_CACHE_PATH = BASE_DIR / "data" / "prices_cache.csv"


@dataclass(frozen=True)
class V4Config:
    start_date: str = "2024-01-01"
    end_date: str | None = None
    min_history: int = 180
    rebalance_every: int = 1
    top_n: int = 8
    max_weight: float = 0.25
    max_rebalance_turnover: float | None = 0.30
    risk_free_rate_annual: float = 0.02
    output_prefix: str = "tactical_score_v4_research"


def _zscore(values: pd.Series) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan)
    std = float(clean.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=values.index, dtype=float)
    return ((clean - float(clean.mean())) / std).fillna(0.0)


def _rolling_return(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    return prices / prices.shift(days) - 1.0


def _rolling_vol(returns: pd.DataFrame, days: int) -> pd.DataFrame:
    return returns.rolling(days).std(ddof=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _path_efficiency(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    net = (prices - prices.shift(days)).abs()
    step = prices.diff().abs().rolling(days).sum()
    out = net / step
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _ma_dist(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    ma = prices.rolling(days).mean()
    out = prices / ma - 1.0
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _ma_cross_10_50(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.rolling(10).mean() / prices.rolling(50).mean() - 1.0
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _cross_sectional_rank(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    return df.rank(axis=1, pct=True, ascending=ascending).fillna(0.0)


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
        "annualized_return": annualized_return,
        "excess_annualized_return": excess_annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(equity),
        "avg_daily_turnover": float(turnover.mean()) if not turnover.empty else 0.0,
        "total_turnover": float(turnover.sum()) if not turnover.empty else 0.0,
    }


def _equal_weight(tickers: list[str]) -> pd.Series:
    if not tickers:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(tickers), index=tickers, dtype=float)


def _top_n_weights(score: pd.Series, n: int, max_weight: float) -> pd.Series:
    clean = score.dropna().astype(float)
    if clean.empty:
        return pd.Series(dtype=float)
    selected = list(clean.sort_values(ascending=False).head(n).index)
    weights = _equal_weight(selected)
    if max_weight > 0:
        weights = weights.clip(upper=max_weight)
        if weights.sum() > 0:
            weights = weights / weights.sum()
    return weights


def _apply_turnover_cap(
    target: pd.Series,
    previous: pd.Series,
    max_rebalance_turnover: float | None,
) -> pd.Series:
    target = target.reindex(previous.index).fillna(0.0)
    previous = previous.reindex(target.index).fillna(0.0)

    if max_rebalance_turnover is None:
        return target

    turnover = float((target - previous).abs().sum() / 2.0)
    if turnover <= max_rebalance_turnover or turnover <= 1e-12:
        return target

    blend = float(max_rebalance_turnover / turnover)
    out = previous + blend * (target - previous)
    if out.sum() > 0:
        out = out / out.sum()
    return out


def load_prices(params: dict[str, object], end_date: str | None) -> pd.DataFrame:
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


def build_v4_feature_scores(prices: pd.DataFrame) -> pd.DataFrame:
    returns = compute_returns(prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    returns = returns.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

    ret_60d = _rolling_return(prices, 60)
    vol_5d = _rolling_vol(returns, 5)
    vol_20d = _rolling_vol(returns, 20)
    vol_60d = _rolling_vol(returns, 60)
    vol_rank_20d = _cross_sectional_rank(vol_20d, ascending=False)
    path_eff_10d = _path_efficiency(prices, 10)
    path_eff_20d = _path_efficiency(prices, 20)
    ma_cross = _ma_cross_10_50(prices)
    ma_dist_50d = _ma_dist(prices, 50)

    # Defensive alignment: all feature frames must support every prices.index date.
    # compute_returns may drop the first price row; reindex avoids timestamp KeyErrors.
    feature_frames = [
        ret_60d,
        vol_5d,
        vol_20d,
        vol_60d,
        vol_rank_20d,
        path_eff_10d,
        path_eff_20d,
        ma_cross,
        ma_dist_50d,
    ]
    feature_frames = [
        frame.reindex(index=prices.index, columns=prices.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for frame in feature_frames
    ]
    (
        ret_60d,
        vol_5d,
        vol_20d,
        vol_60d,
        vol_rank_20d,
        path_eff_10d,
        path_eff_20d,
        ma_cross,
        ma_dist_50d,
    ) = feature_frames

    rows = []
    for date in prices.index:
        frame = pd.DataFrame({"ticker": list(prices.columns)})
        frame["date"] = pd.Timestamp(date)

        raw = {
            "ret_60d": ret_60d.loc[date],
            "vol_5d": vol_5d.loc[date],
            "vol_20d": vol_20d.loc[date],
            "vol_60d": vol_60d.loc[date],
            "cross_sectional_rank_vol_20d": vol_rank_20d.loc[date],
            "path_efficiency_10d": path_eff_10d.loc[date],
            "path_efficiency_20d": path_eff_20d.loc[date],
            "ma_cross_10_50": ma_cross.loc[date],
            "ma_dist_50d": ma_dist_50d.loc[date],
        }

        for name, series in raw.items():
            frame[name] = series.reindex(prices.columns).astype(float).to_numpy()

        # Horizon-robust score based on feature research:
        # Volatility/risk appetite and path trend worked best across 3d/5d/10d.
        # v4a: raw predictive-feature score from univariate forecast research.
        score_v4 = (
            0.24 * _zscore(frame["vol_20d"])
            + 0.20 * _zscore(frame["vol_60d"])
            + 0.14 * _zscore(frame["vol_5d"])
            + 0.14 * _zscore(frame["ret_60d"])
            + 0.10 * _zscore(frame["path_efficiency_20d"])
            + 0.06 * _zscore(frame["path_efficiency_10d"])
            + 0.08 * _zscore(frame["ma_cross_10_50"])
            + 0.04 * _zscore(frame["ma_dist_50d"])
        )

        # v4b: risk-balanced version.
        # The first v4 test showed raw volatility leadership improved return but hurt Sharpe/drawdown.
        # Keep volatility information, but penalize extreme high-vol concentration.
        vol_blend = (
            0.50 * _zscore(frame["vol_20d"])
            + 0.30 * _zscore(frame["vol_60d"])
            + 0.20 * _zscore(frame["vol_5d"])
        )
        extreme_vol_penalty = _zscore(frame["cross_sectional_rank_vol_20d"].clip(lower=0.80))

        score_v4b = (
            0.12 * vol_blend
            + 0.24 * _zscore(frame["ret_60d"])
            + 0.18 * _zscore(frame["path_efficiency_20d"])
            + 0.10 * _zscore(frame["path_efficiency_10d"])
            + 0.20 * _zscore(frame["ma_cross_10_50"])
            + 0.10 * _zscore(frame["ma_dist_50d"])
            - 0.16 * extreme_vol_penalty
        )

        frame["tactical_score_v4_candidate"] = score_v4.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frame["tactical_rank_v4_candidate"] = frame["tactical_score_v4_candidate"].rank(
            ascending=False,
            method="dense",
        ).astype(int)

        frame["tactical_score_v4b_candidate"] = score_v4b.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frame["tactical_rank_v4b_candidate"] = frame["tactical_score_v4b_candidate"].rank(
            ascending=False,
            method="dense",
        ).astype(int)

        rows.append(frame)

    return pd.concat(rows, ignore_index=True)


def run_v4_research(cfg: V4Config) -> dict[str, Path]:
    params = build_params()
    prices = load_prices(params, cfg.end_date)
    tickers = list(prices.columns)
    returns = compute_returns(prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    v4_panel = build_v4_feature_scores(prices)

    all_dates = list(prices.index)
    eligible_dates = [
        date
        for pos, date in enumerate(all_dates)
        if pos >= cfg.min_history
        and pd.Timestamp(date) >= pd.Timestamp(cfg.start_date)
        and pos + 1 < len(all_dates)
    ]

    strategies = [
        "equal_weight",
        "tactical_v3_top_n",
        "tactical_v4_top_n",
        "tactical_v4_cap_top_n",
        "tactical_v4b_top_n",
        "tactical_v4b_cap_top_n",
    ]

    weights = {s: pd.Series(0.0, index=tickers, dtype=float) for s in strategies}
    equity = {s: 1.0 for s in strategies}
    daily_rows = []
    trade_rows = []
    turnover = {s: [] for s in strategies}

    for i, date in enumerate(eligible_dates):
        pos = all_dates.index(date)
        should_rebalance = i % max(cfg.rebalance_every, 1) == 0

        if should_rebalance:
            hist_prices = prices.iloc[: pos + 1].reindex(columns=tickers)
            hist_returns = returns.iloc[: pos + 1].reindex(columns=tickers)
            tactical = build_multi_horizon_forecast(
                prices=hist_prices,
                returns=hist_returns,
                date=date,
                params=params,
                tickers=tickers,
            )
            v3_table = _build_tactical_score_v3_table(tactical.table).set_index("ticker")
            v4_table = v4_panel[v4_panel["date"].eq(pd.Timestamp(date))].set_index("ticker")

            targets = {
                "equal_weight": _equal_weight(tickers),
                "tactical_v3_top_n": _top_n_weights(v3_table["tactical_score_v3_candidate"], cfg.top_n, cfg.max_weight),
                "tactical_v4_top_n": _top_n_weights(v4_table["tactical_score_v4_candidate"], cfg.top_n, cfg.max_weight),
                "tactical_v4_cap_top_n": _top_n_weights(v4_table["tactical_score_v4_candidate"], cfg.top_n, cfg.max_weight),
                "tactical_v4b_top_n": _top_n_weights(v4_table["tactical_score_v4b_candidate"], cfg.top_n, cfg.max_weight),
                "tactical_v4b_cap_top_n": _top_n_weights(v4_table["tactical_score_v4b_candidate"], cfg.top_n, cfg.max_weight),
            }

            for strategy in strategies:
                target = targets[strategy].reindex(tickers).fillna(0.0)
                if strategy in {"tactical_v4_cap_top_n", "tactical_v4b_cap_top_n"}:
                    target = _apply_turnover_cap(target, weights[strategy], cfg.max_rebalance_turnover)

                one_way = float((target - weights[strategy]).abs().sum() / 2.0)
                turnover[strategy].append(one_way)

                for ticker in tickers:
                    delta = float(target.loc[ticker] - weights[strategy].loc[ticker])
                    if abs(delta) > 1e-8:
                        trade_rows.append(
                            {
                                "date": str(pd.Timestamp(date).date()),
                                "strategy": strategy,
                                "ticker": ticker,
                                "weight_before": float(weights[strategy].loc[ticker]),
                                "weight_after": float(target.loc[ticker]),
                                "weight_delta": delta,
                            }
                        )

                weights[strategy] = target

        next_date = all_dates[pos + 1]
        next_ret = returns.loc[next_date].reindex(tickers).fillna(0.0)

        for strategy in strategies:
            portfolio_return = float((weights[strategy] * next_ret).sum())
            equity[strategy] *= 1.0 + portfolio_return
            daily_rows.append(
                {
                    "date": str(pd.Timestamp(next_date).date()),
                    "strategy": strategy,
                    "portfolio_return": portfolio_return,
                    "equity": equity[strategy],
                }
            )

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)

    summary_rows = []
    for strategy in strategies:
        sub = daily[daily["strategy"].eq(strategy)]
        stats = _performance_stats(
            sub["portfolio_return"].astype(float),
            sub["equity"].astype(float),
            pd.Series(turnover[strategy], dtype=float),
            cfg.risk_free_rate_annual,
        )
        summary_rows.append(
            {
                "strategy": strategy,
                "rebalance_every": cfg.rebalance_every,
                "top_n": cfg.top_n,
                "max_weight": cfg.max_weight,
                "max_rebalance_turnover": cfg.max_rebalance_turnover if strategy in {"tactical_v4_cap_top_n", "tactical_v4b_cap_top_n"} else None,
                **stats,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False)

    current = v4_panel[v4_panel["date"].eq(v4_panel["date"].max())].copy()
    current = current.sort_values("tactical_score_v4_candidate", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = OUTPUT_DIR / f"{cfg.output_prefix}_daily_equity_curve.csv"
    trades_path = OUTPUT_DIR / f"{cfg.output_prefix}_trades.csv"
    summary_path = OUTPUT_DIR / f"{cfg.output_prefix}_summary.csv"
    current_path = OUTPUT_DIR / f"{cfg.output_prefix}_current_scores.csv"
    report_path = OUTPUT_DIR / f"{cfg.output_prefix}_report.txt"

    daily.to_csv(daily_path, index=False)
    trades.to_csv(trades_path, index=False)
    summary.to_csv(summary_path, index=False)
    current.to_csv(current_path, index=False)

    lines = [
        "Tactical Score v4 Research Report",
        "",
        "status: research_only_no_order_change",
        f"start_date: {cfg.start_date}",
        f"rebalance_every: {cfg.rebalance_every}",
        f"top_n: {cfg.top_n}",
        f"max_weight: {cfg.max_weight:.4f}",
        f"max_rebalance_turnover: {cfg.max_rebalance_turnover}",
        f"risk_free_rate_annual: {cfg.risk_free_rate_annual:.4f}",
        "",
        "method:",
        "- v4 combines robust univariate predictive features found across 1d/3d/5d/10d horizons.",
        "- v4b is a risk-balanced variant that reduces raw volatility exposure and penalizes extreme-vol assets.",
        "- It does not use global Ridge because Ridge was weak/negative in early forecast research.",
        "- This does not alter production Daily Bot orders.",
        "",
        "v4_weights:",
        "- vol_20d: +0.24",
        "- vol_60d: +0.20",
        "- vol_5d: +0.14",
        "- ret_60d: +0.14",
        "- path_efficiency_20d: +0.10",
        "- path_efficiency_10d: +0.06",
        "- ma_cross_10_50: +0.08",
        "- ma_dist_50d: +0.04",
        "",
        "v4b_weights:",
        "- blended_volatility: +0.12",
        "- ret_60d: +0.24",
        "- path_efficiency_20d: +0.18",
        "- path_efficiency_10d: +0.10",
        "- ma_cross_10_50: +0.20",
        "- ma_dist_50d: +0.10",
        "- extreme_vol_penalty: -0.16",
        "",
        "summary:",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.strategy}: return={row.total_return:.4f}, "
            f"ann_return={row.annualized_return:.4f}, vol={row.annualized_vol:.4f}, "
            f"sharpe={row.sharpe:.3f}, max_dd={row.max_drawdown:.4f}, "
            f"turnover={row.total_turnover:.2f}"
        )

    lines.extend(["", "current_top_v4_assets:"])
    for row in current.head(15).itertuples(index=False):
        lines.append(
            f"- {row.ticker}: v4_rank={row.tactical_rank_v4_candidate}, "
            f"v4_score={row.tactical_score_v4_candidate:.4f}, "
            f"v4b_rank={row.tactical_rank_v4b_candidate}, "
            f"v4b_score={row.tactical_score_v4b_candidate:.4f}, "
            f"vol20={row.vol_20d:.4f}, ret60={row.ret_60d:.4f}, "
            f"path20={row.path_efficiency_20d:.4f}, ma_cross={row.ma_cross_10_50:.4f}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "daily": daily_path,
        "trades": trades_path,
        "summary": summary_path,
        "current_scores": current_path,
        "report": report_path,
    }


def parse_args() -> V4Config:
    parser = argparse.ArgumentParser(description="Research tactical score v4 candidate.")
    parser.add_argument("--rebalance-every", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--max-rebalance-turnover", type=float, default=0.30)
    parser.add_argument("--output-prefix", default="tactical_score_v4_research")
    args = parser.parse_args()
    return V4Config(
        rebalance_every=args.rebalance_every,
        top_n=args.top_n,
        max_weight=args.max_weight,
        max_rebalance_turnover=args.max_rebalance_turnover,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    cfg = parse_args()
    paths = run_v4_research(cfg)
    print("Tactical score v4 research outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")
