"""Search constant tactical feature weights by walk-forward backtest.

Research-only. Does not change Daily Bot orders or production allocation logic.

Goal:
Find fixed feature weights that would have produced the best realized Sharpe
over a historical period, using only information available at each rebalance date.
Sharpe uses excess return over 2% annual risk-free rate by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
from typing import Iterable

import numpy as np
import pandas as pd

from config import build_params
from data import load_price_data
from features import compute_returns
from tactical_forecast import build_multi_horizon_forecast, _build_tactical_score_v2_table


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PRICE_CACHE_PATH = BASE_DIR / "data" / "prices_cache.csv"


FEATURE_COLUMNS = [
    "momentum_3d",
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "momentum_60d",
    "vol_5d",
    "vol_20d",
    "vol_adjusted_momentum_20d",
    "mean_reversion_score",
    "overextension_score",
    "trend_score",
    "drawdown_60d",
    "relative_strength_score",
    "forecast_confidence",
    "risk_adjusted_forecast",
    "excess_expected_return_3d",
    "excess_expected_return_5d",
    "excess_expected_return_10d",
    "excess_expected_return_to_project_end",
]


@dataclass(frozen=True)
class SearchConfig:
    start_date: str = "2024-01-01"
    train_end_date: str = "2025-06-30"
    end_date: str | None = None
    min_history: int = 180
    rebalance_every: int = 1
    top_n: int = 8
    max_weight: float = 0.25
    risk_free_rate_annual: float = 0.02
    random_trials: int = 250
    seed: int = 42


def _zscore_cross_section(values: pd.Series) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan)
    std = float(clean.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=values.index, dtype=float)
    mean = float(clean.mean())
    return ((clean - mean) / std).fillna(0.0)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


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

    periods = max(len(clean), 1)
    final_equity = float(equity.iloc[-1])
    total_return = final_equity - 1.0
    annualized_return = (final_equity ** (252.0 / periods) - 1.0) if final_equity > 0 else -1.0
    annualized_vol = float(clean.std(ddof=0) * math.sqrt(252.0))
    rf_daily = (1.0 + risk_free_rate_annual) ** (1.0 / 252.0) - 1.0
    excess_annualized_return = float((clean - rf_daily).mean() * 252.0)
    sharpe = float(excess_annualized_return / annualized_vol) if annualized_vol > 1e-12 else 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "excess_annualized_return": float(excess_annualized_return),
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(equity),
        "avg_daily_turnover": float(turnover.mean()) if not turnover.empty else 0.0,
        "total_turnover": float(turnover.sum()) if not turnover.empty else 0.0,
    }


def _top_n_weights(score: pd.Series, top_n: int, max_weight: float) -> pd.Series:
    clean = score.dropna().astype(float)
    if clean.empty:
        return pd.Series(dtype=float)
    selected = clean.sort_values(ascending=False).head(top_n).index
    weights = pd.Series(1.0 / len(selected), index=selected, dtype=float)
    weights = weights.clip(upper=max_weight)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    return weights


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


def build_feature_panel(
    *,
    prices: pd.DataFrame,
    params: dict[str, object],
    cfg: SearchConfig,
) -> pd.DataFrame:
    """Precompute daily feature snapshots so weight search is fast."""

    tickers = list(params["tickers"])
    returns = compute_returns(prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    all_dates = list(prices.index)
    start_ts = pd.Timestamp(cfg.start_date)

    rows: list[pd.DataFrame] = []
    for pos, date in enumerate(all_dates):
        if pos < cfg.min_history:
            continue
        if pd.Timestamp(date) < start_ts:
            continue
        if pos + 1 >= len(all_dates):
            break

        # Precompute every trading day; rebalance frequency is applied later.
        hist_prices = prices.iloc[: pos + 1].reindex(columns=tickers)
        hist_returns = returns.iloc[: pos + 1].reindex(columns=tickers)

        tactical = build_multi_horizon_forecast(
            prices=hist_prices,
            returns=hist_returns,
            date=date,
            params=params,
            tickers=tickers,
        )
        table = _build_tactical_score_v2_table(tactical.table).copy()

        if "relative_strength_score" not in table.columns and "relative_strength_rank" in table.columns:
            max_rank = float(table["relative_strength_rank"].max() or 1.0)
            table["relative_strength_score"] = 1.0 - (
                (table["relative_strength_rank"].astype(float) - 1.0) / max(max_rank - 1.0, 1.0)
            )

        table["date"] = pd.Timestamp(date)
        table["next_date"] = pd.Timestamp(all_dates[pos + 1])
        rows.append(table)

    if not rows:
        raise ValueError("No feature snapshots were generated.")

    panel = pd.concat(rows, ignore_index=True)
    available = [c for c in FEATURE_COLUMNS if c in panel.columns]
    for col in available:
        panel[f"z_{col}"] = panel.groupby("date")[col].transform(_zscore_cross_section)

    return panel


def random_weight_vectors(feature_names: list[str], trials: int, seed: int) -> list[dict[str, float]]:
    """Generate random long/short feature weights with L1 normalization."""

    rng = np.random.default_rng(seed)
    vectors: list[dict[str, float]] = []

    # Include useful baselines.
    vectors.append({name: 0.0 for name in feature_names})
    if "risk_adjusted_forecast" in feature_names:
        vectors.append({name: (1.0 if name == "risk_adjusted_forecast" else 0.0) for name in feature_names})
    if "vol_20d" in feature_names:
        vectors.append({name: (1.0 if name == "vol_20d" else 0.0) for name in feature_names})

    for _ in range(trials):
        raw = rng.normal(loc=0.0, scale=1.0, size=len(feature_names))

        # Encourage sparse-ish vectors by zeroing small coefficients.
        mask = rng.random(len(feature_names)) < 0.35
        raw[mask] = 0.0

        denom = float(np.abs(raw).sum())
        if denom <= 1e-12:
            continue
        raw = raw / denom
        vectors.append({feature: float(value) for feature, value in zip(feature_names, raw)})

    return vectors


def score_panel(panel: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=panel.index, dtype=float)
    for feature, weight in weights.items():
        z_col = f"z_{feature}"
        if z_col in panel.columns and abs(weight) > 1e-12:
            score = score + float(weight) * panel[z_col].astype(float)
    return score


def backtest_weight_vector(
    *,
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    weights: dict[str, float],
    cfg: SearchConfig,
    split: str,
) -> dict[str, object]:
    """Backtest one fixed feature-weight vector on train or test period."""

    scored = panel.copy()
    scored["candidate_score"] = score_panel(scored, weights)

    train_end = pd.Timestamp(cfg.train_end_date)
    if split == "train":
        scored = scored[scored["date"] <= train_end].copy()
    elif split == "test":
        scored = scored[scored["date"] > train_end].copy()
    else:
        raise ValueError("split must be train or test")

    if scored.empty:
        return {"split": split, "sharpe": -999.0}

    tickers = list(prices.columns)
    returns = compute_returns(prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    unique_dates = sorted(scored["date"].drop_duplicates().tolist())
    rebalance_dates = unique_dates[:: max(cfg.rebalance_every, 1)]

    equity = 1.0
    daily_rows: list[dict[str, object]] = []
    turnover_values: list[float] = []
    current_weights = pd.Series(0.0, index=tickers, dtype=float)

    for date in unique_dates:
        date_ts = pd.Timestamp(date)
        day_table = scored[scored["date"].eq(date_ts)].set_index("ticker")

        if date_ts in rebalance_dates:
            target = _top_n_weights(day_table["candidate_score"], top_n=cfg.top_n, max_weight=cfg.max_weight)
            target = target.reindex(tickers).fillna(0.0)
            turnover_values.append(float((target - current_weights).abs().sum() / 2.0))
            current_weights = target

        next_date = pd.Timestamp(day_table["next_date"].iloc[0])
        if next_date not in returns.index:
            continue
        next_ret = returns.loc[next_date].reindex(tickers).fillna(0.0)
        portfolio_return = float((current_weights * next_ret).sum())
        equity *= 1.0 + portfolio_return
        daily_rows.append(
            {
                "date": str(next_date.date()),
                "return": portfolio_return,
                "equity": equity,
                "split": split,
            }
        )

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return {"split": split, "sharpe": -999.0}

    stats = _performance_stats(
        daily["return"].astype(float),
        daily["equity"].astype(float),
        pd.Series(turnover_values, dtype=float),
        risk_free_rate_annual=cfg.risk_free_rate_annual,
    )
    return {
        "split": split,
        **stats,
        "rebalance_count": int(len(rebalance_dates)),
        "top_n": int(cfg.top_n),
        "rebalance_every": int(cfg.rebalance_every),
        "max_weight": float(cfg.max_weight),
    }


def run_weight_search(cfg: SearchConfig) -> dict[str, Path]:
    params = build_params()
    prices = load_prices(params, cfg.end_date)
    panel = build_feature_panel(prices=prices, params=params, cfg=cfg)

    feature_names = [c for c in FEATURE_COLUMNS if f"z_{c}" in panel.columns]
    candidates = random_weight_vectors(feature_names, trials=cfg.random_trials, seed=cfg.seed)

    rows: list[dict[str, object]] = []
    for candidate_id, weights in enumerate(candidates):
        train_stats = backtest_weight_vector(panel=panel, prices=prices, weights=weights, cfg=cfg, split="train")
        test_stats = backtest_weight_vector(panel=panel, prices=prices, weights=weights, cfg=cfg, split="test")

        train_sharpe = float(train_stats.get("sharpe", -999.0) or -999.0)
        test_sharpe = float(test_stats.get("sharpe", -999.0) or -999.0)
        test_drawdown = float(test_stats.get("max_drawdown", 0.0) or 0.0)
        is_degenerate = all(abs(float(value)) <= 1e-12 for value in weights.values())
        train_test_gap = abs(train_sharpe - test_sharpe)
        drawdown_penalty = max(0.0, abs(test_drawdown) - 0.10)
        robust_score = min(train_sharpe, test_sharpe) - 0.25 * train_test_gap - 0.50 * drawdown_penalty
        if is_degenerate:
            robust_score -= 0.25

        rows.append(
            {
                "candidate_id": candidate_id,
                "is_degenerate": bool(is_degenerate),
                "train_sharpe": train_sharpe,
                "test_sharpe": test_sharpe,
                "train_test_gap": train_test_gap,
                "robust_score": robust_score,
                "train_total_return": train_stats.get("total_return"),
                "test_total_return": test_stats.get("total_return"),
                "train_max_drawdown": train_stats.get("max_drawdown"),
                "test_max_drawdown": test_drawdown,
                "train_total_turnover": train_stats.get("total_turnover"),
                "test_total_turnover": test_stats.get("total_turnover"),
                "weights_json": json.dumps(weights, sort_keys=True),
            }
        )

    results = pd.DataFrame(rows)
    results = results.sort_values(["robust_score", "test_sharpe", "train_sharpe"], ascending=False).reset_index(drop=True)
    best = results.iloc[0]
    best_train = results.sort_values(["train_sharpe", "test_sharpe"], ascending=False).iloc[0]
    best_test = results.sort_values(["test_sharpe", "train_sharpe"], ascending=False).iloc[0]
    best_weights = json.loads(str(best["weights_json"]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = OUTPUT_DIR / "tactical_weight_search_feature_panel.csv"
    results_path = OUTPUT_DIR / "tactical_weight_search_results.csv"
    best_weights_path = OUTPUT_DIR / "tactical_weight_search_best_weights.json"
    report_path = OUTPUT_DIR / "tactical_weight_search_report.txt"

    panel.to_csv(panel_path, index=False)
    results.to_csv(results_path, index=False)
    best_weights_path.write_text(json.dumps(best_weights, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "Tactical Constant Feature Weight Search Report",
        "",
        "status: research_only_no_order_change",
        f"start_date: {cfg.start_date}",
        f"train_end_date: {cfg.train_end_date}",
        f"rebalance_every: {cfg.rebalance_every}",
        f"top_n: {cfg.top_n}",
        f"max_weight: {cfg.max_weight:.4f}",
        f"risk_free_rate_annual: {cfg.risk_free_rate_annual:.4f}",
        f"random_trials: {cfg.random_trials}",
        f"feature_count: {len(feature_names)}",
        "",
        "objective:",
        "- Maximize train-period 2%-RF Sharpe for a constant feature-weight score.",
        "- Report test-period Sharpe separately to detect overfitting.",
        "- This does not alter production orders.",
        "",
        "best_candidate_by_robust_score:",
        f"- candidate_id: {int(best['candidate_id'])}",
        f"- is_degenerate: {bool(best['is_degenerate'])}",
        f"- robust_score: {float(best['robust_score']):.4f}",
        f"- train_sharpe: {float(best['train_sharpe']):.4f}",
        f"- test_sharpe: {float(best['test_sharpe']):.4f}",
        f"- train_test_gap: {float(best['train_test_gap']):.4f}",
        f"- train_total_return: {float(best['train_total_return']):.4f}",
        f"- test_total_return: {float(best['test_total_return']):.4f}",
        f"- train_max_drawdown: {float(best['train_max_drawdown']):.4f}",
        f"- test_max_drawdown: {float(best['test_max_drawdown']):.4f}",
        "",
        "comparison_winners:",
        f"- best_train_candidate_id: {int(best_train['candidate_id'])}, train_sharpe={float(best_train['train_sharpe']):.4f}, test_sharpe={float(best_train['test_sharpe']):.4f}, robust_score={float(best_train['robust_score']):.4f}",
        f"- best_test_candidate_id: {int(best_test['candidate_id'])}, train_sharpe={float(best_test['train_sharpe']):.4f}, test_sharpe={float(best_test['test_sharpe']):.4f}, robust_score={float(best_test['robust_score']):.4f}",
        "",
        "best_robust_weights:",
    ]

    for name, value in sorted(best_weights.items(), key=lambda kv: abs(kv[1]), reverse=True):
        if abs(float(value)) > 1e-6:
            lines.append(f"- {name}: {float(value):+.4f}")

    lines.extend(["", "top_10_candidates_by_robust_score:"])
    for row in results.head(10).itertuples(index=False):
        lines.append(
            f"- id={row.candidate_id}: robust_score={row.robust_score:.4f}, "
            f"train_sharpe={row.train_sharpe:.4f}, test_sharpe={row.test_sharpe:.4f}, "
            f"gap={row.train_test_gap:.4f}, train_return={row.train_total_return:.4f}, "
            f"test_return={row.test_total_return:.4f}, test_dd={row.test_max_drawdown:.4f}, "
            f"degenerate={row.is_degenerate}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "feature_panel": panel_path,
        "results": results_path,
        "best_weights": best_weights_path,
        "report": report_path,
    }


def parse_args() -> SearchConfig:
    parser = argparse.ArgumentParser(description="Optimize constant tactical feature weights by backtest.")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--train-end-date", default="2025-06-30")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--rebalance-every", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return SearchConfig(
        start_date=parser.parse_args().start_date,
        train_end_date=parser.parse_args().train_end_date,
        end_date=parser.parse_args().end_date,
        rebalance_every=parser.parse_args().rebalance_every,
        top_n=parser.parse_args().top_n,
        max_weight=parser.parse_args().max_weight,
        random_trials=parser.parse_args().trials,
        seed=parser.parse_args().seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    paths = run_weight_search(cfg)
    print("Weight search outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")
