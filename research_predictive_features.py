"""Predictive feature research for tactical asset forecasts.

Research-only. Does not place orders and does not change production logic.

Purpose:
- Build a per-date, per-ticker feature panel.
- Add technical, cross-asset, regime, path-shape, and calendar features.
- Evaluate forward-return predictiveness for 1d/3d/5d/10d horizons.
- Train simple ridge models per horizon and optionally per asset.
- Produce reports for improving tactical forecast logic.

This is intentionally conservative:
- No look-ahead: features at date t use data up to t only.
- Targets are future returns from t to t+h.
- Outputs go to outputs/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd

from config import build_params
from data import load_price_data
from features import compute_returns


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PRICE_CACHE_PATH = BASE_DIR / "data" / "prices_cache.csv"


HORIZONS = (1, 3, 5, 10)

BASE_FEATURES = [
    # Asset-local price/path features
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "mom_accel_5_20",
    "mom_accel_10_60",
    "vol_5d",
    "vol_20d",
    "vol_60d",
    "vol_ratio_5_20",
    "rolling_sharpe_20d",
    "drawdown_20d",
    "drawdown_60d",
    "ma_dist_10d",
    "ma_dist_20d",
    "ma_dist_50d",
    "ma_cross_10_50",
    "up_days_5d",
    "up_days_10d",
    "down_days_5d",
    "path_efficiency_10d",
    "path_efficiency_20d",
    "range_position_20d",
    "range_position_60d",

    # Relative and cross-sectional features
    "rel_ret_vs_spy_5d",
    "rel_ret_vs_spy_20d",
    "rel_ret_vs_universe_5d",
    "rel_ret_vs_universe_20d",
    "cross_sectional_rank_ret_20d",
    "cross_sectional_rank_vol_20d",

    # Cross-asset regime features
    "spy_ret_5d",
    "spy_ret_20d",
    "spy_vol_20d",
    "qqq_ret_5d",
    "qqq_ret_20d",
    "tlt_ret_5d",
    "tlt_ret_20d",
    "sgov_ret_20d",
    "gld_ret_20d",
    "pdbc_ret_20d",
    "ibit_ret_20d",
    "risk_on_spy_minus_tlt_20d",
    "duration_tlt_minus_sgov_20d",
    "commodity_pdbc_minus_spy_20d",
    "gold_gld_minus_spy_20d",
    "crypto_ibit_minus_spy_20d",

    # Calendar features
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "turn_of_month",
]


@dataclass(frozen=True)
class ResearchConfig:
    start_date: str = "2024-01-01"
    train_end_date: str = "2025-06-30"
    end_date: str | None = None
    min_history: int = 180
    ridge_alpha: float = 10.0
    top_quantile: float = 0.20
    max_per_asset_models: int = 33


def _safe_div(a: pd.Series, b: pd.Series | float) -> pd.Series:
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_return(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    return prices / prices.shift(days) - 1.0


def _rolling_vol(returns: pd.DataFrame, days: int) -> pd.DataFrame:
    return returns.rolling(days).std(ddof=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_max_drawdown_like(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    rolling_high = prices.rolling(days).max()
    return prices / rolling_high - 1.0


def _ma_distance(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    ma = prices.rolling(days).mean()
    return prices / ma - 1.0


def _path_efficiency(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    net = (prices - prices.shift(days)).abs()
    step = prices.diff().abs().rolling(days).sum()
    return _safe_div(net, step)


def _range_position(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    lo = prices.rolling(days).min()
    hi = prices.rolling(days).max()
    return _safe_div(prices - lo, hi - lo)


def _cross_sectional_rank(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    return df.rank(axis=1, pct=True, ascending=ascending).fillna(0.0)


def _zscore_train_apply(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_z = train.copy()
    test_z = test.copy()

    for col in features:
        mean = float(train[col].mean())
        std = float(train[col].std(ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            train_z[col] = 0.0
            test_z[col] = 0.0
        else:
            train_z[col] = ((train[col] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            test_z[col] = ((test[col] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return train_z, test_z


def _spearman_corr(x: pd.Series, y: pd.Series) -> float:
    x_rank = x.rank()
    y_rank = y.rank()
    if x_rank.nunique() <= 1 or y_rank.nunique() <= 1:
        return 0.0
    value = x_rank.corr(y_rank)
    return float(value) if np.isfinite(value) else 0.0


def _ridge_fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target_col: str,
    alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    use = train[features + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(50, len(features) * 4):
        return np.zeros(len(test), dtype=float), {feature: 0.0 for feature in features}

    X = use[features].to_numpy(dtype=float)
    y = use[target_col].to_numpy(dtype=float)

    X_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)

    # Add intercept not penalized.
    X_aug = np.column_stack([np.ones(len(X)), X])
    X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])

    penalty = np.eye(X_aug.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0

    try:
        beta = np.linalg.solve(X_aug.T @ X_aug + penalty, X_aug.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_aug.T @ X_aug + penalty) @ X_aug.T @ y

    pred = X_test_aug @ beta
    coefs = {feature: float(beta[i + 1]) for i, feature in enumerate(features)}
    return pred, coefs


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


def build_feature_panel(prices: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    tickers = list(prices.columns)
    returns = compute_returns(prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ret_1d = returns
    ret_3d = _rolling_return(prices, 3)
    ret_5d = _rolling_return(prices, 5)
    ret_10d = _rolling_return(prices, 10)
    ret_20d = _rolling_return(prices, 20)
    ret_60d = _rolling_return(prices, 60)

    vol_5d = _rolling_vol(returns, 5)
    vol_20d = _rolling_vol(returns, 20)
    vol_60d = _rolling_vol(returns, 60)

    universe_ret_5d = ret_5d.mean(axis=1)
    universe_ret_20d = ret_20d.mean(axis=1)

    def col_or_zero(name: str) -> pd.Series:
        if name in prices.columns:
            return prices[name]
        return pd.Series(0.0, index=prices.index, dtype=float)

    spy = col_or_zero("SPY")
    qqq = col_or_zero("QQQ")
    tlt = col_or_zero("TLT")
    sgov = col_or_zero("SGOV")
    gld = col_or_zero("GLD")
    pdbc = col_or_zero("PDBC")
    ibit = col_or_zero("IBIT")

    proxy = {
        "spy_ret_5d": spy / spy.shift(5) - 1.0,
        "spy_ret_20d": spy / spy.shift(20) - 1.0,
        "spy_vol_20d": spy.pct_change().rolling(20).std(ddof=0),
        "qqq_ret_5d": qqq / qqq.shift(5) - 1.0,
        "qqq_ret_20d": qqq / qqq.shift(20) - 1.0,
        "tlt_ret_5d": tlt / tlt.shift(5) - 1.0,
        "tlt_ret_20d": tlt / tlt.shift(20) - 1.0,
        "sgov_ret_20d": sgov / sgov.shift(20) - 1.0,
        "gld_ret_20d": gld / gld.shift(20) - 1.0,
        "pdbc_ret_20d": pdbc / pdbc.shift(20) - 1.0,
        "ibit_ret_20d": ibit / ibit.shift(20) - 1.0,
    }
    proxy["risk_on_spy_minus_tlt_20d"] = proxy["spy_ret_20d"] - proxy["tlt_ret_20d"]
    proxy["duration_tlt_minus_sgov_20d"] = proxy["tlt_ret_20d"] - proxy["sgov_ret_20d"]
    proxy["commodity_pdbc_minus_spy_20d"] = proxy["pdbc_ret_20d"] - proxy["spy_ret_20d"]
    proxy["gold_gld_minus_spy_20d"] = proxy["gld_ret_20d"] - proxy["spy_ret_20d"]
    proxy["crypto_ibit_minus_spy_20d"] = proxy["ibit_ret_20d"] - proxy["spy_ret_20d"]

    month = pd.Series(prices.index.month, index=prices.index)
    dow = pd.Series(prices.index.dayofweek, index=prices.index)
    calendar = {
        "month_sin": np.sin(2.0 * np.pi * month / 12.0),
        "month_cos": np.cos(2.0 * np.pi * month / 12.0),
        "dow_sin": np.sin(2.0 * np.pi * dow / 5.0),
        "dow_cos": np.cos(2.0 * np.pi * dow / 5.0),
        "turn_of_month": ((pd.Series(prices.index.day, index=prices.index) <= 3) | (pd.Series(prices.index.day, index=prices.index) >= 27)).astype(float),
    }

    feature_frames: dict[str, pd.DataFrame] = {
        "ret_1d": ret_1d,
        "ret_3d": ret_3d,
        "ret_5d": ret_5d,
        "ret_10d": ret_10d,
        "ret_20d": ret_20d,
        "ret_60d": ret_60d,
        "mom_accel_5_20": ret_5d - ret_20d,
        "mom_accel_10_60": ret_10d - ret_60d,
        "vol_5d": vol_5d,
        "vol_20d": vol_20d,
        "vol_60d": vol_60d,
        "vol_ratio_5_20": _safe_div(vol_5d, vol_20d),
        "rolling_sharpe_20d": _safe_div(ret_20d, vol_20d * np.sqrt(20.0)),
        "drawdown_20d": _rolling_max_drawdown_like(prices, 20),
        "drawdown_60d": _rolling_max_drawdown_like(prices, 60),
        "ma_dist_10d": _ma_distance(prices, 10),
        "ma_dist_20d": _ma_distance(prices, 20),
        "ma_dist_50d": _ma_distance(prices, 50),
        "ma_cross_10_50": prices.rolling(10).mean() / prices.rolling(50).mean() - 1.0,
        "up_days_5d": (returns > 0).rolling(5).sum(),
        "up_days_10d": (returns > 0).rolling(10).sum(),
        "down_days_5d": (returns < 0).rolling(5).sum(),
        "path_efficiency_10d": _path_efficiency(prices, 10),
        "path_efficiency_20d": _path_efficiency(prices, 20),
        "range_position_20d": _range_position(prices, 20),
        "range_position_60d": _range_position(prices, 60),
        "rel_ret_vs_spy_5d": ret_5d.sub(proxy["spy_ret_5d"], axis=0),
        "rel_ret_vs_spy_20d": ret_20d.sub(proxy["spy_ret_20d"], axis=0),
        "rel_ret_vs_universe_5d": ret_5d.sub(universe_ret_5d, axis=0),
        "rel_ret_vs_universe_20d": ret_20d.sub(universe_ret_20d, axis=0),
        "cross_sectional_rank_ret_20d": _cross_sectional_rank(ret_20d, ascending=False),
        "cross_sectional_rank_vol_20d": _cross_sectional_rank(vol_20d, ascending=True),
    }

    for name, series in {**proxy, **calendar}.items():
        feature_frames[name] = pd.DataFrame(
            np.repeat(series.to_numpy(dtype=float).reshape(-1, 1), len(tickers), axis=1),
            index=prices.index,
            columns=tickers,
        )

    rows = []
    start_ts = pd.Timestamp(cfg.start_date)

    for date_pos, date in enumerate(prices.index):
        if date_pos < cfg.min_history:
            continue
        if pd.Timestamp(date) < start_ts:
            continue
        max_h = max(HORIZONS)
        if date_pos + max_h >= len(prices.index):
            break

        frame = pd.DataFrame({"ticker": tickers})
        frame["date"] = pd.Timestamp(date)

        for feature_name, values in feature_frames.items():
            frame[feature_name] = values.loc[date].reindex(tickers).to_numpy(dtype=float)

        current_price = prices.loc[date].reindex(tickers).astype(float)
        frame["latest_price"] = current_price.to_numpy(dtype=float)

        for horizon in HORIZONS:
            future_price = prices.iloc[date_pos + horizon].reindex(tickers).astype(float)
            frame[f"forward_return_{horizon}d"] = (future_price / current_price - 1.0).to_numpy(dtype=float)

        rows.append(frame)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return panel


def evaluate_univariate(panel: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    rows = []
    train_end = pd.Timestamp(cfg.train_end_date)

    for horizon in HORIZONS:
        target = f"forward_return_{horizon}d"
        for feature in BASE_FEATURES:
            for split_name, split_df in [
                ("train", panel[panel["date"] <= train_end]),
                ("test", panel[panel["date"] > train_end]),
                ("all", panel),
            ]:
                if split_df.empty:
                    continue

                ic_values = []
                top_returns = []
                bottom_returns = []
                hit_rates = []

                for _, day in split_df.groupby("date"):
                    if day[feature].nunique() <= 1:
                        continue
                    q = max(int(len(day) * cfg.top_quantile), 1)
                    ranked = day.sort_values(feature, ascending=False)
                    top = ranked.head(q)
                    bottom = ranked.tail(q)

                    ic_values.append(_spearman_corr(day[feature], day[target]))
                    top_returns.append(float(top[target].mean()))
                    bottom_returns.append(float(bottom[target].mean()))
                    hit_rates.append(float((top[target] > 0).mean()))

                if not ic_values:
                    continue

                rows.append(
                    {
                        "horizon": horizon,
                        "feature": feature,
                        "split": split_name,
                        "rank_ic_mean": float(np.mean(ic_values)),
                        "rank_ic_median": float(np.median(ic_values)),
                        "top_mean_return": float(np.mean(top_returns)),
                        "bottom_mean_return": float(np.mean(bottom_returns)),
                        "top_minus_bottom": float(np.mean(top_returns) - np.mean(bottom_returns)),
                        "top_hit_rate": float(np.mean(hit_rates)),
                        "sample_days": int(len(ic_values)),
                    }
                )

    return pd.DataFrame(rows).sort_values(["split", "horizon", "rank_ic_mean"], ascending=[True, True, False])


def evaluate_per_asset_univariate(panel: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    rows = []
    train_end = pd.Timestamp(cfg.train_end_date)

    for ticker, asset_df in panel.groupby("ticker"):
        for horizon in HORIZONS:
            target = f"forward_return_{horizon}d"
            for feature in BASE_FEATURES:
                for split_name, split_df in [
                    ("train", asset_df[asset_df["date"] <= train_end]),
                    ("test", asset_df[asset_df["date"] > train_end]),
                ]:
                    if len(split_df) < 50 or split_df[feature].nunique() <= 1:
                        continue
                    corr = _spearman_corr(split_df[feature], split_df[target])
                    rows.append(
                        {
                            "ticker": ticker,
                            "horizon": horizon,
                            "feature": feature,
                            "split": split_name,
                            "spearman_ic": corr,
                            "feature_mean": float(split_df[feature].mean()),
                            "target_mean": float(split_df[target].mean()),
                            "sample_count": int(len(split_df)),
                        }
                    )

    return pd.DataFrame(rows).sort_values(["split", "horizon", "ticker", "spearman_ic"], ascending=[True, True, True, False])


def evaluate_ridge_models(panel: pd.DataFrame, cfg: ResearchConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(cfg.train_end_date)
    train_raw = panel[panel["date"] <= train_end].copy()
    test_raw = panel[panel["date"] > train_end].copy()

    features = [f for f in BASE_FEATURES if f in panel.columns]
    train_z, test_z = _zscore_train_apply(train_raw, test_raw, features)

    model_rows = []
    coef_rows = []

    for horizon in HORIZONS:
        target = f"forward_return_{horizon}d"

        # Global model across all tickers.
        pred, coefs = _ridge_fit_predict(train_z, test_z, features, target, cfg.ridge_alpha)
        eval_df = test_z[["date", "ticker", target]].copy()
        eval_df["prediction"] = pred

        by_day_ic = []
        top_returns = []
        bottom_returns = []
        hit_rates = []

        for _, day in eval_df.groupby("date"):
            if day["prediction"].nunique() <= 1:
                continue
            q = max(int(len(day) * cfg.top_quantile), 1)
            ranked = day.sort_values("prediction", ascending=False)
            top = ranked.head(q)
            bottom = ranked.tail(q)
            by_day_ic.append(_spearman_corr(day["prediction"], day[target]))
            top_returns.append(float(top[target].mean()))
            bottom_returns.append(float(bottom[target].mean()))
            hit_rates.append(float((top[target] > 0).mean()))

        model_rows.append(
            {
                "model": "global_ridge",
                "ticker": "ALL",
                "horizon": horizon,
                "test_rank_ic_mean": float(np.mean(by_day_ic)) if by_day_ic else 0.0,
                "test_top_mean_return": float(np.mean(top_returns)) if top_returns else 0.0,
                "test_bottom_mean_return": float(np.mean(bottom_returns)) if bottom_returns else 0.0,
                "test_top_minus_bottom": float(np.mean(top_returns) - np.mean(bottom_returns)) if top_returns else 0.0,
                "test_top_hit_rate": float(np.mean(hit_rates)) if hit_rates else 0.0,
                "test_sample_days": int(len(by_day_ic)),
            }
        )

        for feature, coef in coefs.items():
            coef_rows.append(
                {
                    "model": "global_ridge",
                    "ticker": "ALL",
                    "horizon": horizon,
                    "feature": feature,
                    "coef": coef,
                }
            )

        # Per-asset ridge models.
        for ticker in sorted(panel["ticker"].unique())[: cfg.max_per_asset_models]:
            train_asset = train_z[train_z["ticker"].eq(ticker)].copy()
            test_asset = test_z[test_z["ticker"].eq(ticker)].copy()
            if len(train_asset) < max(60, len(features) * 3) or len(test_asset) < 20:
                continue

            pred, coefs = _ridge_fit_predict(train_asset, test_asset, features, target, cfg.ridge_alpha)
            actual = test_asset[target].to_numpy(dtype=float)
            pred_series = pd.Series(pred)
            actual_series = pd.Series(actual)

            ic = _spearman_corr(pred_series, actual_series)
            hit = float(((pred > 0) == (actual > 0)).mean()) if len(actual) else 0.0
            mse = float(np.mean((pred - actual) ** 2)) if len(actual) else 0.0

            model_rows.append(
                {
                    "model": "per_asset_ridge",
                    "ticker": ticker,
                    "horizon": horizon,
                    "test_rank_ic_mean": ic,
                    "test_top_mean_return": 0.0,
                    "test_bottom_mean_return": 0.0,
                    "test_top_minus_bottom": 0.0,
                    "test_top_hit_rate": hit,
                    "test_mse": mse,
                    "test_sample_days": int(len(test_asset)),
                }
            )

            for feature, coef in coefs.items():
                coef_rows.append(
                    {
                        "model": "per_asset_ridge",
                        "ticker": ticker,
                        "horizon": horizon,
                        "feature": feature,
                        "coef": coef,
                    }
                )

    return pd.DataFrame(model_rows), pd.DataFrame(coef_rows)


def run_research(cfg: ResearchConfig) -> dict[str, Path]:
    params = build_params()
    prices = load_prices(params, cfg.end_date)
    panel = build_feature_panel(prices, cfg)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel_path = OUTPUT_DIR / "predictive_feature_panel.csv"
    univar_path = OUTPUT_DIR / "predictive_feature_univariate.csv"
    per_asset_path = OUTPUT_DIR / "predictive_feature_per_asset.csv"
    model_path = OUTPUT_DIR / "predictive_feature_ridge_models.csv"
    coef_path = OUTPUT_DIR / "predictive_feature_ridge_coefficients.csv"
    report_path = OUTPUT_DIR / "predictive_feature_report.txt"

    panel.to_csv(panel_path, index=False)

    univar = evaluate_univariate(panel, cfg)
    per_asset = evaluate_per_asset_univariate(panel, cfg)
    models, coefs = evaluate_ridge_models(panel, cfg)

    univar.to_csv(univar_path, index=False)
    per_asset.to_csv(per_asset_path, index=False)
    models.to_csv(model_path, index=False)
    coefs.to_csv(coef_path, index=False)

    lines = [
        "Predictive Feature Research Report",
        "",
        "status: research_only_no_order_change",
        f"start_date: {cfg.start_date}",
        f"train_end_date: {cfg.train_end_date}",
        f"ridge_alpha: {cfg.ridge_alpha:.4f}",
        f"rows: {len(panel)}",
        f"tickers: {panel['ticker'].nunique()}",
        "",
        "method:",
        "- Builds technical, cross-asset, regime, path-shape and calendar features.",
        "- Evaluates 1d/3d/5d/10d forward returns without look-ahead.",
        "- Reports univariate Rank IC and top-minus-bottom returns.",
        "- Trains simple ridge models for global and per-asset forecast diagnostics.",
        "- Does not alter trading logic.",
        "",
    ]

    lines.append("top_univariate_test_features:")
    test_uni = univar[univar["split"].eq("test")].copy()
    if not test_uni.empty:
        ranked = test_uni.sort_values(["horizon", "rank_ic_mean"], ascending=[True, False])
        for horizon in HORIZONS:
            subset = ranked[ranked["horizon"].eq(horizon)].head(12)
            lines.append(f"horizon_{horizon}d:")
            for row in subset.itertuples(index=False):
                lines.append(
                    f"- {row.feature}: rank_ic={row.rank_ic_mean:.4f}, "
                    f"top_minus_bottom={row.top_minus_bottom:.5f}, "
                    f"top_hit_rate={row.top_hit_rate:.3f}"
                )
    else:
        lines.append("- no test univariate rows")

    lines.extend(["", "ridge_model_test_summary:"])
    for row in models.sort_values(["model", "horizon", "test_rank_ic_mean"], ascending=[True, True, False]).head(80).itertuples(index=False):
        lines.append(
            f"- {row.model} {row.ticker} {row.horizon}d: "
            f"rank_ic={row.test_rank_ic_mean:.4f}, "
            f"top_minus_bottom={getattr(row, 'test_top_minus_bottom', 0.0):.5f}, "
            f"hit={row.test_top_hit_rate:.3f}, days={row.test_sample_days}"
        )

    lines.extend(["", "largest_global_ridge_coefficients:"])
    global_coefs = coefs[coefs["model"].eq("global_ridge")].copy()
    if not global_coefs.empty:
        global_coefs["abs_coef"] = global_coefs["coef"].abs()
        for row in global_coefs.sort_values("abs_coef", ascending=False).head(40).itertuples(index=False):
            lines.append(
                f"- {row.horizon}d {row.feature}: coef={row.coef:+.6f}"
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "panel": panel_path,
        "univariate": univar_path,
        "per_asset": per_asset_path,
        "models": model_path,
        "coefficients": coef_path,
        "report": report_path,
    }


def parse_args() -> ResearchConfig:
    parser = argparse.ArgumentParser(description="Research predictive features for tactical forecasts.")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--train-end-date", default="2025-06-30")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--min-history", type=int, default=180)
    args = parser.parse_args()

    return ResearchConfig(
        start_date=args.start_date,
        train_end_date=args.train_end_date,
        end_date=args.end_date,
        ridge_alpha=args.ridge_alpha,
        min_history=args.min_history,
    )


if __name__ == "__main__":
    cfg = parse_args()
    paths = run_research(cfg)
    print("Predictive feature research outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")
