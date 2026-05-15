"""Feature engineering utilities for momentum-driven allocation signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import FeatureConfig


@dataclass(slots=True)
class FeatureSnapshot:
    """Cross-sectional feature table computed for a single decision date."""

    as_of: pd.Timestamp
    table: pd.DataFrame

    @property
    def scores(self) -> pd.Series:
        """Return the robust forecast score per ticker."""

        return self.table["score"].copy()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily arithmetic returns from a price matrix."""

    if prices.empty:
        raise ValueError("Price DataFrame is empty.")

    returns = prices.sort_index().pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")
    return returns


def compute_momentum_forecast_at_date(
    prices: pd.DataFrame,
    date: pd.Timestamp | str,
    tickers: list[str] | tuple[str, ...],
    short_window: int,
    long_window: int,
    kappa: float,
    clip_lower: float,
    clip_upper: float,
) -> pd.Series:
    """Compute a clipped robust momentum forecast using only history up to the date."""

    if prices.empty:
        raise ValueError("Price history is empty.")
    if short_window <= 0 or long_window <= 0:
        raise ValueError("Momentum windows must be positive integers.")

    as_of = pd.Timestamp(date)
    history = prices.reindex(columns=list(tickers)).loc[:as_of].sort_index()
    history = history.ffill(limit=3).dropna(how="all")

    required_rows = max(short_window, long_window) + 1
    if len(history) < required_rows:
        raise ValueError(
            f"Not enough history up to {as_of.date()} for momentum forecast: "
            f"{len(history)} rows available, {required_rows} required."
        )

    price_today = history.iloc[-1]
    price_short_base = history.iloc[-(short_window + 1)]
    price_long_base = history.iloc[-(long_window + 1)]

    momentum_short = price_today / price_short_base - 1.0
    momentum_long = price_today / price_long_base - 1.0
    mu_signal = 0.6 * momentum_short + 0.4 * momentum_long
    mu_robust = kappa * mu_signal
    mu_robust = mu_robust.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return mu_robust.clip(lower=clip_lower, upper=clip_upper)


def compute_market_risk_state(prices: pd.DataFrame, date: pd.Timestamp | str) -> str:
    """Classify the market state using SPY trend and 63-day momentum when available."""

    if prices.empty or "SPY" not in prices.columns:
        return "normal"

    as_of = pd.Timestamp(date)
    spy_history = prices.loc[:as_of, "SPY"].dropna().sort_index()
    if len(spy_history) < 201:
        return "normal"

    spy_today = float(spy_history.iloc[-1])
    spy_ma_200 = float(spy_history.tail(200).mean())
    spy_momentum_63 = float(spy_today / spy_history.iloc[-64] - 1.0)

    if spy_today < spy_ma_200 or spy_momentum_63 < -0.08:
        return "risk_off"
    return "normal"


def compute_signal_quality(mu: pd.Series) -> dict[str, float]:
    """Return simple diagnostics for a forecast vector."""

    signal = mu.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if signal.empty:
        return {
            "mean_abs_signal": float("nan"),
            "signal_dispersion": float("nan"),
            "number_positive": 0.0,
            "number_negative": 0.0,
        }

    return {
        "mean_abs_signal": float(signal.abs().mean()),
        "signal_dispersion": float(signal.std(ddof=0)),
        "number_positive": float((signal > 0.0).sum()),
        "number_negative": float((signal < 0.0).sum()),
    }


def compute_feature_snapshot(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: FeatureConfig,
    as_of: pd.Timestamp | None = None,
) -> FeatureSnapshot:
    """Compatibility wrapper returning a cross-sectional forecast table."""

    del returns  # The compatibility API keeps this parameter, but it is not needed here.

    if prices.empty:
        raise ValueError("Prices must not be empty.")

    if as_of is None:
        as_of = pd.Timestamp(prices.index[-1])

    tickers = [str(column) for column in prices.columns]
    history = prices.reindex(columns=tickers).loc[:as_of].sort_index().ffill(limit=3)
    required_rows = max(config.momentum_window_3m, config.momentum_window_6m) + 1
    if len(history.dropna(how="all")) < required_rows:
        raise ValueError(
            f"Not enough history up to {pd.Timestamp(as_of).date()} to compute features."
        )

    price_today = history.iloc[-1]
    price_short_base = history.iloc[-(config.momentum_window_3m + 1)]
    price_long_base = history.iloc[-(config.momentum_window_6m + 1)]

    momentum_short = price_today / price_short_base - 1.0
    momentum_long = price_today / price_long_base - 1.0
    forecast = compute_momentum_forecast_at_date(
        prices=history,
        date=as_of,
        tickers=tickers,
        short_window=config.momentum_window_3m,
        long_window=config.momentum_window_6m,
        kappa=config.kappa,
        clip_lower=config.forecast_clip_lower,
        clip_upper=config.forecast_clip_upper,
    )

    table = pd.DataFrame(
        {
            "momentum_short": momentum_short,
            "momentum_long": momentum_long,
            "mu_signal": 0.6 * momentum_short + 0.4 * momentum_long,
            "mu_robust": forecast,
            "score": forecast,
        }
    )
    table = table.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    table["score_rank"] = table["score"].rank(ascending=False, method="dense")
    return FeatureSnapshot(as_of=pd.Timestamp(as_of), table=table.sort_values("score", ascending=False))
