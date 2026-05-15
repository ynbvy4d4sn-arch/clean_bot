"""Risk estimation and compatibility helpers for the allocation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd

from asset_universe import (
    AssetDefinition,
    CRYPTO_MAX_NORMAL,
    CRYPTO_MAX_RISK_OFF,
    GROUP_LIMITS,
    MAX_EQUITY_LIKE_TOTAL_NORMAL,
    MAX_EQUITY_LIKE_TOTAL_RISK_OFF,
)
from config import (
    CVAR_LIMIT,
    DRAWDOWN_LIMIT,
    RiskConfig,
)
from features import compute_market_risk_state
from risk_free import risk_free_return_for_horizon


class RiskRegime(str, Enum):
    """Discrete regimes used by the existing decision pipeline."""

    RISK_ON = "RISK_ON"
    CAUTION = "CAUTION"
    RISK_OFF = "RISK_OFF"
    PAUSE = "PAUSE"


@dataclass(slots=True)
class RiskSnapshot:
    """Compatibility risk summary for a single decision date."""

    as_of: pd.Timestamp
    regime: RiskRegime
    market_drawdown: float
    realized_volatility: float
    positive_breadth: float
    de_risk_scalar: float
    recommended_risky_cap: float
    diagnostics: dict[str, float | str] = field(default_factory=dict)

    @property
    def pause_new_risk(self) -> bool:
        """Return whether the strategy should avoid new risky exposure."""

        return self.regime == RiskRegime.PAUSE


def _coerce_weight_series(
    weights: pd.Series | np.ndarray | Sequence[float],
    index: pd.Index,
) -> pd.Series:
    """Align an arbitrary weight vector to a target index."""

    if isinstance(weights, pd.Series):
        return weights.reindex(index).fillna(0.0).astype(float)

    array = np.asarray(weights, dtype=float)
    if array.ndim != 1 or len(array) != len(index):
        raise ValueError("Weight vector length must match the target index.")
    return pd.Series(array, index=index, dtype=float)


def _coerce_return_series(returns: pd.Series | Sequence[float] | np.ndarray) -> pd.Series:
    """Return a clean one-dimensional return series."""

    if isinstance(returns, pd.Series):
        return returns.dropna().astype(float)
    array = np.asarray(returns, dtype=float)
    if array.ndim != 1:
        raise ValueError("Return input must be one-dimensional.")
    return pd.Series(array, dtype=float).dropna()


def estimate_robust_covariance_at_date(
    returns: pd.DataFrame,
    date: pd.Timestamp | str,
    window: int,
    horizon_days: int,
    alpha: float,
    jitter: float,
) -> pd.DataFrame:
    """Estimate a shrunk horizon covariance matrix using data up to the supplied date."""

    if returns.empty:
        raise ValueError("Return history is empty.")
    if window < 2:
        raise ValueError("Covariance window must be at least 2.")

    as_of = pd.Timestamp(date)
    returns_window = returns.loc[:as_of].sort_index().tail(window)
    if len(returns_window) < 2:
        raise ValueError(
            f"Not enough return observations up to {as_of.date()} to estimate covariance."
        )

    sample_cov_daily = returns_window.cov()
    sample_cov_3m = sample_cov_daily * float(horizon_days)
    diag_cov = pd.DataFrame(
        np.diag(np.diag(sample_cov_3m.to_numpy(dtype=float))),
        index=sample_cov_3m.index,
        columns=sample_cov_3m.columns,
    )
    sigma_values = (alpha * sample_cov_3m + (1.0 - alpha) * diag_cov).to_numpy(dtype=float, copy=True)
    sigma_values[np.diag_indices_from(sigma_values)] += float(jitter)
    return pd.DataFrame(sigma_values, index=sample_cov_3m.index, columns=sample_cov_3m.columns)


def portfolio_expected_return(
    mu: pd.Series | np.ndarray | Sequence[float],
    w: pd.Series | np.ndarray | Sequence[float],
) -> float:
    """Compute expected portfolio return over the forecast horizon."""

    if isinstance(mu, pd.Series):
        weights = _coerce_weight_series(w, mu.index)
        return float(mu.astype(float) @ weights)

    mu_array = np.asarray(mu, dtype=float)
    w_array = np.asarray(w, dtype=float)
    if mu_array.ndim != 1 or w_array.ndim != 1 or len(mu_array) != len(w_array):
        raise ValueError("Expected return inputs must be one-dimensional with equal length.")
    return float(mu_array @ w_array)


def portfolio_variance(
    Sigma: pd.DataFrame | np.ndarray,
    w: pd.Series | np.ndarray | Sequence[float],
) -> float:
    """Compute portfolio variance for the supplied covariance matrix and weights."""

    if isinstance(Sigma, pd.DataFrame):
        weights = _coerce_weight_series(w, Sigma.index)
        sigma_values = Sigma.reindex(index=weights.index, columns=weights.index).to_numpy(dtype=float)
        w_values = weights.to_numpy(dtype=float)
    else:
        sigma_values = np.asarray(Sigma, dtype=float)
        w_values = np.asarray(w, dtype=float)
        if sigma_values.ndim != 2 or w_values.ndim != 1 or sigma_values.shape[0] != sigma_values.shape[1]:
            raise ValueError("Sigma must be square and weights must be one-dimensional.")
        if sigma_values.shape[0] != len(w_values):
            raise ValueError("Sigma dimensions must match the weight vector.")

    variance = float(w_values @ sigma_values @ w_values)
    return max(variance, 0.0)


def portfolio_volatility(
    Sigma: pd.DataFrame | np.ndarray,
    w: pd.Series | np.ndarray | Sequence[float],
) -> float:
    """Compute portfolio volatility."""

    return float(np.sqrt(portfolio_variance(Sigma=Sigma, w=w)))


def portfolio_sharpe(
    mu: pd.Series | np.ndarray | Sequence[float],
    Sigma: pd.DataFrame | np.ndarray,
    w: pd.Series | np.ndarray | Sequence[float],
    risk_free_rate_annual: float,
    horizon_days: int,
) -> float:
    """Compute a horizon-consistent Sharpe ratio."""

    expected_return = portfolio_expected_return(mu=mu, w=w)
    volatility = portfolio_volatility(Sigma=Sigma, w=w)
    if volatility <= 0.0:
        return 0.0

    risk_free_horizon = risk_free_return_for_horizon(risk_free_rate_annual, horizon_days)
    return float((expected_return - risk_free_horizon) / volatility)


def compute_drawdown(equity_curve: pd.Series | Sequence[float] | np.ndarray) -> pd.Series:
    """Compute the drawdown path of an equity curve."""

    if isinstance(equity_curve, pd.Series):
        curve = equity_curve.astype(float).dropna()
    else:
        curve = pd.Series(np.asarray(equity_curve, dtype=float)).dropna()

    if curve.empty:
        raise ValueError("Equity curve is empty.")

    running_peak = curve.cummax()
    return curve / running_peak - 1.0


def compute_max_drawdown_from_returns(returns: pd.Series | Sequence[float] | np.ndarray) -> float:
    """Compute maximum drawdown from a return series."""

    return_series = _coerce_return_series(returns)
    if return_series.empty:
        raise ValueError("Return series is empty.")
    equity_curve = (1.0 + return_series).cumprod()
    drawdown = compute_drawdown(equity_curve)
    return float(drawdown.min())


def compute_cvar_from_returns(
    returns: pd.Series | Sequence[float] | np.ndarray,
    alpha: float = 0.95,
) -> float:
    """Compute historical CVaR / expected shortfall for a return series."""

    return_series = _coerce_return_series(returns)
    if return_series.empty:
        raise ValueError("Return series is empty.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")

    threshold = float(return_series.quantile(1.0 - alpha))
    tail = return_series[return_series <= threshold]
    if tail.empty:
        return threshold
    return float(tail.mean())


def estimate_portfolio_historical_risk(
    returns_window: pd.DataFrame,
    w: pd.Series | np.ndarray | Sequence[float],
) -> dict[str, float]:
    """Estimate historical portfolio risk statistics from a window of returns."""

    if returns_window.empty:
        raise ValueError("Return window is empty.")

    weights = _coerce_weight_series(w, returns_window.columns)
    portfolio_returns = returns_window.reindex(columns=weights.index).fillna(0.0) @ weights
    realized_vol = float(portfolio_returns.std(ddof=0) * np.sqrt(252.0))

    return {
        "cvar_95": compute_cvar_from_returns(portfolio_returns, alpha=0.95),
        "max_drawdown": compute_max_drawdown_from_returns(portfolio_returns),
        "realized_vol": realized_vol,
    }


def evaluate_risk_regime(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    universe: Sequence[AssetDefinition],
    config: RiskConfig,
    as_of: pd.Timestamp | None = None,
) -> RiskSnapshot:
    """Compatibility regime classifier based on the new market-state rules."""

    if prices.empty or returns.empty:
        raise ValueError("Prices and returns must not be empty.")

    if as_of is None:
        as_of = pd.Timestamp(prices.index[-1])

    risky_assets = [asset.symbol for asset in universe if asset.is_risky and asset.symbol in returns.columns]
    if not risky_assets:
        risky_assets = [str(column) for column in returns.columns]

    risky_returns = returns.loc[:as_of, risky_assets].dropna(how="all").fillna(0.0)
    risky_prices = prices.loc[:as_of, risky_assets].dropna(how="all").ffill(limit=3)
    basket_returns = risky_returns.mean(axis=1).fillna(0.0)

    if basket_returns.empty:
        raise ValueError("Not enough risky-asset history to evaluate the risk regime.")

    basket_curve = (1.0 + basket_returns).cumprod()
    market_drawdown = float(compute_drawdown(basket_curve).iloc[-1])
    realized_volatility = float(basket_returns.tail(min(21, len(basket_returns))).std(ddof=0) * np.sqrt(252.0))

    positive_breadth = 0.5
    if len(risky_prices) >= 64:
        positive_breadth = float((risky_prices.iloc[-1] / risky_prices.iloc[-64] - 1.0 > 0.0).mean())

    historical_risk = estimate_portfolio_historical_risk(
        returns_window=risky_returns.tail(min(config.cov_window, len(risky_returns))),
        w=np.repeat(1.0 / len(risky_assets), len(risky_assets)),
    )
    market_state = compute_market_risk_state(prices=prices, date=as_of)

    normal_risky_cap = min(
        1.0,
        MAX_EQUITY_LIKE_TOTAL_NORMAL + GROUP_LIMITS["commodities"] + CRYPTO_MAX_NORMAL,
    )
    risk_off_cap = min(
        1.0,
        MAX_EQUITY_LIKE_TOTAL_RISK_OFF + GROUP_LIMITS["commodities"] + CRYPTO_MAX_RISK_OFF,
    )

    if config.allow_daily_emergency_trades and (
        market_drawdown <= DRAWDOWN_LIMIT
        or historical_risk["cvar_95"] <= CVAR_LIMIT
    ):
        regime = RiskRegime.PAUSE
        de_risk_scalar = 0.0
        recommended_risky_cap = 0.15
    elif config.use_risk_filter and market_state == "risk_off":
        regime = RiskRegime.RISK_OFF
        de_risk_scalar = MAX_EQUITY_LIKE_TOTAL_RISK_OFF / MAX_EQUITY_LIKE_TOTAL_NORMAL
        recommended_risky_cap = risk_off_cap
    elif positive_breadth < 0.45:
        regime = RiskRegime.CAUTION
        de_risk_scalar = 0.85
        recommended_risky_cap = min(normal_risky_cap, 0.90)
    else:
        regime = RiskRegime.RISK_ON
        de_risk_scalar = 1.0
        recommended_risky_cap = normal_risky_cap

    diagnostics = {
        "market_state": market_state,
        "cvar_95": historical_risk["cvar_95"],
        "max_drawdown": historical_risk["max_drawdown"],
        "realized_vol": historical_risk["realized_vol"],
    }
    return RiskSnapshot(
        as_of=pd.Timestamp(as_of),
        regime=regime,
        market_drawdown=market_drawdown,
        realized_volatility=realized_volatility,
        positive_breadth=positive_breadth,
        de_risk_scalar=de_risk_scalar,
        recommended_risky_cap=recommended_risky_cap,
        diagnostics=diagnostics,
    )


def risky_asset_share(weights: pd.Series, universe: Sequence[AssetDefinition]) -> float:
    """Return the combined weight in assets classified as risky."""

    risky_symbols = {asset.symbol for asset in universe if asset.is_risky}
    return float(weights.reindex(sorted(risky_symbols)).fillna(0.0).sum())
