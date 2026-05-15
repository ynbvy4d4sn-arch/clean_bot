"""Build factor time series from proxy price history."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FactorDataResult:
    """Computed factor time series and diagnostics."""

    factor_data: pd.DataFrame
    diagnostics: list[str]


def _winsorize(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
    window: int = 252,
) -> pd.Series:
    """Apply causal trailing winsorization without using future observations."""

    clean = series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clean.empty:
        return clean

    min_periods = max(20, window // 4)
    lo = clean.rolling(window=window, min_periods=min_periods).quantile(lower)
    hi = clean.rolling(window=window, min_periods=min_periods).quantile(upper)
    lo = lo.combine_first(clean.expanding(min_periods=1).min())
    hi = hi.combine_first(clean.expanding(min_periods=1).max())
    return clean.clip(lower=lo, upper=hi)


def _rolling_zscore(series: pd.Series, window: int = 63) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(10, window // 4)).mean()
    std = series.rolling(window=window, min_periods=max(10, window // 4)).std(ddof=0)
    z = (series - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_factor_data(proxy_prices: pd.DataFrame) -> FactorDataResult:
    """Compute factor proxy time series with robust neutral fallbacks."""

    diagnostics: list[str] = []
    if proxy_prices.empty:
        return FactorDataResult(
            factor_data=pd.DataFrame(index=pd.Index([], dtype="datetime64[ns]")),
            diagnostics=["Proxy price data is empty; conditional factor layer will use direct-only fallback."],
        )

    returns = proxy_prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    idx = returns.index
    factors = pd.DataFrame(index=idx)

    def col(name: str) -> pd.Series:
        return returns[name] if name in returns.columns else pd.Series(0.0, index=idx, dtype=float)

    def mean_col(names: list[str]) -> pd.Series:
        available = [name for name in names if name in returns.columns]
        if not available:
            return pd.Series(0.0, index=idx, dtype=float)
        return returns[available].mean(axis=1).astype(float)

    core_equity = mean_col(["SPHQ", "SPMO", "SPLV", "XLK", "XLI", "XLV", "XLC", "XLY", "XLE", "XLF", "XLRE", "XLB", "XLP", "XLU", "VEA", "VWO", "RPV", "SIZE", "VBR"])
    cyclical_equity = mean_col(["SPMO", "XLK", "XLC", "XLY", "XLI", "XLE", "XLF", "XLB", "VWO", "VBR"])
    defensive_equity = mean_col(["SPLV", "XLP", "XLU", "XLV"])
    duration_bonds = mean_col(["IEF", "AGG", "TLT"])
    commodity_basket = mean_col(["PDBC", "GLD", "SLV"])

    factors["equity_beta"] = core_equity
    factors["growth"] = cyclical_equity - defensive_equity
    factors["quality"] = col("SPHQ") - core_equity
    factors["momentum_factor"] = col("SPMO") - col("SPLV")
    factors["low_volatility"] = col("SPLV") - col("SPMO")
    factors["nominal_rates"] = -duration_bonds
    factors["duration"] = -(col("TLT") if "TLT" in returns.columns else (col("IEF") if "IEF" in returns.columns else duration_bonds))
    factors["inflation"] = col("TIP") - duration_bonds
    factors["credit_spread"] = mean_col(["LQD", "HYG", "EMB"]) - (col("IEF") if "IEF" in returns.columns else duration_bonds)
    factors["usd"] = -commodity_basket
    factors["commodity"] = commodity_basket
    factors["oil"] = col("PDBC")
    factors["volatility"] = col("SH") if "SH" in returns.columns else -core_equity
    factors["risk_appetite"] = (0.75 * core_equity + 0.25 * mean_col(["HYG", "EMB"])) - col("SHY")
    factors["crypto_beta"] = col("IBIT")
    factors["sector_rotation"] = col("XLK") - col("XLU")
    factors["liquidity"] = col("SPHQ") - col("LQD")
    factors["real_rates"] = -(col("TIP") if "TIP" in returns.columns else duration_bonds)
    factors["value"] = mean_col(["RPV", "VBR", "XLF", "XLE", "XLB", "XLI", "XLP"]) - col("XLK")
    factors["size"] = mean_col(["SIZE", "VBR"]) - core_equity
    factors["international_equity"] = mean_col(["VEA", "VWO"]) - core_equity
    factors["emerging_markets"] = col("VWO") - col("VEA")

    factors = factors.apply(_winsorize).apply(_rolling_zscore)
    if (factors.abs().sum(axis=0) == 0.0).all():
        diagnostics.append("All factor series are neutral after proxy construction.")
    return FactorDataResult(factor_data=factors, diagnostics=diagnostics)
