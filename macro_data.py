"""Macro and proxy data helpers for the optional conditional factor layer."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from asset_universe import get_enabled_tickers


DEFAULT_PROXY_TICKERS = get_enabled_tickers()


@dataclass(slots=True)
class MacroProxyBundle:
    """Available and missing macro proxy series."""

    proxy_prices: pd.DataFrame
    available_proxies: list[str]
    missing_proxies: list[str]
    diagnostics: list[str]


def load_macro_proxy_data(
    prices: pd.DataFrame,
    date: pd.Timestamp | str | None = None,
    proxy_tickers: list[str] | None = None,
) -> MacroProxyBundle:
    """Return the subset of proxy prices available up to the requested date."""

    as_of = pd.Timestamp(date if date is not None else prices.index[-1])
    proxies = [str(ticker) for ticker in (proxy_tickers or DEFAULT_PROXY_TICKERS)]
    history = prices.loc[:as_of].sort_index()
    available = [ticker for ticker in proxies if ticker in history.columns]
    missing = [ticker for ticker in proxies if ticker not in history.columns]
    diagnostics: list[str] = []
    if missing:
        diagnostics.append("Missing macro proxies: " + ", ".join(missing))
    if not available:
        diagnostics.append("No macro proxies available; conditional factor layer will fall back to direct-only mode.")
    return MacroProxyBundle(
        proxy_prices=history.reindex(columns=available).copy(),
        available_proxies=available,
        missing_proxies=missing,
        diagnostics=diagnostics,
    )
