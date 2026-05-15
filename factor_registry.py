"""Registry of primary and meta factors used by the optional conditional layer."""

from __future__ import annotations


PRIMARY_FACTORS = [
    "equity_beta",
    "growth",
    "value",
    "quality",
    "momentum_factor",
    "low_volatility",
    "inflation",
    "real_rates",
    "nominal_rates",
    "duration",
    "credit_spread",
    "liquidity",
    "usd",
    "commodity",
    "oil",
    "volatility",
    "risk_appetite",
    "crypto_beta",
    "sector_rotation",
    "size",
    "international_equity",
    "emerging_markets",
]


META_FACTORS = [
    "labor_market",
    "wage_pressure",
    "inflation_surprise",
    "central_bank_policy_path",
    "credit_stress",
    "liquidity_stress",
    "commodity_supply_pressure",
    "earnings_revision_proxy",
    "geopolitical_stress_proxy",
    "dollar_liquidity",
]


def get_all_factors() -> list[str]:
    """Return all supported primary and meta factors."""

    return [*PRIMARY_FACTORS, *META_FACTORS]
