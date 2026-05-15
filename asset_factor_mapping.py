"""Qualitative prior mapping from assets to factors."""

from __future__ import annotations

from asset_universe import get_enabled_tickers


ASSET_FACTOR_PRIORS: dict[str, dict[str, float]] = {
    "XLK": {"growth": 1.0, "real_rates": -0.7, "risk_appetite": 0.8, "volatility": -0.6, "usd": -0.2},
    "XLC": {"growth": 0.8, "risk_appetite": 0.6, "volatility": -0.4},
    "XLY": {"growth": 0.7, "risk_appetite": 0.8, "volatility": -0.5},
    "XLP": {"risk_appetite": -0.3, "volatility": -0.2, "inflation": 0.2},
    "XLV": {"quality": 0.5, "risk_appetite": 0.2, "volatility": -0.2},
    "XLF": {"nominal_rates": 0.5, "credit_spread": -0.5, "risk_appetite": 0.6},
    "XLI": {"equity_beta": 0.6, "risk_appetite": 0.5, "sector_rotation": 0.4},
    "XLE": {"oil": 1.0, "inflation": 0.7, "commodity": 0.8, "usd": -0.4},
    "XLB": {"commodity": 0.7, "inflation": 0.4, "usd": -0.2},
    "XLU": {"nominal_rates": -0.5, "volatility": -0.2, "risk_appetite": -0.1},
    "XLRE": {"real_rates": -0.8, "nominal_rates": -0.5, "risk_appetite": 0.2},
    "SPMO": {"momentum_factor": 1.0, "risk_appetite": 0.6, "volatility": -0.2},
    "SPHQ": {"quality": 1.0, "risk_appetite": 0.3, "volatility": -0.3},
    "SPLV": {"low_volatility": 1.0, "risk_appetite": -0.1, "volatility": -0.5},
    "RPV": {"value": 1.0, "risk_appetite": 0.4, "growth": -0.4, "nominal_rates": 0.2},
    "SIZE": {"size": 1.0, "risk_appetite": 0.5, "liquidity": -0.2},
    "VBR": {"value": 0.8, "size": 0.7, "risk_appetite": 0.5, "growth": -0.2},
    "VEA": {"international_equity": 1.0, "equity_beta": 0.7, "usd": -0.6, "risk_appetite": 0.5},
    "VWO": {"international_equity": 1.0, "emerging_markets": 1.0, "equity_beta": 0.8, "usd": -0.8, "risk_appetite": 0.7},
    "MTUM": {"momentum_factor": 1.0, "risk_appetite": 0.6, "volatility": -0.2},
    "QUAL": {"quality": 1.0, "risk_appetite": 0.3, "volatility": -0.3},
    "USMV": {"low_volatility": 1.0, "risk_appetite": -0.1, "volatility": -0.5},
    "SGOV": {"nominal_rates": 0.1, "liquidity": 0.3, "volatility": 0.0},
    "SHY": {"nominal_rates": -0.3, "duration": 0.2, "liquidity": 0.2},
    "IEF": {"nominal_rates": -0.8, "real_rates": -0.5, "duration": 1.0, "risk_appetite": -0.2},
    "TLT": {"nominal_rates": -1.0, "real_rates": -0.8, "duration": 1.0, "risk_appetite": -0.3},
    "AGG": {"nominal_rates": -0.6, "duration": 0.7, "credit_spread": -0.2},
    "LQD": {"nominal_rates": -0.6, "credit_spread": -0.9, "liquidity": 0.4, "risk_appetite": 0.4},
    "HYG": {"nominal_rates": -0.3, "credit_spread": -1.0, "liquidity": -0.2, "risk_appetite": 0.7},
    "EMB": {"nominal_rates": -0.4, "credit_spread": -0.8, "emerging_markets": 0.8, "usd": -0.6, "risk_appetite": 0.7},
    "TIP": {"inflation": 0.8, "real_rates": -0.5, "duration": 0.4},
    "PDBC": {"commodity": 1.0, "inflation": 0.7, "usd": -0.5, "risk_appetite": 0.2},
    "DBP": {"commodity": 0.8, "inflation": 0.5, "usd": -0.4},
    "CPER": {"commodity": 0.9, "growth": 0.4, "usd": -0.5},
    "DBA": {"commodity": 0.8, "inflation": 0.5, "usd": -0.4},
    "REMX": {"commodity": 0.6, "growth": 0.4, "geopolitical_stress_proxy": 0.2},
    "SH": {"equity_beta": -1.0, "risk_appetite": -0.8, "volatility": 0.5},
    "PSQ": {"growth": -0.8, "risk_appetite": -0.8, "volatility": 0.5},
    "IBIT": {"crypto_beta": 1.0, "liquidity": 0.7, "real_rates": -0.5, "usd": -0.3, "risk_appetite": 0.8},
    "ETHA": {"crypto_beta": 1.0, "liquidity": 0.7, "real_rates": -0.5, "usd": -0.3, "risk_appetite": 0.8},
}


def get_asset_factor_mapping() -> dict[str, dict[str, float]]:
    """Return qualitative factor priors for enabled assets."""

    enabled = set(get_enabled_tickers())
    return {ticker: mapping for ticker, mapping in ASSET_FACTOR_PRIORS.items() if ticker in enabled}
