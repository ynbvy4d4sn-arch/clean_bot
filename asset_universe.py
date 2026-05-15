"""Asset registry and compatibility helpers for the ETF universe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

EXPECTED_CASH_TICKER = "SGOV"
DEFAULT_DISABLED_TICKERS: tuple[str, ...] = ()
REQUIRED_ENABLED_FIELDS = ("name", "group", "subgroup", "max_weight", "enabled")


ASSET_UNIVERSE: dict[str, dict[str, str | float | bool]] = {
    "AGG": {
        "name": "iShares Core U.S. Aggregate Bond ETF",
        "group": "bonds",
        "subgroup": "aggregate_bond",
        "max_weight": 0.40,
        "enabled": True,
    },
    "GLD": {
        "name": "SPDR Gold Shares",
        "group": "commodities",
        "subgroup": "gold",
        "max_weight": 0.10,
        "enabled": True,
    },
    "IBIT": {
        "name": "iShares Bitcoin Trust ETF",
        "group": "crypto",
        "subgroup": "bitcoin",
        "max_weight": 0.05,
        "enabled": True,
    },
    "IEF": {
        "name": "iShares 7-10 Year Treasury Bond ETF",
        "group": "bonds",
        "subgroup": "intermediate_treasury",
        "max_weight": 0.40,
        "enabled": True,
    },
    "LQD": {
        "name": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "group": "bonds",
        "subgroup": "investment_grade_credit",
        "max_weight": 0.25,
        "enabled": True,
    },
    "PDBC": {
        "name": "Invesco Optimum Yield Diversified Commodity Strategy No K-1 ETF",
        "group": "commodities",
        "subgroup": "broad_commodities",
        "max_weight": 0.10,
        "enabled": True,
    },
    "SGOV": {
        "name": "iShares 0-3 Month Treasury Bond ETF",
        "group": "cash",
        "subgroup": "treasury_bills",
        "max_weight": 0.70,
        "enabled": True,
    },
    "SHY": {
        "name": "iShares 1-3 Year Treasury Bond ETF",
        "group": "bonds",
        "subgroup": "short_treasury",
        "max_weight": 0.40,
        "enabled": True,
    },
    "SLV": {
        "name": "iShares Silver Trust",
        "group": "commodities",
        "subgroup": "silver",
        "max_weight": 0.10,
        "enabled": True,
    },
    "SPHQ": {
        "name": "Invesco S&P 500 Quality ETF",
        "group": "factor",
        "subgroup": "quality",
        "max_weight": 0.15,
        "enabled": True,
    },
    "SPLV": {
        "name": "Invesco S&P 500 Low Volatility ETF",
        "group": "factor",
        "subgroup": "low_volatility",
        "max_weight": 0.15,
        "enabled": True,
    },
    "SPMO": {
        "name": "Invesco S&P 500 Momentum ETF",
        "group": "factor",
        "subgroup": "momentum",
        "max_weight": 0.15,
        "enabled": True,
    },
    "TIP": {
        "name": "iShares TIPS Bond ETF",
        "group": "bonds",
        "subgroup": "inflation_linked",
        "max_weight": 0.25,
        "enabled": True,
    },
    "XLC": {
        "name": "Communication Services Select Sector SPDR Fund",
        "group": "us_sector",
        "subgroup": "communication_services",
        "max_weight": 0.15,
        "enabled": True,
    },
    "XLY": {
        "name": "Consumer Discretionary Select Sector SPDR Fund",
        "group": "us_sector",
        "subgroup": "consumer_discretionary",
        "max_weight": 0.15,
        "enabled": True,
    },
    "XLE": {
        "name": "Energy Select Sector SPDR Fund",
        "group": "us_sector",
        "subgroup": "energy",
        "max_weight": 0.15,
        "enabled": True,
    },
    "XLF": {
        "name": "Financial Select Sector SPDR Fund",
        "group": "us_sector",
        "subgroup": "financials",
        "max_weight": 0.15,
        "enabled": True,
    },
    "XLRE": {
        "name": "Real Estate Select Sector SPDR Fund",
        "group": "us_sector",
        "subgroup": "real_estate",
        "max_weight": 0.15,
        "enabled": True,
    },
    "XLB": {
        "name": "Materials Select Sector SPDR Fund",
        "group": "us_sector",
        "subgroup": "materials",
        "max_weight": 0.15,
        "enabled": True,
    },
    "XLI": {
        "name": "Industrial Select Sector SPDR Fund",
        "group": "us_sector",
        "subgroup": "industrials",
        "max_weight": 0.15,
        "enabled": True,
    },
    "XLK": {
        "name": "Technology Select Sector SPDR Fund",
        "group": "us_sector",
        "subgroup": "technology",
        "max_weight": 0.15,
        "enabled": True,
    },
    "XLP": {
        "name": "Consumer Staples Select Sector SPDR Fund",
        "group": "us_sector",
        "subgroup": "consumer_staples",
        "max_weight": 0.15,
        "enabled": True,
    },
    "XLU": {
        "name": "Utilities Select Sector SPDR Fund",
        "group": "us_sector",
        "subgroup": "utilities",
        "max_weight": 0.15,
        "enabled": True,
    },
    "XLV": {
        "name": "Health Care Select Sector SPDR Fund",
        "group": "us_sector",
        "subgroup": "health_care",
        "max_weight": 0.15,
        "enabled": True,
    },
    "VEA": {
        "name": "Vanguard FTSE Developed Markets ETF",
        "group": "international_equity",
        "subgroup": "developed_markets",
        "max_weight": 0.15,
        "enabled": True,
    },
    "VWO": {
        "name": "Vanguard FTSE Emerging Markets ETF",
        "group": "international_equity",
        "subgroup": "emerging_markets",
        "max_weight": 0.15,
        "enabled": True,
    },
    "RPV": {
        "name": "Invesco S&P 500 Pure Value ETF",
        "group": "factor",
        "subgroup": "value",
        "max_weight": 0.15,
        "enabled": True,
    },
    "SIZE": {
        "name": "iShares MSCI USA Size Factor ETF",
        "group": "factor",
        "subgroup": "size",
        "max_weight": 0.15,
        "enabled": True,
    },
    "VBR": {
        "name": "Vanguard Small-Cap Value ETF",
        "group": "factor",
        "subgroup": "small_value",
        "max_weight": 0.15,
        "enabled": True,
    },
    "TLT": {
        "name": "iShares 20+ Year Treasury Bond ETF",
        "group": "bonds",
        "subgroup": "long_treasury",
        "max_weight": 0.25,
        "enabled": True,
    },
    "HYG": {
        "name": "iShares iBoxx $ High Yield Corporate Bond ETF",
        "group": "bonds",
        "subgroup": "high_yield_credit",
        "max_weight": 0.15,
        "enabled": True,
    },
    "EMB": {
        "name": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
        "group": "bonds",
        "subgroup": "emerging_market_debt",
        "max_weight": 0.15,
        "enabled": True,
    },
    "SH": {
        "name": "ProShares Short S&P500",
        "group": "hedge",
        "subgroup": "short_sp500",
        "max_weight": 0.10,
        "enabled": True,
    },
}


GROUP_LIMITS: dict[str, float] = {
    "us_sector": 0.55,
    "international_equity": 0.30,
    "factor": 0.30,
    "cash": 0.70,
    "bonds": 0.70,
    "commodities": 0.25,
    "hedge": 0.15,
    "crypto": 0.10,
}

EQUITY_LIKE_GROUPS = ["us_sector", "international_equity", "factor"]
MAX_EQUITY_LIKE_TOTAL_NORMAL = 0.70
MAX_EQUITY_LIKE_TOTAL_RISK_OFF = 0.50

DEFENSIVE_GROUPS = ["cash", "bonds"]
MIN_DEFENSIVE_WEIGHT_NORMAL = 0.10
MIN_DEFENSIVE_WEIGHT_RISK_OFF = 0.30

CRYPTO_MAX_NORMAL = 0.10
CRYPTO_MAX_RISK_OFF = 0.03

DEFAULT_GROUP_BENCHMARK_WEIGHTS: dict[str, float] = {
    "us_sector": 0.35,
    "international_equity": 0.10,
    "factor": 0.15,
    "cash": 0.10,
    "bonds": 0.20,
    "commodities": 0.10,
    "hedge": 0.00,
    "crypto": 0.10,
}

GROUP_TO_ASSET_CLASS: dict[str, str] = {
    "us_sector": "Equity",
    "international_equity": "Equity",
    "factor": "Equity",
    "cash": "Cash",
    "bonds": "Bond",
    "commodities": "Commodity",
    "hedge": "Hedge",
    "crypto": "Crypto",
}

GROUP_TO_REGION: dict[str, str] = {
    "us_sector": "US",
    "international_equity": "Global ex-US",
    "factor": "US",
    "cash": "US",
    "bonds": "US",
    "commodities": "Global",
    "hedge": "US",
    "crypto": "Global",
}


@dataclass(frozen=True, slots=True)
class AssetDefinition:
    """Compatibility representation for the active asset universe."""

    symbol: str
    name: str
    group: str
    subgroup: str
    min_weight: float
    max_weight: float
    benchmark_weight: float
    enabled: bool = True

    @property
    def asset_class(self) -> str:
        """Return a coarse asset class used by the rest of the pipeline."""

        return GROUP_TO_ASSET_CLASS[self.group]

    @property
    def region(self) -> str:
        """Return a coarse region label for reporting compatibility."""

        return GROUP_TO_REGION[self.group]

    @property
    def is_risky(self) -> bool:
        """Return whether the asset should be treated as risk exposure."""

        return self.group in {"us_sector", "international_equity", "factor", "commodities", "crypto"}

    @property
    def is_defensive(self) -> bool:
        """Return whether the asset belongs to a defensive bucket."""

        return self.group in set(DEFENSIVE_GROUPS)


def _enabled_registry_items() -> list[tuple[str, dict[str, str | float | bool]]]:
    """Return enabled registry items in insertion order."""

    return [
        (ticker, config)
        for ticker, config in ASSET_UNIVERSE.items()
        if bool(config.get("enabled", False))
    ]


def _benchmark_weight_map_for_tickers(tickers: Sequence[str]) -> dict[str, float]:
    """Build neutral benchmark weights across enabled assets by group."""

    present_groups: dict[str, list[str]] = {}
    for ticker in tickers:
        group = str(ASSET_UNIVERSE[ticker]["group"])
        present_groups.setdefault(group, []).append(ticker)

    usable_group_targets = {
        group: target
        for group, target in DEFAULT_GROUP_BENCHMARK_WEIGHTS.items()
        if group in present_groups and target > 0.0
    }
    target_total = float(sum(usable_group_targets.values()))

    if target_total <= 0.0:
        equal_weight = 1.0 / len(tickers) if tickers else 0.0
        return {ticker: equal_weight for ticker in tickers}

    benchmark: dict[str, float] = {ticker: 0.0 for ticker in tickers}
    for group, group_target in usable_group_targets.items():
        group_tickers = present_groups[group]
        per_asset_weight = group_target / target_total / len(group_tickers)
        for ticker in group_tickers:
            benchmark[ticker] = per_asset_weight

    return benchmark


def get_enabled_tickers() -> list[str]:
    """Return all enabled tickers in registry order."""

    return [ticker for ticker, _ in _enabled_registry_items()]


def get_cash_ticker() -> str:
    """Return the single enabled cash ticker or raise if the registry is invalid."""

    cash_tickers = get_tickers_by_group("cash")
    if len(cash_tickers) != 1:
        raise ValueError(
            "Expected exactly one enabled cash ticker and it must be "
            f"{EXPECTED_CASH_TICKER}, found {len(cash_tickers)}: {cash_tickers}"
        )
    if cash_tickers[0] != EXPECTED_CASH_TICKER:
        raise ValueError(
            f"Expected cash ticker {EXPECTED_CASH_TICKER}, found {cash_tickers[0]}."
        )
    return cash_tickers[0]


def get_group_map() -> dict[str, str]:
    """Return a mapping from enabled tickers to their top-level group."""

    return {
        ticker: str(config["group"])
        for ticker, config in _enabled_registry_items()
    }


def get_subgroup_map() -> dict[str, str]:
    """Return a mapping from enabled tickers to their subgroup."""

    return {
        ticker: str(config["subgroup"])
        for ticker, config in _enabled_registry_items()
    }


def get_group_limits() -> dict[str, float]:
    """Return a copy of the configured group limits."""

    return GROUP_LIMITS.copy()


def get_asset_max_weights() -> dict[str, float]:
    """Return max weights for all enabled assets."""

    return {
        ticker: float(config["max_weight"])
        for ticker, config in _enabled_registry_items()
    }


def get_tickers_by_group(group: str) -> list[str]:
    """Return enabled tickers that belong to the supplied group."""

    return [
        ticker
        for ticker, config in _enabled_registry_items()
        if str(config["group"]) == group
    ]


def validate_asset_universe() -> bool:
    """Validate the active asset registry and raise on configuration errors."""

    enabled_entries = _enabled_registry_items()
    enabled_tickers = [ticker for ticker, _ in enabled_entries]
    errors: list[str] = []

    if len(enabled_tickers) < 10:
        errors.append("Asset universe must contain at least 10 enabled assets.")

    if len(enabled_tickers) != len(set(enabled_tickers)):
        errors.append("Asset universe contains duplicate enabled tickers.")

    cash_tickers = get_tickers_by_group("cash")
    if len(cash_tickers) != 1:
        errors.append(
            "Asset universe must contain exactly one enabled cash ticker "
            f"({EXPECTED_CASH_TICKER}), found {len(cash_tickers)}: {cash_tickers}."
        )
    elif cash_tickers[0] != EXPECTED_CASH_TICKER:
        errors.append(
            f"Asset universe cash ticker must be {EXPECTED_CASH_TICKER}, found {cash_tickers[0]}."
        )

    for ticker in DEFAULT_DISABLED_TICKERS:
        if ticker in ASSET_UNIVERSE and bool(ASSET_UNIVERSE[ticker].get("enabled", False)):
            errors.append(f"{ticker} must remain disabled in the default V1 universe.")

    sgov_config = ASSET_UNIVERSE.get(EXPECTED_CASH_TICKER)
    if sgov_config is None:
        errors.append(f"Required cash ticker {EXPECTED_CASH_TICKER} is missing from ASSET_UNIVERSE.")
    elif not bool(sgov_config.get("enabled", False)):
        errors.append(f"Required cash ticker {EXPECTED_CASH_TICKER} must be enabled.")

    present_groups = {str(config["group"]) for _, config in enabled_entries if "group" in config}
    missing_group_limits = sorted(group for group in present_groups if group not in GROUP_LIMITS)
    if missing_group_limits:
        errors.append(
            "Missing group limits for enabled groups: " + ", ".join(missing_group_limits) + "."
        )

    for group in sorted(present_groups):
        limit = GROUP_LIMITS.get(group)
        if not isinstance(limit, (int, float)):
            errors.append(f"Group limit for '{group}' must be numeric.")
        elif float(limit) <= 0.0 or float(limit) > 1.0:
            errors.append(f"Group limit for '{group}' must be > 0 and <= 1.")

    for ticker, config in enabled_entries:
        for required_key in REQUIRED_ENABLED_FIELDS:
            if required_key not in config:
                errors.append(f"{ticker} is missing required field '{required_key}'.")

        name = config.get("name")
        group = config.get("group")
        subgroup = config.get("subgroup")
        max_weight = config.get("max_weight")
        enabled = config.get("enabled")

        if not name or not isinstance(name, str):
            errors.append(f"{ticker} must define a non-empty string name.")

        if not group or not isinstance(group, str):
            errors.append(f"{ticker} must define a non-empty string group.")
        elif group not in GROUP_LIMITS:
            errors.append(f"{ticker} uses unknown group '{group}' with no group limit.")

        if not subgroup or not isinstance(subgroup, str):
            errors.append(f"{ticker} must define a non-empty string subgroup.")

        if not isinstance(max_weight, (int, float)):
            errors.append(f"{ticker} max_weight must be numeric.")
        else:
            weight_value = float(max_weight)
            if weight_value <= 0.0 or weight_value > 1.0:
                errors.append(f"{ticker} max_weight must be > 0 and <= 1.")

        if not isinstance(enabled, bool):
            errors.append(f"{ticker} enabled flag must be boolean.")

    if errors:
        raise ValueError("Invalid asset universe configuration:\n- " + "\n- ".join(errors))
    return True


def build_default_universe() -> list[AssetDefinition]:
    """Return the enabled V1 asset universe as structured asset definitions."""

    validate_asset_universe()
    enabled_tickers = get_enabled_tickers()
    benchmark_map = _benchmark_weight_map_for_tickers(enabled_tickers)

    return [
        AssetDefinition(
            symbol=ticker,
            name=str(ASSET_UNIVERSE[ticker]["name"]),
            group=str(ASSET_UNIVERSE[ticker]["group"]),
            subgroup=str(ASSET_UNIVERSE[ticker]["subgroup"]),
            min_weight=0.0,
            max_weight=float(ASSET_UNIVERSE[ticker]["max_weight"]),
            benchmark_weight=float(benchmark_map[ticker]),
            enabled=bool(ASSET_UNIVERSE[ticker]["enabled"]),
        )
        for ticker in enabled_tickers
    ]


def get_enabled_assets(
    universe: Iterable[AssetDefinition] | None = None,
) -> list[AssetDefinition]:
    """Return enabled assets from a provided universe or from the registry."""

    if universe is None:
        return build_default_universe()
    return [asset for asset in universe if asset.enabled]


def symbols(universe: Iterable[AssetDefinition]) -> list[str]:
    """Return symbols in universe order."""

    return [asset.symbol for asset in universe]


def benchmark_weights(universe: Iterable[AssetDefinition]) -> pd.Series:
    """Return benchmark weights aligned to the supplied universe order."""

    import pandas as pd

    ordered = [asset for asset in universe if asset.enabled]
    benchmark = pd.Series(
        {asset.symbol: asset.benchmark_weight for asset in ordered},
        dtype=float,
    )
    total = float(benchmark.sum())
    if total <= 0.0:
        raise ValueError("Benchmark weights must sum to a positive value.")
    return benchmark / total


validate_asset_universe()
