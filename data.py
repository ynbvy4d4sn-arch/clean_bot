"""Historical price loading utilities backed by yfinance with local caching."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional in stabilization mode
    yf = None

from asset_universe import AssetDefinition, get_enabled_tickers, symbols
from calendar_utils import DEFAULT_PROJECT_CALENDAR_PATH, get_expected_latest_trading_day
from config import COV_WINDOW, DATA_DIR, MOMENTUM_LONG, PRICE_CACHE_PATH


LOGGER = logging.getLogger(__name__)
BERLIN_TZ = ZoneInfo("Europe/Berlin")
MIN_REQUIRED_OBSERVATIONS = max(300, MOMENTUM_LONG + 1, COV_WINDOW + 1)
MIN_REQUIRED_ASSETS = 10
YFINANCE_CACHE_DIR = DATA_DIR / "yfinance_cache"
PROJECT_CALENDAR_PATH = DATA_DIR / "trading_calendar_2026.csv"
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

if yf is not None:
    try:  # pragma: no cover - depends on yfinance internals
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))
        if hasattr(yf, "cache") and hasattr(yf.cache, "set_cache_location"):
            yf.cache.set_cache_location(str(YFINANCE_CACHE_DIR))
    except Exception as exc:  # pragma: no cover - best effort only
        LOGGER.warning("Could not configure local yfinance cache directory: %s", exc)


@dataclass(slots=True)
class MarketDataBundle:
    """Container for market data used by the research pipeline."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    volumes: pd.DataFrame | None = None


@dataclass(slots=True)
class RunDataContext:
    """Canonical run-level data and freshness context shared across reports."""

    data_source: str = "unknown"
    cache_status: str = "unknown"
    synthetic_data: bool = False
    used_cache_fallback: bool = False
    latest_price_date: str | None = None
    expected_latest_trading_day: str | None = None
    staleness_days: int | None = None
    data_freshness_ok: bool = False
    live_data_error: str = ""
    tickers_loaded: list[str] = field(default_factory=list)
    tickers_failed: list[str] = field(default_factory=list)
    price_basis: str = "adjusted_close_proxy"
    yfinance_available: bool = False
    warning: str = ""
    project_calendar_path: str = str(DEFAULT_PROJECT_CALENDAR_PATH)
    calendar_status: str = "unknown"
    current_date_berlin: str | None = None
    current_time_berlin: str | None = None
    is_project_trading_day: bool = False
    allowed_start_berlin: str | None = None
    allowed_end_berlin: str | None = None
    within_allowed_window: bool = False
    execution_allowed_by_calendar: bool = False
    run_context: str = "daily_bot"

    def as_dict(self) -> dict[str, object]:
        """Return a plain dict for diagnostics and report writers."""

        return asdict(self)


def _extract_close_prices(raw: pd.DataFrame | pd.Series, tickers: Sequence[str]) -> pd.DataFrame:
    """Extract close prices from a yfinance download result."""

    if isinstance(raw, pd.Series):
        return raw.to_frame(name=tickers[0] if tickers else "Close")

    if raw.empty:
        raise ValueError("yfinance returned an empty dataset.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded data does not contain a 'Close' field.")
        prices = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded data does not contain a 'Close' column.")
        prices = raw[["Close"]].copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0] if tickers else "Close")

    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices.columns = [str(column) for column in prices.columns]
    if len(tickers) == 1 and prices.shape[1] == 1:
        prices.columns = [tickers[0]]
    return prices


def validate_price_data(prices: pd.DataFrame, min_rows: int = 300) -> pd.DataFrame:
    """Validate and normalize a price DataFrame."""

    if prices.empty:
        raise ValueError("Price history is empty.")

    validated = prices.copy()
    validated.index = pd.to_datetime(validated.index)
    validated = validated.sort_index()
    validated = validated.replace([float("inf"), float("-inf")], pd.NA)

    if len(validated) < int(min_rows):
        raise ValueError(
            f"Too few synchronized price observations after cleaning: {len(validated)} rows "
            f"available, at least {int(min_rows)} required."
        )

    if validated.shape[1] < MIN_REQUIRED_ASSETS:
        raise ValueError(
            f"Too few assets remain after data cleaning: {validated.shape[1]} available, "
            f"at least {MIN_REQUIRED_ASSETS} required."
        )

    return validated


def _cache_metadata_path(path: str | Path) -> Path:
    """Return the sidecar metadata path for a CSV price cache."""

    cache_path = Path(path)
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _atomic_write_text(path: str | Path, text: str) -> Path:
    """Write a text file atomically."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f"{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temp_path = Path(handle.name)
        temp_path.replace(target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return target


def _atomic_write_csv(frame: pd.DataFrame, path: str | Path, **kwargs: object) -> Path:
    """Write a CSV atomically."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            prefix=f"{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        frame.to_csv(temp_path, **kwargs)
        temp_path.replace(target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return target


def save_price_cache(prices: pd.DataFrame, path: str | Path) -> Path:
    """Persist a cleaned price matrix to CSV."""

    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_prices = prices.sort_index().copy()
    incoming_attrs = dict(cache_prices.attrs)
    existing_prices = load_price_cache(cache_path)
    if not existing_prices.empty:
        existing_attrs = dict(existing_prices.attrs)
        existing_notes = list(existing_prices.attrs.get("data_notes", []))
        incoming_notes = list(cache_prices.attrs.get("data_notes", []))
        all_index = existing_prices.index.union(cache_prices.index).sort_values()
        all_columns = list(
            dict.fromkeys([*existing_prices.columns.tolist(), *cache_prices.columns.tolist()])
        )
        existing_aligned = existing_prices.reindex(index=all_index, columns=all_columns)
        incoming_aligned = cache_prices.reindex(index=all_index, columns=all_columns)

        incoming_is_synthetic = bool(cache_prices.attrs.get("synthetic_data", False))
        existing_is_synthetic = bool(existing_prices.attrs.get("synthetic_data", False))
        if incoming_is_synthetic and not existing_is_synthetic:
            # Preserve valid real cached observations and only fill gaps with fallback data.
            cache_prices = existing_aligned.combine_first(incoming_aligned)
            cache_prices.attrs.update(existing_attrs)
            cache_prices.attrs.update(incoming_attrs)
            cache_prices.attrs["synthetic_data"] = True
        elif existing_is_synthetic and not incoming_is_synthetic:
            # Once real market data is available, do not re-mix historical
            # synthetic fallback rows back into the cache.
            cache_prices = cache_prices.sort_index().copy()
            cache_prices.attrs.update(incoming_attrs)
            cache_prices.attrs["synthetic_data"] = False
        else:
            # Prefer the newest data while preserving columns not touched in this run.
            cache_prices = incoming_aligned.combine_first(existing_aligned)
            cache_prices.attrs.update(existing_attrs)
            cache_prices.attrs.update(incoming_attrs)
            cache_prices.attrs["synthetic_data"] = incoming_is_synthetic or existing_is_synthetic

        incoming_data_source = str(cache_prices.attrs.get("data_source", "unknown"))
        if not incoming_is_synthetic and incoming_data_source == "yfinance":
            effective_notes = incoming_notes
        else:
            effective_notes = incoming_notes if incoming_notes else existing_notes
        cache_prices.attrs["data_notes"] = _dedupe_notes(effective_notes)
    else:
        cache_prices.attrs.update(incoming_attrs)

    cache_prices = cache_prices.dropna(how="all").sort_index()
    _atomic_write_csv(cache_prices, cache_path)
    metadata = {
        "synthetic_data": bool(cache_prices.attrs.get("synthetic_data", False)),
        "data_notes": list(cache_prices.attrs.get("data_notes", [])),
        "data_source": str(cache_prices.attrs.get("data_source", "unknown")),
        "cache_status": str(cache_prices.attrs.get("cache_status", "saved")),
        "latest_price_date": str(cache_prices.attrs.get("latest_price_date", "")),
        "price_basis": str(cache_prices.attrs.get("price_basis", "adjusted_close_proxy")),
        "yfinance_available": bool(cache_prices.attrs.get("yfinance_available", yf is not None)),
        "tickers_loaded": list(cache_prices.attrs.get("tickers_loaded", cache_prices.columns.tolist())),
        "tickers_failed": list(cache_prices.attrs.get("tickers_failed", [])),
        "used_cache_fallback": bool(cache_prices.attrs.get("used_cache_fallback", False)),
        "live_data_error": str(cache_prices.attrs.get("live_data_error", "")),
        "cache_written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_text(
        _cache_metadata_path(cache_path),
        json.dumps(metadata, indent=2, sort_keys=True),
    )
    return cache_path


def load_price_cache(path: str | Path) -> pd.DataFrame:
    """Load cached prices from CSV if available."""

    cache_path = Path(path)
    if not cache_path.exists():
        return pd.DataFrame()

    try:
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    except Exception as exc:
        LOGGER.warning("Could not read price cache %s: %s", cache_path, exc)
        return pd.DataFrame()

    if prices.empty:
        return pd.DataFrame()

    parsed_index = pd.to_datetime(prices.index, errors="coerce")
    invalid_rows = int(parsed_index.isna().sum())
    if invalid_rows:
        LOGGER.warning("Dropped %s malformed cache row(s) from %s during load.", invalid_rows, cache_path)
        prices = prices.loc[~parsed_index.isna()].copy()
        parsed_index = parsed_index[~parsed_index.isna()]
    if prices.empty:
        return pd.DataFrame()
    prices.index = parsed_index
    prices.columns = [str(column) for column in prices.columns]
    metadata_path = _cache_metadata_path(cache_path)
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(metadata, dict):
                prices.attrs["synthetic_data"] = bool(metadata.get("synthetic_data", False))
                notes = metadata.get("data_notes", [])
                if isinstance(notes, list):
                    prices.attrs["data_notes"] = [str(note) for note in notes]
                prices.attrs["data_source"] = str(metadata.get("data_source", "cache"))
                prices.attrs["cache_status"] = str(metadata.get("cache_status", "loaded"))
                prices.attrs["latest_price_date"] = str(metadata.get("latest_price_date", ""))
                prices.attrs["price_basis"] = str(metadata.get("price_basis", "adjusted_close_proxy"))
                prices.attrs["yfinance_available"] = bool(yf is not None)
                tickers_loaded = metadata.get("tickers_loaded", [])
                tickers_failed = metadata.get("tickers_failed", [])
                prices.attrs["tickers_loaded"] = [str(ticker) for ticker in tickers_loaded] if isinstance(tickers_loaded, list) else []
                prices.attrs["tickers_failed"] = [str(ticker) for ticker in tickers_failed] if isinstance(tickers_failed, list) else []
                prices.attrs["used_cache_fallback"] = bool(metadata.get("used_cache_fallback", False))
                prices.attrs["live_data_error"] = str(metadata.get("live_data_error", ""))
                prices.attrs["cache_written_at_utc"] = str(metadata.get("cache_written_at_utc", ""))
        except Exception as exc:
            LOGGER.warning("Could not read price-cache metadata %s: %s", metadata_path, exc)
    return prices.sort_index()


def _download_prices(
    tickers: Sequence[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices from yfinance."""

    if yf is None:
        raise ImportError("yfinance is not installed in the current Python environment.")

    raw = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    return _extract_close_prices(raw=raw, tickers=tickers)


def _merge_price_sources(cached_prices: pd.DataFrame, downloaded_prices: pd.DataFrame) -> pd.DataFrame:
    """Combine cache and downloaded prices with downloaded observations taking precedence."""

    if bool(cached_prices.attrs.get("synthetic_data", False)) and not downloaded_prices.empty:
        combined = downloaded_prices.copy()
        combined.attrs["synthetic_data"] = False
        combined.attrs["data_notes"] = [
            "Replaced prior synthetic fallback cache with freshly downloaded market data."
        ]
        combined.index = pd.to_datetime(combined.index)
        combined = combined.sort_index()
        combined.columns = [str(column) for column in combined.columns]
        return combined

    if cached_prices.empty:
        combined = downloaded_prices.copy()
    elif downloaded_prices.empty:
        combined = cached_prices.copy()
    else:
        combined = pd.concat([cached_prices, downloaded_prices], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]

    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index()
    combined.columns = [str(column) for column in combined.columns]
    return combined


def _cache_age_hours(path: str | Path) -> float | None:
    """Return the cache age in hours if a cache file exists."""

    cache_path = Path(path)
    if not cache_path.exists():
        return None
    try:
        modified_at = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        return max((datetime.now(timezone.utc) - modified_at).total_seconds() / 3600.0, 0.0)
    except Exception:
        return None


def _filter_usable_tickers(
    prices: pd.DataFrame,
    expected_tickers: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Forward-fill small gaps and drop only tickers with no usable latest history.

    We keep tickers that have leading missing values as long as they eventually
    provide a valid price history. Tickers with too little usable history for
    the configured windows are removed, instead of truncating the full panel to
    the shortest ETF history.
    """

    candidate = prices.copy()
    candidate = candidate.reindex(columns=[str(ticker) for ticker in expected_tickers])
    candidate = candidate.sort_index().ffill(limit=3)

    removed_tickers: list[str] = []
    usable_tickers: list[str] = []
    for ticker in expected_tickers:
        if ticker not in candidate.columns:
            removed_tickers.append(str(ticker))
            continue

        series = candidate[str(ticker)]
        latest_valid = series.dropna()
        if latest_valid.empty:
            removed_tickers.append(str(ticker))
            continue
        latest_panel_price = series.iloc[-1]
        if not np.isfinite(float(latest_panel_price)) or float(latest_panel_price) <= 0.0:
            removed_tickers.append(str(ticker))
            continue
        if int(series.notna().sum()) < int(MIN_REQUIRED_OBSERVATIONS):
            removed_tickers.append(str(ticker))
            continue
        usable_tickers.append(str(ticker))

    filtered = candidate.reindex(columns=usable_tickers)
    if not filtered.empty:
        filtered = filtered.dropna(how="all")
    return filtered, removed_tickers


def _generate_synthetic_price_history(
    tickers: Sequence[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Generate deterministic synthetic prices when no real data source is available."""

    end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.today().normalize()
    fallback_periods = max(MIN_REQUIRED_OBSERVATIONS + 120, 900)
    index = pd.bdate_range(end=end, periods=fallback_periods)

    rng = np.random.default_rng(42)
    common_factor = rng.normal(loc=0.0002, scale=0.0065, size=len(index))

    prices = pd.DataFrame(index=index)
    for idx, ticker in enumerate(tickers):
        drift = 0.0001 + 0.00002 * (idx % 6)
        volatility = 0.0045 + 0.0005 * (idx % 7)
        idiosyncratic = rng.normal(loc=0.0, scale=volatility, size=len(index))
        daily_returns = np.clip(drift + 0.35 * common_factor + idiosyncratic, -0.12, 0.12)
        start_price = 50.0 + 7.5 * (idx % 9)
        prices[str(ticker)] = start_price * np.cumprod(1.0 + daily_returns)

    prices = prices.astype(float)
    prices.attrs["data_notes"] = [
        "Using deterministic synthetic fallback prices because no usable real-market data "
        "could be downloaded and no valid local cache was available. "
        "These results are for pipeline validation only."
    ]
    prices.attrs["synthetic_data"] = True
    return prices


def _dedupe_notes(notes: Sequence[str]) -> list[str]:
    """Return notes in stable order without duplicates."""

    return list(dict.fromkeys(str(note) for note in notes if str(note).strip()))


def _attach_price_metadata(
    prices: pd.DataFrame,
    *,
    removed_tickers: Sequence[str] | None = None,
    available_tickers: Sequence[str] | None = None,
    notes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Attach stable metadata used by logging, reporting and downstream orchestration."""

    enriched = prices.copy()
    existing_notes = list(enriched.attrs.get("data_notes", []))
    enriched.attrs["data_notes"] = _dedupe_notes([*existing_notes, *(notes or [])])
    enriched.attrs["removed_tickers"] = list(
        dict.fromkeys(str(ticker) for ticker in (removed_tickers or []) if str(ticker).strip())
    )
    enriched.attrs["available_tickers"] = list(
        dict.fromkeys(str(ticker) for ticker in (available_tickers or enriched.columns.tolist()))
    )
    if not enriched.empty:
        enriched.attrs["latest_price_date"] = str(pd.Timestamp(enriched.index.max()).date())
    else:
        enriched.attrs["latest_price_date"] = ""
    enriched.attrs["price_basis"] = str(enriched.attrs.get("price_basis", "adjusted_close_proxy"))
    return enriched


def _assess_investable_universe(
    available_tickers: Sequence[str],
) -> tuple[bool, str]:
    """Return whether the available investable tickers can support the constrained strategy."""

    registry_tickers = set(get_enabled_tickers())
    investable_tickers = [ticker for ticker in available_tickers if ticker in registry_tickers]

    if len(investable_tickers) < MIN_REQUIRED_ASSETS:
        return (
            False,
            f"only {len(investable_tickers)} investable assets remained after cleaning",
        )

    try:
        from config import build_params
        from optimizer import build_feasible_initial_weights

        build_feasible_initial_weights(
            tickers=investable_tickers,
            params=build_params(tickers=investable_tickers),
        )
    except Exception as exc:
        return (
            False,
            f"remaining investable universe is infeasible under allocation limits: {exc}",
        )

    return True, ""


def load_price_data(
    tickers: Sequence[str],
    start_date: str,
    end_date: str | None = None,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
    prefer_live: bool = True,
    allow_cache_fallback: bool = True,
    max_cache_age_hours: int = 24,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load adjusted close prices with cache support and robust single-ticker failure handling."""

    expected_tickers = [str(ticker) for ticker in dict.fromkeys(tickers)]
    if not expected_tickers:
        raise ValueError("No tickers were provided to load_price_data().")

    effective_cache_path = Path(cache_path or PRICE_CACHE_PATH)
    cached_prices = pd.DataFrame()
    cache_age_hours = _cache_age_hours(effective_cache_path)
    if use_cache and (allow_cache_fallback or not prefer_live or not force_refresh):
        cached_prices = load_price_cache(effective_cache_path)
        if not cached_prices.empty:
            LOGGER.info(
                "Loaded %s cached price rows from %s.",
                len(cached_prices),
                effective_cache_path,
            )
            cached_prices.attrs["cache_age_hours"] = cache_age_hours

    downloaded_prices = pd.DataFrame()
    download_failed = False
    should_try_live = bool(force_refresh or prefer_live)
    if not should_try_live and use_cache and not cached_prices.empty:
        if cache_age_hours is not None and cache_age_hours <= float(max_cache_age_hours):
            prices = cached_prices.copy()
            prices = prices.reindex(columns=expected_tickers)
            prices, removed_tickers = _filter_usable_tickers(prices=prices, expected_tickers=expected_tickers)
            available_tickers = list(prices.columns)
            prices = validate_price_data(prices=prices, min_rows=MIN_REQUIRED_OBSERVATIONS)
            prices = _attach_price_metadata(
                prices,
                removed_tickers=removed_tickers,
                available_tickers=available_tickers,
                notes=(
                    [
                        "Temporarily dropped tickers in this run due to missing or unusable latest cached price: "
                        + ", ".join(removed_tickers)
                    ]
                    if removed_tickers
                    else []
                ),
            )
            prices.attrs["data_source"] = "cache"
            prices.attrs["cache_status"] = "preferred_cache"
            prices.attrs["used_cache_fallback"] = False
            prices.attrs["yfinance_available"] = yf is not None
            prices.attrs["tickers_loaded"] = list(available_tickers)
            prices.attrs["tickers_failed"] = list(removed_tickers)
            prices.attrs["live_data_error"] = ""
            return prices
        should_try_live = True

    live_error: Exception | None = None
    if should_try_live:
        try:
            downloaded_prices = _download_prices(
                tickers=expected_tickers,
                start_date=start_date,
                end_date=end_date,
            )
            if not downloaded_prices.empty:
                LOGGER.info("Downloaded fresh prices for %s requested tickers.", len(expected_tickers))
        except Exception as exc:
            download_failed = True
            live_error = exc
            downloaded_prices = pd.DataFrame()

    using_cache_fallback = False
    if downloaded_prices.empty:
        download_failed = should_try_live
        if use_cache and allow_cache_fallback and not cached_prices.empty:
            using_cache_fallback = True
            LOGGER.warning(
                "Price download failed. Continuing with cache fallback. Reason: %s",
                live_error or "live download returned an empty dataset",
            )
            prices = cached_prices.copy()
            prices.attrs["live_data_error"] = str(live_error or "live download returned an empty dataset")
        else:
            if live_error is None and should_try_live:
                live_error = ValueError("yfinance returned an empty dataset.")
            raise ValueError(
                "Price download failed and no usable cache fallback is available. "
                f"Unable to load market data from yfinance or local cache. Reason: {live_error}"
            )
    else:
        prices = downloaded_prices.copy()

    prices.attrs["data_notes"] = []
    prices.attrs["removed_tickers"] = []
    prices.attrs["available_tickers"] = []
    prices = prices.reindex(columns=expected_tickers)
    prices = prices.loc[pd.to_datetime(start_date) :]
    if end_date is not None:
        prices = prices.loc[: pd.to_datetime(end_date)]

    prices, removed_tickers = _filter_usable_tickers(prices=prices, expected_tickers=expected_tickers)
    available_tickers = list(prices.columns)
    if removed_tickers:
        LOGGER.warning(
            "No usable close-price history was available for these tickers and they will be dropped: %s",
            ", ".join(removed_tickers),
        )

    investable_universe_ok, infeasibility_reason = _assess_investable_universe(available_tickers)
    if not investable_universe_ok:
        LOGGER.warning(
            "Available market data is insufficient for a fully feasible constrained run (%s). "
            "Continuing with the available price history and leaving feasibility handling to the main pipeline.",
            infeasibility_reason,
        )

    prices = validate_price_data(prices=prices, min_rows=MIN_REQUIRED_OBSERVATIONS)
    notes: list[str] = []
    if removed_tickers:
        notes.append(
            "Temporarily dropped tickers in this run due to missing or unusable price history: "
            + ", ".join(removed_tickers)
        )
    if using_cache_fallback:
        notes.append("Used cached prices because live download failed during this run.")
    prices = _attach_price_metadata(
        prices,
        removed_tickers=removed_tickers,
        available_tickers=available_tickers,
        notes=notes,
    )
    prices.attrs["data_source"] = "cache_fallback" if using_cache_fallback else "yfinance"
    prices.attrs["cache_status"] = "used_after_live_failure" if using_cache_fallback else "refreshed"
    prices.attrs["synthetic_data"] = bool(prices.attrs.get("synthetic_data", False)) if using_cache_fallback else False
    prices.attrs["yfinance_available"] = yf is not None
    prices.attrs["tickers_loaded"] = list(available_tickers)
    prices.attrs["tickers_failed"] = list(removed_tickers)
    prices.attrs["used_cache_fallback"] = using_cache_fallback
    prices.attrs["cache_age_hours"] = cache_age_hours
    prices.attrs["live_data_error"] = str(live_error) if using_cache_fallback and live_error is not None else ""
    prices.attrs["price_basis"] = "adjusted_close_proxy"
    if using_cache_fallback:
        return prices

    save_price_cache(prices=prices, path=effective_cache_path)
    return prices


def check_data_freshness(prices: pd.DataFrame, max_staleness_days: int = 5) -> dict[str, object]:
    """Assess whether the latest available price date is recent enough."""

    if prices.empty:
        return {
            "latest_price_date": None,
            "data_freshness_ok": False,
            "staleness_days": None,
            "warning": "Price history is empty.",
        }

    latest_price_date = pd.Timestamp(prices.index.max()).normalize()
    today = pd.Timestamp(datetime.now(BERLIN_TZ).date())
    staleness_days = max(int((today - latest_price_date).days), 0)
    freshness_ok = staleness_days <= int(max_staleness_days)
    warning = ""
    if not freshness_ok:
        warning = (
            f"Latest market data is stale by {staleness_days} days; "
            f"maximum allowed staleness is {int(max_staleness_days)} days."
        )
    return {
        "latest_price_date": str(latest_price_date.date()),
        "data_freshness_ok": freshness_ok,
        "staleness_days": staleness_days,
        "warning": warning,
    }


def build_run_data_context(
    prices: pd.DataFrame,
    freshness: dict[str, object] | None,
    market_gate: dict[str, object] | None = None,
    *,
    calendar_path: str | Path = DEFAULT_PROJECT_CALENDAR_PATH,
    run_context: str = "daily_bot",
) -> RunDataContext:
    """Build the canonical shared data context for one run."""

    attrs = dict(getattr(prices, "attrs", {}))
    freshness = dict(freshness or {})
    market_gate = dict(market_gate or {})
    expected_latest_trading_day = get_expected_latest_trading_day(
        now=pd.Timestamp(str(market_gate.get("date"))) if market_gate.get("date") else None,
        calendar_path=calendar_path,
    )
    return RunDataContext(
        data_source=str(attrs.get("data_source", "unknown")),
        cache_status=str(attrs.get("cache_status", "unknown")),
        synthetic_data=bool(attrs.get("synthetic_data", False)),
        used_cache_fallback=bool(attrs.get("used_cache_fallback", False)),
        latest_price_date=str(freshness.get("latest_price_date")) if freshness.get("latest_price_date") else None,
        expected_latest_trading_day=expected_latest_trading_day,
        staleness_days=int(freshness["staleness_days"]) if freshness.get("staleness_days") is not None else None,
        data_freshness_ok=bool(freshness.get("data_freshness_ok", False)),
        live_data_error=str(attrs.get("live_data_error", "") or ""),
        tickers_loaded=[str(item) for item in attrs.get("tickers_loaded", prices.columns.tolist())],
        tickers_failed=[str(item) for item in attrs.get("tickers_failed", [])],
        price_basis=str(attrs.get("price_basis", "adjusted_close_proxy")),
        yfinance_available=bool(attrs.get("yfinance_available", yf is not None)),
        warning=str(freshness.get("warning", "") or ""),
        project_calendar_path=str(calendar_path),
        calendar_status=str(market_gate.get("reason", "unknown")),
        current_date_berlin=str(market_gate.get("date")) if market_gate.get("date") else None,
        current_time_berlin=str(market_gate.get("current_time_berlin")) if market_gate.get("current_time_berlin") else None,
        is_project_trading_day=bool(market_gate.get("is_trading_day", False)),
        allowed_start_berlin=str(market_gate.get("allowed_start_berlin")) if market_gate.get("allowed_start_berlin") else None,
        allowed_end_berlin=str(market_gate.get("allowed_end_berlin")) if market_gate.get("allowed_end_berlin") else None,
        within_allowed_window=bool(market_gate.get("within_allowed_window", False)),
        execution_allowed_by_calendar=bool(market_gate.get("execution_allowed", False)),
        run_context=str(run_context),
    )


def write_data_freshness_report(
    prices: pd.DataFrame,
    freshness: dict[str, object],
    output_path: str | Path,
    market_gate: dict[str, object] | None = None,
    data_context: RunDataContext | dict[str, object] | None = None,
) -> Path:
    """Write a compact current-data report for operators."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    context = (
        data_context.as_dict()
        if isinstance(data_context, RunDataContext)
        else dict(data_context)
        if data_context is not None
        else build_run_data_context(
            prices=prices,
            freshness=freshness,
            market_gate=market_gate,
            calendar_path=PROJECT_CALENDAR_PATH,
        ).as_dict()
    )
    tickers_loaded = context.get("tickers_loaded", prices.attrs.get("tickers_loaded", prices.columns.tolist()))
    tickers_failed = context.get("tickers_failed", prices.attrs.get("tickers_failed", []))
    lines = [
        f"run_context: {context.get('run_context', 'daily_bot')}",
        f"data_source: {context.get('data_source', 'unknown')}",
        f"cache_status: {context.get('cache_status', 'unknown')}",
        f"synthetic_data: {bool(context.get('synthetic_data', False))}",
        f"latest_price_date: {context.get('latest_price_date')}",
        f"expected_latest_trading_day: {context.get('expected_latest_trading_day') or 'none'}",
        f"staleness_days: {context.get('staleness_days')}",
        f"data_freshness_ok: {bool(context.get('data_freshness_ok', False))}",
        f"price_basis: {context.get('price_basis', 'adjusted_close_proxy')}",
        f"yfinance_available: {bool(context.get('yfinance_available', yf is not None))}",
        f"tickers_loaded: {', '.join(map(str, tickers_loaded)) if tickers_loaded else 'none'}",
        f"tickers_failed: {', '.join(map(str, tickers_failed)) if tickers_failed else 'none'}",
        f"used_cache_fallback: {bool(context.get('used_cache_fallback', False))}",
        f"live_data_error: {context.get('live_data_error', '') or 'none'}",
        f"warning: {context.get('warning', '') or 'none'}",
        f"calendar_status: {context.get('calendar_status', 'unknown')}",
    ]
    if market_gate is not None or context.get("current_date_berlin") is not None:
        lines.extend(
            [
                f"project_calendar_path: {context.get('project_calendar_path', PROJECT_CALENDAR_PATH)}",
                f"current_date_berlin: {context.get('current_date_berlin', 'n/a')}",
                f"current_time_berlin: {context.get('current_time_berlin', 'n/a')}",
                f"is_project_trading_day: {bool(context.get('is_project_trading_day', False))}",
                f"allowed_start_berlin: {context.get('allowed_start_berlin') or 'none'}",
                f"allowed_end_berlin: {context.get('allowed_end_berlin') or 'none'}",
                f"within_allowed_window: {bool(context.get('within_allowed_window', False))}",
                f"execution_allowed_by_calendar: {bool(context.get('execution_allowed_by_calendar', False))}",
                f"calendar_reason: {context.get('calendar_status', 'n/a')}",
            ]
        )
    _atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def download_market_data(
    universe: Sequence[AssetDefinition],
    data_config: object,
) -> MarketDataBundle:
    """Compatibility wrapper that loads prices and computes daily returns."""

    tickers = symbols(universe)
    prices = load_price_data(
        tickers=tickers,
        start_date=str(getattr(data_config, "start_date")),
        end_date=getattr(data_config, "end_date", None),
        cache_path=getattr(data_config, "cache_path", PRICE_CACHE_PATH),
        use_cache=bool(getattr(data_config, "use_cache", True)),
    )
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return MarketDataBundle(prices=prices, returns=returns, volumes=None)


def slice_history(
    bundle: MarketDataBundle,
    end_date: pd.Timestamp,
    lookback_days: int | None = None,
) -> MarketDataBundle:
    """Return a historical slice ending at the given trading day."""

    prices = bundle.prices.loc[:end_date]
    returns = bundle.returns.loc[:end_date]

    if lookback_days is not None:
        prices = prices.tail(lookback_days)
        returns = returns.tail(lookback_days)

    volumes = bundle.volumes.loc[:end_date] if bundle.volumes is not None else None
    if lookback_days is not None and volumes is not None:
        volumes = volumes.tail(lookback_days)

    return MarketDataBundle(prices=prices, returns=returns, volumes=volumes)
