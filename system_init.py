"""System initialization helpers for the optimizer and daily bot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data import load_price_cache, load_price_data
from database import init_db
from paper_broker_stub import initialize_paper_account


ENV_EXAMPLE_TEMPLATE = """DRY_RUN=true
ENABLE_EXTERNAL_BROKER=false
ENABLE_LOCAL_PAPER_TRADING=false
PAPER_INITIAL_CASH=10000
ENABLE_INVESTOPEDIA_SIMULATOR=false
INVESTOPEDIA_BASE_URL=https://www.investopedia.com/simulator/portfolio
INVESTOPEDIA_USERNAME=your_username
INVESTOPEDIA_PASSWORD=your_password
INVESTOPEDIA_GAME_ID=your_game_id_or_portfolio_id
INVESTOPEDIA_DRY_RUN=true
INVESTOPEDIA_USE_EXISTING_LIBRARY=false
ENABLE_EMAIL_NOTIFICATIONS=false
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
EMAIL_FROM=your_email@gmail.com
EMAIL_TO=target_email@example.com
EMAIL_SUBJECT_PREFIX=[Portfolio Optimizer]
SEND_WEEKLY_SUMMARY=true
SEND_DAILY_HOLD_WAIT_EMAILS=false
FRED_API_KEY=
FMP_API_KEY=
"""


def ensure_directories(params: dict[str, Any]) -> list[str]:
    """Ensure the core project directories exist and return newly created ones."""

    created: list[str] = []
    for key in ("output_dir", "data_dir"):
        path = Path(params[key])
        if not path.exists():
            created.append(str(path))
        path.mkdir(parents=True, exist_ok=True)

    base_dir = Path(__file__).resolve().parent
    for path in (base_dir / "notebooks", base_dir / "logs"):
        if not path.exists():
            created.append(str(path))
        path.mkdir(parents=True, exist_ok=True)
    return created


def ensure_env_example() -> bool:
    """Ensure that `.env.example` exists."""

    path = Path(__file__).resolve().parent / ".env.example"
    if not path.exists():
        path.write_text(ENV_EXAMPLE_TEMPLATE, encoding="utf-8")
    return path.exists()


def initialize_database(params: dict[str, Any]) -> bool:
    """Initialize SQLite schema."""

    init_db(params["db_path"])
    return True


def initialize_price_cache_if_missing(params: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    """Ensure a price cache exists when possible.

    Returns `(ready, warnings, errors)`.
    """

    warnings: list[str] = []
    errors: list[str] = []
    cache_path = Path(params["price_cache_path"])
    cached = load_price_cache(cache_path)
    if not cached.empty:
        return True, warnings, errors

    try:
        prices = load_price_data(
            tickers=list(params["tickers"]),
            start_date=str(params["start_date"]),
            end_date=params.get("end_date"),
            cache_path=cache_path,
            use_cache=False,
        )
    except Exception as exc:
        errors.append(f"Price cache initialization failed because no cache exists and price loading failed: {exc}")
        return False, warnings, errors

    if bool(prices.attrs.get("synthetic_data", False)):
        warnings.append(
            "Price cache initialization used synthetic fallback data. This is acceptable for dry-run pipeline validation "
            "but not for any real execution path."
        )
        return False, warnings, errors
    return True, warnings, errors


def initialize_paper_account_if_enabled(params: dict[str, Any]) -> tuple[bool, list[str]]:
    """Initialize the local paper account if requested."""

    warnings: list[str] = []
    if not bool(params.get("enable_local_paper_trading", False)):
        return False, warnings
    try:
        initialize_paper_account(
            db_path=params["db_path"],
            initial_cash=float(params.get("paper_initial_cash", 10000.0)),
        )
        return True, warnings
    except Exception as exc:
        warnings.append(f"Local paper account could not be initialized: {exc}")
        return False, warnings


def run_system_initialization(params: dict[str, Any]) -> dict[str, Any]:
    """Run the full safe initialization flow."""

    warnings: list[str] = []
    errors: list[str] = []
    created_directories = ensure_directories(params)
    env_example_exists = ensure_env_example()

    db_initialized = False
    try:
        db_initialized = initialize_database(params)
    except Exception as exc:
        errors.append(f"Database initialization failed: {exc}")

    price_cache_ready, cache_warnings, cache_errors = initialize_price_cache_if_missing(params)
    warnings.extend(cache_warnings)
    errors.extend(cache_errors)

    paper_account_ready, paper_warnings = initialize_paper_account_if_enabled(params)
    warnings.extend(paper_warnings)

    return {
        "ok": db_initialized and env_example_exists and not errors,
        "warnings": warnings,
        "errors": errors,
        "created_directories": created_directories,
        "db_initialized": db_initialized,
        "env_example_exists": env_example_exists,
        "price_cache_ready": price_cache_ready,
        "paper_account_ready": paper_account_ready,
    }
