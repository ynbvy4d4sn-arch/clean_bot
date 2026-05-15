"""Project-wide constants and compatibility configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

from asset_universe import (
    CRYPTO_MAX_NORMAL,
    CRYPTO_MAX_RISK_OFF,
    DEFENSIVE_GROUPS,
    EQUITY_LIKE_GROUPS,
    MAX_EQUITY_LIKE_TOTAL_NORMAL,
    MAX_EQUITY_LIKE_TOTAL_RISK_OFF,
    MIN_DEFENSIVE_WEIGHT_NORMAL,
    MIN_DEFENSIVE_WEIGHT_RISK_OFF,
    get_asset_max_weights,
    get_cash_ticker,
    get_enabled_tickers,
    get_group_limits,
    get_group_map,
    validate_asset_universe,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional before dependencies are installed
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        """Fallback no-op implementation when python-dotenv is unavailable."""

        return False


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DB_PATH = DATA_DIR / "optimizer.sqlite"
PRICE_CACHE_PATH = DATA_DIR / "prices_cache.csv"
load_dotenv(BASE_DIR / ".env", override=False)


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean flag from the environment with a safe default."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment with a safe default."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment with a safe default."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default

validate_asset_universe()
TICKERS = get_enabled_tickers()
CASH_TICKER = get_cash_ticker()

START_DATE = "2018-01-01"
END_DATE = None

HORIZON_DAYS = _env_int("HORIZON_DAYS", _env_int("SOLVER_HORIZON_DAYS", 63))
MOMENTUM_SHORT = 63
MOMENTUM_LONG = 126
COV_WINDOW = 126

KAPPA = 0.20
FORECAST_CLIP_LOWER = -0.20
FORECAST_CLIP_UPPER = 0.20

COV_SHRINK_ALPHA = 0.75
COV_JITTER = 1e-8
RISK_FREE_RATE_ANNUAL = _env_float("RISK_FREE_RATE_ANNUAL", _env_float("SOLVER_RISK_FREE_RATE_ANNUAL", 0.02))

RISK_AVERSION = 3.0
TURNOVER_PENALTY = 0.05
CONCENTRATION_PENALTY = 0.02

SOLVER_CONFIG = {
    "objective": os.getenv("SOLVER_OBJECTIVE", "scenario_weighted_rf_sharpe").strip().lower(),
    "horizon_days": HORIZON_DAYS,
    "risk_free_rate_annual": RISK_FREE_RATE_ANNUAL,
    "eps_variance": _env_float("SOLVER_EPS_VARIANCE", 1.0e-10),
    "lambda_turnover": _env_float("SOLVER_LAMBDA_TURNOVER", 0.03),
    "lambda_concentration": _env_float("SOLVER_LAMBDA_CONCENTRATION", 0.01),
    "lambda_downside": _env_float("SOLVER_LAMBDA_DOWNSIDE", 0.15),
    "max_turnover": _env_float("SOLVER_MAX_TURNOVER", 0.75),
    "optimizer_method": os.getenv("SOLVER_OPTIMIZER_METHOD", "SLSQP").strip().upper(),
    "optimizer_ftol": _env_float("SOLVER_OPTIMIZER_FTOL", 1.0e-9),
    "optimizer_maxiter": _env_int("SOLVER_OPTIMIZER_MAXITER", 1000),
    "allow_short": _env_flag("SOLVER_ALLOW_SHORT", False),
    "long_only": _env_flag("SOLVER_LONG_ONLY", True),
    "use_scenario_covariance": _env_flag("SOLVER_USE_SCENARIO_COVARIANCE", True),
    "use_scenario_probabilities": _env_flag("SOLVER_USE_SCENARIO_PROBABILITIES", True),
    "use_rf_adjusted_sharpe": _env_flag("SOLVER_USE_RF_ADJUSTED_SHARPE", True),
}

REPORTING_CONFIG = {
    "write_scenario_solver_report": _env_flag("REPORTING_WRITE_SCENARIO_SOLVER_REPORT", True),
    "write_pairwise_relationships": _env_flag("REPORTING_WRITE_PAIRWISE_RELATIONSHIPS", True),
    "write_scenario_matrices": _env_flag("REPORTING_WRITE_SCENARIO_MATRICES", True),
}

OPTIMIZATION_OBJECTIVE = os.getenv("OPTIMIZATION_OBJECTIVE", str(SOLVER_CONFIG["objective"])).strip().lower()
SHARPE_EPSILON = _env_float("SHARPE_EPSILON", float(SOLVER_CONFIG["eps_variance"]))
SORTINO_TARGET = os.getenv("SORTINO_TARGET", "defensive_cash").strip().lower()
LAMBDA_CVAR_SHARPE = _env_float("LAMBDA_CVAR_SHARPE", 0.25)
LAMBDA_TURNOVER_SHARPE = _env_float("LAMBDA_TURNOVER_SHARPE", float(SOLVER_CONFIG["lambda_turnover"]))
LAMBDA_COST_SHARPE = _env_float("LAMBDA_COST_SHARPE", 1.0)
LAMBDA_CONCENTRATION_SHARPE = _env_float("LAMBDA_CONCENTRATION_SHARPE", float(SOLVER_CONFIG["lambda_concentration"]))
DIRECT_SCENARIO_OPTIMIZER_ENABLED = _env_flag("DIRECT_SCENARIO_OPTIMIZER_ENABLED", True)
DIRECT_SCENARIO_LAMBDA_TURNOVER = _env_float("DIRECT_SCENARIO_LAMBDA_TURNOVER", float(SOLVER_CONFIG["lambda_turnover"]))
DIRECT_SCENARIO_LAMBDA_CONCENTRATION = _env_float("DIRECT_SCENARIO_LAMBDA_CONCENTRATION", float(SOLVER_CONFIG["lambda_concentration"]))
DIRECT_SCENARIO_LAMBDA_DOWNSIDE = _env_float("DIRECT_SCENARIO_LAMBDA_DOWNSIDE", float(SOLVER_CONFIG["lambda_downside"]))
DIRECT_SCENARIO_OPTIMIZER_MAX_STARTS = _env_int("DIRECT_SCENARIO_OPTIMIZER_MAX_STARTS", 5)
DIRECT_SCENARIO_OPTIMIZER_METHOD = os.getenv("DIRECT_SCENARIO_OPTIMIZER_METHOD", str(SOLVER_CONFIG["optimizer_method"])).strip().upper()
DIRECT_SCENARIO_OPTIMIZER_MAXITER = _env_int("DIRECT_SCENARIO_OPTIMIZER_MAXITER", int(SOLVER_CONFIG["optimizer_maxiter"]))
DIRECT_SCENARIO_OPTIMIZER_FTOL = _env_float("DIRECT_SCENARIO_OPTIMIZER_FTOL", float(SOLVER_CONFIG["optimizer_ftol"]))
DIRECT_SCENARIO_RF_MODE = os.getenv("DIRECT_SCENARIO_RF_MODE", "cash_ticker").strip().lower()
SCENARIO_COVARIANCE_LOOKBACK = _env_int("SCENARIO_COVARIANCE_LOOKBACK", COV_WINDOW)
SCENARIO_COVARIANCE_SHRINK_ALPHA = _env_float("SCENARIO_COVARIANCE_SHRINK_ALPHA", COV_SHRINK_ALPHA)
SCENARIO_PROBABILITIES = {
    "base": _env_float("SCENARIO_PROB_BASE", 0.35),
    "risk_on": _env_float("SCENARIO_PROB_RISK_ON", 0.15),
    "risk_off": _env_float("SCENARIO_PROB_RISK_OFF", 0.15),
    "rates_up": _env_float("SCENARIO_PROB_RATES_UP", 0.10),
    "rates_down": _env_float("SCENARIO_PROB_RATES_DOWN", 0.07),
    "commodity_up": _env_float("SCENARIO_PROB_COMMODITY_UP", 0.08),
    "equity_stress": _env_float("SCENARIO_PROB_EQUITY_STRESS", 0.10),
}
SCENARIO_WEIGHTED_PROBABILITIES = {
    "bull_momentum": _env_float("SCENARIO_WEIGHTED_PROB_BULL_MOMENTUM", 0.25),
    "soft_landing": _env_float("SCENARIO_WEIGHTED_PROB_SOFT_LANDING", 0.25),
    "sideways_choppy": _env_float("SCENARIO_WEIGHTED_PROB_SIDEWAYS_CHOPPY", 0.20),
    "inflation_shock": _env_float("SCENARIO_WEIGHTED_PROB_INFLATION_SHOCK", 0.15),
    "growth_selloff": _env_float("SCENARIO_WEIGHTED_PROB_GROWTH_SELLOFF", 0.10),
    "liquidity_stress": _env_float("SCENARIO_WEIGHTED_PROB_LIQUIDITY_STRESS", 0.05),
}
SCENARIO_RISK_ON_EQUITY_SHOCK = _env_float("SCENARIO_RISK_ON_EQUITY_SHOCK", 0.012)
SCENARIO_RISK_OFF_EQUITY_SHOCK = _env_float("SCENARIO_RISK_OFF_EQUITY_SHOCK", -0.018)
SCENARIO_RATES_UP_DURATION_SHOCK = _env_float("SCENARIO_RATES_UP_DURATION_SHOCK", -0.010)
SCENARIO_RATES_DOWN_DURATION_SHOCK = _env_float("SCENARIO_RATES_DOWN_DURATION_SHOCK", 0.008)
SCENARIO_COMMODITY_UP_SHOCK = _env_float("SCENARIO_COMMODITY_UP_SHOCK", 0.018)
SCENARIO_EQUITY_STRESS_SHOCK = _env_float("SCENARIO_EQUITY_STRESS_SHOCK", -0.035)
SCENARIO_DEFENSIVE_CARRY_SHOCK = _env_float("SCENARIO_DEFENSIVE_CARRY_SHOCK", 0.001)

DEFAULT_COMMISSION_PER_TRADE = _env_float("DEFAULT_COMMISSION_PER_TRADE", 0.0)
DEFAULT_BPS_PER_TURNOVER = _env_float("DEFAULT_BPS_PER_TURNOVER", 5.0)
DEFAULT_SLIPPAGE_BPS = _env_float("DEFAULT_SLIPPAGE_BPS", 3.0)
DEFAULT_SPREAD_BPS = _env_float("DEFAULT_SPREAD_BPS", 2.0)
DEFAULT_MARKET_IMPACT_BPS = _env_float("DEFAULT_MARKET_IMPACT_BPS", 0.0)
USE_LIVE_BID_ASK_IF_AVAILABLE = _env_flag("USE_LIVE_BID_ASK_IF_AVAILABLE", True)
COST_RATE = DEFAULT_BPS_PER_TURNOVER / 10000.0
BASE_BUFFER = 0.0005
VOL_BUFFER_MULTIPLIER = 0.05

MIN_TURNOVER_TO_TRADE = 0.05
PARTIAL_THRESHOLD = 0.25
PARTIAL_FRACTION = 0.50
SCENARIO_EXECUTION_FRACTION = _env_float("SCENARIO_EXECUTION_FRACTION", _env_float("EXECUTION_FRACTION", 1.0))
STRONG_SIGNAL_THRESHOLD = 0.005

MAX_TURNOVER = _env_float("MAX_TURNOVER", float(SOLVER_CONFIG["max_turnover"]))

ENABLE_ACTIVE_PREVIEW = _env_flag("ENABLE_ACTIVE_PREVIEW", True)
ACTIVE_PREVIEW_TRADE_NOW_HURDLE = _env_float("ACTIVE_PREVIEW_TRADE_NOW_HURDLE", 0.00075)
ACTIVE_PREVIEW_EXECUTION_BUFFER = _env_float("ACTIVE_PREVIEW_EXECUTION_BUFFER", 0.00035)
ACTIVE_PREVIEW_MODEL_UNCERTAINTY_MULTIPLIER = _env_float("ACTIVE_PREVIEW_MODEL_UNCERTAINTY_MULTIPLIER", 0.50)
ACTIVE_PREVIEW_DELTA_VS_CASH_MIN = _env_float("ACTIVE_PREVIEW_DELTA_VS_CASH_MIN", 0.00025)
ACTIVE_PREVIEW_P_CURRENT_MIN = _env_float("ACTIVE_PREVIEW_P_CURRENT_MIN", 0.52)
ACTIVE_PREVIEW_P_CASH_MIN = _env_float("ACTIVE_PREVIEW_P_CASH_MIN", 0.51)
ACTIVE_PREVIEW_MAX_TURNOVER = _env_float("ACTIVE_PREVIEW_MAX_TURNOVER", 0.20)
ACTIVE_PREVIEW_MIN_ORDER_VALUE_USD = _env_float("ACTIVE_PREVIEW_MIN_ORDER_VALUE_USD", 10.0)
ACTIVE_PREVIEW_ALLOW_EXECUTION = _env_flag("ACTIVE_PREVIEW_ALLOW_EXECUTION", False)

DRAWDOWN_LIMIT = -0.12
CVAR_LIMIT = -0.10

USE_RISK_FILTER = True
ALLOW_DAILY_EMERGENCY_TRADES = True
DEFAULT_PORTFOLIO_VALUE = 10000.0
PORTFOLIO_NAV_USD = 100000.0
RANDOM_SEED = 42
MAX_DAILY_TURNOVER = 0.60
MAX_WEEKLY_TURNOVER = 1.00
MAX_MONTHLY_TURNOVER = 2.00
MAX_ORDERS_PER_DAY = 3
MIN_MINUTES_BETWEEN_ORDERS = 60
CURRENT_PORTFOLIO_SOURCE = os.getenv("CURRENT_PORTFOLIO_SOURCE", "csv").strip().lower()
CURRENT_PORTFOLIO_PATH = Path(os.getenv("CURRENT_PORTFOLIO_PATH", str(DATA_DIR / "current_portfolio.csv")))
DEFAULT_CURRENT_PORTFOLIO = os.getenv("DEFAULT_CURRENT_PORTFOLIO", "cash").strip().lower()
ALLOW_FRACTIONAL_SHARES = _env_flag("ALLOW_FRACTIONAL_SHARES", False)
MIN_ORDER_VALUE_USD = _env_float("MIN_ORDER_VALUE_USD", 10.0)
CASH_BUFFER_USD = _env_float("CASH_BUFFER_USD", 0.0)
MAX_GROSS_EXPOSURE = _env_float("MAX_GROSS_EXPOSURE", 1.0)
DRY_RUN = _env_flag("DRY_RUN", True)
ENABLE_LOCAL_PAPER_TRADING = _env_flag("ENABLE_LOCAL_PAPER_TRADING", False)
ENABLE_INVESTOPEDIA_SIMULATOR = _env_flag("ENABLE_INVESTOPEDIA_SIMULATOR", False)
ENABLE_EXTERNAL_BROKER = _env_flag("ENABLE_EXTERNAL_BROKER", False)
EMAIL_DRY_RUN = _env_flag("EMAIL_DRY_RUN", True)
EMAIL_SEND_ENABLED = _env_flag("EMAIL_SEND_ENABLED", False)
DAILY_BRIEFING_ONLY = _env_flag("DAILY_BRIEFING_ONLY", True)
MAX_EMAILS_PER_DAY = _env_int("MAX_EMAILS_PER_DAY", 1)
EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "brevo").strip().lower() or "brevo"
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "").strip()
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "").strip() or os.getenv("EMAIL_FROM", "").strip()
BREVO_API_KEY = os.getenv("BREVO_API_KEY", "").strip()
USER_CONFIRMED_EMAIL_PHASE = _env_flag("USER_CONFIRMED_EMAIL_PHASE", False)
PHASE = os.getenv("PHASE", "DAILY_REVIEW_PREVIEW").strip() or "DAILY_REVIEW_PREVIEW"
PAPER_INITIAL_CASH = _env_float("PAPER_INITIAL_CASH", DEFAULT_PORTFOLIO_VALUE)
INVESTOPEDIA_BASE_URL = os.getenv(
    "INVESTOPEDIA_BASE_URL",
    "https://www.investopedia.com/simulator/portfolio",
).strip()
INVESTOPEDIA_GAME_ID = os.getenv("INVESTOPEDIA_GAME_ID", "").strip()
INVESTOPEDIA_DRY_RUN = _env_flag("INVESTOPEDIA_DRY_RUN", True)
INVESTOPEDIA_USE_EXISTING_LIBRARY = _env_flag("INVESTOPEDIA_USE_EXISTING_LIBRARY", False)

ASSET_COST_OVERRIDES = {
    "SGOV": {"bucket": "low_cost_cash_bond_etf", "spread_bps": 1.0, "slippage_bps": 1.0, "market_impact_bps": 0.0, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "SHY": {"bucket": "low_cost_cash_bond_etf", "spread_bps": 1.0, "slippage_bps": 1.0, "market_impact_bps": 0.0, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "IEF": {"bucket": "low_cost_cash_bond_etf", "spread_bps": 1.0, "slippage_bps": 1.0, "market_impact_bps": 0.0, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "AGG": {"bucket": "low_cost_cash_bond_etf", "spread_bps": 1.0, "slippage_bps": 1.0, "market_impact_bps": 0.0, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "LQD": {"bucket": "low_cost_cash_bond_etf", "spread_bps": 1.0, "slippage_bps": 1.0, "market_impact_bps": 0.0, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "TIP": {"bucket": "low_cost_cash_bond_etf", "spread_bps": 1.0, "slippage_bps": 1.0, "market_impact_bps": 0.0, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "XLC": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "XLY": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "XLE": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "XLF": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "XLRE": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "XLB": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "XLI": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "XLK": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "XLP": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "XLU": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "XLV": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "VEA": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "VWO": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "SPHQ": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "SPLV": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "SPMO": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "RPV": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "SIZE": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "VBR": {"bucket": "normal_liquidity_etf", "spread_bps": DEFAULT_SPREAD_BPS, "slippage_bps": DEFAULT_SLIPPAGE_BPS, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "TLT": {"bucket": "low_cost_cash_bond_etf", "spread_bps": 1.0, "slippage_bps": 2.0, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "HYG": {"bucket": "normal_liquidity_credit_etf", "spread_bps": 3.0, "slippage_bps": 5.0, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "EMB": {"bucket": "normal_liquidity_credit_etf", "spread_bps": 3.0, "slippage_bps": 5.0, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "PDBC": {"bucket": "higher_cost_commodity_etf", "spread_bps": 5.0, "slippage_bps": 8.0, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "GLD": {"bucket": "higher_cost_commodity_etf", "spread_bps": 5.0, "slippage_bps": 8.0, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "SLV": {"bucket": "higher_cost_commodity_etf", "spread_bps": 5.0, "slippage_bps": 8.0, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "SH": {"bucket": "higher_cost_inverse_etf", "spread_bps": 5.0, "slippage_bps": 10.0, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
    "IBIT": {"bucket": "higher_cost_crypto_etf", "spread_bps": 8.0, "slippage_bps": 15.0, "market_impact_bps": DEFAULT_MARKET_IMPACT_BPS, "commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE},
}


def _build_defensive_weights(tickers: list[str], cash_ticker: str | None) -> dict[str, float]:
    """Build the default defensive allocation template for V1."""

    weights = {ticker: 0.0 for ticker in tickers}
    primary_defensive = cash_ticker if cash_ticker in weights else ("IEF" if "IEF" in weights else None)
    if primary_defensive is not None:
        weights[primary_defensive] += 0.60

    bond_ticker = "IEF" if "IEF" in tickers else primary_defensive
    if bond_ticker is not None:
        weights[bond_ticker] += 0.30

    hedge_ticker: str | None = None
    for candidate in ("GLD", "PDBC", "SLV", "TIP"):
        if candidate in tickers:
            hedge_ticker = candidate
            break

    if hedge_ticker is None:
        fallback_ticker = bond_ticker or primary_defensive
        if fallback_ticker is not None:
            weights[fallback_ticker] += 0.10
    else:
        weights[hedge_ticker] += 0.10

    return weights


def _setting_bool(
    settings: dict[str, object] | None,
    *keys: str,
    default: bool = False,
) -> bool:
    """Read a boolean setting from a mixed-case config/settings mapping."""

    if settings is None:
        return default
    for key in keys:
        if key in settings:
            value = settings[key]
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            normalized = str(value).strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
            return default
    return default


def _setting_text(
    settings: dict[str, object] | None,
    *keys: str,
    default: str = "",
) -> str:
    """Read a text setting from a mixed-case config/settings mapping."""

    if settings is None:
        return default
    for key in keys:
        if key in settings and settings[key] is not None:
            return str(settings[key]).strip()
    return default


def review_email_send_allowed(settings: dict[str, object] | None = None) -> tuple[bool, list[str]]:
    """Central phase gate for real Daily Review email sending."""

    gate = get_email_gate_status(settings)
    return bool(gate["real_email_send_allowed"]), list(gate["blockers"])


def get_email_gate_status(settings: dict[str, object] | None = None) -> dict[str, object]:
    """Return a structured gate status for real Daily Review email sending."""

    settings = settings or build_params()
    enable_email_notifications = _setting_bool(
        settings,
        "enable_email_notifications",
        "ENABLE_EMAIL_NOTIFICATIONS",
        default=False,
    )
    email_send_enabled = _setting_bool(
        settings,
        "email_send_enabled",
        "EMAIL_SEND_ENABLED",
        default=False,
    )
    email_dry_run = _setting_bool(
        settings,
        "email_dry_run",
        "EMAIL_DRY_RUN",
        default=True,
    )
    email_provider = _setting_text(
        settings,
        "email_provider",
        "EMAIL_PROVIDER",
        default="brevo",
    ).lower() or "brevo"
    recipient = _setting_text(
        settings,
        "email_recipient",
        "EMAIL_RECIPIENT",
        default="",
    )
    email_sender = _setting_text(
        settings,
        "email_sender",
        "EMAIL_SENDER",
        "EMAIL_FROM",
        default="",
    )
    brevo_api_key = _setting_text(
        settings,
        "brevo_api_key",
        "BREVO_API_KEY",
        default="",
    )
    user_confirmed_email_phase = _setting_bool(
        settings,
        "user_confirmed_email_phase",
        "USER_CONFIRMED_EMAIL_PHASE",
        default=False,
    )
    phase = _setting_text(settings, "phase", "PHASE", default="DAILY_REVIEW_PREVIEW") or "DAILY_REVIEW_PREVIEW"
    external_broker_enabled = _setting_bool(
        settings,
        "enable_external_broker",
        "ENABLE_EXTERNAL_BROKER",
        default=False,
    )
    investopedia_enabled = _setting_bool(
        settings,
        "enable_investopedia_simulator",
        "ENABLE_INVESTOPEDIA_SIMULATOR",
        default=False,
    )
    local_paper_trading_enabled = _setting_bool(
        settings,
        "enable_local_paper_trading",
        "ENABLE_LOCAL_PAPER_TRADING",
        default=False,
    )

    blockers: list[str] = []
    warnings: list[str] = []
    if not enable_email_notifications:
        blockers.append("ENABLE_EMAIL_NOTIFICATIONS=false")
    if not email_send_enabled:
        blockers.append("EMAIL_SEND_ENABLED=false")
    if email_dry_run:
        blockers.append("EMAIL_DRY_RUN=true")
    if not recipient:
        blockers.append("EMAIL_RECIPIENT empty")
    if email_provider == "brevo":
        if not email_sender:
            blockers.append("EMAIL_SENDER empty")
        if not brevo_api_key:
            blockers.append("BREVO_API_KEY missing")
    if not user_confirmed_email_phase:
        blockers.append("USER_CONFIRMED_EMAIL_PHASE=false")
    if phase != "DAILY_REVIEW_SEND_READY":
        blockers.append(f"PHASE={phase}")
    if external_broker_enabled:
        blockers.append("ENABLE_EXTERNAL_BROKER must remain false")
    if investopedia_enabled:
        blockers.append("ENABLE_INVESTOPEDIA_SIMULATOR must remain false")
    if local_paper_trading_enabled:
        warnings.append("ENABLE_LOCAL_PAPER_TRADING=true")

    real_email_send_allowed = not blockers
    hard_safety_blockers = {
        "ENABLE_EXTERNAL_BROKER must remain false",
        "ENABLE_INVESTOPEDIA_SIMULATOR must remain false",
    }
    if real_email_send_allowed:
        reason = "send_allowed"
        reason_detail = "All Daily Review email gate conditions are satisfied."
    elif any(blocker in hard_safety_blockers for blocker in blockers):
        reason = "blocked_by_gate"
        reason_detail = "Real email send is blocked by a hard safety gate."
    else:
        reason = "preview_only"
        reason_detail = "Preview-only mode because one or more send conditions are not yet satisfied."
    return {
        "real_email_send_allowed": real_email_send_allowed,
        "phase": phase,
        "reason": reason,
        "reason_detail": reason_detail,
        "blockers": blockers,
        "warnings": warnings,
        "enable_email_notifications": enable_email_notifications,
        "email_send_enabled": email_send_enabled,
        "email_dry_run": email_dry_run,
        "email_provider": email_provider,
        "email_recipient_configured": bool(recipient),
        "email_sender_configured": bool(email_sender),
        "brevo_api_key_configured": bool(brevo_api_key),
        "user_confirmed_email_phase": user_confirmed_email_phase,
        "external_broker_enabled": external_broker_enabled,
        "investopedia_enabled": investopedia_enabled,
        "local_paper_trading_enabled": local_paper_trading_enabled,
    }


DEFENSIVE_WEIGHTS = _build_defensive_weights(TICKERS, CASH_TICKER)


@dataclass(slots=True)
class DataConfig:
    """Configuration for historical price loading."""

    tickers: tuple[str, ...] = field(default_factory=lambda: tuple(TICKERS))
    start_date: str = START_DATE
    end_date: str | None = END_DATE
    lookback_days: int = COV_WINDOW
    auto_adjust: bool = True
    cache_path: Path = PRICE_CACHE_PATH
    use_cache: bool = True


@dataclass(slots=True)
class FeatureConfig:
    """Configuration for momentum forecast feature generation."""

    horizon_days: int = HORIZON_DAYS
    momentum_window_3m: int = MOMENTUM_SHORT
    momentum_window_6m: int = MOMENTUM_LONG
    kappa: float = KAPPA
    forecast_clip_lower: float = FORECAST_CLIP_LOWER
    forecast_clip_upper: float = FORECAST_CLIP_UPPER


@dataclass(slots=True)
class RiskConfig:
    """Configuration for covariance and portfolio risk estimation."""

    cov_window: int = COV_WINDOW
    horizon_days: int = HORIZON_DAYS
    cov_shrink_alpha: float = COV_SHRINK_ALPHA
    cov_jitter: float = COV_JITTER
    drawdown_limit: float = DRAWDOWN_LIMIT
    cvar_limit: float = CVAR_LIMIT
    risk_free_rate_annual: float = RISK_FREE_RATE_ANNUAL
    use_risk_filter: bool = USE_RISK_FILTER
    allow_daily_emergency_trades: bool = ALLOW_DAILY_EMERGENCY_TRADES
    defensive_weights: dict[str, float] = field(
        default_factory=lambda: DEFENSIVE_WEIGHTS.copy()
    )


@dataclass(slots=True)
class OptimizationConfig:
    """Compatibility configuration for the optimizer module."""

    risk_aversion: float = RISK_AVERSION
    turnover_penalty: float = TURNOVER_PENALTY
    concentration_penalty: float = CONCENTRATION_PENALTY
    tracking_error_penalty: float = 0.0
    covariance_lookback: int = COV_WINDOW
    max_single_weight: float = 1.0
    min_cash_buffer: float = 0.0
    max_turnover: float = MAX_TURNOVER
    cost_rate: float = COST_RATE
    base_buffer: float = BASE_BUFFER
    vol_buffer_multiplier: float = VOL_BUFFER_MULTIPLIER


@dataclass(slots=True)
class DecisionConfig:
    """Configuration for turning targets into trading recommendations."""

    min_rebalance_turnover: float = MIN_TURNOVER_TO_TRADE
    full_rebalance_turnover: float = PARTIAL_THRESHOLD
    partial_rebalance_ratio: float = PARTIAL_FRACTION
    strong_signal_threshold: float = STRONG_SIGNAL_THRESHOLD


@dataclass(slots=True)
class BacktestConfig:
    """Configuration for simple research backtests."""

    initial_capital: float = DEFAULT_PORTFOLIO_VALUE
    transaction_cost_bps: float = DEFAULT_BPS_PER_TURNOVER
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL
    max_turnover: float = MAX_TURNOVER


@dataclass(slots=True)
class NotificationConfig:
    """Configuration for optional email notifications."""

    enabled: bool = field(
        default_factory=lambda: os.getenv("ENABLE_EMAIL_NOTIFICATIONS", "false").lower() == "true"
    )
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", ""))
    smtp_port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    smtp_user: str = field(default_factory=lambda: os.getenv("SMTP_USER", ""))
    smtp_password: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    sender: str = field(default_factory=lambda: os.getenv("EMAIL_FROM", ""))
    recipients: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            email.strip()
            for email in os.getenv("EMAIL_TO", "").split(",")
            if email.strip()
        )
    )
    use_tls: bool = field(default_factory=lambda: os.getenv("SMTP_USE_TLS", "true").lower() == "true")


@dataclass(slots=True)
class AppConfig:
    """Compatibility wrapper used by the main pipeline."""

    project_name: str = "robust_3m_active_allocation_optimizer"
    benchmark_name: str = "Active 3M Allocation Benchmark"
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE
    tickers: tuple[str, ...] = field(default_factory=lambda: tuple(TICKERS))
    cash_ticker: str = CASH_TICKER
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    output_dir: Path = OUTPUT_DIR


def load_config() -> AppConfig:
    """Load the project configuration and ensure output paths exist."""

    load_dotenv(BASE_DIR / ".env", override=False)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return AppConfig()


def build_params(tickers: list[str] | tuple[str, ...] | None = None) -> dict[str, object]:
    """Build the central parameter dictionary used by main, backtest and optimizer."""

    active_tickers = list(dict.fromkeys([str(ticker) for ticker in (tickers or TICKERS)]))
    cash_ticker = CASH_TICKER if CASH_TICKER in active_tickers else None

    return {
        "solver": SOLVER_CONFIG.copy(),
        "reporting": REPORTING_CONFIG.copy(),
        "tickers": active_tickers,
        "cash_ticker": cash_ticker,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "momentum_short": MOMENTUM_SHORT,
        "momentum_long": MOMENTUM_LONG,
        "kappa": KAPPA,
        "forecast_clip_lower": FORECAST_CLIP_LOWER,
        "forecast_clip_upper": FORECAST_CLIP_UPPER,
        "cov_window": COV_WINDOW,
        "horizon_days": int(SOLVER_CONFIG["horizon_days"]),
        "cov_shrink_alpha": COV_SHRINK_ALPHA,
        "cov_jitter": COV_JITTER,
        "risk_aversion": RISK_AVERSION,
        "turnover_penalty": TURNOVER_PENALTY,
        "concentration_penalty": CONCENTRATION_PENALTY,
        "optimization_objective": OPTIMIZATION_OBJECTIVE,
        "scenario_solver_objective": str(SOLVER_CONFIG["objective"]),
        "eps_variance": float(SOLVER_CONFIG["eps_variance"]),
        "sharpe_epsilon": SHARPE_EPSILON,
        "sortino_target": SORTINO_TARGET,
        "lambda_cvar_sharpe": LAMBDA_CVAR_SHARPE,
        "lambda_turnover_sharpe": LAMBDA_TURNOVER_SHARPE,
        "lambda_cost_sharpe": LAMBDA_COST_SHARPE,
        "lambda_concentration_sharpe": LAMBDA_CONCENTRATION_SHARPE,
        "lambda_downside_sharpe": float(SOLVER_CONFIG["lambda_downside"]),
        "direct_scenario_optimizer_enabled": DIRECT_SCENARIO_OPTIMIZER_ENABLED,
        "final_allocation_method": "scenario_weighted_rf_sharpe_solver",
        "direct_scenario_lambda_turnover": DIRECT_SCENARIO_LAMBDA_TURNOVER,
        "direct_scenario_lambda_concentration": DIRECT_SCENARIO_LAMBDA_CONCENTRATION,
        "direct_scenario_lambda_downside": DIRECT_SCENARIO_LAMBDA_DOWNSIDE,
        "direct_scenario_optimizer_max_starts": DIRECT_SCENARIO_OPTIMIZER_MAX_STARTS,
        "direct_scenario_optimizer_method": DIRECT_SCENARIO_OPTIMIZER_METHOD,
        "direct_scenario_optimizer_maxiter": DIRECT_SCENARIO_OPTIMIZER_MAXITER,
        "direct_scenario_optimizer_ftol": DIRECT_SCENARIO_OPTIMIZER_FTOL,
        "direct_scenario_rf_mode": DIRECT_SCENARIO_RF_MODE,
        "allow_short": bool(SOLVER_CONFIG["allow_short"]),
        "long_only": bool(SOLVER_CONFIG["long_only"]),
        "use_scenario_covariance": bool(SOLVER_CONFIG["use_scenario_covariance"]),
        "use_scenario_probabilities": bool(SOLVER_CONFIG["use_scenario_probabilities"]),
        "use_rf_adjusted_sharpe": bool(SOLVER_CONFIG["use_rf_adjusted_sharpe"]),
        "write_scenario_solver_report": bool(REPORTING_CONFIG["write_scenario_solver_report"]),
        "write_pairwise_relationships": bool(REPORTING_CONFIG["write_pairwise_relationships"]),
        "write_scenario_matrices": bool(REPORTING_CONFIG["write_scenario_matrices"]),
        "scenario_covariance_lookback": SCENARIO_COVARIANCE_LOOKBACK,
        "scenario_covariance_shrink_alpha": SCENARIO_COVARIANCE_SHRINK_ALPHA,
        "scenario_probabilities": SCENARIO_PROBABILITIES.copy(),
        "scenario_weighted_probabilities": SCENARIO_WEIGHTED_PROBABILITIES.copy(),
        "scenario_risk_on_equity_shock": SCENARIO_RISK_ON_EQUITY_SHOCK,
        "scenario_risk_off_equity_shock": SCENARIO_RISK_OFF_EQUITY_SHOCK,
        "scenario_rates_up_duration_shock": SCENARIO_RATES_UP_DURATION_SHOCK,
        "scenario_rates_down_duration_shock": SCENARIO_RATES_DOWN_DURATION_SHOCK,
        "scenario_commodity_up_shock": SCENARIO_COMMODITY_UP_SHOCK,
        "scenario_equity_stress_shock": SCENARIO_EQUITY_STRESS_SHOCK,
        "scenario_defensive_carry_shock": SCENARIO_DEFENSIVE_CARRY_SHOCK,
        "cost_rate": COST_RATE,
        "default_commission_per_trade_usd": DEFAULT_COMMISSION_PER_TRADE,
        "default_bps_per_turnover": DEFAULT_BPS_PER_TURNOVER,
        "default_slippage_bps": DEFAULT_SLIPPAGE_BPS,
        "default_spread_bps": DEFAULT_SPREAD_BPS,
        "default_market_impact_bps": DEFAULT_MARKET_IMPACT_BPS,
        "use_live_bid_ask_if_available": USE_LIVE_BID_ASK_IF_AVAILABLE,
        "asset_cost_overrides": ASSET_COST_OVERRIDES,
        "base_buffer": BASE_BUFFER,
        "vol_buffer_multiplier": VOL_BUFFER_MULTIPLIER,
        "min_turnover_to_trade": MIN_TURNOVER_TO_TRADE,
        "partial_threshold": PARTIAL_THRESHOLD,
        "partial_fraction": PARTIAL_FRACTION,
        "scenario_execution_fraction": SCENARIO_EXECUTION_FRACTION,
        "execution_fraction": SCENARIO_EXECUTION_FRACTION,
        "strong_signal_threshold": STRONG_SIGNAL_THRESHOLD,
        "max_turnover": MAX_TURNOVER,
        "enable_active_preview": ENABLE_ACTIVE_PREVIEW,
        "active_preview_trade_now_hurdle": ACTIVE_PREVIEW_TRADE_NOW_HURDLE,
        "active_preview_execution_buffer": ACTIVE_PREVIEW_EXECUTION_BUFFER,
        "active_preview_model_uncertainty_multiplier": ACTIVE_PREVIEW_MODEL_UNCERTAINTY_MULTIPLIER,
        "active_preview_delta_vs_cash_min": ACTIVE_PREVIEW_DELTA_VS_CASH_MIN,
        "active_preview_p_current_min": ACTIVE_PREVIEW_P_CURRENT_MIN,
        "active_preview_p_cash_min": ACTIVE_PREVIEW_P_CASH_MIN,
        "active_preview_max_turnover": ACTIVE_PREVIEW_MAX_TURNOVER,
        "active_preview_min_order_value_usd": ACTIVE_PREVIEW_MIN_ORDER_VALUE_USD,
        "active_preview_allow_execution": ACTIVE_PREVIEW_ALLOW_EXECUTION,
        "drawdown_limit": DRAWDOWN_LIMIT,
        "cvar_limit": CVAR_LIMIT,
        "risk_free_rate_annual": float(SOLVER_CONFIG["risk_free_rate_annual"]),
        "risk_filter": USE_RISK_FILTER,
        "use_risk_filter": USE_RISK_FILTER,
        "allow_daily_emergency_trades": ALLOW_DAILY_EMERGENCY_TRADES,
        "asset_max_weights": get_asset_max_weights(),
        "group_map": get_group_map(),
        "group_limits": get_group_limits(),
        "equity_like_groups": EQUITY_LIKE_GROUPS,
        "defensive_groups": DEFENSIVE_GROUPS,
        "max_equity_like_total_normal": MAX_EQUITY_LIKE_TOTAL_NORMAL,
        "max_equity_like_total_risk_off": MAX_EQUITY_LIKE_TOTAL_RISK_OFF,
        "min_defensive_weight_normal": MIN_DEFENSIVE_WEIGHT_NORMAL,
        "min_defensive_weight_risk_off": MIN_DEFENSIVE_WEIGHT_RISK_OFF,
        "crypto_max_normal": CRYPTO_MAX_NORMAL,
        "crypto_max_risk_off": CRYPTO_MAX_RISK_OFF,
        "defensive_weights": _build_defensive_weights(active_tickers, cash_ticker),
        "min_cash_weight": 0.0,
        "default_portfolio_value": DEFAULT_PORTFOLIO_VALUE,
        "portfolio_nav_usd": PORTFOLIO_NAV_USD,
        "random_seed": RANDOM_SEED,
        "current_portfolio_source": CURRENT_PORTFOLIO_SOURCE,
        "current_portfolio_path": CURRENT_PORTFOLIO_PATH,
        "default_current_portfolio": DEFAULT_CURRENT_PORTFOLIO,
        "allow_fractional_shares": ALLOW_FRACTIONAL_SHARES,
        "min_order_value_usd": MIN_ORDER_VALUE_USD,
        "cash_buffer_usd": CASH_BUFFER_USD,
        "max_gross_exposure": MAX_GROSS_EXPOSURE,
        "dry_run": DRY_RUN,
        "enable_email_notifications": _env_flag("ENABLE_EMAIL_NOTIFICATIONS", False),
        "email_dry_run": EMAIL_DRY_RUN,
        "email_send_enabled": EMAIL_SEND_ENABLED,
        "email_provider": EMAIL_PROVIDER,
        "daily_briefing_only": DAILY_BRIEFING_ONLY,
        "max_emails_per_day": MAX_EMAILS_PER_DAY,
        "email_recipient": EMAIL_RECIPIENT,
        "email_sender": EMAIL_SENDER,
        "brevo_api_key": "[REDACTED]" if BREVO_API_KEY else "",
        "brevo_api_key_present": bool(BREVO_API_KEY),
        "user_confirmed_email_phase": USER_CONFIRMED_EMAIL_PHASE,
        "phase": PHASE,
        "enable_local_paper_trading": ENABLE_LOCAL_PAPER_TRADING,
        "enable_investopedia_simulator": ENABLE_INVESTOPEDIA_SIMULATOR,
        "enable_external_broker": ENABLE_EXTERNAL_BROKER,
        "paper_initial_cash": PAPER_INITIAL_CASH,
        "max_daily_turnover": MAX_DAILY_TURNOVER,
        "max_weekly_turnover": MAX_WEEKLY_TURNOVER,
        "max_monthly_turnover": MAX_MONTHLY_TURNOVER,
        "max_orders_per_day": MAX_ORDERS_PER_DAY,
        "min_minutes_between_orders": MIN_MINUTES_BETWEEN_ORDERS,
        "investopedia_base_url": INVESTOPEDIA_BASE_URL,
        "investopedia_game_id": INVESTOPEDIA_GAME_ID,
        "investopedia_dry_run": INVESTOPEDIA_DRY_RUN,
        "investopedia_use_existing_library": INVESTOPEDIA_USE_EXISTING_LIBRARY,
        "investopedia_credentials_present": bool(
            os.getenv("INVESTOPEDIA_USERNAME", "").strip()
            and os.getenv("INVESTOPEDIA_PASSWORD", "").strip()
            and INVESTOPEDIA_GAME_ID
        ),
        "output_dir": OUTPUT_DIR,
        "data_dir": DATA_DIR,
        "db_path": DB_PATH,
        "price_cache_path": PRICE_CACHE_PATH,
    }
