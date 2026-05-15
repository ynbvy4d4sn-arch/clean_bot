"""Optional, experimental adapter for the Investopedia stock simulator.

This module is intentionally conservative: it does not implement login
automation, CAPTCHA workarounds, or any mandatory external connectivity.
The main optimizer must remain fully usable even when this adapter is
disabled or unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path

import pandas as pd

from broker_interface import BrokerInterface

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional before dependencies are installed
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        """Fallback no-op when python-dotenv is unavailable."""

        return False


BASE_DIR = Path(__file__).resolve().parent


@dataclass(slots=True)
class InvestopediaSettings:
    """Environment-backed settings for the optional simulator adapter."""

    enabled: bool = False
    username: str = ""
    password: str = ""
    game_id: str = ""
    base_url: str = "https://www.investopedia.com/simulator/portfolio"
    dry_run: bool = True
    use_existing_library: bool = False
    env_path_exists: bool = False

    @property
    def credentials_present(self) -> bool:
        """Return whether all required credentials are present."""

        return bool(self.username and self.password and self.game_id)


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean flag from the environment with a safe default."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def load_investopedia_settings(env_path: str | Path | None = None) -> InvestopediaSettings:
    """Load Investopedia simulator settings from `.env` only.

    The function never logs secrets and never requires the `.env` file to
    exist. Missing or incomplete configuration simply leaves the adapter
    inactive or not ready.
    """

    resolved_env = Path(env_path) if env_path is not None else BASE_DIR / ".env"
    if resolved_env.exists():
        load_dotenv(resolved_env, override=False)

    return InvestopediaSettings(
        enabled=_env_flag("ENABLE_INVESTOPEDIA_SIMULATOR", False),
        username=os.getenv("INVESTOPEDIA_USERNAME", "").strip(),
        password=os.getenv("INVESTOPEDIA_PASSWORD", "").strip(),
        game_id=os.getenv("INVESTOPEDIA_GAME_ID", "").strip(),
        base_url=os.getenv(
            "INVESTOPEDIA_BASE_URL",
            "https://www.investopedia.com/simulator/portfolio",
        ).strip(),
        dry_run=_env_flag("INVESTOPEDIA_DRY_RUN", True),
        use_existing_library=_env_flag("INVESTOPEDIA_USE_EXISTING_LIBRARY", False),
        env_path_exists=resolved_env.exists(),
    )


class InvestopediaSimulatorAdapter(BrokerInterface):
    """Experimental adapter placeholder for future Investopedia simulator work.

    The current implementation is deliberately non-invasive and preview-only.
    Any execution path that would require unsupported automation fails with a
    clear, non-secret-bearing exception so callers can log and continue.
    """

    def __init__(
        self,
        settings: InvestopediaSettings,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)

    @classmethod
    def from_env(
        cls,
        env_path: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> "InvestopediaSimulatorAdapter":
        """Construct the adapter from `.env`-backed settings."""

        return cls(load_investopedia_settings(env_path=env_path), logger=logger)

    def _ensure_enabled(self) -> None:
        if not self.settings.enabled:
            raise RuntimeError(
                "Investopedia simulator integration is disabled. Set ENABLE_INVESTOPEDIA_SIMULATOR=true in .env to opt in."
            )

    def _ensure_supported(self) -> None:
        self._ensure_enabled()
        if not self.settings.credentials_present:
            raise RuntimeError(
                "Investopedia simulator integration is enabled but incomplete. Set INVESTOPEDIA_USERNAME, INVESTOPEDIA_PASSWORD and INVESTOPEDIA_GAME_ID in .env."
            )
        raise NotImplementedError(
            "Investopedia simulator integration is experimental and intentionally not auto-implemented here. "
            "No stable supported login/API flow is configured in this project, so no simulator orders were sent."
        )

    def login(self) -> dict[str, object]:
        """Attempt login only when explicitly enabled.

        The project intentionally does not implement fragile website automation,
        CAPTCHA bypasses, or MFA workarounds. This method therefore fails closed.
        """

        self._ensure_enabled()
        if not self.settings.credentials_present:
            raise RuntimeError(
                "Investopedia simulator integration is enabled but credentials are incomplete in .env."
            )
        raise NotImplementedError(
            "Investopedia login is intentionally not automated in this project. "
            "A stable supported simulator API or documented manual session flow would be required first."
        )

    def get_account_summary(self) -> dict[str, float | str]:
        self._ensure_supported()
        return {}

    def get_positions(self) -> pd.DataFrame:
        self._ensure_supported()
        return pd.DataFrame(columns=["ticker", "shares", "last_price", "market_value"])

    def get_cash(self) -> float:
        self._ensure_supported()
        return 0.0

    def get_latest_prices(self, tickers: list[str]) -> pd.Series:
        self._ensure_supported()
        return pd.Series(index=tickers, dtype=float)

    def preview_orders(self, order_df: pd.DataFrame) -> pd.DataFrame:
        """Return a preview copy without contacting any external system."""

        required = {"ticker", "side", "estimated_shares"}
        missing = sorted(required - set(order_df.columns))
        if missing:
            raise ValueError(f"Order preview is missing required columns: {', '.join(missing)}")
        preview = order_df.copy()
        preview["ticker"] = preview["ticker"].astype(str)
        preview["side"] = preview["side"].astype(str).str.upper()
        preview["estimated_shares"] = pd.to_numeric(preview["estimated_shares"], errors="coerce")
        if preview["estimated_shares"].isna().any():
            raise ValueError("Order preview contains non-finite estimated_shares values.")
        invalid_sides = sorted(set(preview["side"]) - {"BUY", "SELL", "HOLD"})
        if invalid_sides:
            raise ValueError(f"Order preview contains invalid sides: {', '.join(invalid_sides)}")
        return preview

    def submit_orders(self, order_df: pd.DataFrame) -> dict[str, object]:
        preview = self.preview_orders(order_df)
        self._ensure_enabled()
        if self.settings.dry_run:
            return {
                "submitted": False,
                "mode": "investopedia_dry_run",
                "order_count": int(len(preview)),
                "message": "Investopedia adapter is enabled but dry-run is active, so no external simulator orders were sent.",
            }
        self.login()
        self._ensure_supported()
        return {"submitted": False, "order_count": int(len(preview))}

    def get_order_status(self, order_id: str) -> dict[str, object]:
        self._ensure_supported()
        return {"order_id": order_id, "status": "unavailable"}

    def cancel_order(self, order_id: str) -> dict[str, object]:
        self._ensure_supported()
        return {"order_id": order_id, "cancelled": False}
