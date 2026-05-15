"""Abstract broker interface for future paper-trading or broker adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BrokerInterface(ABC):
    """Interface for broker-like integrations.

    This project does not execute real trades. Concrete implementations are expected
    to support previewing or locally simulating portfolio actions only, unless a
    future extension explicitly adds external connectivity.
    """

    @abstractmethod
    def get_account_summary(self) -> dict[str, float | str]:
        """Return a compact account summary."""

    @abstractmethod
    def get_positions(self) -> pd.DataFrame:
        """Return current positions."""

    @abstractmethod
    def get_cash(self) -> float:
        """Return current cash balance."""

    @abstractmethod
    def get_latest_prices(self, tickers: list[str]) -> pd.Series:
        """Return latest prices for the supplied tickers."""

    @abstractmethod
    def preview_orders(self, order_df: pd.DataFrame) -> pd.DataFrame:
        """Return a validated preview of proposed orders without executing them."""

    @abstractmethod
    def submit_orders(self, order_df: pd.DataFrame) -> dict[str, object]:
        """Submit orders to the implementation.

        For local paper-trading adapters this may only simulate execution.
        """

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict[str, object]:
        """Return the status of a previously submitted order."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict[str, object]:
        """Cancel a previously submitted order if supported."""
