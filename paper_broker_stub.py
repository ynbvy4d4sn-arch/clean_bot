"""Local paper-trading stub backed by SQLite, without any external API."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from broker_interface import BrokerInterface


LOGGER = logging.getLogger(__name__)


def _connect(db_path: str | Path) -> sqlite3.Connection:
    """Return a SQLite connection and ensure the parent directory exists."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def _init_paper_tables(db_path: str | Path) -> None:
    """Create paper-trading tables if they do not exist yet."""

    statements = [
        """
        CREATE TABLE IF NOT EXISTS paper_account (
            id INTEGER PRIMARY KEY,
            cash REAL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS paper_positions (
            ticker TEXT PRIMARY KEY,
            shares REAL,
            last_price REAL,
            market_value REAL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS paper_trades (
            trade_id TEXT PRIMARY KEY,
            timestamp TEXT,
            ticker TEXT,
            side TEXT,
            shares REAL,
            price REAL,
            gross_value REAL,
            cost REAL,
            net_value REAL,
            status TEXT,
            source TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS paper_account_history (
            date TEXT,
            cash REAL,
            positions_value REAL,
            total_equity REAL
        )
        """,
    ]

    with _connect(db_path) as connection:
        for statement in statements:
            connection.execute(statement)
        existing_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(paper_trades)").fetchall()
        }
        if "status" not in existing_columns:
            connection.execute("ALTER TABLE paper_trades ADD COLUMN status TEXT")
        if "source" not in existing_columns:
            connection.execute("ALTER TABLE paper_trades ADD COLUMN source TEXT")
        connection.commit()


def init_paper_tables(db_path: str | Path) -> None:
    """Public helper used by the database module and smoke tests."""

    _init_paper_tables(db_path)


def initialize_paper_account(db_path: str | Path, initial_cash: float = 10000) -> None:
    """Initialize a local paper account if it does not exist yet."""

    _init_paper_tables(db_path)
    with _connect(db_path) as connection:
        existing = connection.execute("SELECT COUNT(*) FROM paper_account").fetchone()
        if existing and int(existing[0]) > 0:
            return
        connection.execute("INSERT INTO paper_account (id, cash) VALUES (1, ?)", (float(initial_cash),))
        connection.commit()


def get_paper_positions(db_path: str | Path) -> pd.DataFrame:
    """Return current paper positions."""

    _init_paper_tables(db_path)
    with _connect(db_path) as connection:
        positions = pd.read_sql_query(
            "SELECT ticker, shares, last_price, market_value FROM paper_positions ORDER BY ticker",
            connection,
        )
    if positions.empty:
        return pd.DataFrame(columns=["ticker", "shares", "last_price", "market_value", "weight"])
    total_value = float(positions["market_value"].sum())
    positions["weight"] = positions["market_value"] / total_value if total_value > 0.0 else 0.0
    return positions


def get_paper_cash(db_path: str | Path) -> float:
    """Return current paper-account cash."""

    initialize_paper_account(db_path)
    with _connect(db_path) as connection:
        row = connection.execute("SELECT cash FROM paper_account WHERE id = 1").fetchone()
    return float(row[0]) if row else 0.0


def _upsert_position(
    connection: sqlite3.Connection,
    ticker: str,
    shares: float,
    last_price: float,
) -> None:
    """Insert or update a paper position."""

    market_value = float(shares) * float(last_price)
    if abs(float(shares)) < 1e-12:
        connection.execute("DELETE FROM paper_positions WHERE ticker = ?", (ticker,))
        return

    connection.execute(
        """
        INSERT INTO paper_positions (ticker, shares, last_price, market_value)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            shares = excluded.shares,
            last_price = excluded.last_price,
            market_value = excluded.market_value
        """,
        (ticker, float(shares), float(last_price), market_value),
    )


def execute_order_preview_locally(
    db_path: str | Path,
    order_preview_df: pd.DataFrame,
    latest_prices: pd.Series,
    cost_rate: float,
) -> pd.DataFrame:
    """Simulate execution of an order preview against the local paper account.

    Orders are executed immediately at the supplied `latest_prices`. No external API,
    no real orders, and no broker connectivity are used.
    """

    initialize_paper_account(db_path)
    preview = order_preview_df.copy()
    if preview.empty:
        return preview

    prices = latest_prices.astype(float).copy()
    prices.index = pd.Index([str(ticker) for ticker in prices.index], name="ticker")

    with _connect(db_path) as connection:
        cash = get_paper_cash(db_path)
        positions_df = get_paper_positions(db_path)
        positions = (
            positions_df.set_index("ticker")
            if not positions_df.empty
            else pd.DataFrame(columns=["shares", "last_price", "market_value"]).set_index(
                pd.Index([], name="ticker")
            )
        )

        trade_rows: list[dict[str, Any]] = []

        for record in preview.itertuples(index=False):
            ticker = str(record.ticker)
            side = str(record.side)
            if side == "HOLD":
                continue

            price = float(prices.get(ticker, 0.0))
            if price <= 0.0:
                LOGGER.warning("Skipping local paper trade for %s because no valid price is available.", ticker)
                continue

            desired_shares = float(record.estimated_shares)
            if desired_shares == 0.0:
                continue

            shares = abs(desired_shares)
            gross_value = shares * price
            modeled_row_cost = getattr(record, "estimated_total_order_cost", None)
            original_order_value = abs(float(getattr(record, "order_value", getattr(record, "estimated_order_value", gross_value))))
            if modeled_row_cost is None or not np.isfinite(float(modeled_row_cost)):
                modeled_cost_rate = float(cost_rate)
            else:
                modeled_cost_rate = max(float(modeled_row_cost), 0.0) / max(original_order_value, 1e-12)
            cost = abs(gross_value) * modeled_cost_rate
            net_value = gross_value + cost if side == "BUY" else gross_value - cost

            current_shares = float(positions.loc[ticker, "shares"]) if ticker in positions.index else 0.0
            new_shares = current_shares

            if side == "BUY":
                if net_value > cash:
                    affordable_shares = max((cash / (price * (1.0 + modeled_cost_rate))), 0.0)
                    shares = float(affordable_shares)
                    gross_value = shares * price
                    cost = abs(gross_value) * modeled_cost_rate
                    net_value = gross_value + cost
                if shares <= 0.0:
                    continue
                cash -= net_value
                new_shares = current_shares + shares
            elif side == "SELL":
                shares = min(shares, current_shares)
                gross_value = shares * price
                cost = abs(gross_value) * modeled_cost_rate
                net_value = gross_value - cost
                if shares <= 0.0:
                    continue
                cash += net_value
                new_shares = current_shares - shares
            else:
                continue

            _upsert_position(connection, ticker=ticker, shares=new_shares, last_price=price)
            positions.loc[ticker, "shares"] = new_shares
            positions.loc[ticker, "last_price"] = price
            positions.loc[ticker, "market_value"] = new_shares * price

            trade_rows.append(
                {
                    "trade_id": f"paper_{uuid4().hex[:12]}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "ticker": ticker,
                    "side": side,
                    "shares": float(shares),
                    "price": price,
                    "gross_value": gross_value,
                    "cost": cost,
                    "net_value": net_value,
                    "status": "filled",
                    "source": "local_paper",
                }
            )

        connection.execute("UPDATE paper_account SET cash = ? WHERE id = 1", (float(cash),))
        if trade_rows:
            connection.executemany(
                """
                INSERT INTO paper_trades (
                    trade_id, timestamp, ticker, side, shares, price, gross_value, cost, net_value, status, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["trade_id"],
                        row["timestamp"],
                        row["ticker"],
                        row["side"],
                        row["shares"],
                        row["price"],
                        row["gross_value"],
                        row["cost"],
                        row["net_value"],
                        row["status"],
                        row["source"],
                    )
                    for row in trade_rows
                ],
            )
        connection.commit()

    return pd.DataFrame(trade_rows)


def save_paper_account_state(db_path: str | Path, date: str | pd.Timestamp) -> None:
    """Snapshot current local paper-account equity to history."""

    initialize_paper_account(db_path)
    cash = get_paper_cash(db_path)
    positions = get_paper_positions(db_path)
    positions_value = float(positions["market_value"].sum()) if not positions.empty else 0.0
    total_equity = cash + positions_value

    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO paper_account_history (date, cash, positions_value, total_equity)
            VALUES (?, ?, ?, ?)
            """,
            (str(pd.Timestamp(date).date()), float(cash), positions_value, total_equity),
        )
        connection.commit()


def get_paper_account_summary(db_path: str | Path) -> dict[str, float | str]:
    """Return a compact summary of the local paper account."""

    initialize_paper_account(db_path)
    cash = get_paper_cash(db_path)
    positions = get_paper_positions(db_path)
    positions_value = float(positions["market_value"].sum()) if not positions.empty else 0.0
    total_equity = cash + positions_value
    return {
        "cash": float(cash),
        "positions_value": positions_value,
        "total_equity": total_equity,
        "number_of_positions": int(len(positions)),
        "currency": "USD",
        "source": "local_paper",
    }


class PaperBrokerStub(BrokerInterface):
    """Local broker-like adapter that only simulates paper-trading in SQLite."""

    def __init__(self, db_path: str | Path, initial_cash: float = 10000) -> None:
        self.db_path = Path(db_path)
        initialize_paper_account(self.db_path, initial_cash=initial_cash)

    def get_account_summary(self) -> dict[str, float | str]:
        return get_paper_account_summary(self.db_path)

    def get_positions(self) -> pd.DataFrame:
        return get_paper_positions(self.db_path)

    def get_cash(self) -> float:
        return get_paper_cash(self.db_path)

    def get_latest_prices(self, tickers: list[str]) -> pd.Series:
        positions = get_paper_positions(self.db_path)
        if positions.empty:
            return pd.Series(0.0, index=tickers, dtype=float)
        prices = positions.set_index("ticker")["last_price"].astype(float)
        return prices.reindex(tickers).fillna(0.0)

    def preview_orders(self, order_df: pd.DataFrame) -> pd.DataFrame:
        required = {
            "ticker",
            "current_weight",
            "target_weight",
            "delta_weight",
            "side",
            "estimated_order_value",
            "estimated_shares",
        }
        missing = sorted(required - set(order_df.columns))
        if missing:
            raise ValueError(f"Order preview is missing required columns: {', '.join(missing)}")
        preview = order_df.copy()
        preview["ticker"] = preview["ticker"].astype(str)
        preview["side"] = preview["side"].astype(str).str.upper()
        invalid_sides = sorted(set(preview["side"]) - {"BUY", "SELL", "HOLD"})
        if invalid_sides:
            raise ValueError(f"Order preview contains invalid sides: {', '.join(invalid_sides)}")
        return preview

    def submit_orders(self, order_df: pd.DataFrame) -> dict[str, object]:
        preview = self.preview_orders(order_df)
        prices = self.get_latest_prices(preview["ticker"].astype(str).tolist())
        trades = execute_order_preview_locally(
            db_path=self.db_path,
            order_preview_df=preview,
            latest_prices=prices,
            cost_rate=0.0,
        )
        return {
            "status": "simulated",
            "trade_count": int(len(trades)),
            "trades": trades.to_dict(orient="records"),
        }

    def get_order_status(self, order_id: str) -> dict[str, object]:
        with _connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT trade_id, timestamp, ticker, side, shares, price FROM paper_trades WHERE trade_id = ?",
                (str(order_id),),
            ).fetchone()
        if row is None:
            return {"order_id": str(order_id), "status": "not_found"}
        return {
            "order_id": row[0],
            "timestamp": row[1],
            "ticker": row[2],
            "side": row[3],
            "shares": float(row[4]),
            "price": float(row[5]),
            "status": "filled",
        }

    def cancel_order(self, order_id: str) -> dict[str, object]:
        return {
            "order_id": str(order_id),
            "status": "not_supported",
            "message": "PaperBrokerStub simulates immediate fills and does not keep open orders.",
        }

    def save_account_history(self, date: str | pd.Timestamp) -> None:
        """Persist a local paper-equity snapshot."""

        save_paper_account_state(self.db_path, date=date)
