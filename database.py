"""SQLite persistence helpers for optimizer research runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

import pandas as pd

from paper_broker_stub import init_paper_tables as init_local_paper_tables


LOGGER = logging.getLogger(__name__)


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Return a SQLite connection and ensure the parent directory exists."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path, timeout=30.0)


def init_db(db_path: str | Path) -> None:
    """Create the required SQLite tables if they do not exist yet."""

    statements = [
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            run_timestamp TEXT,
            start_date TEXT,
            end_date TEXT,
            tickers TEXT,
            parameters_json TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS daily_results (
            run_id TEXT,
            date TEXT,
            next_date TEXT,
            equity REAL,
            portfolio_return_gross REAL,
            portfolio_return_net REAL,
            decision TEXT,
            risk_state TEXT,
            realized_turnover REAL,
            realized_cost REAL,
            net_benefit REAL,
            delta_score REAL,
            score_current REAL,
            score_target REAL,
            weekly_rebalance_day INTEGER,
            emergency_condition INTEGER,
            risk_gate_failed INTEGER,
            target_vol REAL,
            cvar_95 REAL,
            max_drawdown REAL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS weights (
            run_id TEXT,
            date TEXT,
            ticker TEXT,
            weight REAL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS target_weights (
            run_id TEXT,
            date TEXT,
            ticker TEXT,
            weight REAL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS order_preview (
            run_id TEXT,
            date TEXT,
            ticker TEXT,
            current_weight REAL,
            target_weight REAL,
            delta_weight REAL,
            side TEXT,
            estimated_order_value REAL,
            estimated_shares REAL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS performance_summary (
            run_id TEXT,
            strategy TEXT,
            cagr REAL,
            volatility REAL,
            sharpe REAL,
            max_drawdown REAL,
            cvar_95 REAL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS execution_results (
            run_id TEXT,
            timestamp TEXT,
            execution_mode TEXT,
            orders_submitted INTEGER,
            orders_failed INTEGER,
            message TEXT,
            errors_json TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tradability_status (
            run_id TEXT,
            timestamp TEXT,
            ticker TEXT,
            has_price_data INTEGER,
            has_latest_price INTEGER,
            is_enabled_in_universe INTEGER,
            is_tradable_in_local_paper INTEGER,
            is_tradable_in_investopedia INTEGER,
            is_short_or_inverse INTEGER,
            allowed_by_policy INTEGER,
            final_allowed INTEGER,
            reason TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS system_health_checks (
            timestamp TEXT,
            check_name TEXT,
            status TEXT,
            message TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS data_quality (
            run_id TEXT,
            timestamp TEXT,
            ticker TEXT,
            missing_ratio REAL,
            stale_price_flag INTEGER,
            latest_price_available INTEGER,
            history_length INTEGER,
            extreme_return_count INTEGER,
            last_valid_date TEXT,
            data_quality_score REAL,
            global_data_quality_score REAL
        )
        """,
    ]

    with get_connection(db_path) as connection:
        cursor = connection.cursor()
        for statement in statements:
            cursor.execute(statement)
        connection.commit()
    init_local_paper_tables(db_path)
    LOGGER.debug("SQLite schema is ready at %s.", db_path)


def _json_default(value: Any) -> str:
    """Serialize paths, timestamps and other simple objects into JSON strings."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return str(value)


def create_run(db_path: str | Path, params: dict[str, Any], tickers: list[str]) -> str:
    """Create a run header row and return the generated run identifier."""

    init_db(db_path)
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
    payload = (
        run_id,
        datetime.now(timezone.utc).isoformat(),
        str(params.get("start_date")),
        str(params.get("end_date")) if params.get("end_date") is not None else None,
        json.dumps(list(tickers)),
        json.dumps(params, default=_json_default, sort_keys=True),
    )

    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO runs (
                run_id, run_timestamp, start_date, end_date, tickers, parameters_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        connection.commit()

    LOGGER.info("Created SQLite run header %s.", run_id)
    return run_id


def save_daily_results(db_path: str | Path, run_id: str, daily_df: pd.DataFrame) -> None:
    """Persist daily backtest records."""

    if daily_df.empty:
        LOGGER.info("Skipping SQLite daily_results save for %s because the frame is empty.", run_id)
        return

    init_db(db_path)
    rows: list[tuple[Any, ...]] = []
    for _, record in daily_df.iterrows():
        rows.append(
            (
                run_id,
                str(pd.Timestamp(record["date"]).date()),
                str(pd.Timestamp(record["next_date"]).date()),
                float(record["equity"]),
                float(record["portfolio_return_gross"]),
                float(record["portfolio_return_net"]),
                str(record["decision"]),
                str(record["risk_state"]),
                float(record["realized_turnover"]),
                float(record["realized_cost"]),
                float(record["net_benefit"]) if pd.notna(record["net_benefit"]) else None,
                float(record["delta_score"]) if pd.notna(record["delta_score"]) else None,
                float(record["score_current"]) if pd.notna(record["score_current"]) else None,
                float(record["score_target"]) if pd.notna(record["score_target"]) else None,
                int(bool(record["weekly_rebalance_day"])),
                int(bool(record["emergency_condition"])),
                int(bool(record["risk_gate_failed"])),
                float(record["target_vol"]) if pd.notna(record["target_vol"]) else None,
                float(record["cvar_95"]) if pd.notna(record["cvar_95"]) else None,
                float(record["max_drawdown"]) if pd.notna(record["max_drawdown"]) else None,
            )
        )

    with get_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO daily_results (
                run_id, date, next_date, equity, portfolio_return_gross, portfolio_return_net,
                decision, risk_state, realized_turnover, realized_cost, net_benefit,
                delta_score, score_current, score_target, weekly_rebalance_day,
                emergency_condition, risk_gate_failed, target_vol, cvar_95, max_drawdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def save_weights(
    db_path: str | Path,
    run_id: str,
    weights_df: pd.DataFrame,
    table_name: str = "weights",
) -> None:
    """Persist portfolio weights in long format."""

    if weights_df.empty:
        LOGGER.info("Skipping SQLite %s save for %s because the frame is empty.", table_name, run_id)
        return
    if table_name not in {"weights", "target_weights"}:
        raise ValueError("table_name must be 'weights' or 'target_weights'.")

    init_db(db_path)
    long_df = weights_df.copy()
    long_df.index = pd.to_datetime(long_df.index)
    long_df = long_df.reset_index().rename(columns={long_df.index.name or "index": "date"})
    long_df = long_df.melt(id_vars="date", var_name="ticker", value_name="weight")
    long_df["run_id"] = run_id

    rows = [
        (
            str(record.run_id),
            str(pd.Timestamp(record.date).date()),
            str(record.ticker),
            float(record.weight),
        )
        for record in long_df.itertuples(index=False)
    ]

    with get_connection(db_path) as connection:
        connection.executemany(
            f"INSERT INTO {table_name} (run_id, date, ticker, weight) VALUES (?, ?, ?, ?)",
            rows,
        )
        connection.commit()


def save_order_preview(db_path: str | Path, run_id: str, order_preview_df: pd.DataFrame) -> None:
    """Persist the latest order preview table."""

    if order_preview_df.empty:
        LOGGER.info("Skipping SQLite order_preview save for %s because the frame is empty.", run_id)
        return

    init_db(db_path)
    preview = order_preview_df.copy()
    if "date" not in preview.columns:
        preview["date"] = datetime.now(timezone.utc).date().isoformat()

    rows = [
        (
            run_id,
            str(record.date),
            str(record.ticker),
            float(record.current_weight),
            float(record.target_weight),
            float(record.delta_weight),
            str(record.side),
            float(record.estimated_order_value),
            float(record.estimated_shares),
        )
        for record in preview.itertuples(index=False)
    ]

    with get_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO order_preview (
                run_id, date, ticker, current_weight, target_weight, delta_weight,
                side, estimated_order_value, estimated_shares
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def save_performance_summary(db_path: str | Path, run_id: str, perf_df: pd.DataFrame) -> None:
    """Persist performance summary rows."""

    if perf_df.empty:
        LOGGER.info("Skipping SQLite performance_summary save for %s because the frame is empty.", run_id)
        return

    init_db(db_path)
    rows = [
        (
            run_id,
            str(record.name),
            float(record.annualized_return) if pd.notna(record.annualized_return) else None,
            float(record.annualized_volatility) if pd.notna(record.annualized_volatility) else None,
            float(record.sharpe_ratio) if pd.notna(record.sharpe_ratio) else None,
            float(record.max_drawdown) if pd.notna(record.max_drawdown) else None,
            float(record.cvar_95) if pd.notna(record.cvar_95) else None,
        )
        for record in perf_df.itertuples(index=False)
    ]

    with get_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO performance_summary (
                run_id, strategy, cagr, volatility, sharpe, max_drawdown, cvar_95
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def save_execution_result(
    db_path: str | Path,
    run_id: str,
    execution_result: dict[str, Any],
) -> None:
    """Persist the execution-layer result without crashing the main run."""

    init_db(db_path)
    errors = execution_result.get("errors", [])
    payload = (
        str(run_id),
        datetime.now(timezone.utc).isoformat(),
        str(execution_result.get("execution_mode", "unknown")),
        int(execution_result.get("orders_submitted", 0) or 0),
        int(execution_result.get("orders_failed", 0) or 0),
        str(execution_result.get("message", "")),
        json.dumps(errors, default=_json_default, ensure_ascii=True),
    )
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO execution_results (
                run_id, timestamp, execution_mode, orders_submitted,
                orders_failed, message, errors_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        connection.commit()


def save_tradability_status(
    db_path: str | Path,
    run_id: str,
    tradability_df: pd.DataFrame,
) -> None:
    """Persist a tradability report to SQLite."""

    if tradability_df.empty:
        return
    init_db(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            str(run_id),
            timestamp,
            str(record.ticker),
            int(bool(record.has_price_data)),
            int(bool(record.has_latest_price)),
            int(bool(record.is_enabled_in_universe)),
            int(bool(record.is_tradable_in_local_paper)),
            int(bool(record.is_tradable_in_investopedia)),
            int(bool(record.is_short_or_inverse)),
            int(bool(record.allowed_by_policy)),
            int(bool(record.final_allowed)),
            str(record.reason),
        )
        for record in tradability_df.itertuples(index=False)
    ]
    with get_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO tradability_status (
                run_id, timestamp, ticker, has_price_data, has_latest_price,
                is_enabled_in_universe, is_tradable_in_local_paper,
                is_tradable_in_investopedia, is_short_or_inverse,
                allowed_by_policy, final_allowed, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def save_health_check_to_db(
    db_path: str | Path,
    checks_df: pd.DataFrame,
) -> None:
    """Persist health-check rows to SQLite."""

    if checks_df.empty:
        return
    init_db(db_path)
    rows = [
        (
            str(record.timestamp),
            str(record.check_name),
            str(record.status),
            str(record.message),
        )
        for record in checks_df.itertuples(index=False)
    ]
    with get_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO system_health_checks (timestamp, check_name, status, message)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def save_data_quality_to_db(
    db_path: str | Path,
    run_id: str,
    data_quality_report: dict[str, Any],
) -> None:
    """Persist a data-quality report to SQLite."""

    report_df = data_quality_report.get("report_df", pd.DataFrame()).copy()
    if report_df.empty:
        return
    init_db(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            str(run_id),
            timestamp,
            str(record.ticker),
            float(record.missing_ratio),
            int(bool(record.stale_price_flag)),
            int(bool(record.latest_price_available)),
            int(record.history_length),
            int(record.extreme_return_count),
            str(record.last_valid_date),
            float(record.data_quality_score),
            float(getattr(record, "global_data_quality_score", data_quality_report.get("global_data_quality_score", 0.0))),
        )
        for record in report_df.itertuples(index=False)
    ]
    with get_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO data_quality (
                run_id, timestamp, ticker, missing_ratio, stale_price_flag,
                latest_price_available, history_length, extreme_return_count,
                last_valid_date, data_quality_score, global_data_quality_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def init_paper_tables(db_path: str | Path) -> None:
    """Initialize the local paper-trading tables."""

    init_local_paper_tables(db_path)


def save_paper_trade(db_path: str | Path, trade: dict[str, Any]) -> None:
    """Persist a single local paper trade."""

    init_db(db_path)
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO paper_trades (
                trade_id, timestamp, ticker, side, shares, price,
                gross_value, cost, net_value, status, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(trade.get("trade_id")),
                str(trade.get("timestamp")),
                str(trade.get("ticker")),
                str(trade.get("side")),
                float(trade.get("shares", 0.0)),
                float(trade.get("price", 0.0)),
                float(trade.get("gross_value", 0.0)),
                float(trade.get("cost", 0.0)),
                float(trade.get("net_value", 0.0)),
                str(trade.get("status", "filled")),
                str(trade.get("source", "local_paper")),
            ),
        )
        connection.commit()


def save_paper_positions(db_path: str | Path, positions_df: pd.DataFrame) -> None:
    """Persist the current local paper positions."""

    init_db(db_path)
    if positions_df.empty:
        return
    rows = [
        (
            str(record.ticker),
            float(record.shares),
            float(record.last_price),
            float(record.market_value),
        )
        for record in positions_df.itertuples(index=False)
    ]
    with get_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO paper_positions (ticker, shares, last_price, market_value)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def save_paper_account_history(
    db_path: str | Path,
    date: str,
    cash: float,
    positions_value: float,
    total_equity: float,
) -> None:
    """Persist a local paper-account history snapshot."""

    init_db(db_path)
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO paper_account_history (date, cash, positions_value, total_equity)
            VALUES (?, ?, ?, ?)
            """,
            (str(date), float(cash), float(positions_value), float(total_equity)),
        )
        connection.commit()


def save_full_run(
    db_path: str | Path,
    run_id: str,
    result: dict[str, pd.DataFrame],
    order_preview_df: pd.DataFrame,
    perf_df: pd.DataFrame,
) -> None:
    """Persist the full run payload to SQLite."""

    init_db(db_path)
    try:
        save_daily_results(db_path=db_path, run_id=run_id, daily_df=result["daily"])
        save_weights(db_path=db_path, run_id=run_id, weights_df=result["weights"], table_name="weights")
        save_weights(
            db_path=db_path,
            run_id=run_id,
            weights_df=result["target_weights"],
            table_name="target_weights",
        )

        latest_date = str(pd.Timestamp(result["daily"].iloc[-1]["date"]).date()) if not result["daily"].empty else None
        preview_to_store = order_preview_df.copy()
        if latest_date is not None:
            preview_to_store["date"] = latest_date
        save_order_preview(db_path=db_path, run_id=run_id, order_preview_df=preview_to_store)
        save_performance_summary(db_path=db_path, run_id=run_id, perf_df=perf_df)
        if "execution_result" in result:
            save_execution_result(
                db_path=db_path,
                run_id=run_id,
                execution_result=result["execution_result"],
            )
        LOGGER.info("Saved run %s to SQLite database %s.", run_id, db_path)
    except sqlite3.Error as exc:
        LOGGER.exception("SQLite save_full_run failed for %s at %s: %s", run_id, db_path, exc)
        raise
