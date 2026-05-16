"""Build paper order preview from latest target allocation.

Preview-only. Does not send orders.

Inputs:
- outputs/paper_preview_candidate_v3_lower_turnover_allocations.csv
- latest prices from data/prices_cache.csv if available

Output:
- outputs/paper_order_preview.csv
- outputs/paper_order_preview_report.txt
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
ALLOC_PATH = OUTPUT_DIR / "paper_preview_candidate_v3_lower_turnover_allocations.csv"
PRICE_CACHE_PATH = ROOT / "data" / "prices_cache.csv"
POSITIONS_PATH = ROOT / "config" / "paper_positions.csv"

# Preview-only default NAV. This is not actual broker NAV.
DEFAULT_NAV_USD = 100_000.0

# Order-preview safety settings.
MIN_ORDER_USD = 100.0
MAX_ORDER_USD = 25_000.0
ROUND_TO_WHOLE_SHARES = True

# Fallback only. Preferred input is config/paper_positions.csv.
CURRENT_WEIGHTS = {}


def load_current_positions(prices: pd.Series, nav: float) -> tuple[pd.Series, float, str]:
    """Load current paper positions from config/paper_positions.csv.

    Expected columns:
    - ticker
    - shares

    Optional:
    - cash_usd row as ticker CASH or column cash_usd is not required yet.

    If the file is empty/missing, fall back to CURRENT_WEIGHTS.
    """

    if POSITIONS_PATH.exists():
        pos = pd.read_csv(POSITIONS_PATH)
        if not pos.empty and {"ticker", "shares"}.issubset(pos.columns):
            pos["ticker"] = pos["ticker"].astype(str).str.upper().str.strip()
            pos["shares"] = pd.to_numeric(pos["shares"], errors="coerce").fillna(0.0)

            cash_rows = pos[pos["ticker"].isin(["CASH", "USD"])]
            cash_usd = float(cash_rows["shares"].sum()) if not cash_rows.empty else 0.0

            holdings = pos[~pos["ticker"].isin(["CASH", "USD"])].copy()
            holdings = holdings.groupby("ticker", as_index=True)["shares"].sum()

            values = {}
            for ticker, shares in holdings.items():
                px = prices.get(ticker, float("nan"))
                if pd.notna(px) and px > 0:
                    values[ticker] = float(shares) * float(px)

            gross_value = float(sum(values.values()) + cash_usd)
            effective_nav = gross_value if gross_value > 0 else nav
            weights = pd.Series(values, dtype=float) / effective_nav if values else pd.Series(dtype=float)

            source = f"positions_file:{POSITIONS_PATH}"
            return weights, effective_nav, source

    if CURRENT_WEIGHTS:
        return pd.Series(CURRENT_WEIGHTS, dtype=float), nav, "CURRENT_WEIGHTS"

    return pd.Series(dtype=float), nav, "empty_all_cash"


def load_latest_prices() -> pd.Series:
    """Load latest available prices from cache.

    Supports both:
    - long format: date,ticker,close
    - wide format: date index column plus ticker columns
    """

    if not PRICE_CACHE_PATH.exists():
        return pd.Series(dtype=float)

    df = pd.read_csv(PRICE_CACHE_PATH)
    cols = {c.lower(): c for c in df.columns}

    # Long format.
    if {"date", "ticker", "close"}.issubset(cols):
        date_col = cols["date"]
        ticker_col = cols["ticker"]
        close_col = cols["close"]
        df[date_col] = pd.to_datetime(df[date_col])
        latest = (
            df.sort_values(date_col)
            .groupby(ticker_col)
            .tail(1)
            .set_index(ticker_col)[close_col]
            .astype(float)
        )
        return latest

    if {"date", "symbol", "close"}.issubset(cols):
        date_col = cols["date"]
        ticker_col = cols["symbol"]
        close_col = cols["close"]
        df[date_col] = pd.to_datetime(df[date_col])
        latest = (
            df.sort_values(date_col)
            .groupby(ticker_col)
            .tail(1)
            .set_index(ticker_col)[close_col]
            .astype(float)
        )
        return latest

    # Wide format, typical project cache:
    # first column is date-like, remaining columns are tickers.
    date_col = df.columns[0]
    wide = df.copy()
    wide[date_col] = pd.to_datetime(wide[date_col], errors="coerce")
    wide = wide.dropna(subset=[date_col]).sort_values(date_col)
    if wide.empty:
        return pd.Series(dtype=float)

    # Use the last non-null price per ticker, because some tickers can have NaN on latest row.
    price_cols = [c for c in wide.columns if c != date_col]
    prices = {}
    for col in price_cols:
        s = pd.to_numeric(wide[col], errors="coerce").dropna()
        if not s.empty:
            prices[col] = float(s.iloc[-1])

    return pd.Series(prices, dtype=float)


def main() -> None:
    if not ALLOC_PATH.exists():
        raise SystemExit(f"Missing allocation file: {ALLOC_PATH}")

    alloc = pd.read_csv(ALLOC_PATH)
    latest_date = alloc["date"].max()
    target = (
        alloc[alloc["date"].eq(latest_date)]
        .set_index("ticker")["weight"]
        .astype(float)
        .sort_values(ascending=False)
    )

    prices_all = load_latest_prices()
    current_loaded, nav, current_source = load_current_positions(prices_all, DEFAULT_NAV_USD)

    tickers = sorted(set(target.index) | set(current_loaded.index))
    current = current_loaded.reindex(tickers).fillna(0.0)
    target = target.reindex(tickers).fillna(0.0)

    prices = prices_all.reindex(tickers)

    rows = []
    for ticker in tickers:
        current_weight = float(current.loc[ticker])
        target_weight = float(target.loc[ticker])
        delta_weight = target_weight - current_weight
        dollar_delta = delta_weight * nav

        px = prices.loc[ticker] if ticker in prices.index else float("nan")
        raw_shares = dollar_delta / px if pd.notna(px) and px > 0 else float("nan")

        if pd.notna(raw_shares) and ROUND_TO_WHOLE_SHARES:
            estimated_shares = float(int(abs(raw_shares))) * (1.0 if raw_shares >= 0 else -1.0)
        else:
            estimated_shares = raw_shares

        rounded_dollar_delta = estimated_shares * px if pd.notna(estimated_shares) and pd.notna(px) else float("nan")

        reject_reason = ""
        if abs(delta_weight) < 1e-6:
            action = "HOLD"
            reject_reason = "below_weight_threshold"
        elif pd.isna(px) or px <= 0:
            action = "REJECT"
            reject_reason = "missing_or_invalid_price"
        elif abs(dollar_delta) < MIN_ORDER_USD:
            action = "REJECT"
            reject_reason = "below_min_order_usd"
        elif abs(dollar_delta) > MAX_ORDER_USD + 1e-9:
            action = "REJECT"
            reject_reason = "above_max_order_usd"
        elif ROUND_TO_WHOLE_SHARES and abs(estimated_shares) < 1:
            action = "REJECT"
            reject_reason = "rounds_to_zero_shares"
        elif delta_weight > 0:
            action = "BUY"
        else:
            action = "SELL"

        rows.append(
            {
                "preview_date": latest_date,
                "ticker": ticker,
                "action": action,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "delta_weight": delta_weight,
                "preview_nav_usd": nav,
                "dollar_delta": dollar_delta,
                "latest_price": px,
                "raw_estimated_shares": raw_shares,
                "estimated_shares": estimated_shares,
                "rounded_dollar_delta": rounded_dollar_delta,
                "reject_reason": reject_reason,
            }
        )

    out = pd.DataFrame(rows).sort_values("dollar_delta", ascending=False)
    order_path = OUTPUT_DIR / "paper_order_preview.csv"
    report_path = OUTPUT_DIR / "paper_order_preview_report.txt"
    out.to_csv(order_path, index=False)

    actionable = out[out["action"].isin(["BUY", "SELL"])].copy()
    rejected = out[out["action"].eq("REJECT")].copy()
    rounded_cash_impact = float(actionable["rounded_dollar_delta"].sum()) if "rounded_dollar_delta" in actionable else 0.0
    preview_cash_remaining = nav - rounded_cash_impact

    lines = [
        "Paper Order Preview Report",
        "",
        "status: preview_only_no_orders_sent",
        "candidate: v3_lower_turnover",
        f"target_allocation_date: {latest_date}",
        f"preview_nav_usd: {nav:.2f}",
        f"current_position_source: {current_source}",
        f"min_order_usd: {MIN_ORDER_USD:.2f}",
        f"max_order_usd: {MAX_ORDER_USD:.2f}",
        f"round_to_whole_shares: {ROUND_TO_WHOLE_SHARES}",
        f"rounded_cash_impact: {rounded_cash_impact:.2f}",
        f"preview_cash_remaining: {preview_cash_remaining:.2f}",
        "",
        "important:",
        "- This report does not send simulator or broker orders.",
        "- This report uses config/paper_positions.csv when present.",
        "- If config/paper_positions.csv is empty, current positions are assumed empty/all-cash.",
        "- Next step is replacing config/paper_positions.csv with real simulator export.",
        "",
        "target_allocation:",
    ]

    for row in target.sort_values(ascending=False).items():
        ticker, weight = row
        if abs(weight) > 1e-8:
            lines.append(f"- {ticker}: {weight:.4f}")

    lines.extend(["", "preview_orders:"])
    if actionable.empty:
        lines.append("- none")
    for row in actionable.itertuples(index=False):
        lines.append(
            f"- {row.action} {row.ticker}: "
            f"delta_weight={row.delta_weight:+.4f}, "
            f"dollar_delta={row.dollar_delta:+.2f}, "
            f"price={row.latest_price:.4f}, "
            f"shares={row.estimated_shares:.0f}, "
            f"rounded_dollar_delta={row.rounded_dollar_delta:+.2f}"
        )

    lines.extend(["", "rejected_orders:"])
    if rejected.empty:
        lines.append("- none")
    for row in rejected.itertuples(index=False):
        lines.append(
            f"- {row.ticker}: action={row.action}, "
            f"dollar_delta={row.dollar_delta:+.2f}, "
            f"reason={row.reject_reason}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(report_path.read_text())
    print(f"CSV: {order_path}")


if __name__ == "__main__":
    main()
