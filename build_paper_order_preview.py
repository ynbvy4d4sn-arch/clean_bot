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

# Preview-only default NAV. This is not actual broker NAV.
DEFAULT_NAV_USD = 100_000.0

# Replace this later with real simulator positions.
CURRENT_WEIGHTS = {
    # Example: empty portfolio / all cash for first preview.
    # "XLK": 0.10,
    # "SGOV": 0.90,
}


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

    tickers = sorted(set(target.index) | set(CURRENT_WEIGHTS))
    current = pd.Series(CURRENT_WEIGHTS, index=tickers, dtype=float).fillna(0.0)
    target = target.reindex(tickers).fillna(0.0)

    prices = load_latest_prices().reindex(tickers)
    nav = DEFAULT_NAV_USD

    rows = []
    for ticker in tickers:
        current_weight = float(current.loc[ticker])
        target_weight = float(target.loc[ticker])
        delta_weight = target_weight - current_weight
        dollar_delta = delta_weight * nav

        px = prices.loc[ticker] if ticker in prices.index else float("nan")
        shares = dollar_delta / px if pd.notna(px) and px > 0 else float("nan")

        if abs(delta_weight) < 1e-6:
            action = "HOLD"
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
                "estimated_shares": shares,
            }
        )

    out = pd.DataFrame(rows).sort_values("dollar_delta", ascending=False)
    order_path = OUTPUT_DIR / "paper_order_preview.csv"
    report_path = OUTPUT_DIR / "paper_order_preview_report.txt"
    out.to_csv(order_path, index=False)

    actionable = out[out["action"].ne("HOLD")].copy()

    lines = [
        "Paper Order Preview Report",
        "",
        "status: preview_only_no_orders_sent",
        "candidate: v3_lower_turnover",
        f"target_allocation_date: {latest_date}",
        f"preview_nav_usd: {nav:.2f}",
        "",
        "important:",
        "- This report does not send simulator or broker orders.",
        "- Current positions are currently assumed as empty/all-cash unless CURRENT_WEIGHTS is edited.",
        "- Next step is replacing CURRENT_WEIGHTS with real simulator positions.",
        "",
        "target_allocation:",
    ]

    for row in target.sort_values(ascending=False).items():
        ticker, weight = row
        if abs(weight) > 1e-8:
            lines.append(f"- {ticker}: {weight:.4f}")

    lines.extend(["", "preview_orders:"])
    for row in actionable.itertuples(index=False):
        lines.append(
            f"- {row.action} {row.ticker}: "
            f"delta_weight={row.delta_weight:+.4f}, "
            f"dollar_delta={row.dollar_delta:+.2f}, "
            f"price={row.latest_price}, "
            f"estimated_shares={row.estimated_shares}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(report_path.read_text())
    print(f"CSV: {order_path}")


if __name__ == "__main__":
    main()
