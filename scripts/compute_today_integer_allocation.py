"""Research-only integer target holdings helper.

WARNING:
- This script computes target holdings only.
- It does not create executable simulator orders.
- It does not apply the final daily-bot execution gate, reconciliation,
  whole-run validation, or paper-broker state checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def read_latest_prices(path: str) -> tuple[pd.Series, str]:
    df = pd.read_csv(path)

    # Find date column
    date_col = None
    for c in df.columns:
        if c.lower() in {"date", "datetime", "timestamp"}:
            date_col = c
            break

    if date_col is None:
        # assume first column is date-like if not numeric
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    latest_row = df.iloc[-1]
    latest_date = str(latest_row[date_col].date())

    prices = latest_row.drop(labels=[date_col])
    prices = pd.to_numeric(prices, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna()
    prices = prices[prices > 0]

    return prices, latest_date


def read_target_weights(path: str) -> pd.Series:
    df = pd.read_csv(path)

    if {"ticker", "weight"}.issubset(df.columns):
        w = df.set_index("ticker")["weight"]
    elif {"ticker", "optimal_weight"}.issubset(df.columns):
        w = df.set_index("ticker")["optimal_weight"]
    else:
        # target_weights.csv style: date, XLK, XLC, ...
        date_col = None
        for c in df.columns:
            if c.lower() in {"date", "datetime", "timestamp"}:
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]

        latest = df.iloc[-1].drop(labels=[date_col])
        w = pd.to_numeric(latest, errors="coerce")

    w = pd.to_numeric(w, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    w = w[w > 0]
    w = w / w.sum()
    return w


def integer_allocate(
    weights: pd.Series,
    prices: pd.Series,
    nav: float,
    min_order_value: float,
) -> pd.DataFrame:
    common = weights.index.intersection(prices.index)
    weights = weights.loc[common].copy()
    prices = prices.loc[common].copy()

    weights = weights / weights.sum()

    target_values = weights * nav
    raw_shares = target_values / prices

    # Initial whole-share allocation by floor
    shares = np.floor(raw_shares).astype(int)
    spent = float((shares * prices).sum())
    cash_left = nav - spent

    # Greedy improvement: buy one extra share where it most reduces target-value gap
    max_loops = 100000
    loops = 0

    while loops < max_loops:
        affordable = prices[prices <= cash_left]
        if affordable.empty:
            break

        current_values = shares * prices
        gaps = target_values - current_values

        # Only consider assets still under target and affordable
        candidates = gaps[gaps > 0].index.intersection(affordable.index)
        if len(candidates) == 0:
            break

        # Buy where one extra share gives best relative improvement
        before_error = (gaps.loc[candidates].abs() / nav)
        after_values = current_values.loc[candidates] + prices.loc[candidates]
        after_gaps = target_values.loc[candidates] - after_values
        after_error = (after_gaps.abs() / nav)

        improvement = before_error - after_error
        best = improvement.idxmax()

        if improvement.loc[best] <= 0:
            break

        shares.loc[best] += 1
        cash_left -= float(prices.loc[best])
        loops += 1

    final_values = shares * prices
    final_weights = final_values / nav
    drift = final_weights - weights

    out = pd.DataFrame({
        "ticker": weights.index,
        "target_weight": weights.values,
        "latest_price": prices.values,
        "target_value": target_values.values,
        "ideal_fractional_shares": raw_shares.values,
        "whole_shares": shares.values,
        "actual_value": final_values.values,
        "actual_weight": final_weights.values,
        "weight_drift": drift.values,
        "abs_weight_drift": drift.abs().values,
    })

    out["side"] = np.where(out["actual_value"] >= min_order_value, "BUY", "SKIP_TOO_SMALL")
    out["preview_context"] = "research_target_holdings_only"
    out["executable"] = False
    out["preview_note"] = "Target holdings only. Not executable simulator orders."
    out = out.sort_values("target_weight", ascending=False)

    cash_row = pd.DataFrame([{
        "ticker": "CASH_LEFT",
        "target_weight": 0.0,
        "latest_price": 1.0,
        "target_value": 0.0,
        "ideal_fractional_shares": np.nan,
        "whole_shares": np.nan,
        "actual_value": cash_left,
        "actual_weight": cash_left / nav,
        "weight_drift": cash_left / nav,
        "abs_weight_drift": cash_left / nav,
        "side": "HOLD_CASH",
        "preview_context": "research_target_holdings_only",
        "executable": False,
        "preview_note": "Target holdings only. Not executable simulator orders.",
    }])

    out = pd.concat([out, cash_row], ignore_index=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nav", type=float, required=True, help="Portfolio value in USD, e.g. 100000")
    parser.add_argument(
        "--weights",
        default="outputs/selected_candidate_weights.csv",
        help="CSV with ticker,weight or latest row of target_weights.csv",
    )
    parser.add_argument(
        "--prices",
        default="data/prices_cache.csv",
        help="Wide price cache CSV",
    )
    parser.add_argument(
        "--out",
        default="outputs/today_integer_allocation.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--min-order-value",
        type=float,
        default=10.0,
        help="Minimum meaningful order value in USD",
    )
    args = parser.parse_args()

    weights = read_target_weights(args.weights)
    prices, latest_date = read_latest_prices(args.prices)

    allocation = integer_allocate(
        weights=weights,
        prices=prices,
        nav=args.nav,
        min_order_value=args.min_order_value,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    allocation.to_csv(out_path, index=False)

    print("WARNING: This script computes target holdings only. It does not create executable simulator orders.")
    invested = allocation.loc[allocation["ticker"] != "CASH_LEFT", "actual_value"].sum()
    cash_left = allocation.loc[allocation["ticker"] == "CASH_LEFT", "actual_value"].iloc[0]
    total_abs_drift = allocation.loc[allocation["ticker"] != "CASH_LEFT", "abs_weight_drift"].sum()

    print("\nInteger allocation computed")
    print(f"Latest price date: {latest_date}")
    print(f"NAV: ${args.nav:,.2f}")
    print(f"Invested: ${invested:,.2f}")
    print(f"Cash left: ${cash_left:,.2f}")
    print(f"Total absolute weight drift: {total_abs_drift:.4%}")
    print(f"Output: {out_path}")

    print("\nOrders:")
    show = allocation[allocation["ticker"] != "CASH_LEFT"].copy()
    show = show[show["whole_shares"] > 0]
    cols = [
        "ticker",
        "target_weight",
        "latest_price",
        "whole_shares",
        "actual_value",
        "actual_weight",
        "weight_drift",
    ]
    print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()
