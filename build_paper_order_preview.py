"""Build paper order preview from latest target allocation.

Preview-only. Does not send orders.

Inputs:
- outputs/paper_preview_candidate_v3_lower_turnover_allocations.csv
- data/prices_cache.csv
- config/paper_positions.csv

Outputs:
- outputs/paper_order_preview.csv
- outputs/paper_order_preview_report.txt
"""

from __future__ import annotations

from pathlib import Path
import math
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
ALLOC_PATH = OUTPUT_DIR / "paper_preview_candidate_v3_lower_turnover_allocations.csv"
PRICE_CACHE_PATH = ROOT / "data" / "prices_cache.csv"
POSITIONS_PATH = ROOT / "config" / "paper_positions.csv"

DEFAULT_NAV_USD = 100_000.0

MIN_ORDER_USD = 100.0
MAX_ORDER_USD = 25_000.0
ROUND_TO_WHOLE_SHARES = True
MIN_CASH_BUFFER_USD = 500.0

CURRENT_WEIGHTS: dict[str, float] = {}


def load_latest_prices() -> pd.Series:
    """Load latest available prices from cache.

    Supports:
    - long format: date,ticker,close
    - wide format: date index column plus ticker columns
    """

    if not PRICE_CACHE_PATH.exists():
        return pd.Series(dtype=float)

    df = pd.read_csv(PRICE_CACHE_PATH)
    cols = {c.lower(): c for c in df.columns}

    if {"date", "ticker", "close"}.issubset(cols):
        date_col = cols["date"]
        ticker_col = cols["ticker"]
        close_col = cols["close"]
        df[date_col] = pd.to_datetime(df[date_col])
        return (
            df.sort_values(date_col)
            .groupby(ticker_col)
            .tail(1)
            .set_index(ticker_col)[close_col]
            .astype(float)
        )

    if {"date", "symbol", "close"}.issubset(cols):
        date_col = cols["date"]
        ticker_col = cols["symbol"]
        close_col = cols["close"]
        df[date_col] = pd.to_datetime(df[date_col])
        return (
            df.sort_values(date_col)
            .groupby(ticker_col)
            .tail(1)
            .set_index(ticker_col)[close_col]
            .astype(float)
        )

    date_col = df.columns[0]
    wide = df.copy()
    wide[date_col] = pd.to_datetime(wide[date_col], errors="coerce")
    wide = wide.dropna(subset=[date_col]).sort_values(date_col)
    if wide.empty:
        return pd.Series(dtype=float)

    prices = {}
    for col in [c for c in wide.columns if c != date_col]:
        s = pd.to_numeric(wide[col], errors="coerce").dropna()
        if not s.empty:
            prices[col] = float(s.iloc[-1])

    return pd.Series(prices, dtype=float)


def load_current_positions(prices: pd.Series, fallback_nav: float) -> tuple[pd.Series, pd.Series, float, float, str]:
    """Load current paper positions.

    Expected config/paper_positions.csv columns:
    - ticker
    - shares

    Special tickers:
    - CASH or USD are interpreted as cash dollars.
    """

    if POSITIONS_PATH.exists():
        pos = pd.read_csv(POSITIONS_PATH)
        if not pos.empty and {"ticker", "shares"}.issubset(pos.columns):
            pos["ticker"] = pos["ticker"].astype(str).str.upper().str.strip()
            pos["shares"] = pd.to_numeric(pos["shares"], errors="coerce").fillna(0.0)

            cash_rows = pos[pos["ticker"].isin(["CASH", "USD"])]
            cash_usd = float(cash_rows["shares"].sum()) if not cash_rows.empty else 0.0

            holdings = (
                pos[~pos["ticker"].isin(["CASH", "USD"])]
                .groupby("ticker", as_index=True)["shares"]
                .sum()
            )

            values = {}
            valid_shares = {}
            for ticker, shares in holdings.items():
                px = prices.get(ticker, float("nan"))
                if pd.notna(px) and px > 0:
                    valid_shares[ticker] = float(shares)
                    values[ticker] = float(shares) * float(px)

            holdings_value = float(sum(values.values()))
            effective_nav = holdings_value + cash_usd
            if effective_nav <= 0:
                effective_nav = fallback_nav

            current_weights = pd.Series(values, dtype=float) / effective_nav if values else pd.Series(dtype=float)
            current_shares = pd.Series(valid_shares, dtype=float)

            return (
                current_weights,
                current_shares,
                cash_usd,
                effective_nav,
                f"positions_file:{POSITIONS_PATH}",
            )

    if CURRENT_WEIGHTS:
        weights = pd.Series(CURRENT_WEIGHTS, dtype=float)
        return weights, pd.Series(dtype=float), fallback_nav, fallback_nav, "CURRENT_WEIGHTS"

    return pd.Series(dtype=float), pd.Series(dtype=float), fallback_nav, fallback_nav, "empty_all_cash"


def split_order_rows(base_row: dict) -> list[dict]:
    """Split one desired order into max-dollar chunks.

    This is preview-only. It does not send orders.
    """

    action = str(base_row["action"])
    if action not in {"BUY", "SELL"}:
        return [base_row]

    dollar_delta = float(base_row["dollar_delta"])
    abs_dollar = abs(dollar_delta)

    if abs_dollar <= MAX_ORDER_USD + 1e-9:
        base_row["order_part"] = 1
        base_row["order_part_count"] = 1
        return [base_row]

    part_count = int(math.ceil(abs_dollar / MAX_ORDER_USD))
    rows = []

    remaining = dollar_delta
    for part in range(1, part_count + 1):
        signed_part_dollar = math.copysign(min(MAX_ORDER_USD, abs(remaining)), remaining)
        row = dict(base_row)
        row["order_part"] = part
        row["order_part_count"] = part_count
        row["dollar_delta"] = signed_part_dollar

        px = float(row["latest_price"])
        raw_shares = signed_part_dollar / px if px > 0 else float("nan")
        if pd.notna(raw_shares) and ROUND_TO_WHOLE_SHARES:
            estimated_shares = float(int(abs(raw_shares))) * (1.0 if raw_shares >= 0 else -1.0)
        else:
            estimated_shares = raw_shares

        rounded_dollar_delta = estimated_shares * px if pd.notna(estimated_shares) else float("nan")
        row["raw_estimated_shares"] = raw_shares
        row["estimated_shares"] = estimated_shares
        row["rounded_dollar_delta"] = rounded_dollar_delta
        row["reject_reason"] = ""
        rows.append(row)

        remaining -= signed_part_dollar

    return rows


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
    current_loaded, current_shares, current_cash_usd, nav, current_source = load_current_positions(
        prices_all,
        DEFAULT_NAV_USD,
    )

    tickers = sorted(set(target.index) | set(current_loaded.index))
    current = current_loaded.reindex(tickers).fillna(0.0)
    target = target.reindex(tickers).fillna(0.0)
    prices = prices_all.reindex(tickers)

    base_rows = []
    for ticker in tickers:
        current_weight = float(current.loc[ticker])
        target_weight = float(target.loc[ticker])
        delta_weight = target_weight - current_weight
        desired_dollar_delta = delta_weight * nav

        px = prices.loc[ticker] if ticker in prices.index else float("nan")
        raw_shares = desired_dollar_delta / px if pd.notna(px) and px > 0 else float("nan")

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
        elif abs(desired_dollar_delta) < MIN_ORDER_USD:
            action = "REJECT"
            reject_reason = "below_min_order_usd"
        elif ROUND_TO_WHOLE_SHARES and abs(estimated_shares) < 1:
            action = "REJECT"
            reject_reason = "rounds_to_zero_shares"
        elif delta_weight > 0:
            action = "BUY"
        else:
            action = "SELL"

        base_rows.append(
            {
                "preview_date": latest_date,
                "ticker": ticker,
                "action": action,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "delta_weight": delta_weight,
                "preview_nav_usd": nav,
                "current_cash_usd": current_cash_usd,
                "dollar_delta": desired_dollar_delta,
                "latest_price": px,
                "raw_estimated_shares": raw_shares,
                "estimated_shares": estimated_shares,
                "rounded_dollar_delta": rounded_dollar_delta,
                "reject_reason": reject_reason,
                "order_part": 1,
                "order_part_count": 1,
            }
        )

    rows = []
    for row in base_rows:
        if row["action"] in {"BUY", "SELL"}:
            rows.extend(split_order_rows(row))
        else:
            rows.append(row)

    out = pd.DataFrame(rows).sort_values(["action", "dollar_delta"], ascending=[True, False])

    actionable = out[out["action"].isin(["BUY", "SELL"])].copy()
    rejected = out[out["action"].eq("REJECT")].copy()

    def recompute_cash(frame: pd.DataFrame) -> tuple[float, float, float, float]:
        active = frame[frame["action"].isin(["BUY", "SELL"])].copy()
        buy = float(active[active["action"].eq("BUY")]["rounded_dollar_delta"].sum())
        sell = float(active[active["action"].eq("SELL")]["rounded_dollar_delta"].sum())
        net = buy + sell
        cash_after = current_cash_usd - net
        return buy, sell, net, cash_after

    cash_adjustments = []

    buy_total, sell_total, net_cash_impact, preview_cash_after_orders = recompute_cash(out)

    # Cash safety: reduce BUY orders one share at a time until cash buffer is respected.
    # This is preview-only and avoids creating dry-run orders that would overspend available cash.
    if preview_cash_after_orders < MIN_CASH_BUFFER_USD:
        buy_idx = list(
            out[out["action"].eq("BUY")]
            .sort_values("rounded_dollar_delta", ascending=True)
            .index
        )

        changed = True
        while preview_cash_after_orders < MIN_CASH_BUFFER_USD and changed:
            changed = False
            for idx in buy_idx:
                if preview_cash_after_orders >= MIN_CASH_BUFFER_USD:
                    break

                shares = float(out.at[idx, "estimated_shares"])
                px = float(out.at[idx, "latest_price"])

                if shares <= 1 or px <= 0:
                    continue

                out.at[idx, "estimated_shares"] = shares - 1
                out.at[idx, "raw_estimated_shares"] = shares - 1
                out.at[idx, "rounded_dollar_delta"] = (shares - 1) * px

                cash_adjustments.append(
                    {
                        "ticker": out.at[idx, "ticker"],
                        "removed_one_share_at_price": px,
                    }
                )

                buy_total, sell_total, net_cash_impact, preview_cash_after_orders = recompute_cash(out)
                changed = True

        # Reject any BUY that was reduced to zero.
        zero_buy_mask = (out["action"].eq("BUY")) & (out["estimated_shares"].abs() < 1)
        out.loc[zero_buy_mask, "action"] = "REJECT"
        out.loc[zero_buy_mask, "reject_reason"] = "cash_buffer_reduced_to_zero"

    actionable = out[out["action"].isin(["BUY", "SELL"])].copy()
    rejected = out[out["action"].eq("REJECT")].copy()
    buy_total, sell_total, net_cash_impact, preview_cash_after_orders = recompute_cash(out)

    order_path = OUTPUT_DIR / "paper_order_preview.csv"
    report_path = OUTPUT_DIR / "paper_order_preview_report.txt"
    out.to_csv(order_path, index=False)

    lines = [
        "Paper Order Preview Report",
        "",
        "status: preview_only_no_orders_sent",
        "candidate: v3_lower_turnover",
        f"target_allocation_date: {latest_date}",
        f"preview_nav_usd: {nav:.2f}",
        f"current_position_source: {current_source}",
        f"current_cash_usd: {current_cash_usd:.2f}",
        f"min_order_usd: {MIN_ORDER_USD:.2f}",
        f"max_order_usd: {MAX_ORDER_USD:.2f}",
        f"round_to_whole_shares: {ROUND_TO_WHOLE_SHARES}",
        f"min_cash_buffer_usd: {MIN_CASH_BUFFER_USD:.2f}",
        f"cash_adjustment_count: {len(cash_adjustments)}",
        f"rounded_buy_total: {buy_total:.2f}",
        f"rounded_sell_total: {sell_total:.2f}",
        f"net_cash_impact: {net_cash_impact:.2f}",
        f"preview_cash_after_orders: {preview_cash_after_orders:.2f}",
        "",
        "important:",
        "- This report does not send simulator or broker orders.",
        "- This report uses config/paper_positions.csv when present.",
        "- Orders above max_order_usd are split into multiple preview parts.",
        "- Next step is replacing config/paper_positions.csv with real simulator export.",
        "",
        "target_allocation:",
    ]

    for ticker, weight in target.sort_values(ascending=False).items():
        if abs(weight) > 1e-8:
            lines.append(f"- {ticker}: {weight:.4f}")

    lines.extend(["", "preview_orders:"])
    if actionable.empty:
        lines.append("- none")
    for row in actionable.itertuples(index=False):
        part = ""
        if int(row.order_part_count) > 1:
            part = f" part={int(row.order_part)}/{int(row.order_part_count)}"
        lines.append(
            f"- {row.action} {row.ticker}{part}: "
            f"delta_weight={row.delta_weight:+.4f}, "
            f"dollar_delta={row.dollar_delta:+.2f}, "
            f"price={row.latest_price:.4f}, "
            f"shares={row.estimated_shares:.0f}, "
            f"rounded_dollar_delta={row.rounded_dollar_delta:+.2f}"
        )

    lines.extend(["", "cash_safety_adjustments:"])
    if not cash_adjustments:
        lines.append("- none")
    else:
        by_ticker = {}
        for item in cash_adjustments:
            by_ticker[item["ticker"]] = by_ticker.get(item["ticker"], 0) + 1
        for ticker, count in sorted(by_ticker.items()):
            lines.append(f"- {ticker}: reduced_buy_shares={count}")

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
