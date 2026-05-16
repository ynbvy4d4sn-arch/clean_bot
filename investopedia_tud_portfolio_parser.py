"""Parse Investopedia TUD game portfolio snapshot.

Read-only. Does not submit orders.

Input:
- outputs/investopedia_tud_navigator/tud_portfolio.txt

Outputs:
- config/paper_positions.csv
- outputs/investopedia_tud_portfolio_parsed.csv
- outputs/investopedia_tud_portfolio_report.txt
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


ROOT = Path(__file__).resolve().parent
SNAPSHOT_TXT = ROOT / "outputs" / "investopedia_tud_navigator" / "tud_portfolio.txt"
POSITIONS_OUT = ROOT / "config" / "paper_positions.csv"
PARSED_OUT = ROOT / "outputs" / "investopedia_tud_portfolio_parsed.csv"
REPORT_OUT = ROOT / "outputs" / "investopedia_tud_portfolio_report.txt"


TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def money_to_float(text: str) -> float:
    cleaned = (
        text.replace("$", "")
        .replace(",", "")
        .replace("%", "")
        .replace("+", "")
        .replace("−", "-")
        .strip()
    )
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_snapshot_text(text: str) -> tuple[dict, pd.DataFrame]:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    meta = {
        "account_value": None,
        "buying_power": None,
        "cash": None,
        "total_stocks_etfs_value": None,
    }

    for i, line in enumerate(lines):
        upper = line.upper()
        if upper == "ACCOUNT VALUE" and i + 1 < len(lines):
            meta["account_value"] = money_to_float(lines[i + 1])
        elif upper == "BUYING POWER" and i + 1 < len(lines):
            meta["buying_power"] = money_to_float(lines[i + 1])
        elif upper == "CASH" and i + 1 < len(lines):
            meta["cash"] = money_to_float(lines[i + 1])
        elif upper == "TOTAL VALUE" and i > 0 and lines[i - 1].upper() == "STOCKS & ETFS" and i + 1 < len(lines):
            meta["total_stocks_etfs_value"] = money_to_float(lines[i + 1])

    rows = []

    try:
        start = next(i for i, line in enumerate(lines) if line.upper() == "STOCKS & ETFS")
        end_candidates = [i for i, line in enumerate(lines[start + 1 :], start + 1) if line.upper() == "TRADE HISTORY"]
        end = end_candidates[0] if end_candidates else len(lines)
    except StopIteration:
        return meta, pd.DataFrame(columns=["ticker", "shares", "current_price", "total_value"])

    section = lines[start:end]

    for idx, line in enumerate(section):
        ticker = line.upper().strip()
        if not TICKER_RE.match(ticker):
            continue

        # Skip table labels accidentally matching ticker pattern.
        if ticker in {
            "SYMBOL",
            "DESCRIPTION",
            "QTY",
            "BUY",
            "SELL",
            "ETFS",
            "ETF",
            "STOCKS",
            "TOTAL",
            "VALUE",
        }:
            continue

        # Investopedia text layout after each ticker:
        # ticker
        # description
        # current price
        # today's change
        # (...)
        # purchase price
        # qty
        # total value
        # total gain/loss
        #
        # We search the next ~16 lines for money values and integer qty.
        window = section[idx + 1 : idx + 18]

        money_positions = []
        qty_candidates = []

        for j, item in enumerate(window):
            if item.startswith("$") or item.startswith("-$") or item.startswith("+$"):
                money_positions.append((j, item))

            if re.fullmatch(r"\d+", item):
                qty_candidates.append((j, int(item)))

        if len(money_positions) < 3 or not qty_candidates:
            continue

        # In the observed layout:
        # first money = current price
        # third money = purchase price
        # qty follows purchase price
        current_price = money_to_float(money_positions[0][1])

        qty = None
        qty_pos = None
        for qpos, qval in qty_candidates:
            # Usually after purchase price, before total value.
            if qpos > money_positions[1][0]:
                qty = qval
                qty_pos = qpos
                break

        if qty is None:
            qty_pos, qty = qty_candidates[0]

        total_value = None
        for mpos, mval in money_positions:
            if mpos > qty_pos:
                total_value = money_to_float(mval)
                break

        if total_value is None:
            total_value = current_price * qty

        rows.append(
            {
                "ticker": ticker,
                "shares": int(qty),
                "current_price": current_price,
                "total_value": total_value,
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker"], keep="first")
    return meta, df


def main() -> None:
    if not SNAPSHOT_TXT.exists():
        raise SystemExit(f"Missing snapshot: {SNAPSHOT_TXT}")

    text = SNAPSHOT_TXT.read_text(encoding="utf-8", errors="ignore")
    meta, positions = parse_snapshot_text(text)

    if positions.empty:
        raise SystemExit("No positions parsed from Investopedia snapshot.")

    cash = float(meta.get("cash") or 0.0)

    paper_rows = [{"ticker": "CASH", "shares": cash}]
    for row in positions.itertuples(index=False):
        paper_rows.append({"ticker": row.ticker, "shares": row.shares})

    POSITIONS_OUT.parent.mkdir(exist_ok=True)
    PARSED_OUT.parent.mkdir(exist_ok=True)

    pd.DataFrame(paper_rows).to_csv(POSITIONS_OUT, index=False)
    positions.to_csv(PARSED_OUT, index=False)

    positions_value = float(positions["total_value"].sum())
    inferred_nav = cash + positions_value

    lines = [
        "Investopedia TUD Portfolio Parse Report",
        "",
        "status: parsed_read_only",
        "orders_were_submitted: false",
        f"snapshot: {SNAPSHOT_TXT}",
        f"account_value: {meta.get('account_value')}",
        f"buying_power: {meta.get('buying_power')}",
        f"cash: {cash:.2f}",
        f"positions_value_sum: {positions_value:.2f}",
        f"inferred_nav_cash_plus_positions: {inferred_nav:.2f}",
        "",
        "positions:",
    ]

    for row in positions.itertuples(index=False):
        lines.append(
            f"- {row.ticker}: shares={row.shares}, "
            f"current_price={row.current_price:.2f}, "
            f"total_value={row.total_value:.2f}"
        )

    lines.extend(
        [
            "",
            f"wrote_positions_csv: {POSITIONS_OUT}",
            f"wrote_parsed_csv: {PARSED_OUT}",
        ]
    )

    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_OUT.read_text())


if __name__ == "__main__":
    main()
