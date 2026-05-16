from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NAV_DIR = ROOT / "outputs" / "investopedia_tud_navigator"

JSON_TABLES = [
    NAV_DIR / "tud_holdings_live_tables.json",
    NAV_DIR / "after_portfolio_click_tables.json",
]

OUT_CSV = ROOT / "outputs" / "investopedia_tud_portfolio_parsed.csv"
OUT_REPORT = ROOT / "outputs" / "investopedia_tud_portfolio_report.txt"
PAPER_POSITIONS = ROOT / "config" / "paper_positions.csv"

KNOWN_TICKERS = {
    "AGG", "IEF", "PDBC", "SGOV", "SHY", "SPMO", "TIP", "XLE", "XLK", "XLP",
    "XLC", "XLV", "SLV",
}


def clean(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def money_to_float(s: object) -> float | None:
    v = clean(s).replace(",", "").replace("−", "-")
    m = re.search(r"(-?)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", v)
    if not m:
        return None
    return (-1 if m.group(1) == "-" else 1) * float(m.group(2))


def num_to_float(s: object) -> float | None:
    v = clean(s).replace(",", "")
    if not re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", v):
        return None
    return float(v)


def parse_rows_from_json_tables() -> list[dict]:
    parsed: list[dict] = []

    for path in JSON_TABLES:
        if not path.exists():
            continue

        data = json.loads(path.read_text(errors="ignore"))

        # Current format: list of tables, each with rows.
        if isinstance(data, dict):
            tables = data.get("tables", [])
        else:
            tables = data

        for table in tables:
            rows = table.get("rows", []) if isinstance(table, dict) else []
            if not rows:
                continue

            header = [clean(x).lower() for x in rows[0]]
            required = ["symbol", "current price", "purchase price", "qty", "total value", "total gain/loss"]

            if not all(x in header for x in required):
                continue

            idx = {col: header.index(col) for col in required}

            for cells in rows[1:]:
                if len(cells) <= max(idx.values()):
                    continue

                ticker = clean(cells[idx["symbol"]]).upper()
                if ticker not in KNOWN_TICKERS:
                    continue

                shares = num_to_float(cells[idx["qty"]])
                total_value = money_to_float(cells[idx["total value"]])

                if shares is None or total_value is None:
                    continue

                parsed.append({
                    "ticker": ticker,
                    "shares": shares,
                    "current_price": money_to_float(cells[idx["current price"]]),
                    "purchase_price": money_to_float(cells[idx["purchase price"]]),
                    "total_value": total_value,
                    "gain_loss": money_to_float(cells[idx["total gain/loss"]]),
                    "source": path.name,
                    "raw": " | ".join(clean(x) for x in cells),
                })

    return dedupe(parsed)


def dedupe(rows: list[dict]) -> list[dict]:
    out = {}
    for row in rows:
        out[row["ticker"]] = row
    return [out[t] for t in sorted(out)]


def write_outputs(rows: list[dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    PAPER_POSITIONS.parent.mkdir(parents=True, exist_ok=True)

    parsed_fields = [
        "ticker", "shares", "current_price", "purchase_price",
        "total_value", "gain_loss", "source", "raw",
    ]

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=parsed_fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Only overwrite live positions when rows are valid.
    if rows:
        with PAPER_POSITIONS.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["ticker", "shares", "current_price", "purchase_price", "total_value", "gain_loss"],
            )
            w.writeheader()
            for r in rows:
                shares = r["shares"]
                if float(shares).is_integer():
                    shares = int(shares)
                w.writerow({
                    "ticker": r["ticker"],
                    "shares": shares,
                    "current_price": r["current_price"],
                    "purchase_price": r["purchase_price"],
                    "total_value": r["total_value"],
                    "gain_loss": r["gain_loss"],
                })

    report = []
    report.append("Investopedia TUD Portfolio Parser Report")
    report.append("")
    report.append(f"parsed_rows: {len(rows)}")
    report.append("")
    report.append("positions:")

    for r in rows:
        report.append(
            f"- {r['ticker']}: shares={r['shares']} current_price={r['current_price']} "
            f"purchase_price={r['purchase_price']} total_value={r['total_value']} "
            f"gain_loss={r['gain_loss']} source={r['source']}"
        )

    if not rows:
        report.append("")
        report.append("ERROR: No positions parsed. config/paper_positions.csv was not overwritten.")

    OUT_REPORT.write_text("\n".join(report) + "\n")


def main() -> None:
    rows = parse_rows_from_json_tables()
    write_outputs(rows)

    if not rows:
        print("ERROR: No positions parsed from live table JSON.")
        print(f"Report: {OUT_REPORT}")
        sys.exit(1)

    print(f"Parsed {len(rows)} Investopedia holdings.")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {PAPER_POSITIONS}")


if __name__ == "__main__":
    main()
