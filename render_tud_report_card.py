from __future__ import annotations

from pathlib import Path
from datetime import datetime
from string import Template
import base64
import math
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
ASSETS = ROOT / "assets"
DATA_INV = ROOT / "data" / "investopedia"

POSITIONS_CSV = OUTPUTS / "investopedia_tud_portfolio_parsed.csv"
ORDER_CSV = OUTPUTS / "paper_order_preview.csv"
PORTFOLIO_REPORT = OUTPUTS / "investopedia_tud_portfolio_report.txt"

OUT_HTML = OUTPUTS / "tud_daily_report_card.html"
OUT_PNG = OUTPUTS / "tud_daily_report_card.png"
AUDIT_OUT = OUTPUTS / "tud_report_data_audit.txt"

START_DATE = "2026-04-24"
START_TS = pd.Timestamp(START_DATE)
START_NAV = 100000.0
RISK_FREE_RATE_ANNUAL = 0.02
PERIODS_PER_YEAR_DAILY = 252


def asset_b64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def money_plain(x) -> str:
    try:
        if pd.isna(x):
            return "-"
        return f"${float(x):,.2f}"
    except Exception:
        return "-"


def money_signed(x) -> str:
    try:
        if pd.isna(x):
            return "-"
        value = float(x)
        sign = "+" if value > 0 else ""
        return f"{sign}${value:,.2f}"
    except Exception:
        return "-"


def pct_plain(x) -> str:
    try:
        if pd.isna(x):
            return "-"
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return "-"


def pct_signed(x) -> str:
    try:
        if pd.isna(x):
            return "-"
        value = float(x)
        sign = "+" if value > 0 else ""
        return f"{sign}{100.0 * value:.2f}%"
    except Exception:
        return "-"


def num(x) -> str:
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):,.0f}"
    except Exception:
        return "-"


def clean_money(x) -> float:
    s = str(x or "")
    s = s.replace("$", "").replace(",", "").replace("+", "").replace("−", "-")
    s = re.sub(r"[()%]", "", s).strip()
    try:
        return float(s)
    except Exception:
        return np.nan


def read_meta() -> dict:
    meta = {}
    if not PORTFOLIO_REPORT.exists():
        return meta

    text = PORTFOLIO_REPORT.read_text(encoding="utf-8", errors="ignore")
    for key in ["account_value", "cash", "buying_power", "positions_value_sum", "inferred_nav_cash_plus_positions"]:
        m = re.search(rf"^{key}:\s*([0-9.,-]+)", text, flags=re.MULTILINE)
        if m:
            meta[key] = float(m.group(1).replace(",", ""))
    return meta


def load_positions():
    path = ROOT / "config" / "paper_positions.csv"
    df = pd.read_csv(path)

    cols = {c.lower().strip(): c for c in df.columns}

    # Normalize ticker/symbol.
    if "ticker" in cols:
        df = df.rename(columns={cols["ticker"]: "ticker"})
    elif "symbol" in cols:
        df = df.rename(columns={cols["symbol"]: "ticker"})
    else:
        raise ValueError(f"positions file {path} must contain ticker or symbol column; got {list(df.columns)}")

    # Normalize shares/quantity.
    cols = {c.lower().strip(): c for c in df.columns}
    if "shares" in cols:
        df = df.rename(columns={cols["shares"]: "shares"})
    elif "quantity" in cols:
        df = df.rename(columns={cols["quantity"]: "shares"})
    else:
        raise ValueError(f"positions file {path} must contain shares or quantity column; got {list(df.columns)}")

    # Normalize value columns.
    cols = {c.lower().strip(): c for c in df.columns}

    if "total_value" in cols:
        df = df.rename(columns={cols["total_value"]: "total_value"})
    elif "market_value" in cols:
        df = df.rename(columns={cols["market_value"]: "total_value"})
    elif "value" in cols:
        df = df.rename(columns={cols["value"]: "total_value"})

    if "current_price" in cols:
        df = df.rename(columns={cols["current_price"]: "current_price"})
    elif "price" in cols:
        df = df.rename(columns={cols["price"]: "current_price"})

    if "gain_loss" in cols:
        df = df.rename(columns={cols["gain_loss"]: "gain_loss"})

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)

    if "current_price" in df.columns:
        df["current_price"] = pd.to_numeric(df["current_price"], errors="coerce")
    else:
        df["current_price"] = pd.NA

    if "total_value" in df.columns:
        df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
    else:
        # Fallback if only shares + current_price exist.
        df["total_value"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0) * pd.to_numeric(df["current_price"], errors="coerce").fillna(0)

    if "gain_loss" in df.columns:
        df["gain_loss"] = pd.to_numeric(df["gain_loss"], errors="coerce")
    else:
        df["gain_loss"] = pd.NA

    total = float(pd.to_numeric(df["total_value"], errors="coerce").fillna(0).sum())
    if total > 0:
        df["weight"] = pd.to_numeric(df["total_value"], errors="coerce").fillna(0) / total
    else:
        df["weight"] = 0.0

    return df


def load_orders() -> pd.DataFrame:
    if not ORDER_CSV.exists():
        return pd.DataFrame(columns=["action", "ticker", "shares", "volume"])

    df = pd.read_csv(ORDER_CSV)
    df["action"] = df["action"].astype(str).str.upper()
    df = df[df["action"].isin(["BUY", "SELL"])].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["shares"] = pd.to_numeric(df["estimated_shares"], errors="coerce").abs().fillna(0).round().astype(int)

    col = "rounded_dollar_delta" if "rounded_dollar_delta" in df.columns else "dollar_delta"
    df["volume"] = pd.to_numeric(df[col], errors="coerce").abs().fillna(0.0)

    # Wichtig: gesplittete Orders, z.B. SGOV part=1/2 + part=2/2, für den Report zusammenfassen.
    grouped = (
        df.groupby(["action", "ticker"], as_index=False)
        .agg({"shares": "sum", "volume": "sum"})
        .sort_values(["action", "volume"], ascending=[True, False])
    )
    return grouped[["action", "ticker", "shares", "volume"]].copy()


def find_trade_history() -> Path | None:
    DATA_INV.mkdir(parents=True, exist_ok=True)
    candidates = []
    for base in [DATA_INV, Path.home() / "Downloads", ROOT]:
        if base.exists():
            candidates.extend(base.glob("trade-history-*.xls"))
            candidates.extend(base.glob("trade-history-*.html"))
            candidates.extend(base.glob("trade-history-*.csv"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def load_trade_history() -> pd.DataFrame:
    path = find_trade_history()
    if not path:
        return pd.DataFrame(columns=["datetime", "ticker", "side", "qty", "price", "notional"])

    try:
        if path.suffix.lower() == ".csv":
            raw = pd.read_csv(path)
        else:
            raw = pd.read_html(path)[0]
    except Exception:
        return pd.DataFrame(columns=["datetime", "ticker", "side", "qty", "price", "notional"])

    if raw.empty:
        return pd.DataFrame(columns=["datetime", "ticker", "side", "qty", "price", "notional"])

    cols = {str(c).lower().strip(): c for c in raw.columns}
    date_col = cols.get("date") or cols.get("time") or cols.get("datetime")
    ticker_col = cols.get("symbol") or cols.get("ticker")
    side_col = cols.get("trade type") or cols.get("type") or cols.get("side")
    qty_col = cols.get("quantity") or cols.get("qty")
    price_col = cols.get("price")

    if not all([date_col, ticker_col, side_col, qty_col, price_col]):
        return pd.DataFrame(columns=["datetime", "ticker", "side", "qty", "price", "notional"])

    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(raw[date_col], errors="coerce")
    out["ticker"] = raw[ticker_col].astype(str).str.upper().str.strip()
    out["side_raw"] = raw[side_col].astype(str).str.lower()
    out["qty"] = raw[qty_col].map(clean_money)
    out["price"] = raw[price_col].map(clean_money)
    out["side"] = np.where(out["side_raw"].str.contains("buy"), "BUY", np.where(out["side_raw"].str.contains("sell"), "SELL", ""))

    out = out.dropna(subset=["datetime", "qty", "price"])
    out = out[(out["ticker"] != "") & (out["qty"] > 0) & (out["price"] > 0) & (out["side"] != "")]
    out["notional"] = out["qty"] * out["price"]

    try:
        out["datetime"] = out["datetime"].dt.tz_localize(None)
    except Exception:
        pass

    # Daily chart: date-only trades stay on the trade date.

    return out[["datetime", "ticker", "side", "qty", "price", "notional"]].sort_values("datetime")


def add_cost_basis_pnl(positions: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    df = positions.copy()
    if df.empty:
        return df

    cost = {}
    qty = {}

    for row in trades.itertuples(index=False):
        t = row.ticker
        q = float(row.qty)
        n = float(row.notional)

        if row.side == "BUY":
            qty[t] = qty.get(t, 0.0) + q
            cost[t] = cost.get(t, 0.0) + n
        elif row.side == "SELL":
            old_qty = qty.get(t, 0.0)
            old_cost = cost.get(t, 0.0)
            if old_qty > 0:
                avg = old_cost / old_qty
                sold = min(q, old_qty)
                qty[t] = old_qty - sold
                cost[t] = old_cost - avg * sold

    df["cost_basis"] = df["ticker"].map(cost)
    df["cost_basis"] = pd.to_numeric(df["cost_basis"], errors="coerce")
    df["abs_pnl"] = df["total_value"] - df["cost_basis"]
    df["rel_pnl"] = np.where(df["cost_basis"] > 0, df["abs_pnl"] / df["cost_basis"], np.nan)

    # yfinance auto_adjust=True sorgt im Chart für Dividenden-/Ausschüttungsadjustierung.
    # Die Tabellen-P/L nutzt Cost-Basis + aktuellen Marktwert; exakte Cash-Ausschüttungen aus Investopedia sind separat nicht verfügbar.
    df["abs_pnl"] = df["abs_pnl"].fillna(0.0)
    df["rel_pnl"] = df["rel_pnl"].fillna(0.0)
    return df


def fetch_hourly_prices(tickers: list[str]) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame()

    tickers = sorted(set(tickers + ["SPY"]))
    try:
        data = yf.download(
            tickers,
            start=START_DATE,
            interval="1d",
            auto_adjust=True,  # Adjusted for splits/distributions; better total-return proxy than raw close.
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    rows = []
    for ticker in tickers:
        try:
            close = data[(ticker, "Close")].dropna()
            temp = close.reset_index()
            temp.columns = ["datetime", "close"]
            temp["ticker"] = ticker
            rows.append(temp)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    try:
        out["datetime"] = out["datetime"].dt.tz_localize(None)
    except Exception:
        pass
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["datetime", "close"]).sort_values(["datetime", "ticker"])


def build_curves_from_trades(positions: pd.DataFrame, trades: pd.DataFrame, account_value: float) -> tuple[pd.DataFrame, dict]:
    tickers = sorted(set(positions["ticker"].tolist()) | set(trades["ticker"].tolist()))
    prices = fetch_hourly_prices(tickers)

    if prices.empty:
        curve = pd.DataFrame(
            [
                {"datetime": START_TS, "nav": START_NAV, "portfolio_return_pct": 0.0, "spy_return_pct": 0.0},
                {"datetime": pd.Timestamp.now(), "nav": account_value, "portfolio_return_pct": account_value / START_NAV - 1.0, "spy_return_pct": np.nan},
            ]
        )
        return curve, calc_stats(curve)

    wide = prices.pivot_table(index="datetime", columns="ticker", values="close", aggfunc="last").sort_index().ffill()
    if wide.empty:
        curve = pd.DataFrame(
            [
                {"datetime": START_TS, "nav": START_NAV, "portfolio_return_pct": 0.0, "spy_return_pct": 0.0},
                {"datetime": pd.Timestamp.now(), "nav": account_value, "portfolio_return_pct": account_value / START_NAV - 1.0, "spy_return_pct": np.nan},
            ]
        )
        return curve, calc_stats(curve)

    cash = START_NAV
    holdings: dict[str, float] = {}
    nav_rows = []
    trade_rows = list(trades.sort_values("datetime").itertuples(index=False))
    ptr = 0

    for dt in wide.index:
        while ptr < len(trade_rows) and pd.Timestamp(trade_rows[ptr].datetime) <= dt:
            tr = trade_rows[ptr]
            if tr.side == "BUY":
                holdings[tr.ticker] = holdings.get(tr.ticker, 0.0) + float(tr.qty)
                cash -= float(tr.notional)
            elif tr.side == "SELL":
                holdings[tr.ticker] = holdings.get(tr.ticker, 0.0) - float(tr.qty)
                cash += float(tr.notional)
            ptr += 1

        nav = cash
        for ticker, q in holdings.items():
            if ticker in wide.columns and pd.notna(wide.at[dt, ticker]):
                nav += q * float(wide.at[dt, ticker])

        nav_rows.append({"datetime": dt, "raw_nav": nav})

    curve = pd.DataFrame(nav_rows)
    curve = curve[curve["raw_nav"] > 0].copy()

    if curve.empty:
        curve = pd.DataFrame(
            [
                {"datetime": START_TS, "nav": START_NAV, "portfolio_return_pct": 0.0, "spy_return_pct": 0.0},
                {"datetime": pd.Timestamp.now(), "nav": account_value, "portfolio_return_pct": account_value / START_NAV - 1.0, "spy_return_pct": np.nan},
            ]
        )
        return curve, calc_stats(curve)

    # Auf aktuellen Investopedia Account Value kalibrieren.
    curve["nav"] = curve["raw_nav"] * account_value / curve["raw_nav"].iloc[-1]

    # Prozentualer Report-Chart: 0% am Startkapital.
    curve["portfolio_return_pct"] = curve["nav"] / curve["nav"].iloc[0] - 1.0

    if "SPY" in wide.columns:
        spy = wide["SPY"].reindex(curve["datetime"]).ffill()
        curve["spy_return_pct"] = spy.values / spy.values[0] - 1.0
    else:
        curve["spy_return_pct"] = np.nan

    return curve[["datetime", "nav", "portfolio_return_pct", "spy_return_pct"]], calc_stats(curve)


def calc_stats(curve: pd.DataFrame) -> dict:
    if len(curve) < 2:
        return {"sharpe": np.nan, "rel_gain": np.nan, "ann_vol": np.nan, "abs_gain": np.nan}

    rel_gain = curve["nav"].iloc[-1] / START_NAV - 1.0
    abs_gain = curve["nav"].iloc[-1] - START_NAV

    returns = curve["nav"].pct_change().dropna()

    if len(returns) < 2 or returns.std(ddof=0) <= 0:
        return {
            "sharpe": np.nan,
            "rel_gain": rel_gain,
            "ann_vol": np.nan,
            "abs_gain": abs_gain,
        }

    rf_daily = (1.0 + RISK_FREE_RATE_ANNUAL) ** (1.0 / PERIODS_PER_YEAR_DAILY) - 1.0
    excess = returns - rf_daily

    ann_vol = returns.std(ddof=0) * math.sqrt(PERIODS_PER_YEAR_DAILY)
    sharpe = excess.mean() / excess.std(ddof=0) * math.sqrt(PERIODS_PER_YEAR_DAILY) if excess.std(ddof=0) > 0 else np.nan

    # Only hide obviously broken reconstruction artifacts.
    if pd.notna(sharpe) and abs(sharpe) > 8:
        sharpe = np.nan

    return {
        "sharpe": sharpe,
        "rel_gain": rel_gain,
        "ann_vol": ann_vol,
        "abs_gain": abs_gain,
    }


def chart_png_b64(curve: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mtick

    plot = curve.copy().sort_values("datetime")

    # Force visual chart basis to exactly 0.0% at first displayed point.
    if not plot.empty:
        plot["portfolio_return_pct"] = plot["portfolio_return_pct"] - plot["portfolio_return_pct"].iloc[0]
        if "spy_return_pct" in plot.columns and plot["spy_return_pct"].notna().any():
            first_spy = plot["spy_return_pct"].dropna().iloc[0]
            plot["spy_return_pct"] = plot["spy_return_pct"] - first_spy

    fig, ax = plt.subplots(figsize=(7.05, 3.90))

    ax.plot(
        plot["datetime"],
        plot["portfolio_return_pct"],
        color="#031845",
        linewidth=2.35,
        label="Butter Brezel Portfolio",
    )

    if "spy_return_pct" in plot.columns and plot["spy_return_pct"].notna().any():
        ax.plot(
            plot["datetime"],
            plot["spy_return_pct"],
            color="#1F6FD1",
            linewidth=1.85,
            label="S&P 500 / SPY",
        )

    ax.axhline(0.0, color="#8DA5C7", linewidth=0.9, alpha=0.7)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    ax.grid(True, alpha=0.20, linestyle="--", linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m."))

    ax.tick_params(axis="x", labelsize=8, colors="#20314F", rotation=0, pad=7)
    ax.tick_params(axis="y", labelsize=8, colors="#20314F", pad=5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.margins(x=0.025, y=0.16)

    path = OUTPUTS / "tud_curve_chart.png"
    fig.savefig(path, dpi=205, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + asset_b64(path)


def pie_png_b64(positions: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    df = positions.head(8)
    colors = ["#031845", "#0F5EA8", "#2B7BD0", "#5AA9E6", "#88C7F8", "#B8DBFF", "#9CB8D6", "#CAD3E0"]

    fig, ax = plt.subplots(figsize=(4.5, 4.3))
    ax.pie(
        df["weight"],
        colors=colors[: len(df)],
        startangle=90,
        autopct=(lambda pct: f"{pct:.1f}%" if pct >= 10 else ""),
        textprops={'color':'white','fontsize':10,'fontweight':'bold'},
        wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
    )
    ax.axis("equal")

    path = OUTPUTS / "tud_pie_chart.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + asset_b64(path)


def order_rows(df: pd.DataFrame) -> str:
    if df.empty:
        return "<tr><td colspan='3' class='muted'>none</td></tr>"
    rows = []
    for r in df.itertuples(index=False):
        rows.append(f"<tr><td>{r.ticker}</td><td>{num(r.shares)}</td><td>{money_plain(r.volume)}</td></tr>")
    return "\n".join(rows)


def perf_rows(df: pd.DataFrame) -> str:
    rows = []
    for r in df.head(10).itertuples(index=False):
        cls_abs = "pos" if r.abs_pnl >= 0 else "neg"
        cls_rel = "pos" if r.rel_pnl >= 0 else "neg"
        rows.append(
            "<tr>"
            f"<td><b>{r.ticker}</b></td>"
            f"<td>{num(r.shares)}</td>"
            f"<td>{money_plain(r.current_price)}</td>"
            f"<td>{money_plain(r.total_value)}</td>"
            f"<td>{pct_plain(r.weight)}</td>"
            f"<td class='{cls_abs}'>{money_signed(r.abs_pnl)}</td>"
            f"<td class='{cls_rel}'>{pct_signed(r.rel_pnl)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_audit(curve: pd.DataFrame, trades: pd.DataFrame, stats: dict, orders: pd.DataFrame) -> None:
    lines = [
        "TUD Report Data Audit",
        "",
        f"trade_rows: {len(trades)}",
        f"curve_rows_daily: {len(curve)}",
        f"curve_first: {curve['datetime'].min() if not curve.empty else 'none'}",
        f"curve_last: {curve['datetime'].max() if not curve.empty else 'none'}",
        f"risk_free_rate_annual: {RISK_FREE_RATE_ANNUAL:.4f}",
        f"daily_volatility_annualized: {stats.get('ann_vol')}",
        f"sharpe_2pct_rf_daily: {stats.get('sharpe')}",
        "",
        "aggregated_orders:",
    ]
    if orders.empty:
        lines.append("- none")
    else:
        for row in orders.itertuples(index=False):
            lines.append(f"- {row.action} {row.ticker}: shares={row.shares}, volume={row.volume:.2f}")
    AUDIT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_audit(curve: pd.DataFrame, trades: pd.DataFrame, stats: dict, orders: pd.DataFrame) -> None:
    audit_path = OUTPUTS / "tud_report_data_audit.txt"

    median_step = np.nan
    if len(curve) >= 3:
        deltas = curve["datetime"].sort_values().diff().dropna().dt.total_seconds() / 86400.0
        if not deltas.empty:
            median_step = float(deltas.median())

    lines = [
        "TUD Report Data Audit",
        "",
        "chart_source: trade_history_plus_yfinance_daily_adjusted",
        "yfinance_interval: 1d",
        "yfinance_auto_adjust: true",
        f"trade_rows: {len(trades)}",
        f"curve_rows_daily: {len(curve)}",
        f"curve_first: {curve['datetime'].min() if not curve.empty else 'none'}",
        f"curve_last: {curve['datetime'].max() if not curve.empty else 'none'}",
        f"median_step_days: {median_step}",
        f"risk_free_rate_annual: {RISK_FREE_RATE_ANNUAL:.4f}",
        f"ann_vol: {stats.get('ann_vol')}",
        f"sharpe_2pct_rf_daily: {stats.get('sharpe')}",
        "",
        "aggregated_orders:",
    ]

    if orders.empty:
        lines.append("- none")
    else:
        for row in orders.itertuples(index=False):
            lines.append(f"- {row.action} {row.ticker}: shares={row.shares}, volume={row.volume:.2f}")

    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_html() -> str:
    OUTPUTS.mkdir(exist_ok=True)

    meta = read_meta()
    positions = load_positions()
    orders = load_orders()
    trades = load_trade_history()
    positions = add_cost_basis_pnl(positions, trades)

    account_value = float(meta.get("account_value") or meta.get("inferred_nav_cash_plus_positions") or positions["total_value"].sum() or START_NAV)
    curve, stats = build_curves_from_trades(positions, trades, account_value)
    write_report_audit(curve, trades, stats, orders)
    write_audit(curve, trades, stats, orders)

    logo_b64 = ""
    for p in [ASSETS / "butter_brezel_logo.png", ASSETS / "butter_brezel_header.png", ASSETS / "butter_brezel.png"]:
        logo_b64 = asset_b64(p)
        if logo_b64:
            break

    logo_html = f"<img class='brand-logo' src='data:image/png;base64,{logo_b64}'>" if logo_b64 else "<div class='brand-name'>Butter Brezel</div>"

    pie_src = pie_png_b64(positions) if not positions.empty else ""
    curve_src = chart_png_b64(curve)

    top = positions.head(8)
    legend = "\n".join(
        f"<div class='legend-row'><span><i class='dot d{i}'></i>{r.ticker}</span><b>{pct_plain(r.weight)}</b></div>"
        for i, r in enumerate(top.itertuples(index=False))
    )

    buy = orders[orders["action"] == "BUY"]
    sell = orders[orders["action"] == "SELL"]

    stable_rel_gain = account_value / START_NAV - 1.0
    stable_abs_gain = account_value - START_NAV

    sharpe = f"{stats['sharpe']:.3f}" if pd.notna(stats["sharpe"]) else "—"
    rel_gain = pct_plain(stable_rel_gain)
    vol = pct_plain(stats["ann_vol"])
    abs_gain = money_signed(stable_abs_gain)
    today = datetime.now().strftime("%d.%m.%Y")

    template = Template(r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
:root {
  --navy:#031845;
  --navy2:#031845;
  --blue:#031845;
  --blue2:#1F6FD1;
  --soft:#EEF6FF;
  --line:#CFE0F4;
  --text:#12213B;
  --muted:#687893;
  --green:#07833D;
  --red:#D21414;
}
* { box-sizing:border-box; }
body {
  margin:0;
  background:#E8F0FA;
  font-family: Inter, Arial, Helvetica, sans-serif;
  color:var(--text);
}
.page {
  width:1055px;
  min-height:1490px;
  margin:0 auto;
  padding:12px;
  background:#F3F7FC;
}


.header {
  height:170px;
  border-radius:14px 14px 0 0;
  padding:0 34px 0 42px;
  position:relative;
  overflow:hidden;
  color:white;
  background:#031845 !important;
  display:grid;
  grid-template-columns:minmax(0, 1fr) 260px;
  align-items:center;
  gap:20px;
}




.header-title h1 {
  font-size:44px;
  line-height:1.02;
  margin:0;
  padding:0;
  color:#ffffff !important;
  letter-spacing:.01em;
  font-family: Georgia, 'Times New Roman', serif;
  font-weight:700;
  white-space:nowrap;
  overflow:visible;
}


.subtitle { display:none; }
.header-brand {
  display:flex;
  justify-content:flex-end;
  align-items:center;
  width:100%;
  height:100%;
}
.brand-stack {
  width:250px;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  transform:translateX(8px);
}

.brand-date { display:none; }

.brand-logo {
  width:235px;
  max-height:135px;
  object-fit:contain;
  object-position:center center;
  border-radius:0;
  display:block;
}
.brand-name {
  font-size:42px;
  font-weight:900;
  font-family: Georgia, 'Times New Roman', serif;
}
.date { display:none; }


.metrics {
  display:grid;
  grid-template-columns:repeat(4, minmax(0, 1fr));
  gap:16px;
  margin:18px 18px 18px 18px;
  align-items:stretch;
}












.metric {
  width:100%;
  height:108px;
  min-height:108px;
  box-sizing:border-box;
  background:white;
  border:1px solid var(--line);
  border-radius:18px;
  padding:18px 16px 16px 16px;
  box-shadow:0 8px 20px rgba(6,41,92,.08);
  display:flex;
  flex-direction:column;
  justify-content:center;
}


.metric-label {
  color:var(--navy);
  font-size:13px;
  font-weight:900;
  margin-bottom:15px;
}
.metric-value {
  font-size:30px;
  line-height:1;
  font-weight:900;
  color:var(--green);
}
.metric-value.blue { color:var(--navy); }
.grid2 {
  display:grid;
  grid-template-columns:repeat(4, minmax(0, 1fr));
  gap:16px;
  margin:0 18px 18px 18px;
  align-items:stretch;
}
.grid2 > *:first-child {
  grid-column:1 / span 2;
}
.grid2 > *:last-child {
  grid-column:3 / span 2;
}

.card {
  background:white;
  border:1px solid var(--line);
  border-radius:18px;
  box-shadow:0 8px 22px rgba(6,41,92,.07);
  padding:20px;
}
.card h2 {
  margin:0 0 17px 0;
  color:var(--navy);
  font-size:23px;
  font-weight:900;
}
.pie-layout {
  display:grid;
  grid-template-columns:52% 48%;
  align-items:center;
  gap:8px;
}

.pie-img { width:100%; }
.legend-row {
  display:flex;
  justify-content:flex-start;
  gap:12px;
  align-items:center;
  margin:11px 0;
  font-size:14px;
  color:var(--navy);
}
.legend-row span {
  display:flex;
  align-items:center;
  font-weight:900;
}
.dot {
  display:inline-block;
  width:13px;
  height:13px;
  border-radius:50%;
  margin-right:10px;
}
.d0{background:#031845}.d1{background:#0F5EA8}.d2{background:#2B7BD0}.d3{background:#5AA9E6}
.d4{background:#88C7F8}.d5{background:#B8DBFF}.d6{background:#9CB8D6}.d7{background:#CAD3E0}
.chart-img { width:100%; height:auto; }
.order-card { min-height:300px; }
.buy-title { color:var(--green) !important; }
.sell-title { color:var(--red) !important; }
table {
  width:100%;
  border-collapse:collapse;
  font-size:14.5px;
}
th {
  background:#031845 !important;
  background-color:#031845 !important;
  color:#ffffff !important;
  text-align:left;
  padding:10px 14px;
  font-size:13px;
  font-weight:700;
  border:none;
}
td {
  padding:11px 14px;
  border-bottom:1px solid #E4EFFB;
}
tbody tr:nth-child(even) td { background:#F2F8FF; }
.buy-table th {
  background:#031845 !important;
  background-color:#031845 !important;
  color:#ffffff !important;
}
.buy-table tbody tr:nth-child(even) td { background:#F0F8F2; }
.sell-table th {
  background:#031845 !important;
  background-color:#031845 !important;
  color:#ffffff !important;
}
.sell-table tbody tr:nth-child(even) td { background:#FFF1F1; }
.big-card { margin:0 18px 18px 18px; }
.pos { color:var(--green); font-weight:900; }
.neg { color:var(--red); font-weight:900; }
.footer {
  margin:0;
  height:64px;
  background:#031845 !important;
  background-color:#031845 !important;
  color:#DDEBFF;
  border-radius:0 0 14px 14px;
  padding:18px 38px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  font-size:15px;
}
.footer b {
  color:white;
  font-family:Georgia,'Times New Roman',serif;
  font-size:28px;
}
.muted { color:var(--muted); }



/* FINAL TABLE COLOR OVERRIDE ONLY */
table thead tr th,
table th,
.buy-table thead tr th,
.buy-table th,
.sell-table thead tr th,
.sell-table th {
  background:#031845 !important;
  background-color:#031845 !important;
  color:#ffffff !important;
}

</style>
</head>
<body>
<div class="page">
  <div class="header">
    <div class="header-title">
      <h1>Daily Trading Report</h1>
      
    </div>
    <div class="header-brand">
      <div class="brand-stack">
        $logo_html
        
      </div>
    </div></div>

  <div class="metrics">
    <div class="metric"><div class="metric-label">Sharpe Ratio</div><div class="metric-value blue">$sharpe</div></div>
    <div class="metric"><div class="metric-label">Return since 24.04</div><div class="metric-value">$rel_gain</div></div>
    <div class="metric"><div class="metric-label">Volatility p.a.</div><div class="metric-value blue">$vol</div></div>
    <div class="metric"><div class="metric-label">Absolute P/L</div><div class="metric-value">$abs_gain</div></div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Current Portfolio Weights</h2>
      <div class="pie-layout">
        <img class="pie-img" src="$pie_src">
        <div>$legend</div>
      </div>
    </div>

    <div class="card">
      <h2>Portfolio vs. S&P 500</h2>
      <img class="chart-img" src="$curve_src">
    </div>
  </div>

  <div class="grid2">
    <div class="card order-card">
      <h2 class="buy-title">Buy Orders</h2>
      <table class="buy-table">
        <thead><tr><th>Ticker</th><th>Shares</th><th>Volume</th></tr></thead>
        <tbody>$buy_rows</tbody>
      </table>
    </div>

    <div class="card order-card">
      <h2 class="sell-title">Sell Orders</h2>
      <table class="sell-table">
        <thead><tr><th>Ticker</th><th>Shares</th><th>Volume</th></tr></thead>
        <tbody>$sell_rows</tbody>
      </table>
    </div>
  </div>

  <div class="card big-card">
    <h2>Single-Security Performance</h2>
    <table>
      <thead>
        <tr>
          <th>Ticker</th><th>Shares</th><th>Last Price</th><th>Market Value</th>
          <th>Weight</th><th>Absolute P/L</th><th>Relative P/L</th>
        </tr>
      </thead>
      <tbody>$perf_rows</tbody>
    </table>
  </div>

  <div class="footer">
    <div>Butter Brezel Trading Research Report</div>
    <b>Butter Brezel</b>
  </div>
</div>
</body>
</html>
""")

    return template.safe_substitute(
        logo_html=logo_html,
        today=today,
        sharpe=sharpe,
        rel_gain=rel_gain,
        vol=vol,
        abs_gain=abs_gain,
        pie_src=pie_src,
        legend=legend,
        curve_src=curve_src,
        buy_rows=order_rows(buy),
        sell_rows=order_rows(sell),
        perf_rows=perf_rows(positions),
    )


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    html = build_html()
    OUT_HTML.write_text(html, encoding="utf-8")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1055, "height": 1491}, device_scale_factor=1)
        page.goto(OUT_HTML.resolve().as_uri(), wait_until="networkidle")
        page.screenshot(path=str(OUT_PNG), full_page=True)
        browser.close()

    print(f"wrote {OUT_HTML}")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {AUDIT_OUT}")


if __name__ == "__main__":
    main()
