"""Build blue visual HTML report for TUD dry-run emails."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
CONFIG = ROOT / "config"


BLUE = "#0B3A75"
BLUE_2 = "#0F5EA8"
BLUE_3 = "#EAF3FF"
BLUE_4 = "#F6FAFF"
TEXT = "#172033"
MUTED = "#5E6B7A"
BORDER = "#D6E6F8"


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _money(x) -> str:
    try:
        if pd.isna(x):
            return "-"
        return f"${float(x):,.2f}"
    except Exception:
        return "-"


def _pct(x) -> str:
    try:
        if pd.isna(x):
            return "-"
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return "-"


def _num(x) -> str:
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):,.0f}"
    except Exception:
        return "-"


def _html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class='muted'><i>keine</i></p>"
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


def load_current_positions() -> pd.DataFrame:
    parsed_path = OUTPUTS / "investopedia_tud_portfolio_parsed.csv"
    if parsed_path.exists():
        df = pd.read_csv(parsed_path)
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
        df["current_price"] = pd.to_numeric(df["current_price"], errors="coerce")
        df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce").fillna(0)
        nav = float(df["total_value"].sum())
        df["weight"] = np.where(nav > 0, df["total_value"] / nav, 0.0)
        return df

    path = CONFIG / "paper_positions.csv"
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "shares", "current_price", "total_value", "weight"])

    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
    df = df[df["ticker"] != "CASH"].copy()
    df["current_price"] = np.nan
    df["total_value"] = 0.0
    df["weight"] = 0.0
    return df


def load_orders() -> pd.DataFrame:
    path = OUTPUTS / "paper_order_preview.csv"
    if not path.exists():
        return pd.DataFrame(columns=["action", "ticker", "shares", "volume_usd"])

    df = pd.read_csv(path)
    df["action"] = df["action"].astype(str).str.upper()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["shares"] = pd.to_numeric(df["estimated_shares"], errors="coerce").abs().fillna(0).round().astype(int)

    if "rounded_dollar_delta" in df.columns:
        df["volume_usd"] = pd.to_numeric(df["rounded_dollar_delta"], errors="coerce").abs()
    else:
        df["volume_usd"] = pd.to_numeric(df["dollar_delta"], errors="coerce").abs()

    return df[df["action"].isin(["BUY", "SELL"])].copy()


def build_pie_chart(positions: pd.DataFrame) -> str:
    if positions.empty or "weight" not in positions.columns:
        return ""

    df = positions[positions["weight"] > 0.002].copy().sort_values("weight", ascending=False)
    if df.empty:
        return ""

    colors = [
        "#0B3A75", "#0F5EA8", "#2B7BD0", "#5AA9E6", "#88C7F8",
        "#B8DBFF", "#6C8EBF", "#1E4E8C", "#4F83CC", "#9CC9F5",
    ]

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.pie(
        df["weight"],
        labels=df["ticker"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[: len(df)],
        textprops={"fontsize": 9},
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.set_title("Aktuelle Portfolio-Gewichte", fontsize=14, color=BLUE, pad=14)
    return _fig_to_base64(fig)


def load_nav_history() -> pd.DataFrame:
    path = OUTPUTS / "paper_nav_history.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "nav_usd"])
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["nav_usd"] = pd.to_numeric(df["nav_usd"], errors="coerce")
    return df.dropna(subset=["date", "nav_usd"]).sort_values("date")


def load_spy_history() -> pd.DataFrame:
    for path in [OUTPUTS / "spy_history.csv", OUTPUTS / "SPY_history.csv"]:
        if path.exists():
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            close_col = "close" if "close" in df.columns else df.columns[-1]
            df["close"] = pd.to_numeric(df[close_col], errors="coerce")
            return df.dropna(subset=["date", "close"]).sort_values("date")
    return pd.DataFrame(columns=["date", "close"])


def build_performance_chart(nav: pd.DataFrame, spy: pd.DataFrame, start_date: str = "2026-04-24") -> tuple[str, dict]:
    stats = {
        "sharpe": np.nan,
        "ann_return": np.nan,
        "ann_vol": np.nan,
        "abs_gain": np.nan,
        "rel_gain": np.nan,
    }

    if nav.empty:
        return "", stats

    nav = nav[nav["date"] >= pd.Timestamp(start_date)].copy()
    if nav.empty:
        return "", stats

    nav = nav.sort_values("date")
    nav["ret"] = nav["nav_usd"].pct_change().fillna(0.0)

    if len(nav) >= 2:
        abs_gain = float(nav["nav_usd"].iloc[-1] - nav["nav_usd"].iloc[0])
        rel_gain = float(nav["nav_usd"].iloc[-1] / nav["nav_usd"].iloc[0] - 1.0)
        ann_return = float((1.0 + rel_gain) ** (252 / max(len(nav) - 1, 1)) - 1.0)
        ann_vol = float(nav["ret"].std(ddof=0) * np.sqrt(252))
        sharpe = float((ann_return - 0.02) / ann_vol) if ann_vol > 0 else np.nan
        stats.update(
            {
                "sharpe": sharpe,
                "ann_return": ann_return,
                "ann_vol": ann_vol,
                "abs_gain": abs_gain,
                "rel_gain": rel_gain,
            }
        )

    nav["indexed"] = 100.0 * nav["nav_usd"] / nav["nav_usd"].iloc[0]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(nav["date"], nav["indexed"], label="Portfolio", linewidth=2.5, color=BLUE)

    if not spy.empty:
        spy = spy[spy["date"] >= nav["date"].min()].copy()
        if not spy.empty:
            spy["indexed"] = 100.0 * spy["close"] / spy["close"].iloc[0]
            ax.plot(spy["date"], spy["indexed"], label="S&P 500 / SPY", linewidth=2.0, color="#7AA7D9")

    ax.set_title(f"Portfolio vs. S&P 500 seit {nav['date'].min().date()}", fontsize=14, color=BLUE)
    ax.set_ylabel("Indexiert, Start = 100")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _fig_to_base64(fig), stats


def build_tud_visual_email() -> tuple[str, str]:
    positions = load_current_positions()
    orders = load_orders()
    nav = load_nav_history()
    spy = load_spy_history()

    pie_b64 = build_pie_chart(positions)
    curve_b64, stats = build_performance_chart(nav, spy)

    buy = orders[orders["action"] == "BUY"][["ticker", "shares", "volume_usd"]].copy()
    sell = orders[orders["action"] == "SELL"][["ticker", "shares", "volume_usd"]].copy()

    buy.columns = ["Ticker", "Stückzahl", "Volumen"]
    sell.columns = ["Ticker", "Stückzahl", "Volumen"]

    if not buy.empty:
        buy["Volumen"] = buy["Volumen"].map(_money)
    if not sell.empty:
        sell["Volumen"] = sell["Volumen"].map(_money)

    pnl = positions.copy()
    if not pnl.empty:
        pnl["Stückzahl"] = pnl["shares"].map(_num)
        pnl["Preis"] = pnl["current_price"].map(_money)
        pnl["Wert"] = pnl["total_value"].map(_money)
        pnl["Gewicht"] = pnl["weight"].map(_pct)
        pnl = pnl[["ticker", "Stückzahl", "Preis", "Wert", "Gewicht"]].copy()
        pnl.columns = ["Ticker", "Stückzahl", "Preis", "Wert", "Gewicht"]

    sharpe_text = f"{stats['sharpe']:.3f}" if pd.notna(stats["sharpe"]) else "-"
    rel_gain_text = _pct(stats["rel_gain"])
    vol_text = _pct(stats["ann_vol"])
    abs_gain_text = _money(stats["abs_gain"])

    html = f"""
<html>
<head>
<style>
body {{
  margin: 0;
  padding: 0;
  background: #F2F7FD;
  font-family: Arial, Helvetica, sans-serif;
  color: {TEXT};
}}
.wrapper {{
  max-width: 980px;
  margin: 0 auto;
  padding: 24px;
}}
.header {{
  background: linear-gradient(135deg, {BLUE}, {BLUE_2});
  color: white;
  border-radius: 18px;
  padding: 24px 28px;
  box-shadow: 0 10px 24px rgba(11,58,117,0.18);
}}
.header h1 {{
  margin: 0 0 8px 0;
  font-size: 26px;
}}
.header p {{
  margin: 0;
  opacity: 0.92;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 18px 0;
}}
.metric {{
  background: white;
  border: 1px solid {BORDER};
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 6px 16px rgba(15,94,168,0.08);
}}
.metric .label {{
  color: {MUTED};
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: .04em;
}}
.metric .value {{
  color: {BLUE};
  font-size: 20px;
  font-weight: 700;
  margin-top: 4px;
}}
.card {{
  background: white;
  border: 1px solid {BORDER};
  border-radius: 18px;
  padding: 20px;
  margin: 18px 0;
  box-shadow: 0 8px 22px rgba(15,94,168,0.08);
}}
h2 {{
  color: {BLUE};
  margin: 0 0 14px 0;
  font-size: 20px;
}}
.tbl {{
  border-collapse: collapse;
  width: 100%;
}}
.tbl th {{
  text-align: left;
  background: {BLUE_3};
  color: {BLUE};
  border-bottom: 2px solid {BORDER};
  padding: 9px 10px;
  font-size: 13px;
}}
.tbl td {{
  border-bottom: 1px solid #E8F1FB;
  padding: 9px 10px;
  font-size: 13px;
}}
.tbl tr:nth-child(even) td {{
  background: {BLUE_4};
}}
.badge {{
  display: inline-block;
  background: {BLUE_3};
  color: {BLUE};
  border-radius: 999px;
  padding: 6px 10px;
  font-weight: 700;
  font-size: 12px;
}}
.notice {{
  color: {MUTED};
  font-size: 13px;
}}
img {{
  max-width: 100%;
  height: auto;
}}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h1>TUD Dry-Run Trading Report</h1>
    <p>Validierter Handlungsvorschlag. Es wurden keine Orders gesendet.</p>
  </div>

  <div class="metrics">
    <div class="metric"><div class="label">Sharpe</div><div class="value">{sharpe_text}</div></div>
    <div class="metric"><div class="label">Rendite seit 24.04</div><div class="value">{rel_gain_text}</div></div>
    <div class="metric"><div class="label">Vola p.a.</div><div class="value">{vol_text}</div></div>
    <div class="metric"><div class="label">Absoluter P/L</div><div class="value">{abs_gain_text}</div></div>
  </div>

  <div class="card">
    <h2>Aktuelle Portfolio-Gewichte</h2>
    {f"<img src='data:image/png;base64,{pie_b64}'>" if pie_b64 else "<p class='notice'>Kein Kuchendiagramm verfügbar.</p>"}
  </div>

  <div class="card">
    <h2>BUY</h2>
    {_html_table(buy)}
  </div>

  <div class="card">
    <h2>SELL</h2>
    {_html_table(sell)}
  </div>

  <div class="card">
    <h2>Portfolio vs. S&P 500</h2>
    {f"<img src='data:image/png;base64,{curve_b64}'>" if curve_b64 else "<p class='notice'>Noch keine NAV/SPY-Historie verfügbar. Ab jetzt kann sie aufgebaut werden.</p>"}
  </div>

  <div class="card">
    <h2>Aktuelle Positionen</h2>
    {_html_table(pnl)}
  </div>

  <p class="notice">Submit Guard bleibt aktiv. Diese Mail ist kein Orderversand.</p>
</div>
</body>
</html>
""".strip()

    plain = f"""TUD Dry-Run Trading Report

Keine Orders wurden gesendet.

Sharpe: {sharpe_text}
Rendite seit 24.04: {rel_gain_text}
Vola p.a.: {vol_text}
Absoluter P/L: {abs_gain_text}

BUY:
{buy.to_string(index=False) if not buy.empty else "keine"}

SELL:
{sell.to_string(index=False) if not sell.empty else "keine"}
"""

    return html, plain
