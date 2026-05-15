"""Rich rendering helpers for the Daily Review mail/report path."""

from __future__ import annotations

import base64
from html import escape as html_escape
from mimetypes import guess_type
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from notifications import sanitize_for_output

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional
    plt = None
    MATPLOTLIB_AVAILABLE = False


BLANK_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5n2S8AAAAASUVORK5CYII="
)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _parse_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_text(value: object) -> str:
    return "true" if bool(value) else "false"


def _format_usd(value: object) -> str:
    return f"{_safe_float(value, 0.0):,.2f} USD"


def _format_pct(value: object, digits: int = 2) -> str:
    return f"{_safe_float(value, 0.0):.{digits}%}"


def _format_num(value: object, digits: int = 4) -> str:
    return f"{_safe_float(value, 0.0):.{digits}f}"


def _status_style(final_action: str, hard_fail_count: int) -> tuple[str, str, str]:
    action = str(final_action or "").upper()
    if hard_fail_count > 0 or "BLOCK" in action:
        return "BLOCK", "#fdf2f2", "#b42318"
    if "BUY" in action:
        return action, "#ecfdf3", "#067647"
    if "SELL" in action:
        return action, "#fff4ed", "#c4320a"
    if "WAIT" in action:
        return action, "#fffaeb", "#b54708"
    if action == "HOLD":
        return "HOLD", "#fffaeb", "#b54708"
    return action or "REVIEW", "#eff8ff", "#175cd3"


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return path


def _atomic_write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return path


def _write_blank_png(path: Path) -> Path:
    return _atomic_write_bytes(path, base64.b64decode(BLANK_PNG_BASE64))


def _write_chart_placeholder(path: Path, title: str, message: str) -> Path:
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return _write_blank_png(path)
    figure, axis = plt.subplots(figsize=(8, 2.5))
    axis.axis("off")
    axis.text(0.01, 0.82, title, fontsize=13, fontweight="bold", ha="left", va="top")
    axis.text(0.01, 0.48, "Chart generation fallback", fontsize=10, ha="left", va="top")
    axis.text(0.01, 0.15, sanitize_for_output(message), fontsize=8.5, ha="left", va="top", wrap=True)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return path


def _safe_plot(output_path: Path, title: str, plotter: Any, warnings: list[str]) -> Path:
    try:
        if not MATPLOTLIB_AVAILABLE or plt is None:
            raise RuntimeError("matplotlib not available")
        path = plotter()
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError("chart file missing or empty after plot")
        return path
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"{title}: {sanitize_for_output(exc)}")
        return _write_chart_placeholder(output_path, title, str(exc))


def _current_positions_frame(review: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in list(review.get("current_positions", [])):
        rows.append(
            {
                "ticker": str(row.get("ticker", "")),
                "shares": _safe_float(row.get("current_shares", 0.0)),
                "latest_price": _safe_float(row.get("latest_price", 0.0)),
                "market_value_usd": _safe_float(row.get("market_value_usd", 0.0)),
                "current_weight": _safe_float(row.get("current_weight", 0.0)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("market_value_usd", ascending=False).reset_index(drop=True)


def _target_allocation_frame(review: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in list(review.get("target_allocation", [])):
        rows.append(
            {
                "ticker": str(row.get("ticker", "")),
                "target_weight": _safe_float(row.get("target_weight", 0.0)),
                "target_shares": _safe_float(row.get("target_shares", 0.0)),
                "target_market_value_usd": _safe_float(row.get("target_market_value_usd", 0.0)),
                "continuous_target_weight": _safe_float(row.get("continuous_target_weight", 0.0)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("target_weight", ascending=False).reset_index(drop=True)


def _delta_orders_frame(review: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in list(review.get("delta_transactions", [])):
        rows.append(
            {
                "ticker": str(row.get("ticker", "")),
                "action": str(row.get("action", "")),
                "current_shares": _safe_float(row.get("current_shares", 0.0)),
                "target_shares": _safe_float(row.get("target_shares", 0.0)),
                "order_shares": _safe_float(row.get("order_shares", 0.0)),
                "estimated_price": _safe_float(row.get("estimated_price", 0.0)),
                "estimated_order_value": _safe_float(row.get("estimated_order_value", 0.0)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame[frame["action"].isin(["BUY", "SELL"]) & (frame["order_shares"] > 0)].reset_index(drop=True)


def _manual_order_count(review: dict[str, Any]) -> int:
    order_summary = review.get("order_summary", {})
    return int(
        order_summary.get(
            "manual_eligible_order_count",
            order_summary.get("order_count", 0),
        )
        or 0
    )


def _build_operator_instruction(review: dict[str, Any], issues: dict[str, Any]) -> str:
    run_status = review.get("run_status", {})
    data_status = review.get("data_status", {})
    final_action = str(run_status.get("final_action", "") or "").upper()
    manual_order_count = _manual_order_count(review)
    instructions: list[str] = []
    if bool(data_status.get("synthetic_data", False)):
        instructions.append("Blockiert: synthetische Daten; keine Orders.")
    elif not bool(data_status.get("data_freshness_ok", False)):
        instructions.append("Blockiert: Daten nicht frisch.")
    elif "WAIT" in final_action or "BLOCK" in final_action or int(issues.get("hard_fail_count", 0)) > 0:
        instructions.append("Keine Orders eingeben.")
    elif final_action == "HOLD":
        instructions.append("Heute keine Orders eingeben. Beste Aktion laut Bot: HOLD.")
    if manual_order_count == 0:
        instructions.append("Keine Simulator-Orders eingeben.")
    if bool(data_status.get("used_cache_fallback", False)):
        instructions.append("Warnung: Live-Daten nicht genutzt; Bericht nur vorsichtig verwenden.")
    if not instructions:
        instructions.append("Preview only. Nur die freigegebenen Delta-Orders aus outputs/manual_simulator_orders.csv manuell pruefen.")
    return " ".join(dict.fromkeys(instructions))


def _allocation_chart_data(review: dict[str, Any]) -> pd.DataFrame:
    current_frame = _current_positions_frame(review).copy()
    if current_frame.empty:
        return current_frame
    current_frame["value"] = current_frame["market_value_usd"].clip(lower=0.0)
    current_frame["label"] = current_frame["ticker"]
    nav_usd = _safe_float(review.get("current_portfolio", {}).get("nav_usd", 0.0))
    cash_usd = _safe_float(review.get("current_portfolio", {}).get("cash_usd", 0.0))
    if nav_usd > 0.0 and cash_usd > 0.0:
        cash_row = pd.DataFrame([{"ticker": "CASH", "shares": 0.0, "latest_price": 1.0, "market_value_usd": cash_usd, "current_weight": cash_usd / nav_usd, "value": cash_usd, "label": "CASH"}])
        current_frame = pd.concat([current_frame, cash_row], ignore_index=True)
    current_frame = current_frame.sort_values("value", ascending=False).reset_index(drop=True)
    if len(current_frame) > 8:
        top = current_frame.iloc[:7].copy()
        other_value = float(current_frame.iloc[7:]["value"].sum())
        other_weight = float(current_frame.iloc[7:]["current_weight"].sum())
        if other_value > 0.0:
            top = pd.concat(
                [
                    top,
                    pd.DataFrame(
                        [
                            {
                                "ticker": "OTHER",
                                "shares": 0.0,
                                "latest_price": 0.0,
                                "market_value_usd": other_value,
                                "current_weight": other_weight,
                                "value": other_value,
                                "label": "OTHER",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        return top
    return current_frame


def _plot_current_portfolio_allocation(review: dict[str, Any], output_path: Path) -> Path:
    frame = _allocation_chart_data(review)
    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    if frame.empty:
        axis.axis("off")
        axis.text(0.5, 0.5, "No current portfolio positions available", ha="center", va="center", fontsize=13)
    else:
        colors = [
            "#1849a9",
            "#1570ef",
            "#36bffa",
            "#12b76a",
            "#f79009",
            "#ef6820",
            "#7a5af8",
            "#667085",
        ]
        wedges, _ = axis.pie(
            frame["value"],
            startangle=90,
            wedgeprops={"width": 0.44, "edgecolor": "white"},
            colors=colors[: len(frame)],
        )
        axis.set_title("Current Portfolio Allocation", fontsize=15, fontweight="bold")
        legend_labels = [
            f"{row.label}: {_format_pct(row.current_weight)}"
            for row in frame.itertuples(index=False)
        ]
        axis.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_current_vs_target_weights(review: dict[str, Any], output_path: Path) -> Path:
    current_frame = _current_positions_frame(review)[["ticker", "current_weight"]]
    target_frame = _target_allocation_frame(review)[["ticker", "target_weight"]]
    merged = current_frame.merge(target_frame, on="ticker", how="outer").fillna(0.0)
    if merged.empty:
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.axis("off")
        axis.text(0.5, 0.5, "No current/target weights available", ha="center", va="center", fontsize=13)
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return output_path
    merged["max_weight"] = merged[["current_weight", "target_weight"]].max(axis=1)
    merged = merged.sort_values("max_weight", ascending=False).head(10).sort_values("max_weight", ascending=True)
    figure, axis = plt.subplots(figsize=(9.5, 6.5))
    positions = range(len(merged))
    axis.barh([pos - 0.18 for pos in positions], merged["current_weight"], height=0.34, color="#1570ef", label="Current")
    axis.barh([pos + 0.18 for pos in positions], merged["target_weight"], height=0.34, color="#12b76a", label="Target")
    axis.set_yticks(list(positions))
    axis.set_yticklabels(list(merged["ticker"]))
    axis.set_xlabel("Weight")
    axis.set_title("Current vs Target Weights", fontsize=15, fontweight="bold")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_nav_cash_summary(review: dict[str, Any], issues: dict[str, Any], output_path: Path) -> Path:
    current_portfolio = review.get("current_portfolio", {})
    order_summary = review.get("order_summary", {})
    cost_edge = review.get("cost_edge", {})
    figure, axis = plt.subplots(figsize=(10, 3.8))
    axis.axis("off")
    summary_items = [
        ("NAV", _format_usd(current_portfolio.get("nav_usd", 0.0))),
        ("Cash", _format_usd(current_portfolio.get("cash_usd", 0.0))),
        ("Order Count", str(_parse_int(order_summary.get("order_count", 0), 0))),
        ("Manual Orders", str(_parse_int(order_summary.get("manual_eligible_order_count", 0), 0))),
        ("Trade Edge", f"{_safe_float(cost_edge.get('trade_now_edge', 0.0)):.6f}"),
        ("Warnings", str(_parse_int(issues.get("soft_warning_count", 0), 0))),
    ]
    box_colors = ["#eff8ff", "#ecfdf3", "#fffaeb", "#f2f4f7", "#fff4ed", "#fdf2f8"]
    for idx, (label, value) in enumerate(summary_items):
        col = idx % 3
        row = idx // 3
        x0 = 0.03 + col * 0.32
        y0 = 0.56 - row * 0.42
        rect = plt.Rectangle((x0, y0), 0.28, 0.26, facecolor=box_colors[idx], edgecolor="#d0d5dd", linewidth=1.0)
        axis.add_patch(rect)
        axis.text(x0 + 0.02, y0 + 0.18, label, fontsize=10, fontweight="bold", color="#344054", ha="left", va="center")
        axis.text(x0 + 0.02, y0 + 0.08, value, fontsize=12, color="#101828", ha="left", va="center")
    axis.set_title("NAV / Cash / Risk Snapshot", fontsize=15, fontweight="bold", loc="left")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_risk_and_blockers(review: dict[str, Any], issues: dict[str, Any], output_path: Path) -> Path:
    cost_edge = review.get("cost_edge", {})
    metrics = [
        ("trade_now_edge", _safe_float(cost_edge.get("trade_now_edge", 0.0))),
        ("execution_buffer", _safe_float(cost_edge.get("execution_buffer", 0.0))),
        ("model_uncertainty", _safe_float(cost_edge.get("model_uncertainty_buffer", 0.0))),
    ]
    figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.0, 1.1]})
    axis_left.barh([name for name, _ in metrics], [value for _, value in metrics], color=["#f79009", "#1570ef", "#7a5af8"])
    axis_left.set_title("Risk / Edge Metrics", fontsize=13, fontweight="bold")
    axis_left.grid(axis="x", alpha=0.25)
    axis_right.axis("off")
    axis_right.set_title("Top Blockers", fontsize=13, fontweight="bold", loc="left")
    blocker_lines = [str(item.get("message", "")) for item in list(issues.get("issue_table", []))[:6]]
    if not blocker_lines:
        blocker_lines = ["none"]
    axis_right.text(
        0.0,
        0.95,
        "\n".join(f"- {sanitize_for_output(line)}" for line in blocker_lines),
        ha="left",
        va="top",
        fontsize=10,
        wrap=True,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _chart_data_uri(path: Path) -> str:
    if not path.exists() or path.stat().st_size <= 0:
        return ""
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type = guess_type(path.name)[0] or "image/png"
    return f"data:{mime_type};base64,{payload}"


def _table_html(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{html_escape(item)}</th>" for item in headers)
    row_html = []
    for row in rows:
        row_html.append("<tr>" + "".join(f"<td>{html_escape(str(cell))}</td>" for cell in row) + "</tr>")
    return (
        "<table class=\"report-table\">"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody>"
        "</table>"
    )


def _build_orders_table_rows(delta_frame: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in delta_frame.itertuples(index=False):
        rows.append(
            [
                str(row.action),
                str(row.ticker),
                _format_num(row.current_shares, 4),
                _format_num(row.target_shares, 4),
                _format_num(row.order_shares, 4),
                _format_usd(row.estimated_order_value),
            ]
        )
    return rows


def _build_html_report(
    review: dict[str, Any],
    issues: dict[str, Any],
    subject: str,
    plain_text_body: str,
    chart_paths: dict[str, Path],
    render_warnings: list[str],
    pdf_path: Path | None,
) -> str:
    run_status = review.get("run_status", {})
    data_status = review.get("data_status", {})
    current_portfolio = review.get("current_portfolio", {})
    decision_context = review.get("decision_context", {})
    hold_analysis = dict(review.get("hold_analysis", {}) or {})
    cost_edge = review.get("cost_edge", {})
    current_frame = _current_positions_frame(review).head(10)
    target_frame = _target_allocation_frame(review).head(10)
    delta_frame = _delta_orders_frame(review)
    status_label, status_bg, status_fg = _status_style(run_status.get("final_action", ""), int(issues.get("hard_fail_count", 0)))
    chart_sections: list[str] = []
    for title, key in [
        ("Current Portfolio Allocation", "current_portfolio_allocation"),
        ("Current vs Target Weights", "current_vs_target_weights"),
        ("NAV / Cash / Risk Snapshot", "nav_cash_summary"),
        ("Risk and Blockers", "risk_and_blockers"),
    ]:
        chart_path = chart_paths.get(key)
        if chart_path is None:
            continue
        data_uri = _chart_data_uri(chart_path)
        if not data_uri:
            continue
        chart_sections.append(
            "<div class=\"chart-card\">"
            f"<h3>{html_escape(title)}</h3>"
            f"<img src=\"{data_uri}\" alt=\"{html_escape(title)}\" />"
            "</div>"
        )
    current_rows = [
        [
            str(row.ticker),
            _format_num(row.shares, 4),
            _format_usd(row.market_value_usd),
            _format_pct(row.current_weight),
        ]
        for row in current_frame.itertuples(index=False)
    ]
    target_rows = [
        [
            str(row.ticker),
            _format_pct(row.target_weight),
            _format_num(row.target_shares, 4),
            _format_usd(row.target_market_value_usd),
        ]
        for row in target_frame.itertuples(index=False)
    ]
    issue_rows = [
        f"<li><strong>{html_escape(str(item.get('severity', 'INFO')))}</strong>: {html_escape(str(item.get('message', '')))}</li>"
        for item in list(issues.get("issue_table", []))[:8]
    ]
    why_hold_rows = [
        f"<li>{html_escape(str(item))}</li>"
        for item in list(hold_analysis.get("why_hold_lines", []) or [])
    ]
    hold_blocker_rows = [
        [
            html_escape(str(item.get("blocker", "n/a"))),
            html_escape(str(item.get("type", "n/a"))),
            html_escape(str(item.get("impact", "n/a"))),
            html_escape(str(item.get("explanation", "n/a"))),
        ]
        for item in list(hold_analysis.get("blocker_table", []) or [])
    ]
    relevant_files = [
        "outputs/manual_simulator_orders.csv",
        "outputs/order_preview.csv",
        "outputs/daily_portfolio_review.txt",
        "outputs/hold_dominance_analysis.txt",
        "outputs/hold_sensitivity_report.txt",
        "outputs/decision_history.csv",
        "outputs/today_decision_summary.txt",
        "outputs/rebalance_decision_report.txt",
        "outputs/daily_bot_decision_report.txt",
        "outputs/current_data_freshness_report.txt",
        "outputs/daily_review_report.tex",
        "outputs/daily_review_report.pdf" if pdf_path is not None else "outputs/daily_review_report.tex (PDF not available)",
    ]
    warning_html = ""
    if render_warnings:
        warning_html = (
            "<div class=\"warning-box\"><strong>Rendering Warnings:</strong><ul>"
            + "".join(f"<li>{html_escape(item)}</li>" for item in render_warnings)
            + "</ul></div>"
        )
    pdf_status = "Available as attachment." if pdf_path is not None else "PDF not built; HTML and plaintext remain available."
    plain_text_preview = "<br/>".join(html_escape(line) for line in plain_text_body.strip().splitlines()[:5])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html_escape(subject)}</title>
  <style>
    body {{
      margin: 0;
      padding: 0;
      background: #f5f7fb;
      color: #101828;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    .container {{
      max-width: 980px;
      margin: 0 auto;
      padding: 24px 16px 40px;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #e4e7ec;
      border-radius: 16px;
      box-shadow: 0 10px 24px rgba(16, 24, 40, 0.06);
      padding: 24px;
      margin-bottom: 18px;
    }}
    .hero {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: flex-start;
    }}
    .status-chip {{
      display: inline-block;
      padding: 8px 14px;
      border-radius: 999px;
      font-weight: 700;
      letter-spacing: 0.02em;
      background: {status_bg};
      color: {status_fg};
      border: 1px solid {status_fg};
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric {{
      border: 1px solid #e4e7ec;
      border-radius: 14px;
      padding: 14px;
      background: #fcfcfd;
    }}
    .metric .label {{
      color: #475467;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric .value {{
      color: #101828;
      font-size: 20px;
      font-weight: 700;
      margin-top: 6px;
    }}
    .instruction-box {{
      background: #eff8ff;
      border-left: 5px solid #1570ef;
      border-radius: 12px;
      padding: 14px 16px;
      font-size: 16px;
      font-weight: 600;
    }}
    .warning-box {{
      background: #fffaeb;
      border: 1px solid #fedf89;
      border-radius: 12px;
      padding: 12px 14px;
      margin-top: 14px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      line-height: 1.2;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 20px;
    }}
    h3 {{
      margin: 0 0 12px;
      font-size: 16px;
    }}
    .report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .report-table th {{
      background: #f8fafc;
      text-align: left;
      color: #344054;
      font-weight: 700;
      padding: 10px 12px;
      border-bottom: 1px solid #e4e7ec;
    }}
    .report-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid #f0f2f5;
      vertical-align: top;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .chart-card {{
      border: 1px solid #e4e7ec;
      border-radius: 14px;
      padding: 14px;
      background: #fcfcfd;
    }}
    .chart-card img {{
      width: 100%;
      height: auto;
      border-radius: 10px;
      display: block;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
    }}
    .muted {{
      color: #475467;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="panel">
      <div class="hero">
        <div>
          <div class="muted">Daily Portfolio Review</div>
          <h1>{html_escape(subject)}</h1>
          <div class="muted">Review date: {html_escape(str(run_status.get("review_date", "n/a")))} | Time (Berlin): {html_escape(str(run_status.get("review_time_berlin", "n/a")))}</div>
        </div>
        <div class="status-chip">{html_escape(status_label)}</div>
      </div>
      <div class="summary-grid">
        <div class="metric"><div class="label">Final Action</div><div class="value">{html_escape(str(run_status.get("final_action", "n/a")))}</div></div>
        <div class="metric"><div class="label">First Blocker</div><div class="value" style="font-size:16px">{html_escape(str(issues.get("first_blocker", "none")))}</div></div>
        <div class="metric"><div class="label">Trade Now Edge</div><div class="value">{_safe_float(cost_edge.get("trade_now_edge", 0.0)):.6f}</div></div>
        <div class="metric"><div class="label">Manual Orders</div><div class="value">{_manual_order_count(review)}</div></div>
        <div class="metric"><div class="label">NAV</div><div class="value">{_format_usd(current_portfolio.get("nav_usd", 0.0))}</div></div>
        <div class="metric"><div class="label">Cash</div><div class="value">{_format_usd(current_portfolio.get("cash_usd", 0.0))}</div></div>
        <div class="metric"><div class="label">Latest Price Date</div><div class="value" style="font-size:16px">{html_escape(str(data_status.get("latest_price_date", "n/a")))}</div></div>
        <div class="metric"><div class="label">Data Source</div><div class="value" style="font-size:16px">{html_escape(str(data_status.get("data_source", "n/a")))}</div></div>
      </div>
    </div>

    <div class="panel">
      <h2>Heute tun</h2>
      <div class="instruction-box">{html_escape(str(_build_operator_instruction(review, issues)))}</div>
      {warning_html}
    </div>

    <div class="panel two-col">
      <div>
        <h2>Warum</h2>
        <ul>
          <li><strong>continuous_candidate:</strong> {html_escape(str(decision_context.get("continuous_candidate", "n/a")))}</li>
          <li><strong>final_discrete_candidate:</strong> {html_escape(str(decision_context.get("final_discrete_candidate", "n/a")))}</li>
          <li><strong>trade_now_edge:</strong> {_safe_float(cost_edge.get("trade_now_edge", 0.0)):.6f}</li>
          <li><strong>first_blocker:</strong> {html_escape(str(issues.get("first_blocker", "none")))}</li>
          <li><strong>all_blockers:</strong> {html_escape(" | ".join(map(str, issues.get("all_blockers", ["none"]))))}</li>
        </ul>
      </div>
      <div>
        <h2>Warnings / Risks</h2>
        <ul>{''.join(issue_rows) if issue_rows else '<li>none</li>'}</ul>
      </div>
    </div>

    <div class="panel">
      <h2>Warum HOLD?</h2>
      <ul>{''.join(why_hold_rows) if why_hold_rows else '<li>Keine zusaetzliche HOLD-Analyse verfuegbar.</li>'}</ul>
      {_table_html(["Blocker", "Typ", "Einfluss", "Erklaerung"], hold_blocker_rows) if hold_blocker_rows else ""}
    </div>

    <div class="panel">
      <h2>Orders</h2>
      {"<p><strong>Keine BUY/SELL-Delta-Orders.</strong> Diese Orders sind Delta-Orders relativ zum aktuellen Portfolio.</p>" if delta_frame.empty else _table_html(["Action", "Ticker", "Current Shares", "Target Shares", "Order Shares", "Est. Value"], _build_orders_table_rows(delta_frame))}
      {"" if delta_frame.empty else "<p class=\"muted\">Diese Orders sind Delta-Orders relativ zum aktuellen Portfolio.</p>"}
    </div>

    <div class="panel two-col">
      <div>
        <h2>Current Portfolio</h2>
        {_table_html(["Ticker", "Shares", "Market Value", "Weight"], current_rows) if current_rows else "<p>No current positions available.</p>"}
      </div>
      <div>
        <h2>Target Portfolio</h2>
        {_table_html(["Ticker", "Target Weight", "Target Shares", "Target Value"], target_rows) if target_rows else "<p>No target allocation available.</p>"}
      </div>
    </div>

    <div class="panel">
      <h2>Charts</h2>
      <div class="chart-grid">{''.join(chart_sections) if chart_sections else '<p>No charts available for this run.</p>'}</div>
    </div>

    <div class="panel">
      <h2>Files and Fallbacks</h2>
      <p><strong>PDF status:</strong> {html_escape(pdf_status)}</p>
      <ul>{''.join(f'<li class=\"mono\">{html_escape(item)}</li>' for item in relevant_files)}</ul>
      <p class="muted">Plaintext fallback remains available. Preview snippet:</p>
      <div class="mono">{plain_text_preview}</div>
    </div>
  </div>
</body>
</html>
"""


def _latex_escape(value: object) -> str:
    text = sanitize_for_output("" if value is None else str(value))
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _latex_longtable(headers: list[str], rows: list[list[str]], column_spec: str) -> str:
    if not rows:
        return "No rows available.\\\\\n"
    head = " & ".join(_latex_escape(item) for item in headers) + r" \\"
    body_lines = [" & ".join(_latex_escape(cell) for cell in row) + r" \\" for row in rows]
    return (
        rf"""\begin{{longtable}}{{{column_spec}}}
\toprule
{head}
\midrule
\endhead
"""
        + "\n".join(body_lines)
        + "\n\\bottomrule\n\\end{longtable}\n"
    )


def _relative_for_tex(path: Path, base_dir: Path) -> str:
    return _latex_escape(os.path.relpath(path, base_dir))


def _build_latex_report(
    review: dict[str, Any],
    issues: dict[str, Any],
    subject: str,
    chart_paths: dict[str, Path],
    render_warnings: list[str],
    output_dir: Path,
) -> str:
    run_status = review.get("run_status", {})
    data_status = review.get("data_status", {})
    current_portfolio = review.get("current_portfolio", {})
    decision_context = review.get("decision_context", {})
    hold_analysis = dict(review.get("hold_analysis", {}) or {})
    cost_edge = review.get("cost_edge", {})
    order_summary = review.get("order_summary", {})
    current_frame = _current_positions_frame(review).head(14)
    target_frame = _target_allocation_frame(review).head(14)
    delta_frame = _delta_orders_frame(review)
    operator_instruction = _build_operator_instruction(review, issues)
    status_label, _, _ = _status_style(run_status.get("final_action", ""), int(issues.get("hard_fail_count", 0)))
    current_rows = [
        [
            str(row.ticker),
            _format_num(row.shares, 4),
            _format_num(row.latest_price, 4),
            _format_usd(row.market_value_usd),
            _format_pct(row.current_weight),
        ]
        for row in current_frame.itertuples(index=False)
    ]
    target_rows = [
        [
            str(row.ticker),
            _format_pct(row.target_weight),
            _format_num(row.target_shares, 4),
            _format_usd(row.target_market_value_usd),
        ]
        for row in target_frame.itertuples(index=False)
    ]
    delta_rows = _build_orders_table_rows(delta_frame)
    warning_items = [str(item.get("message", "")) for item in list(issues.get("issue_table", []))[:8]]
    why_hold_items = [str(item) for item in list(hold_analysis.get("why_hold_lines", []) or [])]
    hold_blocker_rows = [
        [
            str(item.get("blocker", "n/a")),
            str(item.get("type", "n/a")),
            str(item.get("impact", "n/a")),
            str(item.get("explanation", "n/a")),
        ]
        for item in list(hold_analysis.get("blocker_table", []) or [])
    ]
    chart_blocks: list[str] = []
    for key, title in [
        ("current_portfolio_allocation", "Current Portfolio Allocation"),
        ("current_vs_target_weights", "Current vs Target Weights"),
        ("nav_cash_summary", "NAV / Cash / Risk Snapshot"),
        ("risk_and_blockers", "Risk and Blockers"),
    ]:
        chart_path = chart_paths.get(key)
        if chart_path is None or not chart_path.exists():
            continue
        chart_blocks.append(
            "\\begin{figure}[H]\n"
            "\\centering\n"
            f"\\includegraphics[width=0.92\\linewidth]{{{_relative_for_tex(chart_path, output_dir)}}}\n"
            f"\\caption*{{{_latex_escape(title)}}}\n"
            "\\end{figure}\n"
        )
    render_warning_block = "\n".join(f"\\item {_latex_escape(item)}" for item in render_warnings) or "\\item none"
    issue_block = "\n".join(f"\\item {_latex_escape(item)}" for item in warning_items) or "\\item none"
    relevant_files = [
        "outputs/manual_simulator_orders.csv",
        "outputs/order_preview.csv",
        "outputs/daily_portfolio_review.txt",
        "outputs/hold_dominance_analysis.txt",
        "outputs/hold_sensitivity_report.txt",
        "outputs/decision_history.csv",
        "outputs/daily_review_email.html",
        "outputs/daily_review_report.tex",
        "outputs/daily_review_report.pdf",
    ]
    relevant_files_block = "\n".join(f"\\item \\texttt{{{_latex_escape(item)}}}" for item in relevant_files)
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{xcolor}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\hypersetup{{hidelinks}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.5em}}

\begin{{document}}

\begin{{center}}
{{\LARGE \textbf{{Daily Portfolio Review}}}}\\[0.4em]
{{\large {_latex_escape(subject)}}}\\[0.5em]
\colorbox{{gray!12}}{{\parbox{{0.92\linewidth}}{{\centering
\textbf{{Status}}: {_latex_escape(status_label)} \quad
\textbf{{Review Date}}: {_latex_escape(run_status.get("review_date", "n/a"))} \quad
\textbf{{Time (Berlin)}}: {_latex_escape(run_status.get("review_time_berlin", "n/a"))}\\
\textbf{{First Blocker}}: {_latex_escape(issues.get("first_blocker", "none"))}
}}}}
\end{{center}}

\section*{{Executive Summary}}
\begin{{tabular}}{{p{{0.34\linewidth}}p{{0.6\linewidth}}}}
\toprule
Field & Value \\
\midrule
Final Action & {_latex_escape(run_status.get("final_action", "n/a"))} \\
Operator Instruction & {_latex_escape(operator_instruction)} \\
Manual Order Count & {_latex_escape(order_summary.get("manual_eligible_order_count", 0))} \\
NAV & {_latex_escape(_format_usd(current_portfolio.get("nav_usd", 0.0)))} \\
Cash & {_latex_escape(_format_usd(current_portfolio.get("cash_usd", 0.0)))} \\
Trade Now Edge & {_latex_escape(f"{_safe_float(cost_edge.get('trade_now_edge', 0.0)):.6f}")} \\
\bottomrule
\end{{tabular}}

\section*{{Data Status}}
\begin{{tabular}}{{p{{0.34\linewidth}}p{{0.6\linewidth}}}}
\toprule
Field & Value \\
\midrule
Data Source & {_latex_escape(data_status.get("data_source", "n/a"))} \\
Latest Price Date & {_latex_escape(data_status.get("latest_price_date", "n/a"))} \\
Used Cache Fallback & {_latex_escape(_bool_text(data_status.get("used_cache_fallback", False)))} \\
Synthetic Data & {_latex_escape(_bool_text(data_status.get("synthetic_data", False)))} \\
Data Freshness OK & {_latex_escape(_bool_text(data_status.get("data_freshness_ok", False)))} \\
Live Data Error & {_latex_escape(data_status.get("live_data_error", "") or "none")} \\
\bottomrule
\end{{tabular}}

\section*{{Safety Status}}
\begin{{itemize}}
\item real\_orders\_enabled: false
\item external\_broker\_enabled: false
\item investopedia\_enabled: false
\item email\_phase: {_latex_escape(run_status.get("review_status", "REVIEW"))}
\item hard\_fail\_count: {_latex_escape(issues.get("hard_fail_count", 0))}
\item warning\_count: {_latex_escape(issues.get("soft_warning_count", 0))}
\end{{itemize}}

\section*{{Decision Summary}}
\begin{{itemize}}
\item continuous\_candidate: {_latex_escape(decision_context.get("continuous_candidate", "n/a"))}
\item final\_discrete\_candidate: {_latex_escape(decision_context.get("final_discrete_candidate", "n/a"))}
\item trade\_now\_edge: {_latex_escape(f"{_safe_float(cost_edge.get('trade_now_edge', 0.0)):.6f}")} 
\item first\_blocker: {_latex_escape(issues.get("first_blocker", "none"))}
\item all\_blockers: {_latex_escape(" | ".join(map(str, issues.get("all_blockers", ["none"]))))}
\end{{itemize}}

\section*{{Warum HOLD?}}
\begin{{itemize}}
{''.join(f"\\item {_latex_escape(item)}\n" for item in why_hold_items) or '\\item Keine zusaetzliche HOLD-Analyse verfuegbar.\n'}
\end{{itemize}}
{_latex_longtable(["Blocker", "Typ", "Einfluss", "Erklaerung"], hold_blocker_rows, "p{0.18\\linewidth}p{0.18\\linewidth}p{0.16\\linewidth}p{0.38\\linewidth}") if hold_blocker_rows else ''}

\section*{{Current Portfolio}}
{_latex_longtable(["Ticker", "Shares", "Latest Price", "Market Value", "Weight"], current_rows, "p{0.14\\linewidth}p{0.15\\linewidth}p{0.16\\linewidth}p{0.22\\linewidth}p{0.14\\linewidth}")} 

\section*{{Target Portfolio}}
{_latex_longtable(["Ticker", "Target Weight", "Target Shares", "Target Value"], target_rows, "p{0.16\\linewidth}p{0.18\\linewidth}p{0.18\\linewidth}p{0.24\\linewidth}")}

\section*{{Delta Orders}}
{_latex_longtable(["Action", "Ticker", "Current Shares", "Target Shares", "Order Shares", "Est. Value"], delta_rows, "p{0.12\\linewidth}p{0.12\\linewidth}p{0.16\\linewidth}p{0.16\\linewidth}p{0.16\\linewidth}p{0.18\\linewidth}")}

\section*{{Warnings / Risks}}
\begin{{itemize}}
{issue_block}
\end{{itemize}}

\section*{{Next Manual Action}}
\begin{{itemize}}
\item {_latex_escape(operator_instruction)}
\item Use \texttt{{outputs/manual\_simulator\_orders.csv}} for simulator orders.
\item Do not use \texttt{{outputs/order\_preview.csv}} for manual simulator orders.
\end{{itemize}}

\section*{{Charts}}
{''.join(chart_blocks) if chart_blocks else 'No charts available.\\\\\n'}

\section*{{Render Warnings}}
\begin{{itemize}}
{render_warning_block}
\end{{itemize}}

\section*{{Relevant Files}}
\begin{{itemize}}
{relevant_files_block}
\end{{itemize}}

\end{{document}}
"""


def _compile_pdf(tex_path: Path) -> tuple[Path | None, list[str]]:
    warnings: list[str] = []
    tex_path = tex_path.resolve()
    if shutil.which("pdflatex") is None:
        warnings.append("pdflatex unavailable; PDF build skipped.")
        return None, warnings
    output_dir = tex_path.parent.resolve()
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(output_dir),
        tex_path.name,
    ]
    try:
        run = subprocess.run(
            command,
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"pdflatex execution failed: {sanitize_for_output(exc)}")
        return None, warnings
    pdf_path = output_dir / f"{tex_path.stem}.pdf"
    log_excerpt = sanitize_for_output((run.stdout or "") + "\n" + (run.stderr or ""))
    fatal_markers = ["\n!", "Fatal error", "Emergency stop", "Undefined control sequence"]
    fatal_detected = any(marker in log_excerpt for marker in fatal_markers)
    if pdf_path.exists() and pdf_path.stat().st_size > 0 and not fatal_detected:
        return pdf_path, warnings
    if run.returncode != 0 or not pdf_path.exists() or pdf_path.stat().st_size <= 0:
        first_line = next((line.strip() for line in log_excerpt.splitlines() if line.strip()), "unknown pdflatex failure")
        warnings.append(f"pdflatex failed: {first_line}")
        return None, warnings
    return pdf_path, warnings


def build_daily_review_render_bundle(
    review: dict[str, Any],
    issues: dict[str, Any],
    output_dir: str | Path,
    *,
    subject: str,
    plain_text_body: str,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    charts_dir = output_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    render_warnings: list[str] = []

    chart_paths = {
        "current_portfolio_allocation": _safe_plot(
            charts_dir / "current_portfolio_allocation.png",
            "Current Portfolio Allocation",
            lambda: _plot_current_portfolio_allocation(review, charts_dir / "current_portfolio_allocation.png"),
            render_warnings,
        ),
        "current_vs_target_weights": _safe_plot(
            charts_dir / "current_vs_target_weights.png",
            "Current vs Target Weights",
            lambda: _plot_current_vs_target_weights(review, charts_dir / "current_vs_target_weights.png"),
            render_warnings,
        ),
        "nav_cash_summary": _safe_plot(
            charts_dir / "nav_cash_summary.png",
            "NAV / Cash / Risk Summary",
            lambda: _plot_nav_cash_summary(review, issues, charts_dir / "nav_cash_summary.png"),
            render_warnings,
        ),
        "risk_and_blockers": _safe_plot(
            charts_dir / "risk_and_blockers.png",
            "Risk and Blockers",
            lambda: _plot_risk_and_blockers(review, issues, charts_dir / "risk_and_blockers.png"),
            render_warnings,
        ),
    }

    tex_path = output_path / "daily_review_report.tex"
    html_path = output_path / "daily_review_email.html"
    pdf_path: Path | None = None

    html_text = _build_html_report(
        review,
        issues,
        subject,
        plain_text_body,
        chart_paths,
        render_warnings,
        pdf_path=None,
    )
    latex_text = _build_latex_report(
        review,
        issues,
        subject,
        chart_paths,
        render_warnings,
        output_path,
    )
    _atomic_write_text(html_path, sanitize_for_output(html_text))
    _atomic_write_text(tex_path, sanitize_for_output(latex_text))

    pdf_path, pdf_warnings = _compile_pdf(tex_path)
    render_warnings.extend(pdf_warnings)
    if pdf_path is not None and (not pdf_path.exists() or pdf_path.stat().st_size <= 0):
        render_warnings.append("PDF path returned but file missing or empty.")
        pdf_path = None

    html_text = _build_html_report(
        review,
        issues,
        subject,
        plain_text_body,
        chart_paths,
        render_warnings,
        pdf_path=pdf_path,
    )
    _atomic_write_text(html_path, sanitize_for_output(html_text))

    return {
        "html_path": html_path,
        "html_text": html_text,
        "tex_path": tex_path,
        "tex_text": latex_text,
        "pdf_path": pdf_path,
        "pdf_built": pdf_path is not None,
        "chart_paths": chart_paths,
        "warnings": render_warnings,
        "attachment_paths": [pdf_path] if pdf_path is not None else [],
    }
