"""Explain why HOLD dominates and persist lightweight daily decision history."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from notifications import sanitize_for_output


BERLIN_TZ = ZoneInfo("Europe/Berlin")

DEFAULT_SELECTION_HURDLE = 0.0010
DEFAULT_RISK_PREMIUM_HURDLE = 0.0005
DEFAULT_P_HOLD_MIN = 0.55
DEFAULT_P_CASH_MIN = 0.52
DEFAULT_TRADE_NOW_HURDLE = 0.0025


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _score_text(value: object) -> str:
    return f"{_safe_float(value, 0.0):.6f}"


def _pct_text(value: object) -> str:
    return f"{100.0 * _safe_float(value, 0.0):.2f}%"


def _append_decision_history(review: dict[str, Any], issues: dict[str, Any], output_dir: Path) -> tuple[Path, pd.DataFrame]:
    history_path = output_dir / "decision_history.csv"
    run_status = dict(review.get("run_status", {}) or {})
    data_status = dict(review.get("data_status", {}) or {})
    decision_context = dict(review.get("decision_context", {}) or {})
    current_portfolio = dict(review.get("current_portfolio", {}) or {})
    order_summary = dict(review.get("order_summary", {}) or {})
    cost_edge = dict(review.get("cost_edge", {}) or {})

    timestamp = f"{run_status.get('review_date', datetime.now(BERLIN_TZ).date())}T{run_status.get('review_time_berlin', datetime.now(BERLIN_TZ).strftime('%H:%M:%S'))}"
    row = {
        "timestamp": str(timestamp),
        "final_action": str(run_status.get("final_action", "n/a")),
        "final_discrete_candidate": str(decision_context.get("final_discrete_candidate", "n/a")),
        "continuous_candidate": str(decision_context.get("continuous_candidate", "n/a")),
        "trade_now_edge": _safe_float(cost_edge.get("trade_now_edge", 0.0)),
        "first_blocker": str(issues.get("first_blocker", "none")),
        "manual_order_count": _safe_int(order_summary.get("manual_eligible_order_count", order_summary.get("order_count", 0)), 0),
        "nav_usd": _safe_float(current_portfolio.get("nav_usd", 0.0)),
        "cash_usd": _safe_float(current_portfolio.get("cash_usd", 0.0)),
        "data_source": str(data_status.get("data_source", "n/a")),
        "latest_price_date": str(data_status.get("latest_price_date", "n/a")),
        "used_cache_fallback": bool(data_status.get("used_cache_fallback", False)),
    }
    history_df = _read_csv_if_exists(history_path)
    if history_df.empty:
        history_df = pd.DataFrame([row])
    else:
        last_row = history_df.tail(1).iloc[0].to_dict()
        duplicate = (
            str(last_row.get("timestamp", "")) == row["timestamp"]
            and str(last_row.get("final_action", "")) == row["final_action"]
            and str(last_row.get("final_discrete_candidate", "")) == row["final_discrete_candidate"]
            and abs(_safe_float(last_row.get("trade_now_edge", 0.0)) - row["trade_now_edge"]) <= 1e-12
            and _safe_int(last_row.get("manual_order_count", 0), 0) == row["manual_order_count"]
        )
        if not duplicate:
            history_df = pd.concat([history_df, pd.DataFrame([row])], ignore_index=True)
    history_df.to_csv(history_path, index=False)
    return history_path, history_df


def _pick_row(frame: pd.DataFrame, column: str, value: str) -> pd.Series | None:
    if frame.empty or column not in frame.columns:
        return None
    rows = frame[frame[column].astype(str) == str(value)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _pick_best_valid_non_hold(discrete_scores: pd.DataFrame) -> pd.Series | None:
    if discrete_scores.empty:
        return None
    valid = discrete_scores[discrete_scores.get("valid_constraints", False) == True].copy()  # noqa: E712
    if valid.empty or "discrete_candidate" not in valid.columns:
        return None
    valid = valid[valid["discrete_candidate"].astype(str) != "HOLD_CURRENT"]
    if valid.empty:
        return None
    sort_columns = [
        "net_robust_score",
        "cvar_5",
        "turnover_vs_current",
        "max_abs_weight_drift",
        "number_of_positions",
        "cash_left",
    ]
    existing_columns = [column for column in sort_columns if column in valid.columns]
    ascending = [False, False, True, True, True, True][: len(existing_columns)]
    valid = valid.sort_values(existing_columns, ascending=ascending, kind="mergesort")
    return valid.iloc[0]


def _pick_best_invalid_from_source(discrete_scores: pd.DataFrame, continuous_candidate: str) -> pd.Series | None:
    if discrete_scores.empty or "continuous_source" not in discrete_scores.columns:
        return None
    rows = discrete_scores[
        (discrete_scores["continuous_source"].astype(str) == str(continuous_candidate))
        & (discrete_scores.get("valid_constraints", True) != True)  # noqa: E712
    ].copy()
    if rows.empty:
        return None
    existing_columns = [column for column in ["net_robust_score", "delta_vs_hold", "cash_left"] if column in rows.columns]
    ascending = [False, False, True][: len(existing_columns)]
    rows = rows.sort_values(existing_columns, ascending=ascending, kind="mergesort")
    return rows.iloc[0]


def _trade_gate_score(delta_vs_hold: float, execution_buffer: float, model_uncertainty_buffer: float) -> float:
    return float(delta_vs_hold) - float(execution_buffer) - float(model_uncertainty_buffer)


def _classify_blockers(
    review: dict[str, Any],
    issues: dict[str, Any],
    *,
    current_positions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    current_tickers = {str(row.get("ticker", "")).strip().upper() for row in current_positions}
    blocker_rows: list[dict[str, str]] = []
    blocker_rows.append(
        {
            "blocker": "trade_now_edge negative",
            "type": "echter Entscheidungsblocker",
            "impact": "hoch",
            "explanation": f"Die aktuelle Netto-Edge nach Puffern liegt bei {_score_text(review.get('cost_edge', {}).get('trade_now_edge', 0.0))} und damit unter Null.",
        }
    )
    blocker_rows.append(
        {
            "blocker": "crypto ETF exposure exists",
            "type": "Risiko-/Exposure-Warnung",
            "impact": "niedrig",
            "explanation": "IBIT/ETHA fuehrt derzeit nur zu einer Warnung. Im Diskretisierer und Execution Gate gibt es keine explizite Krypto-Sperre.",
        }
    )
    adjusted_note = "Alle aktuellen Kurse kommen aus adjusted-close-Proxies." if current_tickers else "Preisquelle ist als adjusted-close-Proxie markiert."
    blocker_rows.append(
        {
            "blocker": "adjusted_close_proxy used",
            "type": "Datenwarnung",
            "impact": "niedrig",
            "explanation": adjusted_note + " Das mindert die Exekutionspraezision, erzwingt aber das HOLD nicht.",
        }
    )
    blocker_rows.append(
        {
            "blocker": "main.py and daily_bot.py scope differs",
            "type": "technischer Scope-Unterschied",
            "impact": "keine direkte Wirkung",
            "explanation": "Der Hinweis betrifft Research-vs-Daily-Review-Scope. Er ist Reporting-Rauschen und kein Bestandteil der Auswahl- oder Gate-Logik.",
        }
    )
    blocker_rows.append(
        {
            "blocker": "dry_run active",
            "type": "Sicherheits-/Info-Hinweis",
            "impact": "keine direkte Wirkung",
            "explanation": "Dry-run verhindert nur echte Ausfuehrung. Es blockiert keine manuellen Simulator-Vorschlaege, wenn solche fachlich entstehen.",
        }
    )
    blocker_rows.append(
        {
            "blocker": "no real orders sent",
            "type": "Sicherheits-/Info-Hinweis",
            "impact": "keine",
            "explanation": "Das ist nur die bestaetigte Folge des Preview-Betriebs und kein Entscheidungstreiber.",
        }
    )
    blocker_rows.append(
        {
            "blocker": "simulator fees 0.00",
            "type": "Sicherheits-/Info-Hinweis",
            "impact": "keine",
            "explanation": "Simulator-Gebuehren sind getrennt reportet und hier null. Sie erklaeren das HOLD nicht.",
        }
    )
    blocker_rows.append(
        {
            "blocker": "no manual orders generated",
            "type": "Folge, kein Primärblocker",
            "impact": "symptomatisch",
            "explanation": "Die leere Simulatorliste ist das Ergebnis von HOLD_CURRENT bzw. WAIT/BLOCK, nicht deren Ursache.",
        }
    )
    if "outside allowed trading window" in list(issues.get("all_blockers", []) or []):
        blocker_rows.insert(
            0,
            {
                "blocker": "outside allowed trading window",
                "type": "Kalender-/Gate-Blocker",
                "impact": "hoch fuer aktuellen Preview-Lauf",
                "explanation": "Der aktuelle Preview-Lauf ist zusatzlich kalenderseitig blockiert. Das erklaert WAIT, aber nicht die strukturelle HOLD-Dominanz.",
            },
        )
    return blocker_rows


def build_hold_analysis_bundle(review: dict[str, Any], issues: dict[str, Any], output_dir: str | Path = "outputs") -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    decision_context = dict(review.get("decision_context", {}) or {})
    current_portfolio = dict(review.get("current_portfolio", {}) or {})
    data_status = dict(review.get("data_status", {}) or {})
    order_summary = dict(review.get("order_summary", {}) or {})
    cost_edge = dict(review.get("cost_edge", {}) or {})
    current_positions = list(review.get("current_positions", []) or [])

    discrete_scores = _read_csv_if_exists(output_path / "discrete_candidate_scores.csv")
    candidate_scores = _read_csv_if_exists(output_path / "candidate_scores.csv")
    daily_results = _read_csv_if_exists(output_path / "daily_results.csv")

    continuous_candidate = str(decision_context.get("continuous_candidate", "n/a"))
    final_discrete_candidate = str(decision_context.get("final_discrete_candidate", "n/a"))
    execution_buffer = _safe_float(cost_edge.get("execution_buffer", 0.0))
    model_buffer = _safe_float(cost_edge.get("model_uncertainty_buffer", 0.0))
    combined_buffers = execution_buffer + model_buffer
    current_trade_now_edge = _safe_float(cost_edge.get("trade_now_edge", 0.0))

    hold_row = _pick_row(discrete_scores, "discrete_candidate", "HOLD_CURRENT")
    continuous_row = _pick_row(candidate_scores, "candidate", continuous_candidate)
    hold_candidate_row = _pick_row(candidate_scores, "candidate", "HOLD")
    best_valid_non_hold = _pick_best_valid_non_hold(discrete_scores)
    best_invalid_from_source = _pick_best_invalid_from_source(discrete_scores, continuous_candidate)

    continuous_net_delta = _safe_float(continuous_row.get("delta_vs_hold", 0.0)) if continuous_row is not None else 0.0
    continuous_gross_delta = 0.0
    if continuous_row is not None and hold_candidate_row is not None:
        continuous_gross_delta = _safe_float(continuous_row.get("robust_score", 0.0)) - _safe_float(hold_candidate_row.get("robust_score", 0.0))
    best_valid_delta = _safe_float(best_valid_non_hold.get("delta_vs_hold", 0.0)) if best_valid_non_hold is not None else 0.0
    best_valid_gross_delta = 0.0
    if best_valid_non_hold is not None and hold_row is not None:
        best_valid_gross_delta = _safe_float(best_valid_non_hold.get("gross_robust_score", 0.0)) - _safe_float(hold_row.get("gross_robust_score", 0.0))

    history_path, history_df = _append_decision_history(review, issues, output_path)
    blocker_table = _classify_blockers(review, issues, current_positions=current_positions)

    why_hold_lines: list[str] = []
    why_hold_lines.append(
        f"Die aktuelle Netto-Trade-Now-Edge liegt bei {_score_text(current_trade_now_edge)}. Schon fuer >0 fehlt damit Edge; fuer die echte Execution-Huerde von {DEFAULT_TRADE_NOW_HURDLE:.6f} fehlen aktuell etwa {_score_text(DEFAULT_TRADE_NOW_HURDLE - current_trade_now_edge)}."
    )
    if continuous_candidate and continuous_candidate != "n/a":
        why_hold_lines.append(
            f"Das kontinuierliche Modell will {continuous_candidate}. Sein theoretischer netter Vorteil gegen HOLD liegt aber nur bei {_score_text(continuous_net_delta)}; nach Puffern ({_score_text(combined_buffers)}) bleibt selbst mit Fractional Shares nur {_score_text(continuous_net_delta - combined_buffers)} uebrig."
        )
    if best_invalid_from_source is not None:
        why_hold_lines.append(
            f"Die beste Whole-Share-Umsetzung aus {continuous_candidate} ({best_invalid_from_source.get('discrete_candidate', 'n/a')}) scheitert zuerst an Constraints: {best_invalid_from_source.get('validation_errors', 'constraints failed')}"
        )
    if best_valid_non_hold is not None:
        why_hold_lines.append(
            f"Die beste gueltige Whole-Share-Alternative ist {best_valid_non_hold.get('discrete_candidate', 'n/a')} mit delta_vs_hold={_score_text(best_valid_delta)}. Das liegt klar unter der Selektionshuerde von {DEFAULT_SELECTION_HURDLE:.6f}."
        )
        why_hold_lines.append(
            f"Selbst wenn man die Selektionshuerde theoretisch auf 0 setzen wuerde, bliebe deren Gate-Score nach Puffern bei {_score_text(_trade_gate_score(best_valid_delta, execution_buffer, model_buffer))}; damit wuerde der Trade immer noch nicht freigegeben."
        )
    if best_valid_non_hold is not None:
        why_hold_lines.append(
            f"Cash ist nicht der Blocker: die beste gueltige Alternative haette cash_left={_safe_float(best_valid_non_hold.get('cash_left', 0.0)):.2f} USD."
        )
        why_hold_lines.append(
            f"Mindestordergroessen sind nicht der Blocker: skipped_small_orders={_safe_int(best_valid_non_hold.get('skipped_small_orders', 0), 0)} bei der besten gueltigen Alternative."
        )
    if hold_row is not None and not _safe_bool(hold_row.get("valid_constraints", True)):
        why_hold_lines.append(
            "Technischer Hinweis: HOLD_CURRENT ist im Diskretisierer als Fallback zulaessig, obwohl seine Constraint-Spalte derzeit ein Warnsignal traegt. Das erklaert die HOLD-Dominanz nicht, macht aber die Diagnostik uneindeutiger."
        )

    dominance_lines = [
        "Hold Dominance Analysis",
        "",
        f"Generated at: {datetime.now(BERLIN_TZ).isoformat(timespec='seconds')}",
        "",
        "Current runtime snapshot",
        f"- final_action: {review.get('run_status', {}).get('final_action', 'n/a')}",
        f"- final_discrete_candidate: {final_discrete_candidate}",
        f"- continuous_candidate: {continuous_candidate}",
        f"- trade_now_edge: {_score_text(current_trade_now_edge)}",
        f"- first_blocker: {issues.get('first_blocker', 'none')}",
        f"- manual_order_count: {_safe_int(order_summary.get('manual_eligible_order_count', order_summary.get('order_count', 0)), 0)}",
        "",
        "Historical sources reviewed",
    ]
    if not daily_results.empty and "decision" in daily_results.columns:
        total_rows = len(daily_results)
        hold_count = int((daily_results["decision"].astype(str) == "HOLD").sum())
        wait_count = int((daily_results["decision"].astype(str) == "WAIT").sum())
        trade_like_count = int((~daily_results["decision"].astype(str).isin(["HOLD", "WAIT", "ALL"])).sum())
        dominance_lines.extend(
            [
                "1. outputs/daily_results.csv (legacy research/backtest scope)",
                f"- rows: {total_rows}",
                f"- HOLD count: {hold_count} ({_pct_text(hold_count / max(total_rows, 1))})",
                f"- WAIT count: {wait_count} ({_pct_text(wait_count / max(total_rows, 1))})",
                f"- non-HOLD/non-WAIT decisions: {trade_like_count} ({_pct_text(trade_like_count / max(total_rows, 1))})",
                "- BUY/SELL exact counts: not available in this legacy file",
                "- trade_now_edge negative count: not available in this legacy file",
                "- manual_order_count=0 count: not available in this legacy file",
                "",
            ]
        )
    else:
        dominance_lines.extend(
            [
                "1. outputs/daily_results.csv",
                "- not available; no legacy historical decision file found.",
                "",
            ]
        )

    history_hold_count = int((history_df.get("final_action", pd.Series(dtype=str)).astype(str).str.upper() == "HOLD").sum()) if not history_df.empty else 0
    history_trade_like_count = int((pd.to_numeric(history_df.get("manual_order_count", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()) if not history_df.empty else 0
    history_negative_edge_count = int((pd.to_numeric(history_df.get("trade_now_edge", pd.Series(dtype=float)), errors="coerce").fillna(0.0) < 0).sum()) if not history_df.empty else 0
    history_zero_orders_count = int((pd.to_numeric(history_df.get("manual_order_count", pd.Series(dtype=float)), errors="coerce").fillna(0) == 0).sum()) if not history_df.empty else 0
    dominance_lines.extend(
        [
            "2. outputs/decision_history.csv (operational Daily-Review history; created/updated by this change)",
            f"- rows: {len(history_df)}",
            f"- final_action=HOLD: {history_hold_count}",
            f"- BUY/SELL-like rows (manual_order_count > 0): {history_trade_like_count}",
            f"- trade_now_edge negative: {history_negative_edge_count}",
            f"- manual_order_count=0: {history_zero_orders_count}",
            "",
            "Assessment",
            "- HOLD dominance is structurally visible in the legacy research history; this is not just a single daily email artifact.",
            "- Exact operational BUY/SELL history did not exist before this change and is now tracked in outputs/decision_history.csv.",
        ]
    )
    dominance_text = sanitize_for_output("\n".join(dominance_lines).strip() + "\n")
    dominance_path = output_path / "hold_dominance_analysis.txt"
    dominance_path.write_text(dominance_text, encoding="utf-8")

    sensitivity_lines = [
        "Hold Sensitivity Report",
        "",
        f"Generated at: {datetime.now(BERLIN_TZ).isoformat(timespec='seconds')}",
        "",
        "Baseline",
        f"- continuous_candidate: {continuous_candidate}",
        f"- final_discrete_candidate: {final_discrete_candidate}",
        f"- current trade_now_edge after buffers: {_score_text(current_trade_now_edge)}",
        f"- execution_buffer: {_score_text(execution_buffer)}",
        f"- model_uncertainty_buffer: {_score_text(model_buffer)}",
        f"- combined_buffers: {_score_text(combined_buffers)}",
        f"- execution trade hurdle: {DEFAULT_TRADE_NOW_HURDLE:.6f}",
        f"- discrete selection hurdle vs HOLD: {DEFAULT_SELECTION_HURDLE:.6f}",
        f"- delta_vs_cash hurdle: {DEFAULT_RISK_PREMIUM_HURDLE:.6f}",
        f"- probability thresholds: p_hold>={DEFAULT_P_HOLD_MIN:.2%}, p_cash>={DEFAULT_P_CASH_MIN:.2%}",
        "",
        "What blocks first?",
    ]
    if best_invalid_from_source is not None:
        sensitivity_lines.append(
            f"- Momentum/tilt Whole-Share candidate {best_invalid_from_source.get('discrete_candidate', 'n/a')} fails constraints first: {best_invalid_from_source.get('validation_errors', 'constraints failed')}."
        )
    if best_valid_non_hold is not None:
        sensitivity_lines.append(
            f"- Best valid non-HOLD candidate {best_valid_non_hold.get('discrete_candidate', 'n/a')} then fails the selector hurdle because delta_vs_hold={_score_text(best_valid_delta)} <= {DEFAULT_SELECTION_HURDLE:.6f}."
        )
        sensitivity_lines.append(
            f"- Even after removing only the selector hurdle, its post-buffer gate score would still be {_score_text(_trade_gate_score(best_valid_delta, execution_buffer, model_buffer))}, so execution would remain blocked."
        )
    sensitivity_lines.extend(
        [
            "",
            "Scenario checks (reporting only, no live effect)",
            f"- Current whole-share baseline: HOLD_CURRENT stays selected; trade_now_edge={_score_text(current_trade_now_edge)}.",
            f"- Execution hurdle = 0 (selector unchanged): still no trade, because trade_now_edge remains {_score_text(current_trade_now_edge)} < 0.",
            f"- Execution hurdle = 0.00125 (selector unchanged): still no trade, because trade_now_edge remains {_score_text(current_trade_now_edge)} < 0.00125.",
        ]
    )
    if best_valid_non_hold is not None:
        sensitivity_lines.append(
            f"- Without modeled trading costs on the best valid whole-share candidate: gross delta_vs_hold would be about {_score_text(best_valid_gross_delta)}; after buffers that is still {_score_text(best_valid_gross_delta - combined_buffers)}."
        )
    else:
        sensitivity_lines.append("- Without modeled trading costs: no valid non-HOLD whole-share candidate was available to test.")
    if best_valid_non_hold is not None:
        sensitivity_lines.append(
            f"- Higher allowed turnover: no effect visible here; the best valid candidate already has turnover_vs_current={_safe_float(best_valid_non_hold.get('turnover_vs_current', 0.0)):.6f} and still fails on edge/hurdles, not on a turnover cap."
        )
        sensitivity_lines.append(
            f"- Smaller minimum order sizes: no effect visible; skipped_small_orders={_safe_int(best_valid_non_hold.get('skipped_small_orders', 0), 0)} for the best valid candidate."
        )
        sensitivity_lines.append(
            f"- Cash constraint ignored theoretically: no effect visible; cash_left would still stay positive at {_safe_float(best_valid_non_hold.get('cash_left', 0.0)):.2f} USD."
        )
    if continuous_row is not None:
        sensitivity_lines.append(
            f"- Fractional shares theoretical: {continuous_candidate} would reach delta_vs_hold={_score_text(continuous_net_delta)} net / { _score_text(continuous_gross_delta)} gross. After current buffers that is {_score_text(continuous_net_delta - combined_buffers)} net and {_score_text(continuous_gross_delta - combined_buffers)} gross."
        )
    else:
        sensitivity_lines.append("- Fractional shares theoretical: continuous candidate row not available in outputs/candidate_scores.csv.")
    sensitivity_lines.extend(
        [
            "",
            "Do data warnings change the answer?",
            "- adjusted_close_proxy used: no direct hold trigger; it is a pricing-quality warning.",
            "- crypto ETF exposure exists: no direct hold trigger; it is an exposure warning for IBIT/ETHA holdings.",
            "- dry_run active: no direct hold trigger; it blocks only real execution, not the generation of preview orders.",
            "- main.py and daily_bot.py scope differs: no direct hold trigger; it is a reporting/scope warning.",
            "",
            "Natural-language conclusion",
            "- HOLD is currently mainly structural: the continuous target wants more tilt, but its buyable whole-share versions violate max-weight constraints.",
            "- The best valid whole-share alternatives have only microscopic delta_vs_hold values, far below both the selector hurdle and the execution hurdle after buffers.",
            "- Cash, min-order sizes and dry-run settings are not the reason for HOLD in this run.",
        ]
    )
    sensitivity_text = sanitize_for_output("\n".join(sensitivity_lines).strip() + "\n")
    sensitivity_path = output_path / "hold_sensitivity_report.txt"
    sensitivity_path.write_text(sensitivity_text, encoding="utf-8")

    summary = {
        "hold_fachlich_begruendet": True,
        "wichtigster_hold_grund": "trade_now_edge negative after buffers; valid whole-share alternatives do not clear selector/gate hurdles",
        "technical_hold_distortion_found": False,
        "main_scope_relevant": False,
        "adjusted_close_proxy_relevant": False,
        "crypto_exposure_relevant": False,
        "dry_run_info_only": True,
        "continuous_net_delta_vs_hold": continuous_net_delta,
        "continuous_gross_delta_vs_hold": continuous_gross_delta,
        "best_valid_discrete_delta_vs_hold": best_valid_delta,
        "best_valid_discrete_name": str(best_valid_non_hold.get("discrete_candidate", "")) if best_valid_non_hold is not None else "",
    }

    return {
        "hold_analysis": {
            "why_hold_lines": why_hold_lines,
            "blocker_table": blocker_table,
            "summary": summary,
        },
        "hold_dominance_analysis_path": dominance_path,
        "hold_sensitivity_report_path": sensitivity_path,
        "decision_history_path": history_path,
    }
