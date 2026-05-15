"""Readable explanations for selected decisions and asset changes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def explain_selected_decision(
    selection_result: Any,
    execution_gate_result: Any,
    *,
    data_quality_report: dict[str, Any] | None = None,
    model_confidence: dict[str, Any] | None = None,
    reconciliation_result: dict[str, Any] | None = None,
    validation_result: dict[str, Any] | None = None,
) -> str:
    """Build a compact text explanation for the chosen candidate."""

    selected = selection_result.selected_score
    return "\n".join(
        [
            f"Selected Candidate: {selection_result.selected_candidate.name}",
            "Selection utility: robust candidate ranking based on net_robust_score after risk, costs and execution buffers.",
            f"Why this candidate: net_robust_score={selected.net_robust_score:.6f}, delta_vs_hold={selected.delta_vs_hold:.6f}, delta_vs_cash={selected.delta_vs_cash:.6f}",
            f"Why not HOLD: probability_beats_hold={selected.probability_beats_hold:.2%}",
            f"Why not CASH: probability_beats_cash={selected.probability_beats_cash:.2%}",
            f"Biggest tail risk: CVaR 5={selected.cvar_5:.2%}, worst_scenario={selected.worst_scenario}",
            f"Execution block: {getattr(execution_gate_result, 'reason', 'n/a')}",
            f"Data Quality Score: {float((data_quality_report or {}).get('global_data_quality_score', 0.0)):.3f}",
            f"Model Confidence Score: {float((model_confidence or {}).get('model_confidence_score', 0.0)):.3f}",
            f"Reconciliation Status: {(reconciliation_result or {}).get('status', 'SKIP')}",
            f"Pre-Trade Validation: {'PASS' if (validation_result or {}).get('ok', True) else 'FAIL'}",
        ]
    )


def explain_asset_changes(
    w_current: pd.Series,
    w_target: pd.Series,
    *,
    forecast_table: pd.DataFrame | None = None,
    exposure_matrix: pd.DataFrame | None = None,
    min_delta: float = 0.01,
) -> pd.DataFrame:
    """Explain material asset changes between current and target weights."""

    current = w_current.astype(float)
    target = w_target.astype(float)
    index = pd.Index(sorted(set(current.index).union(target.index)), name="ticker")
    current = current.reindex(index).fillna(0.0)
    target = target.reindex(index).fillna(0.0)
    delta = target - current
    rows: list[dict[str, Any]] = []
    forecast_table = forecast_table if isinstance(forecast_table, pd.DataFrame) else pd.DataFrame()
    exposure_matrix = exposure_matrix if isinstance(exposure_matrix, pd.DataFrame) else pd.DataFrame()
    for ticker in index:
        if abs(float(delta.loc[ticker])) < min_delta:
            continue
        positive_driver = "direct_forecast_positive" if float(forecast_table.get("expected_return_3m", pd.Series(dtype=float)).get(ticker, 0.0)) > 0 else "defensive_rebalance"
        negative_driver = "tail_risk" if float(forecast_table.get("downside_risk_3m", pd.Series(dtype=float)).get(ticker, 0.0)) > 0.10 else "none"
        factor_summary = "n/a"
        if ticker in exposure_matrix.index and not exposure_matrix.empty:
            top_exposures = exposure_matrix.loc[ticker].abs().sort_values(ascending=False).head(3).index.tolist()
            factor_summary = ", ".join(str(name) for name in top_exposures) if top_exposures else "n/a"
        rows.append(
            {
                "ticker": ticker,
                "current_weight": float(current.loc[ticker]),
                "target_weight": float(target.loc[ticker]),
                "delta_weight": float(delta.loc[ticker]),
                "main_positive_drivers": positive_driver,
                "main_negative_drivers": negative_driver,
                "confidence": float(forecast_table.get("signal_confidence", pd.Series(dtype=float)).get(ticker, 0.0)),
                "scenario_contribution": float(forecast_table.get("expected_return_3m", pd.Series(dtype=float)).get(ticker, 0.0)),
                "factor_exposure_summary": factor_summary,
            }
        )
    return pd.DataFrame(rows)


def explain_factor_drivers(exposure_matrix: pd.DataFrame, factor_forecast_df: pd.DataFrame) -> str:
    """Build a compact factor-driver explanation."""

    if exposure_matrix.empty or factor_forecast_df.empty:
        return "Factor drivers unavailable or direct-only fallback active."
    strongest_factors = factor_forecast_df.sort_values("confidence", ascending=False).head(5)["factor"].astype(str).tolist()
    return "Top factor drivers: " + ", ".join(strongest_factors)


def save_explainability_reports(
    text: str,
    asset_changes: pd.DataFrame,
    *,
    text_path: str | Path,
    csv_path: str | Path,
) -> None:
    """Persist explainability outputs."""

    text_file = Path(text_path)
    csv_file = Path(csv_path)
    text_file.parent.mkdir(parents=True, exist_ok=True)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    text_file.write_text(text + "\n", encoding="utf-8")
    asset_changes.to_csv(csv_file, index=False)
