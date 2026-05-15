"""Model-confidence scoring and governance reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def compute_model_confidence(
    forecast_report: Any,
    factor_report: Any,
    scenario_report: Any,
    optimizer_result: Any,
    data_quality_report: dict[str, Any],
) -> dict[str, Any]:
    """Estimate a simple model-confidence score and uncertainty buffer."""

    warnings: list[str] = []
    forecast_table = getattr(forecast_report, "table", forecast_report)
    factor_df = factor_report if isinstance(factor_report, pd.DataFrame) else pd.DataFrame()
    scenario_df = getattr(scenario_report, "summary", scenario_report)
    if not isinstance(scenario_df, pd.DataFrame):
        scenario_df = pd.DataFrame()

    avg_signal_confidence = float(forecast_table["signal_confidence"].mean()) if isinstance(forecast_table, pd.DataFrame) and "signal_confidence" in forecast_table else 0.50
    forecast_dispersion = float(forecast_table["expected_return_3m"].std(ddof=0)) if isinstance(forecast_table, pd.DataFrame) and "expected_return_3m" in forecast_table else 0.0
    data_quality_score = float(data_quality_report.get("global_data_quality_score", 0.50))
    optimizer_success = 1.0 if bool(getattr(optimizer_result, "success", False)) else 0.5
    factor_available = 1.0 if not factor_df.empty else 0.5
    scenario_breadth = float(scenario_df["mean_asset_return"].std(ddof=0)) if "mean_asset_return" in scenario_df else 0.0

    score = 0.35 * avg_signal_confidence
    score += 0.25 * data_quality_score
    score += 0.15 * optimizer_success
    score += 0.15 * factor_available
    score += 0.10 * max(0.0, 1.0 - min(abs(forecast_dispersion - scenario_breadth), 1.0))
    score = float(min(max(score, 0.0), 1.0))

    if data_quality_score < 0.70:
        warnings.append("Data quality is mediocre; model confidence was reduced.")
    if not factor_df.empty:
        mode = "conditional_factor"
    else:
        mode = "direct_only"
    if not bool(getattr(optimizer_result, "success", False)):
        mode = "fallback"
        warnings.append("Optimizer did not report a clean success state.")

    model_uncertainty_buffer = float(0.0005 + (1.0 - score) * 0.0025)
    return {
        "model_confidence_score": score,
        "model_uncertainty_buffer": model_uncertainty_buffer,
        "warnings": warnings,
        "mode": mode,
        "details": {
            "avg_signal_confidence": avg_signal_confidence,
            "forecast_dispersion": forecast_dispersion,
            "data_quality_score": data_quality_score,
            "optimizer_success": optimizer_success,
            "factor_available": factor_available,
            "scenario_breadth": scenario_breadth,
        },
    }


def save_model_governance_report(result: dict[str, Any], csv_path: str | Path, txt_path: str | Path) -> None:
    """Persist model-governance outputs to CSV and TXT."""

    csv_file = Path(csv_path)
    txt_file = Path(txt_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    txt_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_confidence_score": result.get("model_confidence_score", 0.0),
                "model_uncertainty_buffer": result.get("model_uncertainty_buffer", 0.0),
                "mode": result.get("mode", "fallback"),
                **dict(result.get("details", {})),
            }
        ]
    ).to_csv(csv_file, index=False)
    txt_file.write_text(
        "\n".join(
            [
                f"Mode: {result.get('mode', 'fallback')}",
                f"Model Confidence Score: {result.get('model_confidence_score', 0.0):.3f}",
                f"Model Uncertainty Buffer: {result.get('model_uncertainty_buffer', 0.0):.6f}",
                f"Warnings: {'; '.join(result.get('warnings', [])) or 'none'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
