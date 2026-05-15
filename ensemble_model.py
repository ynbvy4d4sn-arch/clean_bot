"""Lightweight ensemble reporting across several portfolio constructions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_model_ensemble_outputs(
    *,
    optimizer_target: pd.Series,
    defensive_cash: pd.Series,
    momentum_tilt: pd.Series,
    conditional_factor_target: pd.Series | None = None,
    risk_parity_like: pd.Series | None = None,
    min_variance_like: pd.Series | None = None,
) -> dict[str, Any]:
    """Build a simple ensemble report and consensus allocation."""

    models: dict[str, pd.Series] = {
        "optimizer_target": optimizer_target.astype(float),
        "defensive_cash": defensive_cash.astype(float),
        "momentum_tilt": momentum_tilt.astype(float),
    }
    if conditional_factor_target is not None and not conditional_factor_target.empty:
        models["conditional_factor_target"] = conditional_factor_target.astype(float)
    if risk_parity_like is not None and not risk_parity_like.empty:
        models["risk_parity_like"] = risk_parity_like.astype(float)
    if min_variance_like is not None and not min_variance_like.empty:
        models["min_variance_like"] = min_variance_like.astype(float)

    index = pd.Index(sorted(set().union(*[series.index.tolist() for series in models.values()])), name="ticker")
    aligned = {name: series.reindex(index).fillna(0.0) for name, series in models.items()}
    model_weights = pd.Series(1.0 / len(aligned), index=list(aligned.keys()), dtype=float)
    consensus = sum(aligned[name] * float(model_weights[name]) for name in aligned)

    pairwise_distances: list[float] = []
    names = list(aligned.keys())
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            pairwise_distances.append(
                float((aligned[left_name] - aligned[right_name]).abs().sum())
            )
    dispersion = float(sum(pairwise_distances) / len(pairwise_distances)) if pairwise_distances else 0.0
    agreement_score = float(max(0.0, 1.0 - min(dispersion / 2.0, 1.0)))

    report_df = pd.DataFrame(
        [
            {
                "model": name,
                "weight": float(model_weights[name]),
                "gross_exposure": float(aligned[name].abs().sum()),
            }
            for name in names
        ]
    )
    text = (
        f"Agreement Score: {agreement_score:.3f}\n"
        f"Dispersion: {dispersion:.3f}\n"
        f"Models: {', '.join(names)}\n"
    )
    return {
        "candidate_weights": aligned,
        "agreement_score": agreement_score,
        "dispersion": dispersion,
        "consensus_allocation": consensus,
        "model_weights": model_weights,
        "report_df": report_df,
        "text": text,
    }


def save_model_ensemble_report(result: dict[str, Any], csv_path: str | Path, txt_path: str | Path) -> None:
    """Persist ensemble outputs."""

    csv_file = Path(csv_path)
    txt_file = Path(txt_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    txt_file.parent.mkdir(parents=True, exist_ok=True)
    result.get("report_df", pd.DataFrame()).to_csv(csv_file, index=False)
    txt_file.write_text(str(result.get("text", "")), encoding="utf-8")
