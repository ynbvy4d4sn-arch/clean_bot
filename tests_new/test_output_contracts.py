from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs")


def test_active_scenario_probability_report_contract():
    path = OUTPUT_DIR / "active_scenario_probabilities.csv"
    assert path.exists(), "Run daily_bot.py --dry-run --skip-submit before checking output contracts."

    frame = pd.read_csv(path)
    required_columns = {
        "model_scope",
        "active_for_final_allocation",
        "source_module",
        "scenario",
        "probability",
        "probability_source",
    }

    assert required_columns.issubset(frame.columns)
    assert set(frame["model_scope"]) == {"active_final_allocation_solver"}
    assert set(frame["source_module"]) == {"scenario_daily_pipeline"}
    assert frame["active_for_final_allocation"].astype(str).str.lower().eq("true").all()
    assert set(frame["probability_source"]) == {"dynamic_regime_probability_model"}
    assert abs(float(frame["probability"].sum()) - 1.0) < 1e-9


def test_scenario_risk_probability_report_is_marked_diagnostic():
    path = OUTPUT_DIR / "scenario_risk_probability_report.csv"
    assert path.exists(), "Run daily_bot.py --dry-run --skip-submit before checking output contracts."

    frame = pd.read_csv(path)

    assert "model_scope" in frame.columns
    assert "active_for_final_allocation" in frame.columns
    assert "source_module" in frame.columns
    assert set(frame["model_scope"]) == {"scenario_risk_distribution"}
    assert set(frame["source_module"]) == {"scenario_risk_model"}
    assert frame["active_for_final_allocation"].astype(str).str.lower().eq("false").all()


def test_outputs_do_not_contain_raw_secret_patterns():
    dangerous_patterns = (
        "github_pat_",
        "SMTP_PASSWORD=",
        "INVESTOPEDIA_PASSWORD=",
        "BREVO_API_KEY=",
        "password=[",
    )

    offenders: list[str] = []
    for path in OUTPUT_DIR.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in dangerous_patterns:
            if pattern in text:
                offenders.append(f"{path}:{pattern}")

    assert offenders == []
