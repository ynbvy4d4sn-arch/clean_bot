"""Robustness validation for tactical Gurobi candidates.

Research-only. Does not place orders.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from backtest_tactical_gurobi_replay import ReplayConfig, run_tactical_gurobi_replay


OUTPUT_DIR = Path("outputs")


def run_candidate(
    *,
    period: str,
    name: str,
    score_name: str,
    start_date: str,
    end_date: str | None,
    top_n_gate: int,
    cap: float,
    lambda_variance: float,
    lambda_turnover: float,
    lambda_concentration: float,
    signal_scale: float,
) -> dict:
    prefix = (
        f"robust_{period}_{name}_{score_name}"
        f"_start{start_date.replace('-', '')}"
        f"_end{(end_date or 'latest').replace('-', '')}"
    )

    cfg = ReplayConfig(
        start_date=start_date,
        end_date=end_date,
        score_name=score_name,
        rebalance_every=1,
        top_n_gate=top_n_gate,
        max_weight=0.25,
        max_rebalance_turnover=cap,
        lambda_variance=lambda_variance,
        lambda_turnover=lambda_turnover,
        lambda_concentration=lambda_concentration,
        signal_scale=signal_scale,
        output_prefix=prefix,
    )

    paths = run_tactical_gurobi_replay(cfg)
    row = pd.read_csv(paths["summary"]).iloc[0].to_dict()
    row["period"] = period
    row["candidate"] = name
    row["output_prefix"] = prefix
    row["start_date_setting"] = start_date
    row["end_date_setting"] = end_date or "latest"
    return row


def main() -> None:
    periods = [
        ("full", "2024-01-01", None),
        ("early_2024", "2024-01-01", "2024-06-30"),
        ("late_2024", "2024-07-01", "2024-12-31"),
        ("early_2025", "2025-01-01", "2025-06-30"),
        ("late_2025", "2025-07-01", "2025-12-31"),
        ("ytd_2026", "2026-01-01", None),
        ("post_2025_h2", "2025-07-01", None),
    ]

    candidates = [
        {
            "name": "v3_best",
            "score_name": "v3",
            "top_n_gate": 8,
            "cap": 0.30,
            "lambda_variance": 10.0,
            "lambda_turnover": 0.03,
            "lambda_concentration": 0.02,
            "signal_scale": 0.015,
        },
        {
            "name": "v3_lower_turnover",
            "score_name": "v3",
            "top_n_gate": 6,
            "cap": 0.20,
            "lambda_variance": 10.0,
            "lambda_turnover": 0.02,
            "lambda_concentration": 0.02,
            "signal_scale": 0.012,
        },
        {
            "name": "v4b_overlay",
            "score_name": "v4b",
            "top_n_gate": 8,
            "cap": 0.30,
            "lambda_variance": 10.0,
            "lambda_turnover": 0.03,
            "lambda_concentration": 0.02,
            "signal_scale": 0.015,
        },
    ]

    rows = []
    for period, start, end in periods:
        for candidate in candidates:
            print(f"RUN {period} {candidate['name']}")
            rows.append(
                run_candidate(
                    period=period,
                    name=candidate["name"],
                    score_name=candidate["score_name"],
                    start_date=start,
                    end_date=end,
                    top_n_gate=candidate["top_n_gate"],
                    cap=candidate["cap"],
                    lambda_variance=candidate["lambda_variance"],
                    lambda_turnover=candidate["lambda_turnover"],
                    lambda_concentration=candidate["lambda_concentration"],
                    signal_scale=candidate["signal_scale"],
                )
            )

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_path = OUTPUT_DIR / "gurobi_candidate_robustness_summary.csv"
    report_path = OUTPUT_DIR / "gurobi_candidate_robustness_report.txt"
    out.to_csv(summary_path, index=False)

    lines = [
        "Tactical Gurobi Candidate Robustness Report",
        "",
        "status: research_only_no_order_change",
        "",
        "best_by_period:",
    ]

    for period, _, _ in periods:
        sub = out[out["period"].eq(period)].sort_values("sharpe", ascending=False)
        if sub.empty:
            continue
        best = sub.iloc[0]
        lines.append(
            f"- {period}: {best['candidate']} score={best['score_name']}, "
            f"return={best['total_return']:.4f}, sharpe={best['sharpe']:.3f}, "
            f"dd={best['max_drawdown']:.4f}, turnover={best['total_turnover']:.2f}"
        )

    lines.extend(["", "candidate_stability:"])
    for candidate, sub in out.groupby("candidate"):
        lines.append(
            f"- {candidate}: mean_sharpe={sub['sharpe'].mean():.3f}, "
            f"median_sharpe={sub['sharpe'].median():.3f}, "
            f"min_sharpe={sub['sharpe'].min():.3f}, "
            f"max_sharpe={sub['sharpe'].max():.3f}, "
            f"positive_sharpe_share={(sub['sharpe'] > 0).mean():.3f}, "
            f"mean_dd={sub['max_drawdown'].mean():.4f}, "
            f"mean_turnover={sub['total_turnover'].mean():.2f}"
        )

    lines.extend(["", "all_rows:"])
    for row in out.sort_values(["period", "sharpe"], ascending=[True, False]).itertuples(index=False):
        lines.append(
            f"- {row.period} {row.candidate}: return={row.total_return:.4f}, "
            f"vol={row.annualized_vol:.4f}, sharpe={row.sharpe:.3f}, "
            f"dd={row.max_drawdown:.4f}, turnover={row.total_turnover:.2f}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Robustness outputs:")
    print(f"- summary: {summary_path}")
    print(f"- report: {report_path}")


if __name__ == "__main__":
    main()
