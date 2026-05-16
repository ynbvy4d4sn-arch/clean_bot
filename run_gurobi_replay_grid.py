"""Focused tactical Gurobi replay grid.

Research-only. Does not place orders and does not change production logic.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

from backtest_tactical_gurobi_replay import ReplayConfig, run_tactical_gurobi_replay


OUTPUT_DIR = Path("outputs")


def run_grid(mode: str = "quick") -> dict[str, Path]:
    if mode == "quick":
        configs = []
        for lambda_variance in (6.0, 8.0, 10.0):
            for lambda_turnover in (0.02, 0.03):
                for signal_scale in (0.012, 0.015, 0.018):
                    configs.append(
                        {
                            "score_name": "v3",
                            "rebalance_every": 1,
                            "top_n_gate": 6,
                            "max_weight": 0.25,
                            "max_rebalance_turnover": 0.30,
                            "lambda_variance": lambda_variance,
                            "lambda_turnover": lambda_turnover,
                            "lambda_concentration": 0.02,
                            "signal_scale": signal_scale,
                        }
                    )
    elif mode == "medium":
        configs = []
        for score_name in ("v2", "v3"):
            for top_n_gate in (6, 8):
                for max_turnover in (0.20, 0.30):
                    for lambda_variance in (6.0, 8.0, 10.0):
                        for lambda_turnover in (0.02, 0.03):
                            for signal_scale in (0.012, 0.015, 0.018):
                                configs.append(
                                    {
                                        "score_name": score_name,
                                        "rebalance_every": 1,
                                        "top_n_gate": top_n_gate,
                                        "max_weight": 0.25,
                                        "max_rebalance_turnover": max_turnover,
                                        "lambda_variance": lambda_variance,
                                        "lambda_turnover": lambda_turnover,
                                        "lambda_concentration": 0.02,
                                        "signal_scale": signal_scale,
                                    }
                                )
    else:
        raise ValueError("mode must be quick or medium")

    rows = []
    for i, cfg_kwargs in enumerate(configs, start=1):
        prefix = (
            f"gurobi_grid_{mode}_{i:03d}"
            f"_{cfg_kwargs['score_name']}"
            f"_top{cfg_kwargs['top_n_gate']}"
            f"_lv{str(cfg_kwargs['lambda_variance']).replace('.', 'p')}"
            f"_lt{str(cfg_kwargs['lambda_turnover']).replace('.', 'p')}"
            f"_ss{str(cfg_kwargs['signal_scale']).replace('.', 'p')}"
            f"_cap{int(cfg_kwargs['max_rebalance_turnover'] * 100)}"
        )
        print(f"RUN {i}/{len(configs)} {prefix}")

        cfg = ReplayConfig(
            output_prefix=prefix,
            **cfg_kwargs,
        )
        paths = run_tactical_gurobi_replay(cfg)
        summary = pd.read_csv(paths["summary"]).iloc[0].to_dict()
        summary["grid_id"] = i
        summary["output_prefix"] = prefix
        rows.append(summary)

    out = pd.DataFrame(rows).sort_values(
        ["sharpe", "total_return"],
        ascending=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / f"gurobi_replay_grid_{mode}_summary.csv"
    report_path = OUTPUT_DIR / f"gurobi_replay_grid_{mode}_report.txt"

    out.to_csv(summary_path, index=False)

    lines = [
        "Focused Tactical Gurobi Replay Grid Report",
        "",
        "status: research_only_no_order_change",
        f"mode: {mode}",
        f"runs: {len(out)}",
        "risk_free_rate_annual: 0.0200",
        "",
        "objective:",
        "- Compare optimizer-near tactical allocation configurations.",
        "- Sharpe uses excess return over 2% annual risk-free rate.",
        "- No simulator orders are placed.",
        "",
        "top_configs_by_sharpe:",
    ]

    for row in out.head(20).itertuples(index=False):
        lines.append(
            f"- {row.output_prefix}: score={row.score_name}, top_n={row.top_n_gate}, "
            f"lambda_var={row.lambda_variance:.3f}, lambda_turnover={row.lambda_turnover:.3f}, "
            f"signal_scale={row.signal_scale:.3f}, cap={row.max_rebalance_turnover}, "
            f"return={row.total_return:.4f}, vol={row.annualized_vol:.4f}, "
            f"sharpe={row.sharpe:.3f}, max_dd={row.max_drawdown:.4f}, "
            f"turnover={row.total_turnover:.2f}"
        )

    lines.extend(["", "best_config:"])
    best = out.iloc[0]
    lines.extend(
        [
            f"- output_prefix: {best['output_prefix']}",
            f"- score_name: {best['score_name']}",
            f"- top_n_gate: {int(best['top_n_gate'])}",
            f"- max_weight: {float(best['max_weight']):.4f}",
            f"- max_rebalance_turnover: {best['max_rebalance_turnover']}",
            f"- lambda_variance: {float(best['lambda_variance']):.4f}",
            f"- lambda_turnover: {float(best['lambda_turnover']):.4f}",
            f"- lambda_concentration: {float(best['lambda_concentration']):.4f}",
            f"- signal_scale: {float(best['signal_scale']):.4f}",
            f"- total_return: {float(best['total_return']):.4f}",
            f"- sharpe: {float(best['sharpe']):.4f}",
            f"- max_drawdown: {float(best['max_drawdown']):.4f}",
            f"- total_turnover: {float(best['total_turnover']):.4f}",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "summary": summary_path,
        "report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "medium"], default="quick")
    args = parser.parse_args()

    paths = run_grid(args.mode)
    print("Grid outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
