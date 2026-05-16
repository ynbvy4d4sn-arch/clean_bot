"""Paper simulator readiness validation.

Research/safety-only. Does not place orders.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import json
import importlib.util


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout.strip()


def check_file(path: str) -> dict:
    p = ROOT / path
    return {
        "check": f"file_exists:{path}",
        "ok": p.exists(),
        "detail": str(p),
    }


def check_git_clean() -> dict:
    code, out = run(["git", "status", "--short"])
    return {
        "check": "git_status_clean",
        "ok": code == 0 and out == "",
        "detail": out or "clean",
    }


def check_branch() -> dict:
    code, out = run(["git", "branch", "--show-current"])
    return {
        "check": "branch_detected",
        "ok": code == 0 and bool(out),
        "detail": out,
    }


def check_compile() -> dict:
    files = [
        "daily_bot.py",
        "tactical_forecast.py",
        "backtest_tactical_gurobi_replay.py",
        "validate_gurobi_candidate_robustness.py",
    ]
    code, out = run([sys.executable, "-m", "compileall", "-q", *files])
    return {
        "check": "compile_core_files",
        "ok": code == 0,
        "detail": out or "compile ok",
    }


def check_gurobi_available() -> dict:
    spec = importlib.util.find_spec("gurobipy")
    return {
        "check": "gurobipy_available",
        "ok": spec is not None,
        "detail": "available" if spec is not None else "missing",
    }


def check_best_candidate_outputs() -> dict:
    required = [
        "outputs/gurobi_candidate_robustness_summary.csv",
        "outputs/gurobi_candidate_robustness_report.txt",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    return {
        "check": "robustness_outputs_present",
        "ok": not missing,
        "detail": "missing=" + ",".join(missing) if missing else "present",
    }


def main() -> None:
    checks = [
        check_branch(),
        check_git_clean(),
        check_compile(),
        check_gurobi_available(),
        check_file("daily_bot.py"),
        check_file("tactical_forecast.py"),
        check_file("backtest_tactical_gurobi_replay.py"),
        check_file("validate_gurobi_candidate_robustness.py"),
        check_best_candidate_outputs(),
    ]

    all_ok = all(c["ok"] for c in checks)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_json = OUTPUT_DIR / "paper_readiness_check.json"
    out_txt = OUTPUT_DIR / "paper_readiness_check_report.txt"

    payload = {
        "status": "ready_for_next_step" if all_ok else "not_ready",
        "paper_candidate": {
            "name": "v3_lower_turnover",
            "score_name": "v3",
            "top_n_gate": 6,
            "max_weight": 0.25,
            "max_rebalance_turnover": 0.20,
            "lambda_variance": 10.0,
            "lambda_turnover": 0.02,
            "lambda_concentration": 0.02,
            "signal_scale": 0.012,
            "risk_free_rate_annual": 0.02,
        },
        "checks": checks,
    }

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "Paper Simulator Readiness Check",
        "",
        f"status: {payload['status']}",
        "",
        "paper_candidate:",
    ]
    for k, v in payload["paper_candidate"].items():
        lines.append(f"- {k}: {v}")

    lines.extend(["", "checks:"])
    for c in checks:
        marker = "OK" if c["ok"] else "FAIL"
        lines.append(f"- {marker}: {c['check']} :: {c['detail']}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out_txt.read_text())


if __name__ == "__main__":
    main()
