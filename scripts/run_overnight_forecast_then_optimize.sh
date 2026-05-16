#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_DIR="$HOME/uni_trading_bot_project/clean_bot"
cd "$PROJECT_DIR" || exit 1

mkdir -p logs outputs

START_TS="$(date +%s)"
MAX_SECONDS="${MAX_SECONDS:-28800}"
LOG_FILE="logs/overnight_forecast_then_optimize_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S %z') | $*" | tee -a "$LOG_FILE"
}

time_left() {
  now="$(date +%s)"
  elapsed=$((now - START_TS))
  echo $((MAX_SECONDS - elapsed))
}

stop_if_time_low() {
  remaining="$(time_left)"
  if [ "$remaining" -le 600 ]; then
    log "STOP: less than 10 minutes left. remaining=${remaining}s"
    exit 0
  fi
}

run_step() {
  name="$1"
  shift
  stop_if_time_low
  log "START STEP: $name"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  status="${PIPESTATUS[0]}"
  if [ "$status" -ne 0 ]; then
    log "FAILED STEP: $name status=$status"
    return "$status"
  fi
  log "DONE STEP: $name"
  return 0
}

log "START overnight forecast then optimize"
log "project=$PROJECT_DIR"
log "max_seconds=$MAX_SECONDS"
log "branch=$(git branch --show-current)"
log "head=$(git log --oneline -1)"
log "python=$(which python)"

log "=== initial git status ==="
git status --short | tee -a "$LOG_FILE"

run_step "compile research scripts" python -m compileall -q \
  research_predictive_features.py \
  optimize_tactical_weights.py \
  backtest_tactical_gurobi_replay.py \
  run_gurobi_replay_grid.py \
  tactical_forecast.py \
  backtest_tactical_scores.py

# 1) Forecast Research: Ridge Alpha Sweep
log "START predictive feature alpha sweep"
for alpha in 0.3 1 3 10 30 100 300; do
  stop_if_time_low
  log "RUN predictive feature research alpha=$alpha"
  python research_predictive_features.py --ridge-alpha "$alpha" 2>&1 | tee -a "$LOG_FILE"

  safe_alpha="$(echo "$alpha" | sed 's/\./p/g')"
  cp outputs/predictive_feature_report.txt "outputs/predictive_feature_report_alpha${safe_alpha}.txt"
  cp outputs/predictive_feature_univariate.csv "outputs/predictive_feature_univariate_alpha${safe_alpha}.csv"
  cp outputs/predictive_feature_ridge_models.csv "outputs/predictive_feature_ridge_models_alpha${safe_alpha}.csv"
  cp outputs/predictive_feature_ridge_coefficients.csv "outputs/predictive_feature_ridge_coefficients_alpha${safe_alpha}.csv"
done
log "DONE predictive feature alpha sweep"

# 2) Aggregate Forecast Research
log "Aggregate predictive feature research"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
from pathlib import Path
import pandas as pd

rows = []
for path in sorted(Path("outputs").glob("predictive_feature_ridge_models_alpha*.csv")):
    alpha = path.stem.replace("predictive_feature_ridge_models_alpha", "")
    df = pd.read_csv(path)
    global_df = df[df["model"].eq("global_ridge")].copy()
    for row in global_df.to_dict(orient="records"):
        row["alpha"] = alpha
        rows.append(row)

alpha_out = pd.DataFrame(rows)
alpha_out.to_csv("outputs/predictive_feature_alpha_sweep_summary.csv", index=False)

print("=== alpha sweep global ridge ===")
if not alpha_out.empty:
    cols = [
        "alpha", "model", "ticker", "horizon",
        "test_rank_ic_mean", "test_top_minus_bottom",
        "test_top_hit_rate", "test_sample_days"
    ]
    print(alpha_out.sort_values(["horizon", "test_rank_ic_mean"], ascending=[True, False])[cols].to_string(index=False))
else:
    print("no alpha rows")

# Aggregate univariate features across alpha files; univariate does not depend on alpha,
# but this also works if the last run overwrote base outputs.
u_path = Path("outputs/predictive_feature_univariate.csv")
if u_path.exists():
    u = pd.read_csv(u_path)
    test = u[u["split"].eq("test")].copy()

    # Robust feature score: prefer features that have positive IC and positive top-minus-bottom.
    test["feature_quality_score"] = (
        test["rank_ic_mean"]
        + 20.0 * test["top_minus_bottom"]
        + 0.10 * (test["top_hit_rate"] - 0.5)
    )

    test.to_csv("outputs/predictive_feature_test_quality_scores.csv", index=False)

    print("")
    print("=== best univariate predictive features by horizon ===")
    cols = ["horizon", "feature", "rank_ic_mean", "top_minus_bottom", "top_hit_rate", "feature_quality_score"]
    for h in sorted(test["horizon"].unique()):
        print(f"\nHORIZON {h}d")
        print(test[test["horizon"].eq(h)].sort_values("feature_quality_score", ascending=False).head(20)[cols].to_string(index=False))

    # Make a candidate feature list for later model v4 research.
    candidate = (
        test[test["rank_ic_mean"].gt(0)]
        .sort_values("feature_quality_score", ascending=False)
        .groupby("horizon")
        .head(15)
    )
    candidate.to_csv("outputs/predictive_feature_v4_candidate_features.csv", index=False)

    print("")
    print("=== candidate v4 features ===")
    print(candidate[cols].to_string(index=False))
PY

# 3) Multi-seed weight search after forecast research
# This does not yet inject new forecast features into v3. It checks whether v3-style weights are stable.
log "START multi-seed tactical weight search"
for seed in 11 22 33 44 55 66 77; do
  stop_if_time_low
  log "RUN tactical weight search seed=$seed"
  python optimize_tactical_weights.py \
    --trials 500 \
    --seed "$seed" \
    --rebalance-every 1 \
    --top-n 8 \
    --max-weight 0.25 \
    2>&1 | tee -a "$LOG_FILE"

  cp outputs/tactical_weight_search_results.csv "outputs/tactical_weight_search_results_seed${seed}.csv"
  cp outputs/tactical_weight_search_best_weights.json "outputs/tactical_weight_search_best_weights_seed${seed}.json"
  cp outputs/tactical_weight_search_report.txt "outputs/tactical_weight_search_report_seed${seed}.txt"
done
log "DONE multi-seed tactical weight search"

# 4) Aggregate multi-seed tactical weight search
log "Aggregate tactical multi-seed search"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json
from pathlib import Path
import pandas as pd

rows = []
for path in sorted(Path("outputs").glob("tactical_weight_search_results_seed*.csv")):
    seed = path.stem.replace("tactical_weight_search_results_seed", "")
    df = pd.read_csv(path)
    if df.empty:
        continue
    best = df.sort_values("robust_score", ascending=False).iloc[0].to_dict()
    best["seed"] = seed
    rows.append(best)

out = pd.DataFrame(rows)
out.to_csv("outputs/tactical_weight_search_multiseed_summary.csv", index=False)

print("=== multiseed best candidates ===")
if not out.empty:
    cols = [
        "seed", "candidate_id", "is_degenerate",
        "robust_score", "train_sharpe", "test_sharpe",
        "train_test_gap", "train_total_return",
        "test_total_return", "test_max_drawdown"
    ]
    print(out[cols].sort_values("robust_score", ascending=False).to_string(index=False))

weight_rows = []
for path in sorted(Path("outputs").glob("tactical_weight_search_best_weights_seed*.json")):
    seed = path.stem.replace("tactical_weight_search_best_weights_seed", "")
    weights = json.loads(path.read_text())
    weights["seed"] = seed
    weight_rows.append(weights)

if weight_rows:
    wdf = pd.DataFrame(weight_rows).set_index("seed").fillna(0.0)
    stability = pd.DataFrame({
        "mean_weight": wdf.mean(),
        "median_weight": wdf.median(),
        "std_weight": wdf.std(ddof=0),
        "positive_share": (wdf > 0).mean(),
        "negative_share": (wdf < 0).mean(),
    })
    stability = stability.reindex(stability["mean_weight"].abs().sort_values(ascending=False).index)
    stability.to_csv("outputs/tactical_weight_search_weight_stability.csv")

    print("")
    print("=== stable tactical weights ===")
    print(stability.head(30).to_string())
PY

# 5) Refined Gurobi search around current best: v3 top8 cap30
log "START refined Gurobi search around v3 top8 cap30"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
from pathlib import Path
import pandas as pd

from backtest_tactical_gurobi_replay import ReplayConfig, run_tactical_gurobi_replay

configs = []
for lambda_variance in (8.0, 9.0, 10.0, 11.0, 12.0):
    for lambda_turnover in (0.02, 0.025, 0.03, 0.04):
        for signal_scale in (0.010, 0.012, 0.015, 0.018):
            for top_n_gate in (6, 8, 10):
                for cap in (0.20, 0.30):
                    configs.append({
                        "score_name": "v3",
                        "rebalance_every": 1,
                        "top_n_gate": top_n_gate,
                        "max_weight": 0.25,
                        "max_rebalance_turnover": cap,
                        "lambda_variance": lambda_variance,
                        "lambda_turnover": lambda_turnover,
                        "lambda_concentration": 0.02,
                        "signal_scale": signal_scale,
                    })

rows = []
for i, kwargs in enumerate(configs, start=1):
    prefix = (
        f"gurobi_refined2_{i:03d}"
        f"_v3_top{kwargs['top_n_gate']}"
        f"_lv{str(kwargs['lambda_variance']).replace('.', 'p')}"
        f"_lt{str(kwargs['lambda_turnover']).replace('.', 'p')}"
        f"_ss{str(kwargs['signal_scale']).replace('.', 'p')}"
        f"_cap{int(kwargs['max_rebalance_turnover'] * 100)}"
    )
    print(f"RUN refined2 {i}/{len(configs)} {prefix}")
    cfg = ReplayConfig(output_prefix=prefix, **kwargs)
    paths = run_tactical_gurobi_replay(cfg)
    row = pd.read_csv(paths["summary"]).iloc[0].to_dict()
    row["output_prefix"] = prefix
    rows.append(row)

out = pd.DataFrame(rows).sort_values(["sharpe", "total_return"], ascending=False)
Path("outputs").mkdir(exist_ok=True)
out.to_csv("outputs/gurobi_replay_grid_refined2_summary.csv", index=False)

lines = [
    "Refined2 Tactical Gurobi Replay Grid Report",
    "",
    "status: research_only_no_order_change",
    "region: v3 top6/top8/top10 cap20/cap30 around current best",
    "",
    "top_configs_by_sharpe:",
]
for row in out.head(40).itertuples(index=False):
    lines.append(
        f"- {row.output_prefix}: score={row.score_name}, top_n={row.top_n_gate}, "
        f"cap={row.max_rebalance_turnover}, lv={row.lambda_variance:.3f}, "
        f"lt={row.lambda_turnover:.3f}, ss={row.signal_scale:.3f}, "
        f"return={row.total_return:.4f}, vol={row.annualized_vol:.4f}, "
        f"sharpe={row.sharpe:.3f}, max_dd={row.max_drawdown:.4f}, "
        f"turnover={row.total_turnover:.2f}"
    )

best = out.iloc[0]
lines.extend([
    "",
    "best_config:",
    f"- output_prefix: {best['output_prefix']}",
    f"- score_name: {best['score_name']}",
    f"- top_n_gate: {int(best['top_n_gate'])}",
    f"- max_rebalance_turnover: {best['max_rebalance_turnover']}",
    f"- lambda_variance: {float(best['lambda_variance']):.4f}",
    f"- lambda_turnover: {float(best['lambda_turnover']):.4f}",
    f"- lambda_concentration: {float(best['lambda_concentration']):.4f}",
    f"- signal_scale: {float(best['signal_scale']):.4f}",
    f"- total_return: {float(best['total_return']):.4f}",
    f"- sharpe: {float(best['sharpe']):.4f}",
    f"- max_drawdown: {float(best['max_drawdown']):.4f}",
    f"- total_turnover: {float(best['total_turnover']):.4f}",
])
Path("outputs/gurobi_replay_grid_refined2_report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
log "DONE refined Gurobi search"

# 6) Final summary
log "Build final overnight summary"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
from pathlib import Path
import pandas as pd

report = []
report.append("Overnight Forecast and Optimization Summary")
report.append("")
report.append("status: research_only_no_simulator_orders")
report.append("")

for label, path in [
    ("predictive_feature_alpha_sweep", Path("outputs/predictive_feature_alpha_sweep_summary.csv")),
    ("medium_gurobi_grid", Path("outputs/gurobi_replay_grid_medium_summary.csv")),
    ("refined2_gurobi_grid", Path("outputs/gurobi_replay_grid_refined2_summary.csv")),
    ("tactical_weight_multiseed", Path("outputs/tactical_weight_search_multiseed_summary.csv")),
]:
    if path.exists():
        report.append(f"source_available: {label} -> {path}")
report.append("")

pf = Path("outputs/predictive_feature_v4_candidate_features.csv")
if pf.exists():
    df = pd.read_csv(pf)
    report.append("top_predictive_v4_candidate_features:")
    cols = ["horizon", "feature", "rank_ic_mean", "top_minus_bottom", "top_hit_rate", "feature_quality_score"]
    for row in df.sort_values(["horizon", "feature_quality_score"], ascending=[True, False]).head(60)[cols].itertuples(index=False):
        report.append(
            f"- {row.horizon}d {row.feature}: ic={row.rank_ic_mean:.4f}, "
            f"tmb={row.top_minus_bottom:.5f}, hit={row.top_hit_rate:.3f}, "
            f"quality={row.feature_quality_score:.4f}"
        )
    report.append("")

for label, path in [
    ("medium_gurobi_grid", Path("outputs/gurobi_replay_grid_medium_summary.csv")),
    ("refined2_gurobi_grid", Path("outputs/gurobi_replay_grid_refined2_summary.csv")),
]:
    if path.exists():
        df = pd.read_csv(path).sort_values("sharpe", ascending=False)
        report.append(f"top_{label}:")
        for row in df.head(15).itertuples(index=False):
            report.append(
                f"- {row.output_prefix}: score={row.score_name}, top={row.top_n_gate}, "
                f"cap={row.max_rebalance_turnover}, lv={row.lambda_variance}, "
                f"lt={row.lambda_turnover}, ss={row.signal_scale}, "
                f"return={row.total_return:.4f}, sharpe={row.sharpe:.3f}, "
                f"dd={row.max_drawdown:.4f}, turnover={row.total_turnover:.2f}"
            )
        report.append("")

ms = Path("outputs/tactical_weight_search_multiseed_summary.csv")
if ms.exists():
    df = pd.read_csv(ms).sort_values("robust_score", ascending=False)
    report.append("top_multiseed_weight_search:")
    for row in df.head(15).itertuples(index=False):
        report.append(
            f"- seed={row.seed}, candidate={row.candidate_id}, "
            f"robust={row.robust_score:.3f}, train={row.train_sharpe:.3f}, "
            f"test={row.test_sharpe:.3f}, gap={row.train_test_gap:.3f}, "
            f"test_dd={row.test_max_drawdown:.4f}"
        )
    report.append("")

ws = Path("outputs/tactical_weight_search_weight_stability.csv")
if ws.exists():
    df = pd.read_csv(ws)
    report.append("stable_tactical_weight_signals:")
    for row in df.head(25).itertuples(index=False):
        feature = row[0]
        report.append(
            f"- {feature}: mean={row.mean_weight:+.4f}, median={row.median_weight:+.4f}, "
            f"std={row.std_weight:.4f}, pos={row.positive_share:.2f}, neg={row.negative_share:.2f}"
        )

out = Path("outputs/overnight_forecast_then_optimize_summary.txt")
out.write_text("\n".join(report) + "\n", encoding="utf-8")
print(out.read_text())
PY

log "=== final git status ==="
git status --short | tee -a "$LOG_FILE"

log "DONE overnight forecast then optimize"
log "log_file=$LOG_FILE"
