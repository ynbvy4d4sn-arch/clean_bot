# Migration Notes

## Active Daily Run Path

The active Daily Bot path now finalizes immediately after the direct
scenario-weighted RF-adjusted Sharpe solver.

The active solver orchestration has been isolated in
`scenario_daily_pipeline.py`. `daily_bot.py` still owns run setup, data loading,
report writing and execution guards, but the final target solve itself is no
longer embedded in the large Daily Bot function.

Active decision flow:

1. Load data and current portfolio state.
2. Compute returns, forecasts, scenarios and covariance-aware scenario inputs.
3. Solve the direct scenario-weighted RF-adjusted Sharpe optimization.
4. Apply optional execution damping.
5. Convert executable weights to share-level order preview rows.
6. Write the final allocation, final order preview and decision report.
7. Stop before legacy candidate ranking.

Final target source:

- `SCENARIO_WEIGHTED_RF_SHARPE_OPTIMAL`
- `HOLD_SOLVER_FAILED` if the solver or post-solver validation fails.

## Disabled In Active Daily Run

The following legacy decision mechanisms are no longer used to determine the
Daily final target:

- `build_candidate_portfolios`
- `select_robust_candidate`
- `generate_discrete_candidates`
- `score_discrete_candidates`
- `select_best_discrete_portfolio`
- candidate families such as `MOMENTUM_TILT_SIMPLE`, `DEFENSIVE_CASH`,
  `PARTIAL_25`, `PARTIAL_50` and `CONDITIONAL_FACTOR_TARGET`

These modules and names may still exist for audits, fixtures, historical tests
and legacy comparison work. They are not imported as active Daily final-decision
dependencies and the Daily run returns before the old candidate-selection block.

The old constraint-repair utility used to live in `candidate_factory.py`; the
active Daily path now imports the same narrow responsibility from
`constraint_repair.py` so a feasible SLSQP starting point does not pull the
legacy candidate factory into the active decision path.

The active Daily path imports `run_scenario_weighted_daily_solve` rather than
calling candidate construction, robust candidate selection or discrete
candidate ranking for the final target.

## Output Files

Use these files as the active Daily decision artifacts:

- `outputs/scenario_weighted_optimal_allocation.csv`
- `outputs/scenario_weighted_order_preview.csv`
- `outputs/scenario_solver_decision.md`
- `outputs/latest_decision_report.txt`
- `outputs/daily_bot_decision_report.txt`

Legacy candidate output filenames are not deleted, because older operator
workflows may still reference them. During active Daily runs, stale candidate
artifacts are overwritten with explicit `disabled_in_active_daily_run` notices
instead of executable-looking candidate rankings.

## Kopie Files

No `* Kopie.py` files were found in the active import path during this cleanup.
If such files appear later, they should be moved to `legacy/copies/` unless an
active import depends on them; in that case the import should first be corrected
to the canonical non-copy module.

## Safety Notes

This cleanup does not enable live broker execution, Investopedia automation or
real email sending. The slim path writes previews and reports only. Execution
remains guarded by data freshness, synthetic-data checks, the project trading
window and DRY_RUN defaults.
