"""Optional conditional factor scenario model with direct-only fallback."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scenario_model import ScenarioSet


@dataclass(slots=True)
class ConditionalScenarioResult:
    """Conditional scenario output plus mode information."""

    scenario_returns_matrix: pd.DataFrame
    scenario_probabilities: pd.Series
    summary: pd.DataFrame
    diagnostics: list[str]
    mode: str


def build_conditional_scenarios(
    direct_scenarios: ScenarioSet,
    factor_forecast_df: pd.DataFrame,
    exposure_matrix: pd.DataFrame,
    residual_volatility: pd.Series,
    *,
    weight_factor_model: float = 0.50,
    weight_direct_model: float = 0.50,
) -> ConditionalScenarioResult:
    """Build conditional factor scenarios or fall back to direct-only mode."""

    diagnostics: list[str] = []
    if factor_forecast_df.empty or exposure_matrix.empty:
        diagnostics.append("Conditional factor mode unavailable; using direct-only fallback.")
        return ConditionalScenarioResult(
            scenario_returns_matrix=direct_scenarios.scenario_returns_matrix.copy(),
            scenario_probabilities=direct_scenarios.scenario_probabilities.copy(),
            summary=direct_scenarios.summary.copy(),
            diagnostics=diagnostics,
            mode="direct_only",
        )

    ff = factor_forecast_df.set_index("factor")
    asset_index = direct_scenarios.scenario_returns_matrix.columns
    factor_index = [factor for factor in exposure_matrix.columns if factor in ff.index]
    if not factor_index:
        diagnostics.append("No overlapping factors for conditional scenario construction; using direct-only fallback.")
        return ConditionalScenarioResult(
            scenario_returns_matrix=direct_scenarios.scenario_returns_matrix.copy(),
            scenario_probabilities=direct_scenarios.scenario_probabilities.copy(),
            summary=direct_scenarios.summary.copy(),
            diagnostics=diagnostics,
            mode="direct_only",
        )

    exposure = exposure_matrix.reindex(index=asset_index, columns=factor_index).fillna(0.0)
    factor_expected = ff.reindex(factor_index)["expected_change_3m"].fillna(0.0)

    scenario_factor_shocks = {
        "base": factor_expected,
        "growth_risk_on": factor_expected.copy(),
        "inflation_shock": factor_expected.copy(),
        "rates_down_easing": factor_expected.copy(),
        "recession_risk_off": factor_expected.copy(),
        "liquidity_stress": factor_expected.copy(),
        "commodity_shock": factor_expected.copy(),
        "crypto_risk_on": factor_expected.copy(),
    }
    if "growth" in factor_expected.index:
        scenario_factor_shocks["growth_risk_on"].loc["growth"] += 0.20
    if "inflation" in factor_expected.index:
        scenario_factor_shocks["inflation_shock"].loc["inflation"] += 0.25
    if "nominal_rates" in factor_expected.index:
        scenario_factor_shocks["rates_down_easing"].loc["nominal_rates"] -= 0.20
    if "risk_appetite" in factor_expected.index:
        scenario_factor_shocks["recession_risk_off"].loc["risk_appetite"] -= 0.25
        scenario_factor_shocks["liquidity_stress"].loc["risk_appetite"] -= 0.20
        scenario_factor_shocks["crypto_risk_on"].loc["risk_appetite"] += 0.20
    if "commodity" in factor_expected.index:
        scenario_factor_shocks["commodity_shock"].loc["commodity"] += 0.25
    if "crypto_beta" in factor_expected.index:
        scenario_factor_shocks["crypto_risk_on"].loc["crypto_beta"] += 0.30

    factor_based = pd.DataFrame(index=scenario_factor_shocks.keys(), columns=asset_index, dtype=float)
    for name, shock in scenario_factor_shocks.items():
        base_asset = exposure.to_numpy(dtype=float) @ shock.reindex(factor_index).fillna(0.0).to_numpy(dtype=float)
        factor_based.loc[name] = base_asset + 0.10 * residual_volatility.reindex(asset_index).fillna(0.0).to_numpy(dtype=float)

    direct_reindexed = direct_scenarios.scenario_returns_matrix.reindex(index=factor_based.index, columns=asset_index).fillna(0.0)
    combined = weight_factor_model * factor_based + weight_direct_model * direct_reindexed
    probs = pd.Series(
        {
            "base": 0.35,
            "growth_risk_on": 0.12,
            "inflation_shock": 0.10,
            "rates_down_easing": 0.10,
            "recession_risk_off": 0.15,
            "liquidity_stress": 0.08,
            "commodity_shock": 0.05,
            "crypto_risk_on": 0.05,
        },
        dtype=float,
    )
    probs = probs / probs.sum()
    summary = pd.DataFrame(
        {
            "scenario_name": probs.index,
            "probability": probs.values,
            "mean_asset_return": [float(combined.loc[name].mean()) for name in probs.index],
            "median_asset_return": [float(combined.loc[name].median()) for name in probs.index],
        }
    )
    diagnostics.append("Conditional factor scenarios combined with direct scenarios at 50/50 default weight.")
    return ConditionalScenarioResult(
        scenario_returns_matrix=combined,
        scenario_probabilities=probs,
        summary=summary,
        diagnostics=diagnostics,
        mode="conditional_factor",
    )
