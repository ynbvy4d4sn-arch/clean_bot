"""Robust candidate evaluation against forward-looking scenario sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from candidate_factory import CandidatePortfolio
from risk_free import risk_free_return_from_params
from scenario_model import ScenarioSet


@dataclass(slots=True)
class CandidateScore:
    """Score object for one candidate portfolio."""

    name: str
    weights: pd.Series
    mean_return: float
    median_return: float
    probability_loss: float
    volatility: float
    downside_deviation: float
    var_5: float
    cvar_5: float
    worst_scenario: str
    robust_sharpe: float
    robust_score: float
    turnover: float
    estimated_cost: float
    dynamic_buffer: float
    net_robust_score: float
    estimated_commission: float = 0.0
    estimated_spread_cost: float = 0.0
    estimated_slippage_cost: float = 0.0
    estimated_market_impact_cost: float = 0.0
    estimated_total_order_cost: float = 0.0
    cost_bps_used: float = 0.0
    cost_model_used: str = "turnover_proxy"
    live_costs_available: bool = False
    delta_vs_hold: float = 0.0
    delta_vs_cash: float = 0.0
    probability_beats_hold: float = 0.0
    probability_beats_cash: float = 0.0
    risk_free_return: float = 0.0
    return_over_volatility_legacy: float = 0.0


@dataclass(slots=True)
class RobustSelectionResult:
    """Selection outcome across all candidates."""

    selected_candidate: CandidatePortfolio
    selected_score: CandidateScore
    scores: list[CandidateScore]
    scores_frame: pd.DataFrame
    mode: str


def _weighted_quantile(values: np.ndarray, probs: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values = values[order]
    probs = probs[order]
    cdf = np.cumsum(probs) / max(np.sum(probs), 1e-12)
    idx = int(np.searchsorted(cdf, quantile, side="left"))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def evaluate_candidate(
    candidate: CandidatePortfolio,
    scenario_set: ScenarioSet,
    w_current: pd.Series,
    params: dict[str, Any],
) -> CandidateScore:
    """Evaluate one candidate against the scenario distribution."""

    matrix = scenario_set.scenario_returns_matrix.reindex(columns=candidate.weights.index).fillna(0.0)
    weights = candidate.weights.reindex(matrix.columns).fillna(0.0).astype(float)
    scenario_portfolio_returns = matrix.to_numpy(dtype=float) @ weights.to_numpy(dtype=float)
    probs = scenario_set.scenario_probabilities.reindex(matrix.index).fillna(0.0).to_numpy(dtype=float)
    probs = probs / max(probs.sum(), 1e-12)

    mean_return = float(np.sum(scenario_portfolio_returns * probs))
    median_return = _weighted_quantile(scenario_portfolio_returns, probs, 0.50)
    probability_loss = float(np.sum(probs[scenario_portfolio_returns < 0.0]))
    volatility = float(np.sqrt(np.sum(probs * np.square(scenario_portfolio_returns - mean_return))))
    negative = np.minimum(scenario_portfolio_returns, 0.0)
    downside_deviation = float(np.sqrt(np.sum(probs * np.square(negative))))
    var_5 = _weighted_quantile(scenario_portfolio_returns, probs, 0.05)
    tail = scenario_portfolio_returns[scenario_portfolio_returns <= var_5]
    cvar_5 = float(np.mean(tail)) if len(tail) else var_5
    worst_idx = int(np.argmin(scenario_portfolio_returns))
    worst_scenario = str(matrix.index[worst_idx])
    risk_free_return = risk_free_return_from_params(params)
    return_over_volatility_legacy = mean_return * (1.0 / volatility) if volatility > 0.0 else 0.0
    robust_sharpe = (mean_return - risk_free_return) / volatility if volatility > 0.0 else 0.0
    turnover = float(np.abs(candidate.weights.reindex(w_current.index).fillna(0.0) - w_current.reindex(candidate.weights.index).fillna(0.0)).sum())
    estimated_cost = turnover * float(params.get("cost_rate", 0.001))
    dynamic_buffer = float(params.get("base_buffer", 0.0005)) + float(params.get("vol_buffer_multiplier", 0.05)) * volatility

    lambda_cvar = float(params.get("lambda_cvar", 0.80))
    lambda_prob_loss = float(params.get("lambda_prob_loss", 0.40))
    lambda_vol = float(params.get("lambda_vol", 0.10))
    robust_score = (
        0.40 * median_return
        + 0.30 * mean_return
        - lambda_cvar * abs(min(cvar_5, 0.0))
        - lambda_prob_loss * probability_loss
        - lambda_vol * volatility
    )
    net_robust_score = robust_score - estimated_cost - dynamic_buffer
    return CandidateScore(
        name=candidate.name,
        weights=candidate.weights,
        mean_return=mean_return,
        median_return=median_return,
        probability_loss=probability_loss,
        volatility=volatility,
        downside_deviation=downside_deviation,
        var_5=var_5,
        cvar_5=cvar_5,
        worst_scenario=worst_scenario,
        robust_sharpe=robust_sharpe,
        robust_score=robust_score,
        turnover=turnover,
        estimated_cost=estimated_cost,
        dynamic_buffer=dynamic_buffer,
        net_robust_score=net_robust_score,
        risk_free_return=risk_free_return,
        return_over_volatility_legacy=return_over_volatility_legacy,
    )


def select_robust_candidate(
    candidates: dict[str, CandidatePortfolio],
    scenario_set: ScenarioSet,
    w_current: pd.Series,
    params: dict[str, Any],
    mode: str = "direct_only",
) -> RobustSelectionResult:
    """Evaluate and select a candidate that robustly beats HOLD and DEFENSIVE_CASH."""

    if "HOLD" not in candidates or "DEFENSIVE_CASH" not in candidates:
        raise ValueError("Candidates must include HOLD and DEFENSIVE_CASH.")

    scores = [
        evaluate_candidate(candidate=candidate, scenario_set=scenario_set, w_current=w_current, params=params)
        for candidate in candidates.values()
    ]
    score_map = {score.name: score for score in scores}
    hold_score = score_map["HOLD"]
    cash_score = score_map["DEFENSIVE_CASH"]

    matrix = scenario_set.scenario_returns_matrix
    probs = scenario_set.scenario_probabilities.reindex(matrix.index).fillna(0.0).to_numpy(dtype=float)
    probs = probs / max(probs.sum(), 1e-12)
    hold_returns = matrix.to_numpy(dtype=float) @ hold_score.weights.reindex(matrix.columns).fillna(0.0).to_numpy(dtype=float)
    cash_returns = matrix.to_numpy(dtype=float) @ cash_score.weights.reindex(matrix.columns).fillna(0.0).to_numpy(dtype=float)

    rows: list[dict[str, object]] = []
    for score in scores:
        candidate_returns = matrix.to_numpy(dtype=float) @ score.weights.reindex(matrix.columns).fillna(0.0).to_numpy(dtype=float)
        score.delta_vs_hold = score.net_robust_score - hold_score.net_robust_score
        score.delta_vs_cash = score.net_robust_score - cash_score.net_robust_score
        score.probability_beats_hold = float(np.sum(probs[candidate_returns > hold_returns]))
        score.probability_beats_cash = float(np.sum(probs[candidate_returns > cash_returns]))
        rows.append(
            {
                "candidate": score.name,
                "mean_return": score.mean_return,
                "median_return": score.median_return,
                "probability_loss": score.probability_loss,
                "volatility": score.volatility,
                "downside_deviation": score.downside_deviation,
                "var_5": score.var_5,
                "cvar_5": score.cvar_5,
                "worst_scenario": score.worst_scenario,
                "risk_free_return": score.risk_free_return,
                "robust_sharpe": score.robust_sharpe,
                "return_over_volatility_legacy": score.return_over_volatility_legacy,
                "robust_score": score.robust_score,
                "turnover": score.turnover,
                "estimated_cost": score.estimated_cost,
                "dynamic_buffer": score.dynamic_buffer,
                "net_robust_score": score.net_robust_score,
                "delta_vs_hold": score.delta_vs_hold,
                "delta_vs_cash": score.delta_vs_cash,
                "probability_beats_hold": score.probability_beats_hold,
                "probability_beats_cash": score.probability_beats_cash,
            }
        )

    scores_frame = pd.DataFrame(rows).sort_values("net_robust_score", ascending=False).reset_index(drop=True)

    hurdle = float(params.get("hurdle", 0.001))
    risk_premium_hurdle = float(params.get("risk_premium_hurdle", 0.0005))
    p_hold_min = float(params.get("p_hold_min", 0.55))
    p_cash_min = float(params.get("p_cash_min", 0.52))

    selected_score = hold_score
    for score in scores_frame["candidate"]:
        candidate_score = score_map[str(score)]
        if (
            candidate_score.delta_vs_hold > hurdle
            and candidate_score.delta_vs_cash > risk_premium_hurdle
            and candidate_score.probability_beats_hold >= p_hold_min
            and candidate_score.probability_beats_cash >= p_cash_min
        ):
            selected_score = candidate_score
            break

    selected_candidate = candidates[selected_score.name]
    return RobustSelectionResult(
        selected_candidate=selected_candidate,
        selected_score=selected_score,
        scores=scores,
        scores_frame=scores_frame,
        mode=mode,
    )
