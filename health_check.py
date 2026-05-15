"""End-to-end project health check with robust PASS/WARN/SKIP semantics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PRICE_CACHE_PATH = BASE_DIR / "data" / "prices_cache.csv"
BERLIN_TZ = ZoneInfo("Europe/Berlin")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project health checks.")
    parser.add_argument("--quick", action="store_true", help="Prefer cache and avoid long checks.")
    parser.add_argument("--full", action="store_true", help="Run the full set including subprocess smoke runs.")
    return parser.parse_args()


def _row(check_name: str, status: str, message: str) -> dict[str, str]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "check_name": check_name,
        "status": status,
        "message": message,
    }


def _save_reports(rows: list[dict[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "system_health_report.csv"
    csv_temp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    df.to_csv(csv_temp, index=False)
    csv_temp.replace(csv_path)
    text = "\n".join(f"{row['status']:>4} | {row['check_name']} | {row['message']}" for row in rows)
    text_path = OUTPUT_DIR / "system_health_report.txt"
    text_temp = text_path.with_suffix(text_path.suffix + ".tmp")
    text_temp.write_text(text + "\n", encoding="utf-8")
    text_temp.replace(text_path)
    return df


def run_health_check(quick: bool = False, full: bool = False) -> pd.DataFrame:
    """Run the project health check."""
    rows: list[dict[str, str]] = []

    if sys.version_info < (3, 10):
        rows.append(_row("python_version", "FAIL", f"Python {sys.version.split()[0]} is below 3.10."))
        df = _save_reports(rows)
        return df
    rows.append(_row("python_version", "PASS", f"Python {sys.version.split()[0]}"))

    cache_exists = PRICE_CACHE_PATH.exists()
    module_policies = {
        "pandas": ("FAIL", "core dataframe dependency"),
        "numpy": ("FAIL", "core numeric dependency"),
        "yfinance": ("WARN" if cache_exists else "FAIL", "live downloader optional when cache is present"),
        "scipy": ("WARN", "optimizer fallback can run without SciPy"),
        "matplotlib": ("WARN", "charts are optional"),
        "dotenv": ("WARN", ".env loading has a safe fallback"),
    }
    for module_name, (failure_status, reason) in module_policies.items():
        try:
            importlib.import_module(module_name)
            rows.append(_row(f"import_{module_name}", "PASS", "import ok"))
        except Exception as exc:
            rows.append(_row(f"import_{module_name}", failure_status, f"{exc} | {reason}"))

    for optional_module in ("gurobipy",):
        try:
            importlib.import_module(optional_module)
            rows.append(_row(f"optional_import_{optional_module}", "PASS", "available"))
        except Exception as exc:
            rows.append(_row(f"optional_import_{optional_module}", "WARN", f"optional: {exc}"))

    if any(
        row["status"] == "FAIL" and row["check_name"].startswith("import_")
        for row in rows
    ):
        return _save_reports(rows)

    from asset_universe import validate_asset_universe
    from calendar_utils import (
        is_project_trading_day,
        is_within_project_trading_window,
        load_trading_calendar,
    )
    from candidate_factory import build_candidate_portfolios
    from config import build_params
    from config_validation import validate_config
    from data import check_data_freshness, load_price_cache, load_price_data, write_data_freshness_report
    from data_quality import compute_data_quality_report
    from database import init_db, save_health_check_to_db
    from discrete_portfolio_optimizer import (
        build_discrete_order_preview,
        generate_discrete_candidates,
        load_current_portfolio_state,
        score_discrete_candidates,
        select_best_discrete_portfolio,
    )
    from ensemble_model import build_model_ensemble_outputs
    from execution_gate import evaluate_execution_gate
    from features import compute_returns
    from feasibility import check_portfolio_feasibility
    from forecast_3m import build_forecast_3m
    from interface_tests import run_interface_smoke_tests
    from investopedia_adapter import InvestopediaSimulatorAdapter
    from model_governance import compute_model_confidence
    from notifications import email_settings_complete, load_email_settings
    from optimizer import build_feasible_initial_weights, optimize_allocation
    from order_preview import generate_order_preview
    from pre_trade_validation import run_pre_trade_validation
    from reconciliation import reconcile_before_execution
    from regime_engine import detect_regime
    from robust_scorer import select_robust_candidate
    from robustness_tests import run_robustness_tests
    from scenario_model import build_3m_scenarios
    from system_init import run_system_initialization
    from trade_sizing import compute_trade_fraction
    from tradability import apply_tradability_filter, build_tradability_report, select_cash_proxy
    from transaction_costs import estimate_order_cost, estimate_order_list_costs

    params = build_params()

    calendar_path = BASE_DIR / "data" / "trading_calendar_2026.csv"
    try:
        calendar_df = load_trading_calendar(calendar_path)
        rows.append(_row("project_calendar_exists", "PASS", str(calendar_path)))
        min_date = str(calendar_df["date"].min())
        max_date = str(calendar_df["date"].max())
        rows.append(
            _row(
                "project_calendar_range",
                "PASS" if (min_date == "2026-04-24" and max_date == "2026-07-24") else "FAIL",
                f"{min_date} -> {max_date}",
            )
        )
        rule_failures: list[str] = []
        for date_text, expected in {
            "2026-05-25": False,
            "2026-06-19": False,
            "2026-07-03": False,
            "2026-04-27": True,
            "2026-04-25": False,
        }.items():
            actual = is_project_trading_day(date_text, calendar_df)
            if bool(actual) != bool(expected):
                rule_failures.append(f"{date_text}: expected {expected} got {actual}")
        rows.append(
            _row(
                "project_calendar_rules",
                "PASS" if not rule_failures else "FAIL",
                "; ".join(rule_failures) if rule_failures else "holiday/weekend rules ok",
            )
        )
        smoke_failures: list[str] = []
        for moment, expected_allowed, expected_reason in [
            (datetime(2026, 4, 24, 16, 30, tzinfo=BERLIN_TZ), True, "within_project_trading_window"),
            (datetime(2026, 4, 24, 15, 45, tzinfo=BERLIN_TZ), False, "outside_allowed_window"),
            (datetime(2026, 5, 25, 17, 0, tzinfo=BERLIN_TZ), False, "Memorial Day"),
            (datetime(2026, 6, 20, 17, 0, tzinfo=BERLIN_TZ), False, "weekend"),
            (datetime(2026, 7, 24, 21, 30, tzinfo=BERLIN_TZ), True, "within_project_trading_window"),
            (datetime(2026, 7, 24, 22, 30, tzinfo=BERLIN_TZ), False, "outside_allowed_window"),
            (datetime(2026, 8, 1, 17, 0, tzinfo=BERLIN_TZ), False, "date_not_in_project_calendar"),
        ]:
            result = is_within_project_trading_window(now=moment, calendar_path=calendar_path)
            if bool(result["execution_allowed"]) != bool(expected_allowed) or str(result["reason"]) != expected_reason:
                smoke_failures.append(
                    f"{moment.isoformat()} expected ({expected_allowed}, {expected_reason}) got ({result['execution_allowed']}, {result['reason']})"
                )
        rows.append(
            _row(
                "project_calendar_window",
                "PASS" if not smoke_failures else "FAIL",
                "; ".join(smoke_failures) if smoke_failures else "time-window behavior ok",
            )
        )
    except Exception as exc:
        rows.append(_row("project_calendar", "FAIL", str(exc)))

    init_result = run_system_initialization(params)
    rows.append(
        _row(
            "system_initialization",
            "PASS" if init_result["ok"] else ("WARN" if not init_result["errors"] else "FAIL"),
            "; ".join([*init_result["warnings"], *init_result["errors"]]) or "ok",
        )
    )

    env_example = BASE_DIR / ".env.example"
    rows.append(_row("env_example", "PASS" if env_example.exists() else "FAIL", str(env_example)))
    rows.append(_row("env_file_optional", "PASS" if (BASE_DIR / ".env").exists() else "WARN", ".env optional"))

    try:
        validate_asset_universe()
        rows.append(_row("asset_universe", "PASS", "registry valid"))
    except Exception as exc:
        rows.append(_row("asset_universe", "FAIL", str(exc)))

    try:
        config_check = validate_config(params)
        rows.append(
            _row(
                "config_validation",
                "PASS" if config_check["ok"] else "FAIL",
                "; ".join([*config_check["warnings"], *config_check["errors"]]) or "ok",
            )
        )
    except Exception as exc:
        rows.append(_row("config_validation", "FAIL", str(exc)))

    prices = pd.DataFrame()
    if quick:
        try:
            prices = load_price_data(
                tickers=list(params["tickers"]),
                start_date=str(params["start_date"]),
                end_date=params.get("end_date"),
                cache_path=PRICE_CACHE_PATH,
                use_cache=True,
                prefer_live=False,
                allow_cache_fallback=True,
                force_refresh=False,
            )
        except Exception as exc:
            rows.append(_row("price_data", "FAIL", f"quick cache load failed: {exc}"))
            df = _save_reports(rows)
            try:
                save_health_check_to_db(params["db_path"], df)
            except Exception:
                pass
            return df
    else:
        try:
            prices = load_price_data(
                tickers=list(params["tickers"]),
                start_date=str(params["start_date"]),
                end_date=params.get("end_date"),
                cache_path=PRICE_CACHE_PATH,
                use_cache=True,
                prefer_live=True,
                allow_cache_fallback=True,
                force_refresh=bool(full),
            )
        except Exception as exc:
            rows.append(_row("price_data", "FAIL", str(exc)))
            df = _save_reports(rows)
            try:
                save_health_check_to_db(params["db_path"], df)
            except Exception:
                pass
            return df

    price_status = "WARN" if bool(prices.attrs.get("synthetic_data", False)) else "PASS"
    price_message = f"{len(prices)} rows, {prices.shape[1]} columns"
    if bool(prices.attrs.get("synthetic_data", False)):
        price_message += " | synthetic fallback active"
    if bool(prices.attrs.get("used_cache_fallback", False)):
        price_message += " | cache fallback used"
    rows.append(_row("price_data", price_status, price_message))
    data_freshness = check_data_freshness(prices)
    market_gate = is_within_project_trading_window(calendar_path=calendar_path)
    write_data_freshness_report(
        prices=prices,
        freshness=data_freshness,
        output_path=OUTPUT_DIR / "current_data_freshness_report.txt",
        market_gate=market_gate,
    )
    rows.append(
        _row(
            "yfinance_live_availability",
            "PASS" if bool(prices.attrs.get("yfinance_available", False)) else "WARN",
            (
                "yfinance import available; quick mode used cache-preferred path."
                if quick and bool(prices.attrs.get("yfinance_available", False))
                else (
                    "yfinance import unavailable in current environment."
                    if not bool(prices.attrs.get("yfinance_available", False))
                    else f"data_source={prices.attrs.get('data_source', 'unknown')}"
                )
            ),
        )
    )
    rows.append(
        _row(
            "data_freshness",
            "PASS" if bool(data_freshness.get("data_freshness_ok", False)) else "WARN",
            data_freshness.get("warning") or f"latest_price_date={data_freshness.get('latest_price_date')}",
        )
    )
    rows.append(_row("price_assets_minimum", "PASS" if prices.shape[1] >= 10 else "FAIL", f"{prices.shape[1]} assets"))
    cash_status = "PASS" if ("SGOV" in prices.columns or "SHY" in prices.columns) else "FAIL"
    rows.append(_row("cash_proxy_available", cash_status, "SGOV/SHY check"))

    try:
        init_db(params["db_path"])
        rows.append(_row("sqlite", "PASS", "database initialized"))
    except Exception as exc:
        rows.append(_row("sqlite", "FAIL", str(exc)))

    smoke_prices = prices.reindex(columns=list(params["tickers"])).copy()
    available_tickers = [ticker for ticker in params["tickers"] if ticker in smoke_prices.columns]
    try:
        tradability_df = build_tradability_report(
            tickers=list(params["tickers"]),
            prices=prices,
            enable_local_paper=bool(params.get("enable_local_paper_trading", False)),
            enable_investopedia=bool(params.get("enable_investopedia_simulator", False)),
            dry_run=bool(params.get("dry_run", True)),
        )
        available_tickers = apply_tradability_filter(
            tickers=list(params["tickers"]),
            tradability_df=tradability_df,
            min_assets=10,
        )
        smoke_prices = prices.reindex(columns=available_tickers).copy()
        rows.append(_row("tradability", "PASS", f"{len(available_tickers)} tradable assets"))
    except Exception as exc:
        rows.append(_row("tradability", "FAIL", str(exc)))
    if len(available_tickers) < 10 and bool(prices.attrs.get("synthetic_data", False)):
        available_tickers = list(params["tickers"])
        smoke_prices = prices.reindex(columns=available_tickers)

    try:
        quality_prices = smoke_prices.reindex(columns=available_tickers).copy()
        quality_prices.attrs.update(smoke_prices.attrs)
        returns = compute_returns(quality_prices)
        data_quality_report = compute_data_quality_report(
            prices=quality_prices,
            returns=returns,
            active_tickers=available_tickers,
            params=build_params(tickers=available_tickers),
        )
        dq_status = "PASS"
        if data_quality_report["errors"]:
            dq_status = "FAIL"
        elif data_quality_report["warnings"]:
            dq_status = "WARN"
        rows.append(
            _row(
                "data_quality",
                dq_status,
                f"score={float(data_quality_report['global_data_quality_score']):.3f}; "
                + "; ".join([*data_quality_report["warnings"], *data_quality_report["errors"]]),
            )
        )
    except Exception as exc:
        rows.append(_row("data_quality", "FAIL", str(exc)))
        data_quality_report = {"global_data_quality_score": 0.5, "warnings": [], "errors": []}

    try:
        feasibility_result = check_portfolio_feasibility(
            active_tickers=available_tickers,
            params={**build_params(tickers=available_tickers), "max_equity_like_total": params["max_equity_like_total_normal"], "min_defensive_weight": params["min_defensive_weight_normal"]},
        )
        rows.append(
            _row(
                "feasibility",
                "PASS" if feasibility_result["feasible"] else "FAIL",
                "; ".join([*feasibility_result["warnings"], *feasibility_result["errors"]]) or "ok",
            )
        )
    except Exception as exc:
        rows.append(_row("feasibility", "FAIL", str(exc)))

    try:
        as_of = pd.Timestamp(returns.index[-1])
        forecast = build_forecast_3m(smoke_prices, returns=returns, date=as_of, params=params, tickers=available_tickers)
        rows.append(_row("forecast_3m", "PASS", f"{len(forecast.table)} assets"))
    except Exception as exc:
        rows.append(_row("forecast_3m", "FAIL", str(exc)))
        forecast = None
        returns = pd.DataFrame()
        as_of = None

    try:
        if forecast is None or as_of is None:
            raise ValueError("forecast unavailable")
        sigma = pd.DataFrame(
            data=returns.cov().reindex(index=available_tickers, columns=available_tickers).fillna(0.0).to_numpy() * 63.0,
            index=available_tickers,
            columns=available_tickers,
        )
        w_current = build_feasible_initial_weights(available_tickers, build_params(tickers=available_tickers))
        optimizer_result = optimize_allocation(
            mu=forecast.table["expected_return_3m"],
            Sigma=sigma,
            w_current=w_current,
            params={
                **build_params(tickers=available_tickers),
                "max_equity_like_total": params["max_equity_like_total_normal"],
                "min_defensive_weight": params["min_defensive_weight_normal"],
            },
        )
        rows.append(_row("optimizer", "PASS", optimizer_result.solver_name))
    except Exception as exc:
        rows.append(_row("optimizer", "FAIL", str(exc)))
        optimizer_result = None
        w_current = pd.Series(dtype=float)

    try:
        if forecast is None or as_of is None or optimizer_result is None:
            raise ValueError("upstream scenario inputs unavailable")
        scenario_set = build_3m_scenarios(
            forecast_table=forecast.table,
            covariance_3m=sigma,
            risk_state=forecast.risk_state,
            as_of=as_of,
        )
        rows.append(_row("scenario_model", "PASS", ",".join(scenario_set.scenario_names)))
    except Exception as exc:
        rows.append(_row("scenario_model", "FAIL", str(exc)))
        scenario_set = None

    try:
        if forecast is None or optimizer_result is None or scenario_set is None:
            raise ValueError("candidate inputs unavailable")
        candidates = build_candidate_portfolios(
            w_current=w_current,
            w_target=optimizer_result.target_weights,
            forecast_table=forecast.table,
            params=build_params(tickers=available_tickers),
        )
        rows.append(_row("candidate_factory", "PASS", ",".join(candidates.keys())))
    except Exception as exc:
        rows.append(_row("candidate_factory", "FAIL", str(exc)))
        candidates = None

    try:
        if candidates is None or scenario_set is None:
            raise ValueError("robust scorer inputs unavailable")
        selection = select_robust_candidate(
            candidates=candidates,
            scenario_set=scenario_set,
            w_current=w_current,
            params=build_params(tickers=available_tickers),
        )
        rows.append(_row("robust_scorer", "PASS", selection.selected_candidate.name))
    except Exception as exc:
        rows.append(_row("robust_scorer", "FAIL", str(exc)))
        selection = None

    try:
        if selection is None:
            raise ValueError("execution gate inputs unavailable")
        gate = evaluate_execution_gate(selection_result=selection, synthetic_data=bool(prices.attrs.get("synthetic_data", False)))
        status = "PASS" if gate.gate_status in {"PASS", "BLOCK"} else "WARN"
        rows.append(_row("execution_gate", status, f"{gate.gate_status}:{gate.action}"))
    except Exception as exc:
        rows.append(_row("execution_gate", "FAIL", str(exc)))
        gate = None

    try:
        if optimizer_result is None or w_current.empty:
            raise ValueError("order preview inputs unavailable")
        latest_prices = smoke_prices.loc[as_of].reindex(available_tickers)
        preview = generate_order_preview(
            w_current=w_current,
            w_target=optimizer_result.target_weights,
            latest_prices=latest_prices,
            portfolio_value=10000.0,
            output_path=None,
        )
        rows.append(_row("order_preview", "PASS", f"{len(preview)} rows"))
    except Exception as exc:
        rows.append(_row("order_preview", "FAIL", str(exc)))
        preview = pd.DataFrame()
        latest_prices = pd.Series(dtype=float)

    try:
        if optimizer_result is None or w_current.empty or preview.empty:
            raise ValueError("pre-trade validation inputs unavailable")
        validation = run_pre_trade_validation(
            w_current=w_current,
            w_target=optimizer_result.target_weights,
            latest_prices=latest_prices,
            order_preview_df=preview,
            params={
                **build_params(tickers=available_tickers),
                "max_equity_like_total": params["max_equity_like_total_normal"],
                "min_defensive_weight": params["min_defensive_weight_normal"],
            },
        )
        rows.append(
            _row(
                "pre_trade_validation",
                "PASS" if validation["ok"] else "FAIL",
                "; ".join([*validation["warnings"], *validation["errors"]]) or "ok",
            )
        )
    except Exception as exc:
        rows.append(_row("pre_trade_validation", "FAIL", str(exc)))

    try:
        if optimizer_result is None:
            raise ValueError("reconciliation inputs unavailable")
        reconciliation = reconcile_before_execution(
            model_weights=optimizer_result.target_weights,
            latest_prices=latest_prices if isinstance(latest_prices, pd.Series) else pd.Series(dtype=float),
            execution_mode="order_preview_only",
        )
        rows.append(
            _row(
                "reconciliation",
                str(reconciliation.get("status", "SKIP")),
                str(reconciliation.get("message", "ok")),
            )
        )
    except Exception as exc:
        rows.append(_row("reconciliation", "WARN", str(exc)))

    try:
        regime = detect_regime(smoke_prices, as_of)
        rows.append(_row("regime_engine", "PASS", f"{regime['primary_regime']} / {regime['risk_state']}"))
    except Exception as exc:
        rows.append(_row("regime_engine", "WARN", str(exc)))
        regime = {"primary_regime": "neutral", "risk_state": "normal"}

    try:
        if candidates is None:
            raise ValueError("ensemble inputs unavailable")
        ensemble = build_model_ensemble_outputs(
            optimizer_target=candidates["OPTIMIZER_TARGET"].weights,
            defensive_cash=candidates["DEFENSIVE_CASH"].weights,
            momentum_tilt=candidates["MOMENTUM_TILT_SIMPLE"].weights,
            conditional_factor_target=candidates.get("CONDITIONAL_FACTOR_TARGET").weights if "CONDITIONAL_FACTOR_TARGET" in candidates else None,
        )
        rows.append(_row("ensemble_model", "PASS", f"agreement={ensemble['agreement_score']:.3f}"))
    except Exception as exc:
        rows.append(_row("ensemble_model", "WARN", str(exc)))
        ensemble = None

    try:
        if optimizer_result is None or forecast is None or scenario_set is None:
            raise ValueError("model governance inputs unavailable")
        governance = compute_model_confidence(
            forecast_report=forecast,
            factor_report=pd.DataFrame(),
            scenario_report=scenario_set.summary,
            optimizer_result=optimizer_result,
            data_quality_report=data_quality_report,
        )
        rows.append(_row("model_governance", "PASS", f"score={governance['model_confidence_score']:.3f}"))
    except Exception as exc:
        rows.append(_row("model_governance", "WARN", str(exc)))
        governance = {"model_confidence_score": 0.5, "model_uncertainty_buffer": 0.001}

    try:
        if selection is None or gate is None:
            raise ValueError("trade sizing inputs unavailable")
        sizing = compute_trade_fraction(
            selection_result=selection,
            execution_gate_result=gate,
            model_confidence=governance,
            data_quality=data_quality_report,
            turnover_budget={"turnover_budget_remaining": 1.0},
        )
        rows.append(_row("trade_sizing", "PASS", f"{sizing['suggested_action']} @ {float(sizing['trade_fraction']):.2f}"))
    except Exception as exc:
        rows.append(_row("trade_sizing", "WARN", str(exc)))

    try:
        if candidates is None or scenario_set is None or forecast is None or as_of is None:
            raise ValueError("discrete optimizer inputs unavailable")
        latest_prices = smoke_prices.loc[as_of].reindex(available_tickers).astype(float)
        cash_proxy_ticker = select_cash_proxy(available_tickers, tradability_df)
        current_state = load_current_portfolio_state(
            params=params,
            active_tickers=available_tickers,
            latest_prices=latest_prices,
            cash_proxy_ticker=cash_proxy_ticker,
            nav=float(params.get("portfolio_nav_usd", 100000.0)),
        )
        rows.append(
            _row(
                "current_portfolio_state",
                "PASS",
                f"source={current_state.source}; cash_weight={current_state.actual_cash_weight:.3f}",
            )
        )
        hold_candidate_from_current = build_candidate_portfolios(
            w_current=current_state.current_weights_proxy.reindex(available_tickers).fillna(0.0),
            w_target=candidates["OPTIMIZER_TARGET"].weights.reindex(available_tickers).fillna(0.0),
            forecast_table=forecast.table,
            params=params,
        )["HOLD"]
        hold_matches_current = bool(
            hold_candidate_from_current.weights.reindex(available_tickers).fillna(0.0).round(10).equals(
                current_state.current_weights_proxy.reindex(available_tickers).fillna(0.0).round(10)
            )
        )
        rows.append(
            _row(
                "hold_means_current_portfolio",
                "PASS" if hold_matches_current else "FAIL",
                "HOLD candidate matches the actual current portfolio proxy weights."
                if hold_matches_current
                else "HOLD candidate does not match the actual current portfolio proxy weights.",
            )
        )
        continuous_target_sum = float(candidates["OPTIMIZER_TARGET"].weights.reindex(available_tickers).fillna(0.0).sum())
        rows.append(
            _row(
                "continuous_target_weights_sum",
                "PASS" if abs(continuous_target_sum - 1.0) <= 1e-6 else "FAIL",
                f"sum={continuous_target_sum:.6f}",
            )
        )
        positions_present = bool(current_state.current_shares.fillna(0.0).abs().gt(1e-9).any())
        current_portfolio_consistent = bool(
            current_state.source == "csv"
            and (
                (positions_present and current_state.actual_cash_weight < 0.999)
                or ((not positions_present) and current_state.actual_cash_weight >= 0.999)
            )
        )
        rows.append(
            _row(
                "current_portfolio_100pct_cash",
                "PASS" if current_portfolio_consistent else "WARN",
                f"cash_weight={current_state.actual_cash_weight:.6f}; positions_present={positions_present}",
            )
        )

        discrete_candidates = generate_discrete_candidates(
            target_weights=candidates["OPTIMIZER_TARGET"].weights.reindex(available_tickers).fillna(0.0),
            latest_prices=latest_prices,
            nav=float(current_state.nav),
            current_positions=current_state.current_shares.reindex(available_tickers).fillna(0.0),
            current_cash=float(current_state.current_cash),
            min_order_value=float(params.get("min_order_value_usd", 10.0)),
            cash_buffer=0.0,
            max_candidates=25,
            allow_fractional_shares=bool(params.get("allow_fractional_shares", False)),
            marginal_priority=forecast.table["expected_return_3m"] * forecast.table["signal_confidence"],
            cash_proxy_ticker=cash_proxy_ticker,
        )
        if not discrete_candidates:
            raise ValueError("no discrete candidates generated for OPTIMIZER_TARGET smoke test")
        rows.append(_row("discrete_candidate_generation", "PASS", f"{len(discrete_candidates)} candidates"))

        discrete_scored = score_discrete_candidates(
            discrete_candidates=discrete_candidates,
            scenario_returns=scenario_set,
            scorer_config={
                "params": params,
                "hold_weights": candidates["HOLD"].weights,
                "cash_weights": candidates["DEFENSIVE_CASH"].weights,
                "continuous_target": candidates["OPTIMIZER_TARGET"].weights,
            },
            current_weights=current_state.current_weights_proxy.reindex(available_tickers).fillna(0.0),
            current_shares=current_state.current_shares.reindex(available_tickers).fillna(0.0),
            current_cash=float(current_state.current_cash),
            latest_prices=latest_prices,
            nav=float(current_state.nav),
        )
        discrete_selection = select_best_discrete_portfolio(discrete_scored)
        best_discrete = discrete_selection["candidate"]
        preview = build_discrete_order_preview(
            current_shares=current_state.current_shares.reindex(available_tickers).fillna(0.0),
            target_shares=best_discrete.shares.reindex(available_tickers).fillna(0.0),
            latest_prices=latest_prices,
            nav=float(current_state.nav),
            min_order_value=float(params.get("min_order_value_usd", 10.0)),
        )
        preview, _ = estimate_order_list_costs(
            order_preview_df=preview,
            latest_prices=latest_prices,
            config={**params, "nav": float(current_state.nav), "current_cash": float(current_state.current_cash)},
        )

        whole_share_ok = bool(
            (
                best_discrete.shares.reindex(available_tickers).fillna(0.0)
                - best_discrete.shares.reindex(available_tickers).fillna(0.0).round()
            ).abs().le(1e-9).all()
        ) if not bool(params.get("allow_fractional_shares", False)) else True
        no_negative_shares = bool(best_discrete.shares.fillna(0.0).ge(-1e-9).all())
        no_negative_cash = bool(best_discrete.cash_left >= -1e-8)
        no_leverage = bool(best_discrete.weights_actual.fillna(0.0).sum() <= float(params.get("max_gross_exposure", 1.0)) + 1e-8)
        weights_sum_ok = bool(abs(float(best_discrete.weights_actual.fillna(0.0).sum()) + float(best_discrete.cash_weight) - 1.0) <= 1e-3)
        has_buy_orders_from_cash = False
        if current_state.actual_cash_weight >= 0.999:
            for candidate in discrete_candidates:
                if candidate.name == "HOLD_CURRENT":
                    continue
                candidate_preview = build_discrete_order_preview(
                    current_shares=current_state.current_shares.reindex(available_tickers).fillna(0.0),
                    target_shares=candidate.shares.reindex(available_tickers).fillna(0.0),
                    latest_prices=latest_prices,
                    nav=float(current_state.nav),
                    min_order_value=float(params.get("min_order_value_usd", 10.0)),
                )
                candidate_preview, _ = estimate_order_list_costs(
                    order_preview_df=candidate_preview,
                    latest_prices=latest_prices,
                    config={**params, "nav": float(current_state.nav), "current_cash": float(current_state.current_cash)},
                )
                if bool((candidate_preview["side"] == "BUY").any()):
                    has_buy_orders_from_cash = True
                    break
        else:
            has_buy_orders_from_cash = True
        discrete_status = "PASS" if all(
            [
                whole_share_ok,
                no_negative_shares,
                no_negative_cash,
                no_leverage,
                weights_sum_ok,
                has_buy_orders_from_cash,
            ]
        ) else "FAIL"
        rows.append(
            _row(
                "discrete_whole_share_logic",
                discrete_status,
                (
                    f"candidate={discrete_selection['best_discrete_candidate_name']}; "
                    f"whole_shares={whole_share_ok}; no_negative_cash={no_negative_cash}; "
                    f"no_negative_shares={no_negative_shares}; no_leverage={no_leverage}; "
                    f"weights_plus_cash_sum_ok={weights_sum_ok}; buy_orders_from_cash={has_buy_orders_from_cash}"
                ),
            )
        )
        rows.append(
            _row(
                "discrete_weights_plus_cash_sum",
                "PASS" if weights_sum_ok else "FAIL",
                f"sum={float(best_discrete.weights_actual.fillna(0.0).sum()) + float(best_discrete.cash_weight):.6f}",
            )
        )
        required_cost_columns = {
            "estimated_commission",
            "estimated_spread_cost",
            "estimated_slippage_cost",
            "estimated_total_order_cost",
            "cost_bps_used",
        }
        rows.append(
            _row(
                "discrete_order_cost_columns",
                "PASS" if required_cost_columns.issubset(set(preview.columns)) else "FAIL",
                ",".join(sorted(required_cost_columns - set(preview.columns))) or "present",
            )
        )
        selected_row = discrete_selection["scores_frame"].loc[
            discrete_selection["scores_frame"]["discrete_candidate"].astype(str)
            == str(discrete_selection["best_discrete_candidate_name"])
        ].iloc[0]
        preview_total_order_cost = float(preview["estimated_total_order_cost"].fillna(0.0).sum())
        rows.append(
            _row(
                "final_decision_uses_discrete_order_cost",
                "PASS" if abs(float(selected_row["total_order_cost"]) - preview_total_order_cost) <= 1e-6 else "FAIL",
                f"selected={float(selected_row['total_order_cost']):.6f}; preview={preview_total_order_cost:.6f}",
            )
        )

        small_nav_candidates = generate_discrete_candidates(
            target_weights=candidates["OPTIMIZER_TARGET"].weights.reindex(available_tickers).fillna(0.0),
            latest_prices=latest_prices,
            nav=10000.0,
            current_positions=pd.Series(0.0, index=available_tickers, dtype=float),
            current_cash=10000.0,
            min_order_value=float(params.get("min_order_value_usd", 10.0)),
            cash_buffer=0.0,
            max_candidates=10,
            allow_fractional_shares=bool(params.get("allow_fractional_shares", False)),
            marginal_priority=forecast.table["expected_return_3m"] * forecast.table["signal_confidence"],
            cash_proxy_ticker=cash_proxy_ticker,
        )
        if not small_nav_candidates:
            raise ValueError("NAV=10000 discrete candidate smoke test produced no candidates.")
        small_nav_scored = score_discrete_candidates(
            discrete_candidates=small_nav_candidates,
            scenario_returns=scenario_set,
            scorer_config={
                "params": params,
                "hold_weights": candidates["HOLD"].weights,
                "cash_weights": candidates["DEFENSIVE_CASH"].weights,
                "continuous_target": candidates["OPTIMIZER_TARGET"].weights,
            },
            current_weights=current_state.current_weights_proxy.reindex(available_tickers).fillna(0.0),
            current_shares=pd.Series(0.0, index=available_tickers, dtype=float),
            current_cash=10000.0,
            latest_prices=latest_prices,
            nav=10000.0,
        )
        small_nav_selection = select_best_discrete_portfolio(small_nav_scored)
        small_nav_preview = build_discrete_order_preview(
            current_shares=pd.Series(0.0, index=available_tickers, dtype=float),
            target_shares=small_nav_selection["candidate"].shares.reindex(available_tickers).fillna(0.0),
            latest_prices=latest_prices,
            nav=10000.0,
            min_order_value=float(params.get("min_order_value_usd", 10.0)),
        )
        small_nav_preview, _ = estimate_order_list_costs(
            order_preview_df=small_nav_preview,
            latest_prices=latest_prices,
            config={**params, "nav": 10000.0, "current_cash": 10000.0},
        )
        rows.append(
            _row(
                "discrete_small_nav_smoke",
                "PASS",
                (
                    f"best={small_nav_selection['best_discrete_candidate_name']}; "
                    f"orders={int((small_nav_preview['side'] != 'HOLD').sum())}; "
                    f"cash_left={float(small_nav_selection['candidate'].cash_left):.2f}"
                ),
            )
        )
        sgov_cost = estimate_order_cost("SGOV", "BUY", 10.0, 100.0, order_value=1000.0, config=params)
        ibit_cost = estimate_order_cost("IBIT", "BUY", 10.0, 100.0, order_value=1000.0, config=params)
        rows.append(
            _row(
                "transaction_cost_differentiation",
                "PASS" if float(ibit_cost["cost_bps_used"]) > float(sgov_cost["cost_bps_used"]) else "FAIL",
                f"SGOV={sgov_cost['cost_bps_used']:.2f}bps IBIT={ibit_cost['cost_bps_used']:.2f}bps",
            )
        )
        rows.append(
            _row(
                "live_bid_ask_costs_available",
                "WARN" if not bool(ibit_cost["live_costs_available"]) else "PASS",
                "modeled costs used because no live bid/ask feed is available in quick health-check mode."
                if not bool(ibit_cost["live_costs_available"])
                else "live bid/ask available",
            )
        )
    except Exception as exc:
        rows.append(_row("discrete_whole_share_logic", "FAIL", str(exc)))

    if quick:
        rows.append(_row("interface_tests_internal", "SKIP", "Skipped in quick mode to avoid subprocess and live-data delays."))
    else:
        try:
            messages = run_interface_smoke_tests()
            rows.append(_row("interface_tests_internal", "PASS", f"{len(messages)} checks"))
        except Exception as exc:
            rows.append(_row("interface_tests_internal", "WARN", str(exc)))

    try:
        adapter = InvestopediaSimulatorAdapter.from_env()
        if not adapter.settings.enabled:
            rows.append(_row("investopedia_adapter", "WARN", "disabled"))
        elif not adapter.settings.credentials_present:
            rows.append(_row("investopedia_adapter", "WARN", "credentials missing"))
        else:
            rows.append(_row("investopedia_adapter", "WARN", "stub/login not tested"))
    except Exception as exc:
        rows.append(_row("investopedia_adapter", "WARN", str(exc)))

    try:
        email_settings = load_email_settings()
        if not bool(email_settings.get("ENABLE_EMAIL_NOTIFICATIONS", False)):
            rows.append(_row("email_status", "WARN", "disabled"))
        elif not bool(email_settings.get("ENV_FILE_PRESENT", False)):
            rows.append(_row("email_status", "WARN", ".env missing"))
        elif not email_settings_complete(email_settings):
            rows.append(
                _row(
                    "email_status",
                    "WARN",
                    f"incomplete {email_settings.get('EMAIL_PROVIDER', 'email')} provider config",
                )
            )
        else:
            rows.append(_row("email_status", "PASS", f"{email_settings.get('EMAIL_PROVIDER', 'email')} configured"))
    except Exception as exc:
        rows.append(_row("email_status", "WARN", str(exc)))

    for name, module_name in (("main_importable", "main"), ("daily_bot_importable", "daily_bot")):
        try:
            importlib.import_module(module_name)
            rows.append(_row(name, "PASS", "import ok"))
        except Exception as exc:
            rows.append(_row(name, "FAIL", str(exc)))

    try:
        robustness_df = run_robustness_tests()
        status = "PASS" if not (robustness_df["status"] == "FAIL").any() else "WARN"
        rows.append(_row("robustness_tests_importable", status, f"{len(robustness_df)} tests"))
    except Exception as exc:
        rows.append(_row("robustness_tests_importable", "WARN", str(exc)))

    if not full:
        rows.append(_row("daily_bot_full_run", "SKIP", "Run with --full to execute subprocess smoke runs."))
    else:
        for name, command in (
            ("main_importable", [sys.executable, "-c", "import main"]),
            ("daily_bot_dry_run", [sys.executable, "daily_bot.py", "--dry-run", "--mode", "single"]),
        ):
            try:
                result = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parent,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                status = "PASS" if result.returncode == 0 else "FAIL"
                message = "ok" if result.returncode == 0 else (result.stderr or result.stdout)[-400:]
                rows.append(_row(name, status, message))
            except Exception as exc:
                rows.append(_row(name, "WARN", str(exc)))

    df = _save_reports(rows)
    try:
        save_health_check_to_db(params["db_path"], df)
    except Exception as exc:
        LOGGER.warning("Could not persist health-check rows to SQLite: %s", exc)
    return df


def main() -> None:
    setup_logging()
    args = parse_args()
    quick = bool(args.quick) or not bool(args.full)
    df = run_health_check(quick=quick, full=bool(args.full))
    print(df[["status", "check_name", "message"]].to_string(index=False))


if __name__ == "__main__":
    main()
