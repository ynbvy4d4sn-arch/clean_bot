# STATUS

Stand: 2026-05-05

Sichere Defaults verifiziert:
- `DRY_RUN=true`
- `ENABLE_INVESTOPEDIA_SIMULATOR=false`
- `ENABLE_LOCAL_PAPER_TRADING=false`
- `ENABLE_EMAIL_NOTIFICATIONS=false`
- `ENABLE_EXTERNAL_BROKER=false`

## Update 2026-05-05 - Letzter run_daily_dry_run.sh-Lauf im offenen Handelsfenster analysiert

Geaenderte Dateien:
- `outputs/today_decision_summary.txt`
- `outputs/rebalance_decision_report.txt`
- `outputs/daily_bot_decision_report.txt`
- `outputs/final_acceptance_report.txt`
- `STATUS.md`

Kernbefunde aus dem letzten One-Command-Lauf:
- `current_time_berlin=16:51`
- `is_project_trading_day=true`
- `within_allowed_window=true`
- `execution_allowed_by_calendar=true`
- `calendar_reason=within_project_trading_window`
- `data_source=cache_fallback`
- `used_cache_fallback=true`
- `latest_price_date=2026-05-05`
- `final_action=HOLD`
- `execution_mode=order_preview_only`

Erster Blocker:
- `execution_gate:trade_now_edge_below_hurdle`

Execution-Gate-Befund:
- Das Kalenderfenster war offen; der Kalender blockiert also diesmal **nicht**.
- Das Execution Gate blockiert fachlich wegen fehlender Net-Edge nach Kosten und Buffern:
  - `trade_now_edge=-0.001923`
  - `best_discrete_candidate=HOLD_CURRENT`
  - keine diskrete BUY/SELL-Liste ueberwindet die Huerde gegen HOLD

Manual-Simulator-Datei:
- `outputs/manual_simulator_orders.csv` ist fachlich korrekt, aber aktuell nicht zur Eingabe bestimmt.
- Die Datei ist leer, weil keine BUY/SELL-Delta-Order freigegeben wurde.

Verifizierte Checks dieser Runde:
- `./run_daily_dry_run.sh` -> PASS

Bekannte offene Grenzen:
- `used_cache_fallback=true` bleibt ein Vorsichtspunkt fuer jeden nicht-nur-Preview-Pfad.
- `price_basis=adjusted_close_proxy` bleibt ein konservativer Bewertungs-/Preview-Preis statt eines Live-Quotes.
- `scenario_model.py`, Local-Paper-State und `paper_broker_stub.py` bleiben grosse offene Systemthemen.

## Update 2026-05-05 - Kostenmodus mit frischem Open-Window-Dry-Run verifiziert

Geaenderte Dateien:
- `transaction_costs.py`
- `daily_bot.py`
- `robustness_tests.py`
- `outputs/transaction_cost_report.txt`
- `outputs/transaction_cost_audit.txt`
- `outputs/rebalance_decision_report.txt`
- `outputs/today_decision_summary.txt`
- `STATUS.md`

Kernbefunde aus dem frischen Dry-Run:
- `current_time_berlin=16:49`
- `is_project_trading_day=true`
- `within_allowed_window=true`
- `execution_allowed_by_calendar=true`
- `calendar_reason=within_project_trading_window`
- `data_source=yfinance`
- `used_cache_fallback=false`
- `latest_price_date=2026-05-05`
- `continuous_candidate=HOLD`
- `final_discrete_candidate=HOLD_CURRENT`
- `final_action=HOLD`
- `execution_mode=order_preview_only`

Kostenmodus jetzt klar und aktuell verifiziert:
- `commission_per_trade_usd=0.00`
- `simulator_order_fee_usd=0.00`
- `total_simulator_order_fees_usd=0.00`
- `modeled_spread_bps=2.00`
- `modeled_slippage_bps=3.00`
- `modeled_bps_per_turnover=5.00`
- `modeled_transaction_costs_usd=0.00`
- `modeled_transaction_costs_pct_nav=0.000000`
- `cost_model_used=no_orders`
- `trade_now_edge_after_modeled_costs=-0.001908`
- `trade_now_edge_without_direct_simulator_fees=-0.001908`

Interpretation:
- Das offene Handelsfenster war diesmal **nicht** der Blocker.
- Der Blocker ist jetzt fachlich die fehlende Rebalance-Edge:
  - `best_discrete_candidate=HOLD_CURRENT`
  - `trade_now_edge=-0.001908`
  - keine diskrete BUY/SELL-Liste ueberwindet die Huerde nach Kosten und Buffern

Manual-Simulator-Datei:
- `outputs/manual_simulator_orders.csv` bleibt fachlich korrekt.
- Sie ist aktuell leer, weil es gegen das echte aktuelle Portfolio keine freigegebenen Delta-Orders gibt.

Verifizierte Checks dieser Runde:
- `./.venv/bin/python -m py_compile *.py` -> PASS
- `./.venv/bin/python health_check.py --quick` -> PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh` -> PASS
- `./.venv/bin/python robustness_tests.py` -> PASS

Bekannte offene Grenzen:
- `price_basis=adjusted_close_proxy` bleibt ein konservativer Close-/Proxy-Preis und kein Live-Bid/Ask-Ausfuehrungspreis.
- `cost_model_used=no_orders` ist korrekt fuer den aktuellen HOLD-Lauf, demonstriert aber keinen nichtleeren Orderfall.
- `scenario_model.py`, Local-Paper-State und `paper_broker_stub.py` bleiben die groessten offenen Systemthemen.

## Update 2026-04-30 - Frischer run_daily_dry_run.sh-Lauf analysiert

Geaenderte Dateien:
- `transaction_costs.py`
- `daily_bot.py`
- `robustness_tests.py`
- `outputs/today_decision_summary.txt`
- `outputs/daily_bot_decision_report.txt`
- `outputs/final_acceptance_report.txt`

Kernbefunde aus dem letzten One-Command-Lauf:
- `current_time_berlin=13:20`
- `is_project_trading_day=true`
- `within_allowed_window=false`
- `execution_allowed_by_calendar=false`
- `calendar_reason=outside_allowed_window`
- `data_source=cache_fallback`
- `used_cache_fallback=true`
- `continuous_candidate=HOLD`
- `final_discrete_candidate=HOLD_CURRENT`
- `final_action=WAIT_OUTSIDE_WINDOW`
- `execution_mode=blocked`

Erster Blocker:
- `calendar:outside_allowed_window`

Sekundaerer fachlicher Blocker:
- Selbst ohne Kalenderblock waere aktuell kein Rebalance freigegeben, weil:
  - `best_discrete_candidate=HOLD_CURRENT`
  - `trade_now_edge_after_modeled_costs=-0.001918`
  - keine diskrete Delta-Order die Huerde gegen HOLD nach Kosten und Buffern ueberwindet

Kostenmodus jetzt klar getrennt:
- `commission_per_trade_usd=0.00`
- `simulator_order_fee_usd=0.00`
- `total_simulator_order_fees_usd=0.00`
- `modeled_spread_bps=2.00`
- `modeled_slippage_bps=3.00`
- `modeled_bps_per_turnover=5.00`
- `modeled_transaction_costs_usd=0.00`
- `modeled_transaction_costs_pct_nav=0.000000`
- `trade_now_edge_after_modeled_costs=-0.001918`
- `trade_now_edge_without_direct_simulator_fees=-0.001918`

Manual-Simulator-Datei:
- `outputs/manual_simulator_orders.csv` ist fachlich korrekt, aber aktuell nicht zur Eingabe bestimmt.
- Die Datei ist leer, weil gegen das echte aktuelle Portfolio derzeit keine freigegebenen BUY/SELL-Delta-Orders existieren.

Verifizierte Checks dieser Runde:
- `./.venv/bin/python -m py_compile *.py` -> PASS
- `./.venv/bin/python robustness_tests.py` -> PASS
- `./run_daily_dry_run.sh` -> PASS

Bekannte offene Grenzen:
- Dieser frische Lauf fand nicht im offenen Handelsfenster statt; der erste Blocker bleibt das Kalender-Gate.
- `used_cache_fallback=true` bleibt ein zusaetzlicher Vorsichtspunkt fuer jede spaetere nicht-nur-Preview-Ausfuehrung.
- `scenario_model.py`, Local-Paper-State und `paper_broker_stub.py` bleiben die groessten offenen Systemthemen.

## Update 2026-04-30 - Delta-Order-Logik gegen aktuelles Simulator-Portfolio verifiziert

Geaenderte Dateien:
- `discrete_portfolio_optimizer.py`
- `order_preview.py`
- `daily_bot.py`
- `robustness_tests.py`

Kernfixes:
- `best_discrete_order_preview.csv` fuehrt jetzt die geforderten Delta-/Preview-Felder konsistent:
  - `current_shares`
  - `target_shares`
  - `order_shares`
  - `action`
  - `estimated_price`
  - `estimated_order_value`
  - `cash_before_orders`
  - `cash_after_orders`
  - `preview_only`
  - `not_executable`
  - `execution_block_reason`
- `order_shares` ist jetzt im finalen Daily-Bot-Preview der absolute Delta-Wert; die Richtung bleibt separat in `action`/`side`.
- `manual_simulator_orders.csv` uebernimmt weiterhin nur echte BUY/SELL-Zeilen mit `shares > 0` und keine HOLD-Zeilen.
- `today_decision_summary.txt` verweist jetzt klar auf:
  - `outputs/manual_simulator_orders.csv` als manuelle Simulator-Datei
  - `outputs/order_preview.csv` nicht fuer manuelle Simulatororders verwenden
- `rebalance_decision_report.txt` zeigt jetzt auch `cash_before_orders`, `cash_after_orders` und den Negativ-Cash-Check.

Verifizierter aktueller Run:
- `final_discrete_candidate=HOLD_CURRENT`
- `final_action=WAIT_OUTSIDE_WINDOW`
- `execution_mode=blocked`
- `order_count=0`
- `buy_count=0`
- `sell_count=0`
- `hold_count=20`
- `cash_before_orders=3.73`
- `cash_after_orders=3.73`
- `manual_simulator_orders.csv` ist leer, weil aktuell keine Delta-Orders gegen das eingespielte Portfolio noetig sind.

Verifizierte Checks dieser Runde:
- `./.venv/bin/python -m py_compile *.py` -> PASS
- `./.venv/bin/python robustness_tests.py` -> PASS
- `./.venv/bin/python health_check.py --quick` -> PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh` -> PASS

Neue / verifizierte Tests:
- `delta_order_preview_uses_absolute_shares_and_action`
- `manual_simulator_orders_only_delta_buy_sell`

Bekannte offene Grenzen:
- Bei `DRY_RUN=true` bleibt die Datei bewusst Preview-only; es werden weiterhin keine echten Orders gesendet.
- Der aktuelle Lauf bleibt fachlich bei `HOLD_CURRENT`; deshalb kann die Delta-Logik zwar verifiziert werden, erzeugt aber derzeit keine BUY/SELL-Zeilen.
- `scenario_model.py`, Local-Paper-State und `paper_broker_stub.py` bleiben die groessten offenen Systemthemen.

## Update 2026-04-30 - Aktuelles Simulator-Portfolio als Current Portfolio eingespielt

Geaenderte Dateien:
- `data/current_portfolio.csv`
- `discrete_portfolio_optimizer.py`
- `health_check.py`
- `interface_tests.py`

Kernfixes:
- `data/current_portfolio.csv` nutzt jetzt ein share-basiertes Ist-Portfolio statt der frueheren `CASH,1.0`-Platzhalterdatei.
- Das kompatible erkannte Schema ist jetzt:
  - `ticker,shares,cash_value`
  - mit einer expliziten `CASH`-Zeile fuer `cash_usd=3.73`
- `outputs/current_portfolio_report.txt` zeigt jetzt:
  - erkannte Schemaform
  - `positions_count`
  - `invested_market_value_usd`
  - `current_portfolio_100pct_cash`
  - Gewichts-Summen mit und ohne Cash
- Der Health Check bewertet ein echtes Positionsportfolio aus CSV jetzt korrekt als PASS statt implizit 100 %-Cash zu erwarten.
- `interface_tests.py` schreibt jetzt sichtbare Fortschrittsmeldungen und liess sich in dieser Runde sauber bis zum Ende verifizieren.

Aktueller verifizierter Portfolio-Stand:
- `current_portfolio_source=csv`
- `recognized_schema=ticker,shares rows plus CASH row with cash_value`
- `cash_usd=3.73`
- `positions_count=19`
- `nav_usd=99574.63`
- `current_portfolio_100pct_cash=False`

Aktuelle Daily-Bot-Entscheidung gegen dieses Portfolio:
- `continuous_model_target_candidate=MOMENTUM_TILT_SIMPLE`
- `final_discrete_candidate=HOLD_CURRENT`
- `final_action=WAIT_OUTSIDE_WINDOW`
- `execution_mode=blocked`
- `trade_now_edge=-0.001892`
- `order_count=0`
- `manual_simulator_orders.csv` ist leer, weil aktuell keine Delta-Orders gegen das eingespielte Portfolio noetig sind.

Verifizierte Checks dieser Runde:
- `./.venv/bin/python -m py_compile *.py` -> PASS
- `./.venv/bin/python health_check.py --quick` -> PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh` -> PASS
- `./.venv/bin/python interface_tests.py` -> PASS

Bekannte offene Grenzen:
- `scenario_model.py` bleibt mit 5 Szenarien / ohne echte `covariance_3m`-Nutzung unvollstaendig.
- Local-Paper-State ist noch nicht die primaere Current-State-Quelle.
- `paper_broker_stub.py` bleibt unterhalb eines realistischen Broker-Lifecycle-Modells.
- `today_decision_summary.txt` kann fuer den menschlichen Review-Pfad noch klarer auf `manual_simulator_orders.csv` statt auf die TXT-Datei fokussiert werden.

## Update 2026-04-30 - Zentraler RunDataContext und saubere Research-/Daily-Report-Trennung

Geaenderte Dateien:
- `calendar_utils.py`
- `data.py`
- `main.py`
- `daily_bot.py`
- `report.py`
- `daily_analysis_report.py`
- `robustness_tests.py`
- `outputs/full_cleanup_audit.md`
- `outputs/redundancy_and_dead_code_report.md`
- `outputs/output_inventory.md`
- `outputs/final_acceptance_report.txt`

Kernfixes:
- Ein zentraler `RunDataContext` wird jetzt aus Preis-Attrs, Freshness und Kalenderstatus aufgebaut.
- `expected_latest_trading_day` wird jetzt zentral berechnet und in Diagnostics sowie Reports weitergereicht.
- `daily_bot.py` nutzt diesen Kontext jetzt konsistent fuer:
  - `outputs/current_data_freshness_report.txt`
  - `outputs/latest_decision_report.txt`
  - `outputs/daily_bot_decision_report.txt`
  - `outputs/execution_gate_report.csv`
  - Diagnostics-/Codex-/Daily-Analysis-Reports
- `main.py` schreibt seine Research-Kontext-Dateien nicht mehr in Daily-Bot-Zieldateien hinein:
  - neu `outputs/research_current_data_freshness_report.txt`
  - neu `outputs/research_latest_decision_report.txt`
- Damit bleiben die finalen Daily-Bot-Reports nach einem spaeteren `main.py`-Lauf erhalten und zeigen weiter die Daily-Bot-Wahrheit.

Verifizierte Checks dieser Runde:
- `./.venv/bin/python -m py_compile *.py` -> PASS
- `./.venv/bin/python robustness_tests.py` -> PASS
- `./.venv/bin/python health_check.py --quick` -> PASS
- `./.venv/bin/python main.py --skip-email` -> PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh` -> PASS

Neue / verifizierte Tests:
- `reports_use_consistent_latest_price_date`
- `daily_bot_reports_share_same_data_context`
- `main_report_marked_research_context_if_different`

Aktuell verifizierte Report-Wahrheit:
- `outputs/current_data_freshness_report.txt` -> `run_context=daily_bot_discrete_simulator`
- `outputs/latest_decision_report.txt` -> Daily-Bot-Finalreport
- `outputs/daily_bot_decision_report.txt` -> Daily-Bot-Finalreport
- `outputs/research_current_data_freshness_report.txt` -> `run_context=research_backtest`
- `outputs/research_latest_decision_report.txt` -> Research-/Backtest-Report

Bekannte offene Grenzen:
- `interface_tests.py` bleibt als Sandbox-Blocker haengend und ist weiter in `outputs/codex_blockers.md` dokumentiert.
- `scenario_model.py` bleibt mit 5 Szenarien / ohne echte `covariance_3m`-Nutzung unvollstaendig.
- Local-Paper-State ist noch nicht die primaere Current-State-Quelle.
- `paper_broker_stub.py` bleibt unterhalb eines realistischen Broker-Lifecycle-Modells.

## Update 2026-04-30 - Cleanup-/Redundanz-Audit und Preview-Kontext-Schaerfung

Geaenderte Dateien:
- `order_preview.py`
- `main.py`
- `daily_bot.py`
- `report.py`
- `scripts/compute_today_integer_allocation.py`
- `robustness_tests.py`
- `README.md`
- `outputs/full_cleanup_audit.md`
- `outputs/redundancy_and_dead_code_report.md`
- `outputs/output_inventory.md`
- `outputs/final_acceptance_report.txt`

Kernbefunde aus dem Audit:
- Der Daily-Bot-Pfad ist stabiler als ein Teil der Repo-Dokumentation; die groessten offenen Themen sind derzeit Wahrheitskonsistenz, Tickerdrift, Output-Namenskonventionen und nicht fehlgeschlagene Grundsicherheit.
- `transaction_costs.py` ist fuer den diskreten Daily-Bot-Pfad zentral, aber `robust_scorer.py` und `backtest.py` nutzen weiter Research-/Turnover-Proxys.
- `scenario_model.py` bleibt mit 5 Szenarien, ohne echte Covariance-Nutzung und ohne `random_seed`-Verwendung klar unvollstaendig.
- `daily_bot.py` nutzt Local-Paper-State noch nicht als primaere Quelle fuer `current_shares/current_cash`.

Kleine sichere Konsistenzfixes:
- `main.py` schreibt jetzt zusaetzlich `outputs/research_order_preview.csv` als kanonische Research-/Backtest-Datei.
- `outputs/order_preview.csv` bleibt nur als Legacy-Kompatibilitaetsalias bestehen.
- `daily_bot.py` annotiert finale diskrete Preview-Dateien jetzt explizit mit:
  - `preview_context=daily_bot_discrete_simulator`
  - `preview_role=final_discrete_preview`
  - `preview_note=...`
  - `not_executable_reason`
  - `executable`
- `scripts/compute_today_integer_allocation.py` ist jetzt explizit als research-only Zielbestandsrechner gekennzeichnet und schreibt nicht-ausfuehrbare Kontextspalten.

Verifizierte Checks dieser Runde:
- `./.venv/bin/python -m py_compile *.py` -> PASS
- `./.venv/bin/python robustness_tests.py` -> PASS
- `./.venv/bin/python health_check.py --quick` -> PASS
- `./.venv/bin/python main.py --skip-email` -> PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh` -> PASS
- `./.venv/bin/python interface_tests.py` -> BLOCKED/HANG in Sandbox, erneut dokumentiert

Wichtige neue/aktualisierte Artefakte:
- `outputs/full_cleanup_audit.md`
- `outputs/redundancy_and_dead_code_report.md`
- `outputs/output_inventory.md`
- `outputs/final_acceptance_report.txt`
- `outputs/research_order_preview.csv`

Bekannte offene Grenzen bleiben ehrlich sichtbar:
- `main.py` bleibt research-/backtest-orientiert.
- `daily_bot.py` bleibt der share-genaue Simulatorpfad.
- `CONDITIONAL_FACTOR_TARGET` ist kein vollstaendig unabhaengiger Faktor-Optimizer.
- Der direkte Forecast bleibt momentum-/trend-lastig.
- FRED/FMP bzw. externe Makro-Release-Timings sind nicht release-date-aware modelliert.
- `paper_broker_stub.py` bleibt funktional klar unterhalb eines realistischen Broker-Lifecycles.

## Update 2026-04-29 - Transaktionskosten- und Slippage-Audit

Geaenderte Dateien:
- `config.py`
- `config_validation.py`
- `transaction_costs.py`
- `discrete_portfolio_optimizer.py`
- `execution_gate.py`
- `pre_trade_validation.py`
- `paper_broker_stub.py`
- `daily_bot.py`
- `health_check.py`
- `robustness_tests.py`
- `transaction_cost_report.txt` Outputpfad in `daily_bot.py`

Kernfixes:
- zentrale modellierte Kostenlogik in `transaction_costs.py`
- Default-Annahmen jetzt explizit:
  - `DEFAULT_COMMISSION_PER_TRADE=0.00`
  - `DEFAULT_BPS_PER_TURNOVER=5`
  - `DEFAULT_SPREAD_BPS=2`
  - `DEFAULT_SLIPPAGE_BPS=3`
- asset-spezifische Overrides fuer:
  - Cash-/Bond-ETFs wie `SGOV`, `SHY`, `IEF`, `AGG`, `LQD`, `TIP`
  - normale breite ETFs wie `XLC`, `XLI`, `XLK`, `XLP`, `XLU`, `XLV`, `SPHQ`, `SPLV`, `SPMO`
  - hoehere Kosten fuer `PDBC`, `GLD`, `SLV`
  - inverse ETFs wie `SH`
  - Crypto-ETF `IBIT`
- `score_discrete_candidates()` re-scored diskrete Kandidaten jetzt mit Kosten auf der echten Whole-Share-Orderliste statt nur mit Prozent-Turnover
- `daily_bot.py` schreibt die finalen Kostenfelder jetzt in `best_discrete_order_preview.csv`
- `execution_gate.py` nutzt den diskreten Net-Edge nach Orderkosten und zieht Spread/Slippage nicht mehr doppelt ab
- `pre_trade_validation.py` und `paper_broker_stub.py` nutzen per-Order-Kostenfelder, wenn sie vorhanden sind
- `compute_trade_now_edge()` berechnet jetzt explizit die finale Umschichtungsedge aus:
  - aktuellem Portfolio-Score
  - Zielscore nach konkreten Orderkosten
  - Execution-Buffer
  - Model-Uncertainty-Buffer
- `rebalance_decision_report.txt` zeigt jetzt explizit:
  - `target_score_before_costs`
  - `target_score_after_costs`
  - `total_order_cost`
  - `total_order_cost_pct_nav`
  - `execution_buffer`
  - `model_uncertainty_buffer`
  - `trade_now_edge`
- `transaction_cost_report.txt` zeigt jetzt:
  - live vs modelliert
  - Kosten pro Assetgruppe
  - teuerste Order
  - hoechste Cost-BPS
  - Cash vor/nach Orders
  - Negativ-Cash-Check

Aktueller verifizierter Run:
- `data_source=yfinance`
- `synthetic_data=false`
- `data_freshness_ok=true`
- `best_discrete_candidate=CONDITIONAL_FACTOR_TARGET::GREEDY_FILL_250`
- `final_action=WAIT`
- `target_score_before_costs=0.001101`
- `target_score_after_costs=0.000692`
- `trade_now_edge=-0.000823`
- `total_estimated_transaction_cost_usd=40.89`
- `total_estimated_transaction_cost_pct_nav=0.000409`
- `weighted_average_cost_bps=4.09`
- `live_costs_available=false`
- `cost_model_used=modeled_bps_assumptions`

Ausgefuehrte Checks:
- `./.venv/bin/python -m py_compile *.py` -> PASS
- `./.venv/bin/python health_check.py --quick` -> PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh` -> PASS
- `./.venv/bin/python main.py --skip-email` -> PASS
- `./.venv/bin/python robustness_tests.py` -> PASS
- `./.venv/bin/python interface_tests.py` -> PASS
- `./.venv/bin/python -m pytest` -> WARN (`0` Tests gesammelt, kein Test-Body vorhanden)

Neue / relevante Outputs:
- `outputs/transaction_cost_audit.txt`
- `outputs/transaction_cost_report.txt`
- `outputs/best_discrete_order_preview.csv`
- `outputs/discrete_candidate_scores.csv`
- `outputs/discrete_optimization_report.txt`
- `outputs/rebalance_decision_report.txt`

Bekannte Grenzen:
- Das System nutzt aktuell modellierte Kostenannahmen, keine Live-Bid/Ask-Daten und keine echten Brokergebuehren.
- `robust_scorer.py` nutzt fuer kontinuierliche Research-Kandidaten weiter einen Turnover-Proxy; die finale Daily-Bot-Entscheidung nutzt aber die diskrete Orderliste.
- `commission_per_trade_usd` bleibt bewusst `0.00`, weil fuer den aktuellen Simulator keine Ordergebuehren angesetzt werden sollen.

Naechster sinnvoller Schritt:
- falls gewuenscht als naechstes einen kleinen Konsistenztest fuer `best_discrete_order_preview.csv` und `transaction_cost_audit.txt` in `interface_tests.py` nachziehen, damit der Kostenreport kuenftig explizit regression-getestet ist

## Update 2026-04-28 - Diskrete Stabilisierungsrunde

Geaenderte Dateien:
- `data.py`
- `health_check.py`

Kleine Fixes:
- atomische Cache-Writes in `data.py` nutzen jetzt eindeutige Tempdateien statt eines festen `.tmp`-Namens
- `load_price_cache()` verwirft unparsebare Restzeilen robust statt den ganzen Lauf zu crashen
- `health_check.py` prueft jetzt die diskrete Whole-Share-Logik explizit:
  - aktuelles Portfolio aus `data/current_portfolio.csv`
  - 100 % Cash wird erkannt
  - diskrete Kandidaten werden erzeugt
  - ganze Stueckzahlen / kein negativer Cash / kein Leverage / BUY-Orders aus Cash werden geprueft

Ausgefuehrte Checks:
- `./.venv/bin/python -m py_compile *.py` -> PASS
- `./.venv/bin/python health_check.py --quick` -> PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh` -> PASS
- `./.venv/bin/python main.py` -> PASS

Relevante Output-Dateien fuer die diskrete Stueckzahl-Logik:
- `outputs/current_portfolio_report.txt`
- `outputs/hold_vs_target_analysis.txt`
- `outputs/discrete_candidate_scores.csv`
- `outputs/best_discrete_allocation.csv`
- `outputs/best_discrete_order_preview.csv`
- `outputs/discrete_optimization_report.txt`
- `outputs/daily_bot_decision_report.txt`

Bekannte Grenzen:
- `main.py` bleibt research-/backtest-orientiert; die finale stueckgenaue Zielallokation kommt aus `daily_bot.py`
- `health_check.py --quick` nutzt absichtlich den cache-preferred Pfad, meldet aber die diskrete Whole-Share-Logik jetzt separat
- `Investopedia` bleibt disabled/stub; kein Auto-Login, kein Code-/MFA-Handling, kein Submit

Naechster sinnvoller Schritt:
- die manuelle Simulator-Orderliste aus `best_discrete_order_preview.csv` finalisieren und als klare Eingabeliste fuer den Simulator ausgeben

## Update 2026-04-28 - Live-Daten-Verifikation

Aktive Projekt-Umgebung:
- `./.venv/bin/python`

Paketstatus in der aktiven `.venv`:
- `yfinance`: verfuegbar
- `scipy`: verfuegbar
- `matplotlib`: verfuegbar
- `pandas`: verfuegbar
- `numpy`: verfuegbar
- `python-dotenv`: verfuegbar

Live-Datenstatus im letzten verifizierten Lauf:
- `data_source=yfinance`
- `cache_status=refreshed`
- `synthetic_data=false`
- `latest_price_date=2026-04-29`
- `staleness_days=0`
- `data_freshness_ok=true`
- `used_cache_fallback=false`
- `tickers_failed=none`

Verifizierte Befehle dieser Runde:
- `./.venv/bin/python health_check.py --quick` -> PASS
- `./.venv/bin/python main.py` -> PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh` -> PASS

## Update 2026-04-28 - Synthetic/Cache-Bereinigung

Ist-Zustand:
- `data/prices_cache.csv.meta.json` steht jetzt wieder korrekt auf:
  - `data_source=yfinance`
  - `cache_status=refreshed`
  - `synthetic_data=false`
  - `latest_price_date=2026-04-28`
  - `used_cache_fallback=false`
- `data/prices_cache.csv` endet sauber auf `2026-04-28`
- `outputs/current_data_freshness_report.txt` ist konsistent dazu

Bewertung:
- kein `cache_backup/` noetig, weil der vorhandene Cache in diesem Lauf nicht als `synthetic_data=true` markiert war
- die Cache-Metadaten-Erhaltung in `data.py` wurde minimal korrigiert, damit echte Live-Refreshes `data_source`, `cache_status` und `latest_price_date` nicht verlieren
- `synthetic_data=true` bleibt execution-blockierend:
  - `execution_gate.py`
  - `main.py`
  - `daily_bot.py`
  - `robustness_tests.py`

Finale Aussage:
- Die Datenbasis ist aktuell **echt und aktuell**, nicht synthetisch
- Cache-Fallback wurde im letzten verifizierten `main.py`-/`daily_bot.py`-Lauf **nicht** genutzt

## 1. Vollstaendig implementiert

- `main.py`
- `health_check.py`
- `asset_universe.py`
- `data.py`
- `optimizer.py`
- `forecast_3m.py`
- `scenario_model.py`
- `candidate_factory.py`
- `robust_scorer.py`
- `execution_gate.py`
- `factor_registry.py`
- `data_quality.py`
- `tradability.py`
- `pre_trade_validation.py`
- `audit.py`
- `database.py`
- `report.py`

## 2. Teilweise implementiert

- `daily_bot.py`
- `macro_data.py`
- `factor_data.py`
- `factor_forecast.py`
- `asset_exposure_model.py`
- `conditional_scenario_model.py`
- `reconciliation.py`
- `model_governance.py`
- `regime_engine.py`
- `trade_sizing.py`
- `explainability.py`
- `simulator_orchestrator.py`

## 3. Stub / Platzhalter

- `investopedia_adapter.py`
  - sicherer fail-closed Stub
  - keine produktive Login-/Order-Automation

## 4. Vorbereitet aber nicht aktiv

- `paper_broker_stub.py`
- `notifications.py`

## 5. Fehlt

- Keine der aktuell angefragten Kern-/Sicherheitsdateien fehlt.

## 6. Welche Befehle laufen?

```bash
python health_check.py --quick
python main.py
python daily_bot.py --dry-run --mode single
python daily_bot.py --dry-run --mode single --force-refresh
python interface_tests.py
python robustness_tests.py
```

Aktuell final verifiziert:
- `python health_check.py --quick`
- `python main.py`
- `python daily_bot.py --dry-run --mode single`
- `python daily_bot.py --dry-run --mode single --force-refresh`
- `python interface_tests.py`
- `python robustness_tests.py`
- `./.venv/bin/python health_check.py --quick`
- `./.venv/bin/python main.py`
- `./.venv/bin/python daily_bot.py --dry-run --mode single`
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`

## 7. Welche Outputs werden erzeugt?

- `outputs/latest_decision_report.txt`
- `outputs/latest_email_notification.txt`
- `outputs/order_preview.csv`
- `outputs/daily_bot_decision_report.txt`
- `outputs/candidate_scores.csv`
- `outputs/execution_gate_report.csv`
- `outputs/system_health_report.txt`
- `outputs/current_data_freshness_report.txt`
- `outputs/tradability_report.csv`
- `outputs/pre_trade_validation_report.csv`
- `outputs/reconciliation_report.csv`
- `outputs/data_quality_report.csv`
- `outputs/data_sources_report.txt`
- `outputs/model_logic_audit.txt`
- `outputs/audit_metadata.json`
- `data/optimizer.sqlite`

## 8. Welche Fallbacks / Risiken bleiben?

- Tatsaechlich genutzte Datenquellen:
  - Live-`yfinance` vor jeder Entscheidung, wenn verfuegbar
  - sonst `data/prices_cache.csv` als Fallback
  - daraus abgeleitete Proxy-/Faktorzeitreihen
  - SQLite
- Bereinigter Real-Data-Cache:
  - in der Projekt-`.venv` wurden im letzten Verifikationslauf frische Yahoo-Daten erfolgreich geladen
  - `main.py` zog Live-Daten fuer 30/30 angefragte Ticker
  - `daily_bot.py --dry-run --mode single --force-refresh` zog Live-Daten fuer 34/34 angefragte Ticker
  - `data/prices_cache.csv` wurde dabei mit echten Marktdaten aktualisiert
  - `data/prices_cache.csv.meta.json` steht aktuell auf:
    - `data_source=yfinance`
    - `cache_status=refreshed`
    - `synthetic_data=false`
    - `latest_price_date=2026-04-28`
    - `used_cache_fallback=false`
  - aktueller Freshness-Status:
    - `data_freshness_ok=true`
    - `staleness_days=0`
  - finaler Abnahmelauf im aktuellen `python` am 2026-04-28:
    - `python health_check.py --quick`: PASS
    - `python main.py`: PASS
    - `python daily_bot.py --dry-run --mode single --force-refresh`: PASS
    - `python interface_tests.py`: PASS
    - `python robustness_tests.py`: PASS
    - dabei wurde wegen fehlendem `yfinance` im aktiven `python` sauber `cache_fallback` genutzt
    - `outputs/current_data_freshness_report.txt` steht nach diesem Lauf auf:
      - `data_source=cache_fallback`
      - `cache_status=used_after_live_failure`
      - `synthetic_data=false`
      - `latest_price_date=2026-04-28`
      - `data_freshness_ok=true`
      - `used_cache_fallback=true`
      - `live_data_error=yfinance is not installed in the current Python environment.`
    - der Cache blieb dabei:
      - `synthetic_data=false`
      - `latest_price_date=2026-04-28`
      - `data_freshness_ok=true`
    - `main.py` blieb bei `Decision=HOLD`
    - `daily_bot.py` waehlte zuletzt `OPTIMIZER_TARGET::ROUND_NEAREST_REPAIR_0`, blieb aber wegen Execution-Hurdle bei `Action=WAIT`
  - Cache-Fallback-Dokumentation:
    - bei Live-Fehler werden jetzt in den relevanten Reports explizit mitgeschrieben:
      - `data_source`
      - `cache_status`
      - `synthetic_data`
      - `latest_price_date`
      - `staleness_days`
      - `data_freshness_ok`
      - `used_cache_fallback`
      - `live_data_error`
- `synthetic_data=true` wird nicht als echte Datenbasis behandelt:
    - `execution_gate.py` blockiert dann auf `PAUSE`
    - `main.py` und `daily_bot.py` blockieren jede Execution-Schicht
    - der Cache darf dann nur fuer technische Tests/Reports dienen
- Look-ahead-Sicherheit:
  - Backtest-Kern und Daily-Bot-Snapshot sind aktuell forward-looking / weitgehend look-ahead-safe
  - externer Makro-Publikations-Timing-Layer ist noch nicht modelliert
  - Re-Audit 2026-04-28:
    - kein neuer Look-ahead-Bias nach den letzten Live-Daten-, Current-Portfolio- und diskreten Share-Optimierungs-Aenderungen gefunden
    - historische Daten bleiben Input fuer heutige Prognosen, nicht Ziel der Optimierung
    - Hilfsfunktionen mit Live-Defaults auf `prices.index[-1]` / `factor_data.index[-1]` bleiben nur dann sicher, wenn historische Caller weiter explizit `date` oder `as_of` uebergeben
  - Re-Audit 2026-04-30:
    - in `factor_data.py` wurde ein echter Zukunftsdaten-Leak gefunden und minimal-invasiv behoben
    - Ursache: volle Stichproben-Quantile in der Winsorisierung konnten historische Faktorwerte von spaeteren Beobachtungen beeinflussen
    - Fix: kausale trailing Rolling-/Expanding-Winsorisierung statt globaler Quantile
    - Backtest-Kern, Daily-Bot-Snapshot und Faktor-Regressionen bleiben danach kausal
    - Restrisiken bleiben bei nicht release-date-aware externen Makro-Feeds und bei Helper-Defaults, wenn historische Caller `date` / `as_of` nicht explizit setzen
- Aktuelles Python-Umfeld:
  - in der Projekt-`.venv` sind `yfinance`, `scipy` und `matplotlib` verfuegbar
  - normale Sandbox-/Netzwerk-Laeufe koennen beim Live-Refresh weiterhin auf Cache zurueckfallen
  - der zuletzt verifizierte Projektlauf hat jedoch echten Live-Refresh erfolgreich genutzt
- Investopedia bleibt Stub
- FRED/FMP sind nur vorbereitet
- Conditional-Factor-Layer ist proxy-basiert
- Leverage/Margin ist nicht implementiert
- Modelllogik passt grob zur Zielidee:
  - 3M-Forward-Forecasts, Szenarien, HOLD/CASH-Vergleich, robustes Candidate-Scoring und Execution Gate sind vorhanden
  - direkter Alpha-Pfad ist aber noch deutlich momentum-/trend-lastig
  - `CONDITIONAL_FACTOR_TARGET` ist aktuell kein klar getrennt optimierter Faktor-Zielkandidat
- Local Paper und E-Mail sind standardmaessig deaktiviert
- Trading-Logic-Audit 2026-04-28:
  - current weights source im verifizierten Daily-Run: `data/current_portfolio.csv` mit `CASH=100%`
  - NAV im Daily-Run: `Cash + Summe(Positionen * aktuelle Preise)`; bei explizitem `cash_value` wird NAV jetzt aus Cash plus Marktwerten abgeleitet
  - Order-Value-Logik im Daily-Bot: `order_value = (target_shares - current_shares) * latest_price`
  - gefixte Bugs:
    - Pre-Trade-Cash-Validation verarbeitet jetzt SELLs vor BUYs und schreibt Verkaufserloese dem Cash korrekt gut
    - Current-Portfolio-NAV wird bei explizitem Cash nicht mehr stillschweigend auf dem konfigurierten Startwert festgehalten
    - Diskrete Kandidaten-Validierung behandelt literales Cash nicht mehr implizit als uebergewichtetes `SGOV`
    - Daily-Bot-Pre-Trade-Validation prueft diskrete Zielportfolios jetzt mit echten investierten Gewichten plus Cash-Rest statt mit Cash-Proxy-Gewichten
  - bekannte Restrisiken:
    - `daily_bot.py` nutzt Broker-/Paper-Positionen noch nicht automatisch als primaere Quelle fuer `w_current`
    - `main.py`-Order-Preview bleibt research-/gewichtsorientiert, nicht share-genau live
    - Cash wird im Scoring ueber `SGOV` als Proxy bewertet
  - Hold-vs-target-Analyse 2026-04-28:
    - der kontinuierliche Modell-Sieger war zuletzt `MOMENTUM_TILT_SIMPLE`
    - nach den Fixes ist der finale diskrete Sieger kein `HOLD_CURRENT` mehr, sondern ein diskreter `CONDITIONAL_FACTOR_TARGET`/`OPTIMIZER_TARGET`-naher Kandidat
    - die finale Aktion bleibt trotzdem `WAIT`, weil der Execution-Hurdle nach Kosten, Spread, Slippage und Unsicherheits-Puffern negativ bleibt
  - Model-Logic-Re-Audit 2026-04-28:
    - aktueller validierter Run nutzt echte Live-Daten mit `data_source=yfinance`, `synthetic_data=false`, `latest_price_date=2026-04-28`
    - die Modelllogik bleibt grundsaetzlich zielkonform:
      - 3M-Forward-Forecast
      - Szenarien
      - HOLD/CASH-Vergleich
      - robuste Kandidatenbewertung
      - diskrete, ausfuehrbare Portfolioauswahl
      - Execution Gate
    - kleiner Fix:
      - `daily_bot.py` loggt am Ende jetzt getrennt `continuous_candidate` und `final_candidate`, damit der kontinuierliche Sieger nicht mit dem finalen diskreten Kandidaten verwechselt wird
    - wichtigster verbleibender konzeptioneller Gap:
      - der direkte Forecast-Pfad ist weiter merklich momentum-/trend-lastig
      - `CONDITIONAL_FACTOR_TARGET` ist noch kein wirklich separat optimierter Faktor-Zielkandidat

## 9. Naechster konkreter Schritt

```bash
cd ~/robust_3m_active_allocation_optimizer
source .venv/bin/activate
python health_check.py --quick
python main.py
python daily_bot.py --dry-run --mode single --force-refresh
```

## 10. Priorisierte TODOs

1. `CONDITIONAL_FACTOR_TARGET` als wirklich eigenes faktorgetriebenes Zielportfolio bauen.
   - Nutzen: hoch
   - Risiko: mittel
   - Aufwand: mittel
   - Timing: spaeter
2. Direkten Alpha-/Forecast-Pfad weniger momentum-/trend-lastig machen.
   - Nutzen: hoch
   - Risiko: mittel
   - Aufwand: mittel
   - Timing: spaeter
3. Faktorlayer ueber reine Preis-Proxys hinaus realistischer machen.
   - Nutzen: hoch
   - Risiko: mittel
   - Aufwand: mittel bis hoch
   - Timing: spaeter
4. Reports staerker auf "warum schlug dieser Kandidat HOLD/CASH?" ausrichten.
   - Nutzen: hoch
   - Risiko: niedrig
   - Aufwand: gering
   - Timing: jetzt teilweise umgesetzt
5. Synthetic-/Cache-Laeufe in den Decision Reports noch klarer markieren.
   - Nutzen: hoch
   - Risiko: niedrig
   - Aufwand: gering
   - Timing: jetzt umgesetzt
6. Utility-/Robust-Score-Logik im User-Report expliziter benennen.
   - Nutzen: hoch
   - Risiko: niedrig
   - Aufwand: gering
   - Timing: jetzt umgesetzt
7. Regime-Einfluss staerker in die Candidate-Konstruktion tragen.
   - Nutzen: mittel
   - Risiko: niedrig bis mittel
   - Aufwand: mittel
   - Timing: spaeter
8. Szenario-Attribution pro Kandidat ausbauen.
   - Nutzen: mittel
   - Risiko: niedrig
   - Aufwand: mittel
   - Timing: spaeter
9. Continuous-Mode und Turnover-Budget im Produktionspfad haerter verifizieren.
   - Nutzen: mittel
   - Risiko: niedrig
   - Aufwand: mittel
   - Timing: spaeter
10. Externe Makro-Releases nur mit sauber modellierten Publikationslatenzen einbinden.
   - Nutzen: hoch
   - Risiko: hoch
   - Aufwand: hoch
   - Timing: deutlich spaeter

Details pro Datei stehen in:
- `outputs/project_status_report.txt`
- `outputs/final_system_audit_report.txt`
- `outputs/final_acceptance_report.txt`
- `outputs/investopedia_interface_audit.txt`
- `outputs/data_sources_report.txt`
- `outputs/lookahead_bias_report.txt`
- `outputs/model_logic_audit.txt`

## 11. Edge-Case- und Betriebsrisiko-Check 2026-04-28

Geaenderte Dateien in dieser Runde:
- `calendar_utils.py`
- `data/trading_calendar_2026.csv`
- `data.py`
- `features.py`
- `daily_bot.py`
- `main.py`
- `report.py`
- `order_preview.py`
- `discrete_portfolio_optimizer.py`
- `health_check.py`
- `README.md`
- `outputs/edge_case_audit_report.txt`

Kleine klare Fixes:
- Lokale Projekt-Handelstage mit Berlin-Zeitfenster eingefuehrt
- `daily_bot.py` blockiert optionale Execution jetzt auch ueber das lokale Kalender-Gate
- `health_check.py --quick` prueft Kalenderdatei und Trading-Window-Smoke-Cases
- `current_data_freshness_report.txt`, `latest_decision_report.txt` und `daily_bot_decision_report.txt` enthalten jetzt auch Kalenderstatus und `price_basis`
- Wichtige Dateien werden jetzt atomischer geschrieben:
  - Preis-Cache und Cache-Metadaten
  - Daily-Bot-State
  - System-Health-Reports
  - zentrale CSV-/TXT-Reports
  - Order-Preview-Ausgaben
- `daily_bot.py` hat jetzt einen Lockfile-Schutz mit `data/daily_bot.lock`
- Kurze ETF-Historien schneiden nicht mehr automatisch den gesamten Return-Panel unnoetig zusammen
- Order Preview markiert Micro-Orders jetzt als `too_small_or_no_change`
- Order Preview dokumentiert jetzt explizit `adjusted_close_proxy`

Checks in dieser Runde:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python calendar_utils.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS
- `./.venv/bin/python main.py`: PASS

Aktueller Kalenderstatus:
- Kalenderdatei erzeugt: ja
- Projektzeitraum: `2026-04-24` bis `2026-07-24`
- Handelstage im Zeitraum: `63`
- Aktueller Market-Gate-Status im letzten verifizierten Lauf:
  - `current_date_berlin=2026-04-28`
  - `current_time_berlin=16:57`
  - `is_project_trading_day=true`
  - `within_allowed_window=true`
  - `execution_allowed_by_calendar=true`

Offene, nur dokumentierte Restrisiken:
- Inverse-ETF- und Crypto-Weekend-Risiken sind dokumentiert, aber noch nicht mit eigener Explainability-Warnzeile hervorgehoben
- FRED/FMP bleiben optional und nicht release-date-aware
- `main.py`-Order-Preview bleibt research-/gewichtsorientiert, waehrend der Daily-Bot share-genauer ist

## 12. Trading-Logic-Audit 2026-04-28

Geaenderte Dateien in dieser Runde:
- `pre_trade_validation.py`
- `discrete_portfolio_optimizer.py`
- `order_preview.py`
- `outputs/trading_logic_audit.txt`

Gepruefte Bereiche:
- aktuelle Gewichte / Current Portfolio
- NAV- und Marktwertberechnung
- Backtest-Walk-Forward / Turnover / Kosten
- Daily-Bot-Rebalancing
- Order Preview
- Reconciliation
- Stale-Data-Schutz
- Cash/SGOV/No-Leverage
- Look-ahead-Sicherheit

Gefixte Bugs:
1. Pre-Trade-Validation hat gueltige diskrete BUY/SELL-Orders faelschlich nur wegen eines harten `delta_weight`-Schwellwerts verworfen.
   - Fix: Validation nutzt jetzt die echten actionable-order Grenzen (Mindestorderwert / nicht-null Shares) statt die starre Delta-Weight-Sperre.
2. Order Preview hat ausserhalb des Handelsfensters BUY/SELL-Zeilen zu `HOLD` umgeschrieben.
   - Fix: BUY/SELL-Richtung bleibt jetzt sichtbar, wird aber sauber als `not_executable=true` mit Blockgrund markiert.
3. `run_pre_trade_validation()` hat bestehende `not_executable`-Flags zuvor auf `False` zurueckgesetzt.
   - Fix: Vorhandene Flags bleiben erhalten und werden nur um lokal geblockte Zeilen ergaenzt.

Verifizierte Kernaussagen:
- Current weights source im Daily-Bot: `data/current_portfolio.csv` (im letzten Lauf `100% Cash`)
- NAV calculation: `Cash + Summe(Positionen * aktuelle Preise)`
- Order value calculation im Daily-Bot: `(target_shares - current_shares) * latest_price`
- Backtest ist walk-forward:
  - Forecast/Kovarianz nur bis `t`
  - Umsetzung nur auf Rendite `t -> t+1`
  - Turnover aus `w_new - w_current`
  - Kosten werden vom Periodenreturn abgezogen
- Daily-Bot nutzt aktuelle Preise vom letzten verfuegbaren Preisdatum
- `data_freshness_ok` und `synthetic_data` bleiben execution-blockierend
- Kein Leverage, keine negativen Shares, kein negativer Cash im verifizierten diskreten Pfad

Checks in dieser Runde:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS
- `./.venv/bin/python main.py`: PASS

Aktueller verifizierter Daily-Bot-Status:
- `data_source=yfinance`
- `synthetic_data=false`
- `latest_price_date=2026-04-28`
- `data_freshness_ok=true`
- `current_portfolio_source=csv`
- `actual_cash_weight=1.000000`
- `best_discrete_candidate=CONDITIONAL_FACTOR_TARGET::ROUND_NEAREST_REPAIR_0`
- `gate_action=WAIT_OUTSIDE_WINDOW`
- `pre_trade_validation=PASS`

Bekannte Grenzen:
- HOLD-Scoring nutzt fuer literal cash weiterhin den Cash-Proxy `SGOV`; NAV/Order-Preview rechnen dagegen mit echtem Cash.
- `main.py`-`order_preview.csv` bleibt research-/backtest-orientiert und ist nicht identisch mit der live-share-genauen Daily-Bot-Preview.
- In Preview-only Mode bleibt Reconciliation bewusst `SKIP`, solange kein echter Broker-/Paper-State aktiv ist.

## 13. Hold-vs-Target-Analyse 2026-04-28

Geaenderte Dateien in dieser Runde:
- `outputs/hold_vs_target_analysis.txt`

Kernaussagen:
- Im letzten `daily_bot.py --dry-run --mode single --force-refresh` gewinnt `HOLD` nicht.
- Kontinuierlicher Sieger: `MOMENTUM_TILT_SIMPLE`
- Finaler diskreter Sieger: `OPTIMIZER_TARGET::ROUND_NEAREST_REPAIR_250`
- Finale Aktion: `WAIT_OUTSIDE_WINDOW`

Warum nicht gehandelt wurde:
- Erster Blocker im echten Lauf: Projektkalender / Handelsfenster
- Zusaetzlich waere auch ohne Kalenderblock das Execution Gate negativ geblieben:
  - `trade_now_score=-0.003431`
  - Ursache: `estimated_cost + spread + slippage + execution_uncertainty_buffer + model_uncertainty_buffer`

Wichtige Befunde:
- `p_hold_min` und `p_cash_min` sind im aktuellen Lauf nicht bindend
- `dynamic_buffer` ist nicht der Hauptblocker
- Der Hauptgrund, warum `MOMENTUM_TILT_SIMPLE` nicht final gewinnt:
  - die besten diskreten Momentum-Varianten verletzen Portfolio-Constraints
- Darum gewinnt diskret `OPTIMIZER_TARGET::GREEDY_FILL_250` als beste gueltige ganze-Stueck-Variante

Checks in dieser Runde:
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python main.py`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS

## 14. Look-Ahead Re-Audit 2026-04-29

Geaenderte Dateien in dieser Runde:
- `outputs/lookahead_bias_report.txt`
- `STATUS.md`

Kernaussage:
- `look-ahead-safe: TEILWEISE JA`
- Kein neuer Look-ahead-Bias wurde nach den letzten Aenderungen an:
  - Current-Portfolio- / 100%-Cash-Logik
  - diskreter Share-Optimierung / Order-Preview
  - Cache-Fallback / Live-Daten-Refresh
  - Kalender-Gate / Execution-Blockierung
  gefunden.
- Kein Code-Fix war in diesem Re-Audit noetig.

Verifizierte Punkte:
- Backtest bleibt walk-forward:
  - Forecast an `t` nutzt nur Daten bis `t`
  - Kovarianz an `t` nutzt nur Returns bis `t`
  - Faktor-Exposures an `t` nutzen nur Daten bis `t`
  - Entscheidung an `t` wird erst auf Rendite `t -> t+1` angewandt
- Daily-Bot bleibt Snapshot-sicher:
  - `prefer_live=True`
  - `--force-refresh` laedt vor der Entscheidung
  - Cache-Fallback wird sauber dokumentiert
- Factor-Layer bleibt kausal:
  - `macro_data.py` schneidet auf `loc[:as_of]`
  - `factor_data.py` nutzt nur den uebergebenen historischen Proxy-Frame
  - `factor_forecast.py` arbeitet mit trailing `tail(...)`
  - `asset_exposure_model.py` regressiert nur auf `loc[:as_of].tail(rolling_window)`

Bekannte Restrisiken:
- Hilfsfunktionen mit Live-Defaults auf letztes Datum bleiben nur dann sicher, wenn historische Caller weiter explizit `date` / `as_of` uebergeben:
  - `forecast_3m.py`
  - `macro_data.py`
  - `factor_forecast.py`
  - `asset_exposure_model.py`
- Externe Makro-/API-Feeds sind weiter nicht release-date-aware modelliert.

Betroffene Dateien:
- `backtest.py`
- `features.py`
- `risk.py`
- `forecast_3m.py`
- `macro_data.py`
- `factor_data.py`
- `factor_forecast.py`
- `asset_exposure_model.py`
- `scenario_model.py`
- `conditional_scenario_model.py`
- `daily_bot.py`
- `main.py`

Checks in dieser Runde:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python main.py`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS

Frischer verifizierter Daily-Bot-Lauf:
- `data_source=yfinance`
- `synthetic_data=false`
- `latest_price_date=2026-04-28`
- `data_freshness_ok=true`
- `continuous_candidate=MOMENTUM_TILT_SIMPLE`
- `best_discrete_candidate=CONDITIONAL_FACTOR_TARGET::ROUND_NEAREST_REPAIR_250`
- `gate_action=WAIT_OUTSIDE_WINDOW`
- `execution_mode=blocked`

## 15. Model-Logic Re-Audit 2026-04-29

Geaenderte Dateien in dieser Runde:
- `outputs/model_logic_audit.txt`
- `STATUS.md`

Kernaussage:
- Das System folgt weiterhin der beabsichtigten Modelllogik:
  - aktuelle Daten laden
  - 3M-Forward-Forecast bauen
  - Szenarien erzeugen
  - mehrere Kandidaten bewerten
  - HOLD und DEFENSIVE_CASH vergleichen
  - diskrete kaufbare Portfolios erneut scoren
  - danach erst Gate / Validation / Execution-Blockierung anwenden

Verifizierte Punkte:
- kein rueckblickender historischer Sharpe-Maximierer
- aktuelle Daten vor Entscheidung
- 3M-Forward-Forecast aktiv
- Szenario-Logik aktiv
- mehrere Kandidatenportfolios aktiv
- `delta_vs_hold` / `delta_vs_cash` aktiv
- `probability_beats_hold` / `probability_beats_cash` aktiv
- CVaR / Tail Risk aktiv
- Kosten / Turnover / Spread / Slippage / Buffer aktiv
- Execution Gate aktiv
- Data Quality / Tradability / Pre-Trade Validation aktiv
- `direct_only`-Fallback fuer Factor-Layer vorhanden
- finale Entscheidung bleibt erklaerbar berichtbar

Wichtige Teillimits:
- `CONDITIONAL_FACTOR_TARGET` ist noch kein wirklich separater Faktor-Optimierungspfad; er wird weiterhin aus dem gleichen `w_target`-Pfad abgeleitet.
- Die direkte Alpha-Schicht in `forecast_3m.py` bleibt momentum-/trend-lastig.

Kein Fix noetig:
- In diesem Re-Audit wurde kein unmittelbarer Logikbug gefunden, der eine Codeaenderung erforderte.

## 16. Finaler End-to-End-Abnahmelauf 2026-04-29

Geaenderte Dateien in dieser Runde:
- `outputs/final_acceptance_report.txt`
- `STATUS.md`

Ausgefuehrte Checks:
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python main.py`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS
- `./.venv/bin/python interface_tests.py`: PASS
- `./.venv/bin/python robustness_tests.py`: PASS

Verifizierter Abnahmestand:
- `data_source=yfinance`
- `synthetic_data=false`
- `latest_price_date=2026-04-29`
- `data_freshness_ok=true`
- `active_tickers_count=29`
- `removed_tickers=none`
- `selected_candidate=CONDITIONAL_FACTOR_TARGET::ROUND_NEAREST_REPAIR_0`
- `final_action=WAIT_OUTSIDE_WINDOW`
- `execution_gate_status=BLOCK`
- `pre_trade_validation_status=PASS`
- `order_preview_path=outputs/best_discrete_order_preview.csv`
- `investopedia_status=disabled/safe_stub`

Bekannte Warnungen:
- `health_check.py --quick` ist cache-preferred und hat im Quick-Pfad weiter einen sicheren temp-cache-Fallback-Hinweis erzeugt.
- `gurobipy` bleibt optional und nicht installiert; SciPy-Fallback ist verifiziert funktionsfaehig.
- `.env` bleibt optional und war im verifizierten Lauf nicht vorhanden.
- Investopedia bleibt ein sicherer Stub ohne produktive Website-Automation.
- `current_data_freshness_report.txt`, `latest_decision_report.txt` und `daily_bot_decision_report.txt` koennen beim `latest_price_date` abweichen, wenn `main.py` und `daily_bot.py` unterschiedliche Ticker-Sets fuer die Frischepruefung laden; das ist im aktuellen Stand ein Scope-Unterschied, kein Cache-Fallback-Fehler.

Naechster sinnvoller Schritt:
- Wenn gewuenscht, die Reporting-Schicht spaeter angleichen, damit `main.py` und `daily_bot.py` fuer `latest_price_date` denselben Ticker-Scope verwenden.

## 17. Finaler Cache-Fallback-Check 2026-04-29

Geaenderte Dateien in dieser Runde:
- `STATUS.md`

Verifizierte Logik:
- Bei `prefer_live=True` oder `--force-refresh` wird zuerst Live-yfinance versucht.
- Wenn Live scheitert und ein brauchbarer Cache existiert:
  - `data_source=cache_fallback`
  - `cache_status=used_after_live_failure`
  - `used_cache_fallback=true`
  - `live_data_error=<urspruengliche Fehlermeldung>`
- `synthetic_data` bleibt getrennt:
  - echter Cache => `false`
  - synthetischer Cache => `true`
- `data_freshness_ok` wird geprueft.
- Stale oder synthetische Daten bleiben execution-blockierend.
- Frischer echter Cache darf den Dry-Run weiter analysieren lassen.

Verifizierte Reports:
- `outputs/current_data_freshness_report.txt`
- `outputs/latest_decision_report.txt`
- `outputs/daily_bot_decision_report.txt`
- `STATUS.md`

Aktueller verifizierter Zustand:
- `health_check.py --quick`
  - Quick-Pfad cache-preferred
  - sicherer temp-cache-Fallback-Hinweis vorhanden
- `main.py`
  - Live-yfinance erfolgreich
  - `data_source=yfinance`
  - `synthetic_data=false`
  - `used_cache_fallback=false`
- `daily_bot.py --dry-run --mode single --force-refresh`
  - Live-yfinance erfolgreich
  - `data_source=yfinance`
  - `synthetic_data=false`
  - `used_cache_fallback=false`
  - `latest_price_date=2026-04-29`
  - `data_freshness_ok=true`

Bekannte Nuance:
- `main.py` und `daily_bot.py` nutzen aktuell unterschiedliche Ticker-Sets fuer die Frischebewertung:
  - `main.py`-Pfad: `latest_price_date=2026-04-28`
  - `daily_bot.py`-Pfad: `latest_price_date=2026-04-29`
- Das ist im aktuellen Stand ein Ticker-Scope-Unterschied im Reporting, kein Hinweis auf fehlerhaften Cache-Fallback und kein Hinweis auf `synthetic_data=true`.

## 18. Finaler Edge-Case- und Betriebsrisiko-Check 2026-04-29

Geaenderte Dateien in dieser Runde:
- `data.py`
- `main.py`
- `daily_bot.py`
- `report.py`
- `README.md`
- `outputs/edge_case_audit_report.txt`
- `STATUS.md`

Ausgefuehrte Checks:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python main.py`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS

Kleine klare Fixes:
- `data.py`
  - Datenfrische-Staleness wird jetzt explizit gegen das Berlin-Datum statt gegen ein implizites lokales Interpreter-Datum berechnet.
- `main.py`
  - atomische Writes wurden auf eindeutige Temp-Dateien umgestellt
  - `order_preview.csv` wird nicht mehr vorzeitig als Roh-Preview geschrieben, sondern erst als validierte/finale Preview persistiert
- `daily_bot.py`
  - atomische Writes fuer Decision-Report, State und CSV-Outputs wurden auf eindeutige Temp-Dateien umgestellt
- `report.py`
  - Report-CSV/TXT-Writes nutzen jetzt eindeutige atomische Temp-Dateien
- `README.md`
  - Cron-/Betriebshinweis ergaenzt: Konsole bzw. Operator-Reports nach Laeufen pruefen

Gepruefte Edge Cases:
- timezone-aware Trading-Gate in `Europe/Berlin`
- Verhalten ausserhalb des Projektzeitraums
- kurze Historien neuer ETFs wie `IBIT` / `ETHA`
- adjusted-close-Proxys in Order Previews
- Cash vs. `SGOV`
- inverse ETFs `SH` / `PSQ`
- Crypto-ETF-Weekend-/Gap-Risk
- Whole-share-Logik / Mindestorderwert / keine Miniorders
- Cache-/Stale-/Synthetic-Blocking
- atomische Writes fuer wichtige Runtime-Outputs
- Lockfile gegen parallele `daily_bot`-Laeufe
- Secrets / `.gitignore` / `.env.example`
- Cron-/Server-Readiness

Aktueller verifizierter Betriebszustand:
- `data_source=yfinance`
- `synthetic_data=false`
- `data_freshness_ok=true`
- `daily_bot`-`latest_price_date=2026-04-29`
- `current_date_berlin=2026-04-29`
- `current_time_berlin=10:12`
- `is_project_trading_day=true`
- `within_allowed_window=false`
- `execution_allowed_by_calendar=false`
- `calendar_reason=outside_allowed_window`
- `daily_bot.lock` wurde nach dem letzten erfolgreichen Lauf wieder sauber entfernt

Nur dokumentierte Restrisiken:
- `main.py` bleibt research-/backtest-orientiert und ist nicht identisch mit der share-genauen Daily-Bot-Logik.
- Fuer nicht-null Crypto-Gewichte gibt es noch keine eigene Weekend-/Gap-Warnzeile im Explainability-Report.
- Fuer gleichzeitige Long-Equity- und Inverse-Hedge-Exposures gibt es noch keine eigene Explainability-Warnzeile.
- `latest_price_date` kann zwischen `main.py` und `daily_bot.py` wegen unterschiedlicher Ticker-Sets weiter abweichen; das ist dokumentierter Scope-Unterschied, kein Cache-Fehler.

Relevante Outputs:
- `outputs/edge_case_audit_report.txt`
- `outputs/current_data_freshness_report.txt`
- `outputs/latest_decision_report.txt`
- `outputs/daily_bot_decision_report.txt`
- `outputs/best_discrete_order_preview.csv`

Naechster sinnvoller Schritt:
- Wenn gewuenscht, spaeter die Reporting-Schicht angleichen, damit `main.py` und `daily_bot.py` fuer `latest_price_date` denselben Ticker-Scope verwenden und Explainability-Warnungen fuer Crypto-/Inverse-Randfaelle ergaenzen.

## 19. Diskrete Re-Scoring-Klarstellung 2026-04-29

Geaenderte Dateien in dieser Runde:
- `discrete_portfolio_optimizer.py`
- `daily_bot.py`
- `health_check.py`
- `outputs/discrete_candidate_scores.csv`
- `outputs/best_discrete_allocation.csv`
- `outputs/best_discrete_order_preview.csv`
- `outputs/discrete_optimization_report.txt`
- `STATUS.md`

Ausgefuehrte Checks:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python main.py`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS

Minimal-invasive Fixes:
- `discrete_portfolio_optimizer.py`
  - Drift-Metriken fuer diskrete Kandidaten werden jetzt gegen den jeweiligen kontinuierlichen Quellkandidaten des Kandidaten gemessen statt immer gegen den globalen kontinuierlichen Sieger.
  - `continuous_source` wird in `discrete_candidate_scores.csv` mitgeschrieben.
- `daily_bot.py`
  - Diskrete Kandidaten speichern jetzt ihre zugehoerigen `continuous_target_weights` und den `continuous_source`.
  - `discrete_optimization_report.txt` vergleicht `closest_rounding_candidate` jetzt innerhalb derselben Quellfamilie wie der gewaehlte diskrete Kandidat.
  - Report ergaenzt um `selected_total_abs_weight_drift`, `selected_max_abs_weight_drift`, `score_difference_vs_closest_rounding` und `score_difference_vs_hold_current`.
- `health_check.py`
  - neuer Smoke-Test `discrete_small_nav_smoke` fuer `NAV=10000`, ganze Stuecke, kein negativer Cash.

Aktueller verifizierter Zustand:
- `continuous_model_optimal_candidate=MOMENTUM_TILT_SIMPLE`
- `continuous_model_optimal_score=0.001690`
- `best_discrete_candidate=CONDITIONAL_FACTOR_TARGET::ROUND_NEAREST_REPAIR_0`
- `best_discrete_source_candidate=CONDITIONAL_FACTOR_TARGET`
- `best_discrete_score=0.000520`
- `cash_left=32.03`
- `selected_total_abs_weight_drift=0.000556`
- `selected_max_abs_weight_drift=0.000176`
- `score_difference_vs_closest_rounding=0.000000`
- `score_difference_vs_hold_current=0.000899`
- `order_count=6`

Wichtige Einordnung:
- Die finale Stueckzahl-Allokation bleibt score-basiert und nicht nur drift-basiert.
- Im aktuellen Lauf faellt der beste diskrete Kandidat zufaellig mit dem besten gueltigen `ROUND_NEAREST_REPAIR` derselben Quellfamilie zusammen; deshalb ist `score_difference_vs_closest_rounding=0.000000`.
- Execution bleibt weiterhin korrekt durch den Projektkalender blockiert (`WAIT_OUTSIDE_WINDOW`), waehrend Analyse und Order-Preview weiter erzeugt werden.

Relevante Outputs:
- `outputs/discrete_candidate_scores.csv`
- `outputs/best_discrete_allocation.csv`
- `outputs/best_discrete_order_preview.csv`
- `outputs/discrete_optimization_report.txt`

Bekannte Grenzen:
- Dass der aktuell beste diskrete Kandidat zugleich die naechste gueltige Rundungsvariante seiner eigenen Quellfamilie ist, ist ein Laufresultat und kein Architekturzwang.
- `main.py` bleibt weiter research-/backtest-orientiert; der share-genaue finale Pfad ist `daily_bot.py`.

## 20. Diskrete Zieltrennung und Rebalance-Report 2026-04-29

Geaenderte Dateien in dieser Runde:
- `config.py`
- `discrete_portfolio_optimizer.py`
- `daily_bot.py`
- `health_check.py`
- `STATUS.md`

Ausgefuehrte Checks:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS
- `./.venv/bin/python main.py`: PASS

Minimal-invasive Fixes:
- `config.py`
  - `CASH_BUFFER_USD` als expliziter Parameter aufgenommen und in `build_params()` verdrahtet.
- `discrete_portfolio_optimizer.py`
  - `current_portfolio_report.txt` enthaelt jetzt klarer `current_positions`, `current_weights`, `cash_usd` und `default_current_portfolio_used`.
  - diskrete Kandidaten reporten jetzt auch `number_of_positions`, `number_of_orders`, `skipped_small_orders`, `turnover_vs_current` und `estimated_transaction_cost`.
  - `select_best_discrete_portfolio()` nutzt jetzt bei gueltigen Kandidaten einen klareren Tie-Break auf Score, Tail Risk, Turnover, Drift, Positionszahl und Cash.
- `daily_bot.py`
  - erzeugt jetzt `outputs/continuous_model_target_weights.csv`
  - erzeugt jetzt `outputs/rebalance_decision_report.txt`
  - erzeugt jetzt `outputs/optimizer_price_usage_audit.txt`
  - `best_discrete_allocation.csv` enthaelt jetzt auch `continuous_target_weight` und `abs_weight_drift`
  - `discrete_optimization_report.txt` nennt jetzt zusaetzlich `CURRENT_PORTFOLIO`, `CONTINUOUS_MODEL_TARGET`, `DISCRETE_MODEL_TARGET`, `final_action`, `executable` und den Gate-Grund
- `health_check.py`
  - prueft jetzt zusaetzlich `hold_means_current_portfolio`
  - prueft `continuous_target_weights_sum`
  - prueft `current_portfolio_100pct_cash`
  - prueft `discrete_weights_plus_cash_sum`

Aktueller verifizierter Zustand:
- `current_portfolio_source=csv`
- `current_portfolio=100% cash`
- `continuous_model_target_candidate=MOMENTUM_TILT_SIMPLE`
- `best_discrete_candidate=CONDITIONAL_FACTOR_TARGET::ROUND_NEAREST_REPAIR_0`
- `best_discrete_score=0.000521`
- `current_portfolio_score=-0.000379`
- `delta_score=0.000900`
- `final_action=WAIT_OUTSIDE_WINDOW`
- `cash_left=32.03`
- `selected_max_abs_weight_drift=0.000176`
- `order_count=6`

Relevante Outputs:
- `outputs/current_portfolio_report.txt`
- `outputs/continuous_model_target_weights.csv`
- `outputs/discrete_candidate_scores.csv`
- `outputs/best_discrete_allocation.csv`
- `outputs/best_discrete_order_preview.csv`
- `outputs/discrete_optimization_report.txt`
- `outputs/rebalance_decision_report.txt`
- `outputs/optimizer_price_usage_audit.txt`

Bekannte Grenzen:
- Die finale Handelsaktion bleibt im aktuellen Lauf weiterhin kalenderbedingt blockiert; die diskrete Zielallokation wird trotzdem korrekt als Preview geschrieben.
- `best_discrete_candidate` und `final_action` sind bewusst getrennt: der beste kaufbare diskrete Zielkandidat ist nicht automatisch die freigegebene Handlung.

## 21. Finale Abnahmekriterien 2026-04-29

Geaenderte Dateien in dieser Runde:
- `outputs/final_acceptance_report.txt`
- `STATUS.md`

Ausgefuehrte Checks:
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS

Verifizierte Abnahmekriterien:
- `data_source=yfinance`
- `synthetic_data=false`
- `data_freshness_ok=true`
- `current_portfolio_source=csv`
- `HOLD` entspricht dem tatsaechlichen aktuellen Portfolio
- `outputs/continuous_model_target_weights.csv` vorhanden
- `outputs/best_discrete_allocation.csv` vorhanden
- `outputs/best_discrete_order_preview.csv` vorhanden
- ganze Stuecke: PASS
- kein negativer Cash: PASS
- keine Shorts: PASS
- kein Leverage: PASS
- keine echten Orders: PASS
- Investopedia bleibt disabled/stub: PASS

Aktueller verifizierter Daily-Bot-Stand:
- `continuous_model_target_candidate=MOMENTUM_TILT_SIMPLE`
- `final_discrete_candidate=CONDITIONAL_FACTOR_TARGET::ROUND_NEAREST_REPAIR_0`
- `final_action=WAIT_OUTSIDE_WINDOW`
- `execution_mode=blocked`
- `order_preview_path=outputs/best_discrete_order_preview.csv`

Relevanter Output:
- `outputs/final_acceptance_report.txt`

## 22. One-Command Dry-Run Script 2026-04-29

Geaenderte Dateien in dieser Runde:
- `run_daily_dry_run.sh`
- `STATUS.md`

Umgesetzte Aenderung:
- neues Startscript `run_daily_dry_run.sh` erstellt
- aktiviert `.venv`, falls vorhanden
- fuehrt `health_check.py --quick` aus
- fuehrt `daily_bot.py --dry-run --mode single --force-refresh` aus
- bricht bei kritischen Fehlern dank `set -euo pipefail` sofort ab
- zeigt am Ende kompakt:
  - `outputs/current_data_freshness_report.txt`
  - `outputs/current_portfolio_report.txt`
  - `outputs/discrete_optimization_report.txt`
  - `outputs/rebalance_decision_report.txt`
  - Pfad zu `outputs/best_discrete_order_preview.csv`
- setzt `MPLCONFIGDIR` lokal auf `outputs/.mplconfig`, damit kein Matplotlib-Cacheproblem ausserhalb des Projektpfads entsteht

Ausgefuehrte Checks:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./run_daily_dry_run.sh`: PASS

Verifizierter Script-Lauf:
- Dry-Run erfolgreich abgeschlossen
- keine echten Orders gesendet
- `daily_bot.py` lief weiter fail-closed und endete mit `final_action=WAIT_OUTSIDE_WINDOW`
- im verifizierten Lauf wurde ein sauber dokumentierter Cache-Fallback genutzt:
  - `data_source=cache_fallback`
  - `used_cache_fallback=True`
  - `data_freshness_ok=True`

Relevante Datei:
- `run_daily_dry_run.sh`

## 23. Manual Simulator Order Sheet 2026-04-29

Geaenderte Dateien in dieser Runde:
- `daily_bot.py`
- `STATUS.md`

Umgesetzte Aenderung:
- `daily_bot.py` erzeugt jetzt zusaetzlich:
  - `outputs/manual_simulator_orders.csv`
  - `outputs/manual_simulator_orders.txt`
- die manuelle Einkaufsliste wird aus dem finalen diskreten Preview abgeleitet
- es werden nur echte `BUY`/`SELL`-Zeilen mit `shares > 0` uebernommen
- `HOLD`-Zeilen werden weggelassen
- beide Outputs markieren klar:
  - `Manual simulator entry only`
  - `latest_price_date`
  - `rest_cash_usd`
- `best_discrete_order_preview.csv` wird im blockierten Gate-Pfad jetzt ebenfalls mit dem finalen `execution_block_reason` nachgeschrieben, damit Manual-Sheet und finale Preview dieselbe Wahrheit zeigen

Ausgefuehrte Checks:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS
- `./.venv/bin/python main.py`: WARN

Verifizierter Zustand:
- `outputs/manual_simulator_orders.txt` enthaelt aktuell:
  - `BUY 258 XLE`
  - `BUY 5 XLRE`
  - `BUY 695 SGOV`
  - `BUY 553 PDBC`
  - `BUY 116 DBA`
  - `BUY 17 REMX`
- `outputs/manual_simulator_orders.csv` enthaelt die geforderten Spalten:
  - `ticker`
  - `action`
  - `shares`
  - `estimated_price`
  - `estimated_order_value`
  - `note`
- finale Preview bleibt korrekt `preview_only` / `not_executable`, weil der Kalender das Handelsfenster blockiert
- keine echten Orders gesendet

Offene Hinweise:
- `main.py` wurde fuer die Sicherheitsrunde erneut gestartet, hat in diesem lokalen Lauf aber innerhalb des Beobachtungsfensters keinen Abschluss geloggt; der geaenderte Codepfad betrifft nur den `daily_bot.py`-Outputpfad, und `main.py` hatte in der vorherigen Runde bereits erfolgreich abgeschlossen.

Relevante Outputs:
- `outputs/manual_simulator_orders.csv`
- `outputs/manual_simulator_orders.txt`

## 24. Human Decision Summary 2026-04-29

Geaenderte Dateien in dieser Runde:
- `daily_bot.py`
- `STATUS.md`

Umgesetzte Aenderung:
- `daily_bot.py` erzeugt jetzt zusaetzlich `outputs/today_decision_summary.txt`
- die Datei erklaert in einfacher Sprache:
  - Datenaktualitaet
  - angenommenes aktuelles Portfolio
  - kontinuierliches Modellziel
  - bestes diskretes kaufbares Portfolio
  - manuelle Simulator-Orders
  - Handelsfenster-Status
  - Preview-vs-Execution
  - Entscheidungsgrund
  - wichtigste Risiken
- die Zusammenfassung nutzt denselben finalen Daily-Bot-Kontext wie Order-Preview und Rebalance-Report

Ausgefuehrte Checks:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS
- `./.venv/bin/python main.py --skip-email`: WARN

Verifizierter Zustand:
- `outputs/today_decision_summary.txt` wurde erfolgreich geschrieben
- aktuelle Kernaussage:
  - Daten aktuell: `Ja`
  - aktuelles Portfolio: `100 % Cash`
  - kontinuierliches Modellziel: `MOMENTUM_TILT_SIMPLE`
  - bestes diskretes Portfolio: `CONDITIONAL_FACTOR_TARGET::ROUND_NEAREST_REPAIR_0`
  - finale Aktion: `WAIT_OUTSIDE_WINDOW`
  - Orderliste: `nur Preview`
- keine echten Orders gesendet

Offene Hinweise:
- `main.py --skip-email` lief im lokalen Beobachtungsfenster weiter ohne Abschlussmeldung; der geaenderte Pfad betrifft nur den Daily-Bot-Reportpfad, nicht die Research-Logik.

Relevanter Output:
- `outputs/today_decision_summary.txt`

## 25. Output Naming Cleanup 2026-04-29

Geaenderte Dateien in dieser Runde:
- `report.py`
- `main.py`
- `daily_bot.py`
- `STATUS.md`

Umgesetzte Aenderung:
- neues `outputs/output_file_guide.txt` eingefuehrt
- `best_discrete_order_preview.csv` ist dort explizit als finale diskrete Daily-Bot-Preview fuer die manuelle Simulatorentscheidung dokumentiert
- `manual_simulator_orders.csv` und `manual_simulator_orders.txt` sind dort explizit als manuelle Einkaufsliste dokumentiert
- `daily_bot_decision_report.txt` verweist jetzt explizit auf:
  - `outputs/best_discrete_order_preview.csv`
  - `outputs/manual_simulator_orders.csv`
  - `outputs/manual_simulator_orders.txt`
- `main.py` markiert `outputs/order_preview.csv` jetzt im CSV selbst mit:
  - `preview_context`
  - `preview_role`
  - `preview_note`
- `latest_decision_report.txt` markiert `outputs/order_preview.csv` jetzt explizit als `research_backtest_preview`

Ausgefuehrte Checks:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS
- `./.venv/bin/python main.py --skip-email`: PASS

Verifizierter Zustand:
- `outputs/output_file_guide.txt` vorhanden
- `outputs/daily_bot_decision_report.txt` zeigt die richtige finale Daily-Bot-Datei
- `outputs/order_preview.csv` ist jetzt im Dateiinhalte selbst als Research-/Backtest-Preview markiert
- `outputs/latest_decision_report.txt` enthaelt:
  - `Research Preview File`
  - `Research Preview Context`
  - `Research Preview Note`
- keine Dateien geloescht
- keine echten Orders gesendet

Relevante Outputs:
- `outputs/output_file_guide.txt`
- `outputs/best_discrete_order_preview.csv`
- `outputs/manual_simulator_orders.csv`
- `outputs/manual_simulator_orders.txt`
- `outputs/order_preview.csv`
- `outputs/daily_bot_decision_report.txt`
- `outputs/latest_decision_report.txt`

## 26. Finaler Stabilisierungs- und Abnahmelauf 2026-04-29

Geaenderte Dateien in dieser Runde:
- `outputs/final_acceptance_report.txt`
- `STATUS.md`

Gepruefte Stabilitaet:
- diskrete Whole-Share-Optimierung ist im Daily-Bot-Pfad sauber integriert
- Begriffe bleiben sauber getrennt:
  - `CURRENT_PORTFOLIO`
  - `CONTINUOUS_MODEL_TARGET`
  - `DISCRETE_MODEL_TARGET`
  - `FINAL_ACTION`
- 100 % Cash als aktuelles Portfolio funktioniert weiterhin
- finale diskrete Preview erzeugt echte BUY-Orders aus Cash
- keine negativen Shares
- kein negativer Cash
- keine Fractional Shares
- kein Leverage
- Dry-Run bleibt Default
- keine echten Orders

Ausgefuehrte Checks:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS
- `./.venv/bin/python main.py --skip-email`: PASS
- `./run_daily_dry_run.sh`: PASS

Verifizierter Endstand:
- `data_source=yfinance`
- `cache_status=refreshed`
- `synthetic_data=false`
- `used_cache_fallback=false`
- `latest_price_date=2026-04-29`
- `current_portfolio_source=csv`
- `current_portfolio=100% cash`
- `continuous_model_target_candidate=MOMENTUM_TILT_SIMPLE`
- `final_discrete_candidate=CONDITIONAL_FACTOR_TARGET::ROUND_NEAREST_REPAIR_0`
- `final_action=WAIT_OUTSIDE_WINDOW`
- `execution_mode=blocked`
- `order_count=5`
- `cash_left=21.09`

Relevante Output-Dateien:
- `outputs/current_portfolio_report.txt`
- `outputs/continuous_model_target_weights.csv`
- `outputs/discrete_candidate_scores.csv`
- `outputs/best_discrete_allocation.csv`
- `outputs/best_discrete_order_preview.csv`
- `outputs/manual_simulator_orders.csv`
- `outputs/manual_simulator_orders.txt`
- `outputs/discrete_optimization_report.txt`
- `outputs/rebalance_decision_report.txt`
- `outputs/today_decision_summary.txt`
- `outputs/output_file_guide.txt`
- `outputs/final_acceptance_report.txt`
- `run_daily_dry_run.sh`

PASS/WARN/FAIL:
- PASS: Kernpfad Daily Bot Dry-Run
- PASS: Abnahmebericht aktualisiert
- PASS: Output-Namen klar dokumentiert
- PASS: finale manuelle Simulator-Datei ist `outputs/best_discrete_order_preview.csv`
- PASS: manuelle Einkaufsliste ist `outputs/manual_simulator_orders.csv/txt`
- WARN: aktuelle finale Aktion bleibt `WAIT_OUTSIDE_WINDOW`, weil das Handelsfenster zum Verifikationszeitpunkt geschlossen war

Bekannte Grenzen:
- `main.py` bleibt research/backtest-orientiert
- `daily_bot.py` bleibt der share-genaue Simulatorpfad
- adjusted close bleibt Preis-Proxy und kein Live-Quote
- Investopedia bleibt Stub / disabled
- keine echte Broker-API
- kein Echtgeld

Naechster sinnvoller Schritt:
- denselben Dry-Run waehrend des erlaubten Berlin-Handelsfensters erneut ausfuehren, um die gleiche finale diskrete Zielallokation einmal mit offenem Kalender-Gate gegen den Execution-Block zu pruefen

## 27. Hold-vs-Rebalance-Vermessung 2026-05-05

Geaenderte Dateien in dieser Runde:
- `daily_bot.py`
- `robustness_tests.py`
- `outputs/hold_vs_target_analysis.txt`
- `outputs/rebalance_decision_report.txt`
- `outputs/today_decision_summary.txt`
- `outputs/discrete_optimization_report.txt`
- `STATUS.md`

Was jetzt neu sauber ausgewiesen wird:
- `current_portfolio_score`
- `target_score_before_costs`
- `target_score_after_costs`
- `delta_score_vs_current`
- `total_order_cost`
- `execution_buffer`
- `model_uncertainty_buffer`
- `trade_now_edge`
- `probability_beats_current`
- `probability_beats_cash`
- `tail_risk_current`
- `tail_risk_target`
- wichtigste Treiber fuer das Zielportfolio
- wichtigste Gruende gegen sofortiges Handeln
- was sich aendern muesste, damit BUY/SELL freigegeben wird

Verifizierter aktueller Stand:
- `data_source=yfinance`
- `latest_price_date=2026-05-05`
- `current_portfolio_source=csv`
- `current_portfolio != 100% cash`
- `continuous_model_target_candidate=MOMENTUM_TILT_SIMPLE`
- `final_discrete_candidate=HOLD_CURRENT`
- `current_portfolio_score=0.000268`
- `target_score_before_costs=0.000268`
- `target_score_after_costs=0.000268`
- `delta_score_vs_current=0.000000`
- `total_order_cost=0.00 USD`
- `execution_buffer=0.001000`
- `model_uncertainty_buffer=0.000901`
- `trade_now_edge=-0.001901`
- `probability_beats_current=0.00%`
- `probability_beats_cash=100.00%`
- `tail_risk_current=0.09%`
- `tail_risk_target=0.09%`
- `final_action=HOLD`
- `execution_mode=order_preview_only`

Interpretation:
- Das kontinuierliche Modell sieht weiter Chancen in einem Momentum-/Risk-on-Ziel.
- Die kaufbare diskrete Umsetzung faellt aber auf `HOLD_CURRENT` zurueck.
- Hauptgruende:
  - negative `trade_now_edge` nach Kosten und Buffern
  - keine positive diskrete Ueberlegenheit gegenueber dem aktuellen Portfolio
  - die besten nicht-HOLD-Kandidaten verletzen weiter individuelle Asset-Max-Gewichte

Ausgefuehrte Checks:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python robustness_tests.py`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS

Neue/verbesserte Reports:
- `outputs/hold_vs_target_analysis.txt` ist jetzt eine lesbare Hold-vs-Rebalance-Vermessung mit Anhang
- `outputs/rebalance_decision_report.txt` enthaelt jetzt Wahrscheinlichkeiten, Tail-Risk sowie Zieltreiber und Blockgruende
- `outputs/discrete_optimization_report.txt` trennt jetzt sauber zwischen aktuellem Portfolio, diskretem Ziel und Freigabebedingungen
- `outputs/today_decision_summary.txt` ist jetzt als schnelle menschliche Entscheidungsdatei strukturiert

Bekannte Restrisiken:
- `probability_beats_current` bleibt bei `HOLD_CURRENT` eine strikte Outperformance-Metrik; ein identisches Portfolio schlaegt sich selbst daher nicht
- das kontinuierliche Zielportfolio bleibt momentum-/trend-lastig
- die besten diskreten Alternativen sind aktuell durch Asset-Max-Gewichte blockiert
- `scenario_model.py`, Local-Paper-State und `paper_broker_stub.py` bleiben die groessten offenen Systemthemen

## 28. Safe Test Loop und Daily Review Preview Stand 2026-05-05

Was in diesem Schritt verifiziert wurde:
- sichere Mail-Preview-Stufe bleibt aktiv
- Daily Review Outputs werden frisch geschrieben
- keine echten Orders
- kein echter Mailversand
- Research- und Daily-Bot-Pfad laufen beide weiterhin

Ausgefuehrte Schritte in Reihenfolge:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python config_validation.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python interface_tests.py`: PASS
- `./.venv/bin/python robustness_tests.py`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS
- `./.venv/bin/python main.py --skip-email`: PASS
- `./run_daily_dry_run.sh`: PASS

Daily Review / Mail Preview bereit:
- `outputs/daily_portfolio_review.txt`
- `outputs/daily_portfolio_review.csv`
- `outputs/daily_email_briefing.txt`
- `outputs/daily_email_subject.txt`
- `outputs/latest_email_notification.txt`
- `outputs/email_safety_report.txt`
- `outputs/daily_review_validation_report.txt`
- `outputs/last_email_state.json`

Aktueller verifizierter Bedienstand:
- `nav_usd=100284.73`
- `cash_usd=3.73`
- `positions_count=19`
- `current_portfolio_source=csv`
- `current_portfolio != 100% cash`
- `data_source=cache_fallback` im letzten `run_daily_dry_run.sh`-Lauf
- `latest_price_date=2026-05-05`
- `final_action=WAIT_OUTSIDE_WINDOW`
- `final_discrete_candidate=HOLD_CURRENT`
- `execution_mode=blocked`
- `first_blocker=cache_fallback used`
- `order_count=0`
- `buy_count=0`
- `sell_count=0`

Konsistenzchecks:
- NAV stimmt logisch: `cash_usd + invested_market_value_usd = nav_usd`
- `manual_simulator_orders.csv` enthaelt nur Delta-Orders; im aktuellen Lauf korrekt `0` Zeilen
- `best_discrete_order_preview.csv` bleibt Delta-Preview relativ zum aktuellen Portfolio
- `simulator_order_fee_usd=0.00`
- `total_simulator_fees_usd=0.00`
- modellierte Kosten bleiben getrennt ausgewiesen
- `first_blocker` und `all_blockers` sind in den Review-/Safety-Reports gesetzt
- Mail-Preview vorhanden, Subject nicht leer, Body nicht leer

Was eingebaut bzw. abgesichert ist:
- Daily-Portfolio-Review als taegliche Beobachtungsstufe
- strukturierte `HARD_FAIL` / `SOFT_WARNING` / `INFO`-Kategorien
- sichere Mail-Preview-Dateien ohne echten Versand
- Dedupe-Schutz ueber `outputs/last_email_state.json`
- klares Phase-Gate fuer spaeteren echten Mailversand

Gefundene und behobene Fehler in dieser Runde:
- keine neuen Codefehler
- laengere Smoke-/Research-Laufzeiten wurden bestaetigt, aber alle Prozesse endeten erfolgreich

Daily Review bereit:
- ja, preview-only

Mailversand weiterhin deaktiviert:
- ja
- echter Versand bleibt blockiert, solange nicht gleichzeitig gilt:
  - `ENABLE_EMAIL_NOTIFICATIONS=true`
  - `EMAIL_SEND_ENABLED=true`
  - `EMAIL_DRY_RUN=false`
  - `EMAIL_RECIPIENT` gesetzt
  - `USER_CONFIRMED_EMAIL_PHASE=true`
  - `PHASE=DAILY_REVIEW_SEND_READY`

PASS/WARN/FAIL:
- PASS: gesamte sichere Arbeits- und Testschleife
- PASS: Daily Review Preview und Mail-Preview-Dateien
- WARN: letzter One-Command-Lauf nutzte `cache_fallback`
- WARN: Handelsfenster war geschlossen (`WAIT_OUTSIDE_WINDOW`)
- WARN: keine freigegebenen BUY/SELL-Orders; `manual_simulator_orders.csv` bleibt korrekt leer

Offene Restrisiken:
- yfinance-/Live-Refresh kann im One-Command-Lauf weiterhin leer zurueckkommen; dann faellt der Bot konservativ auf Cache-Fallback zurueck
- adjusted close bleibt Preis-Proxy und kein Live-Quote
- `main.py` bleibt Research-/Backtest-Pfad und darf nicht mit dem Daily-Simulatorpfad verwechselt werden
- `scenario_model.py`, Local-Paper-State und `paper_broker_stub.py` bleiben die groessten offenen Systemthemen

## 29. Mail Phase Gate 2026-05-05

Der Mailversand ist jetzt zentral und explizit phasengegated.

Technische Wahrheit:
- echter Review-/Daily-Analysis-Mailversand ist nur erlaubt, wenn gleichzeitig gilt:
  - `ENABLE_EMAIL_NOTIFICATIONS=true`
  - `EMAIL_SEND_ENABLED=true`
  - `EMAIL_DRY_RUN=false`
  - `EMAIL_RECIPIENT` gesetzt
  - `USER_CONFIRMED_EMAIL_PHASE=true`
  - `PHASE=DAILY_REVIEW_SEND_READY`
  - `ENABLE_EXTERNAL_BROKER=false`
  - `ENABLE_INVESTOPEDIA_SIMULATOR=false`

Wenn eine dieser Bedingungen fehlt:
- kein echter Versand
- nur Preview
- `outputs/email_safety_report.txt` erklaert warum

Was abgesichert wurde:
- zentrales Gate in `config.py`
- `daily_portfolio_review.py` und `daily_analysis_report.py` nutzen dieselbe Gate-Logik
- `config_validation.py` prueft den Gate-Stand mit
- `ENABLE_INVESTOPEDIA_SIMULATOR=true` wird in diesem Projekt jetzt ebenfalls als unzulaessig validiert

Verifiziert:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python config_validation.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python robustness_tests.py`: PASS
- `./.venv/bin/python interface_tests.py`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS

Aktueller Gate-Status:
- `phase=DAILY_REVIEW_PREVIEW`
- `real_email_send_allowed=false`
- `email_result_reason=preview_only_phase_gate`
- `ENABLE_EXTERNAL_BROKER=false`
- `ENABLE_INVESTOPEDIA_SIMULATOR=false`

Wichtige Regel:
- niemals automatisch von Preview zu Send wechseln

## 30. Hold-vs-Rebalance Update 2026-05-06

Die Hold-vs-Rebalance-Vermessung wurde fuer das echte aktuelle Simulator-Portfolio frisch ueber den Daily-Bot-Dry-Run aktualisiert.

Ausgefuehrte Checks:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS

Aktueller verifizierter Stand:
- `current_portfolio_score=0.000274`
- `target_score_before_costs=0.000274`
- `target_score_after_costs=0.000274`
- `delta_score_vs_current=0.000000`
- `total_order_cost=0.00 USD`
- `execution_buffer=0.001000`
- `model_uncertainty_buffer=0.000901`
- `trade_now_edge=-0.001901`
- `probability_beats_current=0.00%`
- `probability_beats_cash=100.00%`
- `tail_risk_current=0.09%`
- `tail_risk_target=0.09%`
- `continuous_model_optimal_candidate=MOMENTUM_TILT_SIMPLE`
- `final_discrete_candidate=HOLD_CURRENT`
- `final_action=WAIT_OUTSIDE_WINDOW`
- `execution_mode=blocked`

Wichtigste Treiber fuer das Zielportfolio:
- kontinuierliches Modell wollte vor allem Uebergewichte in `XLK`, `SPMO`, `PDBC` und `XLU`
- Top Faktor-/Makrotreiber: `commodity`, `nominal_rates`, `value`, `usd`, `sector_rotation`
- die kaufbare diskrete Umsetzung blieb trotzdem bei `HOLD_CURRENT`

Wichtigste Gruende gegen sofortiges Handeln:
- `trade_now_edge` ist negativ und liegt unter der Execution-Huerde
- aktueller Lauf war ausserhalb des Handelsfensters
- das beste diskrete Zielportfolio ist identisch mit dem aktuellen Portfolio; keine BUY/SELL-Deltas
- die besten Nicht-HOLD-Kandidaten verletzen individuelle Asset-Max-Gewichte

Was sich fuer BUY/SELL aendern muesste:
- eine nicht-HOLD-Variante muss die Trade-Now-Huerde klar schlagen; aktuell fehlen etwa `0.004401` Net-Edge bis zur Huerde `0.002500`
- eine diskrete Alternative muss `delta_vs_current > 0` und `delta_vs_cash > 0.000500` erreichen
- die Wahrscheinlichkeiten muessen mindestens `p_current >= 55%` und `p_cash >= 52%` erreichen
- die kaufbare Whole-Share-Umsetzung muss nach Rounding und Constraints valide bleiben und darf nicht wieder auf `HOLD_CURRENT` zurueckfallen

Aktualisierte Outputs:
- `outputs/hold_vs_target_analysis.txt`
- `outputs/rebalance_decision_report.txt`
- `outputs/today_decision_summary.txt`
- `outputs/discrete_optimization_report.txt`

Bekannte Restrisiken:
- aktueller Lauf ist ein Morgen-/Vorfenster-Lauf und damit weiter `WAIT_OUTSIDE_WINDOW`
- `manual_simulator_orders.csv` bleibt korrekt leer
- die grössten offenen fachlichen Themen bleiben `scenario_model.py`, Local-Paper-State und `paper_broker_stub.py`

## 31. Daily Review Testschleife 2026-05-06

Die angeforderte sichere Arbeits- und Testschleife wurde in der vorgegebenen Reihenfolge komplett durchlaufen. Jeder Schritt wurde erst nach bestandenem Vor-Schritt fortgesetzt.

Was eingebaut bzw. verifiziert wurde:
- Daily-Review-Preview-Pfad bleibt funktionsfaehig und sicher im `DRY_RUN`-Modus
- Current-Portfolio-, NAV-, Delta-Order-, Kosten- und Mail-Preview-Logik greifen konsistent zusammen
- `outputs/daily_portfolio_review.txt`, `outputs/daily_review_validation_report.txt`, `outputs/email_safety_report.txt` und `outputs/today_decision_summary.txt` zeigen dieselbe Blocker-/Warnlogik
- `manual_simulator_orders.csv` bleibt Delta-only und aktuell korrekt leer

Bestandene Tests:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python config_validation.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python interface_tests.py`: PASS
- `./.venv/bin/python robustness_tests.py`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS
- `./.venv/bin/python main.py --skip-email`: PASS
- `./run_daily_dry_run.sh`: PASS

Pruefung der Review-Dateien:
- alle angeforderten Dateien existieren
- alle Dateien sind frisch
- keine Datei ist leer

Inhaltliche Konsistenz:
- NAV logisch korrekt: `cash_usd + invested_market_value_usd = nav_usd`
- `current_portfolio` ist nicht 100% Cash, solange Positionen vorhanden sind
- `manual_simulator_orders.csv` enthaelt nur Delta-Orders; im aktuellen Lauf `0` Zeilen
- Simulator-Gebuehren bleiben `0.00`
- modellierte Kosten sind getrennt ausgewiesen
- `first_blocker` und `all_blockers` sind gesetzt
- keine echten Orders
- keine echte Mail
- Mail-Preview vorhanden, Subject und Body nicht leer

Gefundene und behobene Fehler:
- keine Code-Fehler in dieser Testschleife
- kontrollierte Warnung im `run_daily_dry_run.sh`-Lauf: `used_cache_fallback=true`, weil Live-Refresh im Script-Lauf auf leeres yfinance-Dataset fiel
- kein Fix noetig, weil der Pfad korrekt fail-safe auf Cache-Fallback blieb

Aktueller verifizierter Stand:
- `nav_usd=100284.73`
- `cash_usd=3.73`
- `positions_count=19`
- `latest_price_date=2026-05-05`
- `final_action=WAIT_OUTSIDE_WINDOW`
- `first_blocker=outside allowed trading window`
- `order_count=0`
- `buy_count=0`
- `sell_count=0`

Restrisiken:
- der letzte One-Command-Lauf nutzte `cache_fallback`; fachlich kontrollierter WARN, aber kein FAIL
- der aktuelle Review bleibt ausserhalb des erlaubten Handelsfensters und damit `WAIT_OUTSIDE_WINDOW`
- `manual_simulator_orders.csv` ist deshalb zwar fachlich korrekt, aber aktuell nicht zur Eingabe freigegeben

Bereitschaft:
- Daily Review ist bereit
- Mail-Preview ist bereit
- echter Mailversand bleibt deaktiviert
- `ENABLE_EXTERNAL_BROKER=false`
- `ENABLE_INVESTOPEDIA_SIMULATOR=false`

## 32. Hold-vs-Rebalance Refresh 2026-05-06

Die Hold-vs-Rebalance-Vermessung wurde erneut frisch gegen das echte aktuelle Simulator-Portfolio gerechnet und die Kernreports aktualisiert.

Ausgefuehrte Checks:
- `./.venv/bin/python -m py_compile *.py`: PASS
- `./.venv/bin/python health_check.py --quick`: PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh`: PASS

Aktueller verifizierter Stand:
- `current_portfolio_score=0.000274`
- `target_score_before_costs=0.000274`
- `target_score_after_costs=0.000274`
- `delta_score_vs_current=0.000000`
- `total_order_cost=0.00 USD`
- `execution_buffer=0.001000`
- `model_uncertainty_buffer=0.000901`
- `trade_now_edge=-0.001901`
- `probability_beats_current=0.00%`
- `probability_beats_cash=100.00%`
- `tail_risk_current=0.09%`
- `tail_risk_target=0.09%`
- `continuous_model_optimal_candidate=MOMENTUM_TILT_SIMPLE`
- `final_discrete_candidate=HOLD_CURRENT`
- `final_action=WAIT_OUTSIDE_WINDOW`
- `execution_mode=blocked`

Wichtigste Treiber fuer das Zielportfolio:
- das kontinuierliche Modell wollte vor allem Uebergewichte in `XLK`, `SPMO`, `PDBC` und `XLU`
- Top Faktor-/Makrotreiber: `commodity`, `nominal_rates`, `value`, `usd`, `sector_rotation`
- die kaufbare diskrete Umsetzung blieb trotzdem bei `HOLD_CURRENT`

Wichtigste Gruende gegen sofortiges Handeln:
- `trade_now_edge` ist negativ und liegt unter der Execution-Huerde
- aktueller Lauf ist ausserhalb des Handelsfensters
- das beste diskrete Ziel ist identisch mit dem aktuellen Portfolio; daher keine BUY/SELL-Deltas
- die besten Nicht-HOLD-Kandidaten verletzen individuelle Asset-Max-Gewichte

Was sich fuer BUY/SELL aendern muesste:
- etwa `0.004401` mehr Net-Edge bis zur Huerde `0.002500`
- eine diskrete Alternative mit `delta_vs_current > 0` und `delta_vs_cash > 0.000500`
- mindestens `p_current >= 55%` und `p_cash >= 52%`
- eine Whole-Share-Umsetzung, die nach Rounding/Constraints valide bleibt und nicht wieder auf `HOLD_CURRENT` zurueckfaellt

Aktualisierte Outputs:
- `outputs/hold_vs_target_analysis.txt`
- `outputs/rebalance_decision_report.txt`
- `outputs/today_decision_summary.txt`
- `outputs/discrete_optimization_report.txt`

Bekannte Restrisiken:
- der aktuelle Lauf bleibt ein `WAIT_OUTSIDE_WINDOW`-Lauf
- `manual_simulator_orders.csv` bleibt deshalb korrekt leer
- die groessten offenen fachlichen Themen bleiben Szenario-Robustheit, Local-Paper-State und der vereinfachte Broker-Stub

## 33. Open-Window Snapshot Revalidated 2026-05-06

Der letzte verifizierte `run_daily_dry_run.sh`-Lauf **im offenen Handelsfenster** bleibt der Snapshot vom `2026-05-05 16:51 CEST`. Dieser Befund wurde erneut gegen die vorhandenen Akzeptanz-/Statusreports geprueft und in die Daily-Reports als Referenz uebernommen.

Kernbefunde aus dem letzten Open-Window-Snapshot:
- `current_time_berlin=16:51`
- `is_project_trading_day=true`
- `within_allowed_window=true`
- `execution_allowed_by_calendar=true`
- `calendar_reason=within_project_trading_window`
- `final_action=HOLD`
- `execution_mode=order_preview_only`
- erster Blocker: `execution_gate:trade_now_edge_below_hurdle`

Execution-Gate-Befund:
- Das Kalenderfenster war offen; der Kalender war also **nicht** der Blocker.
- Das Execution Gate blockierte fachlich wegen Kosten/Edge/Buffer:
  - `trade_now_edge=-0.001923`
  - `final_discrete_candidate=HOLD_CURRENT`
  - keine BUY/SELL-Delta-Order ueberwand die Huerde gegen HOLD

Manual-Simulator-Datei:
- `outputs/manual_simulator_orders.csv` war fachlich korrekt, aber nicht zur Eingabe bestimmt
- die Datei blieb leer, weil keine BUY/SELL-Delta-Order freigegeben wurde

Aktualisierte Reports dieser Revalidierung:
- `outputs/today_decision_summary.txt`
- `outputs/rebalance_decision_report.txt`
- `outputs/daily_bot_decision_report.txt`
- `outputs/final_acceptance_report.txt`
- `STATUS.md`

## 34. Daily Review Entry-Point 2026-05-06

Ziel:
- den schnellsten sicheren Tagesbetrieb fuer den Bot bereitstellen, damit einmal taeglich eine klare BUY/SELL/HOLD-Aussage erzeugt wird

Neu:
- `run_daily_email_review.sh`

Was der neue Entry-Point macht:
- `config_validation.py`
- `health_check.py --quick`
- `daily_bot.py --dry-run --mode single --force-refresh`
- zeigt danach sofort die wichtigsten Preview-Dateien
- zeigt eine kompakte Konsole-Zusammenfassung mit:
  - Subject
  - `current_time_berlin`
  - `final_action`
  - `first_blocker`
  - `data_source`
  - `used_cache_fallback`
  - `latest_price_date`
  - `order_count`
  - `buy_count`
  - `sell_count`

Wichtige Sicherheitslage:
- Preview only
- keine echten Orders
- keine echte Mail
- `outputs/manual_simulator_orders.csv` bleibt die relevante Datei fuer den Simulator
- `outputs/order_preview.csv` bleibt ausdruecklich nicht die manuelle Simulator-Datei

Verifiziert:
- `bash run_daily_email_review.sh` -> PASS

Bekannte Restrisiken:
- vor 16:00 Berlin bleibt der Bot weiterhin korrekt bei `WAIT_OUTSIDE_WINDOW`
- bei fehlendem erfolgreichem Live-Refresh kann `used_cache_fallback=true` auftreten
- Dividendencash muss weiterhin ueber `data/current_portfolio.csv` gepflegt werden, solange kein Broker-/State-Feed aktiv ist

## 35. Daily Review Mailtool 2026-05-06

Ziel:
- den bestehenden Daily-Review-Preview in einen echten Daily-Review-Mailpfad ueberfuehren, ohne Broker oder echte Orders zu aktivieren

Geaenderte Dateien:
- `notifications.py`
- `config.py`
- `daily_portfolio_review.py`
- `daily_bot.py`
- `run_daily_email_review.sh`
- `interface_tests.py`
- `.env.example`
- `README.md`

Was jetzt funktioniert:
- zentraler SMTP-/Fake-Mailversand mit strukturiertem Ergebnisobjekt
- Daily Bot sendet im Mailpfad jetzt den **Daily Review** und nicht mehr versehentlich den Research-/Analysis-Body
- `outputs/latest_email_notification.txt`, `outputs/email_safety_report.txt` und `outputs/last_email_state.json` zeigen jetzt:
  - ob gesendet wurde
  - ob der Sendversuch stattfand
  - ob Dedupe blockiert hat
  - ob Auth/SMTP fehlgeschlagen ist
- `last_email_state.json` markiert fehlgeschlagene Sends **nicht** faelschlich als erfolgreich versendet
- `run_daily_email_review.sh` zeigt jetzt Mailstatus in der Konsole an

Verifiziert:
- `./.venv/bin/python -m py_compile *.py` -> PASS
- `./.venv/bin/python config_validation.py` -> PASS
- `./.venv/bin/python health_check.py --quick` -> PASS
- `./.venv/bin/python interface_tests.py` -> PASS
- `./.venv/bin/python robustness_tests.py` -> PASS
- `./.venv/bin/python daily_bot.py --dry-run --mode single --force-refresh` -> PASS
- `bash run_daily_email_review.sh` -> PASS im Preview-Modus

Echter SMTP-Test:
- ein realer Testversuch ueber Gmail SMTP wurde gestartet
- Ergebnis: `smtp_failed`
- Fehlerklasse: `SMTPAuthenticationError`
- der Bot erreichte also den echten SMTP-Pfad, aber Gmail akzeptierte die aktuellen Zugangsdaten nicht
- wichtig: `last_email_state.json` blieb korrekt auf `last_sent_date=""` und `last_send_success=false`

Bekannte Restrisiken:
- fuer echten Versand werden gueltige Gmail-App-Credentials benoetigt
- solange die Authentifizierung fehlschlaegt, bleibt nur Preview plus sauberer Fehlerreport

## 36. Daily Mail Final Test Plan 2026-05-06

Ziel:
- den kompletten Daily-Mail-Pfad in drei Modi end-to-end abnehmen:
  - Default Preview
  - Fake Send Success inklusive Dedupe
  - Fake Send Failure inklusive Retry-Nachweis

Gefixter End-to-End-Punkt:
- `daily_portfolio_review.py` hat den vorhandenen `last_email_state.json`-Status im CLI-Regenerator vorher mit einem Default-`preview_only` ueberschrieben
- dadurch sah `run_daily_email_review.sh` bei Fake-Sends zunaechst nicht den echten Sendstatus
- behoben durch Uebernahme des letzten `email_result` aus `last_email_state.json` beim Regenerieren der Review-Dateien

Durchgefuehrte Modus-Tests:
- Modus A Default Preview:
  - `./run_daily_email_review.sh`
  - Ergebnis:
    - `email_send_attempted=false`
    - `email_send_success=false`
    - `email_result_reason=preview_only`
    - `outputs/email_final_acceptance_report.txt` endet auf `READY_FOR_DAILY_EMAIL_PREVIEW`
    - keine echte Mail
    - keine Orders
- Modus B Fake Send Success:
  - isolierter frischer `outputs/last_email_state.json`
  - `EMAIL_PROVIDER=fake`
  - `EMAIL_FAKE_SEND_SUCCESS=true`
  - Ergebnis erster Lauf:
    - `email_send_attempted=true`
    - `email_send_success=true`
    - `email_result_reason=fake_send_success`
    - `last_email_state.json` als gesendet markiert
  - Ergebnis zweiter identischer Lauf:
    - `email_send_attempted=false`
    - `email_send_success=false`
    - `email_result_reason=already_sent_today`
    - Dedupe greift korrekt
- Modus C Fake Send Failure:
  - erneut isolierter frischer `outputs/last_email_state.json`
  - `EMAIL_PROVIDER=fake`
  - `EMAIL_FAKE_SEND_SUCCESS=false`
  - Ergebnis Failure-Lauf:
    - `email_send_attempted=true`
    - `email_send_success=false`
    - `email_result_reason=fake_send_failure`
    - Fehler sichtbar
    - Script endet bei offenem Gate korrekt mit Nonzero
    - State wird nicht faelschlich als gesendet markiert
  - anschliessender Retry mit `EMAIL_FAKE_SEND_SUCCESS=true`:
    - `email_send_attempted=true`
    - `email_send_success=true`
    - Retry nach Failure funktioniert

Finale Verifikation auf Endstand:
- `./.venv/bin/python -m py_compile *.py` -> PASS
- `./.venv/bin/python config_validation.py` -> PASS
- `./.venv/bin/python health_check.py --quick` -> PASS
- `./.venv/bin/python interface_tests.py` -> PASS
- `./.venv/bin/python robustness_tests.py` -> PASS
- `./run_daily_email_review.sh` -> PASS im sicheren Preview-Endzustand

Aktueller sicherer Endzustand:
- `PHASE=DAILY_REVIEW_PREVIEW`
- `real_email_send_allowed=false`
- `outputs/email_final_acceptance_report.txt` endet auf `READY_FOR_DAILY_EMAIL_PREVIEW`
- keine echten Orders
- kein Broker
- keine Investopedia-Automation
- keine Secrets in Outputs entdeckt

Fazit:
- Preview funktioniert
- Fake Success funktioniert
- Dedupe funktioniert
- Fake Failure funktioniert
- Retry nach Failure funktioniert
- der erste echte Testversand ist technisch fast bereit; es fehlt nur noch ein bewusst freigegebener echter SMTP-Konfigurationslauf mit gueltigen Zugangsdaten

Aktuell fehlend fuer echte Testmails:
- im Repo gibt es derzeit keine `.env`
- dadurch fehlen im Arbeitsverzeichnis die nutzbaren SMTP-Felder fuer einen echten Versand:
  - `SMTP_USERNAME`
  - `SMTP_PASSWORD`
  - `EMAIL_SENDER`
  - optional/empfohlen `EMAIL_RECIPIENT`
- solange diese Daten nicht gueltig im Laufkontext verfuegbar sind, kann das Erfolgskriterium `email_result_reason=sent` fuer echte Mails nicht erreicht werden

## 37. Brevo Mail Provider Integration 2026-05-06

Eingebaut:
- `notifications.py` unterstuetzt jetzt `EMAIL_PROVIDER=brevo` als bevorzugten HTTPS/API-Mailpfad.
- `fake` bleibt fuer sichere Sendepfad-Tests erhalten.
- `smtp` bleibt als Legacy-Fallback erhalten, ist aber nicht mehr der Hauptpfad.
- `.env.example` und `docs/daily_email_operations.md` zeigen jetzt Brevo als Standard-Provider.
- `health_check.py` prueft E-Mail-Konfiguration jetzt provider-spezifisch statt SMTP-only.
- `daily_portfolio_review.py` reportet den aktiven Provider jetzt standardmaessig als `brevo` und fuehrt `error_class` in den Mail-Reports.

Sicherheitslage:
- lokales `.env` wurde bewusst auf sicheren Preview-Endzustand zurueckgestellt:
  - `PHASE=DAILY_REVIEW_PREVIEW`
  - `EMAIL_DRY_RUN=true`
  - `EMAIL_SEND_ENABLED=false`
  - `EMAIL_PROVIDER=brevo`
  - kein API-Key eingetragen
- Trading bleibt deaktiviert:
  - `DRY_RUN=true`
  - `ENABLE_EXTERNAL_BROKER=false`
  - `ENABLE_INVESTOPEDIA_SIMULATOR=false`
  - `ENABLE_LOCAL_PAPER_TRADING=false`

Neue Brevo-Regressionen:
- Brevo ohne API-Key blockiert sauber
- Brevo Dry-Run sendet nicht
- Brevo API-Success setzt `sent=true`
- Brevo API-Failure setzt `sent=false`
- Brevo Failure markiert `last_send_success` nicht faelschlich auf `true`
- Brevo Dedupe nach erfolgreichem Send funktioniert

Verifikation:
- `./.venv/bin/python -m py_compile *.py` -> PASS
- `./.venv/bin/python config_validation.py` -> PASS
- `./.venv/bin/python health_check.py --quick` -> PASS
- `./.venv/bin/python interface_tests.py` -> PASS
- `./.venv/bin/python robustness_tests.py` -> PASS
- `bash run_daily_email_review.sh` -> PASS

Aktueller Endzustand:
- `outputs/email_final_acceptance_report.txt` -> `READY_FOR_DAILY_EMAIL_PREVIEW`
- `outputs/email_safety_report.txt` zeigt:
  - `real_email_send_allowed=false`
  - `email_provider=brevo`
  - `email_result_reason=preview_only`
- keine echten Orders
- keine echten Mails
- keine Secrets in Outputs entdeckt

Was fuer echten Brevo-Versand noch fehlt:
- ein gueltiger `BREVO_API_KEY`
- eine verifizierte `EMAIL_SENDER`-Adresse bei Brevo
- eine Zieladresse in `EMAIL_RECIPIENT`
- explizites Oeffnen des bestehenden Mail-Gates fuer einen echten Send-Lauf

## 38. Brevo Gate Hardening Refresh 2026-05-06

Nachgezogen:
- `config.py` kennt jetzt `EMAIL_PROVIDER=brevo` zentral und fuehrt provider-spezifische Gate-Blocker:
  - `EMAIL_SENDER empty`
  - `BREVO_API_KEY missing`
- `config_validation.py` zeigt jetzt zusaetzlich:
  - `email_provider`
  - `email_sender_configured`
  - `brevo_api_key_configured`
- `.env.example` zeigt jetzt explizit den Brevo-Send-Ready-Satz fuer spaeteren echten Versand.
- zusaetzlicher Regressionstest:
  - `send_email_notification_brevo_missing_recipient_no_send`

Erneut verifiziert:
- `./.venv/bin/python -m py_compile *.py` -> PASS
- `./.venv/bin/python config_validation.py` -> PASS
- `./.venv/bin/python health_check.py --quick` -> PASS
- `./.venv/bin/python interface_tests.py` -> PASS
- `./.venv/bin/python robustness_tests.py` -> PASS
- `bash run_daily_email_review.sh` -> PASS

Aktuelle Brevo-Preview-Lage:
- `outputs/email_safety_report.txt` zeigt:
  - `email_provider=brevo`
  - `real_email_send_allowed=false`
  - `email_result_reason=preview_only`
- `config_validation.py` zeigt aktuell die erwarteten Preview-Blocker:
  - `ENABLE_EMAIL_NOTIFICATIONS=false`
  - `EMAIL_SEND_ENABLED=false`
  - `EMAIL_DRY_RUN=true`
  - `EMAIL_RECIPIENT empty`
  - `EMAIL_SENDER empty`
  - `BREVO_API_KEY missing`
  - `USER_CONFIRMED_EMAIL_PHASE=false`
  - `PHASE=DAILY_REVIEW_PREVIEW`

Sicherer Endzustand bleibt:
- `PHASE=DAILY_REVIEW_PREVIEW`
- `EMAIL_DRY_RUN=true`
- `EMAIL_SEND_ENABLED=false`
- `EMAIL_PROVIDER=brevo`
- keine echten Orders
- kein Broker
- keine Investopedia-Automation

## 39. Erster echter Brevo-SMTP-Testversand 2026-05-06

Durchgefuehrt:
- lokaler Testlauf mit `EMAIL_PROVIDER=smtp` ueber Brevo-SMTP
- Safety-Gate offen:
  - `ENABLE_EMAIL_NOTIFICATIONS=true`
  - `EMAIL_SEND_ENABLED=true`
  - `EMAIL_DRY_RUN=false`
  - `USER_CONFIRMED_EMAIL_PHASE=true`
  - `PHASE=DAILY_REVIEW_SEND_READY`
- Trading blieb deaktiviert:
  - `DRY_RUN=true`
  - `ENABLE_EXTERNAL_BROKER=false`
  - `ENABLE_INVESTOPEDIA_SIMULATOR=false`
  - `ENABLE_LOCAL_PAPER_TRADING=false`

Vor dem Test:
- `outputs/last_email_state.json` wurde kontrolliert auf neutralen Zustand zurueckgesetzt, damit kein Alt-Dedupe den ersten echten Versand blockiert.

Erster echter Sendelauf:
- `source .venv/bin/activate && python config_validation.py` -> PASS
- `source .venv/bin/activate && python health_check.py --quick` -> PASS
- `bash run_daily_email_review.sh` -> PASS
- Ergebnis:
  - `email_send_attempted=true`
  - `email_send_success=true`
  - `email_result_reason=sent`
  - `provider=smtp`
  - `last_send_success=true`
  - `data_source=yfinance`
  - `used_cache_fallback=false`
  - keine echten Orders
  - kein Broker
  - keine Investopedia-Automation

Nachweis-Dateien nach erstem echten Send:
- `outputs/email_safety_report.txt`
- `outputs/latest_email_notification.txt`
- `outputs/last_email_state.json`
- `outputs/email_final_acceptance_report.txt`

Zweiter Lauf:
- derselbe Scriptpfad erneut gestartet
- keine zweite Mail gesendet
- Ergebnis:
  - `email_send_attempted=false`
  - `email_send_success=false`
  - `email_result_reason=max_emails_per_day_reached`

Einordnung:
- der Schutz gegen doppelte Tagesmails hat korrekt gegriffen
- der Reason war hier `max_emails_per_day_reached` statt `already_sent_today`, weil der zweite Live-Lauf einen geaenderten Report-Hash erzeugt hat
- fachlich wurde der gewuenschte No-Second-Send-Schutz trotzdem korrekt eingehalten

Rueckstellung auf sicheren Endzustand:
- lokales `.env` wieder auf Preview gesetzt
- `source .venv/bin/activate && python config_validation.py` bestaetigt wieder:
  - `PHASE=DAILY_REVIEW_PREVIEW`
  - `real_email_send_allowed=false`
  - `EMAIL_DRY_RUN=true`

Sicherer Endzustand jetzt:
- `EMAIL_PROVIDER=brevo`
- `PHASE=DAILY_REVIEW_PREVIEW`
- `EMAIL_DRY_RUN=true`
- `EMAIL_SEND_ENABLED=false`
- keine echten Orders
- kein Broker
- keine Investopedia-Automation

## 40. Brevo-SMTP Real-Mail-Test

Brevo-SMTP Real-Mail-Test:
- erster echter Send: PASS
- Dedupe-Zweitlauf: PASS
- Preview-Endzustand wiederhergestellt: PASS
- keine Secrets in Outputs/Logs: PASS
- keine echten Orders/Broker/Investopedia: PASS
- relevante Fehler:
  - kein Versandfehler im ersten echten Sendelauf
  - Zweitlauf blockierte korrekt den zweiten Versand mit `max_emails_per_day_reached` statt `already_sent_today`

## 41. Brevo-SMTP Zustellungsdiagnose 2026-05-06

Brevo-SMTP Zustellungsdiagnose:
- echter Brevo-SMTP-Test: lokal accepted/success
- Empfaenger meldet: keine Mail erhalten
- Zustellung ist damit nicht bestaetigt
- Diagnose gestartet in `outputs/email_delivery_diagnosis_report.txt`
- keine weiteren echten Mails ohne explizite Freigabe
- erste technische Einordnung:
  - `send_success=true` bedeutet lokal nur, dass Brevo-SMTP die Nachricht ohne Exception angenommen hat
  - Inbox-Zustellung wird damit nicht bewiesen
  - lokal wird derzeit keine providerseitige Message-ID oder Delivery-Bestaetigung persistiert
