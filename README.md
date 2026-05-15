# clean_bot

## Current validated workflow

This repository contains the cleaned working copy of the robust 3M active allocation bot.

Current active daily path:

- final allocation source: `SCENARIO_WEIGHTED_RF_SHARPE_OPTIMAL`
- execution mode by default: dry-run preview only
- active scenario probabilities: dynamic regime-based probabilities in `scenario_daily_pipeline.py`
- active probability report: `outputs/active_scenario_probabilities.csv`
- scenario risk diagnostics report: `outputs/scenario_risk_probability_report.csv`
- no real broker execution by default

## Setup

Run from the project root:

    cd ~/uni_trading_bot_project/clean_bot
    python -m pip install -r requirements-dev.txt
    cp .env.example .env

Fill `.env` only with local credentials/settings. Never commit `.env`.

Dependency files:

- `requirements.txt`: runtime dependencies
- `requirements-dev.txt`: runtime dependencies plus local test tooling
- `requirements-optional.txt`: optional optimizer backends such as `gurobipy`


## Validation

Run the full local validation gate:

    ./scripts/validate_project.sh

Equivalent manual commands:

    python -m pytest tests_new -q
    python -m compileall -q .
    python smoke_test.py
    python daily_bot.py --dry-run --skip-submit

After dry-runs, runtime market-cache changes can be discarded with:

    git checkout -- data/prices_cache.csv data/prices_cache.csv.meta.json

## GitHub

Current remote workflow:

    git status --short
    git add -A
    git commit -m "Describe change"
    git push

---

# robust_3m_active_allocation_optimizer

## Kurzbeschreibung

Ein Python-Projekt fuer einen robusten aktiven 3M-Multi-Asset-Allokationsoptimizer. Das System laedt ETF-Daten, rechnet taeglich ein Zielportfolio, entscheidet zwischen `HOLD`, `WAIT`, `PARTIAL_REBALANCE`, `FULL_REBALANCE`, `DE_RISK` und `PAUSE`, erzeugt Reports, schreibt eine Order Preview und speichert die wichtigsten Ergebnisse zusaetzlich in SQLite.

Der aktuelle Stand kombiniert:

- einen stabilen Walk-forward-Backtest in `main.py`
- einen taeglichen Dry-Run-Bot in `daily_bot.py`
- eine direkte Forward-3M-Forecast-Schicht
- eine optionale Conditional-Factor-Schicht mit robustem `direct_only`-Fallback
- eine lokale Paper-Trading-Architektur
- einen experimentellen, standardmaessig deaktivierten Investopedia-Simulator-Stub

## Warnhinweis

- Keine Anlageberatung.
- Kein echtes Trading.
- Keine Broker-API.
- Keine echten Orders.
- Keine Performancegarantie.

Das Projekt ist fuer Research, Backtests, Signalgenerierung, Order Preview, Reports und optionale E-Mail-Benachrichtigungen gebaut. Es handelt standardmaessig kein echtes Geld und benoetigt keine Zugangsdaten.

## Kurzablauf

`python main.py` erledigt in einem Durchlauf:

1. Asset Registry validieren
2. Preisdaten laden oder Cache verwenden
3. Walk-forward-Backtest rechnen
4. taegliche Decision Engine anwenden
5. Reports und Charts speichern
6. aktuelle Zielallokation berechnen
7. `research_order_preview.csv` schreiben (plus `order_preview.csv` nur als Legacy-Alias)
8. `latest_email_notification.txt` schreiben und optional E-Mail senden
9. Ergebnisse nach `data/optimizer.sqlite` speichern

## Installation auf Mac

### 1. In das Projektverzeichnis wechseln

```bash
cd ~/robust_3m_active_allocation_optimizer
```

Wenn dein Projekt an einem anderen Ort liegt, verwende stattdessen den passenden Pfad.

### 2. Virtuelle Umgebung anlegen

```bash
python3 -m venv .venv
```

### 3. venv aktivieren

```bash
source .venv/bin/activate
```

### 4. Requirements installieren

```bash
pip install -r requirements.txt
```

Verwendete Kernpakete:

- `pandas`
- `numpy`
- `yfinance`
- `scipy`
- `matplotlib`
- `python-dotenv`

`gurobipy` ist optional und nicht Teil von `requirements.txt`.

## Start

```bash
python main.py
```

Optional:

```bash
python main.py --skip-email
python main.py --portfolio-value 25000
python health_check.py --quick
python health_check.py --full
python daily_bot.py --dry-run --mode single --force-refresh
python interface_tests.py
python robustness_tests.py
```

## Aktuell verifizierter Stand

Stand dieses Repos am 2026-04-26:

- `python -m py_compile *.py` laeuft
- `python health_check.py --quick` laeuft
- `python main.py` laeuft
- `python daily_bot.py --dry-run --mode single` laeuft
- `python interface_tests.py` laeuft
- `python robustness_tests.py` laeuft

Wichtig:

- Standard bleibt sicherer Dry-Run
- keine echten Broker-Orders
- keine Investopedia-Pflicht
- keine FRED-/FMP-Pflicht
- bei Live-Datenproblemen wird auf Cache und notfalls auf markierten synthetischen Fallback fuer Pipeline-Validierung gewechselt

## Cron / Betrieb

- `daily_bot.py` arbeitet standardmaessig im `DRY_RUN`.
- Fuer wiederholte Laeufe ist jetzt ein einfacher Lockfile-Schutz aktiv: `data/daily_bot.lock`.
- Starte keinen zweiten parallelen Cron-Lauf, waehrend ein vorheriger Lauf noch aktiv ist.
- Vor jeder Entscheidung werden nach Moeglichkeit Live-Daten geladen; bei Ausfall wird ein klar dokumentierter Cache-Fallback genutzt.
- Pruefe nach Cron-Laeufen die Konsole oder die wichtigsten Operator-Reports, statt still von Erfolg auszugehen.
- Pruefe fuer den Betriebsstatus besonders:
  - `outputs/current_data_freshness_report.txt`
  - `outputs/daily_bot_decision_report.txt`
  - `outputs/latest_decision_report.txt`
  - `outputs/research_current_data_freshness_report.txt` nur fuer `main.py`-Research-Laeufe
  - `outputs/research_latest_decision_report.txt` nur fuer `main.py`-Research-Laeufe
  - `outputs/system_health_report.txt`

## System Initialization and Safety Checks

Das Projekt hat jetzt eine zusaetzliche Sicherheits- und Vorpruefungsschicht vor jeder optionalen Ausfuehrung:

- `system_init.py`
  erstellt Ordner, stellt `.env.example` sicher, initialisiert SQLite und bereitet optional den lokalen Paper-Account vor
- `tradability.py`
  prueft, ob Assets nicht nur Preisdaten haben, sondern im aktuellen Modus ueberhaupt verwendbar sind
- `pre_trade_validation.py`
  prueft Zielgewichte, Preise, Order Preview sowie optional Cash und Positionen vor jeder moeglichen Ausfuehrung
- `reconciliation.py`
  gleicht Modellzustand und Broker-/Paper-Zustand ab; im Default-Modus wird das sauber als `SKIP` dokumentiert
- `health_check.py`
  fuehrt einen robusten End-to-End-Selbsttest mit `PASS`, `WARN`, `FAIL` und `SKIP` aus

Wichtig:

- Der Bot kauft nie blind ein Asset, nur weil der Optimizer es auswaehlt.
- Nicht handelbare oder nicht sauber verfuegbare Assets werden aus dem aktiven Lauf entfernt.
- Vor optionaler Ausfuehrung muessen Tradability, Preise, Gewichte, Cash/Positionen, Reconciliation und Execution Gate passen.
- Standard bleibt sicher: `DRY_RUN=true`, `ENABLE_LOCAL_PAPER_TRADING=false`, `ENABLE_INVESTOPEDIA_SIMULATOR=false`, `ENABLE_EXTERNAL_BROKER=false`.

Pruef-Workflow:

```bash
python health_check.py --quick
python main.py
python daily_bot.py --dry-run --mode single
python interface_tests.py
python robustness_tests.py
```

## Was du morgen pruefen solltest

Nach dem Lauf zuerst diese Dateien ansehen:

- `outputs/daily_results.csv`
- `outputs/weights.csv`
- `outputs/target_weights.csv`
- `outputs/research_order_preview.csv`
- `outputs/order_preview.csv` (nur Legacy-Alias)
- `outputs/research_current_data_freshness_report.txt`
- `outputs/performance_summary.csv`
- `outputs/decision_summary.csv`
- `outputs/research_latest_decision_report.txt`
- `outputs/latest_email_notification.txt`
- `data/prices_cache.csv`
- `data/optimizer.sqlite`

Wenn alles sauber lief, endet der Run in der Konsole mit:

```text
Run completed successfully.
```

## Outputs

Das Projekt erzeugt standardmaessig diese Artefakte:

- `outputs/daily_results.csv`
  Tagesweise Backtest-Ergebnisse mit `date`, `next_date`, Renditen, Entscheidung, Turnover und Risiko-Kennzahlen.
- `outputs/weights.csv`
  Tatsächlich ausgefuehrte Portfolio-Gewichte je Entscheidungstag.
- `outputs/target_weights.csv`
  Optimizer-Zielgewichte vor der finalen Decision-Engine-Ausfuehrung.
- `outputs/research_order_preview.csv`
  Kanonische Research-/Backtest-Vorschau aus `main.py`.
  Nicht identisch mit finalen Daily-Bot-Simulatororders.
- `outputs/order_preview.csv`
  Legacy-Kompatibilitaetsalias fuer `outputs/research_order_preview.csv`.
- `outputs/research_current_data_freshness_report.txt`
  Kanonischer Freshness-Report fuer `main.py`-Research-/Backtest-Laeufe.
- `outputs/research_latest_decision_report.txt`
  Kanonischer Entscheidungsreport fuer `main.py`-Research-/Backtest-Laeufe.
- `outputs/performance_summary.csv`
  Strategie plus Benchmarks mit Rendite-, Volatilitaets- und Drawdown-Kennzahlen.
- `outputs/decision_summary.csv`
  Aggregierte Statistik ueber `HOLD`, `WAIT`, `PARTIAL_REBALANCE`, `FULL_REBALANCE`, `DE_RISK`, `PAUSE`.
- `outputs/latest_email_notification.txt`
  Letzter Benachrichtigungstext, auch wenn keine E-Mail versendet wurde.
- `outputs/daily_email_subject.txt`
  Subject-Zeile der Daily-Review-Mail-Preview.
- `outputs/daily_email_briefing.txt`
  Body der Daily-Review-Mail-Preview.
- `outputs/email_safety_report.txt`
  Erklaert, warum echter Versand blockiert oder erlaubt waere.
- `outputs/last_email_state.json`
  Dedupe-/Status-Snapshot fuer den letzten Daily-Review-Maillauf.
- `outputs/current_data_freshness_report.txt`
  Kanonischer Freshness-Report fuer den finalen Daily-Bot-Lauf.
- `outputs/latest_decision_report.txt`
  Kanonischer finaler Daily-Bot-Entscheidungsreport.
- `outputs/equity_curve.png`
  Equity-Kurve der Strategie und Benchmarks.
- `outputs/drawdown_curve.png`
  Drawdown-Verlauf der Strategie und Benchmarks.
- `outputs/weights_over_time.png`
  Gewichtsentwicklung ueber die Zeit.
- `outputs/daily_bot_decision_report.txt`
  Letzte taegliche Bot-Entscheidung im Dry-Run-Format.
- `outputs/forecast_3m_diagnostics.csv`
  Direktes 3M-Forecast-Diagnostik-Set pro Asset.
- `outputs/scenario_summary.csv`
  Zusammenfassung der direkten 3M-Szenarien.
- `outputs/candidate_scores.csv`
  Kandidatenvergleich mit `HOLD`, `DEFENSIVE_CASH`, Delta- und Tail-Risk-Kennzahlen.
- `outputs/selected_candidate_weights.csv`
  Gewichte des vom Daily Bot ausgewaehlten Kandidaten.
- `outputs/daily_bot_order_preview.csv`
  Daily-Bot-Kompatibilitaetskopie der finalen diskreten Preview.
- `outputs/best_discrete_order_preview.csv`
  Kanonische finale diskrete Daily-Bot-Simulator-Preview.
- `outputs/execution_gate_report.csv`
  Execution-Gate-Entscheidung und Trade-now-Score.
- `outputs/tradability_report.csv`
  Aktive und entfernte Ticker inklusive Grund, warum ein Asset im aktuellen Lauf erlaubt oder blockiert ist.
- `outputs/pre_trade_validation_report.csv`
  Strukturierte Vorpruefung vor optionaler Order-Ausfuehrung.
- `outputs/reconciliation_report.csv`
  Modell-vs.-Broker/Paper-Abgleich oder sauber dokumentiertes `SKIP` im Preview-only-Modus.
- `outputs/system_health_report.csv`
  Tabellarischer System-Health-Check.
- `outputs/system_health_report.txt`
  Lesbare Kurzfassung des Health Checks.
- `outputs/data_quality_report.csv`
  Per-Ticker- und Global-Data-Quality-Bericht.
- `outputs/regime_report.csv`
  Regelbasierter Regime-Snapshot.
- `outputs/regime_report.txt`
  Lesbare Regime-Zusammenfassung.
- `outputs/model_governance_report.csv`
  Model-Confidence und Unsicherheitsbuffer.
- `outputs/model_governance_report.txt`
  Lesbare Governance-Zusammenfassung.
- `outputs/model_ensemble_report.csv`
  Ensemble-/Konsensmodell-Bericht.
- `outputs/model_ensemble_report.txt`
  Lesbare Ensemble-Zusammenfassung.
- `outputs/explainability_report.txt`
  Lesbarer Decision- und Faktor-Treiber-Report.
- `outputs/asset_change_explanations.csv`
  Erklaerungen zu groesseren Gewichtsveraenderungen.
- `outputs/audit_metadata.json`
  Run-Metadaten inklusive `random_seed`, `active_tickers` und `execution_mode`.
- `outputs/robustness_tests_report.csv`
  Ergebnisse der Robustheits-Smoke-Tests.
- `outputs/robustness_tests_report.txt`
  Lesbare Kurzfassung der Robustheits-Smoke-Tests.
- `outputs/factor_forecasts.csv`
  3M-Faktorprognosen, sofern die optionale Faktor-Schicht verfuegbar ist.
- `outputs/factor_data.csv`
  Abgeleitete Faktorzeitreihen aus Proxy-Daten.
- `outputs/asset_factor_exposures.csv`
  Asset-Faktor-Exposures oder Prior-Fallbacks.
- `outputs/conditional_scenario_summary.csv`
  Zusammenfassung der kombinierten Conditional-Factor-Szenarien.
- `outputs/factor_model_diagnostics.txt`
  Klarer Hinweis, ob `conditional_factor` oder `direct_only` genutzt wurde.

Chart-Erzeugung ist robust gekapselt. Wenn ein Plot fehlschlaegt, wird eine Warnung geloggt und der Run laeuft weiter.

## SQLite-Datenbank

Die Datenbank liegt in:

```text
data/optimizer.sqlite
```

Gespeichert werden unter anderem:

- `runs`
- `daily_results`
- `weights`
- `target_weights`
- `order_preview`
- `performance_summary`
- `execution_results`
- `tradability_status`
- `system_health_checks`
- `data_quality`
- `paper_account`
- `paper_positions`
- `paper_trades`
- `paper_account_history`

Die SQLite-Schicht nutzt ausschliesslich `sqlite3` aus der Python-Standardbibliothek. Wenn das DB-Schreiben fehlschlaegt, loggt `main.py` den Fehler, aber CSVs und Reports bleiben erhalten.

## E-Mail-Konfiguration

E-Mail ist optional und standardmaessig deaktiviert.

### Standardverhalten

- Ohne `.env` wird keine E-Mail gesendet.
- Mit `ENABLE_EMAIL_NOTIFICATIONS=false` wird keine E-Mail gesendet.
- Mit `EMAIL_SEND_ENABLED=false` bleibt der Daily-Review-Mailpfad preview-only.
- Mit `EMAIL_DRY_RUN=true` wird nur die Mail-Preview als Datei geschrieben.
- Ohne `EMAIL_RECIPIENT` wird kein echter Versand versucht.
- Ohne `USER_CONFIRMED_EMAIL_PHASE=true` und `PHASE=DAILY_REVIEW_SEND_READY` bleibt echter Versand blockiert.
- Das zentrale Mail-Gate liefert:
  - `reason=preview_only`, wenn der Review absichtlich nur als Preview laufen darf
  - `reason=blocked_by_gate`, wenn harte Sicherheitsblocker wie Broker/Investopedia aktiv sind
  - `reason=send_allowed`, wenn echter Versand freigegeben ist
- Bei fehlenden Provider-Daten wird keine E-Mail gesendet.
- `outputs/latest_email_notification.txt` wird trotzdem immer geschrieben.
- `outputs/daily_email_subject.txt`, `outputs/daily_email_briefing.txt`, `outputs/email_safety_report.txt` und `outputs/last_email_state.json` werden fuer den Daily Review ebenfalls geschrieben.
- Der Dedupe-State in `outputs/last_email_state.json` markiert einen Bericht nur dann als `sent`, wenn der Versand wirklich erfolgreich war.
- Mit `MAX_EMAILS_PER_DAY=1` wird nach dem ersten erfolgreichen Tagesversand jede weitere Mail am selben Tag blockiert, auch wenn sich der Bericht spaeter aendert; die Preview-Dateien werden trotzdem weiter aktualisiert.
- Bevorzugter echter Provider ist `brevo` ueber HTTPS/API; `smtp` bleibt als Legacy-Fallback erhalten.

### Einrichtung

```bash
cp .env.example .env
```

Danach in `.env` eintragen:

- `ENABLE_EMAIL_NOTIFICATIONS=true`
- `EMAIL_DRY_RUN=true`
- `EMAIL_SEND_ENABLED=false`
- `DAILY_BRIEFING_ONLY=true`
- `MAX_EMAILS_PER_DAY=1`
- `EMAIL_RECIPIENT=target_email@example.com`
- `EMAIL_SENDER=verified_sender@example.com`
- `EMAIL_PROVIDER=brevo`
- `BREVO_API_KEY=your_brevo_api_key`
- `USER_CONFIRMED_EMAIL_PHASE=false`
- `PHASE=DAILY_REVIEW_PREVIEW`
- `EMAIL_FROM=verified_sender@example.com`
- `EMAIL_SUBJECT_PREFIX=[Portfolio Optimizer]`
- `SEND_WEEKLY_SUMMARY=true`
- `SEND_DAILY_HOLD_WAIT_EMAILS=false`

Optionaler Legacy-SMTP-Fallback:

- `SMTP_HOST=smtp.gmail.com`
- `SMTP_PORT=587`
- `SMTP_USER=your_email@gmail.com`
- `SMTP_PASSWORD=your_app_password`
- `EMAIL_USE_SSL=false`
- `EMAIL_USE_STARTTLS=true`
- `EMAIL_FROM=your_email@gmail.com`
- `EMAIL_TO=target_email@example.com`

### Brevo-Hinweis

Fuer Brevo wird keine App benoetigt. Es reicht ein Brevo-Account mit verifizierter Absenderadresse und ein API-Key.

### Wann eine E-Mail verschickt wird

Nur wenn E-Mail aktiviert und der konfigurierte Provider vollstaendig eingerichtet ist, plus mindestens eine Bedingung:

- `DE_RISK`
- `FULL_REBALANCE`
- `PAUSE`
- `PARTIAL_REBALANCE` mit `emergency_condition=True`
- Wochenreport am `weekly_rebalance_day`, wenn `SEND_WEEKLY_SUMMARY=true`
- sehr starker Zusatznutzen (`net_benefit > strong_signal_threshold`)
- hoher Turnover mit positivem Nutzen
- oder explizit `SEND_DAILY_HOLD_WAIT_EMAILS=true`

Normale `HOLD`- und `WAIT`-Tage bleiben also standardmaessig stumm.
Der Daily-Review-Mailpfad bleibt ausserdem preview-only, bis `ENABLE_EMAIL_NOTIFICATIONS=true`, `EMAIL_SEND_ENABLED=true`, `EMAIL_DRY_RUN=false`, `EMAIL_RECIPIENT` gesetzt, `USER_CONFIRMED_EMAIL_PHASE=true` und `PHASE=DAILY_REVIEW_SEND_READY` zusammen aktiv sind.

### Daily Review Mailtool

- `run_daily_email_review.sh` ist der Operator-Entry-Point fuer den Daily Review.
- Detailliertes Runbook: `docs/daily_email_operations.md`
- Unterschied:
  - `run_daily_dry_run.sh` ist der allgemeine Dry-Run fuer Portfolio-, Gate- und Order-Preview-Reports.
  - `run_daily_email_review.sh` ist der klare Entry-Point fuer den taeglichen Mailbericht und die dazugehoerigen Preview-Dateien.
- Preview-Modus:
  - Im Default schreibt `run_daily_email_review.sh` nur Preview-Dateien.
  - Broker, Investopedia und lokale Paper-Ausfuehrung werden dabei explizit deaktiviert.
- Wenn das Mail-Gate offen ist, sendet der Bot die Daily-Review-Mail auf Basis von:
  - `outputs/daily_email_subject.txt`
  - `outputs/daily_email_briefing.txt`
- Send-Modus:
  - Echter Versand ist nur moeglich, wenn das zentrale Mail-Gate offen ist.
  - Ein Send-Fehler bei offenem Gate fuehrt im Mailtool zu einem Nonzero-Exit.
- Der Versand bleibt komplett getrennt von Broker-, Investopedia- und Orderausfuehrung.

Cron-Beispiel:

```bash
5 16 * * 1-5 cd "/Users/janseliger/Downloads/BF Trading /robust_3m_active_allocation_optimizer" && ./.venv/bin/python -m py_compile *.py >/dev/null 2>&1 && ./run_daily_email_review.sh >> outputs/cron_daily_email_review.log 2>&1
```

## Mathematische Logik

## Daily Bot

Der Daily Bot laeuft standardmaessig nur als sicherer Dry-Run:

```bash
python daily_bot.py --dry-run --mode single
```

Ablauf im Single-Mode:

1. Daten laden
2. Forward-3M-Direct-Forecast bauen
3. robuste 3M-Kovarianz schaetzen
4. Optimizer-Zielportfolio rechnen
5. Szenarien bauen
6. Kandidatenportfolios vergleichen
7. Robust Scorer auswerten
8. Execution Gate pruefen
9. Order Preview schreiben
10. optionalen Execution-Layer nur als Dry-Run/Preview aufrufen

Wichtige Kandidaten:

- `HOLD`
- `OPTIMIZER_TARGET`
- `PARTIAL_25`
- `PARTIAL_50`
- `DEFENSIVE_CASH`
- `MOMENTUM_TILT_SIMPLE`
- optional `CONDITIONAL_FACTOR_TARGET`

Wichtig:

- `HOLD` und `DEFENSIVE_CASH` werden immer mitbewertet
- Kosten, Buffer und Tail-Risk werden abgezogen
- `synthetic_data=true` blockiert Orders im Execution Gate
- Standard bleibt `execution_mode=order_preview_only`

### Current Portfolio CSV

Der Daily Bot liest das aktuelle Simulator-Portfolio standardmaessig aus:

- `data/current_portfolio.csv`

Empfohlenes Schema fuer den aktuellen Ist-Bestand:

```csv
ticker,shares,cash_value
AGG,29,
XLK,18,
SGOV,250,
CASH,,3.73
```

Auch akzeptiert:

```csv
ticker,shares,cash_usd
AGG,29,
XLK,18,
SGOV,250,
CASH,,3.73
```

Legacy-/Fallback-Schemata bleiben weiterhin zulaessig:

- `ticker,current_weight` plus `CASH,current_weight`
- `ticker,current_weight` plus `CASH,cash_value`
- `ticker,current_weight` plus `CASH,cash_usd`

Regeln:

- `CASH` ist eine eigene Zeile und repraesentiert echtes freies Cash, nicht den Cash-Proxy-ETF.
- Wenn kein expliziter Cash-Wert vorhanden ist und share-basierte Positionen genutzt werden, setzt der Parser `cash_usd=0.00` und schreibt eine Warnung in `outputs/current_portfolio_report.txt`.
- Unbekannte oder im aktuellen Lauf inaktive Ticker werden ignoriert und klar reportet.
- Negative Shares werden nicht uebernommen und als Parser-Fehler reportet.
- Fractional Shares werden nur akzeptiert, wenn `ALLOW_FRACTIONAL_SHARES=true`; sonst werden sie als Parser-Fehler reportet.
- `outputs/current_portfolio_report.txt` zeigt das erkannte Schema, die Cash-Eingabemethode sowie Parser-Warnungen/-Fehler.

### Momentum-Forecast

Fuer jeden Handelstag wird pro Asset ein robustes Momentum-Signal berechnet:

- 63-Tage-Momentum
- 126-Tage-Momentum
- Kombination `0.6 * momentum_63 + 0.4 * momentum_126`

### Forecast-Shrinkage

Das rohe Momentum-Signal wird konservativ ueber `kappa` geschrumpft:

```text
mu_robust = kappa * mu_signal
```

Danach wird auf ein festes Prognoseintervall geclippt, damit Ausreisser nicht zu aggressiven Gewichten fuehren.

## Direct Forecast Layer

`forecast_3m.py` verwendet einen vorwaertsgerichteten 3M-Horizont mit:

- 63T-Momentum
- 126T-Momentum
- relativer Staerke
- Trend relativ zur 126T-Linie
- Signal-Confidence
- Unsicherheits-Multiplikator

## Conditional Factor Layer

Die optionale Faktor-Schicht besteht aus:

- `factor_registry.py`
- `asset_factor_mapping.py`
- `macro_data.py`
- `factor_data.py`
- `factor_forecast.py`
- `asset_exposure_model.py`
- `conditional_scenario_model.py`

Wenn Proxy-/Faktor-Daten nicht sauber verfuegbar sind, wird **nicht** abgebrochen. Stattdessen wechselt der Daily Bot automatisch auf:

```text
mode = direct_only
```

und schreibt einen klaren Hinweis in `outputs/factor_model_diagnostics.txt`.

## Szenarien, Kandidaten, Robust Scorer, Execution Gate

- `scenario_model.py` baut direkte 3M-Szenarien wie `base`, `bull_momentum`, `bear_risk_off`, `correlation_stress`, `mean_reversion`
- `candidate_factory.py` baut mehrere handlungsfaehige Kandidaten statt nur ein Zielportfolio
- `robust_scorer.py` vergleicht Kandidaten gegen `HOLD` und `DEFENSIVE_CASH`
- `execution_gate.py` blockiert Orders bei zu wenig Edge, hoher Unsicherheit oder synthetischen Daten

### Kovarianz-Shrinkage

Die Risiko-Matrix wird aus historischen Tagesrenditen geschaetzt:

- Sample-Kovarianz auf einem Rollfenster
- Hochskalierung auf den 3M-Horizont
- Shrinkage Richtung Diagonalmatrix
- kleiner Jitter auf der Diagonale fuer numerische Stabilitaet

### Mean-Variance-Utility

Der Optimizer maximiert eine risikoadjustierte Nutzfunktion:

```text
mu.T @ w
- risk_aversion * w.T @ Sigma @ w
- turnover_penalty * sum(abs(w - w_current))
- concentration_penalty * sum(w_i^2)
```

### Turnover-Penalty

Jede starke Abweichung vom aktuellen Portfolio kostet im Modell Nutzen. Das stabilisiert die Allokation und verhindert zu haeufiges Umschichten.

### Konzentrationsstrafe

Eine quadratische Strafterm auf Einzelgewichte verhindert unnoetig konzentrierte Portfolios.

### Gruppenlimits

Das Universum ist in Gruppen organisiert:

- `us_sector`
- `factor`
- `cash`
- `bonds`
- `commodities`
- `hedge`
- `crypto`

Neben Einzelgewichtslimits gelten:

- Gruppenlimits
- Equity-like-Limit fuer `us_sector + factor`
- Defensive-Mindestgewicht fuer `cash + bonds`
- restriktivere Limits im Risk-Off-Zustand

### Decision Engine

Die Zielallokation wird nicht blind umgesetzt. Vor einem Trade prueft die Decision Engine:

- Zielscore gegen aktuellen Score
- geschaetzte Kosten
- Volatilitaetsbuffer
- CVaR- und Drawdown-Gates
- Wochenrhythmus
- Emergency-Bedingungen

Moegliche Ausgaben:

- `HOLD`
- `WAIT`
- `PARTIAL_REBALANCE`
- `FULL_REBALANCE`
- `DE_RISK`
- `PAUSE`

## Rebalancing-Logik

### Taeglich rechnen

Das Modell berechnet an jedem verfuegbaren Handelstag:

- Forecast
- robuste Kovarianz
- Risk State
- Zielgewichte
- Entscheidungslogik

### Woechentlich regulaer handeln

Normale Rebalances sollen bevorzugt auf dem letzten verfuegbaren Handelstag der Woche stattfinden.

### Taegliche Notfall-Signale

Ausserhalb des regulaeren Wochenrhythmus kann trotzdem gehandelt werden, wenn:

- der Risk Gate faellt
- der Zusatznutzen sehr stark ist
- hoher Turnover sinnvoll ist
- oder ein Drawdown-/Risiko-Trigger greift

## Look-ahead Bias vermeiden

Der Backtest ist strikt walk-forward aufgebaut:

- an Tag `t` werden Forecast, Kovarianz und Risk State nur mit Daten bis einschliesslich `t` berechnet
- die Entscheidung an Tag `t` wird erst auf die Rendite von `t -> t+1` angewendet
- `daily_results.csv` speichert sowohl `date` als auch `next_date`
- der Start erfolgt erst nach einer Warm-up-Phase von `max(200, momentum_long, cov_window) + 1`

Konkret im Code:

- `compute_momentum_forecast_at_date(..., date=t)` nutzt `prices.loc[:t]`
- `estimate_robust_covariance_at_date(..., date=t)` nutzt `returns.loc[:t]`
- `compute_market_risk_state(prices, t)` nutzt nur Historie bis `t`
- die ausgefuehrte Allokation wird danach auf `returns.loc[next_date]` angewendet

## Asset Registry

Die Asset Registry liegt in `asset_universe.py`.

Aktive V1-Gruppen:

- US-Sektor-ETFs
- Faktor-ETFs
- Cash und Bonds
- Commodities / Real Assets
- Hedges
- Crypto

Wichtige Sicherheitsregeln:

- genau ein aktiver Cash-Ticker: `SGOV`
- `BIL`, `BND`, `BNDX`, `DBC` bleiben standardmaessig deaktiviert
- jeder aktive Ticker braucht `name`, `group`, `subgroup`, `max_weight`, `enabled`
- alle aktiven Gruppen brauchen ein Gruppenlimit

## Optional Gurobi

`gurobipy` ist optional.

- Wenn installiert, wird Gurobi verwendet.
- Wenn nicht installiert oder wenn Gurobi scheitert, faellt das Projekt automatisch auf `scipy.optimize` zurueck.
- Wenn auch SciPy scheitert, wird ein feasible equal-weight-like Fallback-Portfolio verwendet.

## Future Paper Trading Adapter

Die Architektur ist absichtlich so angelegt, dass spaeter ein lokaler oder externer Paper-Broker-Adapter ergaenzt werden kann, ohne die Kernpipeline umzubauen:

- [broker_interface.py](/Users/janseliger/Downloads/BF Trading /robust_3m_active_allocation_optimizer/broker_interface.py)
- [paper_broker_stub.py](/Users/janseliger/Downloads/BF Trading /robust_3m_active_allocation_optimizer/paper_broker_stub.py)
- [investopedia_adapter.py](/Users/janseliger/Downloads/BF Trading /robust_3m_active_allocation_optimizer/investopedia_adapter.py)
- [simulator_orchestrator.py](/Users/janseliger/Downloads/BF Trading /robust_3m_active_allocation_optimizer/simulator_orchestrator.py)

Ausrichtung der Schicht:

- Standardpipeline bleibt unveraendert: `Data -> Forecast -> Optimizer -> Decision Engine -> Order Preview -> Email/Reports`
- Broker-/Simulator-Logik ist optional und darf den Hauptlauf nie blockieren
- der aktuelle `PaperBrokerStub` simuliert nur lokal in SQLite
- keine externe API, kein Login, keine Captcha-Umgehung, keine echten Orders
- Standardwert bleibt `ENABLE_LOCAL_PAPER_TRADING = False`; dann entstehen nur Preview-Dateien, keine echten Orders
- wenn spaeter bewusst aktiviert, wird ausschliesslich lokal gegen SQLite simuliert

## Execution and Simulator Architecture

Standard ist immer `Order Preview only`.

- kein echtes Trading
- keine echte Broker-API
- keine echten Orders ausserhalb eines rein optionalen Simulator-/Paper-Kontexts
- die Kernpipeline bleibt: `Data -> Forecast -> Risk -> Optimizer -> Decision Engine -> Order Preview -> Reports / Email / SQLite`
- erst danach folgt optional die Execution-Schicht

Aktuelle Execution-Pfade:

- Default: nur Research-/Preview-Dateien (`research_order_preview.csv`, Legacy-Alias `order_preview.csv`)
- optional: lokaler `PaperBrokerStub` in SQLite
- optional: experimenteller `InvestopediaSimulatorAdapter`

Sicherheitsregeln:

- Investopedia wird nur genutzt, wenn `ENABLE_INVESTOPEDIA_SIMULATOR=true`
- Credentials kommen nur aus `.env`
- keine CAPTCHA-Umgehung
- keine MFA-Umgehung
- keine Anti-Bot-Workarounds
- wenn Login oder Website-Struktur fehlschlaegt, laeuft das Kernsystem trotzdem weiter

Empfohlener Workflow:

1. `python main.py` ausfuehren und Optimizer/Reports pruefen
2. `outputs/research_order_preview.csv` pruefen
3. optional `ENABLE_LOCAL_PAPER_TRADING=true` fuer den lokalen SQLite-Stub testen
4. erst danach optional und experimentell Investopedia pruefen

Historische Referenzen, aber keine Pflicht-Abhaengigkeiten:

- [dchrostowski/investopedia_simulator_api](https://github.com/dchrostowski/investopedia_simulator_api)
- [kirkthaker/investopedia-trading-api](https://github.com/kirkthaker/investopedia-trading-api)
- [kirkthaker/investope](https://github.com/kirkthaker/investope)

Diese Repositories sollten nur konzeptionell betrachtet werden. Inoffizielle Integrationen koennen jederzeit an Login-Aenderungen, CAPTCHA, MFA, Session-Handling oder HTML-Aenderungen brechen.

Aktueller Realstatus:

- `investopedia_adapter.py` ist ein sicherer experimenteller Stub
- automatische Website-/Simulator-Ausfuehrung ist derzeit nicht produktiv aktiv
- der Default bleibt garantiert `order_preview_only`

## Sichere Defaults

Der Standardlauf ist absichtlich ein sicherer Dry-Run. Ohne explizite Umstellung in `.env` gilt:

- `ENABLE_EMAIL_NOTIFICATIONS=false`
- `ENABLE_LOCAL_PAPER_TRADING=false`
- `ENABLE_INVESTOPEDIA_SIMULATOR=false`
- `ENABLE_EXTERNAL_BROKER=false`
- `DRY_RUN=true`

Damit macht das Projekt standardmaessig nur:

- CSV-Outputs
- Charts
- SQLite-Speicherung
- Order Preview
- `latest_email_notification.txt`

Und standardmaessig nicht:

- keine externen Orders
- keine Simulator-Orders
- keine Broker-API-Aufrufe
- keine Pflicht-Zugangsdaten

Konzeptionelle Orientierung fuer spaetere Evaluierung:

- [dchrostowski/investopedia_simulator_api](https://github.com/dchrostowski/investopedia_simulator_api)
- [kirkthaker/investopedia-trading-api](https://github.com/kirkthaker/investopedia-trading-api)
- [kirkthaker/investope](https://github.com/kirkthaker/investope)

Wichtige Vorsicht:

- alte inoffizielle Investopedia-APIs koennen jederzeit wegen Login-Aenderungen, reCAPTCHA, MFA oder HTML-Aenderungen brechen
- diese Repositories sind hier nur konzeptionelle Referenzen, keine Pflicht-Abhaengigkeiten
- zuerst sollte immer der lokale `PaperBrokerStub` verwendet werden
- eine spaetere Investopedia-Anbindung waere optional, experimentell und duerfte nie Voraussetzung fuer `python main.py` sein

Aktuell gilt weiterhin:

- keine echte Broker-API
- keine echten Orders
- kein Pflicht-Login
- keine Investopedia-Abhaengigkeit fuer den Hauptlauf

## Investopedia Simulator Integration

Die Investopedia-Simulator-Anbindung ist in diesem Projekt nur als optionale und experimentelle Adapter-Schicht vorbereitet. Die Kernpipeline bleibt immer:

`Data -> Forecast -> Optimizer -> Decision Engine -> Order Preview -> Reports/E-Mail`

Wichtige Regeln:

- nur virtuelles Geld / Simulator-Kontext
- standardmaessig deaktiviert ueber `ENABLE_INVESTOPEDIA_SIMULATOR=false`
- Zugangsdaten werden ausschliesslich aus `.env` gelesen
- keine Passwoerter werden hardcodiert oder geloggt
- keine CAPTCHA-Umgehung
- keine Anti-Bot-Workarounds
- wenn Login, MFA, reCAPTCHA, HTML-Aenderungen oder Timeouts stoeren, darf der Adapter nur sauber fehlschlagen; `python main.py` muss trotzdem weiterlaufen

Relevante `.env`-Eintraege:

- `ENABLE_INVESTOPEDIA_SIMULATOR=false`
- `INVESTOPEDIA_USERNAME=your_username`
- `INVESTOPEDIA_PASSWORD=your_password`
- `INVESTOPEDIA_GAME_ID=your_game_id_or_portfolio_id`

Aktueller Stand:

- es gibt einen experimentellen `investopedia_adapter.py`
- dieser fuehrt bewusst keine riskante Login-Automation oder Pflicht-Scraping-Logik aus
- `simulator_orchestrator.py` kapselt die sichere Auswahl zwischen Preview-only, lokalem Paper-Stub und experimentellem Adapter
- `interface_tests.py` bietet einen einfachen Smoke-Test fuer diese Schnittstellen
- falls du ihn spaeter aktivierst und keine stabile unterstuetzte Bibliothek verfuegbar ist, wird nur eine verstaendliche Warnung geloggt und keine Order uebertragen

Hinweis zu inoffiziellen APIs:

- alte inoffizielle Investopedia-Integrationen koennen jederzeit wegen Login-Aenderungen, reCAPTCHA, MFA oder HTML-Aenderungen brechen
- deshalb sollte zuerst immer der lokale `PaperBrokerStub` verwendet werden
- der Hauptoptimizer bleibt absichtlich voll funktionsfaehig, auch wenn Investopedia nicht funktioniert

## Grenzen des Modells

- keine Steuern
- keine Broker-API
- keine echten Orders
- kein Slippage- oder Market-Impact-Modell
- keine echte Bayesian-DRO-Engine
- keine Garantie fuer Datenqualitaet oder Performance
- `yfinance` ist praktisch, aber keine institutionelle Datenquelle
- bei komplett fehlender Live-Datenbasis kann ein klar markierter synthetischer Offline-Fallback fuer Pipeline-Tests verwendet werden

## Naechste moegliche Erweiterungen

- Paper Trading
- Broker API
- bessere Datenquellen
- echter Regimefilter
- Steuerlogik

## Assumptions

- Einzelne fehlende ETF-Serien duerfen pro Lauf temporaer entfernt werden, solange mindestens 10 investierbare Assets uebrig bleiben.
- Ohne `.env` oder ohne gueltige SMTP-Daten werden keine E-Mails versendet.
- Ohne `gurobipy` wird automatisch `scipy.optimize` verwendet.
- `research_order_preview.csv` ist die kanonische hypothetische Research-/Backtest-Rebalance-Vorschau; `order_preview.csv` ist nur ihr Legacy-Alias.
- Ein synthetischer Fallback ist nur fuer Robustheit und Pipeline-Tests gedacht, nicht fuer echte Research-Aussagen.

## Scheduler

Fuer spaetere Automatisierung:

- Windows: Task Scheduler
- macOS / Linux: `cron`

Beispiel:

```cron
0 7 * * 1-5 cd /pfad/zum/projekt && /pfad/zum/python main.py
```

## Projektstruktur

```text
robust_3m_active_allocation_optimizer/
├── config.py
├── asset_universe.py
├── data.py
├── calendar_utils.py
├── features.py
├── risk.py
├── optimizer.py
├── decision.py
├── backtest.py
├── metrics.py
├── report.py
├── notifications.py
├── order_preview.py
├── database.py
├── main.py
├── requirements.txt
├── README.md
├── .env.example
├── outputs/
├── data/
└── notebooks/
```

## Morgen frueh in 5 Befehlen

```bash
cd ~/robust_3m_active_allocation_optimizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```
