# Daily Email Operations

## Zweck

Dieser Betriebsmodus erzeugt einen taeglichen Portfolio-Mailbericht fuer den Operator.

- Kein echter Orderversand
- Kein Broker
- Keine Investopedia-Automation
- Der Mailpfad ist von Portfolio-Recherche und Orderausfuehrung getrennt

Der operative Entry-Point ist:

```bash
./run_daily_email_review.sh
```

## Preview-Betrieb

Preview ist der sichere Default. Dabei wird nur der taegliche Review plus Mail-Preview erzeugt.

Benötigte ENV-Werte:

```bash
ENABLE_EMAIL_NOTIFICATIONS=false
EMAIL_SEND_ENABLED=false
EMAIL_DRY_RUN=true
PHASE=DAILY_REVIEW_PREVIEW
ENABLE_EXTERNAL_BROKER=false
ENABLE_INVESTOPEDIA_SIMULATOR=false
ENABLE_LOCAL_PAPER_TRADING=false
```

Optionale ENV-Werte:

```bash
EMAIL_RECIPIENT=
EMAIL_PROVIDER=brevo
MAX_EMAILS_PER_DAY=1
DAILY_BRIEFING_ONLY=true
```

Start:

```bash
cd "/Users/janseliger/Downloads/BF Trading /robust_3m_active_allocation_optimizer"
./run_daily_email_review.sh
```

Erwartete Outputs:

- `outputs/daily_portfolio_review.txt`
- `outputs/daily_portfolio_review.csv`
- `outputs/daily_email_subject.txt`
- `outputs/daily_email_briefing.txt`
- `outputs/latest_email_notification.txt`
- `outputs/email_safety_report.txt`
- `outputs/daily_review_validation_report.txt`
- `outputs/email_final_acceptance_report.txt`
- `outputs/last_email_state.json`
- `outputs/manual_simulator_orders.csv`

Erwartetes Verhalten:

- Mail-Preview wird geschrieben
- `email_send_attempted=false`
- `email_send_success=false`
- `email_result_reason=preview_only` oder ein anderes Gate-/Dry-Run-Reason
- Keine echten Orders

## Send-Betrieb

Send-Modus ist getrennt vom Preview-Modus. Echter Versand ist nur erlaubt, wenn das zentrale Mail-Gate offen ist.

Pflicht-ENV-Werte fuer echten Send:

```bash
ENABLE_EMAIL_NOTIFICATIONS=true
EMAIL_SEND_ENABLED=true
EMAIL_DRY_RUN=false
EMAIL_RECIPIENT=operator@example.com
USER_CONFIRMED_EMAIL_PHASE=true
PHASE=DAILY_REVIEW_SEND_READY
ENABLE_EXTERNAL_BROKER=false
ENABLE_INVESTOPEDIA_SIMULATOR=false
ENABLE_LOCAL_PAPER_TRADING=false
```

Provider:

```bash
EMAIL_PROVIDER=brevo
```

Brevo-Variante:

```bash
BREVO_API_KEY=your_brevo_api_key
EMAIL_SENDER=verified_sender@example.com
EMAIL_RECIPIENT=operator@example.com
```

Optionaler Legacy-SMTP-Fallback:

```bash
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=bot@example.com
SMTP_PASSWORD=app-password-or-token
EMAIL_SENDER=bot@example.com
EMAIL_RECIPIENT=operator@example.com
EMAIL_USE_SSL=false
EMAIL_USE_STARTTLS=true
```

Fake-Provider fuer sichere Sendpfad-Tests:

```bash
EMAIL_PROVIDER=fake
EMAIL_FAKE_SEND_SUCCESS=true
EMAIL_RECIPIENT=test@example.com
```

Wichtig:

- `EMAIL_PROVIDER=fake` sendet nichts ueber das Netzwerk.
- Auch der Fake-Provider funktioniert nur, wenn das Mail-Gate echten Send erlauben wuerde.
- Mailversand impliziert niemals Broker-, Investopedia- oder Orderaktivierung.

## Cron-Beispiel

Empfohlene Cron-Variante mit `cd`, lokaler `.venv` und Logdatei:

```cron
5 16 * * 1-5 cd "/Users/janseliger/Downloads/BF Trading /robust_3m_active_allocation_optimizer" && . ./.venv/bin/activate && ./run_daily_email_review.sh >> outputs/cron_daily_email_review.log 2>&1
```

Alternative ohne `activate`:

```cron
5 16 * * 1-5 cd "/Users/janseliger/Downloads/BF Trading /robust_3m_active_allocation_optimizer" && ./.venv/bin/python config_validation.py >/dev/null 2>&1 && ./run_daily_email_review.sh >> outputs/cron_daily_email_review.log 2>&1
```

Erwartetes Cron-Verhalten:

- Exit `0`, wenn Daily Review und Mail-Preview erzeugt wurden
- Exit `0`, wenn echter Send korrekt vom Gate blockiert wurde
- Nonzero nur bei kaputter Config, Health-Check-/Daily-Bot-Fehlern oder bei offenem Gate mit fehlgeschlagenem Send

## Fehlersuche

Diese Dateien zuerst pruefen:

- `outputs/email_safety_report.txt`
- `outputs/daily_review_validation_report.txt`
- `outputs/email_final_acceptance_report.txt`
- `outputs/latest_email_notification.txt`
- `outputs/last_email_state.json`
- `outputs/cron_daily_email_review.log`

Schnelle Leselogik:

- `email_safety_report.txt`: Warum echter Send erlaubt oder blockiert ist
- `daily_review_validation_report.txt`: Ob der Mailbody formal sendefaehig ist
- `email_final_acceptance_report.txt`: Gesamtabnahme fuer Preview oder Send
- `latest_email_notification.txt`: Letzte konkrete Mail-Preview
- `last_email_state.json`: Dedupe- und letzter Sendestatus

## Was nie aktiviert werden soll

Folgendes bleibt fuer den Mailroboter tabu:

- `ENABLE_EXTERNAL_BROKER=true`
- `ENABLE_INVESTOPEDIA_SIMULATOR=true`
- echte Orders oder automatische Ausfuehrung

Wenn einer dieser Punkte aktiv ist, muss echter Mailversand blockiert bleiben.

## Sicherheitsregeln

- Keine Secrets committen
- `.env` nicht in Git einchecken
- Outputs regelmaessig pruefen
- `used_cache_fallback=true` immer als Warnsignal behandeln
- `synthetic_data=true` oder `data_freshness_ok=false` niemals ignorieren
- Fuer Simulatororders nur `outputs/manual_simulator_orders.csv` verwenden
- Nicht verwenden: `outputs/order_preview.csv`

## Preview vs. Send

Preview:

- schreibt Reports und Mail-Preview
- sendet nichts
- ist der Default

Send:

- nutzt denselben Review-Inhalt
- braucht explizite Freigabe per Gate
- bleibt weiterhin ohne Broker, Investopedia und echte Orders
