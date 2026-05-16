from __future__ import annotations

import email
import imaplib
import os
import re
import time
from email.header import decode_header
from email.utils import parsedate_to_datetime
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "investopedia_tud_navigator"
PROFILE_DIR = ROOT / ".playwright" / "investopedia_profile"

PORTFOLIO_URL = "https://www.investopedia.com/simulator/portfolio"
SIMULATOR_URL = "https://www.investopedia.com/simulator/"
MY_GAMES_URLS = [
    "https://www.investopedia.com/simulator/games",
    "https://www.investopedia.com/simulator/my-games",
    "https://www.investopedia.com/simulator/",
]


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def short_wait(page, timeout: int = 1500) -> None:
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout)
    except Exception:
        pass


def text_of(page) -> str:
    try:
        return page.locator("body").inner_text(timeout=2500)
    except Exception:
        return ""


def decode_mime(value: str | None) -> str:
    if not value:
        return ""
    out = []
    for payload, charset in decode_header(value):
        if isinstance(payload, bytes):
            out.append(payload.decode(charset or "utf-8", errors="ignore"))
        else:
            out.append(str(payload))
    return "".join(out)


def message_text(msg: email.message.Message) -> str:
    chunks = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            dispo = str(part.get("Content-Disposition", ""))
            if "attachment" in dispo.lower():
                continue
            if ctype in {"text/plain", "text/html"}:
                payload = part.get_payload(decode=True)
                if payload:
                    chunks.append(payload.decode(part.get_content_charset() or "utf-8", errors="ignore"))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            chunks.append(payload.decode(msg.get_content_charset() or "utf-8", errors="ignore"))
    return "\n".join(chunks)


def extract_otp(raw: str) -> str:
    m = re.search(r"\b(\d{5,6})\b", raw or "")
    if not m:
        raise RuntimeError("OTP input did not contain a 5- or 6-digit code")
    return m.group(1)


def read_latest_otp(timeout_seconds: int = 10, min_epoch: float | None = None) -> str:
    host = env("OTP_IMAP_HOST")
    user = env("OTP_IMAP_USER")
    password = env("OTP_IMAP_PASSWORD").replace(" ", "")
    mailbox = env("OTP_MAILBOX", "INBOX")
    subject_hint = env("OTP_SUBJECT_HINT", "access your account").lower()

    if not host or not user or not password:
        raise RuntimeError("OTP mailbox settings fehlen in .env")

    deadline = time.time() + timeout_seconds
    last_error = None

    while time.time() < deadline:
        try:
            with imaplib.IMAP4_SSL(host) as imap:
                imap.login(user, password)
                imap.select(mailbox)

                # Search newest messages only. We still validate Date below.
                status, data = imap.search(None, "ALL")
                if status != "OK":
                    time.sleep(0.4)
                    continue

                ids = data[0].split()[-40:]
                for msg_id in reversed(ids):
                    status, msg_data = imap.fetch(msg_id, "(RFC822)")
                    if status != "OK" or not msg_data or not msg_data[0]:
                        continue

                    raw = msg_data[0][1]
                    msg = email.message_from_bytes(raw)

                    # Ignore old OTP mails from previous login attempts.
                    if min_epoch is not None:
                        try:
                            msg_dt = parsedate_to_datetime(msg.get("Date"))
                            if msg_dt is not None:
                                msg_epoch = msg_dt.timestamp()
                                # Allow tiny clock skew.
                                if msg_epoch < min_epoch - 20:
                                    continue
                        except Exception:
                            # If Date parsing fails, do not trust this mail for fresh OTP.
                            continue

                    blob = "\n".join([
                        decode_mime(msg.get("From")),
                        decode_mime(msg.get("Subject")),
                        message_text(msg),
                    ])
                    blob_lower = blob.lower()

                    # Forwarded mails may be "WG: Access your account".
                    if subject_hint and subject_hint not in blob_lower and "access your account" not in blob_lower:
                        continue

                    return extract_otp(blob)

        except Exception as e:
            last_error = e
            if "AUTHENTICATIONFAILED" in str(e).upper() or "Invalid credentials" in str(e):
                raise

        time.sleep(0.4)

    if last_error:
        raise TimeoutError(f"No fresh OTP found before timeout; last error: {type(last_error).__name__}: {last_error}")
    raise TimeoutError("No fresh OTP found before timeout")

def fill_first(page, selectors: list[str], value: str, timeout: int = 2000) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            loc.wait_for(state="visible", timeout=timeout)
            loc.fill(value, timeout=timeout)
            return True
        except Exception:
            pass
    return False


def click_first(page, selectors: list[str], timeout: int = 2000) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            loc.wait_for(state="visible", timeout=timeout)
            loc.scroll_into_view_if_needed(timeout=timeout)
            loc.click(timeout=timeout)
            short_wait(page)
            return True
        except Exception:
            pass
    return False


def accept_cookies_if_present(page) -> None:
    for sel in [
        "button:has-text('Accept All')",
        "button:has-text('Accept all')",
        "button:has-text('I Accept')",
        "button:has-text('Agree')",
        "button:has-text('Reject All')",
    ]:
        try:
            loc = page.locator(sel).first
            loc.wait_for(state="visible", timeout=1200)
            loc.click(timeout=1200)
            short_wait(page, timeout=800)
            print(f"clicked cookie button: {sel}")
            return
        except Exception:
            pass


def looks_logged_in(page) -> bool:
    url = page.url.lower()
    txt = text_of(page).lower()

    # These mean definitely NOT logged in.
    negative_markers = [
        "sign in to simulator",
        "email address",
        "register now",
        "unlock the power of investing",
        "enter your email",
        "access your account",
        "sign in",
        "log in",
    ]

    if "openid-connect/auth" in url or "login" in url or "sign-in" in url:
        return False

    if any(marker in txt for marker in negative_markers):
        return False

    # These mean likely logged in.
    positive_markers = [
        "portfolio",
        "my games",
        "account value",
        "buying power",
        "holdings",
        "positions",
        "trade",
        "go to game",
    ]

    return "simulator" in url and any(marker in txt for marker in positive_markers)

def enter_otp_code(page, code_raw: str) -> None:
    code = extract_otp(code_raw)

    try:
        page.locator("input").first.wait_for(state="visible", timeout=3000)
    except Exception:
        pass

    inputs = page.locator("input")
    visible = []

    try:
        count = inputs.count()
    except Exception:
        count = 0

    for i in range(count):
        try:
            inp = inputs.nth(i)
            if not inp.is_visible():
                continue
            typ = (inp.get_attribute("type") or "").lower()
            name = (inp.get_attribute("name") or "").lower()
            ident = (inp.get_attribute("id") or "").lower()
            autocomplete = (inp.get_attribute("autocomplete") or "").lower()
            placeholder = (inp.get_attribute("placeholder") or "").lower()
            probe = f"{typ} {name} {ident} {autocomplete} {placeholder}"
            if "email" in probe:
                continue
            if typ in {"text", "tel", "number", ""} or "code" in probe or "otp" in probe or "one-time" in probe:
                visible.append(inp)
        except Exception:
            pass

    if len(visible) >= len(code):
        for digit, inp in zip(code, visible[:len(code)]):
            inp.click(timeout=1000)
            inp.fill(digit, timeout=1000)
        return

    for sel in [
        "input[autocomplete='one-time-code']",
        "input[name*='code']",
        "input[id*='code']",
        "input[type='tel']",
        "input[type='number']",
        "input[type='text']",
    ]:
        try:
            loc = page.locator(sel).first
            loc.wait_for(state="visible", timeout=1200)
            loc.fill(code, timeout=1200)
            return
        except Exception:
            pass

    try:
        page.locator("input").first.click(timeout=1000)
        page.keyboard.insert_text(code)
        return
    except Exception:
        raise RuntimeError("Could not fill OTP code")


def ensure_logged_in(page) -> None:
    investopedia_email = env("INVESTOPEDIA_EMAIL")
    if not investopedia_email:
        raise RuntimeError("INVESTOPEDIA_EMAIL fehlt in .env")

    page.goto(PORTFOLIO_URL, wait_until="domcontentloaded", timeout=30000)
    short_wait(page)
    accept_cookies_if_present(page)

    if looks_logged_in(page):
        print("Investopedia session already logged in.")
        return

    print("Login required. Entering Investopedia email.")

    if not fill_first(
        page,
        [
            "input[type='email']",
            "input[name='email']",
            "input[name='username']",
            "input#email",
            "input#username",
        ],
        investopedia_email,
        timeout=5000,
    ):
        raise RuntimeError("Could not find email input")

    otp_requested_at = time.time()

    click_first(
        page,
        [
            "button:has-text('Continue')",
            "button:has-text('Next')",
            "button:has-text('Sign in')",
            "button:has-text('Log in')",
            "input[type='submit']",
        ],
        timeout=4000,
    )

    try:
        page.locator("input").first.wait_for(state="visible", timeout=3500)
    except Exception:
        pass

    auto_read = env("OTP_AUTO_READ", "true").lower() in {"1", "true", "yes", "y"}
    wait_seconds = int(env("OTP_WAIT_SECONDS", "20") or "20")

    if auto_read:
        print(f"Reading one-time code from mailbox, max {wait_seconds}s.")
        try:
            code = read_latest_otp(timeout_seconds=wait_seconds, min_epoch=otp_requested_at)
            print("OTP found. Entering code.")
        except Exception as e:
            raise RuntimeError(
                "OTP auto-read failed and manual fallback is disabled. "
                f"Original error: {type(e).__name__}: {e}"
            )
    else:
        print("OTP auto-read disabled; asking for code immediately.")
        raw = input("Paste ONLY the 5-digit Investopedia code, then press ENTER: ").strip()
        code = extract_otp(raw)

    enter_otp_code(page, code)

    if not submit_otp_and_wait_for_login(page, timeout_seconds=20):
        try:
            save_snapshot(page, "login_after_otp_failed")
        except Exception:
            pass
        raise RuntimeError("Login did not reach a logged-in simulator session after OTP submit")

    print("Automatic Investopedia login succeeded.")



def submit_otp_and_wait_for_login(page, timeout_seconds: int = 20) -> bool:
    # Some Investopedia OTP forms auto-submit, some require Enter,
    # some require Continue/Verify. Try all without long blind sleeps.
    submit_selectors = [
        "button:has-text('Continue')",
        "button:has-text('Verify')",
        "button:has-text('Submit')",
        "button:has-text('Sign in')",
        "button:has-text('Log in')",
        "button:has-text('Next')",
        "input[type='submit']",
    ]

    try:
        page.keyboard.press("Enter")
    except Exception:
        pass

    deadline = time.time() + timeout_seconds
    clicked_any = False

    while time.time() < deadline:
        # If we are logged in now, stop immediately.
        if looks_logged_in(page):
            return True

        # Click any visible submit-ish button.
        for sel in submit_selectors:
            try:
                loc = page.locator(sel).first
                if loc.is_visible(timeout=500):
                    loc.click(timeout=1000)
                    clicked_any = True
                    short_wait(page, timeout=1200)
                    break
            except Exception:
                pass

        # Press Enter again; harmless if already submitted.
        try:
            page.keyboard.press("Enter")
        except Exception:
            pass

        try:
            page.wait_for_load_state("domcontentloaded", timeout=1200)
        except Exception:
            pass

        if looks_logged_in(page):
            return True

        time.sleep(0.5)

    return looks_logged_in(page)


def choose_tud_game(page) -> bool:
    game_name = env("TUD_GAME_NAME", "TUDSoSe2026 BehavioralFinance")
    game_password = env("TUD_GAME_PASSWORD", "TUDbehavioral2026")

    def save_game_debug(label: str) -> None:
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / f"{label}.txt").write_text(text_of(page), encoding="utf-8", errors="ignore")
            (OUT_DIR / f"{label}.html").write_text(page.content(), encoding="utf-8", errors="ignore")
            try:
                page.screenshot(path=str(OUT_DIR / f"{label}.png"), full_page=True, timeout=15000)
            except Exception:
                pass
            print(f"Saved debug snapshot: {label}")
        except Exception as e:
            print(f"Could not save debug snapshot {label}: {e}")

    print("Opening simulator home after confirmed login.")
    page.goto("https://www.investopedia.com/simulator/", wait_until="domcontentloaded", timeout=30000)
    short_wait(page, timeout=2000)
    accept_cookies_if_present(page)

    # Falls wir schon im richtigen Spiel sind, reicht Portfolio.
    txt = text_of(page)
    if game_name in txt:
        print("TUD game already visible.")
        return True

    # 1) Games öffnen
    for sel in [
        "a:has-text('Games')",
        "button:has-text('Games')",
        "text=Games",
    ]:
        try:
            loc = page.locator(sel).first
            loc.wait_for(state="visible", timeout=3500)
            loc.click(timeout=3500)
            short_wait(page, timeout=2500)
            print(f"clicked Games via {sel}")
            break
        except Exception:
            pass

    # 2) Join Game öffnen
    for sel in [
        "a:has-text('Join Game')",
        "button:has-text('Join Game')",
        "text=Join Game",
        "a:has-text('Join')",
        "button:has-text('Join')",
    ]:
        try:
            loc = page.locator(sel).first
            loc.wait_for(state="visible", timeout=3500)
            loc.click(timeout=3500)
            short_wait(page, timeout=2500)
            print(f"clicked Join Game via {sel}")
            break
        except Exception:
            pass

    # 3) Game Lookup Feld füllen
    filled_lookup = False
    for sel in [
        "input[placeholder*='Game']",
        "input[placeholder*='Lookup']",
        "input[aria-label*='Game']",
        "input[name*='game']",
        "input[type='search']",
        "input[type='text']",
    ]:
        try:
            loc = page.locator(sel).first
            loc.wait_for(state="visible", timeout=4000)
            loc.fill(game_name, timeout=3000)
            short_wait(page, timeout=1500)
            print(f"filled game lookup via {sel}")
            filled_lookup = True
            break
        except Exception:
            pass

    if not filled_lookup:
        save_game_debug("join_game_lookup_not_found")
        return False

    # 4) Ergebnis/Join klicken
    clicked_join = False
    for sel in [
        f"text={game_name}",
        "button:has-text('Join')",
        "a:has-text('Join')",
        "button:has-text('JOIN')",
        "a:has-text('JOIN')",
    ]:
        try:
            loc = page.locator(sel).first
            loc.wait_for(state="visible", timeout=5000)
            loc.click(timeout=3500)
            short_wait(page, timeout=2500)
            print(f"clicked game/join via {sel}")
            clicked_join = True
            break
        except Exception:
            pass

    if not clicked_join:
        # JS fallback: click first visible Join near target game name.
        try:
            clicked = page.evaluate(
                r"""(gameName) => {
                    const norm = s => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();
                    const target = norm(gameName);
                    const visible = el => {
                        const r = el.getBoundingClientRect();
                        const st = window.getComputedStyle(el);
                        return r.width > 0 && r.height > 0 && st.display !== 'none' && st.visibility !== 'hidden';
                    };
                    const all = Array.from(document.querySelectorAll('body *')).filter(visible);
                    const boxes = all.filter(el => norm(el.innerText).includes(target))
                                     .sort((a,b) => {
                                        const ra=a.getBoundingClientRect(), rb=b.getBoundingClientRect();
                                        return (ra.width*ra.height)-(rb.width*rb.height);
                                     });
                    for (const box of boxes) {
                        const actions = Array.from(box.querySelectorAll('a,button,[role=button]')).filter(visible);
                        const join = actions.find(el => norm(el.innerText).includes('join'));
                        if (join) {
                            join.scrollIntoView({block:'center'});
                            join.click();
                            return true;
                        }
                    }
                    return false;
                }""",
                game_name,
            )
            print(f"clicked join via JS={clicked}")
            clicked_join = bool(clicked)
            short_wait(page, timeout=2500)
        except Exception as e:
            print(f"JS join failed: {type(e).__name__}: {e}")

    if not clicked_join:
        save_game_debug("join_button_not_found")
        return False

    # 5) Passwort eingeben, falls abgefragt.
    txt = text_of(page).lower()
    if "password" in txt or "passwort" in txt or "confirm join" in txt:
        filled_pw = False
        for sel in [
            "input[type='password']",
            "input[name*='password']",
            "input[placeholder*='Password']",
            "input[placeholder*='password']",
            "input[type='text']",
        ]:
            try:
                loc = page.locator(sel).first
                loc.wait_for(state="visible", timeout=4000)
                loc.fill(game_password, timeout=3000)
                short_wait(page, timeout=800)
                print(f"filled game password via {sel}")
                filled_pw = True
                break
            except Exception:
                pass

        if not filled_pw:
            save_game_debug("game_password_field_not_found")
            return False

        for sel in [
            "button:has-text('Confirm Join')",
            "a:has-text('Confirm Join')",
            "button:has-text('CONFIRM JOIN')",
            "button:has-text('Confirm')",
            "button:has-text('Join')",
        ]:
            try:
                loc = page.locator(sel).first
                loc.wait_for(state="visible", timeout=4000)
                loc.click(timeout=3500)
                short_wait(page, timeout=3000)
                print(f"clicked confirm via {sel}")
                break
            except Exception:
                pass

    save_game_debug("after_join_game_attempt")

    # 6) Erfolgsprüfung
    txt = text_of(page)
    if game_name in txt or "portfolio" in txt.lower() or "account value" in txt.lower():
        print("TUD game join/open appears successful.")
        return True

    return False

def open_portfolio(page) -> None:
    KNOWN = ["AGG", "IEF", "PDBC", "SGOV", "SHY", "SPMO", "TIP", "XLE", "XLK", "XLP"]

    def save_debug(label: str) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        txt = text_of(page)
        html = page.content()

        (OUT_DIR / f"{label}.txt").write_text(txt, encoding="utf-8", errors="ignore")
        (OUT_DIR / f"{label}.html").write_text(html, encoding="utf-8", errors="ignore")

        try:
            tables = page.locator("table").evaluate_all(
                """tables => tables.map((table, ti) => ({
                    table: ti,
                    text: (table.innerText || '').replace(/\\s+/g, ' ').trim(),
                    rows: Array.from(table.querySelectorAll('tr')).map(tr =>
                        Array.from(tr.querySelectorAll('th,td')).map(td =>
                            (td.innerText || '').replace(/\\s+/g, ' ').trim()
                        )
                    )
                }))"""
            )
            import json
            (OUT_DIR / f"{label}_tables.json").write_text(
                json.dumps(tables, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"WARNING: table dump skipped for {label}: {type(e).__name__}: {e}")

        try:
            page.screenshot(path=str(OUT_DIR / f"{label}.png"), full_page=False, timeout=5000)
        except Exception:
            pass

        print(f"Saved debug snapshot: {label}")

    def ticker_count() -> int:
        txt = text_of(page)
        return sum(1 for t in KNOWN if re.search(rf"\\b{t}\\b", txt))

    def wait_for_real_tickers(seconds: int = 35) -> bool:
        import time
        deadline = time.time() + seconds

        while time.time() < deadline:
            txt = text_of(page)
            count = sum(1 for t in KNOWN if re.search(rf"\\b{t}\\b", txt))

            print(f"ticker_count={count}")

            if count >= 3:
                return True

            # Keine Mausbewegung, kein Wheel. Nur auf DOM/Text warten.
            for t in KNOWN:
                try:
                    page.get_by_text(t, exact=True).first.wait_for(state="visible", timeout=700)
                    return True
                except Exception:
                    pass

            try:
                page.wait_for_load_state("networkidle", timeout=1200)
            except Exception:
                pass

            try:
                page.wait_for_timeout(1000)
            except Exception:
                pass

        return ticker_count() >= 3

    # Nie Trade klicken. Nur Portfolio.
    for sel in [
        "a:has-text('Portfolio')",
        "button:has-text('Portfolio')",
        "[role=button]:has-text('Portfolio')",
        "text=PORTFOLIO",
        "text=Portfolio",
    ]:
        try:
            loc = page.locator(sel).first
            loc.wait_for(state="visible", timeout=5000)
            loc.click(timeout=5000)
            print(f"clicked Portfolio via {sel}")

            try:
                page.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                pass

            if wait_for_real_tickers(seconds=35):
                print(f"Real ticker rows visible. ticker_count={ticker_count()}")
                save_debug("tud_holdings_live")
                return

            save_debug("after_portfolio_click_no_tickers")
            raise RuntimeError("Portfolio opened, but ticker rows did not become visible")

        except Exception as e:
            print(f"Portfolio selector failed/insufficient {sel}: {type(e).__name__}: {e}")

    raise RuntimeError("Could not open real holdings table")

def save_snapshot(page, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html = page.content()
    txt = text_of(page)

    html_path = OUT_DIR / f"{name}.html"
    txt_path = OUT_DIR / f"{name}.txt"
    png_path = OUT_DIR / f"{name}.png"

    html_path.write_text(html, encoding="utf-8", errors="ignore")
    txt_path.write_text(txt, encoding="utf-8", errors="ignore")

    try:
        page.screenshot(path=str(png_path), full_page=False, timeout=5000)
    except Exception as e:
        print(f"WARNING: screenshot skipped for {name}: {type(e).__name__}: {e}")

    print(f"Saved snapshot: {name}")
    print(f"- {txt_path}")
    print(f"- {html_path}")
    print(f"- {png_path}")

def main() -> None:
    load_dotenv(ROOT / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    keep_on_error = env("KEEP_BROWSER_OPEN_ON_ERROR", "true").lower() in {"1", "true", "yes", "y"}
    keep_on_success = env("KEEP_BROWSER_OPEN_ON_SUCCESS", "false").lower() in {"1", "true", "yes", "y"}

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            viewport={"width": 1600, "height": 2600},
        )
        page = browser.pages[0] if browser.pages else browser.new_page()
        page.set_default_timeout(3000)
        page.set_default_navigation_timeout(15000)

        try:
            ensure_logged_in(page)
            learned_url_path = ROOT / "config" / "investopedia_tud_portfolio_url.txt"
            if learned_url_path.exists() and learned_url_path.read_text(encoding="utf-8").strip():
                print("Using learned TUD portfolio URL; skipping game selection.")
            else:
                selected = choose_tud_game(page)
                print(f"selected_tud_game={selected}")

            open_portfolio(page)
            save_snapshot(page, "tud_portfolio")

            if keep_on_success:
                input("Success. Browser stays open. Press ENTER to close...")
            browser.close()

        except Exception as e:
            print(f"ERROR during auto refresh: {type(e).__name__}: {e}")
            try:
                save_snapshot(page, "tud_error")
            except Exception:
                pass
            if keep_on_error:
                input("Browser kept open after error. Inspect it, then press ENTER to close...")
            browser.close()
            raise


if __name__ == "__main__":
    main()
