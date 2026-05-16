"""Investopedia session keeper.

Read-only. Does not submit orders.

Purpose:
- Opens persistent Chromium profile.
- Lets the user log in manually once.
- Lets the user navigate to the desired game/portfolio.
- Saves repeated snapshots without closing the browser until user quits.

Commands after browser opens:
- ENTER: save snapshot of current page
- q: quit
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "investopedia_session_keeper"
PROFILE = ROOT / ".local" / "investopedia_browser_profile"

START_URL = "https://www.investopedia.com/simulator/portfolio"


def save_snapshot(page, label: str = "snapshot") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    html_path = OUT / f"{label}_{stamp}.html"
    text_path = OUT / f"{label}_{stamp}.txt"
    png_path = OUT / f"{label}_{stamp}.png"
    latest_html = OUT / "latest.html"
    latest_txt = OUT / "latest.txt"

    try:
        page.wait_for_load_state("domcontentloaded", timeout=10000)
    except PlaywrightTimeoutError:
        print("Warning: page load-state timeout; saving current DOM anyway.")

    html = page.content()
    text = page.locator("body").inner_text(timeout=30000)

    html_path.write_text(html, encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")
    latest_html.write_text(html, encoding="utf-8")
    latest_txt.write_text(text, encoding="utf-8")

    screenshot_status = "not attempted"
    try:
        page.screenshot(path=str(png_path), full_page=False, timeout=5000)
        screenshot_status = f"saved {png_path.name}"
    except Exception as exc:
        screenshot_status = f"skipped {type(exc).__name__}: {exc}"

    print("")
    print("Saved snapshot:")
    print(f"- {text_path}")
    print(f"- {html_path}")
    print(f"- latest text: {latest_txt}")
    print(f"- screenshot: {screenshot_status}")
    print("")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROFILE.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE),
            headless=False,
            viewport={"width": 1440, "height": 1000},
        )

        page = context.new_page()
        page.goto(START_URL, wait_until="domcontentloaded", timeout=60000)

        print("")
        print("Investopedia browser session is open.")
        print("Log in manually if needed.")
        print("Open the game: TUDSoSe2026 BehavioralFinance.")
        print("Navigate to the portfolio/holdings page.")
        print("")
        print("Terminal commands:")
        print("- ENTER: save current page snapshot")
        print("- q + ENTER: quit and close browser")
        print("")

        while True:
            cmd = input("Command [ENTER=snapshot, q=quit]: ").strip().lower()
            if cmd in {"q", "quit", "exit"}:
                break
            save_snapshot(page, "investopedia")

        context.close()
        print("Closed browser session.")


if __name__ == "__main__":
    main()
