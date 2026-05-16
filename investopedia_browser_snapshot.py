"""Investopedia browser-session snapshot.

Read-only. Does not submit orders.

You log in manually in the opened browser window.
The script then saves the portfolio page HTML/text for parser development.

Uses a persistent browser profile:
.local/investopedia_browser_profile
"""

from __future__ import annotations

from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "investopedia_snapshot"
PROFILE = ROOT / ".local" / "investopedia_browser_profile"

PORTFOLIO_URL = "https://www.investopedia.com/simulator/portfolio"


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
        page.goto(PORTFOLIO_URL, wait_until="domcontentloaded", timeout=60000)

        print("")
        print("Browser opened.")
        print("Log in manually with your email/code if needed.")
        print("Navigate to the portfolio page if needed:")
        print(PORTFOLIO_URL)
        print("")
        print("When the portfolio is visible, come back here and press ENTER.")
        input("Press ENTER after portfolio is visible... ")

        try:
            page.wait_for_load_state("domcontentloaded", timeout=30000)
        except PlaywrightTimeoutError:
            print("Warning: load_state timeout; continuing with current page.")

        html = page.content()
        text = page.locator("body").inner_text(timeout=30000)

        html_path = OUT / "portfolio_page.html"
        text_path = OUT / "portfolio_page.txt"
        png_path = OUT / "portfolio_page.png"

        html_path.write_text(html, encoding="utf-8")
        text_path.write_text(text, encoding="utf-8")

        try:
            page.screenshot(path=str(png_path), full_page=False, timeout=10000)
            screenshot_status = f"saved: {png_path}"
        except PlaywrightTimeoutError:
            screenshot_status = "skipped: screenshot timeout"
        except Exception as exc:
            screenshot_status = f"skipped: {type(exc).__name__}: {exc}"

        print("")
        print("Saved snapshot files:")
        print(f"- {html_path}")
        print(f"- {text_path}")
        print(f"- screenshot: {screenshot_status}")

        context.close()


if __name__ == "__main__":
    main()
