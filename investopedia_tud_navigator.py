"""Navigate Investopedia to the TUD game portfolio.

Read-only. Does not submit orders.

Flow:
1. Open persistent Chromium session.
2. If login appears, user logs in manually.
3. Open Games.
4. Click My Games.
5. Select TUDSoSe2026 BehavioralFinance.
6. Click GO TO GAME.
7. Click Portfolio.
8. Save snapshot for parser development.

No orders are submitted.
"""

from __future__ import annotations

from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


ROOT = Path(__file__).resolve().parent
PROFILE = ROOT / ".local" / "investopedia_browser_profile"
OUT = ROOT / "outputs" / "investopedia_tud_navigator"

GAME_NAME = "TUDSoSe2026 BehavioralFinance"
BASE = "https://www.investopedia.com"
GAMES_URL = f"{BASE}/simulator/games"
PORTFOLIO_URL = f"{BASE}/simulator/portfolio"


def save_snapshot(page, label: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    html_path = OUT / f"{label}.html"
    txt_path = OUT / f"{label}.txt"
    png_path = OUT / f"{label}.png"

    try:
        page.wait_for_load_state("domcontentloaded", timeout=10000)
    except PlaywrightTimeoutError:
        print("Warning: load-state timeout; saving current DOM anyway.")

    html = page.content()
    text = page.locator("body").inner_text(timeout=30000)

    html_path.write_text(html, encoding="utf-8")
    txt_path.write_text(text, encoding="utf-8")

    try:
        page.screenshot(path=str(png_path), full_page=False, timeout=7000)
        screenshot_status = f"saved {png_path}"
    except Exception as exc:
        screenshot_status = f"skipped screenshot: {type(exc).__name__}: {exc}"

    print("")
    print(f"Saved snapshot: {label}")
    print(f"- {txt_path}")
    print(f"- {html_path}")
    print(f"- {screenshot_status}")


def click_if_visible(page, text: str, *, timeout: int = 8000) -> bool:
    try:
        loc = page.get_by_text(text, exact=True).first
        loc.wait_for(timeout=timeout)
        loc.click(timeout=timeout)
        return True
    except Exception:
        return False


def try_select_tud_game(page) -> bool:
    """Try several DOM strategies to click GO TO GAME for the TUD game."""

    # Strategy 1: use XPath around the game name and nearby GO TO GAME.
    xpaths = [
        (
            "//*[contains(normalize-space(.), 'TUDSoSe2026 BehavioralFinance')]"
            "/ancestor::*[contains(., 'GO TO GAME')][1]"
            "//*[normalize-space()='GO TO GAME']"
        ),
        (
            "//*[contains(normalize-space(.), 'TUDSoSe2026 BehavioralFinance')]"
            "/following::*[normalize-space()='GO TO GAME'][1]"
        ),
    ]

    for xp in xpaths:
        try:
            loc = page.locator(f"xpath={xp}").first
            loc.wait_for(timeout=8000)
            loc.click(timeout=8000)
            return True
        except Exception:
            pass

    # Strategy 2: fallback: first GO TO GAME, because in your snapshot TUD was first.
    try:
        page.get_by_text("GO TO GAME", exact=True).first.click(timeout=8000)
        return True
    except Exception:
        return False


def main() -> None:
    PROFILE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE),
            headless=False,
            viewport={"width": 1440, "height": 1000},
        )
        page = context.new_page()

        print("")
        print("Opening Investopedia with persistent local browser profile.")
        print("If login appears, log in manually with your email code.")
        print("Do not paste terminal commands while this script is waiting.")
        print("")

        page.goto(GAMES_URL, wait_until="domcontentloaded", timeout=60000)

        print("If login is required, finish it in the browser.")
        input("Press ENTER once you are logged in and the Investopedia page is visible... ")

        # Go to Games again after login, because redirects may have changed page.
        page.goto(GAMES_URL, wait_until="domcontentloaded", timeout=60000)

        # Click My Games if visible.
        clicked_my_games = click_if_visible(page, "My Games", timeout=10000)
        print(f"clicked_my_games={clicked_my_games}")

        try:
            page.get_by_text(GAME_NAME, exact=True).wait_for(timeout=15000)
            print(f"Found game: {GAME_NAME}")
        except Exception:
            print("")
            print("Could not auto-detect TUD game.")
            print("Please click Games → My Games manually and make sure TUDSoSe2026 BehavioralFinance is visible.")
            input("Press ENTER when the TUD game card is visible... ")

        selected = try_select_tud_game(page)
        print(f"selected_tud_game={selected}")

        if not selected:
            print("")
            print("Could not auto-click GO TO GAME.")
            print("Please click GO TO GAME for TUDSoSe2026 BehavioralFinance manually.")
            input("Press ENTER after you entered the TUD game... ")

        try:
            page.wait_for_load_state("domcontentloaded", timeout=30000)
        except PlaywrightTimeoutError:
            print("Warning: load after GO TO GAME timed out; continuing.")

        # Once game context is selected, open portfolio route.
        page.goto(PORTFOLIO_URL, wait_until="domcontentloaded", timeout=60000)

        print("")
        print("Now check the browser.")
        print("If it still shows Overview / No Portfolio Information, click Get Started or Portfolio/Trade manually.")
        print("When the TUD portfolio/cash/holdings page is visible, press ENTER.")
        input("Press ENTER to save final TUD portfolio snapshot... ")

        save_snapshot(page, "tud_portfolio")
        context.close()


if __name__ == "__main__":
    main()
