"""Inspect Investopedia snapshot HTML for links/buttons.

Read-only diagnostic.
"""

from __future__ import annotations

from pathlib import Path
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
HTML_PATHS = [
    ROOT / "outputs" / "investopedia_session_keeper" / "latest.html",
    ROOT / "outputs" / "investopedia_tud_game_snapshot" / "tud_game_page.html",
]


def main() -> None:
    html_path = next((p for p in HTML_PATHS if p.exists()), None)
    if html_path is None:
        raise SystemExit("No snapshot HTML found.")

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    print(f"HTML: {html_path}")
    print("")
    print("=== LINKS containing simulator / portfolio / trade / game / holdings ===")

    seen = set()
    for a in soup.find_all("a"):
        text = " ".join(a.get_text(" ", strip=True).split())
        href = a.get("href", "")
        blob = f"{text} {href}".lower()
        if any(k in blob for k in ["simulator", "portfolio", "trade", "game", "holding", "position"]):
            item = (text, href)
            if item not in seen:
                seen.add(item)
                print(f"- text={text!r} href={href!r}")

    print("")
    print("=== BUTTONS / clickable text candidates ===")
    seen_btn = set()
    for tag in soup.find_all(["button", "div", "span"]):
        text = " ".join(tag.get_text(" ", strip=True).split())
        if not text:
            continue
        low = text.lower()
        if any(k in low for k in ["get started", "trade", "portfolio", "holdings", "positions", "go to game", "overview"]):
            if text not in seen_btn:
                seen_btn.add(text)
                print(f"- {text!r}")


if __name__ == "__main__":
    main()
