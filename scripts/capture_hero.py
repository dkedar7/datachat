"""Capture the README hero from the live app via Playwright.

Loads a public dataset by URL, opens the Data Analyst, asks for a chart, and
screenshots the rendered result. Run:

    .venv/Scripts/python.exe scripts/capture_hero.py [url] [out]
"""

import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://datachat.fly.dev/"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/hero.png"
DATASET = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
ASK = "Scatter plot of tip vs total bill, colored by day, with an OLS trendline."


def fill_url(page, url):
    for sel in ("#dataset_url input", "#dataset_url", "input#dataset_url"):
        try:
            page.fill(sel, url, timeout=4000)
            return
        except Exception:
            continue
    raise RuntimeError("could not find the dataset URL input")


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(URL, wait_until="load", timeout=60000)
    page.wait_for_selector("#dataset_url", timeout=30000)
    fill_url(page, DATASET)
    try:                                            # populate the preview + summary
        page.click("#submit_inputs", timeout=5000)
        page.wait_for_timeout(6000)
    except Exception:
        pass
    # The analyst chat is stacked in the left sidebar (always visible) — no toggle.
    page.wait_for_selector("#chat-input", timeout=15000)
    page.fill("#chat-input", ASK)
    page.press("#chat-input", "Enter")
    page.wait_for_timeout(26000)                    # code + sandbox + render
    try:                                            # bring the rendered chart into view
        page.locator("#chat-messages .js-plotly-plot").first.scroll_into_view_if_needed(timeout=5000)
        page.wait_for_timeout(1500)
    except Exception:
        pass
    page.screenshot(path=OUT)
    browser.close()

print("saved", OUT)
