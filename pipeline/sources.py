"""
pipeline/sources.py
-------------------
Bonus: Sources retrieval via DuckDuckGo HTML scraping (no API key).

Retrieves real URLs relevant to the topic and injects them into the
generation prompt so the LLM can cite plausible, on-topic sources.
"""

import time
from typing import List
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from .logger import logger

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}

_DDG_URL = "https://html.duckduckgo.com/html/?q={query}"
_TIMEOUT = 8


def fetch_sources(topic: str, language: str = "en", max_results: int = 5) -> List[str]:
    """
    Search DuckDuckGo for URLs relevant to *topic*.

    Returns a list of up to max_results URL strings.
    Falls back to an empty list on any network or parsing error.
    """
    lang_suffix = "lang:fr " if language == "fr" else ""
    query = quote_plus(lang_suffix + topic)
    url = _DDG_URL.format(query=query)

    try:
        response = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("[sources] DuckDuckGo request failed: %s", exc)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    links: List[str] = []

    for result in soup.select(".result__url"):
        href = result.get_text(strip=True)
        if href and href.startswith("http"):
            links.append(href)
            if len(links) >= max_results:
                break

    # Fallback: look for result__a anchors
    if not links:
        for anchor in soup.select(".result__a"):
            href = anchor.get("href", "")
            if href.startswith("http"):
                links.append(href)
                if len(links) >= max_results:
                    break

    # Brief courtesy delay
    time.sleep(0.5)

    logger.info("[sources] Retrieved %d URLs for topic: %s", len(links), topic)
    return links
