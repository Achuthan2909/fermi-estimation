"""Small web-search adapter for verifier experiments.

The default provider uses DuckDuckGo's instant-answer endpoint because it needs
no extra dependency or API key. Set ``SEARCH_PROVIDER=tavily`` with
``TAVILY_API_KEY`` for better web snippets, or ``SEARCH_PROVIDER=brave`` with
``BRAVE_SEARCH_API_KEY`` to use Brave Web Search.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any

import requests


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _parse_duckduckgo_topics(topics: list[Any], max_results: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    for topic in topics:
        if len(results) >= max_results:
            break
        if not isinstance(topic, dict):
            continue
        nested = topic.get("Topics")
        if isinstance(nested, list):
            results.extend(_parse_duckduckgo_topics(nested, max_results - len(results)))
            continue
        text = _clean(topic.get("Text"))
        url = _clean(topic.get("FirstURL"))
        if text or url:
            title = text.split(" - ", 1)[0][:120] if text else url
            results.append(SearchResult(title=title, url=url, snippet=text))
    return results


def _duckduckgo_search(query: str, max_results: int, timeout_s: float) -> list[SearchResult]:
    response = requests.get(
        "https://api.duckduckgo.com/",
        params={
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    results: list[SearchResult] = []
    abstract = _clean(data.get("AbstractText"))
    abstract_url = _clean(data.get("AbstractURL"))
    heading = _clean(data.get("Heading")) or "DuckDuckGo instant answer"
    if abstract:
        results.append(SearchResult(title=heading, url=abstract_url, snippet=abstract))
    related = data.get("RelatedTopics")
    if isinstance(related, list) and len(results) < max_results:
        results.extend(_parse_duckduckgo_topics(related, max_results - len(results)))
    return results[:max_results]


def _brave_search(query: str, max_results: int, timeout_s: float) -> list[SearchResult]:
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("BRAVE_SEARCH_API_KEY is required when SEARCH_PROVIDER=brave")
    response = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": max_results},
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    web_results = data.get("web", {}).get("results", [])
    results: list[SearchResult] = []
    for item in web_results[:max_results]:
        if not isinstance(item, dict):
            continue
        results.append(
            SearchResult(
                title=_clean(item.get("title")),
                url=_clean(item.get("url")),
                snippet=_clean(item.get("description")),
            )
        )
    return results


def _tavily_search(query: str, max_results: int, timeout_s: float) -> list[SearchResult]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is required when SEARCH_PROVIDER=tavily")
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "query": query,
            "search_depth": os.environ.get("TAVILY_SEARCH_DEPTH", "basic"),
            "topic": "general",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    web_results = data.get("results", [])
    results: list[SearchResult] = []
    for item in web_results[:max_results]:
        if not isinstance(item, dict):
            continue
        results.append(
            SearchResult(
                title=_clean(item.get("title")),
                url=_clean(item.get("url")),
                snippet=_clean(item.get("content")),
            )
        )
    return results


def search_web(query: str, max_results: int = 5, timeout_s: float = 10.0) -> list[SearchResult]:
    """Return web snippets for a query.

    Set ``SEARCH_PROVIDER=none`` during tests or offline runs to disable network
    search while preserving the verifier pipeline.
    """
    provider = os.environ.get("SEARCH_PROVIDER", "duckduckgo").strip().lower()
    if provider in {"", "none", "disabled"}:
        return []
    if provider == "brave":
        return _brave_search(query, max_results=max_results, timeout_s=timeout_s)
    if provider == "tavily":
        return _tavily_search(query, max_results=max_results, timeout_s=timeout_s)
    if provider == "duckduckgo":
        return _duckduckgo_search(query, max_results=max_results, timeout_s=timeout_s)
    raise ValueError(f"unknown SEARCH_PROVIDER={provider!r}")


def format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "No search evidence was retrieved."
    lines = []
    for i, result in enumerate(results, start=1):
        lines.append(f"[{i}] {result.title}\nURL: {result.url}\nSnippet: {result.snippet}")
    return "\n\n".join(lines)
