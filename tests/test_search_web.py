from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_search_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "search" / "web_search.py"
    spec = importlib.util.spec_from_file_location("web_search", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["web_search"] = module
    spec.loader.exec_module(module)
    return module


def test_format_search_results_empty():
    web_search = load_search_module()

    assert web_search.format_search_results([]) == "No search evidence was retrieved."


def test_format_search_results_includes_title_url_and_snippet():
    web_search = load_search_module()
    result = web_search.SearchResult(
        title="Furlong",
        url="https://example.com/furlong",
        snippet="A furlong is 660 feet.",
    )

    formatted = web_search.format_search_results([result])

    assert "Furlong" in formatted
    assert "https://example.com/furlong" in formatted
    assert "660 feet" in formatted


def test_tavily_search_parses_results(monkeypatch):
    web_search = load_search_module()
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "title": "Cubic metre",
                        "url": "https://example.com/cubic-metre",
                        "content": "One cubic metre is equal to 1000 litres.",
                    }
                ]
            }

    def fake_post(url, *, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(web_search.requests, "post", fake_post)

    results = web_search.search_web("how many liters are in one cubic meter", max_results=1)

    assert calls[0]["url"] == "https://api.tavily.com/search"
    assert calls[0]["headers"]["Authorization"] == "Bearer tvly-test"
    assert calls[0]["json"]["search_depth"] == "basic"
    assert results == [
        web_search.SearchResult(
            title="Cubic metre",
            url="https://example.com/cubic-metre",
            snippet="One cubic metre is equal to 1000 litres.",
        )
    ]
