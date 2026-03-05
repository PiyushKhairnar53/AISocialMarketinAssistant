from __future__ import annotations

from typing import Optional

from langchain_community.tools.tavily_search import TavilySearchResults


def get_trend_search_tool(tavily_api_key: Optional[str]) -> TavilySearchResults:
    """
    Create a TavilySearchResults tool configured for marketing trend research.

    The API key is passed explicitly so we don't depend on global env.
    """
    if not tavily_api_key:
        # Tavily will attempt to fall back to environment variables.
        # We keep this function simple and let higher layers handle user messaging.
        return TavilySearchResults(
            k=5,
            search_depth="advanced",
        )

    return TavilySearchResults(
        k=5,
        search_depth="advanced",
        tavily_api_key=tavily_api_key,
    )


__all__ = ["get_trend_search_tool"]

