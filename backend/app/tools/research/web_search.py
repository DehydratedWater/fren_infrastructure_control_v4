"""Web Search — Google web search via SearchAPI.io."""

from __future__ import annotations

import asyncio

import httpx
from src import ScriptTool
from pydantic import BaseModel, Field

_SEARCHAPI_BASE = "https://www.searchapi.io/api/v1/search"


class Input(BaseModel):
    command: str = Field(description="search")
    query: str = Field(default="", description="Search query")
    max_results: int = Field(default=10, description="Max results to return (default 10)")
    location: str = Field(default="", description="Location for localized results (e.g. 'Poland')")


class Output(BaseModel):
    success: bool = True
    error: str = ""
    items: list[dict] = Field(default_factory=list)
    count: int = 0
    query: str = ""


class WebSearchTool(ScriptTool[Input, Output]):
    name = "web-search"
    description = "Search the web via Google using SearchAPI.io"
    input_model = Input
    output_model = Output

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.settings import get_settings

        api_key = get_settings().searchapi_key
        if not api_key:
            return Output(success=False, error="SEARCHAPI_KEY not configured")

        if inp.command == "search":
            return await self._search(inp.query, api_key, inp.max_results, inp.location)

        return Output(success=False, error=f"Unknown command: {inp.command}")

    async def _search(self, query: str, api_key: str, max_results: int, location: str) -> Output:
        if not query:
            return Output(success=False, error="No query provided")

        params: dict = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": min(max_results, 20),
        }
        if location:
            params["location"] = location

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(_SEARCHAPI_BASE, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            return Output(success=False, error=f"Search request failed: {e}")

        # Extract organic results
        results = []
        for item in data.get("organic_results", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "position": item.get("position", 0),
                }
            )

        # Include answer box if present
        answer_box = data.get("answer_box")
        if answer_box:
            results.insert(
                0,
                {
                    "title": answer_box.get("title", "Answer"),
                    "link": answer_box.get("link", ""),
                    "snippet": answer_box.get("answer", answer_box.get("snippet", "")),
                    "position": 0,
                    "type": "answer_box",
                },
            )

        return Output(success=True, items=results, count=len(results), query=query)


if __name__ == "__main__":
    WebSearchTool.run()
