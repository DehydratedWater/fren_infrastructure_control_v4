"""FastAPI monitoring dashboard for fren_infrastructure_control_v4.

Read-only server-rendered (Jinja2) dashboard over v4's live data — proactive
messages, agent runs, the context the agents are fed, the chat, and a health
strip. HTMX polls the partial fragments every ~10s for auto-refresh; full pages
embed the same partials for the first paint and graceful no-JS fallback.

Run with: ``python -m app web`` → uvicorn on 0.0.0.0:8000.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.web import data

_HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))


def create_app() -> FastAPI:
    app = FastAPI(title="fren v4 dashboard", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

    def _render(request: Request, template: str, ctx: dict[str, Any]) -> HTMLResponse:
        # Starlette's current signature is (request, name, context); passing the
        # request positionally avoids the deprecated (name, context-with-request)
        # form that mis-parses the context dict as template globals.
        return templates.TemplateResponse(request, template, ctx)

    # ── full page ──────────────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        ctx = {
            "health": await data.health(),
            "persona": await data.recent_persona_responses(),
            "runs": await data.recent_runs(),
            "digest": await data.conversation_digest(),
            "monologue": await data.inner_monologue(),
            "emotional": await data.emotional_state(),
            "blocks": await data.recent_activity_blocks(),
            "chat": await data.recent_chat(),
        }
        return _render(request, "index.html", ctx)

    # ── HTMX partial fragments (auto-refreshed) ──────────────────────────────
    @app.get("/partials/health", response_class=HTMLResponse)
    async def partial_health(request: Request) -> HTMLResponse:
        return _render(request, "partials/health.html", {"health": await data.health()})

    @app.get("/partials/proactive", response_class=HTMLResponse)
    async def partial_proactive(request: Request) -> HTMLResponse:
        return _render(
            request, "partials/proactive.html",
            {"persona": await data.recent_persona_responses()},
        )

    @app.get("/partials/runs", response_class=HTMLResponse)
    async def partial_runs(request: Request) -> HTMLResponse:
        return _render(request, "partials/runs.html", {"runs": await data.recent_runs()})

    @app.get("/partials/context", response_class=HTMLResponse)
    async def partial_context(request: Request) -> HTMLResponse:
        return _render(
            request, "partials/context.html",
            {
                "digest": await data.conversation_digest(),
                "monologue": await data.inner_monologue(),
                "emotional": await data.emotional_state(),
                "blocks": await data.recent_activity_blocks(),
            },
        )

    @app.get("/partials/chat", response_class=HTMLResponse)
    async def partial_chat(request: Request) -> HTMLResponse:
        return _render(request, "partials/chat.html", {"chat": await data.recent_chat()})

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": await data.db_ok()}

    return app


app = create_app()
