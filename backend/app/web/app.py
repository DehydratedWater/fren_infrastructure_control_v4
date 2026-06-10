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

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
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
            "images": await data.recent_images(),
            "mind": await data.mind(),
            "life": await data.life(),
        }
        return _render(request, "index.html", ctx)

    # ── traces (LLM audit log: persona_prose_trace artifacts) ────────────────
    @app.get("/traces", response_class=HTMLResponse)
    async def traces_page(request: Request) -> HTMLResponse:
        return _render(request, "traces.html", {"traces": await data.prose_traces()})

    @app.get("/traces/{run_id}", response_class=HTMLResponse)
    async def trace_detail(request: Request, run_id: str) -> HTMLResponse:
        detail = await data.prose_trace_detail(run_id)
        return _render(
            request, "trace_detail.html",
            {"run_id": run_id, "trace": detail, "missing": detail is None},
        )

    # ── events (extracted life events: timeline + category charts) ──────────
    @app.get("/events", response_class=HTMLResponse)
    async def events_page(request: Request, category: str = "all") -> HTMLResponse:
        # `category` is validated inside data.events_page against the
        # categories actually present in the DB; unknown → "all".
        return _render(request, "events.html", {"events": await data.events_page(category)})

    # ── artifacts (context_cache gallery, read-only) ─────────────────────────
    @app.get("/artifacts", response_class=HTMLResponse)
    async def artifacts_page(
        request: Request,
        atype: str = Query("all", alias="type"),
        q: str = "",
    ) -> HTMLResponse:
        # `type` and `q` are validated/normalised inside data.artifacts_page.
        return _render(
            request, "artifacts.html",
            {"artifacts": await data.artifacts_page(atype, q)},
        )

    # ── run detail (view the session) ────────────────────────────────────────
    @app.get("/run/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: str) -> HTMLResponse:
        detail = await data.run_detail(run_id)
        if detail is None:
            return _render(request, "run_detail.html", {"run_id": run_id, "missing": True})
        return _render(request, "run_detail.html", {"run_id": run_id, **detail})

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

    @app.get("/partials/images", response_class=HTMLResponse)
    async def partial_images(request: Request, kind: str = "all") -> HTMLResponse:
        # `kind` is validated server-side: anything unknown collapses to "all".
        kind = data.normalize_image_filter(kind)
        return _render(
            request, "partials/images.html",
            {"images": await data.recent_images(kind=kind)},
        )

    @app.get("/partials/mind", response_class=HTMLResponse)
    async def partial_mind(request: Request) -> HTMLResponse:
        return _render(request, "partials/mind.html", {"mind": await data.mind()})

    @app.get("/partials/traces", response_class=HTMLResponse)
    async def partial_traces(request: Request) -> HTMLResponse:
        return _render(request, "partials/traces.html", {"traces": await data.prose_traces()})

    @app.get("/partials/life", response_class=HTMLResponse)
    async def partial_life(request: Request) -> HTMLResponse:
        return _render(request, "partials/life.html", {"life": await data.life()})

    @app.get("/partials/events", response_class=HTMLResponse)
    async def partial_events(request: Request, category: str = "all") -> HTMLResponse:
        return _render(
            request, "partials/events.html",
            {"events": await data.events_page(category)},
        )

    @app.get("/partials/artifacts", response_class=HTMLResponse)
    async def partial_artifacts(
        request: Request,
        atype: str = Query("all", alias="type"),
        q: str = "",
    ) -> HTMLResponse:
        return _render(
            request, "partials/artifacts.html",
            {"artifacts": await data.artifacts_page(atype, q)},
        )

    # ── media bytes (read-only, path-traversal-safe) ─────────────────────────
    @app.get("/media/{kind}/{name}")
    async def media(kind: str, name: str) -> FileResponse:
        """Serve a single image from an allowed media dir, safely.

        ``data.safe_media_path`` enforces the whole policy: ``kind`` must be a
        known media kind, ``name`` must be a bare image filename (no traversal,
        no separators, image extension only) that resolves to a direct child of
        ``settings.data_dir``/<kind>. Anything else → 404 (we never reveal which
        rule failed, and never touch the filesystem outside the allowed dir).
        """
        path = data.safe_media_path(kind, name)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": await data.db_ok()}

    return app


app = create_app()
