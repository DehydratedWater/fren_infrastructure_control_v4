"""FastAPI app for Twily's world — observe her life or visit as Vis.

Read endpoints (world/state/events/npcs/research/location) drive the UI; the two
write endpoints run a beat (autonomous, or in reaction to Vis). Served on its own
port (separate docker service), no auth — it's a private, local surface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.world import clock as world_clock
from app.world.loader import DEFAULT_PACKAGE, get_package
from app.world.models import WorldPackage
from app.world.state import WorldStateRepo

logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).parent / "web"
WORLD_ID = DEFAULT_PACKAGE


def _pkg() -> WorldPackage:
    return get_package(WORLD_ID)


def _repo() -> WorldStateRepo:
    return WorldStateRepo(WORLD_ID)


async def _serialize_state(pkg: WorldPackage, repo: WorldStateRepo) -> dict[str, Any]:
    sess = await repo.ensure_session(pkg)
    loc = pkg.location(sess["current_location_id"])
    clk = int(sess["clock_minutes"])
    present = pkg.npcs_at(loc.id) if loc else []
    neighbors = (
        [{"id": d.id, "name": d.name, "label": c.label or d.name} for c, d in pkg.neighbors(loc.id)]
        if loc else []
    )
    return {
        "world_id": WORLD_ID,
        "turn_number": int(sess["turn_count"]),
        "day_count": int(sess["day_count"]),
        "clock_minutes": clk,
        "clock_label": world_clock.clock_label(clk),
        "day_phase": world_clock.day_phase(clk),
        "visitor_present": bool(sess["visitor_present"]),
        "persona_state": sess.get("persona_state") or {},
        "location": (
            {
                "id": loc.id, "name": loc.name, "kind": loc.kind,
                "description": loc.description,
                "activities": [{"tag": a.tag, "label": a.label} for a in loc.activities],
            }
            if loc else None
        ),
        "present_npcs": [{"id": n.id, "name": n.name, "role": n.role} for n in present],
        "neighbors": neighbors,
    }


class VisitorTurn(BaseModel):
    input: str = ""


def create_app() -> FastAPI:
    app = FastAPI(title="Twily's World", docs_url=None, redoc_url=None)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_WEB_DIR / "index.html")

    @app.get("/static/{path:path}")
    async def static_files(path: str) -> FileResponse:
        target = (_WEB_DIR / path).resolve()
        if not str(target).startswith(str(_WEB_DIR.resolve())) or not target.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(target)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        try:
            sess = await _repo().ensure_session(_pkg())
            clk = int(sess["clock_minutes"])
            return {"ok": True, "clock_label": world_clock.clock_label(clk),
                    "day_count": int(sess["day_count"])}
        except Exception as exc:  # noqa: BLE001
            logger.exception("world health failed")
            return {"ok": False, "error": str(exc)}

    @app.get("/api/world")
    async def world() -> dict[str, Any]:
        pkg = _pkg()
        return {
            "id": pkg.id, "name": pkg.name, "setting": pkg.setting,
            "description": pkg.description,
            "protagonist": {
                "id": pkg.protagonist.id, "name": pkg.protagonist.name,
                "appearance": pkg.protagonist.appearance,
                "drives": pkg.protagonist.drives, "goals": pkg.protagonist.goals,
            },
            "visitor": {
                "id": pkg.visitor.id, "name": pkg.visitor.name,
                "appearance": pkg.visitor.appearance,
            },
            "locations": [
                {"id": loc.id, "name": loc.name, "kind": loc.kind, "parent_id": loc.parent_id}
                for loc in pkg.locations
            ],
            "npcs": [{"id": n.id, "name": n.name, "role": n.role} for n in pkg.npcs],
            "counts": {"locations": len(pkg.locations), "npcs": len(pkg.npcs)},
        }

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return await _serialize_state(_pkg(), _repo())

    @app.get("/api/events")
    async def events(limit: int = 60, before_id: int | None = None) -> dict[str, Any]:
        repo = _repo()
        pkg = _pkg()
        rows = await repo.recent_events(limit=min(max(limit, 1), 200), before_id=before_id)
        out = []
        for e in rows:
            loc = pkg.location(e.get("location_id")) if e.get("location_id") else None
            out.append({
                "id": int(e["id"]), "turn": int(e["turn"]), "kind": e["kind"],
                "actor": e["actor"], "content": e["content"],
                "location_id": e.get("location_id"),
                "location_name": loc.name if loc else None,
                "created_at": e["created_at"].isoformat() if e.get("created_at") else None,
            })
        oldest = out[0]["id"] if out else None
        return {"events": out, "oldest_id": oldest, "has_more": bool(out) and len(rows) >= limit}

    @app.get("/api/npcs")
    async def npcs() -> dict[str, Any]:
        pkg = _pkg()
        states = await _repo().npc_states()
        out = []
        for n in pkg.npcs:
            st = states.get(n.id, {})
            out.append({
                "id": n.id, "name": n.name, "role": n.role,
                "location_id": n.home_location_id,
                "affinity": int(st.get("affinity", n.default_affinity)),
                "last_seen_turn": int(st.get("last_seen_turn", 0)),
            })
        out.sort(key=lambda x: x["affinity"], reverse=True)
        return {"npcs": out}

    @app.get("/api/research")
    async def research(limit: int = 20) -> dict[str, Any]:
        rows = await _repo().recent_research(limit=min(max(limit, 1), 50))
        return {"research": [
            {
                "id": int(r["id"]), "turn": int(r["turn"]), "query": r["query"],
                "summary": r["summary"], "results": r.get("results") or [],
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            }
            for r in rows
        ]}

    @app.get("/api/location/{loc_id}")
    async def location(loc_id: str) -> dict[str, Any]:
        pkg = _pkg()
        loc = pkg.location(loc_id)
        if not loc:
            raise HTTPException(status_code=404, detail="location not found")
        return {
            "id": loc.id, "name": loc.name, "kind": loc.kind,
            "description": loc.description,
            "activities": [{"tag": a.tag, "label": a.label, "description": a.description}
                           for a in loc.activities],
            "neighbors": [{"id": d.id, "name": d.name, "label": c.label or d.name}
                          for c, d in pkg.neighbors(loc.id)],
            "present_npcs": [{"id": n.id, "name": n.name} for n in pkg.npcs_at(loc.id)],
        }

    @app.post("/api/turn")
    async def turn() -> JSONResponse:
        from app.world.turn import run_world_turn

        result = await run_world_turn(world_id=WORLD_ID, trigger="manual")
        if not result.get("ok"):
            return JSONResponse(status_code=503, content={"ok": False, "error": result.get("error")})
        result["state"] = await _serialize_state(_pkg(), _repo())
        return JSONResponse(content=result)

    @app.post("/api/visitor/turn")
    async def visitor_turn(body: VisitorTurn) -> JSONResponse:
        from app.world.turn import run_world_turn

        text = (body.input or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty visitor input")
        repo = _repo()
        await repo.ensure_session(_pkg())
        await repo.set_visitor_present(True)
        result = await run_world_turn(world_id=WORLD_ID, trigger="visitor", visitor_input=text)
        if not result.get("ok"):
            return JSONResponse(status_code=503, content={"ok": False, "error": result.get("error")})
        result["state"] = await _serialize_state(_pkg(), _repo())
        return JSONResponse(content=result)

    return app


app = create_app()
