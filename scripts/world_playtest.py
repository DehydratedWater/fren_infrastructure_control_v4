#!/usr/bin/env python3
"""Play-test Twily's world — a dev harness, not a production job.

Runs many beats against an ISOLATED throwaway session (a separate world_id over
the same authored package) so the real twily_haven session stays untouched, then
dumps a clean chronological transcript + summary stats for evaluation. Use it to
check the world is actually engaging before handing it to Twily.

Usage:
    python scripts/world_playtest.py --turns 18 --reset --visitor-every 6 \
        --out /tmp/twily_playtest.md
"""

from __future__ import annotations

import argparse
import asyncio
import sys

# believable visitor (Vis) injections, rotated through during a play-test
_VISITOR_LINES = [
    "Vis lets themself in, shaking rain off a coat, and asks what she's working on.",
    "Vis peers over her shoulder: 'okay, explain the daylight thing to me like I'm five.'",
    "Vis sets down two pastries from the café and says 'Maro says hi, and that you owe him a mug.'",
    "Vis flops onto the nearest seat: 'take a break. tell me something you learned today.'",
    "Vis asks, gently, 'when's the last time you actually ate something?'",
]


async def _reset(world_id: str) -> None:
    from app.db.session import execute_sql, get_async_session

    async with get_async_session() as s:
        for tbl in ("world_events", "world_research", "world_memories",
                    "world_npc_state", "world_sessions"):
            await execute_sql(s, f"DELETE FROM {tbl} WHERE world_id = :w", {"w": world_id})


async def _dump(world_id: str, package_id: str, out_path: str | None) -> str:
    from app.db.session import fetch_all, get_async_session
    from app.world.loader import get_package

    pkg = get_package(package_id)
    async with get_async_session() as s:
        evs = await fetch_all(
            s, "SELECT turn, kind, actor, content, location_id FROM world_events "
               "WHERE world_id = :w ORDER BY id", {"w": world_id})
        research = await fetch_all(
            s, "SELECT turn, query, summary FROM world_research WHERE world_id = :w ORDER BY id",
            {"w": world_id})
        mems = await fetch_all(
            s, "SELECT turn, content, importance FROM world_memories WHERE world_id = :w ORDER BY id",
            {"w": world_id})

    lines: list[str] = [f"# Play-test transcript — {pkg.name} ({world_id})\n"]
    last_loc = None
    for e in evs:
        loc = pkg.location(e["location_id"]) if e["location_id"] else None
        if loc and loc.id != last_loc:
            lines.append(f"\n_— {loc.name} —_\n")
            last_loc = loc.id
        actor = e["actor"]
        kind = e["kind"]
        c = e["content"]
        if kind == "narration":
            lines.append(f"> {c}")
        elif kind == "action":
            lines.append(f"**Twily:** {c}")
        elif kind == "speech":
            lines.append(f"**Twily (aloud):** “{c}”")
        elif kind == "npc":
            name = pkg.npc(actor).name if pkg.npc(actor) else actor
            lines.append(f"**{name}:** “{c}”")
        elif kind == "visitor":
            lines.append(f"**Vis:** {c}")
        elif kind == "research":
            lines.append(f"_[computer]_ {c}")
        elif kind == "move":
            lines.append(f"_[moves {c}]_")
        else:
            lines.append(f"_{c}_")

    # stats
    moves = sum(1 for e in evs if e["kind"] == "move")
    locs = {e["location_id"] for e in evs if e["location_id"]}
    npcs = {e["actor"] for e in evs if e["kind"] == "npc"}
    lines.append("\n\n---\n## Stats")
    lines.append(f"- events: {len(evs)} · moves: {moves} · locations visited: {len(locs)} · "
                 f"NPCs who spoke: {len(npcs)} · researches: {len(research)} · memories: {len(mems)}")
    if research:
        lines.append("- searched: " + "; ".join(r["query"] for r in research))
    if mems:
        lines.append("\n### Distilled memories")
        for m in mems:
            lines.append(f"- ({m['importance']:.2f}) {m['content']}")

    text = "\n".join(lines)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


async def _run(args) -> int:
    from app.db.session import set_null_pool

    set_null_pool(True)
    from app.world.turn import run_world_turn

    if args.reset:
        await _reset(args.world_id)
        print(f"[playtest] reset {args.world_id}", file=sys.stderr)

    vi = 0
    for i in range(1, args.turns + 1):
        visitor = None
        if args.visitor_every and i % args.visitor_every == 0:
            visitor = _VISITOR_LINES[vi % len(_VISITOR_LINES)]
            vi += 1
        r = await run_world_turn(
            world_id=args.world_id, package_id=args.package,
            trigger="visitor" if visitor else "auto", visitor_input=visitor,
        )
        tag = "VIS" if visitor else "   "
        if r.get("ok"):
            print(f"[playtest] {tag} turn {i}/{args.turns} "
                  f"clock={r.get('clock_label')} moved={r.get('moved')} "
                  f"researched={r.get('researched')}", file=sys.stderr)
        else:
            print(f"[playtest] turn {i} FAILED: {r.get('error')}", file=sys.stderr)

    text = await _dump(args.world_id, args.package, args.out)
    if not args.out:
        print(text)
    else:
        print(f"[playtest] transcript → {args.out}", file=sys.stderr)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Play-test Twily's world (isolated session)")
    p.add_argument("--turns", type=int, default=16)
    p.add_argument("--world-id", default="twily_haven_playtest")
    p.add_argument("--package", default="twily_haven")
    p.add_argument("--visitor-every", type=int, default=0, help="inject a Vis line every K turns (0=off)")
    p.add_argument("--reset", action="store_true", help="clear this world_id's rows first")
    p.add_argument("--out", default="", help="write the transcript to this path")
    args = p.parse_args()
    args.out = args.out or None
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
