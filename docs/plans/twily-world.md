# Twily's World — an autonomous roleplay life-sim

A separate subsystem that gives the Twily persona a *life*: rooms she lives,
cooks, and tinkers in; a town (**Mooring Wells**) she roams; NPCs who shape her;
and a computer she researches the real web on. She plays it in the background
(~one beat / 15 min), and her experiences feed back into her real persona memory,
so who she is on Telegram is shaped by who she's been here. You can **observe**
her life or **log in as Vis** (your avatar in her world) and chat into it.

Inspired by `/home/dw/programing/roll_play_learning`'s *shapes* (world-package
format, the pre-classify → generate → extract-world-update → persist turn loop,
the vanilla-JS observe/play UI) but built fresh and fren-native: Postgres, the
local qwen via `src.interactive`, the existing SearchAPI tool, and the
persona-memory repos. The world is invented (a modern-Ponyville blend), not an
rpl package. The original project is untouched.

## Pieces
- `backend/app/world/models.py` — authored `WorldPackage` + per-turn IO (`TURN_SCHEMA`).
- `backend/app/world/packages/twily_haven/` — the world, as modifiable YAML
  (`package.yaml`, `locations.yaml`, `npcs.yaml`, `lorebook.yaml`). See its README.
- `loader.py` — merge + validate the package (referential integrity, nav graph).
- `state.py` (+ migration `003_twily_world`) — Postgres: `world_sessions`,
  `world_events` (the life log), `world_npc_state`, `world_research`, `world_memories`.
- `prompts.py` — one structured beat plays Twily + narrator + NPCs; injects the
  town/cast, a time-of-day cue, restlessness pressure, and a computer nudge.
- `computer.py` — the research mechanic: real SearchAPI lookup → narrated reading.
- `turn.py` — the beat pipeline (assemble → generate → [research two-pass] → persist).
- `integrate.py` — promote important world memories → persona memory; her research
  → persona interests.
- `api.py` + `web/` — FastAPI app + cozy observe / visit-as-Vis UI.

## Running it
- **Service:** `world` in docker-compose, `python -m app world`, host port **8091**
  (container 8000). Open http://localhost:8091.
- **Background beats:** scheduler cron `world_turn` (`scripts/world_turn.py`),
  `*/15 7-23` — silent (script jobs don't message the user).
- **Persona promotion:** cron `world_promote` daily 02:00 (`--promote-only`).
- **Migrations:** the `bot` service runs `alembic upgrade head` at boot (applies 003).

## Dev tools
- `python scripts/world_playtest.py --turns 20 --reset --visitor-every 8 \
    --world-id twily_haven_ptN --out /tmp/pt.md` — exercise the world on an
  ISOLATED throwaway session (never touches the live `twily_haven` session) and
  dump a transcript + engagement stats (moves, locations, NPCs, researches, memories).
- `python scripts/world_turn.py [--promote|--promote-only]` — one beat (+ promote).

## Editing the world
Hand-edit the YAML under `packages/twily_haven/`. The loader validates on load
(every id must resolve; the nav graph must be connected from the start location),
so a typo fails loudly. Add a new world by creating `packages/<id>/` with the same
four files and pointing the service/cron at that `world_id`.

## Engagement notes (from play-test critics)
The first 18-beat play-test scored engagement 4/10: she looped at one workbench,
met none of the 8 NPCs, never used the computer. The prompt now (a) names the
whole cast + where they are as reasons to leave home, (b) surfaces nearby people
when she's solo, (c) gives a time-of-day rhythm, (d) applies restlessness pressure
after N beats in one place, (e) nudges the computer on unknowns, and (f) caps the
verbal/sensory tics so they don't leak into her real persona. Re-test before any
content change to the world; the harness is the regression check.
