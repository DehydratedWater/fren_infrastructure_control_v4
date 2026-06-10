#!/bin/bash
# Run the agent autoloop against an ISOLATED COPY of the prod DB (<db>_autoloop),
# NEVER the live database. Optimization executes real agents whose tools write
# todos/events/artifacts — running against prod would contaminate the real dataset.
#
#   ./run_autoloop.sh --refresh --agent goals/nudge_strategist   # recopy prod data, then run
#   ./run_autoloop.sh --agent support/daily_briefer              # reuse existing copy
#   ./run_autoloop.sh --proactive-probes                         # tune the proactive
#       agents against the context-signal probe suite (variety / anti-repetition /
#       grounded / skip) in backend/app/agents/proactive_probes.py
set -euo pipefail
cd "$(dirname "$0")/backend"
set -a; source ../.env 2>/dev/null || true; set +a

DB_PORT="${FREN_DB_PORT:-5453}"
PROD_DB="${POSTGRES_DB:-frenv4}"
LOOP_DB="${PROD_DB}_autoloop"
DBC=fren-v4-db-1

if [ "${1:-}" = "--refresh" ]; then
  shift
  echo "[autoloop] refreshing $LOOP_DB from $PROD_DB (prod is only READ) ..."
  docker exec "$DBC" sh -c "dropdb -U $POSTGRES_USER --if-exists $LOOP_DB && createdb -U $POSTGRES_USER $LOOP_DB && psql -U $POSTGRES_USER -d $LOOP_DB -qc 'CREATE EXTENSION IF NOT EXISTS vector;' && pg_dump -U $POSTGRES_USER --no-owner --no-privileges $PROD_DB | psql -U $POSTGRES_USER -d $LOOP_DB -q"
fi

export DATABASE_URL="postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${DB_PORT}/${LOOP_DB}"
export PYTHONPATH="/home/dw/programing/OpenCodeCompilerV2:$PWD"

# SANDBOX: autoloop agents run on the HOST with real tools. Capture tools
# (scrot/X attach, ffmpeg /dev/video*) hard-locked the machine on 2026-06-10
# — kill-switch them, mark the run as autoloop for other side-effect guards,
# and strip any display access so nothing can attach to the user's session.
export FREN_DISABLE_CAPTURE=1
export FREN_AUTOLOOP=1
unset DISPLAY WAYLAND_DISPLAY
# Poison X auth outright: even if a child process re-derives DISPLAY, the
# connection fails authentication cleanly instead of attaching to the session.
export XAUTHORITY=/dev/null

echo "[autoloop] DATABASE_URL -> $LOOP_DB  (prod '$PROD_DB' is untouched)"
exec /home/dw/programing/OpenCodeCompilerV2/.venv/bin/python -m app improve "$@"
