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

WITH_CAPTURE=0
if [ "${1:-}" = "--with-capture" ]; then
  shift
  WITH_CAPTURE=1
fi

if [ "${1:-}" = "--refresh" ]; then
  shift
  echo "[autoloop] refreshing $LOOP_DB from $PROD_DB (prod is only READ) ..."
  docker exec "$DBC" sh -c "dropdb -U $POSTGRES_USER --if-exists $LOOP_DB && createdb -U $POSTGRES_USER $LOOP_DB && psql -U $POSTGRES_USER -d $LOOP_DB -qc 'CREATE EXTENSION IF NOT EXISTS vector;' && pg_dump -U $POSTGRES_USER --no-owner --no-privileges $PROD_DB | psql -U $POSTGRES_USER -d $LOOP_DB -q"
fi

export DATABASE_URL="postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${DB_PORT}/${LOOP_DB}"
export PYTHONPATH="/home/dw/programing/OpenCodeCompilerV2:$PWD"

# SANDBOX: autoloop agents run on the HOST with real tools. Capture tools
# (scrot/X attach, ffmpeg /dev/video*) coincided with the 2026-06-10 host
# hard-locks (root cause since traced to a degrading PCIe chipset link on
# 0000:20:01.1 — capture load raises the dice-roll rate, it isn't the disease)
# — default: kill-switch them, mark the run as autoloop for other side-effect
# guards, and strip any display access so nothing attaches to the session.
# `--with-capture` (first arg) opts back in: screen/camera agents exercise
# their real tools using the caller's DISPLAY/XAUTHORITY. Explicit user
# decision only — the marginal link means capture runs can still freeze the
# box until the hardware is fixed.
# POWER GUARD: at 250W caps, synchronized TP4 inference transients hard-lock
# this host (degrading 12V path, confirmed by A/B load test 2026-06-11 —
# 23s to lock at 250W, clean at <=200W). Refuse to drive the GPUs unless
# every inference card is capped at or below the safe ceiling.
MAX_PL="${FREN_AUTOLOOP_MAX_PL:-200}"
over=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | awk -v m="$MAX_PL" '$1 > m + 0.5 {n++} END {print n+0}')
if [ "$over" -gt 0 ]; then
  echo "[autoloop] REFUSING to start: $over GPU(s) capped above ${MAX_PL}W — this hard-locks the host." >&2
  echo "[autoloop] Fix with: sudo ./set_gpu_power.sh -p $MAX_PL -t $MAX_PL   (or raise FREN_AUTOLOOP_MAX_PL after the PSU/12V path is repaired)" >&2
  exit 1
fi

export FREN_AUTOLOOP=1
if [ "$WITH_CAPTURE" = "1" ]; then
  unset FREN_DISABLE_CAPTURE   # .env may set it; the flag overrides for this run
  echo "[autoloop] --with-capture: screen/camera tools LIVE (DISPLAY=${DISPLAY:-unset})"
else
  export FREN_DISABLE_CAPTURE=1
  unset DISPLAY WAYLAND_DISPLAY
  # Poison X auth outright: even if a child process re-derives DISPLAY, the
  # connection fails authentication cleanly instead of attaching to the session.
  export XAUTHORITY=/dev/null
fi

echo "[autoloop] DATABASE_URL -> $LOOP_DB  (prod '$PROD_DB' is untouched)"
exec /home/dw/programing/OpenCodeCompilerV2/.venv/bin/python -m app improve "$@"
