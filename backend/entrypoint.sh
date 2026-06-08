#!/usr/bin/env bash
# Materialise the full opencode.json (z.ai worker + local-vLLM providers so
# persona_prose / rp_prose resolve the local Qwen models — a zai-only config
# made the prose layer fall through to api.openai.com), run migrations once,
# then start the selected service. SERVICE = bot | scheduler | checker | compile.
set -euo pipefail

cd /app/backend
python -m app.opencode_config
echo "[entrypoint] opencode config written (providers: zai + local-vllm*)"

SERVICE="${SERVICE:-bot}"
AGENTS_DIR="${AGENTS_DIR:-/data/agents}"
mkdir -p "$AGENTS_DIR"

# The bot is the single boot writer: run migrations + compile the fleet once.
# Compile FIRST (it cleans AGENTS_DIR) so it can't clobber the scripts symlink.
if [ "$SERVICE" = "bot" ]; then
  alembic upgrade head || echo "[entrypoint] alembic upgrade skipped/failed (continuing)"
  echo "[entrypoint] compiling fleet to $AGENTS_DIR"
  python -m app compile || echo "[entrypoint] fleet compile failed (continuing)"
fi

# Compiled agents run `python scripts/<x>.py` from cwd=AGENTS_DIR; link the
# baked-in scripts there so they resolve (PYTHONPATH already exposes app + src).
# rm -rf first: `ln -sfn` creates a nested dir if the target already exists as a
# directory (e.g. a stale dir left on the persisted volume).
rm -rf "$AGENTS_DIR/scripts"
ln -s /app/scripts "$AGENTS_DIR/scripts"
echo "[entrypoint] scripts linked: $(ls -1 "$AGENTS_DIR/scripts"/*.py 2>/dev/null | wc -l) entrypoints"

echo "[entrypoint] starting service: $SERVICE"
case "$SERVICE" in
  bot)       exec python -m app bot ;;
  scheduler) exec python -m app scheduler ;;
  checker)   exec python -m app checker ;;
  web)       exec python -m app web ;;
  compile)   exec python -m app compile ;;
  *)         exec python -m app "$SERVICE" ;;
esac
