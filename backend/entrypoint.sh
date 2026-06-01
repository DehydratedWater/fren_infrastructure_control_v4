#!/usr/bin/env bash
# Materialise opencode config with the real z.ai key (env:VAR substitution is
# unreliable), run migrations once, then start the selected service.
# SERVICE = bot (default) | scheduler | checker.
set -euo pipefail

mkdir -p /root/.config/opencode
cat > /root/.config/opencode/opencode.json <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "${WORKER_MODEL:-zai-coding-plan/glm-4.5-air}",
  "small_model": "${WORKER_MODEL:-zai-coding-plan/glm-4.5-air}",
  "provider": {
    "zai-coding-plan": { "options": { "apiKey": "${ZAI_API_KEY}" } }
  }
}
EOF
echo "[entrypoint] opencode config written (key length: ${#ZAI_API_KEY})"

cd /app/backend

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
  compile)   exec python -m app compile ;;
  *)         exec python -m app "$SERVICE" ;;
esac
