#!/usr/bin/env bash
# Materialise the opencode config with the real z.ai key (opencode's env:VAR
# substitution is unreliable, so write the literal value at boot), then start.
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
# alembic upgrade head   # enabled once migrations land (P1 DB)
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
