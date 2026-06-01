FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl unzip ca-certificates git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# opencode CLI — the runtime the workers call.
RUN curl -fsSL https://opencode.ai/install | bash
ENV PATH="/root/.opencode/bin:${PATH}"

WORKDIR /app

# Deps installed without the framework (mounted + importable via PYTHONPATH).
# Core + the integration tools' runtime libs (tuya/google/pdf/docx/embeddings/
# media/charts/youtube) so every ported tool runs, matching v3's dependency set.
RUN pip install --upgrade pip && \
    pip install fastapi 'uvicorn[standard]' 'sqlalchemy[asyncio]' asyncpg alembic \
        'pydantic[email]>=2.6' pydantic-settings pyjwt python-dotenv \
        httpx croniter apscheduler python-telegram-bot pyyaml \
        langchain-core langchain-openai \
        tinytuya 'pymupdf>=1.24' python-docx openai \
        google-api-python-client google-auth google-auth-oauthlib \
        pillow matplotlib yt-dlp beautifulsoup4 \
        'loguru>=0.7.3' \
        pytest pytest-asyncio

COPY backend /app/backend
# Script entrypoints compiled agents shell out to (`python scripts/<x>.py`);
# entrypoint symlinks these into AGENTS_DIR so they resolve from opencode's cwd.
COPY scripts /app/scripts
COPY backend/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Framework provided via runtime volume mount at /srv/oac.
ENV PYTHONPATH=/srv/oac:/app:/app/backend

EXPOSE 8000
WORKDIR /app/backend
CMD ["/usr/local/bin/entrypoint.sh"]
