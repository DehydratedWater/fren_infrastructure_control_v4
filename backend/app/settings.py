"""Runtime configuration (pydantic-settings).

Mirrors v3's env surface where it matters for a drop-in replacement: database,
the z.ai worker key, the local-qwen live provider, the compiled-agents
directory, and execution backend. Secrets come from the environment / a
gitignored `.env`; nothing is inlined.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- database -----------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://fren:fren@localhost:5452/fren",
        alias="DATABASE_URL",
    )

    # --- worker provider (z.ai coding plan, drives opencode) ----------------
    zai_api_key: str = Field(default="", alias="ZAI_API_KEY")
    worker_model: str = Field(
        default="zai-coding-plan/glm-4.5-air", alias="WORKER_MODEL",
    )
    execution_backend: str = Field(default="direct", alias="EXECUTION_BACKEND")

    # --- autoresearch (autoloop) --------------------------------------------
    # The STRONG teacher model (z.ai) that proposes prompt rewrites AND judges
    # responses. The agents themselves are tuned to run on the local qwen.
    autoloop_teacher_model: str = Field(default="glm-5.1", alias="AUTOLOOP_TEACHER_MODEL")
    # opencode provider/model the candidate agents are compiled+run on while
    # being tuned (the local Qwen-27B served by vLLM).
    autoloop_target_model: str = Field(
        default="local-vllm-remote/qwen35-27b", alias="AUTOLOOP_TARGET_MODEL",
    )

    # --- live / interactive provider (local OpenAI-compatible qwen) ---------
    local_llm_base_url: str = Field(default="", alias="LOCAL_LLM_BASE_URL")
    local_llm_model: str = Field(default="qwen3.5-27b", alias="LOCAL_LLM_MODEL")
    local_llm_api_key: str = Field(default="not-needed", alias="LOCAL_LLM_API_KEY")

    # --- locale -------------------------------------------------------------
    user_timezone: str = Field(default="Europe/Warsaw", alias="USER_TIMEZONE")

    # --- agents / runtime ---------------------------------------------------
    agents_dir: Path = Field(
        default=Path("/data/agents"), alias="AGENTS_DIR",
        description="Where compiled .opencode agent trees live (NOT /tmp).",
    )
    # How often the checker service runs an intervention tick (seconds).
    checker_interval_seconds: int = Field(
        default=300, alias="CHECKER_INTERVAL_SECONDS",
    )

    # --- telegram -----------------------------------------------------------
    bot_token: str = Field(default="", alias="BOT_TOKEN")
    bot_rp_token: str = Field(default="", alias="BOT_RP_TOKEN")
    chat_id: str = Field(default="", alias="CHAT_ID")
    # ADDED(v4-port): bot.build_application() reads telegram_api_id/hash to switch
    # to the local Bot API server (2GB file support). Default "" mirrors v3.
    telegram_api_id: str = Field(default="", alias="TELEGRAM_API_ID")
    telegram_api_hash: str = Field(default="", alias="TELEGRAM_API_HASH")

    # --- persona_prose (chat persona model) ---------------------------------
    # ADDED(v4-port): handlers.handle_model_chat reads persona_prose_provider +
    # persona_prose_model. Defaults + types mirror v3 fren/config.py.
    persona_prose_provider: str = Field(default="local-vllm-remote", alias="PERSONA_PROSE_PROVIDER")
    persona_prose_model: str = Field(default="qwen35-27b", alias="PERSONA_PROSE_MODEL")
    persona_prose_temperature: float | None = Field(default=0.85, alias="PERSONA_PROSE_TEMPERATURE")
    persona_prose_max_tokens: int | None = Field(default=16384, alias="PERSONA_PROSE_MAX_TOKENS")
    persona_prose_timeout_seconds: int = Field(default=180, alias="PERSONA_PROSE_TIMEOUT_SECONDS")

    # --- tts (voice-message synthesis) --------------------------------------
    tts_host: str = Field(default="localhost:8200", alias="TTS_HOST")
    tts_speed: float = Field(default=0.85, alias="TTS_SPEED")
    tts_output_dir: str = Field(default="./tts_output", alias="TTS_OUTPUT_DIR")

    # --- stt (faster-whisper) -----------------------------------------------
    # ADDED(v4-port): analyze_media's _transcribe_audio reads settings.stt_host.
    # Default + alias mirror v3 fren/config.py.
    stt_host: str = Field(default="192.168.0.95:8201", alias="STT_HOST")

    # --- vLLM ---------------------------------------------------------------
    # ADDED(v4-port): analyze_media's _call_api reads settings.vllm_api_key.
    # Default + alias mirror v3 fren/config.py.
    vllm_api_key: str = Field(default="EMPTY", alias="VLLM_API_KEY")

    # --- ComfyUI ------------------------------------------------------------
    # ADDED(v4-port): comfyui.render reads settings.get_comfyui_hosts(), which
    # parses this field. Default + alias mirror v3 fren/config.py.
    comfyui_instances: str = Field(default="192.168.0.95:8899", alias="COMFYUI_INSTANCES")

    # --- Personality Core ---------------------------------------------------
    # ADDED(v4-port): PersonalityCoreTool._call_model + app.personality helper
    # read settings.personality_core_host. Default + alias mirror v3 config.
    personality_core_host: str = Field(default="192.168.0.42:5506", alias="PERSONALITY_CORE_HOST")

    # --- auth ---------------------------------------------------------------
    jwt_secret: str = Field(default="change-me", alias="JWT_SECRET")
    jwt_alg: str = Field(default="HS256", alias="JWT_ALG")

    # --- openai (embeddings only) -------------------------------------------
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # --- google oauth2 (gmail + calendar services) --------------------------
    google_client_id: str = Field(default="", alias="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field(default="", alias="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = Field(
        default="http://localhost:8642/oauth2callback", alias="GOOGLE_REDIRECT_URI",
    )
    # Comma-separated allowed recipients ("" = all allowed).
    gmail_whitelist: str = Field(default="", alias="GMAIL_WHITELIST")
    # Comma-separated read-only account names (e.g. "work,shared").
    gmail_readonly_accounts: str = Field(default="", alias="GMAIL_READONLY_ACCOUNTS")
    # Twily's own calendar for writes.
    google_calendar_id: str = Field(default="primary", alias="GOOGLE_CALENDAR_ID")

    # --- paths --------------------------------------------------------------
    # Root used by google_auth to locate `.google_token*.json` files.
    project_root: str = Field(
        default_factory=lambda: str(Path.cwd()), alias="PROJECT_ROOT",
    )

    def get_comfyui_hosts(self) -> list[tuple[str, int]]:
        """Parse COMFYUI_INSTANCES into list of (host, port) tuples.

        ADDED(v4-port): faithful port of v3 fren/config.py
        Settings.get_comfyui_hosts; consumed by app.comfyui.render.render_scene.
        """
        hosts: list[tuple[str, int]] = []
        for entry in self.comfyui_instances.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" in entry:
                host, port = entry.rsplit(":", 1)
                hosts.append((host, int(port)))
            else:
                hosts.append((entry, 8188))
        return hosts


@lru_cache
def get_settings() -> Settings:
    return Settings()
