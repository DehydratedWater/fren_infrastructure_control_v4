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

    # --- telegram -----------------------------------------------------------
    bot_token: str = Field(default="", alias="BOT_TOKEN")
    chat_id: str = Field(default="", alias="CHAT_ID")

    # --- auth ---------------------------------------------------------------
    jwt_secret: str = Field(default="change-me", alias="JWT_SECRET")
    jwt_alg: str = Field(default="HS256", alias="JWT_ALG")


@lru_cache
def get_settings() -> Settings:
    return Settings()
