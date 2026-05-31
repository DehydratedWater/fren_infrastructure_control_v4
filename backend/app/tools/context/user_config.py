"""User config tool — read/write user preferences and agent configuration."""

from __future__ import annotations

import asyncio

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="get|set|list")
    key: str = Field(
        default="",
        description="Config key (knowledge_sheet, periodic_check_focus, evening_focus_config, winddown_config)",
    )
    value: str = Field(default="", description="Config value to set")


class Output(BaseModel):
    success: bool = True
    config: dict | None = None
    configs: list[dict] = Field(default_factory=list)
    error: str = ""


class UserConfigTool(ScriptTool[Input, Output]):
    name = "user_config"
    description = "Read and write user preferences and agent configuration"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.user_config import UserConfigRepo

        repo = UserConfigRepo()

        if inp.command == "get":
            if not inp.key:
                return Output(success=False, error="key is required for get")
            config = await repo.get(inp.key)
            if config:
                return Output(success=True, config=config)
            return Output(success=False, error=f"Config key not found: {inp.key}")

        if inp.command == "set":
            if not inp.key:
                return Output(success=False, error="key is required for set")
            config = await repo.set(inp.key, inp.value)
            return Output(success=True, config=config)

        if inp.command == "list":
            configs = await repo.list()
            return Output(success=True, configs=configs)

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    UserConfigTool.run()
