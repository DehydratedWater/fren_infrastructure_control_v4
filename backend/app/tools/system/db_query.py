"""Read-only SQL query tool."""

from __future__ import annotations

import asyncio
import re

from src import ScriptTool, StreamFormat

from pydantic import BaseModel, Field

# Simple safety checks (no external sql_safety dependency)
_BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|COPY)\b",
    re.IGNORECASE,
)
_MULTI = re.compile(r";\s*\S")


def _check_sql(sql: str) -> str | None:
    """Return error message if query is unsafe, else None."""
    if _MULTI.search(sql):
        return "Multi-statement queries are not allowed"
    if _BLOCKED.search(sql):
        return "Only SELECT queries are allowed (read-only)"
    stripped = sql.strip().rstrip(";").strip()
    if not stripped.upper().startswith("SELECT") and not stripped.upper().startswith("WITH"):
        return "Only SELECT/WITH queries are allowed"
    return None


class Input(BaseModel):
    sql: str = Field(description="SQL query to execute (SELECT only)")
    timeout: int = Field(default=60, description="Query timeout in seconds")


class Output(BaseModel):
    success: bool = True
    row_count: int = 0
    data: list[dict] = Field(default_factory=list)
    error: str = ""


class DbQueryTool(ScriptTool[Input, Output]):
    name = "db_query"
    description = "Execute READ-ONLY SQL queries against the database"
    stream_format = StreamFormat.TEXT
    stream_field = "sql"

    def execute(self, inp: Input) -> Output:
        err = _check_sql(inp.sql)
        if err:
            return Output(success=False, error=err)
        return asyncio.run(self._run(inp.sql, inp.timeout))

    async def _run(self, sql: str, timeout: int) -> Output:
        from sqlalchemy import text

        from app.db.session import get_async_session

        try:
            async with get_async_session() as session:
                await session.execute(text(f"SET LOCAL statement_timeout = '{timeout * 1000}'"))
                result = await session.execute(text(sql))
                rows = [dict(r._mapping) for r in result.fetchall()]
                # Rollback to ensure read-only
                await session.rollback()
                return Output(success=True, row_count=len(rows), data=rows)
        except Exception as e:
            return Output(success=False, error=str(e))


if __name__ == "__main__":
    DbQueryTool.run()
