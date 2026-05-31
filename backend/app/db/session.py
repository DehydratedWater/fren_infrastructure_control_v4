"""Async PostgreSQL session + raw-SQL helpers.

Faithful port of v3's `fren/db/session.py`: the whole data layer is raw SQL
(no ORM), so repos stay thin and the schema lives in Alembic. Datetimes are
coerced to Europe/Warsaw on read, matching v3 so downstream formatting is
identical. `set_null_pool()` is used by short-lived script-tool processes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

_LOCAL_TZ = ZoneInfo("Europe/Warsaw")

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_use_null_pool: bool = False


def set_null_pool(enabled: bool = True) -> None:
    """Enable NullPool for short-lived script-tool processes."""
    global _use_null_pool
    _use_null_pool = enabled


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        pool_kwargs: dict = {}
        if _use_null_pool:
            pool_kwargs["poolclass"] = NullPool
        else:
            pool_kwargs.update(
                pool_size=10, max_overflow=20, pool_pre_ping=True, pool_recycle=3600,
            )
        _engine = create_async_engine(
            get_settings().database_url, echo=False, **pool_kwargs,
        )
    return _engine


def _get_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False,
            autocommit=False, autoflush=False,
        )
    return _session_factory


@asynccontextmanager
async def get_async_session() -> "AsyncGenerator[AsyncSession, None]":
    factory = _get_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


def _to_local(row: dict[str, Any]) -> dict[str, Any]:
    """Convert UTC datetimes in a row to Europe/Warsaw (v3 parity)."""
    for key, val in row.items():
        if isinstance(val, datetime):
            if val.tzinfo is None:
                val = val.replace(tzinfo=UTC)
            row[key] = val.astimezone(_LOCAL_TZ)
    return row


async def execute_sql(
    session: AsyncSession, query: str, params: dict[str, Any] | None = None,
) -> Any:
    return await session.execute(text(query), params or {})


async def fetch_one(
    session: AsyncSession, query: str, params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    result = await execute_sql(session, query, params)
    row = result.fetchone()
    return _to_local(dict(row._mapping)) if row else None


async def fetch_all(
    session: AsyncSession, query: str, params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    result = await execute_sql(session, query, params)
    return [_to_local(dict(r._mapping)) for r in result.fetchall()]
