"""Alembic async env — runs migrations against the configured DATABASE_URL.

Raw-SQL schema (no ORM metadata), so autogenerate is disabled; migrations are
hand-written, matching v3's manual-migration discipline.
"""

from __future__ import annotations

import asyncio
import os

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    try:
        from app.settings import get_settings

        return get_settings().database_url
    except Exception:
        return "postgresql+asyncpg://fren:fren@localhost:5452/fren"


def run_migrations_offline() -> None:
    context.configure(url=_database_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def _do_run(connection) -> None:
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_database_url(), echo=False)
    async with engine.connect() as connection:
        await connection.run_sync(_do_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
