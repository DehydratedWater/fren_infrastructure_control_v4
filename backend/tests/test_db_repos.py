"""DB layer smoke — every repo module imports and exposes its Repo classes.

Raw-SQL Postgres repos can't be exercised without a live DB; this guards the
import surface + class presence (catches a broken port / bad import) and is the
hook a live-DB integration suite plugs into later.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

pytest.importorskip("sqlalchemy")


def _repo_modules() -> list[str]:
    import app.db.repos as pkg

    return sorted(m.name for m in pkgutil.iter_modules(pkg.__path__))


def test_all_repo_modules_import():
    mods = _repo_modules()
    # v3 had ~49 repo modules; ensure we ported the full set.
    assert len(mods) >= 49
    for name in mods:
        importlib.import_module(f"app.db.repos.{name}")


def test_repos_expose_repo_classes():
    total = 0
    for name in _repo_modules():
        mod = importlib.import_module(f"app.db.repos.{name}")
        classes = [k for k, v in vars(mod).items()
                   if isinstance(v, type) and k.endswith("Repo")]
        total += len(classes)
    # ~88 repo classes across the fleet's data layer
    assert total >= 80


def test_key_repos_present():
    from app.db.repos.goals import GoalsRepo
    from app.db.repos.todos import TodosRepo
    from app.db.repos.chat import ChatMessagesRepo
    from app.db.repos.ralf import RalfProcessesRepo

    assert all([GoalsRepo, TodosRepo, ChatMessagesRepo, RalfProcessesRepo])


def test_session_helpers_present():
    from app.db import execute_sql, fetch_all, fetch_one, get_async_session

    assert all([execute_sql, fetch_all, fetch_one, get_async_session])
