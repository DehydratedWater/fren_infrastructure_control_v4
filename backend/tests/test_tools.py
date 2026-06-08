"""Tool layer smoke — every ported ScriptTool module imports + is a ScriptTool.

Tools execute against the live DB / external services at runtime; this guards
the import surface + the ScriptTool contract (name/description/execute) across
the whole ported fleet, catching a broken port or bad import.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

pytest.importorskip("sqlalchemy")

from src import ScriptTool  # noqa: E402


def _tool_areas():
    import app.tools as tp

    return [m.name for m in pkgutil.iter_modules(tp.__path__) if m.ispkg]


def test_all_tool_modules_import():
    failures = []
    total = 0
    for area in _tool_areas():
        mod = importlib.import_module(f"app.tools.{area}")
        for sm in pkgutil.iter_modules(mod.__path__):
            total += 1
            try:
                importlib.import_module(f"app.tools.{area}.{sm.name}")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{area}.{sm.name}: {type(exc).__name__}: {exc}")
    assert not failures, "tool import failures:\n" + "\n".join(failures)
    assert total >= 70  # ~75 tools ported


# Helper modules in a tool area that are intentionally NOT ScriptTools
# (imported by sibling tools, e.g. shared search functions) or are background
# scheduler jobs driven by a thin scripts/ entrypoint rather than an agent
# (the proactive-signal ingestion jobs: digest / inner-monologue / camera).
_TOOL_HELPER_MODULES = {
    "research.web_search",
    "context.conversation_digest",
    "system.inner_monologue",
    "system.activity_observer",
}


def test_each_module_exposes_a_scripttool():
    missing = []
    for area in _tool_areas():
        mod = importlib.import_module(f"app.tools.{area}")
        for sm in pkgutil.iter_modules(mod.__path__):
            key = f"{area}.{sm.name}"
            # `scripts` subpackages hold CLI entrypoints, not ScriptTool defs.
            if sm.ispkg or sm.name == "scripts" or key in _TOOL_HELPER_MODULES:
                continue
            m = importlib.import_module(f"app.tools.{area}.{sm.name}")
            tools = [
                v for v in vars(m).values()
                if inspect.isclass(v) and issubclass(v, ScriptTool) and v is not ScriptTool
            ]
            if not tools:
                missing.append(key)
    assert not missing, "modules with no ScriptTool subclass:\n" + "\n".join(missing)


def test_stream_format_tools_use_the_framework_enum():
    from src import StreamFormat

    from app.tools.system.db_query import DbQueryTool

    assert DbQueryTool.stream_format == StreamFormat.TEXT
    assert DbQueryTool.stream_field == "sql"
