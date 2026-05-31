"""Personality package — the personality-core HTTP scorer + helpers.

This is the importable helper surface for personality-core access as
``from app.personality import ...``. v3 had no dedicated
``fren/personality/`` package; the canonical model call lived inside
``fren/tools/personality/personality_core.py``. This package re-exports a
stable helper (``call_personality_core``) plus the shared ``SYSTEM_PROMPT``
and the full ``PersonalityCoreTool`` so callers have one ``app.personality``
namespace, while the verbatim ScriptTool port stays in
:mod:`app.tools.personality.personality_core`.
"""

from __future__ import annotations

from app.personality.core import call_personality_core
from app.tools.personality.personality_core import (
    SYSTEM_PROMPT,
    PersonalityCoreTool,
)

__all__ = ["call_personality_core", "SYSTEM_PROMPT", "PersonalityCoreTool"]
