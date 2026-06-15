"""Exact `--command` verb reference injected into compiled agents.

Run traces showed agents inventing wrong verbs (`list-active` for habit_manager's
`list`, `get-extraction-state` for event_manager's `get-state`) and burning turns
on `--help`. `command_vocab` derives the real verbs from each tool and
`with_tool_command_reference` injects them into the agent's prompt at compile.
"""

from __future__ import annotations

from app.agents._tooldefs import command_vocab, script_of_tool
from app.agents._tools import habit_manager_tool
from app.agents.improve import _TOOLCMD_MARKER, with_tool_command_reference
from app.agents.registry import all_agents


def test_command_vocab_from_field_enum():
    # habit_manager declares its verbs in the Input.command field description.
    vocab = command_vocab("scripts/habit_manager.py")
    assert "list" in vocab.split("|")
    assert "list-active" not in vocab  # the verb the model wrongly guessed


def test_command_vocab_from_dispatch_scan():
    # event_manager's command field has no enum — verbs are recovered by scanning
    # `command == "..."` dispatch arms. `get-state` is the real verb the model
    # missed (it guessed `get-extraction-state`).
    vocab = command_vocab("scripts/event_manager.py")
    verbs = set(vocab.split("|"))
    assert "get-state" in verbs
    assert "get-extraction-state" not in verbs


def test_command_vocab_empty_for_non_command_tool():
    # send_message has no `command` field — no vocab, no crash.
    assert command_vocab("scripts/send_message.py") == ""
    assert command_vocab("scripts/does_not_exist.py") == ""


def test_script_of_tool_extracts_path():
    assert script_of_tool(habit_manager_tool()) == "scripts/habit_manager.py"


def test_reference_injected_with_exact_verbs():
    defs = {a.header.agent_id: a for a in all_agents()}
    out = with_tool_command_reference(defs["support/event_extractor"])
    body = out.system_prompt
    assert _TOOLCMD_MARKER in body
    # the exact verbs the agent flailed on are now stated
    assert "get-state" in body
    assert "scripts/habit_manager.py" in body and "list" in body


def test_reference_is_idempotent():
    a = {x.header.agent_id: x for x in all_agents()}["support/event_extractor"]
    once = with_tool_command_reference(a)
    twice = with_tool_command_reference(once)
    assert once.system_prompt.count(_TOOLCMD_MARKER) == 1
    assert twice.system_prompt.count(_TOOLCMD_MARKER) == 1


def test_pure_prompt_agent_unchanged():
    # an agent with no extra_tools gets no block.
    pure = next(
        (a for a in all_agents() if not getattr(a, "extra_tools", [])), None
    )
    if pure is not None:
        out = with_tool_command_reference(pure)
        assert _TOOLCMD_MARKER not in (out.system_prompt or "")
