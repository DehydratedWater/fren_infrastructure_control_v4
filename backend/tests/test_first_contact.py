"""First-contact tier — spec construction + tool wiring (offline, no LLM)."""

from __future__ import annotations


def test_live_spec_targets_local_qwen_served_id():
    from app.agents.first_contact import _live_spec

    spec = _live_spec()
    # The interactive client calls vLLM directly → model_id MUST be the served id.
    assert spec.model_id == "cyankiwi/Qwen3.5-27B-AWQ-BF16-INT8"
    assert spec.base_url == "http://192.168.0.42:8082/v1"
    assert spec.agent_id == "persona/twily_first_contact"


def test_fc_tools_are_actions_only_no_emit_guidance():
    # emit_guidance must NOT be a loop tool (that caused repeated calls); the
    # reply is the structured output instead.
    from app.agents.first_contact import _live_spec

    names = {t.name for t in _live_spec().tools}
    assert "emit_guidance" not in names
    assert names == {"handoff", "todo_manager", "fetch_context"}


def test_fc_has_guidance_output_schema():
    from app.agents.first_contact import _live_spec

    schema = _live_spec().output_schema
    assert schema is not None
    assert set(schema["required"]) == {"intent", "key_points", "message_kind"}


def test_fast_tier_disables_thinking():
    # The quick tier runs thinking OFF for snap (and non-empty content on qwen).
    from app.agents.config import QWEN35_27B_LIVE

    extra = QWEN35_27B_LIVE.provider_options.get("extra_body", "")
    assert "enable_thinking" in extra and "false" in extra.lower()


def test_handoff_tool_has_agent_and_instruction():
    from app.agents.first_contact import _HANDOFF

    props = _HANDOFF["schema"]["properties"]
    assert "agent" in props and "instruction" in props
    assert _HANDOFF["script"] is None  # handled specially (detached spawn)
