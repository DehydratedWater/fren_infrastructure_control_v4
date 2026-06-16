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
    # Routing is the structured `route` field, NOT a tool — only cheap CRUD tools.
    assert names == {"todo_manager", "fetch_context"}


def test_fc_has_structured_route_schema():
    from app.agents.first_contact import _live_spec

    schema = _live_spec().output_schema
    assert schema is not None
    assert set(schema["required"]) == {"route", "intent", "key_points", "message_kind"}
    assert set(schema["properties"]["route"]["enum"]) == {"direct", "image", "video", "handoff"}


def test_fc_decision_keeps_thinking_render_is_fast():
    # The FC DECISION keeps thinking ON (routing needs reasoning — thinking-off
    # mis-routed greetings). The expensive persona_prose RENDER is what runs
    # thinking-off, via generate_persona_message(fast=True).
    import inspect

    from app.agents import first_contact
    from app.agents.config import QWEN35_27B_LIVE

    assert "enable_thinking" not in QWEN35_27B_LIVE.provider_options.get("extra_body", "")
    src = inspect.getsource(first_contact.run_first_contact)
    assert "fast=True" in src  # render is the fast (thinking-off) pass


def test_handoff_tool_has_agent_and_instruction():
    from app.agents.first_contact import _HANDOFF

    props = _HANDOFF["schema"]["properties"]
    assert "agent" in props and "instruction" in props
    assert _HANDOFF["script"] is None  # handled specially (detached spawn)


def test_routing_is_structured_with_specialist_map():
    from app.agents.first_contact import _ROUTE_TO_AGENT, _live_spec

    names = {t.name for t in _live_spec().tools}
    assert names == {"todo_manager", "fetch_context"}  # no handoff/call_specialist tools
    # media/heavy routing is the structured `route` → specialist (deterministic spawn)
    assert _ROUTE_TO_AGENT["image"] == "persona/twily_selfie"
    assert _ROUTE_TO_AGENT["video"] == "persona/twily_videographer"
    assert _ROUTE_TO_AGENT["handoff"] == "persona/orchestrator"


# ── routing probes: deterministic plumbing via a stub client ──


class _StubClient:
    """Returns a scripted ChatResponse sequence (no LLM)."""

    def __init__(self, *responses):
        from src.interactive.runner import ChatResponse  # noqa: F401

        self._responses = list(responses)
        self._i = 0

    def complete(self, *, messages, tools, model, **params):
        r = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return r


def _final(content='{"intent":"x","key_points":["x"],"message_kind":"reply"}'):
    from src.interactive.runner import ChatResponse

    return ChatResponse(content=content)


def _tool(name, args):
    from src.interactive.runner import ChatResponse, ChatToolCall

    return ChatResponse(content="", tool_calls=[ChatToolCall(id="1", name=name, args=args)])


def test_route_probe_direct_when_no_tool():
    from app.agents.first_contact import route_probe

    assert route_probe("good morning", client=_StubClient(_final())) == "direct"


def test_route_probe_reports_structured_route():
    from app.agents.first_contact import route_probe

    for route in ("image", "video", "handoff", "direct"):
        content = (
            '{"route":"%s","instruction":"x","intent":"x",'
            '"key_points":["x"],"message_kind":"ack"}' % route
        )
        assert route_probe("x", client=_StubClient(_final(content))) == route


def test_routing_probe_pack_targets_are_valid():
    from app.agents.first_contact import FC_ROUTING_PROBES

    valid = {"direct", "image", "video", "handoff"}
    assert len(FC_ROUTING_PROBES) >= 8
    for msg, route in FC_ROUTING_PROBES:
        assert route in valid, f"{msg!r} → unknown route {route!r}"
