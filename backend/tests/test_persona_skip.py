"""SKIP delivery path — a contract-satisfying silent run delivers NOTHING.

A conditional background agent (periodic_checker, nudge_strategist, ...) that has
nothing to send this run emits a PersonaGuidance with message_kind="skip" (or an
empty guidance). The delivery contract is satisfied (emit_guidance ran) but the
user must receive NOTHING — no persona_prose LLM call, no Telegram send. A normal
reply still delivers. All mocked — no live LLM, DB, or Telegram.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from app.telegram.persona_prose import (
    ChatContext,
    PersonaGuidance,
    generate_persona_message,
    is_skip_guidance,
)


# ── is_skip_guidance: the predicate that defines a no-deliver run ──


def test_is_skip_guidance_for_skip_kind():
    g = PersonaGuidance(intent="nothing to send", key_points=[], message_kind="skip")
    assert is_skip_guidance(g) is True


def test_is_skip_guidance_for_empty_content():
    # empty intent + empty key_points + no raw_data → a skip even without the kind
    assert is_skip_guidance(PersonaGuidance(intent="", key_points=[])) is True
    assert is_skip_guidance(PersonaGuidance(intent="   ", key_points=["  "])) is True


def test_is_skip_guidance_false_for_real_reply():
    g = PersonaGuidance(intent="reply", key_points=["2 new commits ingested"])
    assert is_skip_guidance(g) is False


def test_skip_kind_survives_from_dict():
    g = PersonaGuidance.from_dict({"intent": "x", "key_points": [], "message_kind": "skip"})
    assert g.message_kind == "skip"  # not coerced to "reply"


# ── generate_persona_message: a skip delivers NOTHING but satisfies the contract ──


def _ctx():
    return ChatContext(chat_id=1)


@pytest.mark.asyncio
async def test_skip_guidance_delivers_nothing(monkeypatch):
    """A skip never calls the LLM and never sends to Telegram, but DOES write a
    trace (so the silent run is on record / debuggable)."""
    import app.telegram.persona_prose as pp

    sent: list = []
    traced: list = []

    async def fake_deliver(text, attachments, *, kind=""):
        sent.append(text)

    async def fake_trace(trace):
        traced.append(trace)

    def explode_call(*a, **k):  # the LLM must NOT be invoked for a skip
        raise AssertionError("persona_prose made an LLM call for a SKIP")

    monkeypatch.setattr(pp, "_deliver_via_send_message", fake_deliver)
    monkeypatch.setattr(pp, "_write_trace_artifacts", fake_trace)
    # if generate_persona_message ever reached the OpenAI client, this would fire
    monkeypatch.setattr(pp, "load_provider_details",
                        lambda *a, **k: explode_call())

    guidance = PersonaGuidance(intent="nothing to send", key_points=[],
                               message_kind="skip")
    trace = await generate_persona_message(guidance, _ctx(), run_id="run-skip-1")

    # nothing delivered to the user
    assert sent == []
    assert trace["delivered_text"] == ""
    assert trace.get("skipped") is True
    assert trace["suppressed_reason"] == "agent_skip"
    # but the run IS on record (contract satisfied + debuggable)
    assert traced and traced[0]["run_id"] == "run-skip-1"
    assert traced[0]["delivered_text"] == ""


@pytest.mark.asyncio
async def test_empty_guidance_is_treated_as_skip(monkeypatch):
    """An empty guidance (no intent/key_points) is suppressed via the skip path
    and delivers nothing."""
    import app.telegram.persona_prose as pp

    sent: list = []
    monkeypatch.setattr(pp, "_deliver_via_send_message",
                        lambda t, a, *, kind="": sent.append(t))

    async def fake_trace(trace):
        return None

    monkeypatch.setattr(pp, "_write_trace_artifacts", fake_trace)
    monkeypatch.setattr(pp, "load_provider_details",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("LLM called for empty guidance")))

    trace = await generate_persona_message(
        PersonaGuidance(intent="", key_points=[]), _ctx(), run_id="run-empty-1")
    assert sent == []
    assert trace["delivered_text"] == ""
    assert trace.get("skipped") is True


@pytest.mark.asyncio
async def test_normal_reply_still_delivers(monkeypatch):
    """A real reply is NOT skipped — it renders via the LLM and delivers."""
    import app.telegram.persona_prose as pp

    sent: list = []

    async def fake_deliver(text, attachments, *, kind=""):
        sent.append(text)

    async def fake_trace(trace):
        return None

    monkeypatch.setattr(pp, "_deliver_via_send_message", fake_deliver)
    monkeypatch.setattr(pp, "_write_trace_artifacts", fake_trace)
    monkeypatch.setattr(pp, "load_provider_details",
                        lambda *a, **k: ("http://x", "EMPTY", "model-x"))

    async def fake_to_thread(fn, *a, **k):
        # stand in for the OpenAI call: return the rendered text
        return "Hey — 2 new commits landed. 💜"

    monkeypatch.setattr(pp.asyncio, "to_thread", fake_to_thread)

    guidance = PersonaGuidance(intent="share news",
                               key_points=["2 new commits ingested"],
                               message_kind="reply")
    trace = await generate_persona_message(guidance, _ctx(), run_id="run-reply-1")

    assert sent == ["Hey — 2 new commits landed. 💜"]
    assert trace["delivered_text"] == "Hey — 2 new commits landed. 💜"
    assert not trace.get("skipped")


# ── emit_guidance tool: the skip fast-path records the contract, sends nothing ──


@pytest.mark.asyncio
async def test_emit_guidance_skip_fastpath_records_no_delivery(monkeypatch):
    """EmitGuidanceTool._emit routes a skip to _emit_skip: it writes
    persona_guidance + an empty persona_response (contract on record) and NEVER
    calls persona_prose / send_message."""
    from app.tools.telegram.emit_guidance import EmitGuidanceTool

    writes: list = []

    class _FakeRepo:
        async def ensure_run(self, run_id, **k):
            return None

        async def write_artifact(self, **kw):
            writes.append(kw)
            return {"artifact_id": "art-1"}

    import app.db.repos.execution_ledger as el
    monkeypatch.setattr(el, "ExecutionLedgerRepo", _FakeRepo)

    # generate_persona_message must NOT be reached for a skip emit
    import app.telegram.persona_prose as pp

    async def explode(*a, **k):
        raise AssertionError("generate_persona_message reached for a SKIP emit")

    monkeypatch.setattr(pp, "generate_persona_message", explode)

    monkeypatch.setenv("FREN_RUN_ID", "emit-skip-1")
    tool = EmitGuidanceTool()
    out = await tool._emit(
        '{"intent":"nothing to send","key_points":[],"message_kind":"skip"}')

    assert out.success is True
    assert out.delivered_text == ""
    kinds = [w["artifact_type"] for w in writes]
    assert "persona_guidance" in kinds  # contract on record
    assert "persona_response" in kinds  # post-run hook sees a delivered run
    resp = next(w for w in writes if w["artifact_type"] == "persona_response")
    assert resp["payload"]["delivered_text"] == ""
    assert resp["payload"]["skipped"] is True
