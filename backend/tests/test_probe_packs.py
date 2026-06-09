"""Tests for the corpus-driven probe-pack generator (app/agents/probe_packs.py).

All pure / mocked — NO docker, NO DB, NO teacher: `sample_corpus` and
`improve_live._zai_chat` are monkeypatched, and PACKS_DIR is redirected to a
tmp dir so the repo's packs are never touched. These lock:

1. generate_pack happy path — fenced teacher JSON is parsed; probes carry the
   in-code anti-meta/tool-discipline criteria tail; pack_tests converts them to
   AgentTests with _ANTI_META appended and an LLMJudge evaluator each.
2. garbage teacher output → ONE retry (strict-JSON nudge) → RuntimeError.
3. empty corpus → role-only generation still works (source='role-only').
4. write/load round-trip + '/'→'__' filename mapping.
5. pack_tests for a missing pack returns [] (old single-test behaviour).
6. _judge_test_suite merges judge test + pack tests for build_agent_units.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("src")

from src import AgentDefinition, AgentHeader, AgentTest, LLMJudgeEvaluator

import app.agents.improve as im
import app.agents.probe_packs as pp
from app.agents import improve_live


def _agent(aid="goals/nudge_strategist"):
    return AgentDefinition(
        header=AgentHeader(agent_id=aid, name="X", description="d"),
        usage_explanation_long="Plans gentle task nudges from the user's todo list.",
        usage_explanation_short="nudge planner",
    )


_TEACHER_JSON = {
    "probes": [
        {
            "name": "todo_overload",
            "prompt": "I have 14 todos today: gym, taxes, call mum... help me pick 3.",
            "criteria": "The agent plans nudges. A good response picks 3 priorities.",
            "source_hint": "msgs about todo overload",
        },
        {
            "name": "polish_reminder",
            "prompt": "Przypomnij mi jutro o 9 o spotkaniu z Markiem.",
            "criteria": "The agent plans nudges. A good response sets the reminder.",
        },
    ]
}


@pytest.fixture()
def packs_dir(tmp_path, monkeypatch):
    """Redirect PACKS_DIR to a tmp dir for every test."""
    monkeypatch.setattr(pp, "PACKS_DIR", tmp_path)
    return tmp_path


def _patch_teacher(monkeypatch, replies):
    """_zai_chat fake that pops `replies` in order; records calls."""
    calls = []

    def fake(model, messages, **kw):
        calls.append((model, messages))
        return replies[min(len(calls) - 1, len(replies) - 1)]

    monkeypatch.setattr(improve_live, "_zai_chat", fake)
    return calls


# --- 1. happy path ------------------------------------------------------------

def test_generate_pack_happy_path_fenced_json(packs_dir, monkeypatch):
    monkeypatch.setattr(
        pp, "sample_corpus",
        lambda aid, **kw: ["help me sort my todos", "przypomnij mi o zakupach"],
    )
    fenced = "```json\n" + json.dumps(_TEACHER_JSON) + "\n```"
    calls = _patch_teacher(monkeypatch, [fenced])

    pack = pp.generate_pack(_agent(), per_agent=2)

    assert len(calls) == 1  # ONE teacher call
    assert pack.agent_id == "goals/nudge_strategist"
    assert pack.source == "v3-corpus"
    assert [p.name for p in pack.probes] == ["todo_overload", "polish_reminder"]
    # the anti-meta/tool-discipline judge tail is appended IN CODE
    for probe in pack.probes:
        assert "TOOL DISCIPLINE" in probe.criteria
        assert "describes its own role" in probe.criteria
    assert pack.probes[0].source_hint == "msgs about todo overload"
    # the real corpus samples were actually shown to the teacher
    sent = calls[0][1][-1]["content"]
    assert "przypomnij mi o zakupach" in sent


def test_pack_tests_converts_probes_with_anti_meta(packs_dir, monkeypatch):
    monkeypatch.setattr(pp, "sample_corpus", lambda aid, **kw: ["sort my todos"])
    _patch_teacher(monkeypatch, [json.dumps(_TEACHER_JSON)])
    pack = pp.generate_pack(_agent(), per_agent=2)
    pp.write_pack(pack)

    tests = pp.pack_tests("goals/nudge_strategist")
    assert len(tests) == 2
    t = tests[0]
    assert isinstance(t, AgentTest)
    assert t.name == "goals/nudge_strategist::pack:todo_overload"
    assert t.prompt.startswith("I have 14 todos")
    assert t.prompt.endswith(im._ANTI_META)  # anti-meta guard appended
    (ev,) = t.evaluators
    assert isinstance(ev, LLMJudgeEvaluator)
    assert ev.name == "todo_overload"
    assert ev.pass_threshold == 0.7
    assert "TOOL DISCIPLINE" in ev.criteria


def test_per_agent_caps_probe_count(packs_dir, monkeypatch):
    monkeypatch.setattr(pp, "sample_corpus", lambda aid, **kw: ["x" * 20])
    _patch_teacher(monkeypatch, [json.dumps(_TEACHER_JSON)])
    pack = pp.generate_pack(_agent(), per_agent=1)
    assert len(pack.probes) == 1


# --- 2. garbage → retry → RuntimeError -----------------------------------------

def test_teacher_garbage_retries_once_then_raises(packs_dir, monkeypatch):
    monkeypatch.setattr(pp, "sample_corpus", lambda aid, **kw: [])
    calls = _patch_teacher(monkeypatch, ["lol not json", "still { not json"])

    with pytest.raises(RuntimeError, match="unparsable"):
        pp.generate_pack(_agent(), per_agent=3)
    assert len(calls) == 2  # exactly one retry
    # the retry carried the strict-JSON nudge
    assert "ONLY a valid JSON" in calls[1][1][-1]["content"]


def test_teacher_garbage_then_valid_recovers(packs_dir, monkeypatch):
    monkeypatch.setattr(pp, "sample_corpus", lambda aid, **kw: [])
    calls = _patch_teacher(monkeypatch, ["garbage", json.dumps(_TEACHER_JSON)])
    pack = pp.generate_pack(_agent(), per_agent=2)
    assert len(calls) == 2
    assert len(pack.probes) == 2


# --- 3. empty corpus → role-only generation ------------------------------------

def test_empty_corpus_falls_back_to_role_only(packs_dir, monkeypatch):
    monkeypatch.setattr(pp, "sample_corpus", lambda aid, **kw: [])
    calls = _patch_teacher(monkeypatch, [json.dumps(_TEACHER_JSON)])

    pack = pp.generate_pack(_agent(), per_agent=2)
    assert pack.source == "role-only"
    assert len(pack.probes) == 2
    # the teacher was told there is no corpus, but still got the role
    sent = calls[0][1][-1]["content"]
    assert "no corpus sample available" in sent
    assert "gentle task nudges" in sent


# --- 4. write/load round-trip + filename mapping --------------------------------

def test_write_load_round_trip_and_filename_mapping(packs_dir):
    pack = pp.ProbePack(
        agent_id="persona/twily_chat",
        generated_at="2026-06-10T00:00:00+00:00",
        teacher="glm-5.1",
        probes=[pp.PackProbe(name="hello", prompt="hej co tam?",
                             criteria="Friendly persona reply.")],
    )
    path = pp.write_pack(pack)
    assert path.name == "persona__twily_chat.json"  # '/' → '__'
    assert path.parent == packs_dir
    loaded = pp.load_pack("persona/twily_chat")
    assert loaded == pack


def test_load_pack_invalid_json_returns_none(packs_dir):
    (packs_dir / "persona__broken.json").write_text("{not json")
    assert pp.load_pack("persona/broken") is None


# --- 5. missing pack → [] -------------------------------------------------------

def test_pack_tests_missing_pack_returns_empty(packs_dir):
    assert pp.pack_tests("goals/never_generated") == []


# --- 6. build_agent_units merge --------------------------------------------------

def test_judge_test_suite_merges_pack_tests(packs_dir, monkeypatch):
    """In judge-test mode the suite = role-fulfilment judge test + pack probes."""
    agent = _agent("food/food_suggester")
    monkeypatch.setattr(im, "synthesize_probe", lambda a: "Suggest a quick dinner.")
    pp.write_pack(pp.ProbePack(
        agent_id="food/food_suggester",
        generated_at="2026-06-10T00:00:00+00:00",
        probes=[
            pp.PackProbe(name="quick_dinner", prompt="co na obiad? mam 20 min",
                         criteria="Suggests a realistic 20-minute meal."),
            pp.PackProbe(name="fridge_leftovers", prompt="I have eggs and rice, ideas?",
                         criteria="Suggests a meal from the listed ingredients."),
        ],
    ))

    tests = im._judge_test_suite(agent)
    names = [t.name for t in tests]
    assert names[0] == "food/food_suggester::role-fulfilment"  # judge test first
    assert "food/food_suggester::pack:quick_dinner" in names
    assert "food/food_suggester::pack:fridge_leftovers" in names
    assert len(tests) == 3


def test_judge_test_suite_without_pack_is_single_judge_test(packs_dir, monkeypatch):
    """Missing pack → unchanged old behaviour: exactly the one judge test."""
    agent = _agent("food/food_suggester")
    monkeypatch.setattr(im, "synthesize_probe", lambda a: "Suggest a quick dinner.")
    tests = im._judge_test_suite(agent)
    assert [t.name for t in tests] == ["food/food_suggester::role-fulfilment"]


# --- corpus sampling (psql faked, no docker) -------------------------------------

def test_sample_corpus_filters_short_and_slash_lines(monkeypatch):
    rows = "\n".join([
        "help me plan my tasks for today",
        "/start",          # pure command → skipped
        "ok",              # <8 chars → skipped
        "przypomnij mi o deadline w piatek",
        "",
    ])
    captured = {}

    def fake_psql(sql, *, container, db):
        captured["sql"] = sql
        return rows

    monkeypatch.setattr(pp, "_psql", fake_psql)
    msgs = pp.sample_corpus("goals/nudge_strategist", limit=10)
    assert msgs == [
        "help me plan my tasks for today",
        "przypomnij mi o deadline w piatek",
    ]
    sql = captured["sql"]
    assert "sender = 'user'" in sql
    assert "ILIKE '%todo%'" in sql  # goals keyword slice applied
    assert "LIMIT 10" in sql
    assert sql.strip().upper().startswith("SELECT")  # read-only


def test_sample_corpus_broad_domains_have_no_keyword_filter(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        pp, "_psql",
        lambda sql, *, container, db: captured.__setitem__("sql", sql) or "",
    )
    pp.sample_corpus("persona/twily_chat", limit=5)
    assert "ILIKE" not in captured["sql"]  # persona → broad sample


def test_sample_corpus_psql_failure_returns_empty(monkeypatch):
    monkeypatch.setattr(pp, "_psql", lambda sql, *, container, db: "")
    assert pp.sample_corpus("goals/nudge_strategist") == []


def test_full_agent_id_override_beats_domain_prefix():
    assert "sleep" in pp.keywords_for("goals/winddown")
    assert "todo" in pp.keywords_for("goals/nudge_strategist")
    assert pp.keywords_for("persona/twily_chat") == []
