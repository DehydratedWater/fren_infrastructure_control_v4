"""Corpus-driven probe-pack generator — multi-probe autoloop packs per agent.

WHY: the autoloop's `synthesize_probe` gives every agent ONE generic
role-fulfilment probe. That optimises "can it do its job in the abstract", not
"can it do what the user actually asks". This module grounds optimisation in
the REAL v3 chat corpus (19.8k `chat_messages` rows in the read-only v3 DB):

  1. `sample_corpus(agent_id)` pulls real user messages for the agent's domain
     (keyword ILIKE slice, or a broad sample for persona-style agents) via
     `docker exec docker-fren-db-1 psql` — SELECT only, never a write.
  2. `generate_pack(agent)` makes ONE teacher (GLM via opencode — `_zai_chat`,
     never the raw z.ai API) call that writes K self-contained probes in the
     corpus's style/topics/language mix (EN+PL), each with a judge criteria
     string; the anti-meta + tool-discipline judge tail is appended in code so
     it can never be dropped.
  3. Packs persist as JSON under `probe_packs/` (regenerate with `--refresh`
     when the model or usage changes — model-portability) and
     `pack_tests(agent_id)` converts them into graded `AgentTest`s that
     `build_agent_units` ADDS on top of the single judge test.

A missing/broken pack degrades to exactly the old behaviour (judge test only).
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

PACKS_DIR = Path(__file__).parent / "probe_packs"


class PackProbe(BaseModel):
    """One corpus-grounded probe: a self-contained user message + judge rubric."""

    name: str
    prompt: str
    criteria: str
    pass_threshold: float = 0.7
    # Short note which real corpus messages inspired this probe (provenance).
    source_hint: str = ""


class ProbePack(BaseModel):
    """A persisted set of probes for one agent, regenerable from the corpus."""

    agent_id: str
    generated_at: str
    source: str = "v3-corpus"
    teacher: str = ""
    probes: list[PackProbe] = Field(min_length=1)


# --- corpus access (v3 DB, READ-ONLY) ---------------------------------------

# Domain prefix (the part before "/" in agent_id) → ILIKE keyword slices that
# reflect the real usage clusters in the v3 chat corpus. An EMPTY list means
# "broad sample — any user message" (persona-style agents see everything).
# Full agent_id keys override their domain prefix (checked first), so a single
# agent can get its own slice without touching the domain default. Mixed EN/PL
# stems on purpose: the corpus is mixed English/Polish.
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    # --- domain prefixes -----------------------------------------------------
    "goals": ["todo", "task", "remind", "goal", "priorit", "habit", "deadline"],
    "food": ["eat", "meal", "dinner", "lunch", "recipe", "restaurant",
             "hungry", "cook"],
    "persona": [],  # broad sample — any user message
    "profile": [],  # broad sample — profile learns from everything
    "support": ["email", "calendar", "invoice", "brief", "document", "analy"],
    "research": ["research", "find", "video", "youtube", "price", "website",
                 "check"],
    "server": ["camera", "screenshot", "light", "server", "disk", "gpu"],
    "rp": ["adventure", "story", "character", "roleplay"],
    "funfact": ["fact", "interesting", "did you know", "ciekawost"],
    "investigation": ["why", "investigate", "what happened", "figure out",
                      "broken"],
    "retrieval": ["remember", "you said", "last time", "earlier", "history"],
    "vis_simulation": ["image", "picture", "photo", "draw", "generate",
                       "zdjec"],
    "workflow_master": [],  # orchestration sees the full request stream
    "workflows": [],        # orchestration sees the full request stream
    # --- full agent_id overrides (checked before the prefix) -----------------
    "goals/winddown": ["sleep", "tired", "bed", "night", "wind down", "spac"],
}


def _resolve_pg_user(container: str) -> str:
    """POSTGRES_USER from the container env; '' when unresolvable."""
    try:
        proc = subprocess.run(
            ["docker", "exec", container, "env"],
            capture_output=True, text=True, timeout=15,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("POSTGRES_USER="):
                return line.split("=", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _psql(sql: str, *, container: str, db: str) -> str:
    """Run one SELECT via docker exec psql; '' on any failure (never raises).

    Tries the container's POSTGRES_USER first, then the known fallbacks.
    """
    users: list[str] = []
    env_user = _resolve_pg_user(container)
    for u in (env_user, "fren", "postgres"):
        if u and u not in users:
            users.append(u)
    for user in users:
        try:
            proc = subprocess.run(
                ["docker", "exec", container, "psql", "-U", user, "-d", db,
                 "-t", "-A", "-c", sql],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                return proc.stdout
        except Exception:  # noqa: BLE001
            continue
    return ""


def keywords_for(agent_id: str) -> list[str]:
    """Keyword slice for an agent: full-id override first, then domain prefix."""
    if agent_id in DOMAIN_KEYWORDS:
        return DOMAIN_KEYWORDS[agent_id]
    return DOMAIN_KEYWORDS.get(agent_id.split("/", 1)[0], [])


def sample_corpus(
    agent_id: str,
    *,
    limit: int = 40,
    container: str = "docker-fren-db-1",
    db: str = "fren",
) -> list[str]:
    """Random real user messages from the v3 corpus for this agent's domain.

    SELECT-only over `chat_messages` (sender='user'); keyword ILIKE any-of when
    the domain has a slice, broad sample otherwise. Newlines are flattened so
    one psql line == one message. Returns [] on any failure — the generator
    then falls back to role-only probe generation.
    """
    kws = keywords_for(agent_id)
    where = "sender = 'user' AND length(message) >= 8"
    if kws:
        likes = " OR ".join(
            "message ILIKE '%" + k.replace("'", "''") + "%'" for k in kws
        )
        where += f" AND ({likes})"
    sql = (
        "SELECT left(regexp_replace(message, E'[\\n\\r]+', ' ', 'g'), 400) "
        f"FROM chat_messages WHERE {where} ORDER BY random() LIMIT {int(limit)}"
    )
    out = _psql(sql, container=container, db=db)
    msgs: list[str] = []
    for line in out.splitlines():
        m = line.strip()
        if len(m) < 8 or m.startswith("/"):
            continue  # too short / pure slash-command
        msgs.append(m)
    return msgs


# --- teacher generation ------------------------------------------------------

# Appended IN CODE to every probe's judge criteria so the anti-meta and
# tool-discipline clauses (the spirit of make_judge_test's criteria tail) can
# never be dropped by a forgetful teacher.
_CRITERIA_TAIL = (
    "\nScore 0.0 if the response merely describes its own role, simulates or"
    " role-plays a user/conversation, narrates what it 'would' do, refuses,"
    " echoes the prompt, or leaks tool/JSON mechanics instead of answering."
    " Reward a concrete, correct, on-task result a user could use as-is.\n"
    "If instead the response is a bracketed note that the agent acted by"
    " invoking tools/subagents, judge whether THOSE actions are the right ones"
    " for this request: correct delegation or tool use for the role scores"
    " high; flailing on wrong, hallucinated, or blocked tools scores low.\n"
    "TOOL DISCIPLINE: if the response carries a 'TOOL DISCIPLINE' note that the"
    " agent made DENIED/blocked tool attempts (forbidden by its allow-list) or"
    " that the session ERRORED, LOWER the score in proportion to the number of"
    " blocked attempts, and score an errored session near 0 (a failed run, not"
    " an empty answer)."
)

_TEACHER_SYS = (
    "You design evaluation probes for ONE agent of a Telegram persona-bot"
    " fleet. You are given the AGENT ROLE and a sample of REAL user messages"
    " from the production chat corpus (mixed English/Polish).\n"
    "Write exactly {k} probes. Each probe is ONE realistic user message this"
    " agent should handle, grounded in the style, topics and phrasing of the"
    " REAL messages. Rules:\n"
    "- FULLY SELF-CONTAINED: the agent has NO other context. Inline any data"
    " the task needs (text to rewrite, lists, details). NEVER reference 'the"
    " above', 'this plan', 'the earlier message'.\n"
    "- DIVERSE: cover different responsibilities/facets of the agent's role.\n"
    "- LANGUAGE: preserve the corpus mix — if the real messages mix English"
    " and Polish, write some probes in each language.\n"
    "- For each probe also write a judge `criteria` string that (a) restates"
    " the agent's role, and (b) states concretely what a GOOD response to THIS"
    " exact request does.\n"
    "- Optionally add a short `source_hint` noting which real message(s)"
    " inspired the probe.\n"
    "Output STRICT JSON only — no prose, no markdown fences:\n"
    '{{"probes":[{{"name":"snake_case_short_name","prompt":"...",'
    '"criteria":"...","source_hint":"..."}}]}}'
)

_JSON_NUDGE = (
    "Your previous output was not valid JSON. Return ONLY a valid JSON object"
    ' matching {"probes":[{"name","prompt","criteria","source_hint"}...]} —'
    " no markdown fences, no commentary, nothing before or after the JSON."
)

_CORPUS_CHAR_BUDGET = 6000


def _parse_pack_json(raw: str) -> list[dict]:
    """Lenient parse of the teacher's pack JSON; raises ValueError on garbage."""
    txt = (raw or "").strip()
    txt = re.sub(r"^```[a-zA-Z]*\s*", "", txt)
    txt = re.sub(r"\s*```\s*$", "", txt)
    start, end = txt.find("{"), txt.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in teacher output")
    obj = json.loads(txt[start : end + 1])
    probes = obj.get("probes") if isinstance(obj, dict) else None
    if not isinstance(probes, list) or not probes:
        raise ValueError("teacher JSON has no 'probes' list")
    out: list[dict] = []
    for i, p in enumerate(probes):
        if not isinstance(p, dict):
            continue
        prompt = str(p.get("prompt") or "").strip()
        criteria = str(p.get("criteria") or "").strip()
        if not prompt or not criteria:
            continue
        name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(p.get("name") or "")).strip("_")
        out.append({
            "name": (name or f"probe_{i + 1}")[:48],
            "prompt": prompt,
            "criteria": criteria,
            "source_hint": str(p.get("source_hint") or "").strip()[:300],
        })
    if not out:
        raise ValueError("teacher JSON had no usable probes")
    return out


def _corpus_block(samples: list[str]) -> str:
    """Numbered corpus excerpt, truncated to the prompt budget."""
    lines: list[str] = []
    used = 0
    for i, m in enumerate(samples, 1):
        line = f"{i}. {m}"
        if used + len(line) > _CORPUS_CHAR_BUDGET:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def generate_pack(agent, *, per_agent: int = 5, corpus_limit: int = 40) -> ProbePack:
    """ONE teacher call → a ProbePack of K corpus-grounded probes for `agent`.

    The teacher runs through `_zai_chat` (opencode-routed GLM — raw z.ai calls
    are prohibited). Empty corpus → role-only generation (source='role-only').
    Unparsable teacher output is retried once with a strict-JSON nudge, then
    raises RuntimeError so the CLI can report and continue with other agents.
    """
    from app.agents import improve_live as live  # lazy: heavy runtime imports

    aid = agent.header.agent_id
    role = (agent.usage_explanation_long or agent.usage_explanation_short
            or agent.header.description or aid)
    samples = sample_corpus(aid, limit=corpus_limit)
    try:
        from app.settings import get_settings

        teacher = get_settings().autoloop_teacher_model
    except Exception:  # noqa: BLE001
        teacher = "glm-5.1"

    corpus = _corpus_block(samples)
    usr = (
        f"AGENT ID: {aid}\n\nAGENT ROLE:\n{role}\n\n"
        + (
            f"REAL USER MESSAGES (production corpus sample):\n{corpus}\n\n"
            if corpus
            else "(no corpus sample available — ground the probes in the agent"
                 " role alone, keeping them realistic user messages)\n\n"
        )
        + f"Write the {per_agent} probes now as STRICT JSON."
    )
    messages = [
        {"role": "system", "content": _TEACHER_SYS.format(k=per_agent)},
        {"role": "user", "content": usr},
    ]
    last_err: Exception | None = None
    items: list[dict] | None = None
    for attempt in range(2):
        raw = live._zai_chat(teacher, messages, max_tokens=4000, temperature=0.5)
        try:
            items = _parse_pack_json(raw)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            messages = messages + [{"role": "user", "content": _JSON_NUDGE}]
    if items is None:
        raise RuntimeError(
            f"teacher returned unparsable probe-pack JSON for {aid}: {last_err}"
        )
    probes = [
        PackProbe(
            name=it["name"],
            prompt=it["prompt"],
            criteria=it["criteria"].rstrip() + _CRITERIA_TAIL,
            source_hint=it["source_hint"],
        )
        for it in items[:per_agent]
    ]
    return ProbePack(
        agent_id=aid,
        generated_at=datetime.now(timezone.utc).isoformat(),
        source="v3-corpus" if samples else "role-only",
        teacher=teacher,
        probes=probes,
    )


# --- persistence + AgentTest conversion --------------------------------------


def _pack_path(agent_id: str) -> Path:
    return PACKS_DIR / (agent_id.replace("/", "__") + ".json")


def write_pack(pack: ProbePack) -> Path:
    PACKS_DIR.mkdir(parents=True, exist_ok=True)
    path = _pack_path(pack.agent_id)
    path.write_text(json.dumps(pack.model_dump(), indent=2, ensure_ascii=False))
    return path


def load_pack(agent_id: str) -> ProbePack | None:
    """The persisted pack for `agent_id`, or None when missing/invalid."""
    path = _pack_path(agent_id)
    try:
        if not path.exists():
            return None
        return ProbePack.model_validate_json(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def pack_tests(agent_id: str) -> list:
    """Convert the agent's persisted pack into graded judge AgentTests.

    Missing pack → [] (the autoloop then runs the single role-fulfilment judge
    test exactly as before). Every probe prompt gets the anti-meta guard.
    """
    pack = load_pack(agent_id)
    if pack is None:
        return []
    from app.agents.improve import _ANTI_META  # local: improve imports us lazily
    from src import AgentTest, LLMJudgeEvaluator

    return [
        AgentTest(
            name=f"{agent_id}::pack:{probe.name}",
            prompt=probe.prompt.rstrip() + _ANTI_META,
            evaluators=(
                LLMJudgeEvaluator(
                    name=probe.name,
                    criteria=probe.criteria,
                    pass_threshold=probe.pass_threshold,
                ),
            ),
        )
        for probe in pack.probes
    ]


# --- batch generation ---------------------------------------------------------


def generate_packs(
    agent_ids: list[str] | None,
    *,
    per_agent: int = 5,
    refresh: bool = False,
    workers: int = 6,
    corpus_limit: int = 40,
) -> dict[str, str]:
    """Generate (or skip existing) packs for the targeted agents, in parallel.

    Returns {agent_id: "ok" | "skipped" | "error: ..."} — one teacher call per
    generated agent, ThreadPoolExecutor-parallel like `prewarm_probes`.
    """
    from concurrent.futures import ThreadPoolExecutor

    from app.agents.registry import all_agents

    wanted = set(agent_ids) if agent_ids is not None else None
    targets = [a for a in all_agents()
               if wanted is None or a.header.agent_id in wanted]

    def _one(agent) -> tuple[str, str]:
        aid = agent.header.agent_id
        if not refresh and _pack_path(aid).exists():
            return aid, "skipped"
        try:
            write_pack(generate_pack(
                agent, per_agent=per_agent, corpus_limit=corpus_limit,
            ))
            return aid, "ok"
        except Exception as e:  # noqa: BLE001
            return aid, f"error: {e}"

    results: dict[str, str] = {}
    if targets:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for aid, status in ex.map(_one, targets):
                results[aid] = status
    return results
