"""Helpers for the system-parity / health test suite (the "autoloop").

Offline, cheap, deterministic introspection of the v4 fleet + config so the
recurring CLASS of latent problems (missing scripts, tool/allow-list drift,
model/provider mismatch, ephemeral media paths, endpoint hard-fails, stale
emotion state) is caught automatically.

Nothing here hits the network or the DB. The fleet is compiled to a tmp dir; the
configs are parsed straight off disk.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import yaml

# Repo layout: backend/tests/_parity_helpers.py -> repo root is parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIG_DIR = REPO_ROOT / "config"
OPENCODE_JSON = REPO_ROOT / "opencode.json"
SCHEDULE_YML = CONFIG_DIR / "schedule.yml"
DOCKER_COMPOSE = REPO_ROOT / "docker-compose.yml"

# The single persistent volume mount declared in docker-compose.yml. Anything
# that must survive `docker compose down/up` (rendered media, captures, voice)
# has to live UNDER this path; otherwise it is written into the ephemeral image
# layer (WORKDIR /app/backend) and lost on container recreate.
PERSISTENT_VOLUME_MOUNT = "/data"


# ── opencode.json ──────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def opencode_config() -> dict:
    return json.loads(OPENCODE_JSON.read_text())


@functools.lru_cache(maxsize=1)
def declared_provider_models() -> set[str]:
    """Every ``<provider>/<model>`` pair declared in opencode.json.

    A compiled agent's ``model:`` line must resolve to one of these or opencode
    cannot route it (silent fallback to the global default / api.openai.com).
    """
    cfg = opencode_config()
    pairs: set[str] = set()
    for provider, pdef in (cfg.get("provider") or {}).items():
        for model in (pdef.get("models") or {}):
            pairs.add(f"{provider}/{model}")
    # The top-level default `model` / `small_model` are also valid targets even
    # if the provider block lists no explicit `models` map.
    for key in ("model", "small_model"):
        val = cfg.get(key)
        if val:
            pairs.add(val)
    return pairs


@functools.lru_cache(maxsize=1)
def provider_base_urls() -> dict[str, str]:
    """Map each opencode provider key -> its configured baseURL (if any)."""
    cfg = opencode_config()
    out: dict[str, str] = {}
    for provider, pdef in (cfg.get("provider") or {}).items():
        url = (pdef.get("options") or {}).get("baseURL")
        if url:
            out[provider] = url
    return out


# ── schedule.yml ───────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def schedule_jobs() -> dict[str, dict]:
    return (yaml.safe_load(SCHEDULE_YML.read_text()) or {}).get("jobs", {})


def schedule_script_jobs() -> dict[str, str]:
    """job_name -> 'scripts/X.py' for every ``agent: script:scripts/X.py`` job.

    Includes disabled jobs on purpose: a disabled job still documents a v3->v4
    port gap, and re-enabling it should not silently 404.
    """
    out: dict[str, str] = {}
    for name, job in schedule_jobs().items():
        agent = str(job.get("agent", ""))
        if agent.startswith("script:"):
            out[name] = agent.split("script:", 1)[1]
    return out


# ── compiled fleet introspection ───────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def compiled_fleet_dir() -> Path:
    """Compile the default worker variant once into a tmp dir and cache it."""
    import tempfile

    from app.agents.compile import compile_fleet
    from app.agents.config import DEFAULT_WORKER

    target = Path(tempfile.mkdtemp(prefix="parity_fleet_")) / "build"
    compile_fleet(target=target, project_root=target.parent, variants=[DEFAULT_WORKER])
    return target


def compiled_agent_files() -> list[Path]:
    return sorted((compiled_fleet_dir() / ".opencode" / "agents").rglob("*.md"))


_MODEL_RE = re.compile(r"^model:\s*(.+?)\s*$", re.M)
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---", re.S)
# Allow-list entries look like:  ``    python scripts/foo.py*: allow``
_ALLOWED_SCRIPT_RE = re.compile(r"python\s+(scripts/[A-Za-z0-9_./-]+\.py)\*?\s*:\s*allow")


def agent_model(md_text: str) -> str | None:
    m = _MODEL_RE.search(md_text)
    return m.group(1).strip() if m else None


def agent_frontmatter(md_text: str) -> str:
    m = _FRONTMATTER_RE.search(md_text)
    return m.group(1) if m else ""


def agent_allowed_scripts(md_text: str) -> set[str]:
    """The set of ``scripts/X.py`` paths an agent's bash allow-list permits."""
    fm = agent_frontmatter(md_text)
    return set(_ALLOWED_SCRIPT_RE.findall(fm))
