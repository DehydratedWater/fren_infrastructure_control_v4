"""System parity / health suite — the "autoloop" around the recurring problems.

This suite catches the CLASS of latent v3->v4 regressions we keep hitting, all
offline + deterministic (compile the fleet to a tmp dir, parse configs off disk,
mock any external call):

  1. endpoint / model resolution + resilience
  2. image/media persistence (persistent volume vs ephemeral image layer)
  3. emotion network (emotional_state writer + freshness)
  4. port gaps (schedule scripts, tool/allow-list drift)
  5. complete_run wiring (runs stuck status=running)
  6. misc v3->v4 parity

Convention for documenting KNOWN-BROKEN invariants without breaking CI: a test
that asserts a currently-failing invariant is marked ``@pytest.mark.xfail(
strict=True, reason=...)`` so the suite stays green BUT the problem is recorded;
when the user fixes it the test XPASSes (strict=True turns an unexpected pass
into a failure, forcing the marker to be removed). Tests with no xfail assert
invariants that currently HOLD and must keep holding.
"""

from __future__ import annotations

import re

import pytest

from tests._parity_helpers import (
    PERSISTENT_VOLUME_MOUNT,
    REPO_ROOT,
    SCRIPTS_DIR,
    agent_allowed_scripts,
    agent_model,
    compiled_agent_files,
    declared_provider_models,
    opencode_config,
    provider_base_urls,
    schedule_script_jobs,
)

# ── known-DOWN endpoints (per the audit brief, 2026-06) ──────────────────────
# Only 192.168.0.42:8082 is confirmed UP. These three are confirmed DOWN; any
# agent/tool hard-pinned to them with no fallback hard-fails ("model not found").
DOWN_ENDPOINTS = {
    "http://192.168.0.42:8083/v1",  # analytical / qwen35-27b-heretic
    "http://192.168.0.42:5502/v1",  # local-vllm / glm-4.5-air-local
    "http://192.168.0.95:5504/v1",  # vision / qwen3-8b-vl (A4000)
}
UP_ENDPOINT = "http://192.168.0.42:8082/v1"


# ═══════════════════════════════════════════════════════════════════════════
# 4. PORT GAPS — schedule scripts must exist on disk
# ═══════════════════════════════════════════════════════════════════════════


# The script jobs still DISABLED after the 2026-06 final cron-port batch.
# Everything else is ported and enabled (night_analyst, relationship_initiator,
# relationship_reflector, thought_forger and the full topic_synthesizer rebuild
# landed as agents + thin spawn wrappers — see test_cron_port_batch2.py).
# ralf_ping stays PERMANENTLY excluded: it is SUPERSEDED by the framework
# workflow DAG executor (the ralf state-machine tick is subsumed by src
# workflow primitives) — if it is ever re-enabled it should be re-thought,
# not re-ported.
REMAINING_DISABLED_SCRIPT_JOBS = {
    "ralf_ping",  # superseded by the framework workflow DAG (permanent)
}


@pytest.mark.xfail(
    strict=True,
    reason="PERMANENT, BY DESIGN: the disabled ralf_ping job references "
    "scripts/ralf_ping.py, which is intentionally NOT ported — the ralf "
    "state-machine tick is superseded by the framework workflow DAG executor. "
    "The job entry stays in schedule.yml (disabled) to document the v3 "
    "feature; every other script job exists on disk and is enabled.",
)
def test_every_schedule_script_job_targets_an_existing_script():
    """Every ``agent: script:scripts/X.py`` in schedule.yml must exist on disk.

    Catches the "re-enable a job, get a silent 404" class. Disabled jobs are
    included on purpose — the one remaining miss (ralf_ping) documents the
    v3 feature superseded by the workflow DAG.
    """
    missing: dict[str, str] = {}
    for job, script in schedule_script_jobs().items():
        if not (REPO_ROOT / script).is_file():
            missing[job] = script
    assert not missing, (
        "schedule.yml references script jobs with no file on disk "
        f"(re-enabling these 404s): {missing}"
    )


def test_disabled_script_jobs_are_exactly_the_unported_set():
    """The disabled script-job set must equal REMAINING_DISABLED_SCRIPT_JOBS.

    Two-way truthfulness: a ported job flipping back to disabled fails this
    (regression), and enabling/porting ralf_ping without re-thinking it (and
    updating the pin + the xfail above) also fails it.
    """
    from tests._parity_helpers import schedule_jobs

    disabled = {
        name
        for name, job in schedule_jobs().items()
        if str(job.get("agent", "")).startswith("script:") and not job.get("enabled")
    }
    assert disabled == REMAINING_DISABLED_SCRIPT_JOBS, (
        f"disabled script jobs drifted: extra={sorted(disabled - REMAINING_DISABLED_SCRIPT_JOBS)} "
        f"missing={sorted(REMAINING_DISABLED_SCRIPT_JOBS - disabled)}"
    )


def test_ported_cron_batch_jobs_are_enabled_with_v3_schedules():
    """The six ported jobs are enabled and keep their v3 cron expressions.

    activity_daily_summary intentionally diverges from v3's */5: a *daily*
    reconsolidation needs no 5-min cadence and at */5 it saturated the bg vLLM
    lane (~57% timeouts). */15 keeps it fresh at a third of the load.
    """
    from tests._parity_helpers import schedule_jobs

    expected_crons = {
        "activity_daily_summary": "*/15 5-23,0-2 * * *",
        "lesson_extraction": "*/30 6-23,0-2 * * *",
        "event_habit_bridge": "*/10 5-23,0-4 * * *",
        "goal_progress_update": "0 */1 * * *",
        "ralf_cleanup": "0 4 * * *",
        "pending_thoughts_expire": "0 5 * * *",
    }
    jobs = schedule_jobs()
    for name, cron in expected_crons.items():
        job = jobs.get(name)
        assert job is not None, f"job {name} vanished from schedule.yml"
        assert job.get("enabled") is True, f"job {name} is not enabled"
        assert job.get("cron") == cron, f"job {name} cron drifted: {job.get('cron')!r} != {cron!r}"


def test_enabled_schedule_script_jobs_exist():
    """The stricter subset: every ENABLED script job must exist (would run now)."""
    from tests._parity_helpers import schedule_jobs

    jobs = schedule_jobs()
    missing = {
        name: agent.split("script:", 1)[1]
        for name, job in jobs.items()
        if (agent := str(job.get("agent", ""))).startswith("script:")
        and job.get("enabled")
        and not (REPO_ROOT / agent.split("script:", 1)[1]).is_file()
    }
    assert not missing, f"ENABLED schedule script jobs missing on disk: {missing}"


# ═══════════════════════════════════════════════════════════════════════════
# 4. PORT GAPS — agent allow-list ↔ scripts on disk
# ═══════════════════════════════════════════════════════════════════════════


def test_every_allowed_script_in_compiled_fleet_exists_on_disk():
    """Every ``python scripts/X.py`` an agent may run must exist on disk.

    Catches tool/allow-list drift: an agent declares a tool whose script was
    never ported, so the agent can "use" a tool that 404s at runtime.
    """
    offenders: dict[str, set[str]] = {}
    for md in compiled_agent_files():
        text = md.read_text()
        for script in agent_allowed_scripts(text):
            if not (REPO_ROOT / script).is_file():
                offenders.setdefault(md.name, set()).add(script)
    assert not offenders, (
        "compiled agents allow scripts that don't exist on disk: "
        f"{ {k: sorted(v) for k, v in offenders.items()} }"
    )


def test_persona_media_agents_declare_their_media_tools():
    """Delivery/media agents (persona_media) keep their media tools in the
    compiled allow-list — guards against a media tool silently dropping out of
    an agent's permissions during a refactor.
    """
    # twily_selfie must be able to compose + render + dispatch its image.
    selfie = next(
        (md for md in compiled_agent_files() if md.name == "persona/twily_selfie.md".split("/")[-1]),
        None,
    )
    assert selfie is not None, "persona/twily_selfie did not compile"
    allowed = agent_allowed_scripts(selfie.read_text())
    for needed in ("scripts/ponyxl_prompt_composer.py", "scripts/render_ponyxl.py"):
        assert needed in allowed, f"twily_selfie lost media tool {needed}: {sorted(allowed)}"


# ═══════════════════════════════════════════════════════════════════════════
# 1. ENDPOINT / MODEL RESOLUTION — compiled model ↔ opencode.json provider
# ═══════════════════════════════════════════════════════════════════════════


def test_every_compiled_model_maps_to_a_declared_provider_model():
    """Each compiled ``model:`` line must resolve to an opencode.json provider/model.

    Catches the "model: zai/glm-x but opencode.json has no such model -> silent
    fallback to api.openai.com" class.
    """
    declared = declared_provider_models()
    offenders: dict[str, str] = {}
    for md in compiled_agent_files():
        model = agent_model(md.read_text())
        if model and model not in declared:
            offenders[md.name] = model
    assert not offenders, (
        f"compiled agents reference models absent from opencode.json: {offenders}\n"
        f"declared: {sorted(declared)}"
    )


def test_every_compiled_model_across_all_variants_is_declared(tmp_path):
    """Compile EVERY worker variant and assert each ``model:`` resolves in opencode.json.

    The cheap default-variant test above can't see the glm-4.7/5/5.1 gap; this
    one compiles the full matrix.
    """
    import re as _re

    from app.agents.compile import compile_fleet
    from app.agents.config import WORKER_VARIANTS

    target = tmp_path / "allvariants"
    compile_fleet(target=target, project_root=tmp_path, variants=list(WORKER_VARIANTS))
    declared = declared_provider_models()
    seen: set[str] = set()
    for md in target.rglob("*.md"):
        m = _re.search(r"^model:\s*(.+?)\s*$", md.read_text(), _re.M)
        if m:
            seen.add(m.group(1).strip())
    undeclared = seen - declared
    assert not undeclared, (
        f"compiled models (all variants) absent from opencode.json: {sorted(undeclared)}"
    )


def test_config_presets_match_opencode_provider_models():
    """The agents/config.py presets must line up with opencode.json provider keys.

    Compile-time presets produce the ``provider/model`` strings; if they drift
    from opencode.json the compiled fleet routes to nonexistent models.
    """
    from app.agents import config as cfg

    declared = declared_provider_models()
    # Only the three live worker presets remain (default qwen + the two alt
    # cloud passes); QWEN_VL aliases QWEN35_27B (multimodal vision routes here).
    presets = [
        cfg.GLM_47, cfg.GLM_51,
        cfg.QWEN35_27B, cfg.QWEN_VL,
    ]
    offenders = {
        p.name: p.qualified_model_name
        for p in presets
        if p.qualified_model_name not in declared
    }
    assert not offenders, (
        f"config.py presets not declared in opencode.json: {offenders}\n"
        f"declared: {sorted(declared)}"
    )


# NOTE: the former test_vision_config_base_url_matches_opencode_image_provider
# was dropped: the separate A4000 vision model (local-vllm-image, :5504) and the
# config._VLLM_VISION endpoint no longer exist — vision/video route to the
# multimodal local qwen-27B (QWEN_VL = QWEN35_27B on :8082). The "vision is on
# qwen, nothing pins the dead A4000 model" invariant is now covered by
# test_vision_agents_route_to_the_multimodal_qwen_not_the_dead_a4000 below.


# ═══════════════════════════════════════════════════════════════════════════
# 1. ENDPOINT RESILIENCE — vllm_resolve degrades when a role endpoint is down
# ═══════════════════════════════════════════════════════════════════════════


def test_vllm_resolve_falls_back_when_role_endpoint_is_down(monkeypatch):
    """get_llm_endpoint('analytical') must fall back to the UP endpoint when the
    analytical :8083 is unreachable, NOT hard-fail.
    """
    from app import vllm_resolve

    vllm_resolve._MODEL_CACHE.clear()
    monkeypatch.setattr(vllm_resolve, "_get_variant", lambda: "split")

    def fake_served(base_url: str):
        # analytical :8083 is down -> None; the UP fallback :8082 serves a model.
        if base_url == UP_ENDPOINT:
            return "cyankiwi/Qwen3.5-27B-AWQ-BF16-INT8"
        return None

    monkeypatch.setattr(vllm_resolve, "_served_model", fake_served)

    base, model = vllm_resolve.get_llm_endpoint("analytical")
    assert base == UP_ENDPOINT, f"did not fall back to UP endpoint: got {base}"
    assert model, "fell back but returned no served model"
    vllm_resolve._MODEL_CACHE.clear()


def test_vllm_resolve_never_hard_fails_when_everything_is_down(monkeypatch):
    """Even with ALL probes failing, get_llm_endpoint returns a (url, model)
    tuple (static fallback) rather than raising — tools degrade, not crash.
    """
    from app import vllm_resolve

    vllm_resolve._MODEL_CACHE.clear()
    monkeypatch.setattr(vllm_resolve, "_get_variant", lambda: "unknown")
    monkeypatch.setattr(vllm_resolve, "_served_model", lambda _b: None)

    base, model = vllm_resolve.get_llm_endpoint("analytical")
    assert base and model, "resolver hard-failed instead of returning a static fallback"
    vllm_resolve._MODEL_CACHE.clear()


def test_vision_agents_route_to_the_multimodal_qwen_not_the_dead_a4000():
    """Vision-class agents must run on the local qwen-27B (:8082), which is
    multimodal — NOT the separate A4000 vision model (local-vllm-image, :5504),
    which is dropped per requirements (only the one qwen-27B + the small
    emotional-core model are needed). So NO compiled agent may pin the dead
    local-vllm-image model, and the vision agents' model must resolve to a
    declared, on-:8082 provider.
    """
    # No agent should reference the dropped A4000 vision model anymore.
    image_pinned = [
        md.name for md in compiled_agent_files()
        if "local-vllm-image" in md.read_text()
    ]
    assert not image_pinned, (
        f"agents still pin the dropped A4000 vision model: {image_pinned}"
    )
    # Vision-class agents route to the multimodal qwen on :8082.
    from app.agents.config import QWEN_VL
    vision_model = f"{QWEN_VL.provider}/{QWEN_VL.model_id}"
    assert vision_model == "local-vllm-remote/qwen35-27b", vision_model
    assert provider_base_urls().get("local-vllm-remote", "").startswith(
        "http://192.168.0.42:8082"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. MEDIA PERSISTENCE — output must land on the persistent /data volume
# ═══════════════════════════════════════════════════════════════════════════


def test_docker_compose_mounts_the_persistent_data_volume():
    """fren_v4_data must be mounted at /data — the only persistent surface."""
    text = (REPO_ROOT / "docker-compose.yml").read_text()
    assert f"fren_v4_data:{PERSISTENT_VOLUME_MOUNT}" in text, (
        "docker-compose.yml no longer mounts fren_v4_data at /data"
    )


def test_camera_and_screenshot_captures_use_persistent_volume():
    """camera_capture + screenshot must write under the persistent /data mount.

    They currently use ``Path("data/captures")`` (relative -> /app/backend/data),
    which is ephemeral. A correct path is absolute under /data (or derived from a
    /data-anchored settings value).
    """
    for rel in ("app/tools/system/camera_capture.py", "app/tools/system/screenshot.py"):
        src = (REPO_ROOT / "backend" / rel).read_text()
        m = re.search(r'CAPTURES_DIR\s*=\s*Path\((["\'])(.+?)\1\)', src)
        assert m, f"could not find CAPTURES_DIR in {rel}"
        captures_path = m.group(2)
        assert captures_path.startswith(PERSISTENT_VOLUME_MOUNT), (
            f"{rel}: CAPTURES_DIR={captures_path!r} is not under {PERSISTENT_VOLUME_MOUNT} "
            "(ephemeral; lost on container recreate)"
        )


def test_render_output_dir_is_under_persistent_volume():
    """render_and_send's self-review copy dir must be under /data."""
    src = (REPO_ROOT / "scripts" / "render_and_send.py").read_text()
    # The dest_dir for stray-file copies.
    m = re.search(r'dest_dir\s*=\s*(.+)', src)
    assert m, "could not find dest_dir in render_and_send.py"
    line = m.group(1)
    # PROJECT_ROOT / "data" / "rendered" -> ephemeral. Must reference /data.
    assert PERSISTENT_VOLUME_MOUNT in line and "PROJECT_ROOT" not in line, (
        f"render_and_send dest_dir is ephemeral: {line.strip()}"
    )


def test_comfyui_download_lands_outside_tmp():
    """ComfyUI downloads must land on the persistent /data volume, not /tmp.

    Renders pulled from the remote ComfyUI host are written to a download dir
    before send; that dir is now anchored under /data (DATA_DIR-overridable) so
    the render survives a container recreate instead of being lost from /tmp.
    """
    src = (REPO_ROOT / "backend" / "app" / "comfyui" / "client.py").read_text()
    assert "/tmp/comfyui_dl_" not in src, (
        "download_output still writes to /tmp (ephemeral) — anchor it under /data"
    )
    assert PERSISTENT_VOLUME_MOUNT in src, (
        "download_output no longer references the persistent /data volume"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. EMOTION NETWORK — emotional_state writer wired + freshness expectation
# ═══════════════════════════════════════════════════════════════════════════


def test_user_mood_repo_has_a_drift_writer():
    """UserMoodRepo must expose a writer (drift) — not just read-only get/history."""
    from app.db.repos.user_mood import UserMoodRepo

    assert hasattr(UserMoodRepo, "drift"), "UserMoodRepo lost its drift() writer"
    assert hasattr(UserMoodRepo, "get") and hasattr(UserMoodRepo, "history")


def test_user_mood_drift_is_called_somewhere_in_the_codebase():
    """Some non-test module must call UserMoodRepo().drift(...) or the emotion
    state never updates. (VibeStateRepo.drift IS wired — user_mood is not.)
    """
    hits: list[str] = []
    for py in (REPO_ROOT / "backend" / "app").rglob("*.py"):
        text = py.read_text()
        if "UserMoodRepo" in text and ".drift(" in text:
            hits.append(str(py.relative_to(REPO_ROOT)))
    for py in SCRIPTS_DIR.rglob("*.py"):
        text = py.read_text()
        if "UserMoodRepo" in text and ".drift(" in text:
            hits.append(str(py.relative_to(REPO_ROOT)))
    assert hits, "no module calls UserMoodRepo().drift() — emotional_state is never written"


def test_user_mood_freshness_helper_is_available():
    """A freshness expectation needs an updated_at column on the mood state.

    The schema must carry updated_at (the drift() decay logic depends on it); the
    dashboard/health strip can then flag a stale emotional_state row.
    """
    schema = (REPO_ROOT / "backend" / "migrations" / "versions" / "001_initial_schema.py").read_text()
    assert "user_mood_state" in schema, "user_mood_state table missing from schema"
    # The state table must have updated_at for any freshness check to work.
    block = schema[schema.index("user_mood_state"):]
    assert "updated_at" in block[:1200], "user_mood_state lacks updated_at (no freshness check possible)"


# ═══════════════════════════════════════════════════════════════════════════
# 5. complete_run WIRING — runs must not be stuck status=running forever
# ═══════════════════════════════════════════════════════════════════════════


def test_spawn_agent_completes_the_run():
    """spawn.py must call complete_run() so runs don't stay 'running' forever."""
    src = (REPO_ROOT / "backend" / "app" / "telegram" / "spawn.py").read_text()
    assert "complete_run" in src, (
        "spawn.py never calls complete_run() — execution_runs are stuck status=running"
    )


def test_complete_run_exists_and_sets_completed_status():
    """The ledger primitive itself is correct (the wiring, not the repo, is the bug)."""
    src = (REPO_ROOT / "backend" / "app" / "db" / "repos" / "execution_ledger.py").read_text()
    assert "async def complete_run" in src
    assert "completed_at" in src


# ═══════════════════════════════════════════════════════════════════════════
# 6. MISC PARITY — provider keys referenced by config all exist in opencode.json
# ═══════════════════════════════════════════════════════════════════════════


def test_all_config_provider_keys_declared_in_opencode():
    """Every provider key used by a compile-time preset exists in opencode.json."""
    from app.agents import config as cfg

    declared_providers = set((opencode_config().get("provider") or {}).keys())
    # The three live WORKER presets must resolve to declared providers. The
    # split-profile presets (SPLIT_*) belong to the interactive `live_profile`,
    # not the opencode worker fleet, and resolve via vllm_resolve's own probed
    # endpoints — they are intentionally NOT in opencode.json.
    presets = [
        cfg.GLM_47, cfg.GLM_51,
        cfg.QWEN35_27B, cfg.QWEN_VL,
    ]
    offenders = {p.name: p.provider for p in presets if p.provider not in declared_providers}
    assert not offenders, f"config presets reference undeclared providers: {offenders}"


def test_split_endpoints_in_resolver_match_known_hosts():
    """vllm_resolve's split/fallback endpoints stay on the known vLLM hosts.

    Guards against a stray edit pointing the resolver at a host that was never
    part of the fleet.
    """
    from app import vllm_resolve

    all_urls = {u for u, _ in vllm_resolve._SPLIT_ENDPOINTS.values()}
    all_urls |= {u for u, _ in vllm_resolve._SINGLE_MODEL.values()}
    all_urls.add(vllm_resolve._FALLBACK[0])
    for url in all_urls:
        assert url.startswith("http://192.168.0.42:"), (
            f"resolver endpoint on an unexpected host: {url}"
        )
