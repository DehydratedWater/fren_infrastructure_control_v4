"""File-based scheduler — runs agent jobs on cron expressions inside the bot process."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from croniter import croniter

from app.settings import get_settings
from app.telegram.spawn import spawn_agent

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCHEDULE_PATH = _PROJECT_ROOT / "config" / "schedule.yml"
_ONE_TIME_PATH = _PROJECT_ROOT / "config" / "one_time_schedule.yml"
_STATE_PATH = _PROJECT_ROOT / "data" / "scheduler_state.json"
_RUN_LOGS_PATH = _PROJECT_ROOT / "run_logs"


def _load_schedule() -> dict[str, Any]:
    """Read schedule.yml, returning an empty dict on error."""
    try:
        with open(_SCHEDULE_PATH) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data.get("jobs", {})
    except FileNotFoundError:
        logger.warning("Schedule file not found: %s", _SCHEDULE_PATH)
    except Exception:
        logger.exception("Failed to parse schedule file")
    return {}


def _load_one_time_schedule() -> dict[str, Any]:
    """Read one_time_schedule.yml, returning an empty dict on error."""
    try:
        with open(_ONE_TIME_PATH) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data.get("jobs", {})
    except FileNotFoundError:
        pass
    except Exception:
        logger.exception("Failed to parse one-time schedule file")
    return {}


def _save_one_time_schedule(jobs: dict[str, Any]) -> None:
    """Atomically write the one-time schedule YAML."""
    _ONE_TIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=_ONE_TIME_PATH.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump({"jobs": jobs}, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        os.replace(tmp, _ONE_TIME_PATH)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _is_one_time_due(at_expr: str, last_run: str | None, now: datetime) -> bool:
    """Check if a one-time job is due.

    ``at_expr`` is either an ISO 8601 datetime or a cron expression.
    ISO datetimes fire once when now >= at and the job hasn't run since that time.
    Cron expressions use the regular ``_is_due()`` logic.
    """
    try:
        target = datetime.fromisoformat(at_expr)
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        if now < target:
            return False
        if last_run is None:
            return True
        try:
            last_dt = datetime.fromisoformat(last_run)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=UTC)
        except ValueError:
            return True
        return target > last_dt
    except ValueError:
        # Not ISO — treat as cron expression
        return _is_due(at_expr, last_run, now)


def _load_state() -> dict[str, str]:
    """Load last-run timestamps per job from the state file."""
    try:
        with open(_STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state: dict[str, str]) -> None:
    """Atomically write state to disk (temp + rename)."""
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=_STATE_PATH.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, _STATE_PATH)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _is_due(cron_expr: str, last_run: str | None, now: datetime) -> bool:
    """Return True if the job should fire in this tick window.

    Uses croniter.get_prev() from *now* and checks whether the most recent
    scheduled instant is after the last recorded run.
    """
    try:
        it = croniter(cron_expr, now)
        prev_fire = it.get_prev(datetime).replace(tzinfo=UTC)
    except (ValueError, KeyError):
        logger.warning("Invalid cron expression: %s", cron_expr)
        return False

    if last_run is None:
        return True

    try:
        last_dt = datetime.fromisoformat(last_run)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=UTC)
    except ValueError:
        return True

    return prev_fire > last_dt


class Scheduler:
    """Tick-based cron scheduler running inside the bot's asyncio loop."""

    def __init__(self) -> None:
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._running_jobs: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        logger.info("Scheduler starting (schedule=%s)", _SCHEDULE_PATH)
        await self._reconcile_stale_runs()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        logger.info("Scheduler stopping (%d jobs running)", len(self._running_jobs))
        self._stop_event.set()

        # Cancel running job subprocesses
        for job_id, task in self._running_jobs.items():
            logger.info("Cancelling running job: %s", job_id)
            task.cancel()

        if self._running_jobs:
            await asyncio.gather(*self._running_jobs.values(), return_exceptions=True)

        if self._task:
            await asyncio.gather(self._task, return_exceptions=True)

        logger.info("Scheduler stopped")

    async def _loop(self) -> None:
        logger.info("Scheduler started — ticking every 60s")
        while not self._stop_event.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("Scheduler tick failed")
            # Wait 60s or until stop_event
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=60)
                break  # stop_event was set
            except TimeoutError:
                pass  # normal 60s tick

    async def _tick(self) -> None:
        jobs = _load_schedule()
        one_time_jobs = _load_one_time_schedule()
        if not jobs and not one_time_jobs:
            return

        state = _load_state()
        now = datetime.now(UTC)

        # --- Regular recurring jobs ---
        for job_id, cfg in jobs.items():
            if not isinstance(cfg, dict):
                continue
            if not cfg.get("enabled", True):
                continue
            cron_expr = cfg.get("cron")
            if not cron_expr:
                continue

            # Skip if already running
            if job_id in self._running_jobs and not self._running_jobs[job_id].done():
                continue

            # Clean up finished tasks
            if job_id in self._running_jobs and self._running_jobs[job_id].done():
                del self._running_jobs[job_id]

            # Circuit breaker: skip if 5+ failures within the last 2 hours
            # Uses exponential backoff cooldown: 1h → 2h → 4h → 8h (max)
            # After 3 cycles, stops retrying until manual reset
            ftimes_key = f"{job_id}_failure_times"
            cycle_key = f"{job_id}_cb_cycle"
            failure_times = state.get(ftimes_key, [])
            if isinstance(failure_times, list):
                # Prune failures older than 2 hours
                cutoff = (now - __import__("datetime").timedelta(hours=2)).isoformat()
                failure_times = [t for t in failure_times if isinstance(t, str) and t > cutoff]
                state[ftimes_key] = failure_times
            else:
                failure_times = []
            # Also migrate old counter format to new format
            old_key = f"{job_id}_consecutive_failures"
            if old_key in state:
                del state[old_key]
                _save_state(state)
            if len(failure_times) >= 5:
                cb_cycle = state.get(cycle_key, 0)
                if isinstance(cb_cycle, int) and cb_cycle >= 3:
                    continue  # Permanently paused — needs manual reset
                cooldown_seconds = min(3600 * (2**cb_cycle), 28800)  # 1h, 2h, 4h, max 8h
                last_fail = failure_times[-1] if failure_times else ""
                if last_fail:
                    try:
                        last_fail_dt = datetime.fromisoformat(last_fail)
                        if last_fail_dt.tzinfo is None:
                            last_fail_dt = last_fail_dt.replace(tzinfo=UTC)
                        if (now - last_fail_dt).total_seconds() < cooldown_seconds:
                            continue  # Still in cooldown
                    except ValueError:
                        pass
                # Cooldown expired — clear window and increment cycle
                state[ftimes_key] = []
                state[cycle_key] = (cb_cycle if isinstance(cb_cycle, int) else 0) + 1
                _save_state(state)

            if _is_due(cron_expr, state.get(job_id), now):
                # Record last-run BEFORE starting to prevent double-fire
                state[job_id] = now.isoformat()
                _save_state(state)
                logger.info("Firing job: %s (%s)", job_id, cfg.get("description", ""))
                self._running_jobs[job_id] = asyncio.create_task(self._execute_job(job_id, cfg))

        # --- One-time / limited-run jobs ---
        for job_id, cfg in one_time_jobs.items():
            if not isinstance(cfg, dict):
                continue
            at_expr = cfg.get("at")
            if not at_expr:
                continue

            # Skip if already running
            if job_id in self._running_jobs and not self._running_jobs[job_id].done():
                continue
            if job_id in self._running_jobs and self._running_jobs[job_id].done():
                del self._running_jobs[job_id]

            if _is_one_time_due(str(at_expr), state.get(job_id), now):
                state[job_id] = now.isoformat()
                _save_state(state)
                logger.info("Firing one-time job: %s (%s)", job_id, cfg.get("description", ""))
                self._running_jobs[job_id] = asyncio.create_task(self._execute_job(job_id, cfg, one_time=True))

    async def _execute_job(self, job_id: str, cfg: dict[str, Any], *, one_time: bool = False) -> None:
        from app.telegram.state import get_postfix, get_scheduler_model

        base_agent = cfg.get("agent", "")
        prompt = cfg.get("prompt", "")
        timeout = cfg.get("timeout", 600)
        description = cfg.get("description", job_id)
        is_script = base_agent.startswith("script:")

        if not is_script:
            job_model = cfg.get("model") or get_scheduler_model()
            postfix = get_postfix(job_model)
            agent = f"{base_agent}{postfix}"
        else:
            postfix = ""
            agent = base_agent

        execution_id = f"sched_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(UTC)

        try:
            from app.db.repos.cron import CronExecutionsRepo

            await CronExecutionsRepo().reconcile_stale_running(
                mode=job_id,
                older_than_seconds=max(int(timeout) + 60, 600),
                error_output=f"Recovered stale running execution before starting new {job_id} job",
            )
        except Exception:
            logger.exception("Failed to reconcile stale executions for %s", job_id)

        # Log start to DB
        try:
            from app.db.repos.cron import CronExecutionsRepo

            repo = CronExecutionsRepo()
            await repo.create(
                execution_id=execution_id,
                mode=job_id,
                started_at=started_at,
                triggered_by="scheduler",
            )
        except Exception:
            logger.exception("Failed to log execution start for %s", job_id)

        settings = get_settings()
        project_root = Path(settings.project_root)

        from app.telegram.state import get_postfix, get_tts_model

        run_id = f"run_{uuid.uuid4().hex[:16]}"

        env = {
            **os.environ,
            "XDG_DATA_HOME": str(project_root / ".opencode" / "data"),
            "FREN_MODEL_POSTFIX": postfix,
            "FREN_TTS_POSTFIX": get_postfix(get_tts_model()) if not is_script else "",
            "FREN_RUN_ID": run_id,
            "FREN_JOB_ID": job_id,
        }

        if is_script:
            script_path = base_agent.removeprefix("script:")
            cmd = ["uv", "run", script_path]
            if prompt:
                cmd.extend(prompt.split())
        else:
            # Wrap the task prompt with a scheduled-trigger header when targeting
            # chat/persona agents — they see both user messages and scheduled
            # instructions and must not confuse the two.
            if base_agent.startswith("persona/twily_chat") or base_agent.startswith("persona/fren_orchestrator"):
                prompt = (
                    "## ⚙️ SCHEDULED TRIGGER — NOT A USER MESSAGE\n"
                    f"This prompt was fired by the scheduler (job: {job_id}). The user did NOT send it. "
                    "The user typed nothing. Do NOT quote, praise, or thank the user for this content — "
                    "they have no idea it exists. Reach out first-person as if initiating the interaction.\n\n"
                    f"## TASK INSTRUCTION:\n{prompt}"
                )

            # Prepend conversation digest to agent prompt for situational context
            enriched_prompt = await self._enrich_prompt(prompt)

            cmd = [
                "uv",
                "run",
                "scripts/opencode_manager.py",
                "run",
                "--agent",
                agent,
                "--prefix",
                job_id,
                "--no-server",
                "--no-attach",
                enriched_prompt,
            ]

        logger.info("[%s] Starting agent %s (timeout=%ds)", job_id, agent, timeout)
        return_code = 1
        stderr_text: str | None = None
        status = "failed"
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(project_root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return_code = proc.returncode or 0
            if return_code != 0:
                stderr_text = stderr.decode("utf-8", errors="replace")[:2000] if stderr else None
                logger.error("[%s] Agent failed (exit %d): %s", job_id, return_code, (stderr_text or "")[:500])
            else:
                status = "completed"
                logger.info("[%s] Completed successfully", job_id)
        except TimeoutError:
            logger.error("[%s] Timed out after %ds", job_id, timeout)
            stderr_text = f"TIMEOUT: exceeded {timeout}s budget"
            return_code = 124
            if proc is not None:
                proc.kill()
        except asyncio.CancelledError:
            logger.info("[%s] Cancelled by scheduler shutdown", job_id)
            stderr_text = "CANCELLED: scheduler shutdown"
            return_code = 130
            status = "cancelled"
            if proc is not None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
        except Exception:
            logger.exception("[%s] Failed to run agent", job_id)
            stderr_text = "UNEXPECTED ERROR: scheduler failed to run job"
            return_code = 1

        # Log completion to DB (with stderr capture for failures)
        log_file = self._find_job_log(job_id, started_at)
        try:
            from app.db.repos.cron import CronExecutionsRepo

            repo = CronExecutionsRepo()
            await repo.complete(
                execution_id,
                exit_code=return_code,
                status=status,
                error_output=stderr_text if return_code != 0 else None,
                log_file=str(log_file) if log_file else None,
            )
        except Exception:
            logger.exception("Failed to log execution completion for %s", job_id)

        # Post-run persona_prose delivery hook. For script jobs this is a no-op
        # (no agent was fired). For agent jobs it reads the guidance from the
        # ledger and delivers via persona_prose IF the agent emitted one.
        # Excluded agents are skipped. CRITICALLY: synth_fallback=False so
        # background scheduler jobs (event_extraction, conversation_digest,
        # periodic_check, etc.) that aren't supposed to message the user don't
        # have a fallback fired at them — that would spam the user every tick.
        # Scheduler-fired persona-voice agents (nudge_strategist, evening_focus,
        # winddown, periodic) deliver via their own emit_guidance call inline,
        # which writes persona_response and the hook short-circuits at step 1.
        if not is_script and return_code == 0:
            try:
                from app.telegram.persona_prose import deliver_guidance_from_ledger, is_excluded_agent

                if not is_excluded_agent(agent):
                    await deliver_guidance_from_ledger(run_id=run_id, synth_fallback=False)
            except Exception:
                logger.exception("[%s] post-run persona_prose delivery failed", job_id)

        logger.info("[%s] %s — %s", job_id, description, status)

        # Circuit breaker: track failures in a 2-hour time window
        cb_state = _load_state()
        ftimes_key = f"{job_id}_failure_times"
        cycle_key = f"{job_id}_cb_cycle"
        failure_times = cb_state.get(ftimes_key, [])
        if not isinstance(failure_times, list):
            failure_times = []
        # Migrate old format
        old_key = f"{job_id}_consecutive_failures"
        if old_key in cb_state:
            del cb_state[old_key]
        if return_code != 0:
            failure_times.append(datetime.now(UTC).isoformat())
            # Prune older than 2 hours
            cutoff = (datetime.now(UTC) - __import__("datetime").timedelta(hours=2)).isoformat()
            failure_times = [t for t in failure_times if t > cutoff]
            cb_state[ftimes_key] = failure_times
            if len(failure_times) == 5:
                cb_cycle = cb_state.get(cycle_key, 0)
                if not isinstance(cb_cycle, int):
                    cb_cycle = 0
                cooldown_h = min(2**cb_cycle, 8)
                logger.warning(
                    "[%s] Circuit breaker: 5 failures in 2h — pausing for %dh (cycle %d)", job_id, cooldown_h, cb_cycle
                )
                try:
                    await self._send_circuit_breaker_alert(
                        job_id, description, cooldown_hours=cooldown_h, cycle=cb_cycle
                    )
                except Exception:
                    logger.debug("Failed to send circuit breaker alert", exc_info=True)
        else:
            cb_state[ftimes_key] = []
            # Reset cycle on success — the job recovered
            if cb_state.get(cycle_key, 0):
                cb_state[cycle_key] = 0
        _save_state(cb_state)

        if one_time:
            self._complete_one_time_job(job_id)

        if status == "cancelled":
            raise asyncio.CancelledError

    @staticmethod
    async def _send_circuit_breaker_alert(
        job_id: str, description: str, *, cooldown_hours: int = 1, cycle: int = 0
    ) -> None:
        """Send Telegram alert when a job hits the circuit breaker threshold."""
        from telegram import Bot

        settings = get_settings()
        bot = Bot(token=settings.bot_token)
        await bot.initialize()
        if cycle >= 2:
            msg = (
                f"*adjusts glasses nervously* Job '{description}' ({job_id}) has failed 5 times AGAIN "
                f"(cycle {cycle + 1}). Pausing for {cooldown_hours}h. This keeps happening — probably needs a fix!"
            )
        else:
            msg = (
                f"*adjusts glasses* Job '{description}' ({job_id}) has failed 5 times in a row. "
                f"Pausing it for {cooldown_hours}h. Might need attention!"
            )
        await bot.send_message(chat_id=settings.chat_id, text=msg)

    async def _reconcile_stale_runs(self) -> None:
        try:
            from app.db.repos.cron import CronExecutionsRepo

            reconciled = await CronExecutionsRepo().reconcile_stale_running(
                older_than_seconds=600,
                error_output="Recovered stale running execution on scheduler startup",
            )
            if reconciled:
                logger.warning("Reconciled %d stale running cron executions on startup", reconciled)
        except Exception:
            logger.exception("Failed to reconcile stale running cron executions on startup")

    @staticmethod
    def _find_job_log(job_id: str, started_at: datetime) -> Path | None:
        if not _RUN_LOGS_PATH.exists():
            return None

        candidates: list[Path] = []
        for path in _RUN_LOGS_PATH.glob(f"{job_id}_*.log"):
            try:
                stat = path.stat()
            except OSError:
                continue
            modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            if modified >= started_at - timedelta(seconds=30):
                candidates.append(path)

        if not candidates:
            return None

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    @staticmethod
    async def _enrich_prompt(prompt: str) -> str:
        """Prepend conversation digest and user rules to a prompt for situational context."""
        prefix_parts: list[str] = []

        # User rules — persistent directives
        try:
            from app.db.repos.user_rules import UserRulesRepo

            rules_text = await UserRulesRepo().format_rules_prompt()
            if rules_text:
                prefix_parts.append(rules_text)
        except Exception:
            logger.debug("Failed to fetch user rules", exc_info=True)

        # Agent lessons — learned from past mistakes
        try:
            from app.db.repos.agent_lessons import AgentLessonsRepo

            lessons_text = await AgentLessonsRepo().format_lessons_prompt()
            if lessons_text:
                prefix_parts.append(lessons_text)
        except Exception:
            logger.debug("Failed to fetch agent lessons", exc_info=True)

        # Conversation digest — rolling situational summary
        try:
            from app.db.repos.agent_notes import AgentNotesRepo

            note = await AgentNotesRepo().get("conversation_digest")
            if note and note.get("note_value"):
                val = note["note_value"]
                digest = val.get("digest", "") if isinstance(val, dict) else str(val)
                if digest:
                    prefix_parts.append(digest)
        except Exception:
            logger.debug("Failed to fetch conversation digest", exc_info=True)

        # Chat history — last 24h, capped at 40k chars (~10k tokens)
        try:
            from app.db.repos.chat import ChatMessagesRepo

            repo = ChatMessagesRepo()
            since_ts = datetime.now(UTC).timestamp() - (24 * 3600)
            msgs = await repo.get_history(days=2, limit=300, clearance="full")
            msgs = [m for m in msgs if m.get("timestamp_unix", 0) > since_ts]
            if msgs:
                lines = ["## Chat History (last 24h)"]
                total = 0
                for m in msgs:
                    ts = str(m.get("timestamp", ""))[:16]
                    sender = m.get("sender", "?")
                    text = str(m.get("message", ""))[:300]
                    line = f"[{ts}] {sender}: {text}"
                    if total + len(line) > 40000:
                        break
                    lines.append(line)
                    total += len(line) + 1
                prefix_parts.append("\n".join(lines))
        except Exception:
            logger.debug("Failed to fetch chat history for enrichment", exc_info=True)

        if prefix_parts:
            prefix = "\n\n".join(prefix_parts)
            return f"{prefix}\n\n---\n\n{prompt}"
        return prompt

    @staticmethod
    def _complete_one_time_job(job_id: str) -> None:
        """Decrement runs_left for a one-time job; remove it when exhausted."""
        jobs = _load_one_time_schedule()
        if job_id not in jobs:
            return
        runs_left = jobs[job_id].get("runs_left", 1) - 1
        if runs_left <= 0:
            logger.info("One-time job exhausted, removing: %s", job_id)
            del jobs[job_id]
        else:
            jobs[job_id]["runs_left"] = runs_left
            logger.info("One-time job %s: %d runs left", job_id, runs_left)
        _save_one_time_schedule(jobs)
