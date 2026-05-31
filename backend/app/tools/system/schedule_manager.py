"""Schedule Manager — view and modify the file-based scheduler config."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from src import ScriptTool
from pydantic import BaseModel, Field

_LOCAL_TZ = ZoneInfo("Europe/Warsaw")


def _utc_str_to_local(iso_str: str | None) -> str | None:
    """Convert a UTC ISO string to Europe/Warsaw local time."""
    if not iso_str:
        return iso_str
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is not None:
            return dt.astimezone(_LOCAL_TZ).isoformat()
    except (ValueError, TypeError):
        pass
    return iso_str


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SCHEDULE_PATH = _PROJECT_ROOT / "config" / "schedule.yml"
_ONE_TIME_PATH = _PROJECT_ROOT / "config" / "one_time_schedule.yml"
_STATE_PATH = _PROJECT_ROOT / "data" / "scheduler_state.json"


def _load_schedule() -> dict[str, Any]:
    try:
        with open(_SCHEDULE_PATH) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _save_schedule(data: dict[str, Any]) -> None:
    _SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SCHEDULE_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _load_one_time_schedule() -> dict[str, Any]:
    try:
        with open(_ONE_TIME_PATH) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _save_one_time_schedule(data: dict[str, Any]) -> None:
    _ONE_TIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_ONE_TIME_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _load_state() -> dict[str, str]:
    try:
        with open(_STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class Input(BaseModel):
    command: str = Field(
        description="list|get|enable|disable|update-cron|update-prompt|add|remove|status"
        "|add-one-time|list-one-time|remove-one-time"
    )
    job_id: str = Field(default="", description="Job identifier")
    cron: str = Field(default="", description="Cron expression (for add/update-cron)")
    at: str = Field(default="", description="ISO 8601 datetime or cron expression (for add-one-time)")
    agent: str = Field(default="", description="Agent path (for add/add-one-time)")
    prompt: str = Field(default="", description="Agent prompt (for add/add-one-time/update-prompt)")
    description: str = Field(default="", description="Job description (for add/add-one-time)")
    timeout: int = Field(default=600, description="Timeout in seconds (for add/add-one-time)")
    runs_left: int = Field(default=1, description="Number of runs before removal (for add-one-time)")


class Output(BaseModel):
    success: bool = True
    job: dict = Field(default_factory=dict)
    jobs: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class ScheduleManagerTool(ScriptTool[Input, Output]):
    name = "schedule_manager"
    description = "View and modify the file-based agent scheduler"

    def execute(self, inp: Input) -> Output:
        cmd = inp.command

        if cmd == "list":
            return self._list_jobs()
        if cmd == "get":
            return self._get_job(inp.job_id)
        if cmd == "enable":
            return self._set_enabled(inp.job_id, True)
        if cmd == "disable":
            return self._set_enabled(inp.job_id, False)
        if cmd == "update-cron":
            return self._update_cron(inp.job_id, inp.cron)
        if cmd == "update-prompt":
            return self._update_prompt(inp.job_id, inp.prompt)
        if cmd == "add":
            return self._add_job(inp)
        if cmd == "remove":
            return self._remove_job(inp.job_id)
        if cmd == "status":
            return self._status()
        if cmd == "add-one-time":
            return self._add_one_time_job(inp)
        if cmd == "list-one-time":
            return self._list_one_time_jobs()
        if cmd == "remove-one-time":
            return self._remove_one_time_job(inp.job_id)

        return Output(success=False, error=f"Unknown command: {cmd}")

    def _list_jobs(self) -> Output:
        data = _load_schedule()
        jobs = data.get("jobs", {})
        result = []
        for jid, cfg in jobs.items():
            if isinstance(cfg, dict):
                result.append({"id": jid, **cfg})
        return Output(success=True, jobs=result, count=len(result))

    def _get_job(self, job_id: str) -> Output:
        if not job_id:
            return Output(success=False, error="job_id is required")
        data = _load_schedule()
        job = data.get("jobs", {}).get(job_id)
        if not job:
            return Output(success=False, error=f"Job not found: {job_id}")
        return Output(success=True, job={"id": job_id, **job})

    def _set_enabled(self, job_id: str, enabled: bool) -> Output:
        if not job_id:
            return Output(success=False, error="job_id is required")
        data = _load_schedule()
        jobs = data.get("jobs", {})
        if job_id not in jobs:
            return Output(success=False, error=f"Job not found: {job_id}")
        jobs[job_id]["enabled"] = enabled
        _save_schedule(data)
        return Output(success=True, job={"id": job_id, **jobs[job_id]})

    def _update_cron(self, job_id: str, cron: str) -> Output:
        if not job_id:
            return Output(success=False, error="job_id is required")
        if not cron:
            return Output(success=False, error="cron expression is required")
        data = _load_schedule()
        jobs = data.get("jobs", {})
        if job_id not in jobs:
            return Output(success=False, error=f"Job not found: {job_id}")
        jobs[job_id]["cron"] = cron
        _save_schedule(data)
        return Output(success=True, job={"id": job_id, **jobs[job_id]})

    def _update_prompt(self, job_id: str, prompt: str) -> Output:
        if not job_id:
            return Output(success=False, error="job_id is required")
        if not prompt:
            return Output(success=False, error="prompt is required")
        data = _load_schedule()
        jobs = data.get("jobs", {})
        if job_id not in jobs:
            return Output(success=False, error=f"Job not found: {job_id}")
        jobs[job_id]["prompt"] = prompt
        _save_schedule(data)
        return Output(success=True, job={"id": job_id, **jobs[job_id]})

    def _add_job(self, inp: Input) -> Output:
        if not inp.job_id:
            return Output(success=False, error="job_id is required")
        if not inp.cron:
            return Output(success=False, error="cron expression is required")
        if not inp.agent:
            return Output(success=False, error="agent is required")
        data = _load_schedule()
        if "jobs" not in data:
            data["jobs"] = {}
        if inp.job_id in data["jobs"]:
            return Output(success=False, error=f"Job already exists: {inp.job_id}")
        data["jobs"][inp.job_id] = {
            "cron": inp.cron,
            "agent": inp.agent,
            "prompt": inp.prompt or f"Run {inp.job_id}",
            "description": inp.description or inp.job_id,
            "enabled": True,
            "timeout": inp.timeout,
        }
        _save_schedule(data)
        return Output(success=True, job={"id": inp.job_id, **data["jobs"][inp.job_id]})

    def _remove_job(self, job_id: str) -> Output:
        if not job_id:
            return Output(success=False, error="job_id is required")
        data = _load_schedule()
        jobs = data.get("jobs", {})
        if job_id not in jobs:
            return Output(success=False, error=f"Job not found: {job_id}")
        removed = jobs.pop(job_id)
        _save_schedule(data)
        return Output(success=True, job={"id": job_id, **removed})

    def _status(self) -> Output:
        data = _load_schedule()
        state = _load_state()
        jobs = data.get("jobs", {})
        result = []
        for jid, cfg in jobs.items():
            if isinstance(cfg, dict):
                entry = {"id": jid, **cfg}
                entry["last_run"] = _utc_str_to_local(state.get(jid))
                result.append(entry)
        return Output(success=True, jobs=result, count=len(result))

    # --- One-time schedule commands ---

    def _add_one_time_job(self, inp: Input) -> Output:
        if not inp.job_id:
            return Output(success=False, error="job_id is required")
        if not inp.at:
            return Output(success=False, error="at (ISO datetime or cron) is required")
        if not inp.agent:
            return Output(success=False, error="agent is required")
        data = _load_one_time_schedule()
        if "jobs" not in data:
            data["jobs"] = {}
        if inp.job_id in data["jobs"]:
            return Output(success=False, error=f"One-time job already exists: {inp.job_id}")
        data["jobs"][inp.job_id] = {
            "at": inp.at,
            "agent": inp.agent,
            "prompt": inp.prompt or f"Run {inp.job_id}",
            "description": inp.description or inp.job_id,
            "timeout": inp.timeout,
            "runs_left": inp.runs_left,
        }
        _save_one_time_schedule(data)
        return Output(success=True, job={"id": inp.job_id, **data["jobs"][inp.job_id]})

    def _list_one_time_jobs(self) -> Output:
        data = _load_one_time_schedule()
        jobs = data.get("jobs", {})
        result = []
        for jid, cfg in jobs.items():
            if isinstance(cfg, dict):
                result.append({"id": jid, **cfg})
        return Output(success=True, jobs=result, count=len(result))

    def _remove_one_time_job(self, job_id: str) -> Output:
        if not job_id:
            return Output(success=False, error="job_id is required")
        data = _load_one_time_schedule()
        jobs = data.get("jobs", {})
        if job_id not in jobs:
            return Output(success=False, error=f"One-time job not found: {job_id}")
        removed = jobs.pop(job_id)
        _save_one_time_schedule(data)
        return Output(success=True, job={"id": job_id, **removed})


if __name__ == "__main__":
    ScheduleManagerTool.run()
