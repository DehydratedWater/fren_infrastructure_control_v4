"""Lock Manager — single-instance enforcement via file-based locks."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field

LOCKS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "locks"


class Input(BaseModel):
    command: str = Field(description="check|acquire|release|force-release|list")
    name: str = Field(default="", description="Lock name (e.g. master_organizer)")
    pid: int = Field(default=0, description="PID to associate with lock (default: current process)")
    metadata: str = Field(default="", description="Optional JSON metadata string")


class Output(BaseModel):
    success: bool = True
    locked: bool = False
    owner_pid: int = 0
    owner_alive: bool = False
    started_at: str = ""
    metadata: dict = Field(default_factory=dict)
    locks: list[dict] = Field(default_factory=list)
    error: str = ""


def _is_pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _lock_path(name: str) -> Path:
    return LOCKS_DIR / f"{name}.lock"


def _read_lock(name: str) -> dict | None:
    """Read lock file, return None if not found or invalid."""
    path = _lock_path(name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_lock(name: str, data: dict) -> None:
    """Write lock file."""
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    _lock_path(name).write_text(json.dumps(data, indent=2))


def _remove_lock(name: str) -> None:
    """Remove lock file."""
    path = _lock_path(name)
    path.unlink(missing_ok=True)


class LockManagerTool(ScriptTool[Input, Output]):
    name = "lock_manager"
    description = "Single-instance enforcement via file-based locks"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        cmd = inp.command

        if cmd == "check":
            return self._check(inp.name)
        if cmd == "acquire":
            return self._acquire(inp.name, inp.pid or os.getpid(), inp.metadata)
        if cmd == "release":
            return self._release(inp.name, inp.pid or os.getpid())
        if cmd == "force-release":
            return self._force_release(inp.name)
        if cmd == "list":
            return self._list_locks()

        return Output(success=False, error=f"Unknown command: {cmd}")

    def _check(self, name: str) -> Output:
        if not name:
            return Output(success=False, error="name is required")

        lock = _read_lock(name)
        if lock is None:
            return Output(success=True, locked=False)

        pid = lock.get("pid", 0)
        alive = _is_pid_alive(pid) if pid else False

        # Auto-clean stale lock (dead PID)
        if not alive:
            _remove_lock(name)
            return Output(success=True, locked=False)

        return Output(
            success=True,
            locked=True,
            owner_pid=pid,
            owner_alive=alive,
            started_at=lock.get("started_at", ""),
            metadata=lock.get("metadata", {}),
        )

    def _acquire(self, name: str, pid: int, metadata_str: str) -> Output:
        if not name:
            return Output(success=False, error="name is required")

        # Check existing lock
        lock = _read_lock(name)
        if lock is not None:
            existing_pid = lock.get("pid", 0)
            if _is_pid_alive(existing_pid):
                return Output(
                    success=False,
                    locked=True,
                    owner_pid=existing_pid,
                    owner_alive=True,
                    started_at=lock.get("started_at", ""),
                    metadata=lock.get("metadata", {}),
                    error=f"Lock '{name}' already held by PID {existing_pid}",
                )
            # Stale lock — clean and proceed
            _remove_lock(name)

        # Parse metadata
        meta = {}
        if metadata_str:
            try:
                meta = json.loads(metadata_str)
            except json.JSONDecodeError:
                meta = {"raw": metadata_str}

        now = datetime.now(UTC).isoformat()
        _write_lock(name, {"pid": pid, "started_at": now, "metadata": meta})

        return Output(
            success=True,
            locked=True,
            owner_pid=pid,
            owner_alive=True,
            started_at=now,
            metadata=meta,
        )

    def _release(self, name: str, pid: int) -> Output:
        if not name:
            return Output(success=False, error="name is required")

        lock = _read_lock(name)
        if lock is None:
            return Output(success=True, locked=False)

        existing_pid = lock.get("pid", 0)
        if existing_pid != pid:
            return Output(
                success=False,
                locked=True,
                owner_pid=existing_pid,
                owner_alive=_is_pid_alive(existing_pid),
                error=f"Lock '{name}' owned by PID {existing_pid}, not {pid}",
            )

        _remove_lock(name)
        return Output(success=True, locked=False)

    def _force_release(self, name: str) -> Output:
        if not name:
            return Output(success=False, error="name is required")

        _remove_lock(name)
        return Output(success=True, locked=False)

    def _list_locks(self) -> Output:
        if not LOCKS_DIR.exists():
            return Output(success=True, locks=[])

        locks = []
        for lock_file in LOCKS_DIR.glob("*.lock"):
            try:
                data = json.loads(lock_file.read_text())
                pid = data.get("pid", 0)
                alive = _is_pid_alive(pid) if pid else False
                if not alive:
                    # Auto-clean stale locks
                    lock_file.unlink(missing_ok=True)
                    continue
                locks.append(
                    {
                        "name": lock_file.stem,
                        "pid": pid,
                        "alive": alive,
                        "started_at": data.get("started_at", ""),
                        "metadata": data.get("metadata", {}),
                    }
                )
            except (json.JSONDecodeError, OSError):
                lock_file.unlink(missing_ok=True)

        return Output(success=True, locks=locks)


if __name__ == "__main__":
    LockManagerTool.run()
