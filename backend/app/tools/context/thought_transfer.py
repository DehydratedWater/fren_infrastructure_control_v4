"""Thought Transfer — file-based message passing between agents."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field

THOUGHTS_DIR = Path(".agent_workspace/thoughts")


def _ensure_dir() -> None:
    THOUGHTS_DIR.mkdir(parents=True, exist_ok=True)


def _path(key: str) -> Path:
    safe = hashlib.sha256(key.encode()).hexdigest()[:16]
    return THOUGHTS_DIR / f"{safe}.json"


class Input(BaseModel):
    command: str = Field(description="write|read|peek|exists|delete|list|clear")
    key: str = Field(default="", description="Thought key")
    content: str = Field(default="", description="Content to write")


class Output(BaseModel):
    success: bool = True
    key: str = ""
    content: str = ""
    count: int = 0
    exists: bool = False
    error: str = ""
    thoughts: list[dict] = Field(default_factory=list)


class ThoughtTransferTool(ScriptTool[Input, Output]):
    name = "thought_transfer"
    description = "File-based message passing between agents"
    stream_field = "content"

    def execute(self, inp: Input) -> Output:
        cmd = inp.command

        if cmd == "write":
            _ensure_dir()
            path = _path(inp.key)
            data = {"key": inp.key, "content": inp.content, "created_at": datetime.now().isoformat()}
            path.write_text(json.dumps(data, indent=2))
            return Output(success=True, key=inp.key, exists=True)

        if cmd in ("read", "peek"):
            path = _path(inp.key)
            if not path.exists():
                return Output(success=False, error=f"Thought not found: {inp.key}")
            data = json.loads(path.read_text())
            if cmd == "read":
                path.unlink()
            return Output(success=True, key=data["key"], content=data["content"])

        if cmd == "exists":
            path = _path(inp.key)
            return Output(success=True, key=inp.key, exists=path.exists())

        if cmd == "delete":
            path = _path(inp.key)
            if not path.exists():
                return Output(success=False, error=f"Thought not found: {inp.key}")
            path.unlink()
            return Output(success=True, key=inp.key)

        if cmd == "list":
            _ensure_dir()
            thoughts = []
            for p in THOUGHTS_DIR.glob("*.json"):
                try:
                    d = json.loads(p.read_text())
                    thoughts.append({"key": d["key"], "created_at": d["created_at"], "size": len(d["content"])})
                except Exception:
                    continue
            return Output(success=True, count=len(thoughts), thoughts=thoughts)

        if cmd == "clear":
            _ensure_dir()
            count = sum(1 for p in THOUGHTS_DIR.glob("*.json") if not p.unlink())
            # unlink returns None, so count all
            count = 0
            for p in list(THOUGHTS_DIR.glob("*.json")):
                p.unlink()
                count += 1
            return Output(success=True, count=count)

        return Output(success=False, error=f"Unknown command: {cmd}")


if __name__ == "__main__":
    ThoughtTransferTool.run()
