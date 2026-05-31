"""Workflow master file operations — safe file CRUD in allowed directories."""

from __future__ import annotations

from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field

from app.settings import get_settings

ALLOWED_DIRS = [".opencode/agents/workflows", "workflow_scripts"]


def _resolve(file_path: str) -> tuple[bool, str]:
    """Resolve and validate file path is in allowed directories."""
    root = Path(get_settings().project_root)
    try:
        resolved = (root / file_path).resolve() if not Path(file_path).is_absolute() else Path(file_path).resolve()
    except Exception as e:
        return False, f"Invalid path: {e}"

    for d in ALLOWED_DIRS:
        try:
            resolved.relative_to((root / d).resolve())
            return True, str(resolved)
        except ValueError:
            continue
    return False, f"Path not in allowed directories: {ALLOWED_DIRS}"


class Input(BaseModel):
    command: str = Field(description="create|modify|append|read|list|delete")
    path: str = Field(default="", description="File path (relative to project root)")
    content: str = Field(default="", description="File content")
    directory: str = Field(default="", description="Directory to list: workflows|workflow_scripts")


class Output(BaseModel):
    success: bool = True
    path: str = ""
    content: str = ""
    files: list[dict] = Field(default_factory=list)
    message: str = ""
    error: str = ""


class WmFileOperationsTool(ScriptTool[Input, Output]):
    name = "wm_file_operations"
    description = "Safe file operations for workflow creation (restricted directories)"

    def execute(self, inp: Input) -> Output:
        if inp.command == "create":
            return self._create(inp.path, inp.content)
        if inp.command == "modify":
            return self._modify(inp.path, inp.content)
        if inp.command == "append":
            return self._append(inp.path, inp.content)
        if inp.command == "read":
            return self._read(inp.path)
        if inp.command == "list":
            return self._list(inp.directory)
        if inp.command == "delete":
            return self._delete(inp.path)
        return Output(success=False, error=f"Unknown command: {inp.command}")

    def _create(self, file_path: str, content: str) -> Output:
        ok, result = _resolve(file_path)
        if not ok:
            return Output(success=False, error=result)
        p = Path(result)
        if p.exists():
            return Output(success=False, error=f"File exists: {p}. Use 'modify' to update.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return Output(success=True, path=str(p), message=f"Created: {file_path}")

    def _modify(self, file_path: str, content: str) -> Output:
        ok, result = _resolve(file_path)
        if not ok:
            return Output(success=False, error=result)
        p = Path(result)
        if not p.exists():
            return Output(success=False, error=f"File not found: {p}. Use 'create'.")
        p.write_text(content)
        return Output(success=True, path=str(p), message=f"Modified: {file_path}")

    def _append(self, file_path: str, content: str) -> Output:
        ok, result = _resolve(file_path)
        if not ok:
            return Output(success=False, error=result)
        p = Path(result)
        if not p.exists():
            return Output(success=False, error=f"File not found: {p}. Use 'create'.")
        with p.open("a") as f:
            f.write(content)
        return Output(success=True, path=str(p), message=f"Appended to: {file_path}")

    def _read(self, file_path: str) -> Output:
        ok, result = _resolve(file_path)
        if not ok:
            return Output(success=False, error=result)
        p = Path(result)
        if not p.exists():
            return Output(success=False, error=f"File not found: {p}")
        return Output(success=True, path=str(p), content=p.read_text())

    def _list(self, directory: str) -> Output:
        root = Path(get_settings().project_root)
        dir_map = {"workflows": ".opencode/agents/workflows", "workflow_scripts": "workflow_scripts"}
        if directory not in dir_map:
            return Output(success=False, error=f"Invalid directory. Use: {list(dir_map.keys())}")
        d = root / dir_map[directory]
        if not d.exists():
            return Output(success=True, files=[], message="Directory empty or missing")
        files = [
            {"name": f.name, "path": str(f.relative_to(root)), "size": f.stat().st_size}
            for f in sorted(d.iterdir())
            if f.is_file()
        ]
        return Output(success=True, files=files)

    def _delete(self, file_path: str) -> Output:
        ok, result = _resolve(file_path)
        if not ok:
            return Output(success=False, error=result)
        p = Path(result)
        if not p.exists():
            return Output(success=False, error=f"File not found: {p}")
        p.unlink()
        return Output(success=True, path=str(p), message=f"Deleted: {file_path}")


if __name__ == "__main__":
    WmFileOperationsTool.run()
