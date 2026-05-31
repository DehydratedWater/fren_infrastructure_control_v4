"""Report Writer — write, list, read, and resolve bug/feature reports."""

from __future__ import annotations

import contextlib
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from src import ScriptTool
from pydantic import BaseModel, Field

REPORTS_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def _sanitize_slug(raw: str) -> str:
    """Lowercase, hyphens only, strip leading/trailing hyphens."""
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower().strip())
    return slug.strip("-")[:80]


class Input(BaseModel):
    command: str = Field(description="write|list|get|resolve")
    report_type: str = Field(default="bug", description="bug|feature")
    slug: str = Field(default="", description="Report slug (for write)")
    content: str = Field(default="", description="Markdown report content (for write)")
    filename: str = Field(default="", description="Report filename (for get|resolve)")
    status: str = Field(default="pending", description="pending|fixed|all (for list)")


class Output(BaseModel):
    success: bool = True
    path: str = ""
    items: list[dict] = Field(default_factory=list)
    content: str = ""
    error: str = ""


class ReportWriterTool(ScriptTool[Input, Output]):
    name: ClassVar[str] = "report_writer"
    description: ClassVar[str] = "Write, list, read, and resolve bug/feature reports"
    stream_field: ClassVar[str] = "content"

    def execute(self, inp: Input) -> Output:
        cmd = inp.command

        if cmd == "write":
            return self._write(inp.report_type, inp.slug, inp.content)
        if cmd == "list":
            return self._list(inp.report_type, inp.status)
        if cmd == "get":
            return self._get(inp.report_type, inp.filename)
        if cmd == "resolve":
            return self._resolve(inp.report_type, inp.filename)

        return Output(success=False, error=f"Unknown command: {cmd}")

    def _report_dir(self, report_type: str, status: str) -> Path:
        plural = f"{report_type}s"
        return REPORTS_ROOT / plural / status

    def _write(self, report_type: str, slug: str, content: str) -> Output:
        if report_type not in ("bug", "feature"):
            return Output(success=False, error=f"Invalid report_type: {report_type}")
        if not slug:
            return Output(success=False, error="slug is required")
        if not content:
            return Output(success=False, error="content is required")

        safe_slug = _sanitize_slug(slug)
        if not safe_slug:
            return Output(success=False, error=f"Invalid slug after sanitization: {slug}")

        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{date_str}_{safe_slug}.md"
        target_dir = self._report_dir(report_type, "pending")
        target_dir.mkdir(parents=True, exist_ok=True)

        path = target_dir / filename
        path.write_text(content)

        return Output(path=str(path.relative_to(REPORTS_ROOT)))

    def _list(self, report_type: str, status: str) -> Output:
        if report_type not in ("bug", "feature"):
            return Output(success=False, error=f"Invalid report_type: {report_type}")

        statuses = ["pending", "fixed"] if status == "all" else [status]
        items: list[dict] = []

        for s in statuses:
            d = self._report_dir(report_type, s)
            if not d.exists():
                continue
            for f in sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
                first_line = ""
                with contextlib.suppress(OSError):
                    first_line = f.read_text().split("\n", 1)[0].strip()
                items.append(
                    {
                        "filename": f.name,
                        "status": s,
                        "title": first_line,
                        "mtime": f.stat().st_mtime,
                    }
                )

        return Output(items=items)

    def _get(self, report_type: str, filename: str) -> Output:
        if not filename:
            return Output(success=False, error="filename is required")

        for s in ("pending", "fixed"):
            path = self._report_dir(report_type, s) / filename
            if path.exists():
                return Output(content=path.read_text(), path=str(path.relative_to(REPORTS_ROOT)))

        return Output(success=False, error=f"Report not found: {filename}")

    def _resolve(self, report_type: str, filename: str) -> Output:
        if not filename:
            return Output(success=False, error="filename is required")

        src = self._report_dir(report_type, "pending") / filename
        if not src.exists():
            return Output(success=False, error=f"Pending report not found: {filename}")

        dest_dir = self._report_dir(report_type, "fixed")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        shutil.move(str(src), str(dest))

        return Output(path=str(dest.relative_to(REPORTS_ROOT)))


if __name__ == "__main__":
    ReportWriterTool.run()
