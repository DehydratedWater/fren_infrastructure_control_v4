"""Screenshot tool — capture desktop for situational awareness."""

from __future__ import annotations

import asyncio
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(default="capture", description="capture")


class Output(BaseModel):
    success: bool = True
    path: str = ""
    error: str = ""


CAPTURES_DIR = Path("data/captures")


class ScreenshotTool(ScriptTool[Input, Output]):
    name = "screenshot"
    description = "Capture desktop screenshot for situational awareness"
    output_note = "NEXT STEPS: 1) Read the returned path to view the screenshot. 2) Send your description to the user via send_message.py."

    def execute(self, inp: Input) -> Output:
        if inp.command != "capture":
            return Output(success=False, error=f"Unknown command: {inp.command}")

        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{ts}.jpg"
        filepath = CAPTURES_DIR / filename

        # Try scrot (X11), then grim (Wayland)
        for cmd in [
            ["scrot", "-z", "-q", "80", str(filepath)],
            ["grim", str(filepath)],
        ]:
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=10)
                # Cache the screenshot artifact
                try:
                    from app.db.repos.context_cache import add_to_cache

                    asyncio.run(
                        add_to_cache(
                            "screenshot",
                            f"Desktop screenshot captured at {ts}",
                            file_path=str(filepath),
                            tags=["screenshot", "desktop"],
                            source_agent="screenshot",
                        )
                    )
                except Exception:
                    pass
                return Output(success=True, path=str(filepath))
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue

        return Output(success=False, error="No screenshot tool available (tried scrot, grim)")


if __name__ == "__main__":
    ScreenshotTool.run()
