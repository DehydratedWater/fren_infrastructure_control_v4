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


# Captures land on the persistent /data volume (fren_v4_data:/data) so they
# survive a container recreate. The literal default matches settings.data_dir's
# default; dev can repoint the whole tree via DATA_DIR.
CAPTURES_DIR = Path("/data/captures")


def _captures_dir() -> Path:
    """Resolve the captures dir from settings (DATA_DIR-overridable for dev)."""
    from app.settings import get_settings

    return Path(get_settings().data_dir) / "captures"


class ScreenshotTool(ScriptTool[Input, Output]):
    name = "screenshot"
    description = "Capture desktop screenshot for situational awareness"
    output_note = "NEXT STEPS: 1) Read the returned path to view the screenshot. 2) Send your description to the user via send_message.py."

    def execute(self, inp: Input) -> Output:
        if inp.command != "capture":
            return Output(success=False, error=f"Unknown command: {inp.command}")

        # HARD KILL-SWITCH: attaching to the display (scrot/grim) hard-locked
        # the host twice on 2026-06-10 (the v3 xorg-attach crash mode). When
        # set, refuse WITHOUT touching X — agents proceed without the capture
        # (the grounded-absence probes train exactly that behavior).
        import os
        if os.getenv("FREN_DISABLE_CAPTURE"):
            return Output(
                success=False,
                error="screen capture disabled (FREN_DISABLE_CAPTURE) — proceed without it",
            )

        captures_dir = _captures_dir()
        captures_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{ts}.jpg"
        filepath = captures_dir / filename

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
