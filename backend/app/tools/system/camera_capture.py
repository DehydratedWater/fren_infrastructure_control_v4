"""Camera capture tool — grab a frame from webcam or desk camera."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        default="webcam",
        description="webcam (room camera /dev/video0), desk (desk camera /dev/video2), or both",
    )


class Output(BaseModel):
    success: bool = True
    webcam_path: str = ""
    desk_path: str = ""
    error: str = ""


CAPTURES_DIR = Path("data/captures")


def _capture(device: str, out_path: Path) -> bool:
    """Capture a single frame from a V4L2 camera device."""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "v4l2",
                "-video_size",
                "1280x720",
                "-i",
                device,
                "-frames:v",
                "1",
                str(out_path),
            ],
            timeout=5,
            check=True,
            capture_output=True,
        )
        return out_path.exists() and out_path.stat().st_size > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


class CameraCaptureTool(ScriptTool[Input, Output]):
    name = "camera-capture"
    description = "Capture a photo from webcam or desk camera"
    output_note = "NEXT STEPS: 1) Read the returned path(s) to view the image. 2) Send your description to the user via send_message.py."

    def execute(self, inp: Input) -> Output:
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        webcam_path = ""
        desk_path = ""
        errors = []

        if inp.command in ("webcam", "both"):
            path = CAPTURES_DIR / f"cam_{ts}.jpg"
            if _capture("/dev/video0", path):
                webcam_path = str(path)
            else:
                errors.append("webcam capture failed")

        if inp.command in ("desk", "both"):
            path = CAPTURES_DIR / f"cam_desk_{ts}.jpg"
            if _capture("/dev/video2", path):
                desk_path = str(path)
            else:
                errors.append("desk camera capture failed")

        if inp.command not in ("webcam", "desk", "both"):
            return Output(success=False, error=f"Unknown command: {inp.command}. Use: webcam, desk, both")

        if not webcam_path and not desk_path:
            return Output(success=False, error="; ".join(errors) or "no camera available")

        return Output(success=True, webcam_path=webcam_path, desk_path=desk_path)


if __name__ == "__main__":
    CameraCaptureTool.run()
