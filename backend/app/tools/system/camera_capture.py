"""Camera capture tool — grab a frame from webcam or desk camera."""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field

# Device candidates per logical camera, preferred first. Multi-node UVC cameras
# expose several /dev/video* nodes where only one is the live capture node (others
# are metadata-only or busy), so we fall back across them and use the first that
# yields a frame. Overridable via FREN_WEBCAM_DEVICE / FREN_DESK_DEVICE.
_WEBCAM_DEVICES = [os.getenv("FREN_WEBCAM_DEVICE", "/dev/video0"), "/dev/video2", "/dev/video1", "/dev/video3"]
_DESK_DEVICES = [os.getenv("FREN_DESK_DEVICE", "/dev/video2"), "/dev/video0", "/dev/video3", "/dev/video1"]


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


# Captures land on the persistent /data volume (fren_v4_data:/data) so they
# survive a container recreate. The literal default below matches
# settings.data_dir's default; dev can repoint the whole tree via DATA_DIR.
CAPTURES_DIR = Path("/data/captures")


def _captures_dir() -> Path:
    """Resolve the captures dir from settings (DATA_DIR-overridable for dev)."""
    from app.settings import get_settings

    return Path(get_settings().data_dir) / "captures"


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


def _capture_first(devices: list[str], out_path: Path) -> bool:
    """Try each candidate device in order; succeed on the first that yields a frame.

    Tolerates a busy/metadata-only preferred node (e.g. /dev/video0 'Device or
    resource busy') by falling back to the next working camera node.
    """
    seen: set[str] = set()
    for device in devices:
        if device in seen:
            continue
        seen.add(device)
        if _capture(device, out_path):
            return True
    return False


class CameraCaptureTool(ScriptTool[Input, Output]):
    name = "camera-capture"
    description = "Capture a photo from webcam or desk camera"
    output_note = "NEXT STEPS: 1) Read the returned path(s) to view the image. 2) Send your description to the user via send_message.py."

    def execute(self, inp: Input) -> Output:
        # HARD KILL-SWITCH: ffmpeg grabbing /dev/video* (uvcvideo) is implicated
        # in the 2026-06-10 host hard-locks alongside the X grab. When set,
        # refuse WITHOUT opening any video device.
        import os
        if os.getenv("FREN_DISABLE_CAPTURE"):
            return Output(
                success=False,
                error="camera capture disabled (FREN_DISABLE_CAPTURE) — proceed without it",
            )

        captures_dir = _captures_dir()
        captures_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        webcam_path = ""
        desk_path = ""
        errors = []

        if inp.command in ("webcam", "both"):
            path = captures_dir / f"cam_{ts}.jpg"
            if _capture_first(_WEBCAM_DEVICES, path):
                webcam_path = str(path)
            else:
                errors.append("webcam capture failed (no working /dev/video* node)")

        if inp.command in ("desk", "both"):
            path = captures_dir / f"cam_desk_{ts}.jpg"
            if _capture_first(_DESK_DEVICES, path):
                desk_path = str(path)
            else:
                errors.append("desk camera capture failed (no working /dev/video* node)")

        if inp.command not in ("webcam", "desk", "both"):
            return Output(success=False, error=f"Unknown command: {inp.command}. Use: webcam, desk, both")

        if not webcam_path and not desk_path:
            return Output(success=False, error="; ".join(errors) or "no camera available")

        return Output(success=True, webcam_path=webcam_path, desk_path=desk_path)


if __name__ == "__main__":
    CameraCaptureTool.run()
