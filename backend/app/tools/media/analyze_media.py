"""Analyze image or video using a vision-capable vLLM model.

For videos: extracts audio via ffmpeg, transcribes with Whisper STT,
sends video to vision model, and combines visual + audio descriptions.
Long videos are split into 30s chunks with progress messages to Telegram.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field

from app.vllm_resolve import get_llm_endpoint

# Defaults — resolved dynamically from vLLM state
DEFAULT_BASE_URL, DEFAULT_MODEL = get_llm_endpoint()
DEFAULT_MAX_TOKENS = 16384
MAX_CHUNK_SECONDS = 30

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Env vars to capture for background dispatch (needed by follow-up orchestrator session)
_ENV_KEYS = (
    "XDG_DATA_HOME",
    "FREN_MSG_HEADER",
    "FREN_CONTENT_CLASS",
    "FREN_CLEARANCE",
    "FREN_MODEL_POSTFIX",
    "FREN_TTS_POSTFIX",
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".m4v": "video/mp4",
}


class Input(BaseModel):
    file_path: str = Field(description="Path to image or video file (relative or absolute)")
    prompt: str = Field(
        default="Describe what you see in detail.",
        description="Analysis prompt / question about the media",
    )
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, description="Max response tokens")


class Output(BaseModel):
    success: bool = True
    description: str = ""
    audio_transcript: str = ""
    media_type: str = ""
    chunks_processed: int = 0
    duration_seconds: float = 0
    dispatched: bool = False
    error: str = ""


# ── Helpers ──


def _resolve_path(file_path: str) -> Path:
    """Resolve relative path from project root."""
    p = Path(file_path)
    if p.is_absolute():
        return p
    from app.settings import get_settings

    return Path(get_settings().project_root) / p


def _send_progress(msg: str) -> None:
    """Send a progress message to Telegram."""
    with contextlib.suppress(Exception):
        from app.tools.telegram.send_message import Input as MsgInput
        from app.tools.telegram.send_message import SendMessageTool

        SendMessageTool().execute(MsgInput(message=msg))


def _encode_image(path: Path, max_longer_edge: int = 1024) -> str:
    """Read and optionally downscale an image, return base64 JPEG."""
    from PIL import Image

    with Image.open(path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        longer = max(img.size)
        if longer > max_longer_edge:
            scale = max_longer_edge / longer
            new_size = (round(img.width * scale), round(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()


def _encode_video(path: Path) -> tuple[str, str]:
    """Read video file and return (base64_data, mime_type)."""
    mime = VIDEO_MIME.get(path.suffix.lower(), "video/mp4")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return b64, mime


def _get_video_duration(path: Path) -> float:
    """Get video duration in seconds using cv2."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return total / fps if fps > 0 else 0.0


def _extract_audio(video_path: Path) -> Path | None:
    """Extract audio from video to temp OGG file using ffmpeg."""
    audio_path = video_path.parent / f"{video_path.stem}.audio.ogg"
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-c:a",
                "libopus",
                "-b:a",
                "64k",
                str(audio_path),
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0 or not audio_path.exists() or audio_path.stat().st_size == 0:
            return None
        return audio_path
    except (subprocess.TimeoutExpired, OSError):
        return None


def _transcribe_audio(audio_path: Path) -> str:
    """Send audio to fren-stt Whisper service, return transcript text."""
    import httpx

    from app.settings import get_settings

    settings = get_settings()
    url = f"http://{settings.stt_host}/v1/audio/transcriptions"

    try:
        with httpx.Client(timeout=120.0) as client, open(audio_path, "rb") as f:
            resp = client.post(url, files={"file": ("audio.ogg", f, "audio/ogg")})
        return resp.json().get("text", "")
    except Exception as e:
        print(f"[analyze_media] STT transcription failed: {e}")
        return ""


def _split_video(path: Path, chunk_seconds: int = MAX_CHUNK_SECONDS) -> list[tuple[Path, float, float]]:
    """Split video into chunks using ffmpeg. Returns [(chunk_path, start_sec, end_sec), ...]."""
    duration = _get_video_duration(path)
    if duration <= 0:
        return []

    chunks: list[tuple[Path, float, float]] = []
    start = 0.0
    idx = 0

    while start < duration:
        end = min(start + chunk_seconds, duration)
        chunk_path = path.parent / f"{path.stem}_chunk{idx}{path.suffix}"
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start),
                    "-t",
                    str(chunk_seconds),
                    "-i",
                    str(path),
                    "-c",
                    "copy",
                    str(chunk_path),
                ],
                capture_output=True,
                timeout=60,
            )
            if result.returncode == 0 and chunk_path.exists() and chunk_path.stat().st_size > 0:
                chunks.append((chunk_path, start, end))
        except (subprocess.TimeoutExpired, OSError):
            pass
        start = end
        idx += 1

    return chunks


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _cleanup_files(*paths: Path | None) -> None:
    """Delete temp files, ignoring errors."""
    for p in paths:
        if p is not None:
            with contextlib.suppress(Exception):
                p.unlink(missing_ok=True)


# ── Tool ──


class AnalyzeMediaTool(ScriptTool[Input, Output]):
    name = "analyze_media"
    description = (
        "Analyze an image or video file using a vision model. Returns a text description and audio transcript."
    )
    stream_field = "description"

    def execute(self, inp: Input) -> Output:
        path = _resolve_path(inp.file_path)
        if not path.exists():
            return Output(success=False, error=f"File not found: {path}")

        ext = path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return self._analyze_image(path, inp)
        elif ext in VIDEO_EXTENSIONS:
            # Check if video is long enough to dispatch to background
            duration = _get_video_duration(path)
            if duration > MAX_CHUNK_SECONDS:
                num_chunks = -(-int(duration) // MAX_CHUNK_SECONDS)
                if num_chunks > 2:
                    return self._dispatch_background(path, inp, duration, num_chunks)
            return self._analyze_video(path, inp)
        else:
            return Output(success=False, error=f"Unsupported file type: {ext}")

    def _analyze_image(self, path: Path, inp: Input) -> Output:
        b64 = _encode_image(path)
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": inp.prompt},
        ]
        result = self._call_api(content, inp.max_tokens)
        if result.startswith("ERROR:"):
            return Output(success=False, media_type="image", error=result)
        return Output(success=True, description=result, media_type="image")

    def _dispatch_background(self, path: Path, inp: Input, duration: float, num_chunks: int) -> Output:
        """Dispatch long video analysis to background worker, return immediately."""
        job = {
            "file_path": str(path),
            "prompt": inp.prompt,
            "max_tokens": inp.max_tokens,
            "duration": duration,
            "num_chunks": num_chunks,
            "env": {k: os.environ.get(k, "") for k in _ENV_KEYS},
        }

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="video_analysis_job_", dir="/tmp", delete=False
            ) as f:
                json.dump(job, f)
                job_path = f.name
        except Exception as e:
            return Output(success=False, error=f"Failed to write job file: {e}")

        worker_script = PROJECT_ROOT / "scripts" / "analyze_and_notify.py"
        log_path = job_path.replace(".json", ".log")
        try:
            with open(log_path, "w") as log_file:
                subprocess.Popen(
                    ["python", str(worker_script), job_path],
                    cwd=str(PROJECT_ROOT),
                    start_new_session=True,
                    stdout=log_file,
                    stderr=log_file,
                )
        except Exception as e:
            return Output(success=False, error=f"Failed to spawn background worker: {e}")

        _send_progress(f"Processing long video ({duration:.0f}s, {num_chunks} chunks) in background...")

        return Output(
            success=True,
            dispatched=True,
            media_type="video",
            duration_seconds=round(duration, 1),
            description=f"Dispatched {num_chunks}-chunk video analysis to background. Results will arrive as a follow-up message.",
        )

    def _analyze_video(self, path: Path, inp: Input) -> Output:
        duration = _get_video_duration(path)
        audio_path: Path | None = None
        chunk_files: list[Path] = []

        try:
            # Step 1: Extract and transcribe audio (from full video, once)
            audio_path = _extract_audio(path)
            transcript = ""
            if audio_path:
                transcript = _transcribe_audio(audio_path)
                print(f"[analyze_media] Audio transcript ({len(transcript)} chars): {transcript[:100]}...")

            # Step 2: Visual analysis — single or chunked
            if duration <= MAX_CHUNK_SECONDS:
                # Short video — process whole
                _send_progress(f"Analyzing video ({duration:.0f}s)...")
                visual = self._vision_for_video(path, inp.prompt, inp.max_tokens)
                chunks_processed = 1
            else:
                # Long video — split into chunks
                chunks = _split_video(path)
                chunk_files = [c[0] for c in chunks]
                n = len(chunks)
                _send_progress(f"Analyzing video ({duration:.0f}s, {n} chunks)...")

                visual_parts: list[str] = []
                for i, (chunk_path, start, end) in enumerate(chunks, 1):
                    ts_start = _format_time(start)
                    ts_end = _format_time(end)
                    _send_progress(f"Chunk {i}/{n} ({ts_start}-{ts_end})...")

                    chunk_visual = self._vision_for_video(
                        chunk_path,
                        f"This is segment {ts_start}-{ts_end} of a longer video. {inp.prompt}",
                        inp.max_tokens,
                    )
                    visual_parts.append(f"[{ts_start}-{ts_end}] {chunk_visual}")
                    running = "\n\n".join(visual_parts)
                    print(f"[analyze_media] After chunk {i}/{n}: {len(running)} chars accumulated")

                visual = "\n\n".join(visual_parts)
                chunks_processed = n

            # Step 3: Combine visual + audio
            description = visual
            if transcript:
                description += f"\n\n## Audio Transcript\n{transcript}"

            return Output(
                success=True,
                description=description,
                audio_transcript=transcript,
                media_type="video",
                chunks_processed=chunks_processed,
                duration_seconds=round(duration, 1),
            )

        except Exception as e:
            return Output(success=False, media_type="video", error=str(e))

        finally:
            _cleanup_files(audio_path, *chunk_files)

    def _vision_for_video(self, path: Path, prompt: str, max_tokens: int) -> str:
        """Send a single video file to the vision API."""
        b64, mime = _encode_video(path)
        content = [
            {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]
        result = self._call_api(content, max_tokens)
        if result.startswith("ERROR:"):
            return f"(Vision error: {result})"
        return result

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Strip thinking/reasoning blocks from thinking-model output."""
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
        return text.strip()

    def _call_api(self, content: list[dict], max_tokens: int) -> str:
        from openai import OpenAI

        from app.settings import get_settings

        settings = get_settings()
        client = OpenAI(
            api_key=settings.vllm_api_key,
            base_url=DEFAULT_BASE_URL,
            timeout=180,
        )

        try:
            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content or ""
            return self._strip_thinking(raw)
        except Exception as e:
            return f"ERROR: {e}"


if __name__ == "__main__":
    AnalyzeMediaTool.run()
