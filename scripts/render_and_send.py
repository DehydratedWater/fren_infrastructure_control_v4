#!/usr/bin/env python3
"""Background worker: render PonyXL image/video and send to Telegram.

Ported from v3 scripts/render_and_send.py. Adapted to v4 conventions:
`fren.*` -> `app.*`, `uv run` -> `python`, and the self-review session uses v4's
opencode_manager CLI (--command run --agent ... --model_postfix ...).

Takes a job JSON file path as argument. Spawned (detached) by
app.tools.vis_simulation.render_ponyxl.RenderPonyXLTool._dispatch via
`python scripts/render_and_send.py <job.json>`.

Modes:
  image: render_scene(T2I) -> send_photo(output, caption)
  video: render_scene(T2I) -> scale to 768px -> render_scene(LTX2 I2V) -> send_video

The ComfyUI host/port is NOT hardcoded here — render_scene() reads it from
settings.get_comfyui_hosts() (COMFYUI_INSTANCES env / default 192.168.0.95:8899).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# I2V workflow — matches a filename in data/comfyui_workflows/.
# ltx23_i2v_gguf = LTX-2.3 (22B GGUF), ltx2_i2v = LTX-2 (19B, legacy).
LTX2_I2V_WORKFLOW_ID = os.environ.get("LTX_I2V_WORKFLOW_ID", "ltx23_i2v_gguf")

# Fixed video resolution — 768px longer edge produces 704x1280 portrait after 8px alignment.
VIDEO_RESOLUTION = 768


def _render_scene_sync(**kwargs: object) -> dict:
    """Call app.comfyui.render.render_scene synchronously."""
    from app.comfyui.render import render_scene

    return asyncio.run(render_scene(**kwargs))


def _extract_output_path(render_result: dict) -> str | None:
    """Extract file path from render result. Downloads from remote ComfyUI if needed."""
    from app.comfyui.client import download_output

    output_files = render_result.get("data", {}).get("output_files", [])
    base_url = render_result.get("data", {}).get("base_url", "")
    if not output_files:
        return None
    of = output_files[0]
    filename = of.get("filename", "")
    if not filename:
        return None
    # Always download from remote (ComfyUI runs on a different machine).
    if base_url:
        local = download_output(base_url, of)
        if local:
            print(f"[download] {filename} -> {local}")
            return local
    return None


def _send_photo(file_path: str, caption: str) -> dict:
    """Send photo via v4's SendImageTool."""
    from app.tools.telegram.send_image import Input, SendImageTool

    result = SendImageTool().execute(Input(image_path=file_path, caption=caption))
    return {"success": result.success, "error": result.error}


def _send_video(file_path: str, caption: str) -> dict:
    """Send video via v4's SendVideoTool."""
    from app.tools.telegram.send_video import Input, SendVideoTool

    result = SendVideoTool().execute(Input(file_path=file_path, caption=caption))
    return {"success": result.success, "error": result.error}


def _scale_reference_image(ref_path: str) -> tuple[str, int, int]:
    """Scale reference image so its longer edge matches VIDEO_RESOLUTION (768px).

    Aligns both dimensions to 8px for VAE compatibility.
    Returns (scaled_path, new_width, new_height).
    """
    from PIL import Image

    with Image.open(ref_path) as img:
        orig_w, orig_h = img.size

    longer = max(orig_w, orig_h)
    if longer <= VIDEO_RESOLUTION:
        new_w = max(64, round(orig_w / 8) * 8)
        new_h = max(64, round(orig_h / 8) * 8)
        if new_w == orig_w and new_h == orig_h:
            return ref_path, orig_w, orig_h
    else:
        scale = VIDEO_RESOLUTION / longer
        new_w = max(64, round(orig_w * scale / 8) * 8)
        new_h = max(64, round(orig_h * scale / 8) * 8)

    with Image.open(ref_path) as img:
        scaled = img.resize((new_w, new_h), Image.LANCZOS)
        scaled_path = ref_path.replace(".", "_scaled.", 1)
        scaled.save(scaled_path, quality=95)

    print(f"[render_and_send] Scaled {orig_w}x{orig_h} -> {new_w}x{new_h}")
    return scaled_path, new_w, new_h


def _send_error_message(error: str) -> None:
    """Send error notification to Telegram."""
    from app.tools.telegram.send_message import Input, SendMessageTool

    with contextlib.suppress(Exception):
        SendMessageTool().execute(Input(message=f"[Render Error] {error[:500]}"))


def _trigger_self_review(file_path: str, media_type: str, caption: str) -> None:
    """Trigger a chat session so Twily can see what she just sent."""
    import shutil

    # Build relative @path for the vision model — file must be inside project root.
    p = Path(file_path)
    try:
        rel = str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        # File is outside project root (e.g. /tmp/) — copy into the persistent
        # /data volume (DATA_DIR-overridable for dev) so the render survives a
        # container recreate. opencode resolves an absolute @path fine, so we
        # hand it the absolute dest rather than one relative to PROJECT_ROOT.
        dest_dir = Path(os.environ.get("DATA_DIR", "/data")) / "rendered"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / p.name
        shutil.copy2(file_path, dest)
        rel = str(dest)
        print(f"[render_and_send] Copied {file_path} -> {rel} for self-review")

    postfix = os.environ.get("FREN_MODEL_POSTFIX", "")

    prompt = f"Twily glances at the {media_type} she just sent to the user: @{rel}"
    if caption:
        prompt += f"\nCaption was: {caption}"
    prompt += "\nReact naturally — comment, tease, or admire the result. Keep it brief and organic."

    # v4 opencode_manager CLI: --command run --agent <name> --prompt <p> [--model_postfix <pf>].
    # The tool appends the postfix + "-primary" internally, so pass the bare agent
    # name and the postfix separately (do NOT pre-suffix the agent name).
    cmd = [
        "python",
        "scripts/opencode_manager.py",
        "--command", "run",
        "--agent", "persona/twily_chat",
        "--prompt", prompt,
    ]
    if postfix:
        cmd += ["--model_postfix", postfix]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=os.environ,
            capture_output=True,
            timeout=300,
        )
        if result.returncode == 0:
            print("[render_and_send] Self-review session completed")
        else:
            print(f"[render_and_send] Self-review failed (exit {result.returncode})")
    except Exception as e:
        print(f"[render_and_send] Self-review trigger failed: {e}")


def run_image_mode(job: dict) -> None:
    """Render T2I image and send to Telegram."""
    print(f"[render_and_send] Image mode: {job['filename_prefix']}")

    result = _render_scene_sync(
        workflow_id=job["workflow_id"],
        positive_prompt=job["positive_prompt"],
        negative_prompt=job["negative_prompt"],
        filename_prefix=job["filename_prefix"],
        seed=job.get("seed"),
        instance_id=job.get("instance_id"),
    )

    if not result.get("success"):
        err = result.get("error", "Unknown render error")
        print(f"[render_and_send] T2I render failed: {err}", file=sys.stderr)
        _send_error_message(f"Image render failed: {err}")
        return

    output_path = _extract_output_path(result)
    if not output_path or not Path(output_path).exists():
        print("[render_and_send] No output file found", file=sys.stderr)
        _send_error_message("Image render produced no output file")
        return

    elapsed = result.get("data", {}).get("elapsed_seconds", 0)
    print(f"[render_and_send] T2I done in {elapsed}s -> {output_path}")

    caption = job.get("caption", "")
    send_result = _send_photo(output_path, caption)
    if send_result.get("success"):
        print("[render_and_send] Photo sent to Telegram")
        _trigger_self_review(output_path, "image", caption)
    else:
        print(f"[render_and_send] Telegram send failed: {send_result.get('error')}", file=sys.stderr)


def run_video_mode(job: dict) -> None:
    """Render T2I image, scale to 768px, then LTX2 I2V, and send to Telegram."""
    print(f"[render_and_send] Video mode: {job['filename_prefix']}")

    # Step 1: T2I render (reference frame).
    t2i_result = _render_scene_sync(
        workflow_id=job["workflow_id"],
        positive_prompt=job["positive_prompt"],
        negative_prompt=job["negative_prompt"],
        filename_prefix=f"{job['filename_prefix']}_ref",
        seed=job.get("seed"),
        instance_id=job.get("instance_id"),
    )

    if not t2i_result.get("success"):
        err = t2i_result.get("error", "Unknown T2I error")
        print(f"[render_and_send] T2I render failed: {err}", file=sys.stderr)
        _send_error_message(f"Video T2I render failed: {err}")
        return

    ref_path = _extract_output_path(t2i_result)
    if not ref_path or not Path(ref_path).exists():
        print("[render_and_send] No T2I output file", file=sys.stderr)
        _send_error_message("Video T2I produced no output file")
        return

    t2i_elapsed = t2i_result.get("data", {}).get("elapsed_seconds", 0)
    print(f"[render_and_send] T2I done in {t2i_elapsed}s -> {ref_path}")

    # Step 2: Scale reference image to 768px longer edge (-> 704x1280 portrait).
    scaled_ref_path, scaled_w, scaled_h = _scale_reference_image(ref_path)

    # Step 3: LTX2 I2V render.
    dialog = job.get("dialog", "")
    base_neg = job.get("negative_prompt", "")
    i2v_neg_extra = (
        "blurry, low quality, still frame, frames, watermark, overlay, titles, multiple scenes, cuts, "
        "photorealistic, real life, live action, real person, photograph, uncanny valley, "
        "pixar, disney, realistic skin, human face, human ears, no horn, "
        "western cartoon, chibi, furry realistic, hyper-realistic"
    )
    i2v_style_anchor = (
        "sfm source filmmaker style, sfm blender model style, "
        "3d anime style, 3d anime cel shading, anthro pony character, "
        "lavender purple skin, dark navy blue hair with pink magenta streaks, "
        "large expressive purple eyes, pointed unicorn horn, pony ears, "
        "soft anime lighting, clean linework, consistent character design"
    )
    # Pass the full T2I positive prompt to I2V so it inherits all character tags,
    # quality scores, clothing, expression, etc. — prevents drift from reference image.
    t2i_positive = job.get("positive_prompt", "")
    if dialog:
        i2v_positive = (
            f"{t2i_positive}, {i2v_style_anchor}, {dialog}" if t2i_positive else f"{i2v_style_anchor}, {dialog}"
        )
    else:
        i2v_positive = f"{t2i_positive}, {i2v_style_anchor}, gentle movement, subtle animation"
    i2v_negative = f"{base_neg}, {i2v_neg_extra}" if base_neg else i2v_neg_extra

    # Overrides: set dimensions from scaled image so LTX2 doesn't upscale.
    i2v_overrides: dict = {}
    if scaled_w != 720:
        i2v_overrides["width"] = scaled_w
    if scaled_h != 1280:
        i2v_overrides["height"] = scaled_h
    scaled_longer = max(scaled_w, scaled_h)
    i2v_overrides["longer_edge"] = scaled_longer

    print(f"[render_and_send] LTX2 I2V: {scaled_w}x{scaled_h}, 25fps, dialog={bool(dialog)}")

    i2v_result = _render_scene_sync(
        workflow_id=LTX2_I2V_WORKFLOW_ID,
        positive_prompt=i2v_positive,
        negative_prompt=i2v_negative,
        filename_prefix=f"{job['filename_prefix']}_vid",
        input_image=scaled_ref_path,
        instance_id=job.get("instance_id"),
        extra_overrides=i2v_overrides if i2v_overrides else None,
    )

    if not i2v_result.get("success"):
        err = i2v_result.get("error", "Unknown I2V error")
        print(f"[render_and_send] I2V render failed: {err}", file=sys.stderr)
        # Fallback: send the reference image instead.
        caption = job.get("caption", "")
        caption = (caption + " (video render failed, here's the image)") if caption else "(video render failed, here's the image)"
        _send_photo(ref_path, caption)
        return

    video_path = _extract_output_path(i2v_result)
    if not video_path or not Path(video_path).exists():
        print("[render_and_send] No I2V output file", file=sys.stderr)
        _send_photo(ref_path, job.get("caption", "") + " (no video output, sending image)")
        return

    i2v_elapsed = i2v_result.get("data", {}).get("elapsed_seconds", 0)
    print(f"[render_and_send] I2V done in {i2v_elapsed}s -> {video_path}")

    caption = job.get("caption", "")
    send_result = _send_video(video_path, caption)
    if send_result.get("success"):
        print("[render_and_send] Video sent to Telegram")
        _trigger_self_review(video_path, "video", caption)
    else:
        print(f"[render_and_send] Video send failed: {send_result.get('error')}", file=sys.stderr)
        _send_photo(ref_path, caption)  # fallback to photo


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: render_and_send.py <job_file.json>", file=sys.stderr)
        sys.exit(1)

    job_path = sys.argv[1]
    try:
        with open(job_path) as f:
            job = json.load(f)
    except Exception as e:
        print(f"Failed to read job file {job_path}: {e}", file=sys.stderr)
        sys.exit(1)

    mode = job.get("mode", "image")
    try:
        if mode == "image":
            run_image_mode(job)
        elif mode == "video":
            run_video_mode(job)
        else:
            print(f"Unknown mode: {mode}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"[render_and_send] Unhandled error: {e}", file=sys.stderr)
        _send_error_message(f"Unhandled error in {mode} mode: {e}")
    finally:
        with contextlib.suppress(Exception):
            Path(job_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
