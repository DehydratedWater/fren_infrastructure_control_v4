#!/usr/bin/env python3
"""Render a T2I -> LTX2 I2V video via ComfyUI, save to DB, print JSON result.

Ported from v3 scripts/render_video.py (fren.* -> app.*). Like render_image.py
but produces video. Expensive (several minutes). Returns the output video path
via stdout JSON for agent consumption.

Two modes:
  - Full: --positive_prompt X -> renders T2I reference frame, scales it, then I2V
  - I2V-only: --reference_path data/rendered/XXX.png --positive_prompt X

Output (stdout JSON):
    {"success": true, "media_id": "vid_...", "path": "data/rendered/...mp4",
     "reference_media_id": "img_...", "seed": 12345, "elapsed_seconds": 312.5}

The LTX2 I2V stage alone can take 2-5+ minutes. Plan for timeouts >= 900s.
The ComfyUI host/port is read from settings.get_comfyui_hosts() inside
render_scene() — not hardcoded here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_RESOLUTION = 768
LTX2_I2V_WORKFLOW_ID = os.environ.get("LTX_I2V_WORKFLOW_ID", "ltx23_i2v_gguf")

ASPECT_TO_WORKFLOW = {
    "portrait": "ponyxl_t2i_portrait",
    "square": "ponyxl_t2i_square",
    "cinematic": "ponyxl_t2i",
}


def _extract_output_path(result: dict) -> str | None:
    data = result.get("data", {}) or {}
    files = data.get("output_files") or []
    if not files:
        return None
    first = files[0]
    if isinstance(first, dict):
        return first.get("local_path") or first.get("path")
    return str(first)


def _ensure_under_project(path_str: str, filename_prefix: str) -> Path:
    src = Path(path_str)
    try:
        src.resolve().relative_to(PROJECT_ROOT.resolve())
        return src
    except ValueError:
        dest_dir = PROJECT_ROOT / "data" / "rendered"
        dest_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        dest = dest_dir / f"{filename_prefix}_{stamp}{src.suffix}"
        shutil.copy(src, dest)
        return dest


def _scale_reference_image(ref_path: str) -> tuple[str, int, int]:
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

    return scaled_path, new_w, new_h


async def _render(args: argparse.Namespace) -> dict:
    from app.comfyui.render import render_scene
    from app.db.repos.rendered_media import RenderedMediaRepo
    from app.db.session import set_null_pool

    set_null_pool(True)
    repo = RenderedMediaRepo()
    filename_prefix = args.filename_prefix or "ralf_video"
    reference_media_id = ""

    # Step 1: T2I (unless reference provided).
    if args.reference_path:
        ref_path = args.reference_path
        if not Path(ref_path).exists():
            return {"success": False, "error": f"reference_path not found: {ref_path}"}
        t2i_positive = args.positive_prompt  # still needed for I2V drift anchor
    else:
        t2i_workflow = ASPECT_TO_WORKFLOW.get(args.aspect, "ponyxl_t2i_portrait")
        t2i_result = await render_scene(
            workflow_id=t2i_workflow,
            positive_prompt=args.positive_prompt,
            negative_prompt=args.negative_prompt or "",
            filename_prefix=f"{filename_prefix}_ref",
            seed=args.seed,
        )
        if not t2i_result.get("success"):
            return {"success": False, "error": f"T2I failed: {t2i_result.get('error')}"}
        ref_out = _extract_output_path(t2i_result)
        if not ref_out or not Path(ref_out).exists():
            return {"success": False, "error": "T2I produced no output file"}
        ref_final = _ensure_under_project(ref_out, f"{filename_prefix}_ref")
        ref_path = str(ref_final)
        t2i_positive = args.positive_prompt

        # Record T2I reference in DB.
        ref_data = t2i_result.get("data", {}) or {}
        ref_row = await repo.create(
            media_type="image",
            file_path=str(ref_final.relative_to(PROJECT_ROOT)),
            workflow_id=t2i_workflow,
            positive_prompt=args.positive_prompt,
            negative_prompt=args.negative_prompt or "",
            seed=ref_data.get("seed") or args.seed,
            width=ref_data.get("width"),
            height=ref_data.get("height"),
            elapsed_seconds=ref_data.get("elapsed_seconds"),
            source_agent=args.source_agent,
            source_ralf_id=args.source_ralf_id,
            source_stage_number=args.source_stage_number or None,
            source_attempt_number=args.source_attempt_number or None,
            notes=f"Reference frame for video {filename_prefix}",
        )
        reference_media_id = ref_row.get("media_id", "")

    # Step 2: Scale reference.
    scaled_path, scaled_w, scaled_h = _scale_reference_image(ref_path)

    # Step 3: LTX2 I2V.
    dialog = args.dialog or ""
    i2v_neg_extra = "blurry, low quality, still frame, frames, watermark, overlay, multiple scenes, cuts"
    i2v_positive = f"{t2i_positive}, {dialog}" if dialog else f"{t2i_positive}, gentle movement, subtle animation"
    i2v_negative = f"{args.negative_prompt}, {i2v_neg_extra}" if args.negative_prompt else i2v_neg_extra

    i2v_overrides = {"longer_edge": max(scaled_w, scaled_h)}
    if scaled_w != 720:
        i2v_overrides["width"] = scaled_w
    if scaled_h != 1280:
        i2v_overrides["height"] = scaled_h

    i2v_result = await render_scene(
        workflow_id=LTX2_I2V_WORKFLOW_ID,
        positive_prompt=i2v_positive,
        negative_prompt=i2v_negative,
        filename_prefix=f"{filename_prefix}_vid",
        input_image=scaled_path,
        extra_overrides=i2v_overrides,
    )
    if not i2v_result.get("success"):
        return {
            "success": False,
            "error": f"I2V failed: {i2v_result.get('error')}",
            "reference_media_id": reference_media_id,
        }

    vid_out = _extract_output_path(i2v_result)
    if not vid_out or not Path(vid_out).exists():
        return {
            "success": False,
            "error": "I2V produced no output file",
            "reference_media_id": reference_media_id,
        }

    vid_final = _ensure_under_project(vid_out, f"{filename_prefix}_vid")
    rel_path = str(vid_final.relative_to(PROJECT_ROOT))
    i2v_data = i2v_result.get("data", {}) or {}

    # Record video in DB.
    row = await repo.create(
        media_type="video",
        file_path=rel_path,
        workflow_id=LTX2_I2V_WORKFLOW_ID,
        positive_prompt=i2v_positive,
        negative_prompt=i2v_negative,
        seed=i2v_data.get("seed") or args.seed,
        width=scaled_w,
        height=scaled_h,
        elapsed_seconds=i2v_data.get("elapsed_seconds"),
        source_agent=args.source_agent,
        source_ralf_id=args.source_ralf_id,
        source_stage_number=args.source_stage_number or None,
        source_attempt_number=args.source_attempt_number or None,
        reference_media_id=reference_media_id,
        notes=args.notes or f"dialog={bool(dialog)}",
    )

    return {
        "success": True,
        "media_id": row.get("media_id"),
        "path": rel_path,
        "reference_media_id": reference_media_id,
        "workflow_id": LTX2_I2V_WORKFLOW_ID,
        "seed": i2v_data.get("seed") or args.seed,
        "width": scaled_w,
        "height": scaled_h,
        "elapsed_seconds": i2v_data.get("elapsed_seconds"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Render T2I->I2V video via ComfyUI, save to DB, print JSON")
    p.add_argument("--positive_prompt", required=True)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--aspect", choices=["portrait", "square", "cinematic"], default="portrait")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dialog", default="")
    p.add_argument("--reference_path", default="", help="Skip T2I and use this image as reference frame")
    p.add_argument("--filename_prefix", default="")
    p.add_argument("--source_agent", default="")
    p.add_argument("--source_ralf_id", default="")
    p.add_argument("--source_stage_number", type=int, default=0)
    p.add_argument("--source_attempt_number", type=int, default=0)
    p.add_argument("--notes", default="")
    args = p.parse_args()

    try:
        result = asyncio.run(_render(args))
    except Exception as e:
        result = {"success": False, "error": f"{type(e).__name__}: {e}"}

    print(json.dumps(result))
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
