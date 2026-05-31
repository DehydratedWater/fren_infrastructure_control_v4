#!/usr/bin/env python3
"""Render a T2I image via ComfyUI, save to DB, print JSON result.

Unlike render_and_send.py (which posts to Telegram as a side-effect), this
script RETURNS the rendered image path via stdout JSON so agents can chain
it into vision analysis, prompt refinement, iteration comparison, etc.

Usage:
    uv run scripts/render_image.py --positive_prompt "..." [options]

Options:
    --positive_prompt TEXT       (required) subject + style tags
    --negative_prompt TEXT       default empty
    --aspect {portrait|square|cinematic}   default portrait
    --seed INT                   default random
    --source_agent TEXT          e.g. "workflows/twily_ralf_execution"
    --source_ralf_id TEXT        ralf_id if invoked from a Ralf run
    --source_stage_number INT
    --source_attempt_number INT
    --reference_media_id TEXT    media_id of prior iteration (for lineage)
    --notes TEXT

Output (stdout JSON):
    {"success": true, "media_id": "img_...", "path": "data/rendered/...png",
     "seed": 12345, "width": 768, "height": 1152, "elapsed_seconds": 42.3}

Timeout: render_scene's default is 2400s (40 min); this script inherits it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]

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


async def _render(args: argparse.Namespace) -> dict:
    from app.comfyui.render import render_scene

    workflow_id = ASPECT_TO_WORKFLOW.get(args.aspect, "ponyxl_t2i_portrait")
    filename_prefix = args.filename_prefix or "ralf_render"

    result = await render_scene(
        workflow_id=workflow_id,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt or "",
        filename_prefix=filename_prefix,
        seed=args.seed,
    )
    if not result.get("success"):
        return {"success": False, "error": result.get("error", "render failed")}

    out_path = _extract_output_path(result)
    if not out_path or not Path(out_path).exists():
        return {"success": False, "error": "no output file produced"}

    # Ensure file is inside project (copy if ComfyUI wrote to /tmp)
    final = _ensure_under_project(out_path, filename_prefix)
    rel_path = str(final.relative_to(PROJECT_ROOT))

    data = result.get("data", {}) or {}
    seed_used = data.get("seed") or args.seed
    width = data.get("width")
    height = data.get("height")
    elapsed = data.get("elapsed_seconds")

    # Store in DB
    from app.db.repos.rendered_media import RenderedMediaRepo
    from app.db.session import set_null_pool

    set_null_pool(True)
    repo = RenderedMediaRepo()
    row = await repo.create(
        media_type="image",
        file_path=rel_path,
        workflow_id=workflow_id,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt or "",
        seed=seed_used,
        width=width,
        height=height,
        elapsed_seconds=elapsed,
        source_agent=args.source_agent,
        source_ralf_id=args.source_ralf_id,
        source_stage_number=args.source_stage_number or None,
        source_attempt_number=args.source_attempt_number or None,
        reference_media_id=args.reference_media_id,
        notes=args.notes,
    )

    return {
        "success": True,
        "media_id": row.get("media_id"),
        "path": rel_path,
        "workflow_id": workflow_id,
        "seed": seed_used,
        "width": width,
        "height": height,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Render T2I image via ComfyUI, save to DB, print JSON")
    p.add_argument("--positive_prompt", required=True)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--aspect", choices=["portrait", "square", "cinematic"], default="portrait")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--filename_prefix", default="")
    p.add_argument("--source_agent", default="")
    p.add_argument("--source_ralf_id", default="")
    p.add_argument("--source_stage_number", type=int, default=0)
    p.add_argument("--source_attempt_number", type=int, default=0)
    p.add_argument("--reference_media_id", default="")
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
