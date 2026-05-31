"""Render PonyXL — blocking render + non-blocking dispatch to background worker."""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Default ComfyUI instance for Twily renders (192.168.0.95:8899 = index 0 in COMFYUI_INSTANCES)
TWILY_INSTANCE_ID = 0


class Input(BaseModel):
    command: str = Field(description="render|dispatch_image|dispatch_video")
    workflow_id: str = Field(default="", description="ComfyUI workflow ID (resolved from aspect if empty)")
    positive_prompt: str = Field(default="", description="Positive prompt from composer")
    negative_prompt: str = Field(default="", description="Negative prompt from composer")
    filename_prefix: str = Field(default="twily_selfie", description="Output filename prefix")
    seed: int | None = Field(default=None, description="Random seed (None = random)")
    instance_id: int | None = Field(default=None, description="Force GPU instance (None = auto)")
    num: int = Field(
        default=1,
        description="Number of images to generate (each with random seed). Only use when user explicitly asks for multiple.",
    )
    caption: str = Field(default="", description="Telegram caption for dispatched image/video")
    dialog: str = Field(default="", description="Narrative/dialog prompt for LTX2 I2V (dispatch_video only)")
    aspect: str = Field(
        default="portrait",
        description="Aspect ratio hint: cinematic|portrait|square (resolves workflow_id if empty)",
    )


class Output(BaseModel):
    success: bool = True
    data: dict = Field(default_factory=dict)
    error: str = ""


class RenderPonyXLTool(ScriptTool[Input, Output]):
    name = "render_ponyxl"
    description = "Render PonyXL image (blocking) or dispatch image/video render to background worker (non-blocking)"

    def execute(self, inp: Input) -> Output:
        if inp.command == "render":
            return self._render_blocking(inp)
        elif inp.command == "dispatch_image":
            return self._dispatch(inp, mode="image")
        elif inp.command == "dispatch_video":
            return self._dispatch(inp, mode="video")
        else:
            return Output(
                success=False, error=f"Unknown command: {inp.command}. Use render|dispatch_image|dispatch_video"
            )

    def _resolve_workflow_id(self, inp: Input) -> str:
        if inp.workflow_id:
            return inp.workflow_id
        workflow_map = {
            "cinematic": "ponyxl_t2i",
            "portrait": "ponyxl_t2i_portrait",
            "square": "ponyxl_t2i_square",
        }
        return workflow_map.get(inp.aspect, "ponyxl_t2i_portrait")

    @staticmethod
    def _normalize_seed(seed: int | None) -> int | None:
        """Treat -1 or None as random (let render_scene pick)."""
        return None if seed is None or seed < 0 else seed

    def _render_blocking(self, inp: Input) -> Output:
        """Blocking render — calls fren.comfyui.render directly."""
        # TODO(v4-port): app.comfyui not yet ported
        from app.comfyui.client import download_output
        from app.comfyui.render import render_scene

        workflow_id = self._resolve_workflow_id(inp)
        seed = self._normalize_seed(inp.seed)
        instance_id = inp.instance_id if inp.instance_id is not None else TWILY_INSTANCE_ID

        try:
            result = asyncio.run(
                render_scene(
                    workflow_id=workflow_id,
                    positive_prompt=inp.positive_prompt,
                    negative_prompt=inp.negative_prompt,
                    filename_prefix=inp.filename_prefix,
                    seed=seed,
                    instance_id=instance_id,
                    timeout=2400,
                )
            )
        except Exception as e:
            return Output(success=False, error=f"Render failed: {e}")

        if not result.get("success"):
            return Output(success=False, error=result.get("error", "Unknown render error"))

        # Resolve output file path (download from remote ComfyUI)
        output_files = result.get("data", {}).get("output_files", [])
        base_url = result.get("data", {}).get("base_url", "")
        output_path = ""
        if output_files and base_url:
            output_path = download_output(base_url, output_files[0]) or ""

        # Cache the rendered image artifact
        if output_path:
            try:
                from app.db.repos.context_cache import add_to_cache

                prompt_preview = (inp.positive_prompt or "")[:150].replace("\n", " ")
                asyncio.run(
                    add_to_cache(
                        "selfie_image",
                        f"Selfie: {prompt_preview}" if prompt_preview else f"Generated selfie ({workflow_id})",
                        file_path=output_path,
                        tags=["selfie", "generated", "ponyxl"],
                        content_class="nsfw",
                        source_agent="render_ponyxl",
                    )
                )
            except Exception:
                pass

        return Output(
            success=True,
            data={
                "output_path": output_path,
                "workflow_id": workflow_id,
                "elapsed_seconds": result.get("data", {}).get("elapsed_seconds", 0),
                "output_files": output_files,
            },
        )

    def _dispatch(self, inp: Input, mode: str) -> Output:
        """Non-blocking dispatch — write job JSON, spawn background worker, return immediately."""
        import random

        workflow_id = self._resolve_workflow_id(inp)
        num = max(1, min(inp.num, 20))  # clamp to 1-20

        seed = self._normalize_seed(inp.seed)
        worker_script = PROJECT_ROOT / "scripts" / "render_and_send.py"
        job_files: list[str] = []

        for i in range(num):
            # For batch: each job gets a unique random seed (unless user specified one)
            job_seed = seed if (seed is not None and num == 1) else random.randint(0, 2**32 - 1)

            job = {
                "mode": mode,
                "workflow_id": workflow_id,
                "positive_prompt": inp.positive_prompt,
                "negative_prompt": inp.negative_prompt,
                "filename_prefix": inp.filename_prefix,
                "caption": inp.caption,
                "seed": job_seed,
                "instance_id": inp.instance_id if inp.instance_id is not None else TWILY_INSTANCE_ID,
            }
            if mode == "video" and inp.dialog:
                job["dialog"] = inp.dialog

            # Write job to temp file
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    prefix=f"render_job_{i}_",
                    dir="/tmp",
                    delete=False,
                ) as f:
                    json.dump(job, f)
                    job_path = f.name
            except Exception as e:
                return Output(success=False, error=f"Failed to write job file {i}: {e}")

            # Spawn background worker (detached, logs to /tmp for debugging)
            log_path = job_path.replace(".json", ".log")
            try:
                with open(log_path, "w") as log_file:
                    subprocess.Popen(
                        ["uv", "run", str(worker_script), job_path],
                        cwd=str(PROJECT_ROOT),
                        start_new_session=True,
                        stdout=log_file,
                        stderr=log_file,
                    )
            except Exception as e:
                return Output(success=False, error=f"Failed to spawn background worker {i}: {e}")

            job_files.append(job_path)

        return Output(
            success=True,
            data={
                "dispatched": True,
                "mode": mode,
                "num": num,
                "job_files": job_files,
                "workflow_id": workflow_id,
            },
        )


if __name__ == "__main__":
    RenderPonyXLTool.run()
