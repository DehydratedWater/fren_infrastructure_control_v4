#!/usr/bin/env python3
"""Render RP scene illustration — dispatches to background worker that sends via RP bot.

Same dispatch mechanism as render_ponyxl.py but spawns render_and_send_rp.py
instead of render_and_send.py, so the final image goes through BOT_RP_TOKEN.
"""

from __future__ import annotations

import json
import random
import subprocess
import tempfile
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Default ComfyUI instance (same as main bot)
TWILY_INSTANCE_ID = 0


class Input(BaseModel):
    command: str = Field(description="dispatch_image (non-blocking)")
    workflow_id: str = Field(default="", description="ComfyUI workflow ID")
    positive_prompt: str = Field(default="", description="Positive prompt from composer")
    negative_prompt: str = Field(default="", description="Negative prompt from composer")
    filename_prefix: str = Field(default="rp_scene", description="Output filename prefix")
    seed: int | None = Field(default=None, description="Random seed (None = random)")
    instance_id: int | None = Field(default=None, description="Force GPU instance (None = auto)")
    caption: str = Field(default="", description="Telegram caption for the illustration")
    aspect: str = Field(
        default="portrait",
        description="Aspect ratio hint: cinematic|portrait|square",
    )


class Output(BaseModel):
    success: bool = True
    data: dict = Field(default_factory=dict)
    error: str = ""


class RenderRPSceneTool(ScriptTool[Input, Output]):
    name = "render_rp_scene"
    description = "Dispatch RP scene illustration to background worker (sends image via RP bot)"

    def _resolve_workflow_id(self, inp: Input) -> str:
        if inp.workflow_id:
            return inp.workflow_id
        workflow_map = {
            "cinematic": "ponyxl_t2i",
            "portrait": "ponyxl_t2i_portrait",
            "square": "ponyxl_t2i_square",
        }
        return workflow_map.get(inp.aspect, "ponyxl_t2i_portrait")

    def execute(self, inp: Input) -> Output:
        if inp.command != "dispatch_image":
            return Output(success=False, error=f"Unknown command: {inp.command}. Use dispatch_image")

        workflow_id = self._resolve_workflow_id(inp)
        worker_script = PROJECT_ROOT / "scripts" / "render_and_send_rp.py"

        job = {
            "mode": "image",
            "workflow_id": workflow_id,
            "positive_prompt": inp.positive_prompt,
            "negative_prompt": inp.negative_prompt,
            "filename_prefix": inp.filename_prefix,
            "caption": inp.caption,
            "seed": inp.seed if inp.seed is not None and inp.seed >= 0 else random.randint(0, 2**32 - 1),
            "instance_id": inp.instance_id if inp.instance_id is not None else TWILY_INSTANCE_ID,
        }

        # Write job to temp file
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="render_rp_job_", dir="/tmp", delete=False
            ) as f:
                json.dump(job, f)
                job_path = f.name
        except Exception as e:
            return Output(success=False, error=f"Failed to write job file: {e}")

        # Spawn background worker (detached)
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

        return Output(
            success=True,
            data={"dispatched": True, "workflow_id": workflow_id, "job_file": job_path},
        )


if __name__ == "__main__":
    RenderRPSceneTool.run()
