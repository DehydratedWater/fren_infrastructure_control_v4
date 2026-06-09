"""Low-level async HTTP client for ComfyUI API."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx


async def queue_prompt(base_url: str, workflow_json: dict) -> str:
    """Queue a workflow on ComfyUI. Returns prompt_id."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{base_url}/prompt", json={"prompt": workflow_json})
        resp.raise_for_status()
        result = resp.json()
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"No prompt_id returned: {result}")
        return prompt_id


async def poll_history(
    base_url: str,
    prompt_id: str,
    *,
    timeout: int = 2400,
    interval: int = 5,
) -> list[dict]:
    """Poll ComfyUI history until render completes. Returns list of output file dicts."""
    start = time.time()

    while time.time() - start < timeout:
        await asyncio.sleep(interval)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{base_url}/history/{prompt_id}")
                resp.raise_for_status()
                history = resp.json()
        except Exception:
            continue

        if prompt_id not in history:
            continue

        entry = history[prompt_id]
        status = entry.get("status", {})

        if status.get("status_str") == "error":
            msgs = status.get("messages", [])
            raise RuntimeError(f"Render failed: {msgs}")

        if status.get("completed", False) or status.get("status_str") == "success":
            output_files: list[dict] = []
            outputs = entry.get("outputs", {})
            for _node_id, node_out in outputs.items():
                for key in ("videos", "images", "gifs"):
                    for item in node_out.get(key, []):
                        output_files.append(
                            {
                                "filename": item.get("filename", ""),
                                "subfolder": item.get("subfolder", ""),
                                "type": item.get("type", "output"),
                            }
                        )
            return output_files

    raise TimeoutError(f"Render timed out after {timeout}s (prompt_id={prompt_id})")


async def upload_image(base_url: str, local_path: str) -> str | None:
    """Upload a local image to ComfyUI. Returns server-side filename."""
    p = Path(local_path)
    if not p.is_file():
        return None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(local_path, "rb") as f:
                resp = await client.post(
                    f"{base_url}/upload/image",
                    files={"image": (p.name, f, "image/png")},
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("name")
    except Exception:
        return None


def _download_dir() -> Path:
    """Persistent dir for ComfyUI downloads (DATA_DIR-overridable for dev).

    Renders pulled from the remote ComfyUI host land on the /data volume
    (fren_v4_data:/data) so they survive a container recreate, not /tmp.
    """
    import os

    d = Path(os.environ.get("DATA_DIR", "/data")) / "comfyui_downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_output(base_url: str, output_file: dict) -> str | None:
    """Download an output file from ComfyUI to the persistent /data volume.

    Returns local path.
    """
    import urllib.request

    fn = output_file.get("filename", "")
    if not fn:
        return None
    params = urlencode(
        {
            "filename": fn,
            "subfolder": output_file.get("subfolder", ""),
            "type": output_file.get("type", "output"),
        }
    )
    url = f"{base_url}/view?{params}"
    local_path = str(_download_dir() / f"comfyui_dl_{fn}")
    try:
        urllib.request.urlretrieve(url, local_path)
        if Path(local_path).exists() and Path(local_path).stat().st_size > 0:
            return local_path
    except Exception:
        pass
    # Handle trailing underscore issue
    if fn.endswith("_"):
        base_name = fn[:-1]
        params = urlencode(
            {
                "filename": base_name,
                "subfolder": output_file.get("subfolder", ""),
                "type": output_file.get("type", "output"),
            }
        )
        url2 = f"{base_url}/view?{params}"
        try:
            urllib.request.urlretrieve(url2, local_path)
            if Path(local_path).exists() and Path(local_path).stat().st_size > 0:
                return local_path
        except Exception:
            pass
    return None


async def find_idle_instance(hosts: list[tuple[str, int]]) -> int | None:
    """Find a ComfyUI instance with an empty queue. Returns index or None."""
    for idx, (host, port) in enumerate(hosts):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"http://{host}:{port}/queue")
                resp.raise_for_status()
                data = resp.json()
                running = len(data.get("queue_running", []))
                pending = len(data.get("queue_pending", []))
                if running == 0 and pending == 0:
                    return idx
        except Exception:
            continue
    return None
