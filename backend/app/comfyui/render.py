"""High-level render function — load workflow, parameterize, queue, poll, return result."""

from __future__ import annotations

import random
import time

from app.comfyui import client, workflows


async def render_scene(
    *,
    workflow_id: str,
    positive_prompt: str,
    negative_prompt: str,
    filename_prefix: str = "render",
    seed: int | None = None,
    instance_id: int | None = None,
    input_image: str | None = None,
    extra_overrides: dict | None = None,
    timeout: int = 2400,
    poll_interval: int = 5,
) -> dict:
    """Render a scene on ComfyUI. Blocking async call.

    Returns dict with {success, data: {output_files, base_url, elapsed_seconds, ...}, error}.
    """
    from app.settings import get_settings

    settings = get_settings()
    hosts = settings.get_comfyui_hosts()

    if not hosts:
        return {"success": False, "error": "No ComfyUI instances configured"}

    # 1. Load and parameterize workflow
    try:
        workflow_json, parameter_schema = workflows.load_workflow(workflow_id)
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}

    overrides: dict = {
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "seed": seed if seed is not None else random.randint(0, 2**32 - 1),
    }
    if extra_overrides:
        overrides.update(extra_overrides)

    workflow_json = workflows.apply_overrides(workflow_json, parameter_schema, overrides)
    workflows.set_filename_prefix(workflow_json, filename_prefix)

    # 2. Select instance
    selected = instance_id
    if selected is None:
        selected = await client.find_idle_instance(hosts)
        if selected is None:
            return {"success": False, "error": "No idle ComfyUI instances available"}

    if selected < 0 or selected >= len(hosts):
        return {"success": False, "error": f"Invalid instance_id {selected}. Range: 0-{len(hosts) - 1}"}

    host, port = hosts[selected]
    base_url = f"http://{host}:{port}"

    # 3. Upload input image if provided
    if input_image and "input_image" in parameter_schema:
        uploaded_name = await client.upload_image(base_url, input_image)
        if uploaded_name:
            entry = parameter_schema["input_image"]
            node_id = entry["node_id"]
            input_key = entry["input_key"]
            if node_id in workflow_json and "inputs" in workflow_json[node_id]:
                workflow_json[node_id]["inputs"][input_key] = uploaded_name

    # 4. Queue and poll
    start_time = time.time()
    try:
        prompt_id = await client.queue_prompt(base_url, workflow_json)
        output_files = await client.poll_history(base_url, prompt_id, timeout=timeout, interval=poll_interval)
    except Exception as e:
        return {"success": False, "error": str(e)}

    elapsed = round(time.time() - start_time, 1)

    return {
        "success": True,
        "data": {
            "prompt_id": prompt_id,
            "instance_id": selected,
            "base_url": base_url,
            "elapsed_seconds": elapsed,
            "output_files": output_files,
            "filename_prefix": filename_prefix,
        },
    }
