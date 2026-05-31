"""Load and parameterize ComfyUI workflows from local JSON files."""

from __future__ import annotations

import copy
import json
from pathlib import Path

_WORKFLOWS_DIR = Path(__file__).resolve().parents[3] / "data" / "comfyui_workflows"


def load_workflow(workflow_id: str) -> tuple[dict, dict]:
    """Load a workflow JSON and its parameter schema from disk.

    Returns (workflow_json, parameter_schema).
    Raises FileNotFoundError if workflow doesn't exist.
    """
    path = _WORKFLOWS_DIR / f"{workflow_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Workflow '{workflow_id}' not found at {path}")

    with open(path) as f:
        data = json.load(f)

    return data["workflow_json"], data.get("parameter_schema", {})


def apply_overrides(
    workflow_json: dict,
    parameter_schema: dict,
    overrides: dict,
) -> dict:
    """Apply parameter overrides to a workflow JSON using the parameter schema.

    Returns a deep copy with overrides applied.
    """
    wf = copy.deepcopy(workflow_json)

    for param_name, param_value in overrides.items():
        if param_name in parameter_schema:
            entry = parameter_schema[param_name]
            node_id = entry["node_id"]
            input_key = entry["input_key"]
            if node_id in wf and "inputs" in wf[node_id]:
                wf[node_id]["inputs"][input_key] = param_value

    return wf


def set_filename_prefix(workflow_json: dict, prefix: str) -> None:
    """Set filename_prefix on all SaveVideo/SaveImage/VHS_VideoCombine nodes (in-place)."""
    for node_data in workflow_json.values():
        if isinstance(node_data, dict):
            ct = node_data.get("class_type", "")
            if (
                ct in ("SaveVideo", "SaveImage", "VHS_VideoCombine")
                and "inputs" in node_data
                and "filename_prefix" in node_data["inputs"]
            ):
                node_data["inputs"]["filename_prefix"] = prefix
