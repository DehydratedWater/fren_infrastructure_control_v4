"""Compile the fleet to opencode worker artifacts.

`compile_fleet()` runs one `CompileScript` over all 7 worker variants (the GLM
cloud passes + local + the split profile), so every agent is emitted as
`.opencode/agents/<path><postfix>.md`. This is the v4 replacement for v3's
`scripts/compile_agents.sh` + `src/fren/compile.py` split dispatcher — the
split routing now lives in the framework (SplitProfile), not hand-rolled here.
"""

from __future__ import annotations

from pathlib import Path

from app.agents.config import WORKER_VARIANTS
from app.agents.registry import build_registry
from src import CompileScript


def compile_fleet(
    *,
    target: Path,
    project_root: Path | None = None,
    variants=None,
    clean: bool = True,
    verbose: bool = False,
) -> list[Path]:
    """Compile every agent × every worker variant into `target`.

    Returns the written file paths. Pass a subset of `variants` (e.g. just the
    default) for a fast dev/test compile.
    """
    def factory():
        return build_registry(project_root=project_root)

    script = CompileScript(
        target=target,
        config="prod",
        factory=factory,
        variants=list(variants if variants is not None else WORKER_VARIANTS),
        clean=clean,
        verbose=verbose,
    )
    result = script.run()
    return list(getattr(result, "written_files", []) or [])
