"""Route finder tool — BFS graph traversal over compiled agent .md files."""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field

from app.settings import get_settings

KNOWN_POSTFIXES = ("-local", "-gptoss", "-glm47", "-glm5")
_MAX_DEPTH = 6


class Input(BaseModel):
    command: str = Field(description="list-capabilities|find-route")
    start_agent: str = Field(
        description="Agent path, e.g. persona/fren_orchestrator or persona/fren_orchestrator-glm47"
    )
    use_case: str = Field(default="", description="Capability description to search for (find-route only)")


class RouteStep(BaseModel):
    agent: str
    description: str
    invocation: str  # task | bash | start


class Route(BaseModel):
    path: list[RouteStep]
    depth: int


class Capability(BaseModel):
    agent: str
    description: str
    mode: str
    depth: int
    scripts: list[str]
    trigger: str


class Output(BaseModel):
    success: bool = True
    capabilities: list[Capability] = Field(default_factory=list)
    routes: list[Route] = Field(default_factory=list)
    count: int = 0
    error: str = ""


def _extract_postfix(stem: str) -> str:
    """Return the postfix (e.g. '-glm47') or '' if none."""
    for pf in KNOWN_POSTFIXES:
        if stem.endswith(pf):
            return pf
    return ""


def _strip_postfix(stem: str) -> str:
    pf = _extract_postfix(stem)
    return stem[: -len(pf)] if pf else stem


def _parse_agent_md(path: Path) -> dict:
    """Parse a compiled agent .md file for metadata."""
    content = path.read_text(encoding="utf-8")
    info: dict = {
        "stem": path.stem,
        "dir": path.parent.name,
        "path": f"{path.parent.name}/{path.stem}",
        "base_name": _strip_postfix(path.stem),
        "postfix": _extract_postfix(path.stem),
    }

    # Parse YAML frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end > 0:
            frontmatter = content[3:end]
            for line in frontmatter.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    info[key.strip()] = val.strip()

    # First heading as description fallback
    desc_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
    if desc_match:
        info.setdefault("description", desc_match.group(1))
    info.setdefault("description", "")
    info.setdefault("mode", "")
    info.setdefault("trigger", "")

    # Script references
    bash_refs = re.findall(r"uv run scripts/(\S+\.py)", content)
    info["script_refs"] = sorted(set(bash_refs))

    # Task tool edges (subagent references)
    task_refs = re.findall(r'subagent_type:\s*"([^"]+)"', content)
    # Also capture --agent patterns (bash invocation of other agents)
    bash_agent_refs = re.findall(r"--agent\s+(\S+/\S+)", content)
    info["task_edges"] = sorted(set(task_refs))
    info["bash_edges"] = sorted(set(bash_agent_refs))

    return info


def _build_graph(agents_dir: Path, postfix: str) -> dict[str, dict]:
    """Build adjacency list from compiled .md files, filtered by postfix."""
    graph: dict[str, dict] = {}

    for md_file in sorted(agents_dir.rglob("*.md")):
        info = _parse_agent_md(md_file)
        if info["postfix"] != postfix:
            continue
        key = f"{info['dir']}/{info['stem']}"
        graph[key] = info

    return graph


def _resolve_edge(graph: dict[str, dict], ref: str, postfix: str) -> str | None:
    """Resolve an agent reference to a graph key, trying with and without postfix."""
    # Direct match
    if ref in graph:
        return ref

    # Try appending postfix
    if postfix and not ref.endswith(postfix):
        candidate = ref + postfix
        if candidate in graph:
            return candidate

    # Try matching by dir/base_name
    for key, info in graph.items():
        ref_parts = ref.rsplit("/", 1)
        if len(ref_parts) == 2:
            ref_dir, ref_name = ref_parts
            if info["dir"] == ref_dir and info["base_name"] == _strip_postfix(ref_name):
                return key

    return None


def _list_capabilities(graph: dict[str, dict], start_key: str) -> list[Capability]:
    """BFS from start node, collect all reachable agents."""
    visited: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque([(start_key, 0)])
    postfix = graph[start_key]["postfix"]

    while queue:
        current, depth = queue.popleft()
        if current in visited:
            continue
        visited[current] = depth

        info = graph.get(current)
        if not info:
            continue

        # Traverse edges
        for ref in info.get("task_edges", []):
            resolved = _resolve_edge(graph, ref, postfix)
            if resolved and resolved not in visited:
                queue.append((resolved, depth + 1))

        for ref in info.get("bash_edges", []):
            resolved = _resolve_edge(graph, ref, postfix)
            if resolved and resolved not in visited:
                queue.append((resolved, depth + 1))

    capabilities = []
    for key, depth in sorted(visited.items(), key=lambda x: x[1]):
        info = graph[key]
        capabilities.append(
            Capability(
                agent=key,
                description=info.get("description", ""),
                mode=info.get("mode", ""),
                depth=depth,
                scripts=info.get("script_refs", []),
                trigger=info.get("trigger", ""),
            )
        )

    return capabilities


def _find_routes(graph: dict[str, dict], start_key: str, use_case: str) -> list[Route]:
    """Find all paths from start to agents matching use_case keywords."""
    postfix = graph[start_key]["postfix"]
    keywords = [w.lower() for w in use_case.split() if len(w) > 2]

    # Find target agents matching keywords
    targets: set[str] = set()
    for key, info in graph.items():
        if info["postfix"] != postfix:
            continue
        searchable = " ".join(
            [
                info.get("description", ""),
                " ".join(info.get("script_refs", [])),
                info.get("trigger", ""),
                info.get("base_name", ""),
            ]
        ).lower()
        if any(kw in searchable for kw in keywords):
            targets.add(key)

    if not targets:
        return []

    # BFS to find shortest paths to each target
    # Track (current_node, path_so_far)
    queue: deque[tuple[str, list[RouteStep]]] = deque()
    queue.append(
        (
            start_key,
            [RouteStep(agent=start_key, description=graph[start_key].get("description", ""), invocation="start")],
        )
    )
    visited: set[str] = set()
    routes: list[Route] = []

    while queue:
        current, path = queue.popleft()

        if len(path) > _MAX_DEPTH:
            continue

        if current in targets and len(path) > 1:
            routes.append(Route(path=path, depth=len(path) - 1))
            # Don't stop — continue to find longer paths through this node
            # But don't re-explore from targets
            continue

        if current in visited:
            continue
        visited.add(current)

        info = graph.get(current)
        if not info:
            continue

        for ref in info.get("task_edges", []):
            resolved = _resolve_edge(graph, ref, postfix)
            if resolved and resolved not in visited:
                ref_info = graph.get(resolved, {})
                step = RouteStep(
                    agent=resolved,
                    description=ref_info.get("description", ""),
                    invocation="task",
                )
                queue.append((resolved, [*path, step]))

        for ref in info.get("bash_edges", []):
            resolved = _resolve_edge(graph, ref, postfix)
            if resolved and resolved not in visited:
                ref_info = graph.get(resolved, {})
                step = RouteStep(
                    agent=resolved,
                    description=ref_info.get("description", ""),
                    invocation="bash",
                )
                queue.append((resolved, [*path, step]))

    # Sort by path length (shortest first)
    routes.sort(key=lambda r: r.depth)
    return routes


class RouteFinderTool(ScriptTool[Input, Output]):
    name = "route_finder"
    description = "Find agent routes and list reachable capabilities via graph traversal"

    def execute(self, inp: Input) -> Output:
        root = Path(get_settings().project_root)
        agents_dir = root / ".opencode" / "agents"

        if not agents_dir.exists():
            return Output(success=False, error=f"Agents dir not found: {agents_dir}")

        # Determine postfix from start_agent
        start_parts = inp.start_agent.rsplit("/", 1)
        if len(start_parts) == 2:
            postfix = _extract_postfix(start_parts[1])
        else:
            postfix = _extract_postfix(inp.start_agent)

        graph = _build_graph(agents_dir, postfix)

        # Resolve start_agent to a graph key
        start_key = _resolve_edge(graph, inp.start_agent, postfix)
        if not start_key:
            return Output(
                success=False,
                error=f"Start agent not found: {inp.start_agent} (postfix={postfix!r})",
            )

        if inp.command == "list-capabilities":
            caps = _list_capabilities(graph, start_key)
            return Output(success=True, capabilities=caps, count=len(caps))

        if inp.command == "find-route":
            if not inp.use_case:
                return Output(success=False, error="use_case is required for find-route")
            routes = _find_routes(graph, start_key, inp.use_case)
            return Output(success=True, routes=routes, count=len(routes))

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    RouteFinderTool.run()
