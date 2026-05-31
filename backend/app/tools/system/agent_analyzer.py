"""Agent analyzer tool — scan agent definitions and build dependency graph."""

from __future__ import annotations

import re
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field

from app.settings import get_settings


class Input(BaseModel):
    command: str = Field(description="list-agents|get-agent|get-graph|get-stats")
    agent_name: str = Field(default="", description="Agent name for get-agent")


class Output(BaseModel):
    success: bool = True
    agents: list[dict] = Field(default_factory=list)
    agent: dict | None = None
    graph: dict | None = None
    stats: dict | None = None
    count: int = 0
    error: str = ""


def _parse_agent_md(path: Path) -> dict:
    """Parse an agent markdown file for metadata."""
    content = path.read_text(encoding="utf-8")
    info: dict = {"name": path.stem, "path": str(path), "file": path.name}

    # Parse YAML frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end > 0:
            frontmatter = content[3:end]
            for line in frontmatter.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    info[key.strip()] = val.strip()

    # Extract description from first heading or description field
    desc_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
    if desc_match:
        info.setdefault("title", desc_match.group(1))

    # Detect bash patterns
    bash_refs = re.findall(r"uv run scripts/(\S+\.py)", content)
    info["script_refs"] = list(set(bash_refs))

    # Detect subagent references
    task_refs = re.findall(r"(?:--agent|Task tool.*?)\s+(\S+/\S+)", content)
    info["task_refs"] = list(set(task_refs))

    return info


class AgentAnalyzerTool(ScriptTool[Input, Output]):
    name = "agent_analyzer"
    description = "Scan agent definitions and build dependency graphs"

    def execute(self, inp: Input) -> Output:
        root = Path(get_settings().project_root)
        agents_dir = root / ".opencode" / "agents"

        if not agents_dir.exists():
            # Try build directory
            agents_dir = root / "build" / ".opencode" / "agents"

        if not agents_dir.exists():
            return Output(success=False, error=f"Agents dir not found: {agents_dir}")

        all_agents = [_parse_agent_md(f) for f in sorted(agents_dir.rglob("*.md"))]

        if inp.command == "list-agents":
            return Output(success=True, agents=all_agents, count=len(all_agents))

        if inp.command == "get-agent":
            for a in all_agents:
                if a["name"] == inp.agent_name or inp.agent_name in a.get("path", ""):
                    return Output(success=True, agent=a)
            return Output(success=False, error=f"Agent not found: {inp.agent_name}")

        if inp.command == "get-graph":
            nodes = [{"id": a["name"], "dir": str(Path(a["path"]).parent.name)} for a in all_agents]
            edges = []
            for a in all_agents:
                for ref in a.get("task_refs", []):
                    edges.append({"from": a["name"], "to": ref.split("/")[-1], "type": "task"})
            return Output(success=True, graph={"nodes": nodes, "edges": edges}, count=len(nodes))

        if inp.command == "get-stats":
            by_dir: dict[str, int] = {}
            for a in all_agents:
                d = str(Path(a["path"]).parent.name)
                by_dir[d] = by_dir.get(d, 0) + 1
            return Output(
                success=True,
                stats={"total": len(all_agents), "by_directory": by_dir},
                count=len(all_agents),
            )

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    AgentAnalyzerTool.run()
