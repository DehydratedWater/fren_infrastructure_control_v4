"""Load a modifiable world package from YAML and validate it.

A package is a directory under `packages/<id>/` split into a few hand-editable
files that merge into one `WorldPackage`:

    package.yaml    top-level meta + protagonist + visitor + scenario + factions
    locations.yaml  { locations: [...], connections: [...] }
    npcs.yaml       { npcs: [...] }
    lorebook.yaml   { lorebook: [...] }

Splitting keeps the world easy to edit by hand (the user asked for "modifiable
packages"). `load_package` merges + Pydantic-validates, then `validate_refs`
checks referential integrity (every referenced id resolves, the nav graph is
connected) so a typo fails loudly at load instead of mid-turn.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

from app.world.models import WorldPackage

_PACKAGES_DIR = Path(__file__).parent / "packages"
DEFAULT_PACKAGE = "twily_haven"


class WorldPackageError(RuntimeError):
    """Raised when a package is missing, malformed, or referentially broken."""


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise WorldPackageError(f"{path.name}: expected a YAML mapping, got {type(data).__name__}")
    return data


def package_dir(package_id: str = DEFAULT_PACKAGE) -> Path:
    return _PACKAGES_DIR / package_id


def list_packages() -> list[str]:
    if not _PACKAGES_DIR.exists():
        return []
    return sorted(p.name for p in _PACKAGES_DIR.iterdir() if (p / "package.yaml").exists())


def load_package(package_id: str = DEFAULT_PACKAGE) -> WorldPackage:
    """Merge the package's YAML files into a validated `WorldPackage`."""
    root = package_dir(package_id)
    if not (root / "package.yaml").exists():
        raise WorldPackageError(
            f"world package {package_id!r} not found at {root} (no package.yaml)"
        )

    merged = _read_yaml(root / "package.yaml")
    locations = _read_yaml(root / "locations.yaml")
    merged["locations"] = locations.get("locations", merged.get("locations", []))
    merged["connections"] = locations.get("connections", merged.get("connections", []))
    merged["npcs"] = _read_yaml(root / "npcs.yaml").get("npcs", merged.get("npcs", []))
    merged["lorebook"] = _read_yaml(root / "lorebook.yaml").get("lorebook", merged.get("lorebook", []))

    try:
        pkg = WorldPackage.model_validate(merged)
    except Exception as exc:  # pydantic ValidationError → friendly
        raise WorldPackageError(f"world package {package_id!r} failed validation: {exc}") from exc

    problems = validate_refs(pkg)
    if problems:
        raise WorldPackageError(
            f"world package {package_id!r} has broken references:\n  - "
            + "\n  - ".join(problems)
        )
    return pkg


def validate_refs(pkg: WorldPackage) -> list[str]:
    """Return a list of referential-integrity problems (empty == clean)."""
    problems: list[str] = []
    loc_ids = {loc.id for loc in pkg.locations}
    npc_ids = {n.id for n in pkg.npcs}

    if len(loc_ids) != len(pkg.locations):
        problems.append("duplicate location ids")
    if len(npc_ids) != len(pkg.npcs):
        problems.append("duplicate npc ids")

    for loc in pkg.locations:
        if loc.parent_id and loc.parent_id not in loc_ids:
            problems.append(f"location {loc.id}: parent_id {loc.parent_id!r} not found")
        for nid in loc.default_npcs:
            if nid not in npc_ids:
                problems.append(f"location {loc.id}: default_npc {nid!r} not found")

    for c in pkg.connections:
        if c.from_id not in loc_ids:
            problems.append(f"connection from_id {c.from_id!r} not found")
        if c.to_id not in loc_ids:
            problems.append(f"connection to_id {c.to_id!r} not found")

    for n in pkg.npcs:
        if n.home_location_id and n.home_location_id not in loc_ids:
            problems.append(f"npc {n.id}: home_location_id {n.home_location_id!r} not found")

    if pkg.scenario.starting_location_id not in loc_ids:
        problems.append(
            f"scenario.starting_location_id {pkg.scenario.starting_location_id!r} not found"
        )

    # navigation graph connectivity (treat bidirectional edges both ways; rooms
    # nest under parents via parent_id which also implies passage)
    if loc_ids:
        adj: dict[str, set[str]] = {lid: set() for lid in loc_ids}
        for c in pkg.connections:
            if c.from_id in adj and c.to_id in adj:
                adj[c.from_id].add(c.to_id)
                if c.bidirectional:
                    adj[c.to_id].add(c.from_id)
        for loc in pkg.locations:
            if loc.parent_id and loc.parent_id in adj:
                adj[loc.id].add(loc.parent_id)
                adj[loc.parent_id].add(loc.id)
        start = pkg.scenario.starting_location_id
        if start in adj:
            seen = {start}
            stack = [start]
            while stack:
                cur = stack.pop()
                for nxt in adj[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            unreachable = loc_ids - seen
            if unreachable:
                problems.append(
                    "locations unreachable from start: " + ", ".join(sorted(unreachable))
                )
    return problems


@functools.lru_cache(maxsize=4)
def get_package(package_id: str = DEFAULT_PACKAGE) -> WorldPackage:
    """Cached package load (packages are static at runtime)."""
    return load_package(package_id)
