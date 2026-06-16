"""twily_haven world package — integrity (offline, no DB/LLM).

These pin the *authored* world: it must load, validate, be fully navigable, and
carry the affordances the engine relies on (a computer to research at). World
prose quality is a human concern; this is the structural contract.
"""

from __future__ import annotations

import pytest

from app.world.loader import DEFAULT_PACKAGE, WorldPackageError, list_packages, load_package, validate_refs


def test_default_package_is_listed():
    assert DEFAULT_PACKAGE in list_packages()


def test_package_loads_and_validates():
    pkg = load_package(DEFAULT_PACKAGE)
    assert pkg.id == "twily_haven"
    assert pkg.protagonist.name  # Twily
    assert pkg.visitor.id == "vis"
    assert validate_refs(pkg) == []


def test_world_has_substance():
    pkg = load_package(DEFAULT_PACKAGE)
    assert len(pkg.locations) >= 10
    assert len(pkg.npcs) >= 6
    assert len(pkg.lorebook) >= 6


def test_start_location_resolves():
    pkg = load_package(DEFAULT_PACKAGE)
    start = pkg.location(pkg.scenario.starting_location_id)
    assert start is not None


def test_there_is_a_computer_to_research_at():
    # the research mechanic routes through an activity tagged "computer"
    pkg = load_package(DEFAULT_PACKAGE)
    tags = {a.tag for loc in pkg.locations for a in loc.activities}
    assert "computer" in tags


def test_navigation_graph_is_connected_from_start():
    pkg = load_package(DEFAULT_PACKAGE)
    # validate_refs already asserts reachability; assert neighbors exist for start
    start = pkg.scenario.starting_location_id
    assert pkg.neighbors(start), "start location has no exits"


def test_every_npc_home_and_default_npc_resolves():
    pkg = load_package(DEFAULT_PACKAGE)
    loc_ids = {loc.id for loc in pkg.locations}
    npc_ids = {n.id for n in pkg.npcs}
    for n in pkg.npcs:
        assert (n.home_location_id is None) or (n.home_location_id in loc_ids)
    for loc in pkg.locations:
        for nid in loc.default_npcs:
            assert nid in npc_ids


def test_missing_package_raises():
    with pytest.raises(WorldPackageError):
        load_package("does_not_exist_xyz")
