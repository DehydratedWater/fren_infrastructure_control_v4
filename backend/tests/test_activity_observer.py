"""Activity observer port — fully mocked (no camera / no vision / no DB).

Asserts the capture→describe→store pipeline writes an activity_blocks row with
the observation, carries NO health snapshot (camera never fabricates health),
and skips cleanly (no row) when capture or vision fails.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from app.tools.system import activity_observer as ao


async def _async(value):
    return value


def test_strip_thinking():
    assert ao._strip_thinking("<think>x</think>\nA person sits at the desk.") == "A person sits at the desk."


async def test_store_block_writes_observation_no_health(monkeypatch):
    captured: dict = {}

    class _Repo:
        async def insert_blocks(self, block_date, blocks):
            captured["block_date"] = block_date
            captured["blocks"] = blocks
            return len(blocks)

    import app.db.repos.activity_blocks as ab

    monkeypatch.setattr(ab, "ActivityBlocksRepo", _Repo)
    # device not synced / no token -> no health attached
    monkeypatch.setattr(ao, "_fetch_health_snapshot", lambda: _async({}))
    ok = await ao._store_block("A person is at the desk, screen on.", "data/captures/cam_x.jpg")
    assert ok is True
    block = captured["blocks"][0]
    assert block["activity_type"] == "observation"
    assert "person is at the desk" in block["description"]
    assert "camera" in block["tags"] and "room_state" in block["tags"]
    # camera carries NO health — must be empty so it can never be fabricated
    assert block["health_snapshot"] == {}
    assert block["environment"]["image_path"].endswith("cam_x.jpg")


async def test_store_block_attaches_garmin_when_present(monkeypatch):
    captured: dict = {}

    class _Repo:
        async def insert_blocks(self, block_date, blocks):
            captured["blocks"] = blocks
            return len(blocks)

    import app.db.repos.activity_blocks as ab

    monkeypatch.setattr(ab, "ActivityBlocksRepo", _Repo)
    # device synced -> live snapshot attaches to the block
    monkeypatch.setattr(ao, "_fetch_health_snapshot", lambda: _async({"body_battery": 14, "stress": 60}))
    await ao._store_block("Desk occupied.", "data/captures/y.jpg")
    assert captured["blocks"][0]["health_snapshot"] == {"body_battery": 14, "stress": 60}


async def test_fetch_health_snapshot_empty_without_token(monkeypatch):
    monkeypatch.delenv("GRAPHANA_GARMIN_DATA_KEY", raising=False)
    assert await ao._fetch_health_snapshot() == {}


async def test_run_skips_when_capture_fails(monkeypatch):
    monkeypatch.setattr(ao, "_capture_frame", lambda command="webcam": "")
    stored = {"called": False}

    async def _store(*a, **k):
        stored["called"] = True
        return True

    monkeypatch.setattr(ao, "_store_block", _store)
    assert await ao.run() is None
    assert stored["called"] is False


async def test_run_skips_when_no_description(monkeypatch):
    monkeypatch.setattr(ao, "_capture_frame", lambda command="webcam": "data/captures/x.jpg")
    monkeypatch.setattr(ao, "_describe", lambda path: _async(""))
    stored = {"called": False}

    async def _store(*a, **k):
        stored["called"] = True
        return True

    monkeypatch.setattr(ao, "_store_block", _store)
    assert await ao.run() is None
    assert stored["called"] is False


async def test_run_full_pipeline(monkeypatch):
    monkeypatch.setattr(ao, "_capture_frame", lambda command="webcam": "data/captures/x.jpg")
    monkeypatch.setattr(ao, "_describe", lambda path: _async("Desk is empty, lights off."))
    seen: dict = {}

    async def _store(desc, path):
        seen["desc"] = desc
        seen["path"] = path
        return True

    monkeypatch.setattr(ao, "_store_block", _store)
    out = await ao.run(command="desk")
    assert out == "Desk is empty, lights off."
    assert seen["desc"] == "Desk is empty, lights off."
    assert seen["path"] == "data/captures/x.jpg"
