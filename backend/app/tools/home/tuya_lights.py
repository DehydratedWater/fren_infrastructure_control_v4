"""Tuya smart device control — lights, plugs, switches via LAN."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src import ScriptTool
from pydantic import BaseModel, Field

# NOTE(v4-port): `tinytuya` is a runtime dependency imported lazily inside the
# connection helpers so the tool module can be imported without it installed.

_REGISTRY_PATH = Path(__file__).resolve().parents[4] / "tuya_devices.json"

# DPS mappings for v3.3 RGB bulbs (DPS 20-25)
_BULB_DPS = {
    "switch": "20",
    "mode": "21",
    "brightness": "22",
    "color_temp": "23",
    "color": "24",
    "scene": "25",
}

_PLUG_DPS = {
    "switch": "1",
    "current": "18",
    "power": "19",
    "voltage": "20",
}


def _load_registry(*, agent_only: bool = True) -> list[dict[str, Any]]:
    path = os.environ.get("TUYA_REGISTRY_PATH", str(_REGISTRY_PATH))
    with open(path) as f:
        devices = json.load(f)
    if agent_only:
        devices = [d for d in devices if d.get("agent_accessible", True)]
    return devices


def _normalize(s: str) -> str:
    """Collapse whitespace and lowercase for matching."""
    return " ".join(s.lower().split())


_GROUP_TARGETS = {"all", "all_bulbs", "all_plugs"}


def _is_group_target(query: str) -> bool:
    return _normalize(query) in _GROUP_TARGETS


def _resolve_group(query: str) -> list[dict[str, Any]]:
    """Resolve a group target to a list of controllable LAN devices."""
    devices = _load_registry()
    q = _normalize(query)
    candidates = [d for d in devices if d["online_lan"] and d["ip"] and not d.get("read_only")]
    if q == "all":
        return candidates
    if q == "all_bulbs":
        return [d for d in candidates if d["category"] == "bulb"]
    if q == "all_plugs":
        return [d for d in candidates if d["category"] == "plug"]
    return []


def _find_device(devices: list[dict[str, Any]], query: str) -> dict[str, Any] | None:
    """Find device by ID or fuzzy name match."""
    q = _normalize(query)
    # exact ID match
    for d in devices:
        if d["id"] == query:
            return d
    # exact name match (normalized)
    for d in devices:
        if _normalize(d["name"]) == q:
            return d
    # substring match (normalized)
    for d in devices:
        if q in _normalize(d["name"]):
            return d
    return None


def _connect_bulb(dev: dict[str, Any]) -> "tinytuya.BulbDevice":
    import tinytuya

    d = tinytuya.BulbDevice(dev["id"], dev["ip"], dev["local_key"], version=dev["version"])
    d.set_socketTimeout(5)
    d.set_socketRetryLimit(3)
    d.set_socketRetryDelay(2)
    d.set_socketPersistent(True)
    # Auto-detect bulb type (A vs B DPS layout) by fetching status first
    d.status()
    return d


def _connect_device(dev: dict[str, Any]) -> "tinytuya.Device":
    import tinytuya

    d = tinytuya.Device(dev["id"], dev["ip"], dev["local_key"], version=dev["version"])
    d.set_socketTimeout(5)
    d.set_socketRetryLimit(3)
    d.set_socketRetryDelay(2)
    return d


def _parse_bulb_status(dps: dict[str, Any]) -> dict[str, Any]:
    return {
        "on": dps.get("20", False),
        "mode": dps.get("21", "unknown"),
        "brightness": dps.get("22", 0),
        "brightness_pct": round(dps.get("22", 0) / 10, 1),
        "color_temp": dps.get("23", 0),
        "color_temp_pct": round(dps.get("23", 0) / 10, 1),
        "color_hex": dps.get("24", ""),
        "scene": dps.get("25", ""),
    }


def _parse_plug_status(dps: dict[str, Any]) -> dict[str, Any]:
    return {
        "on": dps.get("1", False),
        "current_ma": dps.get("18", 0),
        "power_w": round(dps.get("19", 0) / 10, 1),
        "voltage_v": round(dps.get("20", 0) / 10, 1),
    }


class Input(BaseModel):
    command: str = Field(
        description=(
            "list|status|on|off|brightness|color|temperature|scene|white — "
            "list: show all devices; status: get device state; on/off: toggle power; "
            "brightness: set brightness 0-100; color: set RGB; temperature: set warm/cool 0-100; "
            "scene: set scene (1=nature,3=rave,4=rainbow); white: set white mode"
        )
    )
    device: str = Field(
        default="",
        description=(
            "Device name, ID, or group target. "
            "Single: 'gu10', 'gu10 2', 'gu10 3', 'a60' (RGB bulb), 'cluster', 'sunlight'. "
            "Groups: 'all' (every controllable device), 'all_bulbs' (all bulbs), 'all_plugs' (all plugs). "
            "Groups apply the command to every matching device in one call."
        ),
    )
    brightness: int = Field(default=-1, description="Brightness percentage 0-100")
    r: int = Field(default=-1, description="Red value 0-255")
    g: int = Field(default=-1, description="Green value 0-255")
    b: int = Field(default=-1, description="Blue value 0-255")
    temperature: int = Field(default=-1, description="Color temperature 0-100 (0=warm, 100=cool)")
    scene: int = Field(default=-1, description="Scene number: 1=nature, 3=rave, 4=rainbow")


class Output(BaseModel):
    success: bool = True
    device: dict[str, Any] = Field(default_factory=dict)
    devices: list[dict[str, Any]] = Field(default_factory=list)
    status: dict[str, Any] = Field(default_factory=dict)
    count: int = 0
    error: str = ""


class TuyaLightsTool(ScriptTool[Input, Output]):
    name = "tuya_lights"
    description = "Control Tuya smart home devices: lights, plugs, switches"

    def execute(self, inp: Input) -> Output:
        return self._dispatch(inp)

    def _dispatch(self, inp: Input) -> Output:
        cmd = inp.command.lower().strip()

        if cmd == "list":
            return self._list()

        if cmd == "status":
            return self._status(inp)

        if cmd in ("on", "off"):
            return self._power(inp, on=cmd == "on")

        if cmd == "brightness":
            return self._brightness(inp)

        if cmd == "color":
            return self._color(inp)

        if cmd == "temperature":
            return self._temperature(inp)

        if cmd == "scene":
            return self._scene(inp)

        if cmd == "white":
            return self._white(inp)

        return Output(success=False, error=f"Unknown command: {cmd}")

    def _list(self) -> Output:
        devices = _load_registry()
        items = []
        for d in devices:
            items.append(
                {
                    "name": d["name"],
                    "id": d["id"],
                    "ip": d["ip"],
                    "category": d["category"],
                    "capabilities": d["capabilities"],
                    "online_lan": d["online_lan"],
                }
            )
        return Output(success=True, devices=items, count=len(items))

    def _resolve(self, inp: Input, *, allow_read_only: bool = False) -> tuple[dict[str, Any] | None, Output | None]:
        if not inp.device:
            return None, Output(success=False, error="device is required")
        devices = _load_registry()
        dev = _find_device(devices, inp.device)
        if not dev:
            names = [d["name"] for d in devices]
            return None, Output(success=False, error=f"Device not found: '{inp.device}'. Available: {names}")
        if not dev["online_lan"] or not dev["ip"]:
            return None, Output(success=False, error=f"Device '{dev['name']}' is not available on LAN")
        if dev.get("read_only") and not allow_read_only:
            return None, Output(
                success=False, error=f"Device '{dev['name']}' is read-only (status/power monitoring only)"
            )
        return dev, None

    def _status(self, inp: Input) -> Output:
        # If no device specified, get status of all LAN devices
        if not inp.device:
            devices = _load_registry()
            results = []
            for dev in devices:
                if not dev["online_lan"] or not dev["ip"]:
                    results.append({"name": dev["name"], "reachable": False})
                    continue
                try:
                    if dev["category"] == "bulb":
                        d = _connect_bulb(dev)
                        data = d.status()
                        if "dps" in data:
                            parsed = _parse_bulb_status(data["dps"])
                            parsed["name"] = dev["name"]
                            parsed["reachable"] = True
                            results.append(parsed)
                        else:
                            results.append({"name": dev["name"], "reachable": False, "error": str(data)})
                    else:
                        d = _connect_device(dev)
                        data = d.status()
                        if "dps" in data:
                            parsed = _parse_plug_status(data["dps"])
                            parsed["name"] = dev["name"]
                            parsed["reachable"] = True
                            results.append(parsed)
                        else:
                            results.append({"name": dev["name"], "reachable": False, "error": str(data)})
                except Exception as e:
                    results.append({"name": dev["name"], "reachable": False, "error": str(e)})
            return Output(success=True, devices=results, count=len(results))

        dev, err = self._resolve(inp, allow_read_only=True)
        if err:
            return err

        try:
            if dev["category"] == "bulb":
                d = _connect_bulb(dev)
                data = d.status()
                if "dps" in data:
                    return Output(success=True, status=_parse_bulb_status(data["dps"]), device={"name": dev["name"]})
                return Output(success=False, error=f"Bad response: {data}")
            else:
                d = _connect_device(dev)
                data = d.status()
                if "dps" in data:
                    return Output(success=True, status=_parse_plug_status(data["dps"]), device={"name": dev["name"]})
                return Output(success=False, error=f"Bad response: {data}")
        except Exception as e:
            return Output(success=False, error=str(e))

    def _apply_to_group(self, inp: Input, action: str) -> Output:
        """Apply a command to all devices in a group target."""
        group_devs = _resolve_group(inp.device)
        if not group_devs:
            return Output(success=False, error=f"No controllable devices matched group '{inp.device}'")

        results = []
        for dev in group_devs:
            if action == "on":
                r = self._power_single(dev, on=True)
            elif action == "off":
                r = self._power_single(dev, on=False)
            elif action == "brightness":
                r = self._brightness_single(dev, inp)
            elif action == "color":
                r = self._color_single(dev, inp)
            elif action == "temperature":
                r = self._temperature_single(dev, inp)
            elif action == "scene":
                r = self._scene_single(dev, inp)
            elif action == "white":
                r = self._white_single(dev, inp)
            else:
                r = {"name": dev["name"], "error": f"unsupported group action: {action}"}
            results.append(r)

        ok = sum(1 for r in results if r.get("success", False))
        return Output(success=ok > 0, devices=results, count=len(results))

    # ── Power ──

    def _power(self, inp: Input, *, on: bool) -> Output:
        if inp.device and _is_group_target(inp.device):
            return self._apply_to_group(inp, "on" if on else "off")
        dev, err = self._resolve(inp)
        if err:
            return err
        r = self._power_single(dev, on=on)
        return Output(success=r.get("success", False), status=r, device={"name": dev["name"]}, error=r.get("error", ""))

    def _power_single(self, dev: dict[str, Any], *, on: bool) -> dict[str, Any]:
        try:
            d = _connect_bulb(dev) if dev["category"] == "bulb" else _connect_device(dev)
            if on:
                d.turn_on()
            else:
                d.turn_off()
            return {"name": dev["name"], "success": True, "action": "on" if on else "off"}
        except Exception as e:
            return {"name": dev["name"], "success": False, "error": str(e)}

    # ── Brightness ──

    def _brightness(self, inp: Input) -> Output:
        if inp.device and _is_group_target(inp.device):
            return self._apply_to_group(inp, "brightness")
        dev, err = self._resolve(inp)
        if err:
            return err
        r = self._brightness_single(dev, inp)
        return Output(success=r.get("success", False), status=r, device={"name": dev["name"]}, error=r.get("error", ""))

    def _brightness_single(self, dev: dict[str, Any], inp: Input) -> dict[str, Any]:
        if dev["category"] != "bulb":
            return {"name": dev["name"], "success": False, "error": "not a bulb"}
        if not 0 <= inp.brightness <= 100:
            return {"name": dev["name"], "success": False, "error": "brightness must be 0-100"}
        try:
            d = _connect_bulb(dev)
            d.set_brightness(max(10, int(inp.brightness * 10)))
            return {"name": dev["name"], "success": True, "brightness_pct": inp.brightness}
        except Exception as e:
            return {"name": dev["name"], "success": False, "error": str(e)}

    # ── Color ──

    def _color(self, inp: Input) -> Output:
        if inp.device and _is_group_target(inp.device):
            return self._apply_to_group(inp, "color")
        dev, err = self._resolve(inp)
        if err:
            return err
        r = self._color_single(dev, inp)
        return Output(success=r.get("success", False), status=r, device={"name": dev["name"]}, error=r.get("error", ""))

    def _color_single(self, dev: dict[str, Any], inp: Input) -> dict[str, Any]:
        if "color_rgb" not in dev.get("capabilities", []):
            return {"name": dev["name"], "success": False, "error": "no RGB support"}
        if not (0 <= inp.r <= 255 and 0 <= inp.g <= 255 and 0 <= inp.b <= 255):
            return {"name": dev["name"], "success": False, "error": "r,g,b must be 0-255"}
        try:
            d = _connect_bulb(dev)
            d.set_colour(inp.r, inp.g, inp.b)
            return {"name": dev["name"], "success": True, "color": {"r": inp.r, "g": inp.g, "b": inp.b}}
        except Exception as e:
            return {"name": dev["name"], "success": False, "error": str(e)}

    # ── Temperature ──

    def _temperature(self, inp: Input) -> Output:
        if inp.device and _is_group_target(inp.device):
            return self._apply_to_group(inp, "temperature")
        dev, err = self._resolve(inp)
        if err:
            return err
        r = self._temperature_single(dev, inp)
        return Output(success=r.get("success", False), status=r, device={"name": dev["name"]}, error=r.get("error", ""))

    def _temperature_single(self, dev: dict[str, Any], inp: Input) -> dict[str, Any]:
        if dev["category"] != "bulb":
            return {"name": dev["name"], "success": False, "error": "not a bulb"}
        if not 0 <= inp.temperature <= 100:
            return {"name": dev["name"], "success": False, "error": "temperature must be 0-100"}
        try:
            d = _connect_bulb(dev)
            d.set_colourtemp(int(inp.temperature * 10))
            return {"name": dev["name"], "success": True, "color_temp_pct": inp.temperature}
        except Exception as e:
            return {"name": dev["name"], "success": False, "error": str(e)}

    # ── Scene ──

    def _scene(self, inp: Input) -> Output:
        if inp.device and _is_group_target(inp.device):
            return self._apply_to_group(inp, "scene")
        dev, err = self._resolve(inp)
        if err:
            return err
        r = self._scene_single(dev, inp)
        return Output(success=r.get("success", False), status=r, device={"name": dev["name"]}, error=r.get("error", ""))

    def _scene_single(self, dev: dict[str, Any], inp: Input) -> dict[str, Any]:
        if "scene" not in dev.get("capabilities", []):
            return {"name": dev["name"], "success": False, "error": "no scene support"}
        if inp.scene not in (1, 2, 3, 4):
            return {"name": dev["name"], "success": False, "error": "scene must be 1-4"}
        try:
            d = _connect_bulb(dev)
            d.set_scene(inp.scene)
            names = {1: "nature", 2: "scene_2", 3: "rave", 4: "rainbow"}
            return {"name": dev["name"], "success": True, "scene": names.get(inp.scene, str(inp.scene))}
        except Exception as e:
            return {"name": dev["name"], "success": False, "error": str(e)}

    # ── White ──

    def _white(self, inp: Input) -> Output:
        if inp.device and _is_group_target(inp.device):
            return self._apply_to_group(inp, "white")
        dev, err = self._resolve(inp)
        if err:
            return err
        r = self._white_single(dev, inp)
        return Output(success=r.get("success", False), status=r, device={"name": dev["name"]}, error=r.get("error", ""))

    def _white_single(self, dev: dict[str, Any], inp: Input) -> dict[str, Any]:
        if dev["category"] != "bulb":
            return {"name": dev["name"], "success": False, "error": "not a bulb"}
        try:
            d = _connect_bulb(dev)
            d.set_socketPersistent(True)
            d.set_mode("white")
            brightness_pct = inp.brightness if inp.brightness >= 0 else 100
            temp_pct = inp.temperature if inp.temperature >= 0 else 50
            d.set_brightness(max(10, int(brightness_pct * 10)))
            d.set_colourtemp(int(temp_pct * 10))
            return {"name": dev["name"], "success": True, "brightness_pct": brightness_pct, "color_temp_pct": temp_pct}
        except Exception as e:
            return {"name": dev["name"], "success": False, "error": str(e)}


if __name__ == "__main__":
    TuyaLightsTool.run()
