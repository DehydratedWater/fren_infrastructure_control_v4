"""Garmin health data tool -- on-demand health metrics for agents."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, date, datetime, timedelta

import httpx
from src import ScriptTool
from pydantic import BaseModel, Field

GRAFANA_URL = "http://192.168.0.70:3000"
GRAFANA_DATASOURCE_ID = 1  # garmin_influxdb


class Input(BaseModel):
    command: str = Field(description="current|sleep|daily|trend|summary")
    date: str = Field(default="", description="Date YYYY-MM-DD (default: today)")
    hours: int = Field(default=24, description="Hours lookback for trend (default: 24)")


class Output(BaseModel):
    success: bool = True
    data: dict = Field(default_factory=dict)
    error: str = ""


def _get_token() -> str | None:
    return os.environ.get("GRAPHANA_GARMIN_DATA_KEY")


async def _influx_query(client: httpx.AsyncClient, token: str, query: str) -> list[dict]:
    """Run an InfluxQL query via Grafana datasource proxy."""
    resp = await client.get(
        f"{GRAFANA_URL}/api/datasources/proxy/{GRAFANA_DATASOURCE_ID}/query",
        params={"db": "GarminStats", "q": query},
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [{}])
    series = results[0].get("series", [])
    if not series:
        return []
    cols = series[0]["columns"]
    return [dict(zip(cols, row, strict=False)) for row in series[0]["values"]]


def _parse_date(date_str: str) -> date:
    if date_str:
        return date.fromisoformat(date_str)
    return datetime.now(UTC).date()


def _time_range(target: date) -> tuple[str, str]:
    start = f"'{target.isoformat()}T00:00:00Z'"
    end = f"'{(target + timedelta(days=1)).isoformat()}T00:00:00Z'"
    return start, end


async def _fetch_current(client: httpx.AsyncClient, token: str) -> dict:
    """Latest body battery + stress + HR snapshot."""
    bb = await _influx_query(
        client,
        token,
        "SELECT last(BodyBatteryLevel) FROM BodyBatteryIntraday ORDER BY time DESC LIMIT 1",
    )
    stress = await _influx_query(
        client,
        token,
        "SELECT last(stressLevel) FROM StressIntraday WHERE stressLevel > 0 ORDER BY time DESC LIMIT 1",
    )
    hr = await _influx_query(
        client,
        token,
        "SELECT last(HeartRate) FROM HeartRateIntraday ORDER BY time DESC LIMIT 1",
    )
    result: dict = {}
    if bb and bb[0].get("last") is not None:
        result["body_battery"] = int(bb[0]["last"])
        result["timestamp"] = bb[0].get("time", "")
    if stress and stress[0].get("last") is not None:
        result["stress_level"] = int(stress[0]["last"])
    if hr and hr[0].get("last") is not None:
        result["heart_rate"] = int(hr[0]["last"])
    return result


async def _fetch_sleep(client: httpx.AsyncClient, token: str, target: date) -> dict:
    """Last night's sleep summary."""
    start, end = _time_range(target)
    rows = await _influx_query(
        client,
        token,
        f"SELECT sleepScore,sleepTimeSeconds,deepSleepSeconds,remSleepSeconds,"
        f"avgOvernightHrv,restingHeartRate,avgSleepStress "
        f"FROM SleepSummary WHERE time >= {start} AND time < {end} ORDER BY time DESC LIMIT 1",
    )
    if not rows:
        return {}
    r = rows[0]
    sleep_s = r.get("sleepTimeSeconds", 0) or 0
    deep_s = r.get("deepSleepSeconds", 0) or 0
    rem_s = r.get("remSleepSeconds", 0) or 0
    return {
        "sleep_score": r.get("sleepScore"),
        "total_hours": round(sleep_s / 3600, 1),
        "deep_minutes": round(deep_s / 60),
        "rem_minutes": round(rem_s / 60),
        "hrv": r.get("avgOvernightHrv"),
        "resting_hr": r.get("restingHeartRate"),
        "sleep_stress": r.get("avgSleepStress"),
    }


async def _fetch_daily(client: httpx.AsyncClient, token: str, target: date) -> dict:
    """Daily stats (steps, distance, calories, BB range, stress duration)."""
    start, end = _time_range(target)
    rows = await _influx_query(
        client,
        token,
        f"SELECT totalSteps,bodyBatteryHighestValue,bodyBatteryLowestValue,"
        f"activeKilocalories,restingHeartRate,highStressDuration,lowStressDuration,"
        f"restStressDuration,totalDistanceMeters "
        f"FROM DailyStats WHERE time >= {start} AND time < {end} ORDER BY time DESC LIMIT 1",
    )
    if not rows:
        return {}
    r = rows[0]
    dist_m = r.get("totalDistanceMeters", 0) or 0
    high_s = r.get("highStressDuration", 0) or 0
    low_s = r.get("lowStressDuration", 0) or 0
    rest_s = r.get("restStressDuration", 0) or 0
    return {
        "steps": r.get("totalSteps", 0),
        "distance_km": round(dist_m / 1000, 1),
        "active_kcal": round(r.get("activeKilocalories", 0) or 0),
        "body_battery_high": r.get("bodyBatteryHighestValue"),
        "body_battery_low": r.get("bodyBatteryLowestValue"),
        "high_stress_minutes": round(high_s / 60),
        "low_stress_minutes": round(low_s / 60),
        "rest_stress_minutes": round(rest_s / 60),
    }


async def _fetch_trend(client: httpx.AsyncClient, token: str, target: date, hours: int) -> dict:
    """Body battery + stress + HR intraday samples (15-min intervals)."""
    end_dt = datetime(target.year, target.month, target.day, tzinfo=UTC) + timedelta(days=1)
    start_dt = end_dt - timedelta(hours=hours)
    start = f"'{start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
    end = f"'{end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}'"

    bb_rows = await _influx_query(
        client,
        token,
        f"SELECT mean(BodyBatteryLevel) FROM BodyBatteryIntraday "
        f"WHERE time >= {start} AND time < {end} GROUP BY time(15m) fill(none)",
    )
    stress_rows = await _influx_query(
        client,
        token,
        f"SELECT mean(stressLevel) FROM StressIntraday "
        f"WHERE time >= {start} AND time < {end} AND stressLevel > 0 GROUP BY time(15m) fill(none)",
    )
    hr_rows = await _influx_query(
        client,
        token,
        f"SELECT mean(HeartRate) FROM HeartRateIntraday "
        f"WHERE time >= {start} AND time < {end} GROUP BY time(15m) fill(none)",
    )

    def _to_samples(rows: list[dict], key: str = "mean") -> list[dict]:
        out = []
        for r in rows:
            if r.get(key) is not None:
                t = r.get("time", "")[:16]
                time_part = t[11:16] if len(t) >= 16 else t
                out.append({"time": time_part, "value": round(r[key])})
        return out

    return {
        "body_battery": _to_samples(bb_rows),
        "stress": _to_samples(stress_rows),
        "heart_rate": _to_samples(hr_rows),
    }


class GarminHealthTool(ScriptTool[Input, Output]):
    name = "garmin_health"
    description = "Query Garmin health data: body battery, stress, heart rate, sleep, daily stats"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        token = _get_token()
        if not token:
            return Output(success=False, error="GRAPHANA_GARMIN_DATA_KEY not set")

        target = _parse_date(inp.date)

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                if inp.command == "current":
                    data = await _fetch_current(client, token)
                elif inp.command == "sleep":
                    data = await _fetch_sleep(client, token, target)
                elif inp.command == "daily":
                    data = await _fetch_daily(client, token, target)
                elif inp.command == "trend":
                    data = await _fetch_trend(client, token, target, inp.hours)
                elif inp.command == "summary":
                    data = {
                        "current": await _fetch_current(client, token),
                        "sleep": await _fetch_sleep(client, token, target),
                        "daily": await _fetch_daily(client, token, target),
                        "trend": await _fetch_trend(client, token, target, inp.hours),
                    }
                else:
                    return Output(success=False, error=f"Unknown command: {inp.command}")
        except httpx.HTTPError as e:
            return Output(success=False, error=f"Grafana request failed: {e}")

        return Output(success=True, data=data)


if __name__ == "__main__":
    GarminHealthTool.run()
