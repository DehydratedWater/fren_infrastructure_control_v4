"""fren v4 entrypoint — `python -m app <service>`.

Mirrors v3's process model as three standalone services (one container each in
compose):
  bot       — the Telegram bot (blocking run_polling)
  scheduler — the long-lived cron Scheduler (start → wait → stop, signal-aware)
  checker   — periodic intervention checker (one-shot tick on an interval loop)
  compile   — build the fleet into AGENTS_DIR (run once at boot before the bot)
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

log = logging.getLogger("app")


def _run_bot() -> None:
    from app.telegram.bot import run as run_bot

    run_bot()


async def _scheduler_main() -> None:
    from app.scheduler import Scheduler

    scheduler = Scheduler()
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _request_stop() -> None:
        log.info("shutdown signal received")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda *_: _request_stop())

    await scheduler.start()
    try:
        await stop.wait()
    finally:
        await scheduler.stop()


def _run_scheduler() -> None:
    asyncio.run(_scheduler_main())


async def _checker_main() -> None:
    """Run the periodic checker as a long-lived loop.

    PeriodicCheckerTool is one-shot (a ScriptTool-style tick); the service wraps
    it in an interval loop so the container stays up and intervenes on schedule.
    """
    from app.checker import Input, PeriodicCheckerTool
    from app.settings import get_settings

    interval = getattr(get_settings(), "checker_interval_seconds", 300)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda *_: stop.set())

    tool = PeriodicCheckerTool()
    while not stop.is_set():
        try:
            await tool.execute(Input(command="check"))
        except Exception:
            log.exception("checker tick failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def _run_checker() -> None:
    asyncio.run(_checker_main())


def _run_compile() -> None:
    """Compile the whole fleet (all worker variants) into settings.agents_dir."""
    from pathlib import Path

    from app.agents.compile import compile_fleet
    from app.settings import get_settings

    target = Path(get_settings().agents_dir)
    target.mkdir(parents=True, exist_ok=True)
    files = compile_fleet(target=target, project_root=target, clean=True)
    print(f"[compile] wrote {len(files)} files to {target}")


def _dispatch(service: str) -> None:
    if service == "bot":
        _run_bot()
    elif service == "scheduler":
        _run_scheduler()
    elif service == "checker":
        _run_checker()
    elif service == "compile":
        _run_compile()
    else:
        print(
            f"unknown service: {service!r} (use bot|scheduler|checker|compile)",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _dispatch(sys.argv[1] if len(sys.argv) > 1 else "bot")
