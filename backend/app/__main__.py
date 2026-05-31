"""fren v4 entrypoint — `python -m app <service>`.

Mirrors v3's process model: the Telegram bot, the cron scheduler, and the
periodic checker each run as their own service (one container each in compose).
`compile` builds the fleet into AGENTS_DIR (run once at boot before the bot).
"""

from __future__ import annotations

import asyncio
import signal
import sys


def _run_bot() -> None:
    from app.telegram.bot import run as run_bot

    run_bot()


def _run_scheduler() -> None:
    from app.scheduler import Scheduler

    async def _main() -> None:
        sched = Scheduler()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, sched.stop)
            except (NotImplementedError, AttributeError):
                pass
        await sched.run()

    asyncio.run(_main())


def _run_checker() -> None:
    from app.checker import Checker

    async def _main() -> None:
        chk = Checker()
        await chk.run()

    asyncio.run(_main())


def _run_compile() -> None:
    """Compile the whole fleet (all worker variants) into settings.agents_dir.

    Run once at boot before the bot/scheduler start so opencode can resolve
    `--agent <name><postfix>` from the compiled tree.
    """
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
    _dispatch(sys.argv[1] if len(sys.argv) > 1 else "bot")
