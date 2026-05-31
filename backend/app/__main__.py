"""Package entrypoint: ``python -m app <command>``.

v3 had no top-level ``__main__``: the scheduler (``fren/telegram/scheduler.py``)
ran inside the long-lived Telegram bot process, and the checker / cron-manager
were ``script:`` ScriptTools invoked per-tick. This v4 module provides a thin
dispatcher so the same background-job runtime is runnable standalone:

  python -m app scheduler   # run the long-lived cron scheduler (signal-aware)
  python -m app checker      # run a single periodic-checker tick
  python -m app cron ...     # drive cron/workflow execution logging

Signal handling (SIGINT/SIGTERM → graceful Scheduler.stop) is installed for the
scheduler command, mirroring how v3's bot drove Scheduler.start()/stop().
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys


async def _run_scheduler() -> None:
    from app.scheduler import Scheduler

    scheduler = Scheduler()
    loop = asyncio.get_running_loop()
    stop_requested = asyncio.Event()

    def _request_stop() -> None:
        logging.getLogger("app").info("shutdown signal received")
        stop_requested.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # add_signal_handler is unavailable on some platforms (e.g. Windows).
            signal.signal(sig, lambda *_: _request_stop())

    await scheduler.start()
    try:
        await stop_requested.wait()
    finally:
        await scheduler.stop()


def _cmd_scheduler(_args: argparse.Namespace) -> int:
    asyncio.run(_run_scheduler())
    return 0


def _cmd_checker(args: argparse.Namespace) -> int:
    from app.checker import Input, PeriodicCheckerTool

    out = PeriodicCheckerTool().execute(Input(command=args.command, force=args.force))
    import json

    print(json.dumps(out.model_dump(), default=str))
    return 0 if out.success else 1


def _cmd_cron(args: argparse.Namespace) -> int:
    from app.cron_runner import CronManagerTool, Input

    out = CronManagerTool().execute(
        Input(
            command=args.cron_command,
            execution_id=args.execution_id,
            mode=args.mode,
            triggered_by=args.triggered_by,
            exit_code=args.exit_code,
            status=args.status,
            limit=args.limit,
        )
    )
    import json

    print(json.dumps(out.model_dump(), default=str))
    return 0 if out.success else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app", description="fren infra control background-job runtime")
    sub = parser.add_subparsers(dest="command_name", required=True)

    p_sched = sub.add_parser("scheduler", help="run the long-lived cron scheduler")
    p_sched.set_defaults(func=_cmd_scheduler)

    p_check = sub.add_parser("checker", help="run a single periodic-checker tick")
    p_check.add_argument("--command", default="check", help="check|get-state|dry-run")
    p_check.add_argument("--force", action="store_true")
    p_check.set_defaults(func=_cmd_checker)

    p_cron = sub.add_parser("cron", help="drive cron/workflow execution logging")
    p_cron.add_argument("cron_command", help="log-start|log-complete|list-recent|workflow-*")
    p_cron.add_argument("--execution-id", default="")
    p_cron.add_argument("--mode", default="")
    p_cron.add_argument("--triggered-by", default="cron")
    p_cron.add_argument("--exit-code", type=int, default=0)
    p_cron.add_argument("--status", default="completed")
    p_cron.add_argument("--limit", type=int, default=20)
    p_cron.set_defaults(func=_cmd_cron)

    return parser


def _main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
