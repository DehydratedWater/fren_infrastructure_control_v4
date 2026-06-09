"""fren v4 entrypoint — `python -m app <service>`.

Mirrors v3's process model as three standalone services (one container each in
compose):
  bot       — the Telegram bot (blocking run_polling)
  scheduler — the long-lived cron Scheduler (start → wait → stop, signal-aware)
  checker   — periodic intervention checker (one-shot tick on an interval loop)
  compile   — build the fleet into AGENTS_DIR (run once at boot before the bot)
  probe-packs — generate corpus-grounded autoloop probe packs (offline, one-shot)
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
            # await the async path directly: execute() wraps asyncio.run(), which
            # raises "cannot be called from a running event loop" inside this loop.
            await tool._dispatch(Input(command="check"))
        except Exception:
            log.exception("checker tick failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def _run_checker() -> None:
    asyncio.run(_checker_main())


def _run_web() -> None:
    """Serve the read-only monitoring dashboard on 0.0.0.0:8000 via uvicorn."""
    import uvicorn

    uvicorn.run(
        "app.web.app:app",
        host="0.0.0.0",  # noqa: S104 — container service, port published in compose
        port=8000,
        log_level="info",
    )


def _run_compile() -> None:
    """Compile the whole fleet (all worker variants) into settings.agents_dir."""
    from pathlib import Path

    from app.agents.compile import compile_fleet
    from app.settings import get_settings

    target = Path(get_settings().agents_dir)
    target.mkdir(parents=True, exist_ok=True)
    # Promotions live at the REPO root (.oac/promoted), not in the agents output
    # dir. Walk up from this module to find the dir that actually holds them so
    # register_with_improvements applies the optimized prompts (else: baselines).
    here = Path(__file__).resolve()
    project_root = target
    for parent in here.parents:
        if (parent / ".oac" / "promoted").is_dir():
            project_root = parent
            break
    files = compile_fleet(target=target, project_root=project_root, clean=True)
    print(
        f"[compile] wrote {len(files)} files to {target}"
        f" (promotions from {project_root}/.oac/promoted)"
    )


def _run_improve(argv: list[str]) -> None:
    """Autoresearch the fleet: live LLM prompt-rewriting + opencode scoring,
    promoting winners into .oac/promoted/.

    Usage:
      python -m app improve [--agent ID]... [--rounds N] [--workers N]
                            [--threshold F] [--no-branches] [--list]
    """
    import argparse
    from pathlib import Path

    from app.agents.improve import GRADED, run_improvement
    from app.agents.improve_live import (
        ZaiJudge,
        ZaiPromptRewriter,
        live_agent_runner_factory,
        live_branch_invoker_factory_for,
    )
    from app.agents.registry import PROJECT_ROOT, all_agents

    p = argparse.ArgumentParser(prog="app improve")
    p.add_argument("--agent", action="append", default=[], help="restrict to agent id(s)")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.75,
                   help="promote when winner score_floor >= this (graded judge)")
    p.add_argument("--no-branches", action="store_true")
    p.add_argument("--substring-tests", action="store_true",
                   help="use the agents' authored (substring) tests instead of the "
                        "generated graded judge test")
    p.add_argument("--proactive-probes", action="store_true",
                   help="optimise the PROACTIVE agents against the context-signal "
                        "probe suite (variety / anti-repetition / grounded / skip). "
                        "Runs authored tests on nudge_strategist, periodic_checker, "
                        "winddown. Equivalent to --substring-tests --agent <each>.")
    p.add_argument("--list", action="store_true", help="list improvable agents and exit")
    args = p.parse_args(argv)

    # --proactive-probes: target exactly the proactive agents that carry the
    # context-signal probe suite, in authored-tests mode so the probes run.
    if args.proactive_probes:
        args.substring_tests = True
        if not args.agent:
            args.agent = [
                "goals/nudge_strategist",
                "goals/periodic_checker",
                "goals/winddown",
            ]

    # Default mode: generated graded judge test on EVERY agent (137 improvable).
    use_judge_test = not args.substring_tests
    if use_judge_test:
        improvable = [a.header.agent_id for a in all_agents()]
    else:
        improvable = [a.header.agent_id for a in all_agents() if a.agent_tests]
    if args.list:
        mode = "judge-test (all agents)" if use_judge_test else "authored agent_tests"
        print(f"{len(improvable)} improvable agents [{mode}]:")
        for aid in improvable:
            print(" ", aid)
        return

    only = set(args.agent) or None
    snaps = PROJECT_ROOT / ".oac" / "snapshots"
    snaps.mkdir(parents=True, exist_ok=True)

    judge = None if args.substring_tests else ZaiJudge()
    criterion = run_improvement_criterion = (
        GRADED if use_judge_test else None
    )

    log.info(
        "autoloop starting: %s agents%s, rounds=%d, workers=%d, threshold=%.2f, "
        "mode=%s (teacher rewrites+judges, agents tuned on local qwen)",
        len(only) if only else len(improvable),
        "" if only else " (all)",
        args.rounds, args.workers, args.threshold,
        "graded-judge" if use_judge_test else "substring",
    )
    kw = {}
    if criterion is not None:
        kw["criterion"] = criterion
    result = run_improvement(
        agent_runner_factory=live_agent_runner_factory,
        branch_invoker_factory_for=live_branch_invoker_factory_for,
        snapshots_dir=snaps,
        promote_threshold=args.threshold,
        project_root=PROJECT_ROOT,
        max_workers=args.workers,
        llm=ZaiPromptRewriter(),
        judge=judge,
        use_judge_test=use_judge_test,
        only=only,
        max_rounds=args.rounds,
        **kw,
        include_branches=not args.no_branches,
    )
    s = result.summary()
    print("\n========== AUTORESEARCH SUMMARY ==========")
    print(f"units={s['units']} succeeded={s['succeeded']} failed={s['failed']} "
          f"promoted={s['promoted']} mean_winner_score={s['mean_winner_score']:.3f}")
    for o in result.outcomes:
        flag = "PROMOTED" if o.promoted else ("ERR" if o.error else "kept")
        print(f"  [{flag:8}] {o.unit_id:42} score={o.winner_score:.3f}"
              + (f"  {o.error}" if o.error else ""))
    if result.failed():
        print(f"\n{len(result.failed())} unit(s) errored (see above).")


def _run_probe_packs(argv: list[str]) -> None:
    """Generate corpus-grounded probe packs (one teacher call per agent).

    Usage:
      python -m app probe-packs [--agents id ...] [--domains prefix ...]
                                [--per-agent K] [--refresh] [--workers N]
                                [--corpus-limit N]

    Samples REAL user messages from the read-only v3 DB (docker-fren-db-1) and
    asks the GLM teacher (via opencode) for K self-contained probes + judge
    criteria per agent, persisted under app/agents/probe_packs/. The autoloop's
    judge-test mode picks them up automatically. Re-run with --refresh after a
    model switch or usage drift (packs are regenerable, not sacred).
    """
    import argparse

    from app.agents.probe_packs import generate_packs
    from app.agents.registry import all_agents

    p = argparse.ArgumentParser(prog="app probe-packs")
    p.add_argument("--agents", nargs="+", default=None,
                   help="restrict to these agent id(s)")
    p.add_argument("--domains", nargs="+", default=None,
                   help="restrict to these domain prefixes (e.g. goals food)")
    p.add_argument("--per-agent", type=int, default=5,
                   help="probes per agent (default 5)")
    p.add_argument("--refresh", action="store_true",
                   help="regenerate packs that already exist")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--corpus-limit", type=int, default=40,
                   help="corpus messages sampled per agent (default 40)")
    args = p.parse_args(argv)

    all_ids = [a.header.agent_id for a in all_agents()]
    ids: list[str] | None = None
    if args.agents or args.domains:
        picked: set[str] = set()
        if args.agents:
            known = set(all_ids)
            for aid in args.agents:
                if aid in known:
                    picked.add(aid)
                else:
                    print(f"[probe-packs] WARNING: unknown agent id {aid!r}",
                          file=sys.stderr)
        if args.domains:
            prefixes = set(args.domains)
            picked.update(i for i in all_ids if i.split("/", 1)[0] in prefixes)
        ids = sorted(picked)
        if not ids:
            print("[probe-packs] no agents matched the filters", file=sys.stderr)
            sys.exit(2)

    results = generate_packs(
        ids,
        per_agent=args.per_agent,
        refresh=args.refresh,
        workers=args.workers,
        corpus_limit=args.corpus_limit,
    )
    counts = {"ok": 0, "skipped": 0, "error": 0}
    print("\n========== PROBE-PACK SUMMARY ==========")
    for aid in sorted(results):
        status = results[aid]
        counts["error" if status.startswith("error") else status] += 1
        print(f"  [{status.split(':')[0]:7}] {aid:42} "
              + (status if status.startswith("error") else ""))
    print(f"\nok={counts['ok']} skipped={counts['skipped']} "
          f"error={counts['error']} (of {len(results)})")
    if results and counts["ok"] == 0 and counts["skipped"] == 0:
        sys.exit(1)


def _dispatch(service: str, argv: list[str]) -> None:
    if service == "bot":
        _run_bot()
    elif service == "scheduler":
        _run_scheduler()
    elif service == "checker":
        _run_checker()
    elif service == "web":
        _run_web()
    elif service == "compile":
        _run_compile()
    elif service == "improve":
        _run_improve(argv)
    elif service == "probe-packs":
        _run_probe_packs(argv)
    else:
        print(
            f"unknown service: {service!r} "
            "(use bot|scheduler|checker|web|compile|improve|probe-packs)",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _dispatch(
        sys.argv[1] if len(sys.argv) > 1 else "bot",
        sys.argv[2:],
    )
