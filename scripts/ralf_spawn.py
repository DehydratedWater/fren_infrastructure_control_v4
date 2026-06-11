"""ralf_spawn — detached hand-off between RALF chain stages.

Usage (what the RALF agents' prompts instruct):
    python scripts/ralf_spawn.py workflows/twily_ralf_plan_evaluation ralf_id=<id>
    python scripts/ralf_spawn.py workflows/twily_ralf_execution ralf_id=<id> stage_number=2 attempt_number=1

Spawns the next chain agent as a DETACHED subprocess (start_new_session,
output discarded) via opencode_manager, so the calling agent's session can
end immediately — the chain self-drives stage to stage exactly like v3's
ralf_ping, but event-driven instead of polled. The child inherits env
(DATABASE_URL etc.), and cwd stays the compiled workspace so `scripts/` and
agent discovery keep working.

This script exists because the 2026-06-11 ralf-smoke probe caught the chain
stalling at plan_review: the agents' prompts referenced ralf_spawn.py, but
the script (v3's driver replacement) had never been created.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


def main() -> int:
    args = sys.argv[1:]
    if not args or "/" not in args[0]:
        print(json.dumps({"ok": False, "error": "usage: ralf_spawn.py <agent_id> [k=v ...]"}))
        return 2
    agent, params = args[0], args[1:]
    prompt = " ".join(params) if params else ""

    child = subprocess.Popen(  # noqa: S603 — fixed argv, params passed as one prompt arg
        [
            sys.executable, "scripts/opencode_manager.py",
            "--command", "run",
            "--agent", agent,
            "--prompt", prompt,
            "--timeout", "900",
        ],
        cwd=os.getcwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(json.dumps({"ok": True, "spawned": agent, "prompt": prompt, "pid": child.pid}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
