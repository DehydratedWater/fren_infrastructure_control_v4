"""Session Inspector — browse opencode agent session data, messages, and tool calls.

Reads from the opencode SQLite database at .opencode/data/opencode/opencode.db.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, ClassVar

from src import ScriptTool
from pydantic import BaseModel, Field

DB_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent / ".opencode" / "data" / "opencode" / "opencode.db"
)


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _session_summary(row: sqlite3.Row) -> dict:
    return {
        "session_id": row["id"],
        "title": row["title"],
        "parent_id": row["parent_id"] or "",
        "created": row["time_created"],
        "updated": row["time_updated"],
        "version": row["version"],
    }


class Input(BaseModel):
    command: str = Field(description="find-by-time|find-by-text|get-session-tree|get-messages|list-recent")
    timestamp: float = Field(default=0, description="Unix seconds for find-by-time")
    window_seconds: int = Field(default=600, description="Search window for find-by-time (default 600s = 10min)")
    query: str = Field(default="", description="Text to search in message parts (for find-by-text)")
    session_id: str = Field(default="", description="Session ID for get-session-tree|get-messages")
    limit: int = Field(default=10, description="Max items to return")


class Output(BaseModel):
    success: bool = True
    sessions: list[dict] = Field(default_factory=list)
    messages: list[dict] = Field(default_factory=list)
    error: str = ""


class SessionInspectorTool(ScriptTool[Input, Output]):
    name: ClassVar[str] = "session_inspector"
    description: ClassVar[str] = "Browse opencode agent session data, messages, and tool calls"

    def execute(self, inp: Input) -> Output:
        if not DB_PATH.exists():
            return Output(success=False, error=f"Database not found: {DB_PATH}")

        cmd = inp.command
        if cmd == "find-by-time":
            return self._find_by_time(inp.timestamp, inp.window_seconds, inp.limit)
        if cmd == "find-by-text":
            return self._find_by_text(inp.query, inp.limit)
        if cmd == "get-session-tree":
            return self._get_session_tree(inp.session_id)
        if cmd == "get-messages":
            return self._get_messages(inp.session_id, inp.limit)
        if cmd == "list-recent":
            return self._list_recent(inp.limit)

        return Output(success=False, error=f"Unknown command: {cmd}")

    def _find_by_time(self, timestamp: float, window: int, limit: int) -> Output:
        if not timestamp:
            return Output(success=False, error="timestamp is required")

        # Convert unix seconds to milliseconds for DB comparison
        ts_ms = int(timestamp * 1000)
        window_ms = window * 1000

        conn = _get_db()
        try:
            rows = conn.execute(
                """
                SELECT * FROM session
                WHERE parent_id IS NULL OR parent_id = ''
                ORDER BY ABS(time_created - ?) ASC
                LIMIT ?
                """,
                (ts_ms, limit * 3),
            ).fetchall()

            # Filter by window and take limit
            matches = []
            for row in rows:
                delta_ms = abs(row["time_created"] - ts_ms)
                if delta_ms <= window_ms:
                    matches.append(row)
                if len(matches) >= limit:
                    break

            return Output(sessions=[_session_summary(r) for r in matches])
        finally:
            conn.close()

    def _find_by_text(self, query: str, limit: int) -> Output:
        """Search message/part text content for matching sessions."""
        if not query:
            return Output(success=False, error="query is required")

        conn = _get_db()
        try:
            # Search part text content for the query string
            rows = conn.execute(
                """
                SELECT DISTINCT s.* FROM session s
                JOIN part p ON p.session_id = s.id
                WHERE p.data LIKE ?
                ORDER BY s.time_created DESC
                LIMIT ?
                """,
                (f"%{query}%", limit),
            ).fetchall()

            return Output(sessions=[_session_summary(r) for r in rows])
        finally:
            conn.close()

    def _get_session_tree(self, session_id: str) -> Output:
        if not session_id:
            return Output(success=False, error="session_id is required")

        conn = _get_db()
        try:
            # Find the session
            row = conn.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
            if not row:
                return Output(success=False, error=f"Session not found: {session_id}")

            # Walk up to root
            root_id = session_id
            visited: set[str] = set()
            while True:
                if root_id in visited:
                    break
                visited.add(root_id)
                r = conn.execute("SELECT parent_id FROM session WHERE id = ?", (root_id,)).fetchone()
                if not r or not r["parent_id"]:
                    break
                root_id = r["parent_id"]

            # BFS from root
            tree: list[dict[str, Any]] = []
            queue: list[tuple[str, int]] = [(root_id, 0)]
            visited_bfs: set[str] = set()

            while queue:
                sid, depth = queue.pop(0)
                if sid in visited_bfs:
                    continue
                visited_bfs.add(sid)

                r = conn.execute("SELECT * FROM session WHERE id = ?", (sid,)).fetchone()
                if r:
                    summary = _session_summary(r)
                    summary["depth"] = depth
                    tree.append(summary)

                    children = conn.execute(
                        "SELECT id FROM session WHERE parent_id = ? ORDER BY time_created",
                        (sid,),
                    ).fetchall()
                    for child in children:
                        queue.append((child["id"], depth + 1))

            return Output(sessions=tree)
        finally:
            conn.close()

    def _get_messages(self, session_id: str, limit: int) -> Output:
        if not session_id:
            return Output(success=False, error="session_id is required")

        conn = _get_db()
        try:
            msg_rows = conn.execute(
                "SELECT * FROM message WHERE session_id = ? ORDER BY time_created LIMIT ?",
                (session_id, limit),
            ).fetchall()

            if not msg_rows:
                return Output(success=False, error=f"No messages for session: {session_id}")

            messages: list[dict] = []
            for msg in msg_rows:
                msg_data = json.loads(msg["data"]) if msg["data"] else {}
                msg_id = msg["id"]

                # Load parts
                part_rows = conn.execute(
                    "SELECT * FROM part WHERE message_id = ? ORDER BY time_created",
                    (msg_id,),
                ).fetchall()

                parts: list[dict] = []
                for p in part_rows:
                    part_data = json.loads(p["data"]) if p["data"] else {}
                    # Truncate large text
                    text = part_data.get("text", "")
                    if len(text) > 2000:
                        part_data["text"] = text[:2000] + f"\n... [truncated, {len(text)} chars total]"
                    parts.append(part_data)

                messages.append(
                    {
                        "message_id": msg_id,
                        "role": msg_data.get("role", ""),
                        "agent": msg_data.get("agent", ""),
                        "created": msg["time_created"],
                        "summary_title": msg_data.get("summary", {}).get("title", ""),
                        "model": msg_data.get("model", msg_data.get("modelID", "")),
                        "parts": parts,
                    }
                )

            return Output(messages=messages)
        finally:
            conn.close()

    def _list_recent(self, limit: int) -> Output:
        conn = _get_db()
        try:
            rows = conn.execute(
                """
                SELECT * FROM session
                WHERE parent_id IS NULL OR parent_id = ''
                ORDER BY time_created DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return Output(sessions=[_session_summary(r) for r in rows])
        finally:
            conn.close()


if __name__ == "__main__":
    SessionInspectorTool.run()
