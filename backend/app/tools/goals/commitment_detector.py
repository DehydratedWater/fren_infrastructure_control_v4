"""Commitment detector — detect commitments in user messages."""

import asyncio
import hashlib
import re
import uuid
from typing import ClassVar

from src import ScriptTool
from pydantic import BaseModel, Field

# Commitment patterns with types and confidence
PATTERNS: list[tuple[str, str, float]] = [
    (r"I (?:will|'ll)\s+(.+?)(?:\.|$)", "will_statement", 0.8),
    (r"I must\s+(.+?)(?:\.|$)", "must_statement", 0.9),
    (r"I need to\s+(.+?)(?:\.|$)", "need_to", 0.7),
    (r"I (?:have to|gotta)\s+(.+?)(?:\.|$)", "have_to", 0.75),
    (r"I (?:should|ought to)\s+(.+?)(?:\.|$)", "should_statement", 0.5),
    (r"I promise\s+(.+?)(?:\.|$)", "promise", 0.95),
    (r"I (?:plan to|intend to)\s+(.+?)(?:\.|$)", "plan", 0.7),
    (r"I'm going to\s+(.+?)(?:\.|$)", "going_to", 0.75),
    (r"I want to\s+(.+?)(?:\.|$)", "want_to", 0.5),
    (r"I (?:commit|pledge) to\s+(.+?)(?:\.|$)", "commitment", 0.95),
    (r"my goal is to\s+(.+?)(?:\.|$)", "goal_statement", 0.7),
    (r"I'm determined to\s+(.+?)(?:\.|$)", "determination", 0.85),
    (r"I (?:aim|aspire) to\s+(.+?)(?:\.|$)", "aspiration", 0.6),
]

NEGATION_WORDS = {"not", "never", "don't", "won't", "can't", "couldn't", "shouldn't"}


class Input(BaseModel):
    command: str = Field(description="scan-message|get-pending|get-today|update-status")
    message: str = Field(default="", description="Message text to scan")
    commitment_id: str = Field(default="", description="Commitment ID for status update")
    status: str = Field(default="", description="New status (completed|cancelled|in-progress)")
    goal_id: str = Field(default="", description="Optional linked goal ID")


class Output(BaseModel):
    success: bool = True
    commitments: list[dict] = Field(default_factory=list)
    commitment: dict = Field(default_factory=dict)
    count: int = 0
    error: str = ""


class CommitmentDetectorTool(ScriptTool[Input, Output]):
    name: ClassVar[str] = "commitment_detector"
    description: ClassVar[str] = "Detect commitments in user messages"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        if inp.command == "scan-message":
            return self._scan(inp.message)
        if inp.command == "get-pending":
            return await self._get_pending()
        if inp.command == "get-today":
            return await self._get_today()
        if inp.command == "update-status":
            return await self._update(inp)

        return Output(success=False, error=f"Unknown command: {inp.command}")

    def _scan(self, message: str) -> Output:
        if not message:
            return Output(commitments=[], count=0)

        found: list[dict] = []
        seen: set[str] = set()
        for pattern, ptype, confidence in PATTERNS:
            for match in re.finditer(pattern, message, re.IGNORECASE):
                text = match.group(1).strip()
                full = match.group(0).strip()

                # Check for negation
                start = max(0, match.start() - 20)
                prefix = message[start : match.start()].lower()
                if any(w in prefix.split() for w in NEGATION_WORDS):
                    continue

                key = hashlib.md5(text.lower().encode()).hexdigest()[:8]
                if key in seen:
                    continue
                seen.add(key)

                found.append(
                    {
                        "commitment_id": str(uuid.uuid4())[:8],
                        "pattern_type": ptype,
                        "commitment_text": text,
                        "confidence": confidence,
                        "full_match": full,
                    }
                )

        return Output(commitments=found, count=len(found))

    async def _get_pending(self) -> Output:
        from app.db.repos.commitments import CommitmentsRepo

        rows = await CommitmentsRepo().get_pending()
        return Output(commitments=rows, count=len(rows))

    async def _get_today(self) -> Output:
        from app.db.repos.commitments import CommitmentsRepo

        rows = await CommitmentsRepo().get_today()
        return Output(commitments=rows, count=len(rows))

    async def _update(self, inp: Input) -> Output:
        from app.db.repos.commitments import CommitmentsRepo

        row = await CommitmentsRepo().update_status(inp.commitment_id, inp.status, goal_id=inp.goal_id or None)
        return Output(commitment=row or {}, success=row is not None)
