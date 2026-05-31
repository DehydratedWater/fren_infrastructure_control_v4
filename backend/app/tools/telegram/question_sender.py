"""Question Sender — Telegram questions with inline keyboards, rate limiting, dedup."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import UTC, datetime, timedelta

from src import ScriptTool
from pydantic import BaseModel, Field

MAX_QUESTIONS_PER_PERIOD = 3
PERIOD_HOURS = 4
DEDUP_HOURS = 4
OVERLAP_THRESHOLD = 0.75


def _normalize(text: str) -> str:
    """Normalize question text for dedup comparison."""
    t = text.lower().strip()
    t = re.sub(r"~\s*twily\b", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _hash_text(text: str) -> str:
    """8-char MD5 hash of first 50 chars of normalized text."""
    return hashlib.md5(_normalize(text)[:50].encode()).hexdigest()[:8]


def _word_overlap(a: str, b: str) -> float:
    """Word-level Jaccard overlap between two normalized texts."""
    wa = set(_normalize(a).split())
    wb = set(_normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class Input(BaseModel):
    command: str = Field(
        description="send-yesno|send-multiselect|record-answer|check-rate-limit|check-similar|"
        "get-question|get-unanswered|get-today|get-recent|get-hash"
    )
    question: str = Field(default="", description="Question text")
    question_id: str = Field(default="", description="Question ID")
    options: str = Field(default="", description="JSON options array for multiselect")
    answer: str = Field(default="", description="Answer text")
    selected: str = Field(default="", description="JSON selected options array")
    context_goal_id: str = Field(default="", description="Related goal ID")
    force: bool = Field(default=False, description="Bypass rate limit")
    hours: float = Field(default=4.0, description="Hours to look back for dedup")
    days: int = Field(default=7, description="Days to look back")


class Output(BaseModel):
    success: bool = True
    question_record: dict = Field(default_factory=dict)
    questions: list[dict] = Field(default_factory=list)
    count: int = 0
    allowed: bool = True
    remaining: int = 0
    similar_found: bool = False
    hash: str = ""
    normalized: str = ""
    error: str = ""


class QuestionSenderTool(ScriptTool[Input, Output]):
    name = "question_sender"
    description = "Send Telegram questions with inline keyboards, rate limiting, and dedup"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.chat import ChatMessagesRepo

        chat_repo = ChatMessagesRepo()
        cmd = inp.command

        # ── Rate Limit Check ──
        if cmd == "check-rate-limit":
            return await self._check_rate_limit(chat_repo)

        # ── Dedup Check ──
        if cmd == "check-similar":
            return await self._check_similar(chat_repo, inp.question, inp.hours)

        if cmd == "get-hash":
            n = _normalize(inp.question)
            return Output(success=True, hash=_hash_text(inp.question), normalized=n)

        # ── Send Yes/No ──
        if cmd == "send-yesno":
            return await self._send_question(chat_repo, inp, ["Yes", "No"], "yesno")

        # ── Send Multiselect ──
        if cmd == "send-multiselect":
            if not inp.options:
                return Output(success=False, error="options required for multiselect")
            opts = json.loads(inp.options)
            return await self._send_question(chat_repo, inp, opts, "multiselect")

        # ── Record Answer ──
        if cmd == "record-answer":
            return await self._record_answer(chat_repo, inp)

        # ── Get Question ──
        if cmd == "get-question":
            recent = await chat_repo.get_history(days=30, limit=1000)
            for msg in recent:
                meta = msg.get("metadata") or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except (json.JSONDecodeError, TypeError):
                        meta = {}
                if meta.get("question_id") == inp.question_id:
                    return Output(success=True, question_record=meta)
            return Output(success=False, error=f"Question not found: {inp.question_id}")

        # ── Get Unanswered ──
        if cmd == "get-unanswered":
            questions = await self._get_questions(chat_repo, days=inp.days)
            unanswered = [q for q in questions if not q.get("answered_at")]
            return Output(success=True, questions=unanswered, count=len(unanswered))

        # ── Get Today ──
        if cmd == "get-today":
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            msgs = await chat_repo.get_by_date(today, limit=200)
            questions = self._extract_questions(msgs)
            return Output(success=True, questions=questions, count=len(questions))

        # ── Get Recent ──
        if cmd == "get-recent":
            questions = await self._get_questions(chat_repo, days=inp.days)
            return Output(success=True, questions=questions, count=len(questions))

        return Output(success=False, error=f"Unknown command: {cmd}")

    async def _check_rate_limit(self, chat_repo: object) -> Output:
        """Check if we can send more questions (max 3 per 4 hours)."""
        questions = await self._get_questions(chat_repo, hours=PERIOD_HOURS)  # type: ignore[arg-type]
        count = len(questions)
        remaining = max(0, MAX_QUESTIONS_PER_PERIOD - count)
        return Output(
            success=True,
            allowed=remaining > 0,
            remaining=remaining,
            count=count,
        )

    async def _check_similar(self, chat_repo: object, question: str, hours: float) -> Output:
        """Check if a similar question was asked recently."""
        questions = await self._get_questions(chat_repo, hours=hours)  # type: ignore[arg-type]
        qhash = _hash_text(question)
        for q in questions:
            if q.get("hash") == qhash:
                return Output(success=True, similar_found=True, hash=qhash)
            overlap = _word_overlap(question, q.get("question", ""))
            if overlap >= OVERLAP_THRESHOLD:
                return Output(success=True, similar_found=True, hash=qhash)
        return Output(success=True, similar_found=False, hash=qhash)

    async def _send_question(self, chat_repo: object, inp: Input, options: list[str], qtype: str) -> Output:
        """Send a question via Telegram with inline keyboard."""
        # Rate limit check
        if not inp.force:
            rl = await self._check_rate_limit(chat_repo)
            if not rl.allowed:
                return Output(success=False, error="Rate limited: max 3 questions per 4 hours")
            # Dedup check
            sim = await self._check_similar(chat_repo, inp.question, DEDUP_HOURS)
            if sim.similar_found:
                return Output(success=False, error="Similar question already asked recently")

        qid = f"q_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
        qhash = _hash_text(inp.question)

        # Build inline keyboard
        from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

        from app.settings import get_settings

        settings = get_settings()
        bot = Bot(token=settings.bot_token)

        if qtype == "yesno":
            keyboard = [
                [
                    InlineKeyboardButton("Yes", callback_data=f"{qid}_yes"),
                    InlineKeyboardButton("No", callback_data=f"{qid}_no"),
                ]
            ]
        else:
            rows = []
            for i, opt in enumerate(options):
                rows.append([InlineKeyboardButton(opt, callback_data=f"{qid}_toggle_{i}")])
            rows.append([InlineKeyboardButton("Done", callback_data=f"{qid}_done")])
            keyboard = rows

        markup = InlineKeyboardMarkup(keyboard)
        msg = await bot.send_message(
            chat_id=settings.chat_id,
            text=inp.question,
            reply_markup=markup,
        )

        # Save to chat history with question metadata
        now = datetime.now(UTC)
        meta = {
            "question_id": qid,
            "question_type": qtype,
            "options": options,
            "hash": qhash,
            "context_goal_id": inp.context_goal_id or None,
            "telegram_message_id": msg.message_id,
            "sent_at": now.isoformat(),
            "answered_at": None,
            "answer": None,
            "selected_options": [],
        }
        await chat_repo.save(  # type: ignore[union-attr]
            sender="twily_question",
            message=inp.question,
            date=now.date(),
            timestamp=now,
            timestamp_unix=now.timestamp(),
            metadata=json.dumps(meta),
        )

        return Output(success=True, question_record=meta)

    async def _record_answer(self, chat_repo: object, inp: Input) -> Output:
        """Record an answer to a previously sent question."""
        from app.db.session import fetch_all, get_async_session

        # Find the question in recent messages
        async with get_async_session() as s:
            rows = await fetch_all(
                s,
                """
                SELECT id, metadata FROM chat_messages
                WHERE sender = 'twily_question'
                  AND date >= CURRENT_DATE - 7
                ORDER BY timestamp DESC LIMIT 200
            """,
            )

        for row in rows:
            meta = row.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    continue
            if meta.get("question_id") == inp.question_id:
                meta["answered_at"] = datetime.now(UTC).isoformat()
                meta["answer"] = inp.answer
                if inp.selected:
                    meta["selected_options"] = json.loads(inp.selected)

                from app.db.session import execute_sql

                async with get_async_session() as s:
                    await execute_sql(
                        s,
                        """
                        UPDATE chat_messages SET metadata = CAST(:meta AS jsonb) WHERE id = :id
                    """,
                        {"meta": json.dumps(meta), "id": row["id"]},
                    )
                return Output(success=True, question_record=meta)

        return Output(success=False, error=f"Question not found: {inp.question_id}")

    async def _get_questions(self, chat_repo: object, *, days: int = 7, hours: float = 0) -> list[dict]:
        """Get question records from chat history."""
        from app.db.session import fetch_all, get_async_session

        if hours > 0:
            cutoff = datetime.now(UTC) - timedelta(hours=hours)
            async with get_async_session() as s:
                rows = await fetch_all(
                    s,
                    """
                    SELECT metadata FROM chat_messages
                    WHERE sender = 'twily_question'
                      AND timestamp >= CAST(:cutoff AS timestamptz)
                    ORDER BY timestamp DESC LIMIT 200
                """,
                    {"cutoff": cutoff},
                )
        else:
            async with get_async_session() as s:
                rows = await fetch_all(
                    s,
                    """
                    SELECT metadata FROM chat_messages
                    WHERE sender = 'twily_question'
                      AND date >= CURRENT_DATE - CAST(:days AS integer)
                    ORDER BY timestamp DESC LIMIT 200
                """,
                    {"days": days},
                )

        questions = []
        for row in rows:
            meta = row.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    continue
            if meta.get("question_id"):
                questions.append(meta)
        return questions

    @staticmethod
    def _extract_questions(msgs: list[dict]) -> list[dict]:
        """Extract question metadata from chat message rows."""
        questions = []
        for msg in msgs:
            meta = msg.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    continue
            if meta.get("question_id"):
                questions.append(meta)
        return questions


if __name__ == "__main__":
    QuestionSenderTool.run()
