"""Chat messages repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import datetime as dt

from app.db.session import fetch_all, fetch_one, get_async_session


class ChatMessagesRepo:
    async def save(
        self,
        sender: str,
        message: str,
        date: dt.date,
        timestamp: dt.datetime,
        timestamp_unix: float,
        *,
        chat_id: str | None = None,
        message_id: int | None = None,
        username: str | None = None,
        metadata: str = "{}",
        content_class: str = "public",
        sfw_summary: str | None = None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "ts": timestamp,
            "ts_unix": timestamp_unix,
            "sender": sender,
            "message": message,
            "chat_id": chat_id,
            "msg_id": message_id,
            "username": username,
            "meta": metadata,
            "date": date,
            "content_class": content_class,
            "sfw_summary": sfw_summary,
        }
        if embedding:
            vec = "[" + ",".join(str(v) for v in embedding) + "]"
            params["embedding"] = vec
            sql = """
                INSERT INTO chat_messages (timestamp, timestamp_unix, sender, message,
                    chat_id, message_id, username, metadata, date,
                    content_class, sfw_summary, embedding)
                VALUES (:ts, :ts_unix, :sender, :message,
                    :chat_id, :msg_id, :username, CAST(:meta AS jsonb), :date,
                    :content_class, :sfw_summary, CAST(:embedding AS vector))
                RETURNING *
            """
        else:
            sql = """
                INSERT INTO chat_messages (timestamp, timestamp_unix, sender, message,
                    chat_id, message_id, username, metadata, date,
                    content_class, sfw_summary)
                VALUES (:ts, :ts_unix, :sender, :message,
                    :chat_id, :msg_id, :username, CAST(:meta AS jsonb), :date,
                    :content_class, :sfw_summary)
                RETURNING *
            """
        async with get_async_session() as s:
            return await fetch_one(s, sql, params)  # type: ignore[return-value]

    async def get_recent(self, *, limit: int = 50, offset: int = 0, clearance: str = "full") -> list[dict[str, Any]]:
        async with get_async_session() as s:
            if clearance == "full":
                return await fetch_all(
                    s,
                    """
                    SELECT * FROM chat_messages
                    ORDER BY timestamp DESC LIMIT :limit OFFSET :offset
                """,
                    {"limit": limit, "offset": offset},
                )
            return await fetch_all(
                s,
                """
                SELECT id, timestamp, timestamp_unix, sender,
                    CASE WHEN content_class = 'public' THEN message
                         ELSE COALESCE(sfw_summary, '[redacted]') END AS message,
                    chat_id, message_id, username, metadata, date, content_class
                FROM chat_messages
                ORDER BY timestamp DESC LIMIT :limit OFFSET :offset
            """,
                {"limit": limit, "offset": offset},
            )

    async def get_by_date(
        self, date: dt.date, *, limit: int = 200, offset: int = 0, clearance: str = "full"
    ) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            if clearance == "full":
                return await fetch_all(
                    s,
                    """
                    SELECT * FROM (
                        SELECT * FROM chat_messages WHERE date = CAST(:date AS date)
                        ORDER BY timestamp DESC LIMIT :limit OFFSET :offset
                    ) t ORDER BY timestamp ASC
                """,
                    {"date": date, "limit": limit, "offset": offset},
                )
            return await fetch_all(
                s,
                """
                SELECT * FROM (
                    SELECT id, timestamp, timestamp_unix, sender,
                        CASE WHEN content_class = 'public' THEN message
                             ELSE COALESCE(sfw_summary, '[redacted]') END AS message,
                        chat_id, message_id, username, metadata, date, content_class
                    FROM chat_messages WHERE date = CAST(:date AS date)
                    ORDER BY timestamp DESC LIMIT :limit OFFSET :offset
                ) t ORDER BY timestamp ASC
            """,
                {"date": date, "limit": limit, "offset": offset},
            )

    async def get_since_id(
        self, message_id: int, *, limit: int = 200, offset: int = 0, clearance: str = "full"
    ) -> list[dict[str, Any]]:
        """Get messages with id > message_id, ordered chronologically."""
        async with get_async_session() as s:
            if clearance == "full":
                return await fetch_all(
                    s,
                    """
                    SELECT * FROM chat_messages
                    WHERE id > :mid
                    ORDER BY id ASC LIMIT :limit OFFSET :offset
                """,
                    {"mid": message_id, "limit": limit, "offset": offset},
                )
            return await fetch_all(
                s,
                """
                SELECT id, timestamp, timestamp_unix, sender,
                    CASE WHEN content_class = 'public' THEN message
                         ELSE COALESCE(sfw_summary, '[redacted]') END AS message,
                    chat_id, message_id, username, metadata, date, content_class
                FROM chat_messages
                WHERE id > :mid
                ORDER BY id ASC LIMIT :limit OFFSET :offset
            """,
                {"mid": message_id, "limit": limit, "offset": offset},
            )

    async def get_since_hours(
        self, *, hours: int = 4, limit: int = 100, clearance: str = "full"
    ) -> list[dict[str, Any]]:
        """Get messages from the last N hours, ordered chronologically (oldest first)."""
        async with get_async_session() as s:
            if clearance == "full":
                return await fetch_all(
                    s,
                    """
                    SELECT * FROM (
                        SELECT * FROM chat_messages
                        WHERE timestamp >= NOW() - make_interval(hours => :hours)
                        ORDER BY timestamp DESC LIMIT :limit
                    ) t ORDER BY timestamp ASC
                """,
                    {"hours": hours, "limit": limit},
                )
            return await fetch_all(
                s,
                """
                SELECT * FROM (
                    SELECT id, timestamp, timestamp_unix, sender,
                        CASE WHEN content_class = 'public' THEN message
                             ELSE COALESCE(sfw_summary, '[redacted]') END AS message,
                        chat_id, message_id, username, metadata, date, content_class
                    FROM chat_messages
                    WHERE timestamp >= NOW() - make_interval(hours => :hours)
                    ORDER BY timestamp DESC LIMIT :limit
                ) t ORDER BY timestamp ASC
            """,
                {"hours": hours, "limit": limit},
            )

    async def get_range(
        self,
        *,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        only_with_urls: bool = False,
        sender: str | None = None,
        limit: int = 500,
        offset: int = 0,
        clearance: str = "full",
    ) -> list[dict[str, Any]]:
        """Fetch messages within an explicit date range with optional URL / sender filters.

        Uses `date >= :from_date AND date <= :to_date` when bounds given; omit a bound
        to leave it open. Results ordered oldest→newest so the caller sees the timeline.
        """
        where: list[str] = []
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if from_date is not None:
            where.append("date >= CAST(:from_date AS date)")
            params["from_date"] = from_date
        if to_date is not None:
            where.append("date <= CAST(:to_date AS date)")
            params["to_date"] = to_date
        if sender:
            where.append("sender = :sender")
            params["sender"] = sender
        if only_with_urls:
            where.append("(message LIKE '%http://%' OR message LIKE '%https://%')")
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        if clearance == "full":
            sql = f"""
                SELECT * FROM (
                    SELECT * FROM chat_messages{where_sql}
                    ORDER BY timestamp DESC LIMIT :limit OFFSET :offset
                ) t ORDER BY timestamp ASC
            """
        else:
            sql = f"""
                SELECT * FROM (
                    SELECT id, timestamp, timestamp_unix, sender,
                        CASE WHEN content_class = 'public' THEN message
                             ELSE COALESCE(sfw_summary, '[redacted]') END AS message,
                        chat_id, message_id, username, metadata, date, content_class
                    FROM chat_messages{where_sql}
                    ORDER BY timestamp DESC LIMIT :limit OFFSET :offset
                ) t ORDER BY timestamp ASC
            """
        async with get_async_session() as s:
            return await fetch_all(s, sql, params)

    async def get_history(
        self, *, days: int = 7, limit: int = 200, offset: int = 0, clearance: str = "full"
    ) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            if clearance == "full":
                return await fetch_all(
                    s,
                    """
                    SELECT * FROM (
                        SELECT * FROM chat_messages
                        WHERE date >= CURRENT_DATE - CAST(:days AS integer)
                        ORDER BY timestamp DESC LIMIT :limit OFFSET :offset
                    ) t ORDER BY timestamp ASC
                """,
                    {"days": days, "limit": limit, "offset": offset},
                )
            return await fetch_all(
                s,
                """
                SELECT * FROM (
                    SELECT id, timestamp, timestamp_unix, sender,
                        CASE WHEN content_class = 'public' THEN message
                             ELSE COALESCE(sfw_summary, '[redacted]') END AS message,
                        chat_id, message_id, username, metadata, date, content_class
                    FROM chat_messages
                    WHERE date >= CURRENT_DATE - CAST(:days AS integer)
                    ORDER BY timestamp DESC LIMIT :limit OFFSET :offset
                ) t ORDER BY timestamp ASC
            """,
                {"days": days, "limit": limit, "offset": offset},
            )
