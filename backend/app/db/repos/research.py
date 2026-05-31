"""Research tracker repositories — topics, channels, videos, analyses, knowledge."""

from __future__ import annotations

import json
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class ResearchTopicRepo:
    async def create(
        self,
        topic_id: str,
        name: str,
        *,
        prism: str = "",
        status: str = "active",
        description: str = "",
        criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO research_topics (topic_id, name, prism, status, description, criteria)
                VALUES (:topic_id, :name, :prism, :status, :description, CAST(:criteria AS jsonb))
                RETURNING *
                """,
                {
                    "topic_id": topic_id,
                    "name": name,
                    "prism": prism,
                    "status": status,
                    "description": description,
                    "criteria": json.dumps(criteria or {}),
                },
            )  # type: ignore[return-value]

    async def get(self, topic_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM research_topics WHERE topic_id = :tid", {"tid": topic_id})

    async def list_active(self, *, limit: int = 50) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM research_topics WHERE status = 'active' ORDER BY created_at DESC LIMIT :limit",
                {"limit": limit},
            )

    async def list_all(self, *, limit: int = 100) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM research_topics ORDER BY created_at DESC LIMIT :limit",
                {"limit": limit},
            )

    async def update(self, topic_id: str, **fields: Any) -> dict[str, Any] | None:
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {"tid": topic_id}
        idx = 1
        for k, v in fields.items():
            if v is not None:
                pk = f"p{idx}"
                params[pk] = v
                sets.append(f"{k} = :{pk}")
                idx += 1
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"UPDATE research_topics SET {', '.join(sets)} WHERE topic_id = :tid RETURNING *",
                params,
            )

    async def delete(self, topic_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(
                s, "DELETE FROM research_topics WHERE topic_id = :tid RETURNING id", {"tid": topic_id}
            )
            return r.fetchone() is not None

    async def get_with_channels(self, topic_id: str) -> dict[str, Any] | None:
        topic = await self.get(topic_id)
        if not topic:
            return None
        channels = await TopicChannelLinkRepo().get_channels_for_topic(topic_id)
        topic["channels"] = channels
        return topic


class YouTubeChannelRepo:
    async def create(self, channel_id: str, yt_channel_id: str, name: str, **kw: Any) -> dict[str, Any]:
        cols = ["channel_id", "yt_channel_id", "name"]
        vals = [":channel_id", ":yt_channel_id", ":name"]
        params: dict[str, Any] = {"channel_id": channel_id, "yt_channel_id": yt_channel_id, "name": name}
        idx = 4
        for k, v in kw.items():
            if v is not None:
                pk = f"p{idx}"
                cols.append(k)
                vals.append(f":{pk}")
                params[pk] = v
                idx += 1
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"INSERT INTO youtube_channels ({', '.join(cols)}) VALUES ({', '.join(vals)}) RETURNING *",
                params,
            )  # type: ignore[return-value]

    async def get(self, channel_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM youtube_channels WHERE channel_id = :cid", {"cid": channel_id})

    async def get_by_yt_id(self, yt_channel_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s, "SELECT * FROM youtube_channels WHERE yt_channel_id = :ytid", {"ytid": yt_channel_id}
            )

    async def list_all(self, *, limit: int = 100) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM youtube_channels ORDER BY created_at DESC LIMIT :limit",
                {"limit": limit},
            )

    async def mark_fetched(self, channel_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "UPDATE youtube_channels SET last_fetched_at = NOW(), updated_at = NOW() WHERE channel_id = :cid RETURNING *",
                {"cid": channel_id},
            )


class TopicChannelLinkRepo:
    async def link(self, topic_id: str, channel_id: str) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO topic_channel_links (topic_id, channel_id)
                VALUES (:tid, :cid)
                ON CONFLICT (topic_id, channel_id) DO NOTHING
                RETURNING *
                """,
                {"tid": topic_id, "cid": channel_id},
            )  # type: ignore[return-value]

    async def unlink(self, topic_id: str, channel_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(
                s,
                "DELETE FROM topic_channel_links WHERE topic_id = :tid AND channel_id = :cid RETURNING id",
                {"tid": topic_id, "cid": channel_id},
            )
            return r.fetchone() is not None

    async def get_channels_for_topic(self, topic_id: str) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT yc.* FROM youtube_channels yc
                JOIN topic_channel_links tcl ON tcl.channel_id = yc.channel_id
                WHERE tcl.topic_id = :tid
                ORDER BY yc.name
                """,
                {"tid": topic_id},
            )

    async def get_all_active_channel_ids(self) -> list[str]:
        async with get_async_session() as s:
            rows = await fetch_all(
                s,
                """
                SELECT DISTINCT tcl.channel_id FROM topic_channel_links tcl
                JOIN youtube_channels yc ON yc.channel_id = tcl.channel_id
                WHERE yc.status = 'active'
                """,
            )
            return [r["channel_id"] for r in rows]


class YouTubeVideoRepo:
    async def create(
        self,
        video_id: str,
        yt_video_id: str,
        channel_id: str,
        *,
        title: str = "",
        raw_api_response: Any = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "video_id": video_id,
            "yt_video_id": yt_video_id,
            "channel_id": channel_id,
            "title": title,
            "raw": json.dumps(raw_api_response or {}),
        }
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO youtube_videos (video_id, yt_video_id, channel_id, title, raw_api_response)
                VALUES (:video_id, :yt_video_id, :channel_id, :title, CAST(:raw AS jsonb))
                RETURNING *
                """,
                params,
            )  # type: ignore[return-value]

    async def create_standalone(
        self,
        video_id: str,
        yt_video_id: str,
        *,
        title: str = "",
    ) -> dict[str, Any]:
        """Insert a video with channel_id=NULL (user-shared, not from a tracked channel)."""
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO youtube_videos (video_id, yt_video_id, channel_id, title, raw_api_response)
                VALUES (:video_id, :yt_video_id, NULL, :title, CAST(:raw AS jsonb))
                RETURNING *
                """,
                {"video_id": video_id, "yt_video_id": yt_video_id, "title": title, "raw": "{}"},
            )  # type: ignore[return-value]

    async def get(self, video_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM youtube_videos WHERE video_id = :vid", {"vid": video_id})

    async def get_by_yt_id(self, yt_video_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM youtube_videos WHERE yt_video_id = :ytid", {"ytid": yt_video_id})

    async def exists_by_yt_id(self, yt_video_id: str) -> bool:
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                "SELECT 1 AS found FROM youtube_videos WHERE yt_video_id = :ytid LIMIT 1",
                {"ytid": yt_video_id},
            )
            return row is not None

    async def update_transcript(
        self,
        video_id: str,
        transcript: str,
        transcript_raw: Any = None,
        transcript_status: str = "done",
    ) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE youtube_videos
                SET transcript = :txt, transcript_raw = CAST(:raw AS jsonb),
                    transcript_status = :status, updated_at = NOW()
                WHERE video_id = :vid RETURNING *
                """,
                {
                    "vid": video_id,
                    "txt": transcript,
                    "raw": json.dumps(transcript_raw or []),
                    "status": transcript_status,
                },
            )

    async def list_pending_transcripts(self, *, limit: int = 50) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM youtube_videos
                WHERE transcript_status = 'pending'
                ORDER BY created_at ASC LIMIT :limit
                """,
                {"limit": limit},
            )

    async def list_pending_for_channel(self, channel_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        """List videos for a specific channel that still need transcripts."""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM youtube_videos
                WHERE channel_id = :cid AND transcript_status = 'pending'
                ORDER BY created_at ASC LIMIT :limit
                """,
                {"cid": channel_id, "limit": limit},
            )

    async def list_for_channel(self, channel_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM youtube_videos WHERE channel_id = :cid ORDER BY created_at DESC LIMIT :limit",
                {"cid": channel_id, "limit": limit},
            )

    async def list_for_channel_summary(self, channel_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        """List videos for a channel — metadata only, no transcript or raw_api_response."""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT video_id, yt_video_id, channel_id, title, transcript_status, created_at
                FROM youtube_videos WHERE channel_id = :cid
                ORDER BY created_at DESC LIMIT :limit
                """,
                {"cid": channel_id, "limit": limit},
            )

    async def list_recent_summary(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """List all recent videos — metadata only, no transcript or raw_api_response."""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT video_id, yt_video_id, channel_id, title, transcript_status, created_at
                FROM youtube_videos
                ORDER BY created_at DESC LIMIT :limit
                """,
                {"limit": limit},
            )

    async def list_new_for_topic(self, topic_id: str, *, since_analysis_id: str | None = None) -> list[dict[str, Any]]:
        """List videos for a topic's channels that haven't been analyzed yet."""
        if since_analysis_id:
            async with get_async_session() as s:
                analysis = await fetch_one(
                    s,
                    "SELECT created_at FROM topic_analyses WHERE analysis_id = :aid",
                    {"aid": since_analysis_id},
                )
                if analysis:
                    return await fetch_all(
                        s,
                        """
                        SELECT yv.* FROM youtube_videos yv
                        JOIN topic_channel_links tcl ON tcl.channel_id = yv.channel_id
                        WHERE tcl.topic_id = :tid
                          AND yv.transcript_status = 'done'
                          AND yv.created_at > :since
                        ORDER BY yv.created_at ASC
                        """,
                        {"tid": topic_id, "since": analysis["created_at"]},
                    )
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT yv.* FROM youtube_videos yv
                JOIN topic_channel_links tcl ON tcl.channel_id = yv.channel_id
                WHERE tcl.topic_id = :tid AND yv.transcript_status = 'done'
                ORDER BY yv.created_at ASC
                """,
                {"tid": topic_id},
            )


class TopicAnalysisRepo:
    async def create(
        self,
        analysis_id: str,
        topic_id: str,
        *,
        video_ids: list[str] | None = None,
        analysis_text: str = "",
        new_insights: list[Any] | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO topic_analyses (analysis_id, topic_id, video_ids, analysis_text, new_insights)
                VALUES (:aid, :tid, CAST(:vids AS jsonb), :txt, CAST(:ins AS jsonb))
                RETURNING *
                """,
                {
                    "aid": analysis_id,
                    "tid": topic_id,
                    "vids": json.dumps(video_ids or []),
                    "txt": analysis_text,
                    "ins": json.dumps(new_insights or []),
                },
            )  # type: ignore[return-value]

    async def list_for_topic(self, topic_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM topic_analyses WHERE topic_id = :tid ORDER BY created_at DESC LIMIT :limit",
                {"tid": topic_id, "limit": limit},
            )

    async def get_latest_for_topic(self, topic_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM topic_analyses WHERE topic_id = :tid ORDER BY created_at DESC LIMIT 1",
                {"tid": topic_id},
            )


class TopicKnowledgeRepo:
    async def get_or_create(self, knowledge_id: str, topic_id: str) -> dict[str, Any]:
        async with get_async_session() as s:
            row = await fetch_one(s, "SELECT * FROM topic_knowledge WHERE topic_id = :tid", {"tid": topic_id})
            if row:
                return row
            return await fetch_one(
                s,
                """
                INSERT INTO topic_knowledge (knowledge_id, topic_id)
                VALUES (:kid, :tid)
                ON CONFLICT (topic_id) DO UPDATE SET updated_at = NOW()
                RETURNING *
                """,
                {"kid": knowledge_id, "tid": topic_id},
            )  # type: ignore[return-value]

    async def get_for_topic(self, topic_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM topic_knowledge WHERE topic_id = :tid", {"tid": topic_id})

    async def update_knowledge(
        self,
        topic_id: str,
        cumulative_summary: str,
        key_facts: list[Any] | None = None,
    ) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE topic_knowledge
                SET cumulative_summary = :summary,
                    key_facts = CAST(:facts AS jsonb),
                    version = version + 1,
                    updated_at = NOW()
                WHERE topic_id = :tid RETURNING *
                """,
                {
                    "tid": topic_id,
                    "summary": cumulative_summary,
                    "facts": json.dumps(key_facts or []),
                },
            )


class TopicWebsiteRepo:
    async def create(
        self,
        website_id: str,
        topic_id: str,
        url: str,
        *,
        name: str = "",
        scrape_selector: str = "",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO topic_websites (website_id, topic_id, url, name, scrape_selector)
                VALUES (:wid, :tid, :url, :name, :sel) RETURNING *
                """,
                {"wid": website_id, "tid": topic_id, "url": url, "name": name, "sel": scrape_selector},
            )  # type: ignore[return-value]

    async def get(self, website_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM topic_websites WHERE website_id = :wid", {"wid": website_id})

    async def list_for_topic(self, topic_id: str) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM topic_websites WHERE topic_id = :tid AND status = 'active' ORDER BY name",
                {"tid": topic_id},
            )

    async def list_active(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT tw.*, rt.name AS topic_name FROM topic_websites tw
                JOIN research_topics rt ON rt.topic_id = tw.topic_id
                WHERE tw.status = 'active' AND rt.status = 'active'
                ORDER BY tw.topic_id, tw.name
                """,
            )

    async def mark_checked(self, website_id: str, content_hash: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE topic_websites SET last_checked_at = NOW(), last_content_hash = :hash, updated_at = NOW()
                WHERE website_id = :wid RETURNING *
                """,
                {"wid": website_id, "hash": content_hash},
            )

    async def delete(self, website_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(
                s, "DELETE FROM topic_websites WHERE website_id = :wid RETURNING id", {"wid": website_id}
            )
            return r.fetchone() is not None


class WebsiteSnapshotRepo:
    async def create(
        self,
        snapshot_id: str,
        website_id: str,
        *,
        content_text: str = "",
        content_hash: str = "",
        has_changes: bool = False,
        diff_summary: str = "",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO website_snapshots (snapshot_id, website_id, content_text, content_hash, has_changes, diff_summary)
                VALUES (:sid, :wid, :txt, :hash, :changes, :diff) RETURNING *
                """,
                {
                    "sid": snapshot_id,
                    "wid": website_id,
                    "txt": content_text,
                    "hash": content_hash,
                    "changes": has_changes,
                    "diff": diff_summary,
                },
            )  # type: ignore[return-value]

    async def get_latest(self, website_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM website_snapshots WHERE website_id = :wid ORDER BY created_at DESC LIMIT 1",
                {"wid": website_id},
            )

    async def list_changed_for_topic(self, topic_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT ws.*, tw.name AS website_name, tw.url FROM website_snapshots ws
                JOIN topic_websites tw ON tw.website_id = ws.website_id
                WHERE tw.topic_id = :tid AND ws.has_changes = TRUE
                ORDER BY ws.created_at DESC LIMIT :limit
                """,
                {"tid": topic_id, "limit": limit},
            )


class TopicSearchQueryRepo:
    async def create(self, query_id: str, topic_id: str, query: str) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO topic_search_queries (query_id, topic_id, query)
                VALUES (:qid, :tid, :query) RETURNING *
                """,
                {"qid": query_id, "tid": topic_id, "query": query},
            )  # type: ignore[return-value]

    async def get(self, query_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM topic_search_queries WHERE query_id = :qid", {"qid": query_id})

    async def list_for_topic(self, topic_id: str) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM topic_search_queries WHERE topic_id = :tid AND status = 'active' ORDER BY created_at",
                {"tid": topic_id},
            )

    async def list_active(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT tsq.*, rt.name AS topic_name FROM topic_search_queries tsq
                JOIN research_topics rt ON rt.topic_id = tsq.topic_id
                WHERE tsq.status = 'active' AND rt.status = 'active'
                ORDER BY tsq.topic_id
                """,
            )

    async def mark_run(self, query_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "UPDATE topic_search_queries SET last_run_at = NOW() WHERE query_id = :qid RETURNING *",
                {"qid": query_id},
            )

    async def delete(self, query_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(
                s, "DELETE FROM topic_search_queries WHERE query_id = :qid RETURNING id", {"qid": query_id}
            )
            return r.fetchone() is not None


class KnowledgeDiffRepo:
    async def create(
        self,
        diff_id: str,
        topic_id: str,
        *,
        from_version: int = 0,
        to_version: int = 0,
        new_facts: list[Any] | None = None,
        removed_facts: list[Any] | None = None,
        changed_facts: list[Any] | None = None,
        summary: str = "",
        source_type: str = "",
        source_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO knowledge_diffs
                    (diff_id, topic_id, from_version, to_version, new_facts, removed_facts, changed_facts,
                     summary, source_type, source_ids)
                VALUES (:did, :tid, :fv, :tv, CAST(:nf AS jsonb), CAST(:rf AS jsonb), CAST(:cf AS jsonb),
                        :summary, :st, CAST(:sids AS jsonb))
                RETURNING *
                """,
                {
                    "did": diff_id,
                    "tid": topic_id,
                    "fv": from_version,
                    "tv": to_version,
                    "nf": json.dumps(new_facts or []),
                    "rf": json.dumps(removed_facts or []),
                    "cf": json.dumps(changed_facts or []),
                    "summary": summary,
                    "st": source_type,
                    "sids": json.dumps(source_ids or []),
                },
            )  # type: ignore[return-value]

    async def list_for_topic(self, topic_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM knowledge_diffs WHERE topic_id = :tid ORDER BY created_at DESC LIMIT :limit",
                {"tid": topic_id, "limit": limit},
            )

    async def get_latest(self, topic_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM knowledge_diffs WHERE topic_id = :tid ORDER BY created_at DESC LIMIT 1",
                {"tid": topic_id},
            )
