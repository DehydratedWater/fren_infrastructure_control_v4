"""Research Manager — CRUD for topics, channels, links, videos, analyses, knowledge."""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="Topic: create-topic|get-topic|list-topics|update-topic|delete-topic|get-topic-full; "
        "Channel: add-channel|get-channel|list-channels; "
        "Link: link-channel|unlink-channel|get-topic-channels; "
        "Video: list-videos|get-video|list-pending-transcripts; "
        "Analysis: list-analyses; Knowledge: get-knowledge; "
        "Website: add-website|list-websites|remove-website; "
        "Search: add-search-query|list-search-queries|remove-search-query"
    )
    # IDs
    topic_id: str = Field(default="", description="Topic ID")
    channel_id: str = Field(default="", description="Channel ID")
    video_id: str = Field(default="", description="Video ID")
    website_id: str = Field(default="", description="Website ID")
    query_id: str = Field(default="", description="Search query ID")
    # Topic fields
    name: str = Field(default="", description="Name for topic or channel")
    prism: str = Field(default="", description="LLM analysis instructions for topic")
    status: str = Field(default="", description="Status (active/paused/archived)")
    description: str = Field(default="", description="Description for topic")
    criteria: str = Field(default="", description="JSON criteria object for topic")
    # Channel fields
    yt_channel_id: str = Field(default="", description="YouTube channel ID (e.g. UC...)")
    # Website fields
    url: str = Field(default="", description="Website URL to monitor")
    scrape_selector: str = Field(default="", description="CSS selector for scraping")
    # Search query fields
    query: str = Field(default="", description="Search query text")


class Output(BaseModel):
    success: bool = True
    item: dict = Field(default_factory=dict)
    items: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class ResearchManagerTool(ScriptTool[Input, Output]):
    name = "research_manager"
    description = "Manage research topics, YouTube channels, and analysis data"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.research import (
            ResearchTopicRepo,
            TopicAnalysisRepo,
            TopicChannelLinkRepo,
            TopicKnowledgeRepo,
            YouTubeChannelRepo,
            YouTubeVideoRepo,
        )

        cmd = inp.command

        # ── Topics ──
        if cmd == "create-topic":
            repo = ResearchTopicRepo()
            tid = f"topic_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            criteria = {}
            if inp.criteria:
                with contextlib.suppress(json.JSONDecodeError):
                    criteria = json.loads(inp.criteria)
            t = await repo.create(tid, inp.name, prism=inp.prism, description=inp.description, criteria=criteria)
            return Output(success=True, item=t)

        if cmd == "get-topic":
            t = await ResearchTopicRepo().get_with_channels(inp.topic_id)
            return Output(success=bool(t), item=t or {}, error="" if t else "Topic not found")

        if cmd == "list-topics":
            ts = await ResearchTopicRepo().list_all()
            return Output(success=True, items=ts, count=len(ts))

        if cmd == "update-topic":
            fields = {}
            if inp.name:
                fields["name"] = inp.name
            if inp.prism:
                fields["prism"] = inp.prism
            if inp.status:
                fields["status"] = inp.status
            if inp.description:
                fields["description"] = inp.description
            if inp.criteria:
                with contextlib.suppress(json.JSONDecodeError):
                    fields["criteria"] = json.dumps(json.loads(inp.criteria))
            t = await ResearchTopicRepo().update(inp.topic_id, **fields)
            return Output(success=bool(t), item=t or {}, error="" if t else "Topic not found")

        if cmd == "delete-topic":
            ok = await ResearchTopicRepo().delete(inp.topic_id)
            return Output(success=ok)

        # ── Channels ──
        if cmd == "add-channel":
            repo = YouTubeChannelRepo()
            cid = f"chan_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            c = await repo.create(cid, inp.yt_channel_id, inp.name)
            return Output(success=True, item=c)

        if cmd == "get-channel":
            c = await YouTubeChannelRepo().get(inp.channel_id)
            return Output(success=bool(c), item=c or {}, error="" if c else "Channel not found")

        if cmd == "list-channels":
            cs = await YouTubeChannelRepo().list_all()
            return Output(success=True, items=cs, count=len(cs))

        # ── Links ──
        if cmd == "link-channel":
            link = await TopicChannelLinkRepo().link(inp.topic_id, inp.channel_id)
            return Output(success=True, item=link or {})

        if cmd == "unlink-channel":
            ok = await TopicChannelLinkRepo().unlink(inp.topic_id, inp.channel_id)
            return Output(success=ok)

        if cmd == "get-topic-channels":
            cs = await TopicChannelLinkRepo().get_channels_for_topic(inp.topic_id)
            return Output(success=True, items=cs, count=len(cs))

        # ── Videos ──
        if cmd == "list-videos":
            repo = YouTubeVideoRepo()
            if inp.channel_id:
                vs = await repo.list_for_channel_summary(inp.channel_id)
            else:
                vs = await repo.list_recent_summary()
            return Output(success=True, items=vs, count=len(vs))

        if cmd == "get-video":
            v = await YouTubeVideoRepo().get(inp.video_id)
            return Output(success=bool(v), item=v or {}, error="" if v else "Video not found")

        if cmd == "list-pending-transcripts":
            vs = await YouTubeVideoRepo().list_pending_transcripts()
            return Output(success=True, items=vs, count=len(vs))

        # ── Analyses ──
        if cmd == "list-analyses":
            analyses = await TopicAnalysisRepo().list_for_topic(inp.topic_id)
            return Output(success=True, items=analyses, count=len(analyses))

        # ── Knowledge ──
        if cmd == "get-knowledge":
            k = await TopicKnowledgeRepo().get_for_topic(inp.topic_id)
            return Output(success=bool(k), item=k or {}, error="" if k else "No knowledge yet")

        # ── Websites ──
        from app.db.repos.research import TopicWebsiteRepo

        if cmd == "add-website":
            repo = TopicWebsiteRepo()
            wid = f"web_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            w = await repo.create(wid, inp.topic_id, inp.url, name=inp.name, scrape_selector=inp.scrape_selector)
            return Output(success=True, item=w)

        if cmd == "list-websites":
            if inp.topic_id:
                ws = await TopicWebsiteRepo().list_for_topic(inp.topic_id)
            else:
                ws = await TopicWebsiteRepo().list_active()
            return Output(success=True, items=ws, count=len(ws))

        if cmd == "remove-website":
            ok = await TopicWebsiteRepo().delete(inp.website_id)
            return Output(success=ok, error="" if ok else "Website not found")

        # ── Search Queries ──
        from app.db.repos.research import TopicSearchQueryRepo

        if cmd == "add-search-query":
            repo = TopicSearchQueryRepo()
            qid = f"sq_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            q = await repo.create(qid, inp.topic_id, inp.query)
            return Output(success=True, item=q)

        if cmd == "list-search-queries":
            if inp.topic_id:
                qs = await TopicSearchQueryRepo().list_for_topic(inp.topic_id)
            else:
                qs = await TopicSearchQueryRepo().list_active()
            return Output(success=True, items=qs, count=len(qs))

        if cmd == "remove-search-query":
            ok = await TopicSearchQueryRepo().delete(inp.query_id)
            return Output(success=ok, error="" if ok else "Query not found")

        # ── Full Topic View ──
        if cmd == "get-topic-full":
            from app.db.repos.research import KnowledgeDiffRepo, WebsiteSnapshotRepo

            topic = await ResearchTopicRepo().get_with_channels(inp.topic_id)
            if not topic:
                return Output(success=False, error="Topic not found")
            topic["websites"] = await TopicWebsiteRepo().list_for_topic(inp.topic_id)
            topic["search_queries"] = await TopicSearchQueryRepo().list_for_topic(inp.topic_id)
            knowledge = await TopicKnowledgeRepo().get_for_topic(inp.topic_id)
            topic["knowledge"] = knowledge or {}
            topic["recent_diffs"] = await KnowledgeDiffRepo().list_for_topic(inp.topic_id, limit=5)
            topic["recent_changes"] = await WebsiteSnapshotRepo().list_changed_for_topic(inp.topic_id, limit=5)
            return Output(success=True, item=topic)

        return Output(success=False, error=f"Unknown command: {cmd}")


if __name__ == "__main__":
    ResearchManagerTool.run()
