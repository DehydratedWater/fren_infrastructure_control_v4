"""Topic Analyzer — prepare data for agent analysis and save results."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="prepare-analysis|save-analysis|get-status")
    topic_id: str = Field(default="", description="Topic ID")
    analysis_text: str = Field(default="", description="Agent's analysis text to save")
    new_insights: str = Field(default="", description="JSON array of new insights")
    cumulative_summary: str = Field(default="", description="Updated cumulative summary")
    key_facts: str = Field(default="", description="JSON array of key facts")


class Output(BaseModel):
    success: bool = True
    item: dict = Field(default_factory=dict)
    items: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class TopicAnalyzerTool(ScriptTool[Input, Output]):
    name = "topic_analyzer"
    description = "Prepare data for topic analysis and save analysis results"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        cmd = inp.command

        if cmd == "prepare-analysis":
            return await self._prepare(inp.topic_id)

        if cmd == "save-analysis":
            return await self._save(inp)

        if cmd == "get-status":
            return await self._get_status(inp.topic_id)

        return Output(success=False, error=f"Unknown command: {cmd}")

    async def _prepare(self, topic_id: str) -> Output:
        from app.db.repos.research import (
            ResearchTopicRepo,
            TopicAnalysisRepo,
            TopicKnowledgeRepo,
            YouTubeVideoRepo,
        )

        topic = await ResearchTopicRepo().get(topic_id)
        if not topic:
            return Output(success=False, error=f"Topic not found: {topic_id}")

        # Get existing knowledge
        knowledge = await TopicKnowledgeRepo().get_for_topic(topic_id)

        # Get last analysis to determine "since when"
        latest_analysis = await TopicAnalysisRepo().get_latest_for_topic(topic_id)
        since_id = latest_analysis["analysis_id"] if latest_analysis else None

        # Get new videos with transcripts
        videos = await YouTubeVideoRepo().list_new_for_topic(topic_id, since_analysis_id=since_id)

        # Build structured data for the agent
        video_data: list[dict[str, Any]] = []
        for v in videos:
            transcript_raw = v.get("transcript_raw", {})
            chunks = []
            if isinstance(transcript_raw, dict):
                chunks = transcript_raw.get("chunks", [])
            if not chunks and v.get("transcript"):
                chunks = [v["transcript"]]

            video_data.append(
                {
                    "video_id": v["video_id"],
                    "title": v.get("title", ""),
                    "transcript_chunks": chunks,
                }
            )

        # Get recent website changes
        from app.db.repos.research import WebsiteSnapshotRepo

        website_changes = await WebsiteSnapshotRepo().list_changed_for_topic(topic_id, limit=10)
        website_data = [
            {
                "website_name": ws.get("website_name", ""),
                "url": ws.get("url", ""),
                "diff_summary": ws.get("diff_summary", ""),
                "content_preview": (ws.get("content_text", "") or "")[:2000],
                "snapshot_id": ws["snapshot_id"],
            }
            for ws in website_changes
        ]

        result = {
            "topic_id": topic_id,
            "topic_name": topic["name"],
            "prism": topic.get("prism", ""),
            "description": topic.get("description", ""),
            "criteria": topic.get("criteria", {}),
            "existing_knowledge": {
                "cumulative_summary": knowledge.get("cumulative_summary", "") if knowledge else "",
                "key_facts": knowledge.get("key_facts", []) if knowledge else [],
                "version": knowledge.get("version", 0) if knowledge else 0,
            },
            "new_videos": video_data,
            "video_count": len(video_data),
            "website_changes": website_data,
            "website_change_count": len(website_data),
        }

        return Output(success=True, item=result)

    async def _save(self, inp: Input) -> Output:
        from app.db.repos.research import TopicAnalysisRepo, TopicKnowledgeRepo

        if not inp.topic_id:
            return Output(success=False, error="topic_id required")

        # Parse insights
        insights: list[Any] = []
        if inp.new_insights:
            try:
                insights = json.loads(inp.new_insights)
            except json.JSONDecodeError:
                insights = [inp.new_insights]

        # Save analysis
        aid = f"analysis_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
        analysis = await TopicAnalysisRepo().create(
            aid,
            inp.topic_id,
            analysis_text=inp.analysis_text,
            new_insights=insights,
        )

        # Update knowledge if provided
        knowledge = None
        knowledge_diff = None
        if inp.cumulative_summary:
            key_facts: list[Any] = []
            if inp.key_facts:
                try:
                    key_facts = json.loads(inp.key_facts)
                except json.JSONDecodeError:
                    key_facts = [inp.key_facts]

            # Get old knowledge for diff
            kid = f"know_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            old_knowledge = await TopicKnowledgeRepo().get_or_create(kid, inp.topic_id)
            old_version = old_knowledge.get("version", 0) if old_knowledge else 0
            old_facts = (
                set(json.dumps(f) for f in (old_knowledge.get("key_facts") or []) if old_knowledge)
                if old_knowledge
                else set()
            )

            knowledge = await TopicKnowledgeRepo().update_knowledge(inp.topic_id, inp.cumulative_summary, key_facts)

            # Create knowledge diff
            from app.db.repos.research import KnowledgeDiffRepo

            new_version = knowledge.get("version", old_version + 1) if knowledge else old_version + 1
            new_facts_set = set(json.dumps(f) for f in key_facts)
            added = [json.loads(f) for f in new_facts_set - old_facts]
            removed = [json.loads(f) for f in old_facts - new_facts_set]

            if added or removed or insights:
                did = f"diff_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
                knowledge_diff = await KnowledgeDiffRepo().create(
                    did,
                    inp.topic_id,
                    from_version=old_version,
                    to_version=new_version,
                    new_facts=added,
                    removed_facts=removed,
                    summary=inp.analysis_text[:500] if inp.analysis_text else "",
                    source_type="analysis",
                    source_ids=[aid],
                )

        # Cache the analysis artifact
        try:
            from app.db.repos.context_cache import add_to_cache

            summary_text = (inp.analysis_text or "")[:150]
            await add_to_cache(
                "research_analysis",
                f"Analysis for topic '{inp.topic_id}': {summary_text}",
                entity_type="topic_analyses",
                entity_id=aid,
                tags=["research", "analysis", inp.topic_id],
                source_agent="topic_analyzer",
            )
        except Exception:
            pass

        return Output(
            success=True,
            item={
                "analysis": analysis,
                "knowledge": knowledge,
                "knowledge_diff": knowledge_diff,
            },
        )

    async def _get_status(self, topic_id: str | None) -> Output:
        from app.db.repos.research import (
            ResearchTopicRepo,
            TopicAnalysisRepo,
            TopicKnowledgeRepo,
            YouTubeVideoRepo,
        )

        if topic_id:
            topics = [await ResearchTopicRepo().get(topic_id)]
            topics = [t for t in topics if t]
        else:
            topics = await ResearchTopicRepo().list_active()

        statuses = []
        for topic in topics:
            tid = topic["topic_id"]
            latest = await TopicAnalysisRepo().get_latest_for_topic(tid)
            knowledge = await TopicKnowledgeRepo().get_for_topic(tid)
            pending = await YouTubeVideoRepo().list_new_for_topic(
                tid, since_analysis_id=latest["analysis_id"] if latest else None
            )
            statuses.append(
                {
                    "topic_id": tid,
                    "topic_name": topic["name"],
                    "last_analysis_date": str(latest["date"]) if latest else None,
                    "pending_video_count": len(pending),
                    "knowledge_version": knowledge["version"] if knowledge else 0,
                }
            )

        return Output(success=True, items=statuses, count=len(statuses))


if __name__ == "__main__":
    TopicAnalyzerTool.run()
