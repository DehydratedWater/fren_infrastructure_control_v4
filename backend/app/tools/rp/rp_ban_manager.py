"""RP Ban Manager — dynamic anti-cliche system.

Maintains a list of banned narrative patterns per adventure. Rules can be:
- Auto-detected by AI analyzing recent story entries for repetitive patterns
- Manually added by the user via /ban add <phrase>

Inspired by Megumin Suite V5's AI-powered ban list that:
1. Collects last 50 AI messages, strips formatting
2. Sends to AI as literary critique task
3. Returns generalized rules (not exact phrases)
4. Injects as hard constraints in every subsequent generation
"""

from __future__ import annotations

import asyncio
import json

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="list|analyze|add|remove")
    adventure_id: int = Field(default=0, description="Adventure ID")
    rule: str = Field(default="", description="Ban rule text (add/remove)")
    rule_id: int = Field(default=0, description="Rule ID to remove")
    limit: int = Field(default=20, description="Max rules to return")


class Output(BaseModel):
    success: bool = True
    rules: list[dict] = Field(default_factory=list)
    rule: dict | None = None
    error: str = ""


class RPBanManagerTool(ScriptTool[Input, Output]):
    name = "rp_ban_manager"
    description = "Manage anti-cliche ban rules for RP adventures"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.rp_ban import BanRuleRepo

        repo = BanRuleRepo()
        cmd = inp.command

        if cmd == "list":
            rules = await repo.list_active(inp.adventure_id, limit=inp.limit)
            return Output(success=True, rules=[_serialize_rule(r) for r in rules])

        if cmd == "add":
            if not inp.rule:
                return Output(success=False, error="rule text is required for add")
            rule = await repo.create(inp.adventure_id, inp.rule, source="manual")
            return Output(success=True, rule=_serialize_rule(rule))

        if cmd == "remove":
            if not inp.rule_id:
                return Output(success=False, error="rule_id is required for remove")
            rule = await repo.deactivate(inp.rule_id)
            return Output(success=True, rule=_serialize_rule(rule) if rule else None)

        if cmd == "analyze":
            return await self._analyze(inp.adventure_id, repo)

        return Output(success=False, error=f"unknown command: {cmd}")

    async def _analyze(self, adventure_id: int, repo: object) -> Output:
        """AI-powered cliche detection from recent story entries."""
        from app.db.repos.rp_adventure import StoryLogRepo

        # Get recent narration/dialogue entries
        story_repo = StoryLogRepo()
        narration = await story_repo.get_by_entry_type(adventure_id, "narration", limit=30)
        dialogue = await story_repo.get_by_entry_type(adventure_id, "dialogue", limit=20)

        all_entries = narration + dialogue
        if len(all_entries) < 5:
            return Output(
                success=False,
                error="Not enough story entries to analyze (need at least 5)",
            )

        # Clean and join text
        cleaned = []
        for entry in all_entries:
            text = entry.get("content", "")
            # Strip common RP formatting
            for prefix in ["*Narrator:*", "*Action:*"]:
                text = text.replace(prefix, "")
            text = text.strip()
            if text:
                cleaned.append(text)

        combined = "\n---\n".join(cleaned)

        # Generate analysis prompt
        analysis_prompt = f"""Analyze these RP story entries for repetitive patterns and cliches.
Find the TOP 5 most overused patterns — generalized as rules, NOT exact quotes.

Examples of good rules:
- "Characters releasing breaths they didn't know they were holding"
- "Ending every scene with a dramatic pause"
- "NPCs always understanding the player immediately"
- "Repeated use of 'a shiver ran down their spine'"
- "Describing eyes too frequently (narrowing, widening, flashing)"

Story entries:
{combined}

Return a JSON array of strings, each being a generalized cliche rule.
Example: ["Rule 1", "Rule 2", ...]
Only return the JSON array, nothing else."""

        # Use vLLM for analysis
        try:
            import httpx

            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "http://192.168.0.42:8082/v1/chat/completions",
                    json={
                        "model": "qwen35-27b",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a literary critique expert. Analyze text for repetitive patterns and cliches. Return only a JSON array of generalized rules.",
                            },
                            {"role": "user", "content": analysis_prompt},
                        ],
                        "max_tokens": 1024,
                        "temperature": 0.3,
                    },
                )
                data = resp.json()
                content = data["choices"][0]["message"]["content"]

                # Parse JSON array from response
                # Handle markdown code blocks
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]

                rules_text = json.loads(content.strip())
                if not isinstance(rules_text, list):
                    rules_text = [rules_text]

                # Create ban rules
                created = []
                for rule_text in rules_text:
                    if isinstance(rule_text, str) and len(rule_text) > 10:
                        rule = await repo.create(adventure_id, rule_text, source="auto")
                        created.append(_serialize_rule(rule))

                return Output(success=True, rules=created)

        except Exception as e:
            return Output(success=False, error=f"Analysis failed: {e}")


def _serialize_rule(row: dict) -> dict:
    return {
        "id": row.get("id"),
        "adventure_id": row.get("adventure_id"),
        "rule": row.get("rule"),
        "source": row.get("source"),
        "is_active": row.get("is_active"),
        "created_at": str(row.get("created_at", "")),
    }


if __name__ == "__main__":
    RPBanManagerTool.run()
