"""Investigation domain — ports v3 `investigation/*`.

A single subagent today: investigation/youtube_scout, invoked (via the Task tool)
from the investigator orchestrator to do focused YouTube research. It is not an
orchestrator with its own dispatch chain, so this file exposes only `agents()`.

v3 routed youtube_scout through MODEL_CODER with no per-agent `.model_class`, so
it keeps model_class="default".
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from src import (
    AgentDefinition,
    AgentTest,
    CapabilityTest,
    SubstringEvaluator,
)

_YOUTUBE_SCOUT_PROMPT = """\
# YouTube Scout — Video Research Subagent

You search YouTube for videos, evaluate their relevance to the user's interests,
fetch transcripts for the most promising ones, and return ranked
recommendations.

## Guidelines
- Focus on content matching the user's interests and current research topics.
- Prioritise quality over quantity — deeply evaluate 3-5 videos rather than skim 20.
- Extract specific key points from transcripts, not generic summaries.
- Always explain WHY a video is relevant to the user's specific interests.

## Flow
1. Search YouTube using the provided queries (run several query variations for
   diverse results) and collect the results.
2. Evaluate and rank by relevance, view count / channel credibility, recency, and
   title/description quality; cross-check existing research topics for context.
   Select the top 3-5.
3. Fetch each selected video's transcript and extract its main thesis, 3-5 key
   points, notable quotes/data, and the connection to the user's interests; skip
   and note any failed fetch.
4. Compile a structured report — per video: title + link, why it's relevant, key
   points, and a High/Medium/Low watch priority with reasoning — ordered by
   priority. This report is returned to the investigator via the Task result.
"""


def agents() -> list[AgentDefinition]:
    return [
        define_agent(
            "investigation/youtube_scout",
            short="search, evaluate, and analyze YouTube video transcripts",
            long=(
                "YouTube research subagent. Searches with provided queries, ranks"
                " results by relevance/credibility/recency, fetches transcripts"
                " for the top 3-5, extracts key points, and returns a"
                " priority-ordered recommendation report to the investigator."
            ),
            prompt=_YOUTUBE_SCOUT_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="scout-explains-relevance",
                    description="Recommendations must explain why each video is relevant.",
                    evaluators=(
                        SubstringEvaluator(needle="relevan", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="report-uses-transcripts",
                    prompt="Find the best YouTube videos on local LLM inference.",
                    evaluators=(
                        SubstringEvaluator(needle="transcript", case_sensitive=False),
                    ),
                ),
            ],
        ),
    ]
