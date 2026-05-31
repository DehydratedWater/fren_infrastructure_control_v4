"""Pose Selector — select character poses based on emotional criteria."""

from __future__ import annotations

import json
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field

EMOTION_KEYWORDS = {
    "happiness": ["happy", "joy", "joyful", "cheerful", "pleased", "delighted"],
    "confidence": ["confident", "assured", "certain", "bold"],
    "curiosity": ["curious", "interested", "intrigued", "wondering"],
    "anger": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated"],
    "surprise": ["surprised", "shocked", "amazed", "astonished"],
    "sadness": ["sad", "unhappy", "disappointed", "melancholy"],
    "embarrassment": ["embarrassed", "awkward", "shy", "nervous", "flustered"],
}


def _parse_level(value: str) -> tuple[str, float]:
    v = value.lower().strip()
    if v in ("high", "h"):
        return ("high", 0.8)
    if v in ("moderate", "med", "m", "medium"):
        return ("moderate", 0.5)
    if v in ("low", "l"):
        return ("low", 0.2)
    try:
        return ("exact", float(v))
    except ValueError:
        return ("high", 0.8)


class Input(BaseModel):
    query: str = Field(default="", description="Natural language pose description")
    happiness: str = Field(default="", description="Happiness level")
    confidence: str = Field(default="", description="Confidence level")
    curiosity: str = Field(default="", description="Curiosity level")
    anger: str = Field(default="", description="Anger level")
    surprise: str = Field(default="", description="Surprise level")
    sadness: str = Field(default="", description="Sadness level")
    embarrassment: str = Field(default="", description="Embarrassment level")
    top: int = Field(default=5, description="Number of results")
    poses_file: str = Field(default="data/pose_indexes.json", description="Path to poses JSON")


class Output(BaseModel):
    success: bool = True
    poses: list[dict] = Field(default_factory=list)
    recommended: str = ""
    error: str = ""


class PoseSelectorTool(ScriptTool[Input, Output]):
    name = "select_pose"
    description = "Select character poses based on emotional criteria"

    def execute(self, inp: Input) -> Output:
        poses_path = Path(inp.poses_file)
        if not poses_path.exists():
            return Output(success=False, error=f"Poses file not found: {poses_path}")

        poses = json.loads(poses_path.read_text())

        # Build criteria
        if inp.query:
            criteria, expressions = self._nl_to_criteria(inp.query)
        else:
            criteria = {}
            expressions: list[str] = []
            for emotion in ("happiness", "confidence", "curiosity", "anger", "surprise", "sadness", "embarrassment"):
                val = getattr(inp, emotion)
                if val:
                    criteria[emotion] = val

        if not criteria and not expressions:
            criteria = {"curiosity": "moderate", "confidence": "moderate"}

        # Score and rank
        scored = []
        for pose in poses:
            score = self._score_emotion(pose, criteria) * 1.5 + self._score_expression(pose, expressions)
            scored.append((pose, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: inp.top]

        result_poses = []
        for pose, score in top:
            result_poses.append(
                {
                    "pose_id": pose.get("pose_id", ""),
                    "file_name": pose.get("file_name", ""),
                    "short_description": pose.get("short_description", ""),
                    "score": round(score, 2),
                    "emotional_tags": pose.get("emotional_tags", {}),
                }
            )

        recommended = result_poses[0]["file_name"] if result_poses else ""
        return Output(success=True, poses=result_poses, recommended=recommended)

    def _nl_to_criteria(self, query: str) -> tuple[dict[str, str], list[str]]:
        q = query.lower()
        criteria: dict[str, str] = {}
        expressions: list[str] = []
        for emotion, keywords in EMOTION_KEYWORDS.items():
            for kw in keywords:
                if kw in q:
                    criteria[emotion] = "high"
                    break
        expr_kws = [
            "playful",
            "mischievous",
            "serious",
            "stern",
            "skeptical",
            "confident",
            "joyful",
            "nervous",
            "shocked",
            "bored",
            "welcoming",
            "peaceful",
            "focused",
            "thoughtful",
            "smug",
            "sassy",
        ]
        for e in expr_kws:
            if e in q:
                expressions.append(e)
        return criteria, expressions

    def _score_emotion(self, pose: dict, criteria: dict[str, str]) -> float:
        score = 0.0
        tags = pose.get("emotional_tags", {})
        for emotion, level_str in criteria.items():
            if emotion not in tags:
                continue
            val = tags[emotion]
            mode, target = _parse_level(level_str)
            if mode == "high":
                score += val if val >= 0.7 else val * 0.5 if val >= 0.5 else 0
            elif mode == "moderate":
                score += 1.0 - abs(val - 0.5)
            elif mode == "low":
                score += (1.0 - val) if val <= 0.3 else (1.0 - val) * 0.5 if val <= 0.5 else 0
            elif mode == "exact":
                score += 1.0 - abs(val - target)
        return score

    def _score_expression(self, pose: dict, expressions: list[str]) -> float:
        score = 0.0
        expr_tags = pose.get("expression_tags", [])
        pose_exprs = {t["tag"].lower(): t["value"] for t in expr_tags if isinstance(t, dict)}
        for expr in expressions:
            el = expr.lower()
            if el in pose_exprs:
                score += pose_exprs[el]
            else:
                for pe, v in pose_exprs.items():
                    if el in pe or pe in el:
                        score += v * 0.5
        return score


if __name__ == "__main__":
    PoseSelectorTool.run()
