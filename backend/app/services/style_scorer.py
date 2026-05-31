"""Rule-based text scorer — polishes Twily's outgoing messages before send.

Enforces emoji budget, strips banned phrases, removes tildes in ironic modes,
caps stage directions, logs repetition + sycophancy markers. Deterministic
regex only — no LLM, no network. Called from src/fren/tools/telegram/send_message.py
right before Bot.send_message().
"""

from __future__ import annotations

import re
from typing import Any

from app.services.persona_palettes import (
    BANNED_PHRASES,
    SYCOPHANCY_MARKERS,
    emoji_budget_for_weights,
    tildes_banned_for_weights,
)

# Unicode emoji ranges — simple heuristic catches the vast majority of emojis
# without pulling in the `emoji` package. Covers dingbats, misc symbols,
# transport/map, supplemental symbols, emoticons, misc symbols & pictographs.
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f6ff"  # misc symbols + pictographs, transport, etc.
    "\U0001f900-\U0001f9ff"  # supplemental symbols + pictographs
    "\U00002600-\U000027bf"  # misc symbols + dingbats (incl. ✨ ☀ ♥)
    "\U0001fa70-\U0001faff"  # symbols + pictographs extended-A
    "\U0001f000-\U0001f02f"  # mahjong etc
    "\u2700-\u27bf"
    "\U0001f600-\U0001f64f"  # emoticons
    "]"
)

# Stage direction: *single-star text*. Excludes markdown bold (`**bold**`) and
# list markers (`* `) via lookbehind/lookahead on the `*` char.
_STAGE_DIR_RE = re.compile(r"(?<!\*)\*(?!\*)(?!\s)[^*\n]{1,80}?(?<!\*)(?<!\s)\*(?!\*)")

# Heuristics for detecting workflow-generated markdown reports. Reports are
# already-strict structured info (priority reviews, daily briefings) and
# should bypass conversational enforcement entirely — rewriting them
# destroys the document. ANY ONE of these signals marks a message as a report.
_MARKDOWN_HEADER_RE = re.compile(r"^\s*#{2,6}\s+\S", re.MULTILINE)
_MARKDOWN_TABLE_RE = re.compile(r"\|[\s:-]{3,}\|")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*[^*\n]+?\*\*")
# Bullets: markdown `-`/`*`, plus common Unicode bullets (⦁ ▪ • ● ◦ ‣).
_MARKDOWN_LIST_RE = re.compile(r"^\s*(?:[-*]|[\u2022\u25AA\u25CF\u25E6\u2023\u2981])\s+\S", re.MULTILINE)
# ALL-CAPS section headers like "HIGH PRIORITY (8):" or "WHAT WORKED:".
_ALLCAPS_HEADER_RE = re.compile(r"^[A-Z][A-Z &/()\d-]{3,60}:\s*$", re.MULTILINE)
_LONG_REPORT_CHARS = 800


def _is_structured_report(text: str) -> bool:
    """Return True if this looks like a workflow-generated markdown report.

    Reports bypass banned phrases / emoji cap / tilde rule / stage direction
    cap. Hearts are still normalized to 💜. Any ONE of these signals is enough:

    - 2+ markdown H2+ headers (## or deeper)
    - markdown table separator (|---|)
    - 5+ **bold** spans
    - 3+ list-item lines AND >= 400 chars
    - >= 800 chars AND (any **bold** OR any header)
    """
    if len(_MARKDOWN_HEADER_RE.findall(text)) >= 2:
        return True
    if _MARKDOWN_TABLE_RE.search(text):
        return True
    # ALL-CAPS section headers (2+ of them) → report format.
    if len(_ALLCAPS_HEADER_RE.findall(text)) >= 2:
        return True
    bold_count = len(_MARKDOWN_BOLD_RE.findall(text))
    if bold_count >= 5:
        return True
    # Bullet list (markdown OR unicode bullets) — 5+ items at any length,
    # or 3+ items at >=400 chars.
    list_items = len(_MARKDOWN_LIST_RE.findall(text))
    if list_items >= 5:
        return True
    if list_items >= 3 and len(text) >= 400:
        return True
    return bool(
        len(text) >= _LONG_REPORT_CHARS and (bold_count >= 1 or _MARKDOWN_HEADER_RE.search(text) or list_items >= 1)
    )


# Heart variants that should always be rewritten to 💜 (Twily's signature).
_HEART_REWRITES = {
    "\u2764\ufe0f": "\U0001f49c",  # ❤️ red heart → 💜
    "\u2764": "\U0001f49c",  # ❤ red heart (no VS16) → 💜
    "\U0001f496": "\U0001f49c",  # 💖 sparkling heart → 💜
    "\U0001f495": "\U0001f49c",  # 💕 two hearts → 💜
    "\U0001f493": "\U0001f49c",  # 💓 beating heart → 💜
    "\U0001f497": "\U0001f49c",  # 💗 growing heart → 💜
    "\U0001f9e1": "\U0001f49c",  # 🧡 orange heart → 💜
    "\U0001f49b": "\U0001f49c",  # 💛 yellow heart → 💜
    "\U0001f49a": "\U0001f49c",  # 💚 green heart → 💜
    "\U0001f499": "\U0001f49c",  # 💙 blue heart → 💜
    "\U0001f90d": "\U0001f49c",  # 🤍 white heart → 💜
    "\U0001f5a4": "\U0001f49c",  # 🖤 black heart → 💜
    "\U0001f90e": "\U0001f49c",  # 🤎 brown heart → 💜
}


def _count_emojis(text: str) -> int:
    return len(_EMOJI_RE.findall(text))


def _normalize_hearts(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Rewrite non-purple hearts to 💜 (Twily's signature)."""
    found: list[str] = []
    for wrong, right in _HEART_REWRITES.items():
        if wrong in text:
            found.append(wrong)
            text = text.replace(wrong, right)
    if not found:
        return text, []
    return text, [
        {
            "violation_type": "wrong_heart",
            "details": f"rewrote {''.join(found)} → 💜",
            "enforced": True,
        }
    ]


def _strip_banned_phrases(text: str) -> tuple[str, list[dict[str, Any]]]:
    violations: list[dict[str, Any]] = []
    for pattern, replacement in BANNED_PHRASES:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            for m in matches:
                violations.append(
                    {
                        "violation_type": "banned_phrase",
                        "details": m.group(0)[:200],
                        "enforced": True,
                    }
                )
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text, violations


def _log_emoji_excess(text: str, budget: int) -> tuple[str, list[dict[str, Any]]]:
    """Count emojis vs. budget — log only, never rewrite.

    Hard capping was breaking message formatting (emojis were sometimes load-
    bearing for tone, and the strip+whitespace-collapse pipeline left mangled
    prose behind). The system prompt now tells the model the soft budget and
    we trust it to self-regulate. We still emit a non-enforced violation so
    telemetry can show drift.
    """
    count = _count_emojis(text)
    if count <= budget:
        return text, []
    emojis_found = _EMOJI_RE.findall(text)
    return text, [
        {
            "violation_type": "emoji_over_budget",
            "details": f"had {count}, budget {budget} (logged, not enforced): "
            + "".join(emojis_found[-(count - budget) :])[:80],
            "enforced": False,
        }
    ]


def _strip_tildes(text: str, banned: bool) -> tuple[str, list[dict[str, Any]]]:
    count = text.count("~")
    if count == 0:
        return text, []
    if banned:
        return text.replace("~", ""), [
            {
                "violation_type": "tilde_abuse",
                "details": f"stripped {count} tildes (ironic mode)",
                "enforced": True,
            }
        ]
    if count > 1:
        # Keep max 1 tilde even in warm mode.
        return re.sub(r"~{1,}", lambda m: "~" if m.start() == text.rfind("~") else "", text, count=count - 1), [
            {
                "violation_type": "tilde_abuse",
                "details": f"warm mode, capped {count} → 1",
                "enforced": True,
            }
        ]
    return text, []


def _cap_stage_directions(text: str) -> tuple[str, list[dict[str, Any]]]:
    matches = _STAGE_DIR_RE.findall(text)
    if len(matches) <= 1:
        return text, []
    # Keep only the first stage direction, strip the rest.
    kept_one = False

    def _sub(m: re.Match[str]) -> str:
        nonlocal kept_one
        if not kept_one:
            kept_one = True
            return m.group(0)
        return ""

    rewritten = _STAGE_DIR_RE.sub(_sub, text)
    return rewritten, [
        {
            "violation_type": "stage_direction_over",
            "details": f"had {len(matches)} directions, kept 1",
            "enforced": True,
        }
    ]


def _detect_repetition(text: str, recent: list[str]) -> list[dict[str, Any]]:
    """Log if a 4-word phrase from current msg appears in any of the last 5 Twily msgs."""
    if not recent:
        return []
    words = re.findall(r"\w+", text.lower())
    if len(words) < 4:
        return []
    current_4grams = {tuple(words[i : i + 4]) for i in range(len(words) - 3)}
    for old in recent[:5]:
        old_words = re.findall(r"\w+", old.lower())
        if len(old_words) < 4:
            continue
        old_4grams = {tuple(old_words[i : i + 4]) for i in range(len(old_words) - 3)}
        overlap = current_4grams & old_4grams
        if overlap:
            # Filter out common function-word 4-grams.
            significant = [
                " ".join(g)
                for g in overlap
                if any(len(w) > 4 for w in g)
                and not all(w in {"the", "and", "you", "that", "this", "with", "have"} for w in g)
            ]
            if significant:
                return [
                    {
                        "violation_type": "repetition",
                        "details": f"phrase repeated: '{significant[0]}'",
                        "enforced": False,
                    }
                ]
    return []


def _detect_sycophancy(text: str) -> list[dict[str, Any]]:
    """Log sycophancy markers that slipped past banned-phrase stripping."""
    out: list[dict[str, Any]] = []
    for pattern in SYCOPHANCY_MARKERS:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            out.append(
                {
                    "violation_type": "sycophancy_marker",
                    "details": m.group(0)[:100],
                    "enforced": False,
                }
            )
    return out


async def score_and_rewrite(
    text: str,
    chat_id: int,
    recent_twily_msgs: list[str] | None = None,
    vibe_weights: dict[str, float] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply all enforcement rules, return (rewritten_text, violations).

    Violations is a list of dicts suitable for StyleEventsRepo.log_many().
    Each dict has: violation_type, details, before, after, enforced.
    Caller attaches before/after automatically via this function.
    """
    original = text
    recent = recent_twily_msgs or []
    all_violations: list[dict[str, Any]] = []

    # If vibe weights not provided, look them up.
    if vibe_weights is None:
        from app.db.repos.persona_vibe import VibeStateRepo

        state = await VibeStateRepo().get(chat_id)
        vibe_weights = {
            k: float(state[k])
            for k in ("w_warm_snarky", "w_dry_ironic", "w_caring_edge", "w_playful_flirt", "w_debate_socratic")
        }

    emoji_cap = emoji_budget_for_weights(vibe_weights)
    tildes_ban = tildes_banned_for_weights(vibe_weights)

    # 0. Normalize hearts → 💜 (always Twily's signature). ALWAYS applies.
    text, v = _normalize_hearts(text)
    all_violations.extend(v)

    # Structured markdown reports bypass conversational enforcement.
    # Section headers (📊/📅/etc), markdown bold, and long content are
    # structural, not sycophantic — rewriting them destroys the document.
    structured = _is_structured_report(text)
    if structured:
        all_violations.append(
            {
                "violation_type": "report_bypass",
                "details": f"len={len(text)}, skipped conversational enforcement",
                "enforced": False,
            }
        )
    else:
        # 1. Strip banned phrases.
        text, v = _strip_banned_phrases(text)
        all_violations.extend(v)

        # 2. Strip / cap tildes.
        text, v = _strip_tildes(text, banned=tildes_ban)
        all_violations.extend(v)

        # 3. Log emoji excess (soft — model self-regulates from prompt).
        text, v = _log_emoji_excess(text, emoji_cap)
        all_violations.extend(v)

        # 4. Cap stage directions.
        text, v = _cap_stage_directions(text)
        all_violations.extend(v)

    # 5. Log (don't rewrite) repetition.
    all_violations.extend(_detect_repetition(text, recent))

    # 6. Log (don't rewrite) sycophancy markers.
    all_violations.extend(_detect_sycophancy(text))

    # Collapse runs of spaces/tabs introduced by strip operations.
    # Do NOT collapse newlines — `\n\n` paragraph breaks must survive so
    # send_message.py's split("\n\n") fan-out produces multi-bubble replies
    # instead of a single blob.
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Normalize runs of 3+ newlines down to a single paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    # Collapse stray punctuation that banned-phrase removal may leave.
    text = re.sub(r"^[,.!?;:]+\s*", "", text)
    text = re.sub(r"[ \t]+([,.!?;:])", r"\1", text)

    # Attach before/after to each enforced violation.
    for v in all_violations:
        if v.get("enforced"):
            v["before"] = original
            v["after"] = text

    return text, all_violations
