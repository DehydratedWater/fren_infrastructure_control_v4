"""RP Scene Composer — generic PonyXL prompt composer for RP adventure illustrations.

Unlike ponyxl_prompt_composer (MLP-character-locked), this tool composes free-form
scene prompts suitable for any RP adventure setting — fantasy, sci-fi, horror, etc.
Still uses PonyXL score tags and anime-style quality tags for best results with
the PonyXL model on ComfyUI.
"""

from __future__ import annotations

from src import ScriptTool
from pydantic import BaseModel, Field

# ── Quality tags (critical for PonyXL) ──────────────────────────────────────
QUALITY_PREFIX = (
    "score_9, score_8_up, score_7_up, "
    "masterpiece, best quality, highly detailed, sharp focus, "
    "intricate details, cinematic lighting, professional illustration"
)
NEG_QUALITY = "score_6, score_5, score_4, worst quality, low quality, normal quality"

# ── Style presets ───────────────────────────────────────────────────────────
STYLE_PRESETS: dict[str, dict[str, str]] = {
    "anime": {
        "source": "source_anime",
        "positive": "anime style, anime art, detailed anime, vibrant colors",
        "negative": "photorealistic, real person, photograph, 3d render, western cartoon",
    },
    "dark_fantasy": {
        "source": "source_anime",
        "positive": "dark fantasy art, anime style, detailed, dramatic lighting, moody atmosphere, rich colors",
        "negative": "photorealistic, real person, photograph, bright, cheerful, chibi, cute",
    },
    "fantasy": {
        "source": "source_anime",
        "positive": "fantasy art, anime style, detailed, magical atmosphere, vibrant, epic",
        "negative": "photorealistic, real person, photograph, modern, sci-fi, spaceship",
    },
    "sci_fi": {
        "source": "source_anime",
        "positive": "sci-fi art, anime style, detailed, futuristic, neon lighting, high tech",
        "negative": "photorealistic, real person, photograph, medieval, fantasy, sword",
    },
    "horror": {
        "source": "source_anime",
        "positive": "dark horror art, anime style, detailed, eerie, unsettling, dark shadows, dramatic lighting",
        "negative": "photorealistic, real person, photograph, bright, cheerful, cute, chibi",
    },
    "pixel_art": {
        "source": "source_anime",
        "positive": "pixel art style, retro game aesthetic, detailed pixel art, 16-bit",
        "negative": "photorealistic, real person, photograph, smooth shading, 3d render",
    },
    "painterly": {
        "source": "source_anime",
        "positive": "painterly style, oil painting, detailed brushstrokes, rich colors, atmospheric",
        "negative": "photorealistic, real person, photograph, 3d render, flat colors",
    },
}

# ── Mood presets ────────────────────────────────────────────────────────────
MOOD_PRESETS: dict[str, str] = {
    "epic": "epic scale, grand vista, dramatic perspective, sweeping composition",
    "dramatic": "dramatic lighting, high contrast, intense atmosphere, cinematic",
    "peaceful": "peaceful atmosphere, soft lighting, warm tones, serene",
    "eerie": "eerie atmosphere, fog, mist, low visibility, mysterious, unsettling",
    "tense": "tense atmosphere, dramatic shadows, sharp contrast, suspenseful",
    "mysterious": "mysterious atmosphere, shadows, hidden details, enigmatic",
    "action": "dynamic action scene, motion blur, dramatic angle, intense, fast-paced",
    "melancholic": "melancholic atmosphere, muted colors, rain, somber, reflective",
    "joyful": "bright vibrant colors, warm lighting, cheerful atmosphere, sunny",
    "romantic": "soft romantic lighting, warm tones, intimate atmosphere, gentle",
}

# ── Camera presets ──────────────────────────────────────────────────────────
CAMERA_PRESETS: dict[str, str] = {
    "close_up": "close-up, face focus, detailed features",
    "portrait": "portrait shot, head and shoulders, centered",
    "upper_body": "upper body shot, waist up",
    "full_body": "full body shot, entire figure visible",
    "wide_shot": "wide shot, full scene, environmental, establishing shot",
    "cinematic": "cinematic composition, widescreen, dramatic framing, rule of thirds",
    "birds_eye": "bird's eye view, overhead, looking down at scene",
    "low_angle": "low angle shot, looking up, imposing perspective",
    "dutch_angle": "dutch angle, tilted composition, dynamic, unsettling",
    "over_shoulder": "over-the-shoulder shot, perspective from behind character",
}

# ── Default negative (shared) ──────────────────────────────────────────────
NEG_SHARED = (
    "source_cartoon, 3d, chibi, (censor), monochrome, blurry, lowres,"
    " watermark, bad quality, jpeg artifacts, child, loli, underage,"
    " text, signature, logo, border, frame, username,"
    " bad anatomy, bad hands, extra fingers, missing fingers, extra limbs,"
    " deformed, mutated, disfigured, poorly drawn, out of frame, cropped,"
    " duplicate, blurry face, asymmetric eyes, cross-eyed, malformed"
)

NEG_NSFW_BLOCK = "nude, naked, nsfw, explicit, sexual, suggestive"


def _subject_count_token(count: int, gender_hint: str) -> str:
    """Build a PonyXL count-anchor token from subject count + optional gender hint.

    Examples:
        count=0 → "" (no anchor; good for empty landscapes)
        count=1, hint="girl" → "1girl, solo"
        count=1, hint="boy"  → "1boy, solo"
        count=2, hint="girl" → "2girls"
        count=2, hint="mixed" → "1girl, 1boy"
        count=3, hint=""     → "3girls"
    """
    if count <= 0:
        return ""
    hint = (gender_hint or "").strip().lower() or "girl"
    if count == 1:
        if hint == "boy":
            return "1boy, solo"
        if hint == "ambiguous":
            return "1other, solo"
        return "1girl, solo"
    if count == 2 and hint == "mixed":
        return "1girl, 1boy"
    plural = "boys" if hint == "boy" else ("others" if hint == "ambiguous" else "girls")
    return f"{count}{plural}"


class Input(BaseModel):
    command: str = Field(description="compose|list-styles|list-moods|list-cameras")
    # Scene description (free-form)
    scene: str = Field(
        default="",
        description="Free-form scene description — what's happening, who's there, what's visible",
    )
    characters: str = Field(
        default="",
        description="Character descriptions — appearance, clothing, pose (free-form text)",
    )
    environment: str = Field(
        default="",
        description="Environment/location — where the scene takes place",
    )
    action: str = Field(
        default="",
        description="What's happening — action, interaction, event",
    )
    # Style controls
    style: str = Field(
        default="anime",
        description="Art style preset: anime|dark_fantasy|fantasy|sci_fi|horror|pixel_art|painterly",
    )
    mood: str = Field(
        default="",
        description="Mood preset: epic|dramatic|peaceful|eerie|tense|mysterious|action|melancholic|joyful|romantic",
    )
    camera: str = Field(
        default="",
        description="Camera preset: close_up|portrait|upper_body|full_body|wide_shot|cinematic|birds_eye|low_angle|dutch_angle|over_shoulder",
    )
    camera_custom: str = Field(
        default="",
        description="Free-form camera/composition override (ignored if 'camera' is set)",
    )
    lighting: str = Field(
        default="",
        description="Lighting description (e.g. 'moonlit', 'torch-lit dungeon', 'golden hour')",
    )
    # Output control
    aspect: str = Field(
        default="portrait",
        description="Aspect ratio: portrait (720x1280), cinematic (1280x720), square (1024x1024)",
    )
    extra_positive: str = Field(
        default="",
        description="Additional positive prompt tags to append",
    )
    extra_negative: str = Field(
        default="",
        description="Additional negative prompt tags to append",
    )
    subject_count: int = Field(
        default=0,
        description=(
            "Number of named characters visible in the scene (0 = ambient/no named "
            "subjects, 1, 2, 3, etc.). Used to emit PonyXL count tokens like '1girl', "
            "'2girls', '1boy 1girl' for stronger subject anchoring."
        ),
    )
    subject_gender_hint: str = Field(
        default="",
        description=(
            "Optional gender hint for the count token: 'girl', 'boy', 'mixed', "
            "or 'ambiguous'. If empty, defaults to 'girl'."
        ),
    )
    nsfw: bool = Field(default=False, description="Allow NSFW content")


class Output(BaseModel):
    success: bool = True
    data: dict = Field(default_factory=dict)
    error: str = ""


class RPSceneComposerTool(ScriptTool[Input, Output]):
    name = "rp_scene_composer"
    description = (
        "Compose free-form PonyXL prompts for RP adventure scene illustrations. "
        "Supports any setting — fantasy, sci-fi, horror, etc. "
        "Uses PonyXL score tags for quality. NOT character-locked like ponyxl_prompt_composer."
    )

    def execute(self, inp: Input) -> Output:
        if inp.command == "compose":
            return self._compose(inp)
        elif inp.command == "list-styles":
            return self._list_presets("styles", STYLE_PRESETS)
        elif inp.command == "list-moods":
            return self._list_presets("moods", MOOD_PRESETS)
        elif inp.command == "list-cameras":
            return self._list_presets("cameras", CAMERA_PRESETS)
        else:
            return Output(success=False, error=f"Unknown command: {inp.command}")

    def _compose(self, inp: Input) -> Output:
        # Resolve style preset
        style = STYLE_PRESETS.get(inp.style, STYLE_PRESETS["anime"])

        # Build positive prompt in a stable order regardless of which fields arrive:
        #   1. quality + style source
        #   2. subject count token (PonyXL anchors heavily on these)
        #   3. mood
        #   4. environment  → subject  → action  → scene summary
        #   5. camera + lighting
        #   6. style positive modifiers
        #   7. user extras
        parts: list[str] = [QUALITY_PREFIX, style["source"]]

        # Subject count token (critical anchoring for PonyXL)
        count_token = _subject_count_token(inp.subject_count, inp.subject_gender_hint)
        if count_token:
            parts.append(count_token)

        # Mood
        if inp.mood and inp.mood in MOOD_PRESETS:
            parts.append(MOOD_PRESETS[inp.mood])

        # Scene elements (strict order — never let the caller shuffle them)
        if inp.environment:
            parts.append(inp.environment)
        if inp.characters:
            parts.append(inp.characters)
        if inp.action:
            parts.append(inp.action)
        if inp.scene:
            parts.append(inp.scene)

        # Camera
        if inp.camera and inp.camera in CAMERA_PRESETS:
            parts.append(CAMERA_PRESETS[inp.camera])
        elif inp.camera_custom:
            parts.append(inp.camera_custom)

        # Lighting
        if inp.lighting:
            parts.append(inp.lighting)

        # Style descriptors come AFTER scene content so they modify rather than override.
        parts.append(style["positive"])

        # Extra tags
        if inp.extra_positive:
            parts.append(inp.extra_positive)

        positive = ", ".join(parts)

        # Build negative prompt
        neg_parts: list[str] = [NEG_QUALITY, NEG_SHARED, style["negative"]]
        if not inp.nsfw:
            neg_parts.append(NEG_NSFW_BLOCK)
        if inp.extra_negative:
            neg_parts.append(inp.extra_negative)

        negative = ", ".join(neg_parts)

        # Resolve workflow
        workflow_map = {
            "cinematic": "ponyxl_t2i",
            "portrait": "ponyxl_t2i_portrait",
            "square": "ponyxl_t2i_square",
        }
        workflow_id = workflow_map.get(inp.aspect, "ponyxl_t2i_portrait")

        return Output(
            success=True,
            data={
                "positive_prompt": positive,
                "negative_prompt": negative,
                "workflow_id": workflow_id,
                "aspect": inp.aspect,
                "style": inp.style,
                "mood": inp.mood,
            },
        )

    def _list_presets(self, name: str, presets: dict) -> Output:
        items = []
        for key, val in presets.items():
            desc = val if isinstance(val, str) else val.get("positive", "")
            items.append({"name": key, "description": desc[:200]})
        return Output(success=True, data={"presets": items})


if __name__ == "__main__":
    RPSceneComposerTool.run()
