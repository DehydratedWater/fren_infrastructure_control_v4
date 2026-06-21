"""PonyXL Prompt Composer - build structured prompts for PonyXL T2I with MLP characters."""

from __future__ import annotations

from typing import Literal

from src import ScriptTool
from pydantic import BaseModel, Field

# Character definitions with per-form tags, eyes, and body descriptions
CHARACTERS: dict[str, dict[str, str]] = {
    "twilight_sparkle": {
        "name": "Twilight Sparkle",
        "species": "unicorn",
        "cutie_mark": "six-pointed star cutie mark",
        "eyes": "large violet eyes, detailed beautiful eyes, long eyelashes",
        "pony": ("Twilight Sparkle, purple coat, unicorn horn, dark blue mane with pink streak"),
        "anthro": (
            "Twilight Sparkle, purple fur, unicorn horn, dark blue mane with pink streak,"
            " slender feminine body, medium chest, wide hips, long legs"
        ),
        "human": (
            "Twilight Sparkle, long dark blue hair with prominent pink streak, vivid violet eyes,"
            " light skin, anime girl, adult woman, mature face, slender elegant figure, medium bust, long legs"
        ),
        "neg": "Cozy Glow",
    },
    "rainbow_dash": {
        "name": "Rainbow Dash",
        "species": "pegasus",
        "cutie_mark": "rainbow lightning bolt cutie mark",
        "eyes": "large magenta eyes, detailed beautiful eyes, sharp confident gaze",
        "pony": ("Rainbow Dash, cyan coat, pegasus wings, rainbow-colored mane and tail"),
        "anthro": (
            "Rainbow Dash, cyan fur, pegasus wings, rainbow-colored mane and tail,"
            " toned athletic body, medium chest, fit abs, strong legs"
        ),
        "human": (
            "Rainbow Dash, short rainbow-colored hair, magenta eyes,"
            " tanned skin, adult woman, mature face, athletic figure, toned body, medium chest"
        ),
    },
    "fluttershy": {
        "name": "Fluttershy",
        "species": "pegasus",
        "cutie_mark": "three butterflies cutie mark",
        "eyes": "large teal eyes, gentle doe eyes, beautiful detailed eyes, long eyelashes",
        "pony": ("Fluttershy, yellow coat, pegasus wings, long flowing pink mane"),
        "anthro": (
            "Fluttershy, yellow fur, pegasus wings, long flowing pink mane,"
            " soft delicate body, large chest, wide hips, slender waist"
        ),
        "human": (
            "Fluttershy, long flowing pink hair, teal eyes, pale skin, soft delicate figure, large bust, shy posture"
        ),
    },
    "rarity": {
        "name": "Rarity",
        "species": "unicorn",
        "cutie_mark": "three diamonds cutie mark",
        "eyes": "large sapphire blue eyes, elegant detailed eyes, long curled eyelashes",
        "pony": ("Rarity, white coat, unicorn horn, purple curly mane"),
        "anthro": (
            "Rarity, white fur, unicorn horn, purple curly mane,"
            " elegant hourglass body, medium chest, slim waist, long graceful legs"
        ),
        "human": (
            "Rarity, curly purple hair, sapphire blue eyes,"
            " fair porcelain skin, elegant hourglass figure, sophisticated beauty"
        ),
    },
    "applejack": {
        "name": "Applejack",
        "species": "earth pony",
        "cutie_mark": "three apples cutie mark",
        "eyes": "large green eyes, honest bright eyes, detailed beautiful eyes",
        "pony": ("Applejack, orange coat, blonde mane in ponytail, freckles, cowboy hat"),
        "anthro": (
            "Applejack, orange fur, blonde mane in ponytail, freckles, cowboy hat,"
            " strong athletic body, toned arms, medium chest, muscular legs"
        ),
        "human": (
            "Applejack, long blonde hair in ponytail, green eyes, freckles, cowboy hat,"
            " tanned skin, strong athletic figure, toned body"
        ),
    },
    "pinkie_pie": {
        "name": "Pinkie Pie",
        "species": "earth pony",
        "cutie_mark": "three balloons cutie mark",
        "eyes": "large sky blue eyes, bright sparkling eyes, detailed beautiful eyes",
        "pony": ("Pinkie Pie, pink coat, curly pink mane and tail"),
        "anthro": (
            "Pinkie Pie, pink fur, curly pink mane and tail,"
            " curvy energetic body, large chest, thick thighs, bouncy figure"
        ),
        "human": (
            "Pinkie Pie, curly pink hair, sky blue eyes, fair skin, curvy energetic figure, large bust, cheerful pose"
        ),
    },
    "celestia": {
        "name": "Princess Celestia",
        "species": "alicorn",
        "cutie_mark": "golden sun cutie mark",
        "eyes": "large magenta eyes, wise regal eyes, detailed beautiful eyes",
        "pony": ("Princess Celestia, white coat, alicorn horn and wings, flowing pastel rainbow mane, tall, majestic"),
        "anthro": (
            "Princess Celestia, white fur, alicorn horn and wings,"
            " flowing pastel rainbow mane, tall statuesque body, large chest,"
            " long elegant legs, regal bearing"
        ),
        "human": (
            "Princess Celestia, long flowing pastel rainbow hair, magenta eyes,"
            " fair radiant skin, tall statuesque figure, large bust, regal elegant beauty"
        ),
    },
    "luna": {
        "name": "Princess Luna",
        "species": "alicorn",
        "cutie_mark": "crescent moon cutie mark",
        "eyes": "large teal eyes, mysterious ethereal eyes, detailed beautiful eyes, slit pupils",
        "pony": (
            "Princess Luna, dark blue coat, dark blue fur, dark blue body, alicorn horn and wings, flowing starry dark mane, ethereal, tall"
        ),
        "anthro": (
            "Princess Luna, dark blue fur, alicorn horn and wings,"
            " flowing starry dark mane, ethereal, tall elegant body, medium chest,"
            " long legs, mysterious dark beauty"
        ),
        "human": (
            "Princess Luna, long flowing dark blue starry hair, teal eyes,"
            " pale moonlit skin, tall elegant figure, mysterious dark beauty"
        ),
    },
    "trixie": {
        "name": "Trixie Lulamoon",
        "species": "unicorn",
        "cutie_mark": "magic wand and crescent moon cutie mark",
        "eyes": "large purple eyes, dramatic expressive eyes, detailed beautiful eyes",
        "pony": ("Trixie, light blue coat, unicorn horn, silver-white mane, showmare"),
        "anthro": (
            "Trixie, light blue fur, unicorn horn, silver-white mane,"
            " showmare, slim theatrical body, medium chest, long legs"
        ),
        "human": (
            "Trixie, silver-white hair, purple eyes, light skin, slim theatrical figure, dramatic showgirl beauty"
        ),
    },
    "starlight": {
        "name": "Starlight Glimmer",
        "species": "unicorn",
        "cutie_mark": "star with streaming trails cutie mark",
        "eyes": "large aqua eyes, intense intelligent eyes, detailed beautiful eyes",
        "pony": ("Starlight Glimmer, lilac coat, unicorn horn, purple and teal streaked mane"),
        "anthro": (
            "Starlight Glimmer, lilac fur, unicorn horn,"
            " purple and teal streaked mane, slender toned body, medium chest, long legs"
        ),
        "human": (
            "Starlight Glimmer, purple and teal streaked hair, aqua eyes,"
            " light skin, slender toned figure, intelligent beauty"
        ),
    },
}

# Base quality tags
QUALITY_PREFIX = "score_9, score_8_up, score_7_up"

# Form-specific prompt config: source tag, body tags, negative species tags
FORM_CONFIG: dict[str, dict[str, str | bool]] = {
    "pony": {
        "source": "source_furry",
        "body_tags": "mlp, pony, cute, fluffy ears",
        "face_tags": "",
        "neg_species": "anthro, bipedal, human, humanoid, human face, human skin, human body, catgirl, cat ears, feline, rabbit, bunny, rabbit ears, foxgirl, fox ears, boobs, breasts, chest, cleavage, spike, dragon",
        "neg_face": "realistic, photorealistic, real horse, muscular, detailed anatomy, uncanny",
        "allow_nsfw": True,
    },
    "anthro": {
        "source": "source_furry",
        "body_tags": "anthro, pony, equine, pony ears, horse ears, furry female, bipedal, humanoid body, breasts, tail, slim athletic build, fingers, hands, hooves on legs, feminine body, short face",
        "face_tags": "short snout, cute face, round face",
        # exclude OTHER species' ears/features — the furry model otherwise defaults
        # to cat/rabbit/fox ears on the anthro pony (she must stay an equine).
        "neg_species": "feral, quadruped, spike, dragon, catgirl, cat ears, cat tail, feline, whiskers, nekomimi, neko, rabbit, bunny, rabbit ears, bunny ears, foxgirl, fox ears, dog ears, wolf ears, deer antlers, antlers, generic furry, wrong species, kemonomimi",
        "neg_face": "horse head, long snout, long face, equine face",
        "allow_nsfw": True,
    },
    "human": {
        "source": "source_anime",
        "body_tags": "1girl, anime girl, humanoid, anime style",
        "face_tags": "cute face, small nose, round face, anime eyes",
        "neg_species": "furry, anthro, feral, pony, animal ears, tail, snout, fur, spike, dragon",
        "neg_face": "horse head, long snout, long face, equine face, animal face, photorealistic, real person, photograph",
        "allow_nsfw": True,
    },
}

# Negative prompt components (shared across forms)
NEG_QUALITY = "score_6, score_5, score_4"
NEG_STYLE = (
    "source_cartoon, 3d, chibi, (censor), monochrome, blurry, lowres,"
    " watermark, worst quality, bad quality, jpeg artifacts, child, loli, underage"
)
NEG_STYLE_PONY = "3d, (censor), monochrome, blurry, lowres, watermark, worst quality, bad quality, jpeg artifacts"
NEG_SFW = "nude, naked, nsfw, explicit, sexual, suggestive"
NEG_SFW_CONTENT_ONLY = "nsfw, explicit, sexual, suggestive"  # pony form: no nudity neg (they're naturally unclothed)

# Camera angle presets
CAMERA_ANGLES: dict[str, str] = {
    "front": "front view, facing viewer",
    "back": "from behind, back view, ass, back",
    "side": "from side, side view, profile",
    "above": "from above, bird's eye view, high angle",
    "below": "from below, low angle, looking up",
    "portrait": "close-up portrait, head and shoulders, face focus",
    "waist_up": "upper body, waist up shot",
    "full_body": "full body shot",
}

# Parametric expression scales: emotion -> list of (threshold, sfw_tags, nsfw_tags)
EXPRESSION_SCALES: dict[str, list[tuple[float, str, str]]] = {
    "happiness": [
        (0.2, "slight smile", "slight smile"),
        (0.4, "warm smile", "warm smile"),
        (0.6, "happy smile, bright eyes", "happy smile, bright eyes"),
        (0.8, "beaming, joyful grin, sparkling eyes", "beaming, joyful grin, sparkling eyes"),
        (1.0, "ecstatic, overjoyed, radiant expression", "ecstatic, overjoyed, radiant expression"),
    ],
    "anger": [
        (0.2, "slightly annoyed expression", "slightly annoyed expression"),
        (0.4, "furrowed brow, stern look", "furrowed brow, stern look"),
        (0.6, "angry scowl, clenched teeth", "angry scowl, clenched teeth"),
        (0.8, "furious, snarling, intense glare", "furious, snarling, intense glare"),
        (1.0, "enraged, wrathful expression, veins bulging", "enraged, wrathful expression, veins bulging"),
    ],
    "sadness": [
        (0.2, "wistful, melancholy look", "wistful, melancholy look"),
        (0.4, "sad eyes, downcast gaze", "sad eyes, downcast gaze"),
        (0.6, "crying, tears streaming", "crying, tears streaming"),
        (0.8, "sobbing, distraught expression", "sobbing, distraught expression"),
        (1.0, "devastated, anguished, tears flowing", "devastated, anguished, tears flowing"),
    ],
    "surprise": [
        (0.2, "curious look, wide eyes", "curious look, wide eyes"),
        (0.4, "surprised, open mouth", "surprised, open mouth"),
        (0.6, "shocked, gasping", "shocked, gasping"),
        (0.8, "stunned, jaw dropped", "stunned, jaw dropped"),
        (1.0, "extreme shock, mind blown expression", "extreme shock, mind blown expression"),
    ],
    "fear": [
        (0.2, "nervous, worried look", "nervous, worried look"),
        (0.4, "anxious, trembling slightly", "anxious, trembling slightly"),
        (0.6, "scared, frightened expression", "scared, frightened expression"),
        (0.8, "terrified, cowering", "terrified, cowering"),
        (1.0, "paralyzed with terror, wide eyes dilated", "paralyzed with terror, wide eyes dilated"),
    ],
    "confidence": [
        (0.2, "calm, composed expression", "calm, composed expression"),
        (0.4, "self-assured, knowing look", "self-assured, knowing look"),
        (0.6, "confident smirk", "confident smirk"),
        (0.8, "bold, commanding presence", "bold, commanding presence"),
        (1.0, "supremely powerful aura, dominant gaze", "supremely powerful aura, dominant gaze"),
    ],
    "suggestiveness": [
        (0.2, "coy look", "coy look, slight blush"),
        (0.4, "flirty wink", "flirty bedroom eyes, blushing"),
        (0.6, "seductive pose, alluring gaze", "seductive pose, sultry look, heavy blush"),
        (0.8, "provocative, lidded eyes", "lewd expression, blushing, panting, half-lidded eyes"),
        (1.0, "very provocative, smoldering gaze", "ahegao, tongue out, drooling, heart eyes, heavy blush"),
    ],
    "determination": [
        (0.2, "focused look", "focused look"),
        (0.4, "determined expression", "determined expression"),
        (0.6, "resolute, steely gaze", "resolute, steely gaze"),
        (0.8, "fierce determination, gritted teeth", "fierce determination, gritted teeth"),
        (1.0, "unwavering resolve, blazing eyes", "unwavering resolve, blazing eyes"),
    ],
}


def resolve_expression_tags(scales: dict[str, float], nsfw: bool) -> list[str]:
    """Convert numerical emotion scales to prompt tags."""
    tags: list[str] = []
    for emotion, value in scales.items():
        if value <= 0 or emotion not in EXPRESSION_SCALES:
            continue
        thresholds = EXPRESSION_SCALES[emotion]
        best_tags = ""
        for threshold, sfw_tags, nsfw_tags in thresholds:
            if value >= threshold:
                best_tags = nsfw_tags if nsfw else sfw_tags
        if best_tags:
            tags.append(best_tags)
    return tags


FormType = Literal["pony", "anthro", "human"]


class Input(BaseModel):
    command: str = Field(description="compose|list-characters|character-info|list-expressions")
    character: str = Field(
        default="twilight_sparkle",
        description="Character ID for single-character scenes",
    )
    characters: str = Field(
        default="",
        description="Comma-separated character IDs for multi-character scenes (overrides 'character')",
    )
    form: FormType = Field(
        default="anthro",
        description="Character body form: pony (feral quadruped), anthro (furry bipedal), human (anime girl)",
    )
    action: str = Field(default="", description="What character(s) are doing (e.g. reading a book)")
    location: str = Field(default="", description="Scene location (e.g. cozy library, moonlit garden)")
    clothing: str = Field(default="", description="Outfit description (e.g. purple wizard robe)")
    pose: str = Field(default="", description="Body pose (e.g. sitting at desk)")
    expression: str = Field(
        default="",
        description="Manual expression override (bypasses scales). Use scales for parametric control.",
    )
    camera: str = Field(default="", description="Free-form camera/composition (e.g. cinematic wide shot)")
    camera_angle: str = Field(
        default="",
        description="Preset camera angle: front|back|side|above|below|portrait|waist_up|full_body",
    )
    style: str = Field(default="detailed", description="Art style suffix (e.g. detailed, painterly)")
    aspect: str = Field(
        default="portrait",
        description="portrait (720x1280) or cinematic (1280x720) or square (1024x1024)",
    )
    seed: int = Field(default=-1, description="RNG seed for reproducibility (-1 = random)")
    nsfw: bool = Field(default=False, description="Allow NSFW content")
    include_cutie_mark: bool = Field(default=False, description="Include cutie mark in character tags")
    # Parametric expression scales (0.0 to 1.0)
    happiness: float = Field(default=0.0, description="Happiness scale 0.0-1.0")
    anger: float = Field(default=0.0, description="Anger scale 0.0-1.0")
    sadness: float = Field(default=0.0, description="Sadness scale 0.0-1.0")
    surprise: float = Field(default=0.0, description="Surprise scale 0.0-1.0")
    fear: float = Field(default=0.0, description="Fear scale 0.0-1.0")
    confidence: float = Field(default=0.0, description="Confidence scale 0.0-1.0")
    suggestiveness: float = Field(
        default=0.0,
        description="Suggestiveness scale 0.0-1.0 (NSFW tags at higher levels when nsfw=True)",
    )
    determination: float = Field(default=0.0, description="Determination scale 0.0-1.0")


class Output(BaseModel):
    success: bool = True
    data: dict = Field(default_factory=dict)
    items: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class PonyXLPromptComposerTool(ScriptTool[Input, Output]):
    name = "ponyxl_prompt_composer"
    description = (
        "Compose structured PonyXL prompts for MLP characters with dynamic scene parameters and parametric expressions"
    )

    def execute(self, inp: Input) -> Output:
        if inp.command == "compose":
            return self._compose(inp)
        elif inp.command == "list-characters":
            return self._list_characters()
        elif inp.command == "character-info":
            return self._character_info(inp)
        elif inp.command == "list-expressions":
            return self._list_expressions()
        else:
            return Output(success=False, error=f"Unknown command: {inp.command}")

    def _resolve_characters(self, inp: Input) -> list[dict[str, str]] | str:
        """Resolve character list. Returns list of char dicts or error string."""
        if inp.characters:
            char_ids = [c.strip() for c in inp.characters.split(",") if c.strip()]
        else:
            char_ids = [inp.character]

        chars = []
        for cid in char_ids:
            char = CHARACTERS.get(cid)
            if not char:
                available = ", ".join(sorted(CHARACTERS.keys()))
                return f"Unknown character: {cid}. Available: {available}"
            chars.append({"id": cid, **char})
        return chars

    def _compose(self, inp: Input) -> Output:
        resolved = self._resolve_characters(inp)
        if isinstance(resolved, str):
            return Output(success=False, error=resolved)

        char_list: list[dict[str, str]] = resolved
        num_chars = len(char_list)
        form = inp.form
        form_cfg = FORM_CONFIG[form]

        # Pony form: force SFW
        nsfw = inp.nsfw and bool(form_cfg["allow_nsfw"])

        # Build positive prompt
        parts: list[str] = [QUALITY_PREFIX, str(form_cfg["source"]), str(form_cfg["body_tags"])]

        # Face structure tags
        if form_cfg["face_tags"]:
            parts.append(str(form_cfg["face_tags"]))

        # Multi-character count tag
        if num_chars >= 2:
            if form == "pony":
                parts.append(f"{num_chars} ponies")
            else:
                parts.append(f"{num_chars}girls")

        # Action and location come before character descriptions
        if inp.action:
            parts.append(inp.action)
        if inp.location:
            parts.append(inp.location)

        # Character tags (form-specific body + eyes)
        for char in char_list:
            parts.append(char[form])
            parts.append(char["eyes"])
            if inp.include_cutie_mark:
                parts.append(char["cutie_mark"])

        if inp.clothing:
            parts.append(inp.clothing)
        elif not nsfw and form in ("anthro", "human"):
            # Default clothing for SFW anthro/human to prevent naked renders
            parts.append("wearing clothes, clothed, casual outfit")
        if inp.pose:
            parts.append(inp.pose)

        # Expression: manual override OR parametric scales
        if inp.expression:
            parts.append(inp.expression)
        else:
            scales = {
                "happiness": inp.happiness,
                "anger": inp.anger,
                "sadness": inp.sadness,
                "surprise": inp.surprise,
                "fear": inp.fear,
                "confidence": inp.confidence,
                "suggestiveness": inp.suggestiveness,
                "determination": inp.determination,
            }
            expr_tags = resolve_expression_tags(scales, nsfw)
            parts.extend(expr_tags)
        if inp.camera:
            parts.append(inp.camera)
        if inp.camera_angle:
            angle_tags = CAMERA_ANGLES.get(inp.camera_angle)
            if angle_tags:
                parts.append(angle_tags)
        if inp.style:
            parts.append(inp.style)

        # NSFW: move SFW-negative tags into positive prompt
        if nsfw:
            parts.append(NEG_SFW)

        positive = ", ".join(parts)

        # Build negative prompt
        neg_style = NEG_STYLE_PONY if form == "pony" else NEG_STYLE
        neg_parts: list[str] = [NEG_QUALITY, neg_style, str(form_cfg["neg_species"])]
        if form_cfg["neg_face"]:
            neg_parts.append(str(form_cfg["neg_face"]))
        for char in char_list:
            char_neg = char.get("neg")
            if char_neg:
                neg_parts.append(char_neg)
        if not nsfw:
            # Pony form: only block sexual content, not nudity (they're naturally unclothed)
            neg_parts.append(NEG_SFW_CONTENT_ONLY if form == "pony" else NEG_SFW)
        if num_chars == 1:
            neg_parts.append("multiple characters, multiple girls, 2girls, 3girls, group")

        negative = ", ".join(neg_parts)

        # Resolve workflow
        workflow_map = {
            "cinematic": "ponyxl_t2i",
            "portrait": "ponyxl_t2i_portrait",
            "square": "ponyxl_t2i_square",
        }
        workflow_id = workflow_map.get(inp.aspect, "ponyxl_t2i_portrait")

        # Collect active scales for response
        active_scales = {}
        for emo in EXPRESSION_SCALES:
            val = getattr(inp, emo, 0.0)
            if val > 0:
                active_scales[emo] = val

        char_names = [c["name"] for c in char_list]
        char_ids = [c["id"] for c in char_list]

        return Output(
            success=True,
            data={
                "positive_prompt": positive,
                "negative_prompt": negative,
                "workflow_id": workflow_id,
                "seed": inp.seed,
                "aspect": inp.aspect,
                "form": form,
                "characters": char_names,
                "character_ids": char_ids,
                "num_characters": num_chars,
                "nsfw": nsfw,
                "expression_scales": active_scales,
                "parameters": {
                    "action": inp.action or "(default)",
                    "location": inp.location or "(default)",
                    "clothing": inp.clothing or "(default)",
                    "pose": inp.pose or "(default)",
                    "expression": (
                        inp.expression or "(parametric)" if active_scales else inp.expression or "(default)"
                    ),
                    "camera": inp.camera or "(default)",
                    "style": inp.style or "detailed",
                },
            },
        )

    def _list_characters(self) -> Output:
        items = [
            {
                "id": char_id,
                "name": char["name"],
                "species": char["species"],
                "forms": ["pony", "anthro", "human"],
            }
            for char_id, char in CHARACTERS.items()
        ]
        return Output(success=True, items=items, count=len(items))

    def _character_info(self, inp: Input) -> Output:
        char = CHARACTERS.get(inp.character)
        if not char:
            available = ", ".join(sorted(CHARACTERS.keys()))
            return Output(
                success=False,
                error=f"Unknown character: {inp.character}. Available: {available}",
            )

        return Output(
            success=True,
            data={
                "id": inp.character,
                "name": char["name"],
                "species": char["species"],
                "eyes": char["eyes"],
                "cutie_mark": char["cutie_mark"],
                "forms": {
                    "pony": char["pony"],
                    "anthro": char["anthro"],
                    "human": char["human"],
                },
            },
        )

    def _list_expressions(self) -> Output:
        items = []
        for emotion, thresholds in EXPRESSION_SCALES.items():
            levels = []
            for threshold, sfw_tags, nsfw_tags in thresholds:
                level: dict = {"threshold": threshold, "sfw_tags": sfw_tags}
                if nsfw_tags != sfw_tags:
                    level["nsfw_tags"] = nsfw_tags
                levels.append(level)
            items.append({"emotion": emotion, "levels": levels})
        return Output(success=True, items=items, count=len(items))


if __name__ == "__main__":
    PonyXLPromptComposerTool.run()
