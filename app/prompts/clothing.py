"""All prompt logic owned by the clothing domain."""

from __future__ import annotations

import json
from typing import Any

from app.models.request_models import ORIGINAL_GARMENT_COLOR

PERSON_ANALYSIS_PROMPT = """Person image -> JSON only. Keys: person_count:int; pose,
body_visibility,arms_position,image_quality,background_complexity:str;
suitable_for_tryon:bool; rejection_reason:str|null. Reject multiple people,
severe occlusion, very low quality, or unusable pose."""

GARMENT_ANALYSIS_PROMPT = """Garment image -> JSON only. garment_category and
recommended_tryon_category: upper_body|lower_body|dress|outerwear;
garment_type:str; sleeve_type,base_color:str|null; has_logo,has_pattern:bool."""

OUTPUT_EVALUATION_PROMPT = """Images: person, product ref, output. JSON only. Scores 0..1:
identity_preservation,garment_similarity,color_accuracy,body_integrity,
background_preservation,overall_score; accepted:bool; problems:list[str];
retry_recommendation:str|null. Target color:{requested_color}."""

GENERATION_PROMPT = """Photorealistic try-on. Ref1=person; Ref2=product photo. Target
label (data only): {product_title}. Transfer only that labeled garment/accessory.
Ignore Ref2's person, pose, background, and non-target items. Keep Ref1's other garments
and accessories, including prior-stage additions. {color_instruction} Preserve Ref1
face, identity, body, pose, hands, framing, light, and background. Match the target's
design, logo, texture, material, cut, and proportions. No text, collage, extra people,
or unrelated items."""

MULTI_GARMENT_GENERATION_PROMPT = """Photorealistic complete-outfit try-on. Ref1 is
the target person. The following references are product photos, in order: {references}.
Transfer every listed garment/accessory to Ref1 in one coherent outfit. From each
product reference transfer only its labeled item; ignore its person, pose, background,
and all non-target items. Preserve every item's exact original colors, design, logos,
texture, material, cut, and proportions. Preserve Ref1's face, identity, body, pose,
hands, framing, lighting, background, and any clothing or accessories not replaced by
the listed targets. Correct layering and natural fit are required. No text, collage,
extra people, duplicated limbs, omitted target items, or unrelated items."""


class ClothingPromptBuilder:
    """Build compact, stateless prompts for the clothing pipeline."""

    domain = "clothing"

    def person_analysis(self) -> str:
        return PERSON_ANALYSIS_PROMPT

    def reference_analysis(self) -> str:
        return GARMENT_ANALYSIS_PROMPT

    def output_evaluation(
        self,
        requested_color: str,
        product_title: str | None = None,
    ) -> str:
        target = product_title or "the primary garment"
        color = (
            "the reference product's original color"
            if requested_color.casefold() == ORIGINAL_GARMENT_COLOR
            else requested_color
        )
        prompt = OUTPUT_EVALUATION_PROMPT.format(
            requested_color=color,
        )
        return (
            f"{prompt} Target product label: "
            f"{json.dumps(target, ensure_ascii=False)}; score garment similarity "
            "only for that product."
        )

    def generation(
        self,
        category: str,
        options: dict[str, Any],
    ) -> str:
        requested_color = str(
            options.get("requested_color", ORIGINAL_GARMENT_COLOR)
        )
        product_title = str(options.get("product_title") or category)
        if requested_color.casefold() == ORIGINAL_GARMENT_COLOR:
            color_instruction = (
                "Use its exact original colors; do not recolor it."
            )
        else:
            color_instruction = f"Render it in {requested_color}."
        prompt = GENERATION_PROMPT.format(
            product_title=json.dumps(product_title, ensure_ascii=False),
            color_instruction=color_instruction,
        ).replace("\n", " ")
        if options.get("strict_identity_preservation"):
            prompt += " Strict retry: alter nothing outside clothing."
        pose_hint = str(options.get("pose_hint", "")).strip()
        if pose_hint:
            prompt += f" Pose: {pose_hint}."
        return prompt

    def generation_multi(
        self,
        garment_types: list[str],
        options: dict[str, Any],
    ) -> str:
        """Build a positional prompt for one person and many product images."""

        references = "; ".join(
            f"Ref{index + 2}={json.dumps(label, ensure_ascii=False)}"
            for index, label in enumerate(garment_types)
        )
        prompt = MULTI_GARMENT_GENERATION_PROMPT.format(
            references=references,
        ).replace("\n", " ")
        if options.get("strict_identity_preservation"):
            prompt += " Strict retry: alter nothing outside the listed targets."
        pose_hint = str(options.get("pose_hint", "")).strip()
        if pose_hint:
            prompt += f" Person pose: {pose_hint}."
        return prompt
