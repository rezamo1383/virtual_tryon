"""Prompt extension points for wallpaper visualization."""

from __future__ import annotations

from typing import Any


class WallpaperPromptBuilder:
    """Build future analysis and generation prompts for wallpaper jobs."""

    domain = "wallpaper"

    def wall_analysis(self) -> str:
        return (
            "Room image -> JSON only. Keys: wall_detected:bool; confidence:0..1; "
            "wall_polygon:list of exactly 4 {x,y} normalized points ordered "
            "top-left,top-right,bottom-right,bottom-left; wall_count:int; "
            "occlusions:list[str]; lighting:str; warnings:list[str]. Select the "
            "largest visible wall suitable for wallpaper. If none, use an empty "
            "polygon."
        )

    def output_evaluation(self) -> str:
        return (
            "Images: room, wallpaper reference, output. JSON only. Scores 0..1: "
            "wall_coverage,pattern_fidelity,perspective_accuracy,"
            "lighting_preservation,scene_integrity,overall_score; accepted:bool; "
            "problems:list[str]; retry_recommendation:str|null. Reject changes "
            "outside walls, distorted pattern, lost furniture, or bad perspective."
        )

    def generation(
        self,
        category: str,
        options: dict[str, Any],
    ) -> str:
        lighting = (
            "preserve original lighting"
            if options.get("preserve_lighting", True)
            else "use physically plausible lighting"
        )
        scale = round(float(options.get("pattern_scale", 0.18)) * 100)
        prompt = (
            "Create one photorealistic edit, not a redesign. Image 1 is the "
            "original room. Image 2 is the exact wallpaper material; ignore its "
            "text, URL, logo, watermark, border, and product-label elements. "
            "Install the wallpaper only on visible wall surfaces in Image 1, "
            "behind every existing "
            "TV, curtain, door, cabinet, lamp, sofa and decoration. Keep the exact "
            "camera, crop, architecture and object pixels. Repeat the motif "
            f"seamlessly at about {scale}% image-width scale; correct perspective "
            f"at corners and wall planes; {lighting}, texture and existing shadows. "
            "No text, seams, stickers, new objects or changes outside wall finish."
        )
        if options.get("strict_wall_only"):
            prompt += " Strict retry: change wall finish only."
        return prompt
