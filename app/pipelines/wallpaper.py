"""Production-oriented wallpaper visualization pipeline."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from app.clients.qwen_client import QwenClient
from app.clients.tryon_api_client import TryOnAPIClient
from app.core.config import Settings
from app.models.request_models import GenerationRequest, WallpaperOptions
from app.models.wallpaper_models import (
    LightingPreservationResult,
    PerspectiveEstimationResult,
    TextureRepetitionResult,
    WallAnalysisResult,
    WallSegmentationResult,
    WallpaperCandidateResult,
    WallpaperJobResult,
)
from app.pipelines.base import BasePipeline
from app.prompts.wallpaper import WallpaperPromptBuilder
from app.services.output_manager import OutputManager
from app.services.retry_manager import RetryManager
from app.services.wallpaper_processing import (
    OpenCVLightingPreserver,
    OpenCVPerspectiveEstimator,
    OpenCVTextureRepeater,
    PolygonWallSegmenter,
    SemanticWallAnalyzer,
    SemanticWallSegmentationEngine,
    SemanticWallSegmenter,
    VisionWallAnalyzer,
    WallpaperGenerationService,
    WallpaperOutputEvaluator,
    WallpaperReferencePreprocessor,
)
from app.utils.file_utils import remove_tree
from app.utils.hashing import create_job_id
from app.validators.wallpaper import WallpaperValidator

LOGGER = logging.getLogger(__name__)


class WallAnalyzer(Protocol):
    async def analyze(self, room_image: Path) -> WallAnalysisResult:
        """Analyze the largest visible wall."""


class WallSegmenter(Protocol):
    async def segment(
        self,
        room_image: Path,
        analysis: WallAnalysisResult,
        output_directory: Path,
    ) -> WallSegmentationResult:
        """Create the selected wall mask."""


class PerspectiveEstimator(Protocol):
    async def estimate(
        self,
        room_image: Path,
        segmentation: WallSegmentationResult,
    ) -> PerspectiveEstimationResult:
        """Estimate the wall homography."""


class TextureRepeater(Protocol):
    async def repeat(
        self,
        wallpaper_image: Path,
        perspective: PerspectiveEstimationResult,
        output_directory: Path,
        *,
        pattern_scale: float,
    ) -> TextureRepetitionResult:
        """Build a perspective-aware texture preview."""


class LightingPreserver(Protocol):
    async def preserve(
        self,
        room_image: Path,
        generated_image: Path,
        segmentation: WallSegmentationResult,
        output_directory: Path,
        *,
        preserve_lighting: bool,
    ) -> LightingPreservationResult:
        """Restore original lighting and non-wall pixels."""


class WallpaperPipeline(BasePipeline):
    """Analyze, generate, evaluate, retry, and finalize wallpaper jobs."""

    pipeline_name = "wallpaper"

    def __init__(
        self,
        *,
        tenant_id: str,
        settings: Settings,
        analysis_client: QwenClient,
        generation_client: TryOnAPIClient,
        prompt_builder: WallpaperPromptBuilder,
        wall_analyzer: WallAnalyzer | None = None,
        wall_segmenter: WallSegmenter | None = None,
        perspective_estimator: PerspectiveEstimator | None = None,
        texture_repeater: TextureRepeater | None = None,
        lighting_preserver: LightingPreserver | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.settings = settings
        self.analysis_client = analysis_client
        self.generation_client = generation_client
        self.prompts = prompt_builder
        self.validator = WallpaperValidator(settings)
        self.output_manager = OutputManager(settings.output_directory)
        if settings.wallpaper_segmentation_backend == "semantic":
            semantic_engine = SemanticWallSegmentationEngine(settings)
            default_analyzer: WallAnalyzer = SemanticWallAnalyzer(
                semantic_engine
            )
            default_segmenter: WallSegmenter = SemanticWallSegmenter(
                semantic_engine
            )
        else:
            default_analyzer = VisionWallAnalyzer(
                analysis_client,
                prompt_builder,
            )
            default_segmenter = PolygonWallSegmenter()
        self.wall_analyzer = wall_analyzer or default_analyzer
        self.wall_segmenter = wall_segmenter or default_segmenter
        self.perspective_estimator = (
            perspective_estimator or OpenCVPerspectiveEstimator()
        )
        self.texture_repeater = texture_repeater or OpenCVTextureRepeater()
        self.reference_preprocessor = WallpaperReferencePreprocessor()
        self.generation_service = WallpaperGenerationService(
            generation_client
        )
        self.evaluator = WallpaperOutputEvaluator(
            analysis_client,
            prompt_builder,
        )
        self.lighting_preserver = lighting_preserver or OpenCVLightingPreserver(
            settings.wallpaper_mask_feather_radius
        )
        self.retry_manager = RetryManager()

    async def run(self, request: GenerationRequest) -> WallpaperJobResult:
        started_at = datetime.now(UTC)
        validated = self.validator.validate(request)
        options = WallpaperOptions.model_validate(validated.options)
        job_id = create_job_id()
        job_directory = self.output_manager.create_job_directory(job_id)
        temp_directory = self.settings.temp_directory / job_id
        temp_directory.mkdir(parents=True, exist_ok=False)
        self.output_manager.write_request(
            job_directory,
            {
                "tenant_id": self.tenant_id,
                "pipeline": self.pipeline_name,
                **validated.model_dump(mode="json"),
            },
        )
        stages: list[str] = []
        analysis: WallAnalysisResult | None = None
        segmentation: WallSegmentationResult | None = None
        perspective: PerspectiveEstimationResult | None = None
        candidates: list[WallpaperCandidateResult] = []
        retry_count = 0
        try:
            analysis = await self.wall_analyzer.analyze(
                validated.source_image
            )
            stages.append("wall_analysis")
            self.output_manager.write_metadata(
                job_directory,
                "wall_analysis.json",
                analysis,
            )
            if not analysis.wall_detected:
                result = self._rejected_result(
                    job_id=job_id,
                    request=validated,
                    started_at=started_at,
                    stages=stages,
                    analysis=analysis,
                )
                self.output_manager.write_result(job_directory, result)
                LOGGER.info(
                    "wallpaper_job_rejected",
                    extra={
                        "job_id": job_id,
                        "tenant_id": self.tenant_id,
                        "stage": "wall_analysis",
                        "accepted": False,
                    },
                )
                return result

            artifact_directory = job_directory / "wallpaper"
            prepared_reference = await self.reference_preprocessor.prepare(
                validated.reference_image,
                artifact_directory,
            )
            stages.append("reference_preprocessing")
            segmentation = await self.wall_segmenter.segment(
                validated.source_image,
                analysis,
                artifact_directory,
            )
            stages.append("wall_segmentation")
            perspective = await self.perspective_estimator.estimate(
                validated.source_image,
                segmentation,
            )
            stages.append("perspective_estimation")
            texture = await self.texture_repeater.repeat(
                prepared_reference,
                perspective,
                artifact_directory,
                pattern_scale=options.pattern_scale,
            )
            stages.append("texture_repetition")
            base_options: dict[str, object] = {
                "preserve_lighting": options.preserve_lighting,
                "preserve_room_geometry": options.preserve_room_geometry,
                "pattern_scale": options.pattern_scale,
                "generation_quality": "high",
                "wall_region_hint": [
                    [round(point.x, 3), round(point.y, 3)]
                    for point in analysis.wall_polygon
                ],
                "prompt": self.prompts.generation(
                    "wallpaper",
                    options.model_dump(),
                ),
            }
            best: WallpaperCandidateResult | None = None
            while True:
                attempt_options = self.retry_manager.options_for_attempt(
                    base_options,
                    retry_count,
                    strict_options={
                        "strict_wall_only": True,
                        "preserve_lighting": True,
                        "preserve_room_geometry": True,
                    },
                )
                generated = await self.generation_service.generate_candidates(
                    room_image=validated.source_image,
                    wallpaper_image=prepared_reference,
                    segmentation=segmentation,
                    output_directory=(
                        job_directory
                        / "candidates"
                        / f"attempt_{retry_count:02d}"
                    ),
                    count=options.candidates_per_job,
                    attempt=retry_count,
                    start_index=len(candidates) + 1,
                    options=attempt_options,
                )
                for candidate in generated:
                    candidate.evaluation = await self.evaluator.evaluate(
                        validated.source_image,
                        prepared_reference,
                        candidate.path,
                    )
                candidates.extend(generated)
                best = max(
                    candidates,
                    key=lambda item: (
                        item.evaluation.overall_score
                        if item.evaluation is not None
                        else -1
                    ),
                )
                accepted = self._is_accepted(best)
                if not self.retry_manager.should_retry(
                    accepted,
                    retry_count,
                    options.max_retries,
                ):
                    break
                retry_count += 1
            stages.extend(["wallpaper_generation", "output_evaluation"])
            if best is None or best.evaluation is None:
                raise RuntimeError("Wallpaper candidate selection failed.")
            final = await self.lighting_preserver.preserve(
                validated.source_image,
                best.path,
                segmentation,
                job_directory / "final",
                preserve_lighting=options.preserve_lighting,
            )
            stages.append("lighting_preservation")
            accepted = self._is_accepted(best)
            resolved_job_directory = job_directory.resolve()
            relative_output = final.image_path.relative_to(
                resolved_job_directory
            )
            result = WallpaperJobResult(
                job_id=job_id,
                tenant_id=self.tenant_id,
                status=(
                    "completed" if accepted else "completed_with_failures"
                ),
                source_image=validated.source_image,
                reference_image=validated.reference_image,
                completed_stages=stages,
                analysis=analysis,
                segmentation=segmentation,
                perspective=perspective,
                texture_preview=texture.texture_path.relative_to(
                    resolved_job_directory
                ),
                output=relative_output,
                score=best.evaluation.overall_score,
                accepted=accepted,
                retry_count=retry_count,
                candidates_evaluated=len(candidates),
                problems=best.evaluation.problems,
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )
            self.output_manager.write_metadata(
                job_directory,
                "candidate_metadata.json",
                {
                    "candidates": [
                        item.model_dump(mode="json") for item in candidates
                    ]
                },
            )
            self.output_manager.write_result(job_directory, result)
            LOGGER.info(
                "wallpaper_job_finished",
                extra={
                    "job_id": job_id,
                    "tenant_id": self.tenant_id,
                    "pipeline": self.pipeline_name,
                    "status": result.status,
                    "score": result.score,
                    "retry_count": retry_count,
                },
            )
            return result
        except Exception as exc:
            failure = WallpaperJobResult(
                job_id=job_id,
                tenant_id=self.tenant_id,
                status="failed",
                source_image=validated.source_image,
                reference_image=validated.reference_image,
                completed_stages=stages,
                analysis=analysis,
                segmentation=segmentation,
                perspective=perspective,
                retry_count=retry_count,
                candidates_evaluated=len(candidates),
                error=str(exc)[:1000],
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )
            self.output_manager.write_result(job_directory, failure)
            LOGGER.error(
                "wallpaper_job_failed",
                extra={
                    "job_id": job_id,
                    "tenant_id": self.tenant_id,
                    "error_type": type(exc).__name__,
                },
            )
            raise
        finally:
            if self.settings.delete_temp_files:
                remove_tree(temp_directory, self.settings.temp_directory)

    def _is_accepted(self, candidate: WallpaperCandidateResult) -> bool:
        evaluation = candidate.evaluation
        return bool(
            evaluation
            and evaluation.accepted
            and evaluation.overall_score
            >= self.settings.wallpaper_min_acceptance_score
        )

    def _rejected_result(
        self,
        *,
        job_id: str,
        request: GenerationRequest,
        started_at: datetime,
        stages: list[str],
        analysis: WallAnalysisResult,
    ) -> WallpaperJobResult:
        return WallpaperJobResult(
            job_id=job_id,
            tenant_id=self.tenant_id,
            status="rejected",
            source_image=request.source_image,
            reference_image=request.reference_image,
            completed_stages=stages,
            analysis=analysis,
            rejection_reason="No wallpaper-suitable wall was detected.",
            started_at=started_at,
            completed_at=datetime.now(UTC),
        )

    async def aclose(self) -> None:
        """Close tenant-scoped analysis and generation clients."""

        await asyncio.gather(
            self.analysis_client.aclose(),
            self.generation_client.aclose(),
        )
