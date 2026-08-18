"""End-to-end virtual try-on orchestration."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.clients.qwen_client import QwenClient
from app.clients.tryon_api_client import TryOnAPIClient
from app.core.config import Settings
from app.core.exceptions import InputValidationError
from app.models.request_models import (
    ORIGINAL_GARMENT_COLOR,
    PreparedTryOnRequest,
    TryOnRequest,
)
from app.models.result_models import CandidateResult, ColorResult, TryOnJobResult
from app.preprocessing.image_preprocessor import LocalImagePreprocessor
from app.preprocessing.preprocessing_exceptions import PreprocessingError
from app.preprocessing.preprocessing_models import PreprocessingResult
from app.prompts.clothing import ClothingPromptBuilder
from app.providers.factory import build_provider_bundle
from app.services.garment_analyzer import GarmentAnalyzer
from app.services.garment_colorizer import GarmentColorizer
from app.services.garment_segmenter import GarmentSegmenter
from app.services.input_validator import InputValidator
from app.services.output_evaluator import OutputEvaluator
from app.services.output_manager import OutputManager
from app.services.person_analyzer import PersonAnalyzer
from app.services.result_selector import ResultSelector
from app.services.retry_manager import RetryManager
from app.services.tryon_service import TryOnService
from app.repositories.prepared_garments import (
    FilesystemPreparedGarmentRepository,
    PreparedGarmentRepository,
)
from app.services.garment_preparation import GarmentPreparationService
from app.services.generation_policy import GenerationPolicy
from app.utils.file_utils import remove_tree, safe_slug
from app.utils.hashing import create_job_id
from app.utils.image_utils import normalize_color
from app.utils.json_utils import write_json
from app.utils.timing import log_stage_timing, timed_stage

LOGGER = logging.getLogger(__name__)


def color_slug(color: str) -> str:
    """Return friendly safe directory names for common colors and hex values."""

    canonical, _ = normalize_color(color)
    friendly = {
        "#C62828": "red",
        "#1565C0": "blue",
        "#000000": "black",
        "#FFFFFF": "white",
    }
    return friendly.get(canonical, safe_slug(canonical.lstrip("#"), "color"))


class VirtualTryOnPipeline:
    """Coordinate validation, analysis, processing, generation, and selection."""

    def __init__(
        self,
        *,
        settings: Settings,
        qwen_client: QwenClient,
        tryon_client: TryOnAPIClient,
        local_preprocessor: LocalImagePreprocessor | None = None,
        prepared_garment_repository: PreparedGarmentRepository | None = None,
    ) -> None:
        self.settings = settings
        self.qwen_client = qwen_client
        self.tryon_client = tryon_client
        self.local_preprocessor = local_preprocessor
        if self.local_preprocessor is None and settings.local_preprocessing_enabled:
            self.local_preprocessor = LocalImagePreprocessor(settings)
        self._preprocessing_semaphore = asyncio.Semaphore(
            settings.preprocessing_max_concurrency
        )
        self.validator = InputValidator(settings)
        self.person_analyzer = PersonAnalyzer(qwen_client)
        self.garment_analyzer = GarmentAnalyzer(qwen_client)
        self.segmenter = GarmentSegmenter()
        self.colorizer = GarmentColorizer()
        self.tryon_service = TryOnService(tryon_client)
        self.evaluator = OutputEvaluator(qwen_client, settings.min_acceptance_score)
        self.selector = ResultSelector(settings.min_acceptance_score)
        self.retry_manager = RetryManager()
        self.output_manager = OutputManager(settings.output_directory)
        self.prepared_garments = prepared_garment_repository or (
            FilesystemPreparedGarmentRepository(settings.prepared_garment_directory)
        )
        self.garment_preparation_service = GarmentPreparationService(
            settings=settings,
            repository=self.prepared_garments,
            preprocessor=(self.local_preprocessor or LocalImagePreprocessor(settings)),
        )

    async def prepare_garment(
        self,
        *,
        tenant_id: str,
        product_id: str,
        garment_image: Path,
        force: bool = False,
    ) -> Any:
        """Prepare and persist a tenant product independently of Try-On."""

        return await self.garment_preparation_service.prepare(
            tenant_id=tenant_id,
            product_id=product_id,
            garment_image=garment_image,
            force=force,
        )

    async def run_prepared(
        self,
        request: PreparedTryOnRequest,
    ) -> TryOnJobResult:
        """Run the product-id path without garment analysis or preprocessing."""

        total_started = time.perf_counter()
        started_at = datetime.now(UTC)
        policy = GenerationPolicy.from_settings(self.settings)
        job_id = create_job_id()
        job_directory = self.output_manager.create_job_directory(job_id)
        temp_directory = self.settings.temp_directory / job_id
        temp_directory.mkdir(parents=True, exist_ok=False)
        person_preprocessing = None
        prepared = None
        status = "failed"
        try:
            with timed_stage(
                LOGGER,
                pipeline="prepared_clothing",
                stage="validation",
                job_id=job_id,
                mode=policy.mode.value,
            ):
                person_image = self.validator.validate_image(
                    request.person_image,
                    role="person",
                )
            self.output_manager.write_request(job_directory, request)

            processing_person = person_image
            if self.settings.local_preprocessing_enabled:
                with timed_stage(
                    LOGGER,
                    pipeline="prepared_clothing",
                    stage="person_preprocessing",
                    job_id=job_id,
                    mode=policy.mode.value,
                ):
                    async with self._preprocessing_semaphore:
                        if self.local_preprocessor is None:
                            self.local_preprocessor = LocalImagePreprocessor(
                                self.settings
                            )
                        person_preprocessing = await asyncio.wait_for(
                            asyncio.to_thread(
                                self.local_preprocessor.preprocess_person,
                                person_image,
                                job_directory,
                            ),
                            timeout=self.settings.preprocessing_timeout_seconds,
                        )
                if not person_preprocessing.validation.accepted:
                    reasons = "; ".join(
                        person_preprocessing.validation.rejection_reasons
                    )
                    raise InputValidationError(
                        reasons or "Person image is unsuitable for virtual try-on."
                    )
                processing_person = person_preprocessing.normalized_image_path

            with timed_stage(
                LOGGER,
                pipeline="prepared_clothing",
                stage="prepared_garment_lookup",
                job_id=job_id,
                mode=policy.mode.value,
            ):
                prepared = self.prepared_garments.get(
                    request.tenant_id,
                    request.product_id,
                )
            variant_path = job_directory / "variants" / "original.png"
            self.output_manager.copy_artifact(
                prepared.processing.normalized_image_path,
                variant_path,
            )
            base_options: dict[str, Any] = {
                "product_title": request.product_title,
                "preserve_original_color": True,
                "preserve_face": True,
                "preserve_pose": True,
                "preserve_background": True,
            }
            if person_preprocessing is not None:
                if self.tryon_client.supports_mask:
                    base_options["replace_mask_path"] = str(
                        person_preprocessing.replace_mask_path
                    )
                if self.tryon_client.supports_text_prompt:
                    pose = person_preprocessing.pose
                    base_options["pose_hint"] = (
                        f"{pose.person_orientation}, arms {pose.arms_position}"
                    )

            candidates: list[CandidateResult] = []
            retry_count = 0
            with timed_stage(
                LOGGER,
                pipeline="prepared_clothing",
                stage="generation",
                job_id=job_id,
                mode=policy.mode.value,
            ):
                while True:
                    generated = await self.tryon_service.generate_candidates(
                        person_image=processing_person,
                        garment_image=variant_path,
                        category=request.category.value,
                        color=ORIGINAL_GARMENT_COLOR,
                        output_directory=(
                            job_directory / "candidates" / ORIGINAL_GARMENT_COLOR
                        ),
                        count=policy.candidates,
                        attempt=retry_count,
                        start_index=len(candidates) + 1,
                        options=self.retry_manager.options_for_attempt(
                            base_options,
                            retry_count,
                        ),
                    )
                    candidates.extend(generated)
                    if not policy.evaluate_outputs:
                        best = generated[0]
                        accepted = True
                        score = 1.0
                        problems: list[str] = []
                        break

                    evaluation_started = time.perf_counter()
                    for candidate in generated:
                        candidate.evaluation = await self.evaluator.evaluate(
                            processing_person,
                            variant_path,
                            candidate.path,
                            ORIGINAL_GARMENT_COLOR,
                            request.product_title,
                        )
                    log_stage_timing(
                        LOGGER,
                        pipeline="prepared_clothing",
                        stage="evaluation",
                        started=evaluation_started,
                        job_id=job_id,
                        mode=policy.mode.value,
                    )
                    best = self.selector.select_best(candidates)
                    accepted = self.selector.is_accepted(best)
                    if not self.retry_manager.should_retry(
                        accepted,
                        retry_count,
                        policy.max_retries,
                    ):
                        evaluation = best.evaluation
                        if evaluation is None:  # pragma: no cover
                            raise RuntimeError("Evaluated candidate has no score.")
                        score = evaluation.overall_score
                        problems = evaluation.problems
                        break
                    retry_count += 1

            final_path = job_directory / "final" / "original.png"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best.path, final_path)
            color_result = ColorResult(
                color=ORIGINAL_GARMENT_COLOR,
                output=final_path.relative_to(job_directory),
                score=score,
                accepted=accepted,
                retry_count=retry_count,
                candidates_evaluated=len(candidates),
                problems=problems,
            )
            self.output_manager.write_candidate_metadata(
                job_directory,
                candidates,
            )
            status = "completed" if accepted else "completed_with_failures"
            result = TryOnJobResult(
                job_id=job_id,
                tenant_id=request.tenant_id,
                pipeline="clothing",
                status=status,
                person_image=person_image,
                garment_image=prepared.processing.normalized_image_path,
                results=[color_result],
                product_id=request.product_id,
                category=request.category,
                mode=policy.mode,
                person_preprocessing=person_preprocessing,
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )
            self.output_manager.write_result(job_directory, result)
            return result
        except Exception as exc:
            failure = TryOnJobResult(
                job_id=job_id,
                tenant_id=request.tenant_id,
                pipeline="clothing",
                status="failed",
                person_image=request.person_image,
                garment_image=(
                    prepared.processing.normalized_image_path
                    if prepared is not None
                    else Path("unavailable")
                ),
                product_id=request.product_id,
                category=request.category,
                mode=policy.mode,
                person_preprocessing=person_preprocessing,
                error=type(exc).__name__,
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )
            self.output_manager.write_result(job_directory, failure)
            raise
        finally:
            log_stage_timing(
                LOGGER,
                pipeline="prepared_clothing",
                stage="total_request",
                started=total_started,
                status=status,
                job_id=job_id,
                mode=policy.mode.value,
            )
            if self.settings.delete_temp_files:
                remove_tree(temp_directory, self.settings.temp_directory)

    async def run(self, request: TryOnRequest) -> TryOnJobResult:
        """Execute one complete job and return its durable result."""

        started_at = datetime.now(UTC)
        validated = self.validator.validate(request)
        job_id = create_job_id()
        job_directory = self.output_manager.create_job_directory(job_id)
        temp_directory = self.settings.temp_directory / job_id
        temp_directory.mkdir(parents=True, exist_ok=False)
        self.output_manager.write_request(job_directory, validated)
        LOGGER.info(
            "job_started",
            extra={
                "job_id": job_id,
                "stage": "validate",
                "colors": len(validated.colors),
                "candidates_per_color": validated.candidates_per_color,
            },
        )
        person_analysis = None
        garment_analysis = None
        preprocessing = None
        processing_request = validated
        try:
            if self.settings.local_preprocessing_enabled:
                try:
                    preprocessing = await self.preprocess_inputs(
                        validated.person_image,
                        validated.garment_image,
                        job_directory,
                    )
                except (PreprocessingError, TimeoutError) as exc:
                    if not self.settings.preprocessing_fail_open:
                        raise
                    LOGGER.warning(
                        "local_preprocessing_fail_open",
                        extra={
                            "job_id": job_id,
                            "stage": "local_preprocessing",
                            "fallback_used": True,
                            "warning": type(exc).__name__,
                        },
                    )
                if preprocessing is not None:
                    person_valid = preprocessing.person.validation
                    garment_valid = preprocessing.garment.validation
                    if not person_valid.accepted or not garment_valid.accepted:
                        reasons = [
                            *person_valid.rejection_reasons,
                            *garment_valid.rejection_reasons,
                        ]
                        reason = "; ".join(reasons) or (
                            "Local preprocessing suitability score is below "
                            "the configured threshold."
                        )
                        result = TryOnJobResult(
                            job_id=job_id,
                            status="rejected",
                            person_image=validated.person_image,
                            garment_image=validated.garment_image,
                            preprocessing=preprocessing,
                            rejection_reason=reason,
                            started_at=started_at,
                            completed_at=datetime.now(UTC),
                        )
                        self.output_manager.write_result(job_directory, result)
                        LOGGER.info(
                            "job_rejected",
                            extra={
                                "job_id": job_id,
                                "stage": "local_preprocessing",
                                "accepted": False,
                                "degraded_mode": preprocessing.degraded_mode,
                            },
                        )
                        return result
                    processing_request = validated.model_copy(
                        update={
                            "person_image": (
                                preprocessing.person.normalized_image_path
                            ),
                            "garment_image": (
                                preprocessing.garment.normalized_image_path
                            ),
                        }
                    )
            if self.settings.person_analysis_enabled:
                person_analysis = await self.person_analyzer.analyze(
                    processing_request.person_image
                )
                write_json(job_directory / "person_analysis.json", person_analysis)
                if (
                    self.settings.reject_unsuitable_person_images
                    and not person_analysis.suitable_for_tryon
                ):
                    result = TryOnJobResult(
                        job_id=job_id,
                        status="rejected",
                        person_image=validated.person_image,
                        garment_image=validated.garment_image,
                        person_analysis=person_analysis,
                        rejection_reason=person_analysis.rejection_reason
                        or "Person image is unsuitable for virtual try-on.",
                        started_at=started_at,
                        completed_at=datetime.now(UTC),
                    )
                    self.output_manager.write_result(job_directory, result)
                    LOGGER.info(
                        "job_rejected",
                        extra={"job_id": job_id, "stage": "person_analysis"},
                    )
                    return result
                if not person_analysis.suitable_for_tryon:
                    LOGGER.warning(
                        "job_person_rejection_bypassed",
                        extra={
                            "job_id": job_id,
                            "stage": "person_analysis",
                            "rejection_reason": (person_analysis.rejection_reason),
                        },
                    )
            else:
                LOGGER.info(
                    "job_person_analysis_skipped",
                    extra={"job_id": job_id, "stage": "person_analysis"},
                )

            garment_analysis = await self.garment_analyzer.analyze(
                processing_request.garment_image
            )
            write_json(job_directory / "garment_analysis.json", garment_analysis)
            category = garment_analysis.recommended_tryon_category
            if preprocessing is not None:
                normalized = preprocessing.garment.normalized_image_path
                mask = preprocessing.garment.garment_mask_path
            else:
                normalized, mask = self.segmenter.segment(
                    processing_request.garment_image,
                    temp_directory,
                )
                self.output_manager.copy_artifact(
                    mask,
                    job_directory / "garment_mask.png",
                )

            all_candidates: list[CandidateResult] = []
            color_results: list[ColorResult] = []
            for requested_color in validated.colors:
                preserve_original_color = (
                    requested_color.casefold() == ORIGINAL_GARMENT_COLOR
                )
                if preserve_original_color:
                    canonical = ORIGINAL_GARMENT_COLOR
                    slug = ORIGINAL_GARMENT_COLOR
                else:
                    canonical, _ = normalize_color(requested_color)
                    slug = color_slug(requested_color)
                variant_path = job_directory / "variants" / f"{slug}.png"
                if preserve_original_color:
                    self.output_manager.copy_artifact(normalized, variant_path)
                else:
                    self.colorizer.create_variant(
                        normalized, mask, requested_color, variant_path
                    )
                result, candidates = await self._process_color(
                    request=processing_request,
                    job_id=job_id,
                    job_directory=job_directory,
                    variant_path=variant_path,
                    category=category,
                    color=canonical,
                    slug=slug,
                    preprocessing=preprocessing,
                )
                color_results.append(result)
                all_candidates.extend(candidates)

            self.output_manager.write_candidate_metadata(job_directory, all_candidates)
            status = (
                "completed"
                if all(item.accepted for item in color_results)
                else "completed_with_failures"
            )
            result = TryOnJobResult(
                job_id=job_id,
                status=status,
                person_image=validated.person_image,
                garment_image=validated.garment_image,
                person_analysis=person_analysis,
                garment_analysis=garment_analysis,
                preprocessing=preprocessing,
                results=color_results,
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )
            self.output_manager.write_result(job_directory, result)
            LOGGER.info(
                "job_completed",
                extra={
                    "job_id": job_id,
                    "stage": "complete",
                    "status": status,
                    "candidate_count": len(all_candidates),
                    "retry_count": sum(item.retry_count for item in color_results),
                },
            )
            return result
        except Exception as exc:
            failure = TryOnJobResult(
                job_id=job_id,
                status="failed",
                person_image=validated.person_image,
                garment_image=validated.garment_image,
                person_analysis=person_analysis,
                garment_analysis=garment_analysis,
                preprocessing=preprocessing,
                error=str(exc)[:1000],
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )
            try:
                self.output_manager.write_result(job_directory, failure)
            except Exception:
                LOGGER.error(
                    "failed_result_persistence_error",
                    extra={"job_id": job_id},
                    exc_info=LOGGER.isEnabledFor(logging.DEBUG),
                )
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.exception(
                    "job_failed", extra={"job_id": job_id, "stage": "pipeline"}
                )
            else:
                LOGGER.error(
                    "job_failed",
                    extra={
                        "job_id": job_id,
                        "stage": "pipeline",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            raise
        finally:
            if self.settings.delete_temp_files:
                remove_tree(temp_directory, self.settings.temp_directory)

    async def _process_color(
        self,
        *,
        request: TryOnRequest,
        job_id: str,
        job_directory: Path,
        variant_path: Path,
        category: str,
        color: str,
        slug: str,
        preprocessing: PreprocessingResult | None,
    ) -> tuple[ColorResult, list[CandidateResult]]:
        base_options: dict[str, Any] = {
            "product_title": request.product_title,
            "preserve_original_color": color == ORIGINAL_GARMENT_COLOR,
            "preserve_face": request.preserve_face,
            "preserve_pose": request.preserve_pose,
            "preserve_background": request.preserve_background,
        }
        if preprocessing is not None:
            if self.tryon_client.supports_mask:
                base_options["replace_mask_path"] = str(
                    preprocessing.person.replace_mask_path
                )
            if self.tryon_client.supports_text_prompt:
                pose = preprocessing.person.pose
                base_options["pose_hint"] = (
                    f"{pose.person_orientation}, arms {pose.arms_position}"
                )
        candidates: list[CandidateResult] = []
        retry_count = 0
        while True:
            options = self.retry_manager.options_for_attempt(base_options, retry_count)
            generated = await self.tryon_service.generate_candidates(
                person_image=request.person_image,
                garment_image=variant_path,
                category=category,
                color=color,
                output_directory=job_directory / "candidates" / slug,
                count=request.candidates_per_color,
                attempt=retry_count,
                start_index=len(candidates) + 1,
                options=options,
            )
            for candidate in generated:
                candidate.evaluation = await self.evaluator.evaluate(
                    request.person_image,
                    variant_path,
                    candidate.path,
                    color,
                    request.product_title,
                )
                LOGGER.info(
                    "candidate_evaluated",
                    extra={
                        "job_id": job_id,
                        "color": color,
                        "candidate_index": candidate.candidate_index,
                        "score": candidate.evaluation.overall_score,
                        "accepted": candidate.evaluation.accepted,
                    },
                )
            candidates.extend(generated)
            best = self.selector.select_best(candidates)
            accepted = self.selector.is_accepted(best)
            if not self.retry_manager.should_retry(
                accepted, retry_count, request.max_retries
            ):
                break
            retry_count += 1
            LOGGER.info(
                "quality_retry",
                extra={
                    "job_id": job_id,
                    "color": color,
                    "retry_count": retry_count,
                },
            )

        if best.evaluation is None:
            raise RuntimeError("Best candidate unexpectedly has no evaluation.")
        final_path = job_directory / "final" / f"{slug}.png"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best.path, final_path)
        relative_output = final_path.relative_to(job_directory)
        LOGGER.info(
            "best_candidate_selected",
            extra={
                "job_id": job_id,
                "color": color,
                "candidate_index": best.candidate_index,
                "score": best.evaluation.overall_score,
                "accepted": accepted,
            },
        )
        return (
            ColorResult(
                color=color,
                output=relative_output,
                score=best.evaluation.overall_score,
                accepted=accepted,
                retry_count=retry_count,
                candidates_evaluated=len(candidates),
                problems=best.evaluation.problems,
            ),
            candidates,
        )

    async def preprocess_inputs(
        self,
        person_image: Path,
        garment_image: Path,
        job_directory: Path,
        *,
        human_parsing_enabled: bool | None = None,
    ) -> PreprocessingResult:
        """Run bounded local preprocessing without blocking the event loop."""

        async with self._preprocessing_semaphore:
            if self.local_preprocessor is None:
                self.local_preprocessor = LocalImagePreprocessor(self.settings)
            operation = asyncio.to_thread(
                self.local_preprocessor.preprocess,
                person_image,
                garment_image,
                job_directory,
                human_parsing_enabled=human_parsing_enabled,
            )
            return await asyncio.wait_for(
                operation,
                timeout=self.settings.preprocessing_timeout_seconds,
            )

    async def warmup(self) -> None:
        """Prime expensive local models without changing generation behavior."""

        if (
            not self.settings.preprocessing_warmup_enabled
            or self.local_preprocessor is None
        ):
            return
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self.local_preprocessor.warmup),
                timeout=self.settings.preprocessing_timeout_seconds,
            )
        except Exception as exc:
            if self.settings.human_parsing_required:
                raise
            LOGGER.warning(
                "local_preprocessing_warmup_failed",
                extra={
                    "stage": "human_parsing_warmup",
                    "error_type": type(exc).__name__,
                    "warning": str(exc),
                },
            )

    async def aclose(self) -> None:
        """Close both injected clients."""

        await self.qwen_client.aclose()
        await self.tryon_client.aclose()


def build_pipeline(
    settings: Settings,
    *,
    qwen_client: QwenClient | None = None,
    tryon_client: TryOnAPIClient | None = None,
    prompt_builder: ClothingPromptBuilder | None = None,
) -> VirtualTryOnPipeline:
    """Build a pipeline once per process or application lifespan."""

    providers = build_provider_bundle(
        settings,
        prompt_builder=prompt_builder,
        analysis_client=qwen_client,
        generation_client=tryon_client,
    )
    return VirtualTryOnPipeline(
        settings=settings,
        qwen_client=providers.analysis,
        tryon_client=providers.generation,
    )
