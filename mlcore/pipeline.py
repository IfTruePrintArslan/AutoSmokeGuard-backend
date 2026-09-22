"""End-to-end smoke-emission analysis orchestration.

:func:`analyze_media` is the single entry point the web layer needs: hand it a
file path, an output directory and an :class:`~mlcore.config.MLConfig`, and it
returns a JSON-serialisable result dict plus annotated artifacts on disk.

Every path in the returned dict is **relative to** ``output_dir`` so the caller
can join it onto ``MEDIA_ROOT`` (or an S3 prefix) without knowing anything
about this package's layout.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .annotate import draw, save_frame
from .config import MLConfig, device_string
from .detector import VehicleDetector
from .intensity import aggregate, classify_severity, extract_features, severity_rank, smoke_density
from .preprocess import MediaError, crop as crop_region, enhance, iter_frames, load_image, media_kind, probe_video
from .segmenter import SmokeSegmenter

logger = logging.getLogger("asg.ml")

ProgressCallback = Callable[[int, str], None]

__all__ = ["analyze_media", "warmup", "MLPipelineError"]


class MLPipelineError(RuntimeError):
    """Unrecoverable problem with the *input* to the pipeline.

    Raised for a missing/undecodable file or an unusable output directory.
    Per-frame failures never raise -- they are logged and skipped.
    """


# --------------------------------------------------------------------------- #
# Detection floors
#
# A segmenter always returns *some* probability mass, so three floors decide
# whether a response is reported as an emission.  All three must be cleared.
#
#   MIN_SMOKE_AREA_RATIO   Total thresholded plume area, as a fraction of the
#       exhaust ROI.  Filters out the handful of stray pixels every model
#       produces on a hard crop.
#
#   MIN_SMOKE_BLOB_RATIO   Area of the *largest connected component*, same
#       units.  This is the one that actually does the work.  Measured over
#       the bundled sample media, false activations on clean vehicles are
#       scattered speckle along the bumper line -- individually tiny, but they
#       can sum to more total area than a genuine plume occupies in the very
#       large ROI of a bus.  Real smoke is one coherent mass.  The measured
#       split on the sample media is:
#           clean vehicles  : largest blob <= 0.0169 of the ROI (49 ROIs)
#           smoking vehicles: largest blob  = 0.0207 (bus) .. 0.153 (truck)
#       so the floor sits at 0.018, between the two. The margin on the bus
#       side is modest; it is a calibration point, not a law of nature, and
#       should be re-measured if the segmenter is retrained.
#
#   MIN_SMOKE_DENSITY      Combined density score, i.e. roughly "half of the
#       low severity band".  Rejects a plume that is large but utterly
#       transparent, textureless and pale.
# --------------------------------------------------------------------------- #
MIN_SMOKE_AREA_RATIO = 0.02
MIN_SMOKE_BLOB_RATIO = 0.018
MIN_SMOKE_DENSITY = 0.15

#: Annotated frames written to ``frames/``.
MAX_ANNOTATED_FRAMES = 12
# Artifacts are written *during* the scan (so no video frame is retained after
# it has been processed), which means the caps must be simple running budgets
# rather than a global ranking.  Smoking vehicles get their own, larger budget
# so a long clean stretch cannot starve the interesting detections.
#: Mask PNGs written to ``masks/``.
MAX_MASK_ARTIFACTS = 150
#: Crop JPEGs written to ``crops/`` for *smoking* vehicles.
MAX_SMOKE_CROP_ARTIFACTS = 150
#: Crop JPEGs written to ``crops/`` for clean vehicles.
MAX_CLEAN_CROP_ARTIFACTS = 80


def _emit(progress_cb: ProgressCallback | None, percent: int, stage: str) -> None:
    """Invoke *progress_cb* defensively -- a broken callback must not abort a run."""
    if progress_cb is None:
        return
    try:
        progress_cb(int(percent), str(stage))
    except Exception as exc:  # noqa: BLE001
        logger.warning("progress_cb(%s, %r) raised %s; continuing.", percent, stage, exc)


def _relative(path: Path, root: Path) -> str:
    """POSIX-style path of *path* relative to *root* (falls back to the name)."""
    try:
        return Path(os.path.relpath(path, root)).as_posix()
    except ValueError:  # pragma: no cover - different drives on Windows
        return path.name


def _frame_interest(vehicles: Sequence[Mapping[str, Any]]) -> tuple[int, int, int, float]:
    """Sort key for "how worth saving is this frame".

    Ordered by worst severity present, then number of smoking vehicles, then
    number of vehicles, then total confidence.
    """
    worst = -1
    smoking = 0
    confidence = 0.0
    for vehicle in vehicles:
        confidence += float(vehicle.get("confidence") or 0.0)
        smoke = vehicle.get("smoke")
        if isinstance(smoke, Mapping):
            smoking += 1
            worst = max(worst, severity_rank(str(smoke.get("severity", "low"))))
    return (worst, smoking, len(vehicles), round(confidence, 4))


class _FrameAnalyzer:
    """Holds the per-run models and turns one frame into vehicle records."""

    def __init__(self, config: MLConfig) -> None:
        self.config = config
        self.detector = VehicleDetector(config)
        self.segmenter = SmokeSegmenter(config)

    def process(self, frame_bgr: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
        """Detect vehicles in *frame_bgr* and segment smoke under each one.

        Returns:
            ``(vehicles, working_frame)``.  ``working_frame`` is the enhanced
            frame actually used for inference (annotation draws on it so the
            artifact matches what the model saw).
        """
        working = enhance(frame_bgr) if self.config.enhance else frame_bgr
        detections = self.detector.detect(working)

        # Crop every exhaust ROI first, then segment them in one batched
        # forward pass -- see SmokeSegmenter.segment_many.
        roi_boxes = [self.detector.exhaust_roi(d["bbox"], working.shape) for d in detections]
        rois = [crop_region(working, box) for box in roi_boxes]
        segmented = self.segmenter.segment_many(rois)

        vehicles: list[dict[str, Any]] = []
        for detection, roi_box, roi, (mask, seg_confidence) in zip(detections, roi_boxes, rois, segmented):
            record: dict[str, Any] = dict(detection)
            record["smoke"] = None
            try:
                if roi.size == 0:
                    vehicles.append(record)
                    continue

                features = extract_features(mask, roi, binarise_at=self.config.mask_threshold)
                density = smoke_density(features)

                if (
                    features["area_ratio"] >= MIN_SMOKE_AREA_RATIO
                    and features["largest_blob_ratio"] >= MIN_SMOKE_BLOB_RATIO
                    and density >= MIN_SMOKE_DENSITY
                ):
                    severity = classify_severity(
                        density, self.config.severity_low_max, self.config.severity_moderate_max
                    )
                    record["smoke"] = {
                        "mask": mask,
                        "roi": roi_box,
                        "intensity": density,
                        "severity": severity,
                        "confidence": round(float(seg_confidence), 6),
                        "area_ratio": features["area_ratio"],
                        "opacity": features["mean_opacity"],
                        "features": features,
                    }
            except Exception as exc:  # noqa: BLE001 - one bad ROI must not kill the frame
                logger.warning("Smoke analysis failed for one vehicle: %s", exc)
            vehicles.append(record)

        return vehicles, working


def analyze_media(
    media_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    config: MLConfig | None = None,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Analyse an image or a video for vehicle smoke emissions.

    Args:
        media_path: Image or video to analyse.
        output_dir: Directory for artifacts.  ``frames/``, ``masks/`` and
            ``crops/`` subdirectories are created inside it.
        config: Pipeline configuration; defaults are used when omitted.
        progress_cb: Optional ``(percent, stage)`` callback.  It is invoked at
            least at 5 ``loading``, 15 ``detecting``, 40 ``segmenting``,
            75 ``aggregating``, 90 ``writing artifacts`` and 100 ``done``.
            Exceptions raised by the callback are logged and swallowed.

    Returns:
        A JSON-serialisable result dict (see the module docstring of the
        project spec).  All paths are relative to *output_dir*.

    Raises:
        MLPipelineError: The media could not be opened/decoded, or the output
            directory could not be created.
    """
    started = time.time()
    cfg = (config or MLConfig()).validate()

    source = Path(media_path)
    out_root = Path(output_dir)
    try:
        (out_root / "frames").mkdir(parents=True, exist_ok=True)
        (out_root / "masks").mkdir(parents=True, exist_ok=True)
        (out_root / "crops").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MLPipelineError(f"Cannot create output directory {out_root}: {exc}") from exc

    if not source.is_file():
        raise MLPipelineError(f"Media file not found: {source}")

    kind = media_kind(source)
    if kind == "unknown":
        # Fall back to sniffing: try image decode, then video open.
        kind = "image"
        try:
            load_image(source)
        except MediaError:
            kind = "video"

    _emit(progress_cb, 5, "loading")
    analyzer = _FrameAnalyzer(cfg)

    media_meta: dict[str, Any] = {"kind": kind, "filename": source.name, "size_bytes": source.stat().st_size}
    frames: Iterable[tuple[int, float, np.ndarray]]

    if kind == "image":
        try:
            frame = load_image(source)
        except MediaError as exc:
            raise MLPipelineError(str(exc)) from exc
        media_meta.update(width=int(frame.shape[1]), height=int(frame.shape[0]), fps=None, frame_count=1, duration=0.0)
        frames = [(0, 0.0, frame)]
        expected_frames = 1
    else:
        try:
            probe = probe_video(source)
        except MediaError as exc:
            raise MLPipelineError(str(exc)) from exc
        media_meta.update(probe)
        frames = iter_frames(
            source,
            sample_rate=cfg.frame_sample_rate,
            max_frames=cfg.max_frames,
            max_seconds=float(cfg.max_video_seconds),
        )
        sampled = int(probe["frame_count"] or 0) // max(1, cfg.frame_sample_rate)
        expected_frames = max(1, min(cfg.max_frames, sampled or cfg.max_frames))

    _emit(progress_cb, 15, "detecting")

    # -- main loop --------------------------------------------------------- #
    all_vehicles: list[dict[str, Any]] = []
    smoke_regions: list[dict[str, Any]] = []
    # (interest_key, frame_index, timestamp, annotated_bgr)
    best_frames: list[tuple[tuple[int, int, int, float], int, float, np.ndarray]] = []
    frames_processed = 0
    detection_confidences: list[float] = []
    segmenting_announced = False
    budget = _ArtifactBudget()

    for frame_index, timestamp, raw_frame in frames:
        try:
            vehicles, working = analyzer.process(raw_frame)
        except Exception as exc:  # noqa: BLE001 - never let one frame kill the run
            logger.warning("Frame %s failed and was skipped: %s", frame_index, exc)
            continue

        frames_processed += 1

        for vehicle in vehicles:
            record: dict[str, Any] = {
                "vehicle_type": vehicle["vehicle_type"],
                "bounding_box": vehicle["bbox"],
                "confidence": vehicle["confidence"],
                "frame_number": int(frame_index),
                "timestamp_seconds": float(timestamp),
                "crop_path": None,
                "smoke": None,
            }
            detection_confidences.append(float(vehicle["confidence"]))
            smoke = vehicle.get("smoke")
            if isinstance(smoke, Mapping):
                record["smoke"] = {
                    "mask_path": None,
                    "intensity": smoke["intensity"],
                    "severity": smoke["severity"],
                    "confidence": smoke["confidence"],
                    "area_ratio": smoke["area_ratio"],
                    "opacity": smoke["opacity"],
                }
                smoke_regions.append(record["smoke"])
            _write_vehicle_artifacts(record, vehicle, working, out_root, len(all_vehicles), budget)
            all_vehicles.append(record)

        if not segmenting_announced:
            _emit(progress_cb, 40, "segmenting")
            segmenting_announced = True
        elif expected_frames > 1:
            # 40 -> 72 across the body of the run.
            pct = 40 + int(32 * min(1.0, frames_processed / float(expected_frames)))
            _emit(progress_cb, pct, "segmenting")

        # Only pay for annotation when the frame can still make the shortlist.
        interest = _frame_interest(vehicles)
        if vehicles and (len(best_frames) < MAX_ANNOTATED_FRAMES or interest > best_frames[-1][0]):
            try:
                annotated = draw(
                    working,
                    vehicles,
                    frame_number=frame_index,
                    timestamp=timestamp if kind == "video" else None,
                    mask_threshold=cfg.mask_threshold,
                )
                best_frames.append((interest, int(frame_index), float(timestamp), annotated))
                # Sort worst-severity-first; ties break toward the earlier frame.
                best_frames.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                del best_frames[MAX_ANNOTATED_FRAMES:]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Annotation failed for frame %s: %s", frame_index, exc)

        # The frame (and every mask taken from it) is no longer needed.
        del raw_frame, working, vehicles

    if not segmenting_announced:
        _emit(progress_cb, 40, "segmenting")

    # -- aggregation ------------------------------------------------------- #
    _emit(progress_cb, 75, "aggregating")
    summary = aggregate(smoke_regions)

    # -- artifacts --------------------------------------------------------- #
    _emit(progress_cb, 90, "writing artifacts")

    annotated_paths: list[str] = []
    preview_path: str | None = None
    for rank, (_interest, frame_index, _timestamp, annotated) in enumerate(best_frames):
        target = out_root / "frames" / f"frame_{frame_index:06d}.jpg"
        try:
            save_frame(annotated, target)
        except OSError as exc:
            logger.warning("Could not write annotated frame %s: %s", target, exc)
            continue
        rel = _relative(target, out_root)
        annotated_paths.append(rel)
        if rank == 0:
            preview_path = rel

    # `best_frames` is already sorted worst-severity-first, so the first saved
    # frame is the highest-severity one; with no smoke anywhere the sort key
    # degrades to "most vehicles", which is exactly the required fallback.
    if preview_path and annotated_paths:
        try:
            preview_target = out_root / "preview.jpg"
            save_frame(best_frames[0][3], preview_target)
            preview_path = _relative(preview_target, out_root)
        except OSError as exc:
            logger.warning("Could not write preview: %s", exc)

    elapsed = time.time() - started
    result: dict[str, Any] = {
        "total_vehicles": len(all_vehicles),
        "total_smoke": len(smoke_regions),
        "frames_processed": frames_processed,
        "avg_confidence": round(float(np.mean(detection_confidences)), 6) if detection_confidences else 0.0,
        "overall_severity": summary["overall_severity"] if smoke_regions else "none",
        "severity_counts": summary["severity_counts"],
        "mean_intensity": summary["mean_intensity"],
        "preview_path": preview_path,
        "annotated_frames": annotated_paths,
        "vehicles": all_vehicles,
        "segmenter_mode": analyzer.segmenter.resolved_mode(),
        "device": device_string(cfg.torch_device()),
        "elapsed_seconds": round(elapsed, 3),
        "media_meta": media_meta,
    }
    _emit(progress_cb, 100, "done")
    logger.info(
        "Analysed %s: %d frame(s), %d vehicle(s), %d smoke region(s), severity=%s in %.2fs (%s).",
        source.name, frames_processed, result["total_vehicles"], result["total_smoke"],
        result["overall_severity"], elapsed, result["segmenter_mode"],
    )
    return result


class _ArtifactBudget:
    """Running counters that cap how many files one analysis may emit."""

    __slots__ = ("masks", "smoke_crops", "clean_crops")

    def __init__(self) -> None:
        self.masks = 0
        self.smoke_crops = 0
        self.clean_crops = 0


def _write_vehicle_artifacts(
    record: dict[str, Any],
    detection: Mapping[str, Any],
    frame: np.ndarray,
    out_root: Path,
    index: int,
    budget: "_ArtifactBudget",
) -> None:
    """Write this vehicle's crop JPEG and (if it smokes) its mask PNG.

    Called while the frame is still in hand so nothing has to be retained
    after the scan moves on.  Failures are logged and leave the corresponding
    path as ``None`` -- an unwritable artifact must not lose the measurement.
    """
    frame_index = int(record["frame_number"])
    smoke = record.get("smoke")
    smoking = smoke is not None

    if smoking:
        allowed = budget.smoke_crops < MAX_SMOKE_CROP_ARTIFACTS
    else:
        allowed = budget.clean_crops < MAX_CLEAN_CROP_ARTIFACTS
    if allowed:
        try:
            patch = crop_region(frame, record["bounding_box"])
            if patch.size:
                target = out_root / "crops" / f"v{index:05d}_f{frame_index:06d}.jpg"
                save_frame(patch, target, quality=88)
                record["crop_path"] = _relative(target, out_root)
                if smoking:
                    budget.smoke_crops += 1
                else:
                    budget.clean_crops += 1
        except (OSError, KeyError, cv2.error) as exc:
            logger.warning("Could not write vehicle crop for frame %s: %s", frame_index, exc)

    if smoking and budget.masks < MAX_MASK_ARTIFACTS:
        raw = (detection.get("smoke") or {}).get("mask")
        if raw is None:
            return
        try:
            mask_u8 = np.clip(np.asarray(raw, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
            target = out_root / "masks" / f"m{index:05d}_f{frame_index:06d}.png"
            save_frame(mask_u8, target)
            smoke["mask_path"] = _relative(target, out_root)
            budget.masks += 1
        except (OSError, cv2.error) as exc:
            logger.warning("Could not write smoke mask for frame %s: %s", frame_index, exc)


def warmup(config: MLConfig | None = None) -> dict[str, Any]:
    """Preload the detector and segmenter so the first request is not slow.

    Call this once from the worker's startup hook.

    Args:
        config: Pipeline configuration; defaults are used when omitted.

    Returns:
        ``{'device', 'detector_ready', 'segmenter_ready', 'segmenter_mode',
        'elapsed_seconds'}``.
    """
    started = time.time()
    cfg = (config or MLConfig()).validate()

    detector = VehicleDetector(cfg)
    segmenter = SmokeSegmenter(cfg)
    detector_ready = detector.warmup()
    segmenter_ready = segmenter.warmup()

    info = {
        "device": device_string(cfg.torch_device()),
        "detector_ready": bool(detector_ready),
        "segmenter_ready": bool(segmenter_ready),
        "segmenter_mode": segmenter.mode,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    logger.info("Warmup complete: %s", info)
    return info
