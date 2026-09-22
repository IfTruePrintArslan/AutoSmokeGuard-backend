"""Smoke feature extraction, density scoring and severity classification.

The segmenter says *where* smoke is; this module says *how bad* it is.  The
density score is a transparent, hand-weighted combination of five physically
motivated cues rather than a second learned model, so an assessor can read the
numbers off a report and understand why a vehicle was flagged.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Literal, Mapping, Sequence

import cv2
import numpy as np

logger = logging.getLogger("asg.ml")

Severity = Literal["low", "moderate", "high"]

#: Ordering used whenever severities are compared or aggregated.
SEVERITY_ORDER: tuple[str, ...] = ("low", "moderate", "high")
_SEVERITY_RANK = {name: i for i, name in enumerate(SEVERITY_ORDER)}

# --------------------------------------------------------------------------- #
# Density weights
#
# These sum to 1.0.  Each cue answers a different question about the plume and
# they were ordered by how strongly a roadworthiness inspector weighs them:
#
#   AREA      How much of the exhaust region is covered.  A plume that fills
#             the ROI is the single most damning signal, so it carries the most
#             weight -- but it saturates (see AREA_SATURATION) because beyond
#             roughly a third of the ROI the emission is already "gross".
#   OPACITY   Mean confidence of the segmenter inside the plume.  Thin, wispy
#             vapour scores low; a solid opaque column scores high.  This is
#             the classic opacity criterion used by Ringelmann / SAE J1667.
#   DARKNESS  How dark the plume is relative to white.  Black soot (unburnt
#             diesel) is a far worse emission than white steam, so darkness is
#             a real severity multiplier, not just an appearance cue.
#   EDGE      Edge density inside the mask.  Dense, churning smoke is
#             turbulent and textured; a flat grey blob is more likely to be a
#             shadow or a wall, so this term also suppresses false positives.
#   COMPACT   Compactness (how blob-like versus stringy the region is).  Real
#             plumes near the pipe are coherent masses; scattered speckle from
#             a noisy mask is not.  Smallest weight -- it is a sanity term.
# --------------------------------------------------------------------------- #
W_AREA = 0.34
W_OPACITY = 0.28
W_DARKNESS = 0.18
W_EDGE = 0.12
W_COMPACT = 0.08

#: ``area_ratio`` at/above which the area term is already maxed out.
AREA_SATURATION = 0.35
#: ``edge_density`` at/above which the texture term is maxed out.
EDGE_SATURATION = 0.18
#: Opacity below this contributes nothing (the segmenter is merely unsure).
OPACITY_FLOOR = 0.30
#: Luminance considered "fully bright" when computing darkness.
LUMA_MAX = 255.0

_EMPTY_FEATURES: dict[str, float] = {
    "area_ratio": 0.0,
    "mean_opacity": 0.0,
    "edge_density": 0.0,
    "darkness": 0.0,
    "compactness": 0.0,
    "largest_blob_ratio": 0.0,
    "largest_blob_pixels": 0.0,
    "mask_pixels": 0.0,
    "roi_pixels": 0.0,
}


def _as_prob_mask(mask: np.ndarray) -> np.ndarray:
    """Coerce any mask representation to ``float32`` probabilities in ``[0, 1]``."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32, copy=False)
    if arr.size and arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def extract_features(
    mask: np.ndarray,
    roi: np.ndarray,
    binarise_at: float = 0.5,
) -> dict[str, float]:
    """Measure the plume described by *mask* inside the image region *roi*.

    Args:
        mask: Smoke probability map, ``(H, W)`` in ``[0, 1]`` (``uint8`` 0-255
            is accepted and rescaled).  Must match *roi*'s spatial size; it is
            resized if it does not.
        roi: The BGR ``uint8`` exhaust region the mask was computed on.
        binarise_at: Probability threshold separating plume from background.

    Returns:
        ``{'area_ratio', 'mean_opacity', 'edge_density', 'darkness',
        'compactness', 'largest_blob_ratio', 'largest_blob_pixels',
        'mask_pixels', 'roi_pixels'}``.  All values are in ``[0, 1]`` except
        the three pixel counts.  Returns zeros for empty or mismatched input
        rather than raising.
    """
    if mask is None or roi is None or roi.size == 0:
        return dict(_EMPTY_FEATURES)

    prob = _as_prob_mask(mask)
    if prob.size == 0:
        return dict(_EMPTY_FEATURES)

    roi_h, roi_w = roi.shape[:2]
    if prob.shape[:2] != (roi_h, roi_w):
        prob = cv2.resize(prob, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

    roi_pixels = float(roi_h * roi_w)
    binary = (prob >= float(binarise_at)).astype(np.uint8)
    mask_pixels = float(int(binary.sum()))
    if mask_pixels < 1.0:
        empty = dict(_EMPTY_FEATURES)
        empty["roi_pixels"] = roi_pixels
        return empty

    area_ratio = mask_pixels / max(roi_pixels, 1.0)
    mean_opacity = float(prob[binary.astype(bool)].mean())

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = gray.astype(np.uint8, copy=False)

    # Edge density: Canny response confined to the plume, normalised by plume area.
    edges = cv2.Canny(gray, 50, 140)
    edge_pixels = float(int(np.count_nonzero(edges & (binary * 255))))
    edge_density = edge_pixels / mask_pixels

    # Darkness: 1 - normalised mean luminance inside the plume.
    mean_luma = float(gray[binary.astype(bool)].mean())
    darkness = 1.0 - (mean_luma / LUMA_MAX)

    # Compactness and coherence of the largest connected component.
    compactness, largest_blob_px = _largest_component_stats(binary)

    return {
        "area_ratio": round(float(np.clip(area_ratio, 0.0, 1.0)), 6),
        "mean_opacity": round(float(np.clip(mean_opacity, 0.0, 1.0)), 6),
        "edge_density": round(float(np.clip(edge_density, 0.0, 1.0)), 6),
        "darkness": round(float(np.clip(darkness, 0.0, 1.0)), 6),
        "compactness": round(float(np.clip(compactness, 0.0, 1.0)), 6),
        # Share of the ROI taken up by the single biggest blob.  A real plume
        # is one coherent mass; segmentation speckle is many tiny ones, and
        # the two are indistinguishable by total area alone.
        "largest_blob_ratio": round(float(np.clip(largest_blob_px / max(roi_pixels, 1.0), 0.0, 1.0)), 6),
        # Absolute size of that blob.  ``largest_blob_ratio`` is scale-free,
        # which is exactly its weakness: 15 pixels in a 112-pixel ROI is 13% of
        # the area and sails through every ratio gate.  The pipeline pairs the
        # ratio with a floor on this number -- see MIN_SMOKE_BLOB_PIXELS.
        "largest_blob_pixels": float(largest_blob_px),
        "mask_pixels": mask_pixels,
        "roi_pixels": roi_pixels,
    }


def _largest_component_stats(binary: np.ndarray) -> tuple[float, float]:
    """Compactness and pixel area of the biggest connected component.

    Isoperimetric compactness ``4*pi*A / P^2`` is 1.0 for a perfect disc and
    tends to 0 for stringy or speckled regions.

    Returns:
        ``(compactness, largest_component_pixels)``.
    """
    num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return 0.0, 0.0
    largest_px = float(stats[1:, cv2.CC_STAT_AREA].max())

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, largest_px
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    perimeter = float(cv2.arcLength(largest, closed=True))
    if perimeter <= 1e-6 or area <= 0.0:
        return 0.0, largest_px
    compactness = float(np.clip(4.0 * np.pi * area / (perimeter * perimeter), 0.0, 1.0))
    return compactness, largest_px


def smoke_density(features: Mapping[str, float]) -> float:
    """Combine plume features into a single severity-ready score in ``[0, 1]``.

    The score is ``W_AREA*area' + W_OPACITY*opacity' + W_DARKNESS*darkness +
    W_EDGE*edge' + W_COMPACT*compactness`` where the primed terms are the raw
    features passed through saturating / floored ramps (see the module-level
    weight documentation).  Saturation keeps a single extreme cue from pinning
    the score at 1.0 on its own.

    Args:
        features: The mapping returned by :func:`extract_features`.

    Returns:
        Density in ``[0, 1]``; ``0.0`` when there is no plume at all.
    """
    if not features or features.get("mask_pixels", 0.0) < 1.0:
        return 0.0

    area = float(features.get("area_ratio", 0.0))
    opacity = float(features.get("mean_opacity", 0.0))
    darkness = float(features.get("darkness", 0.0))
    edge = float(features.get("edge_density", 0.0))
    compact = float(features.get("compactness", 0.0))

    area_term = min(area / AREA_SATURATION, 1.0)
    # Rescale opacity so "barely above the decision threshold" scores ~0.
    opacity_term = np.clip((opacity - OPACITY_FLOOR) / (1.0 - OPACITY_FLOOR), 0.0, 1.0)
    edge_term = min(edge / EDGE_SATURATION, 1.0)

    score = (
        W_AREA * area_term
        + W_OPACITY * float(opacity_term)
        + W_DARKNESS * darkness
        + W_EDGE * edge_term
        + W_COMPACT * compact
    )
    return round(float(np.clip(score, 0.0, 1.0)), 6)


def classify_severity(
    density: float,
    low_max: float = 0.33,
    moderate_max: float = 0.66,
) -> Severity:
    """Bucket a density score into the product's three severity levels.

    Args:
        density: Score from :func:`smoke_density`.
        low_max: Upper bound (exclusive) of ``"low"``.
        moderate_max: Upper bound (exclusive) of ``"moderate"``.

    Returns:
        ``"low"``, ``"moderate"`` or ``"high"``.
    """
    value = float(np.clip(density, 0.0, 1.0))
    if value < float(low_max):
        return "low"
    if value < float(moderate_max):
        return "moderate"
    return "high"


# --------------------------------------------------------------------------- #
# Media-level severity: worst-case-present, not modal or mean
#
# This is an emission-enforcement evidence system, not a general-purpose
# summary statistic: the question a media-level verdict has to answer is "did
# this vehicle emit a high-severity plume at any point?", because that single
# event is the finding an inspector needs to act on.
#
# An earlier version of this function required a severity to occur in at
# least 10% of regions before it could win, falling back to the modal
# severity otherwise.  That rule averages the finding away: a truck that is
# genuinely high-severity for one frame out of twenty-six (3.8%) read as
# "moderate", which is a false reassurance -- exactly the failure mode this
# product exists to catch.  A mean/average density score has the identical
# problem for the same reason: it lets many clean frames dilute the one frame
# that matters.
#
# So ``overall_severity`` is simply the strongest severity present in *any*
# region, full stop.  There is deliberately no other rule left in this
# module -- see :func:`aggregate`, and :func:`analysis.worker._persist` on
# the database side, which must not recompute this differently.
# ``severity_counts`` still carries the full per-region distribution for the
# UI/report to show the spread; only the single-value verdict is worst-case.
# --------------------------------------------------------------------------- #


def aggregate(regions: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Roll per-region smoke findings up into a media-level verdict.

    Args:
        regions: Iterable of mappings, each with at least ``severity`` and
            optionally ``intensity`` / ``confidence``.

    Returns:
        ``{'severity_counts', 'overall_severity', 'mean_intensity',
        'mean_confidence', 'region_count'}``.  ``overall_severity`` is the
        single strongest severity present in *any* region (worst-case-present
        -- see the module-level comment above this function for why).  With
        no regions it is ``"low"`` and the counts are all zero.
    """
    items = [r for r in (regions or []) if r]
    counts = {name: 0 for name in SEVERITY_ORDER}
    if not items:
        return {
            "severity_counts": counts,
            "overall_severity": "low",
            "mean_intensity": 0.0,
            "mean_confidence": 0.0,
            "region_count": 0,
        }

    intensities: list[float] = []
    confidences: list[float] = []
    for region in items:
        severity = str(region.get("severity") or "low")
        if severity not in counts:
            severity = "low"
        counts[severity] += 1
        if region.get("intensity") is not None:
            intensities.append(float(region["intensity"]))
        if region.get("confidence") is not None:
            confidences.append(float(region["confidence"]))

    total = len(items)
    overall = "low"
    for name in reversed(SEVERITY_ORDER):  # high -> moderate -> low
        if counts[name] > 0:
            overall = name
            break

    return {
        "severity_counts": counts,
        "overall_severity": overall,
        "mean_intensity": round(float(np.mean(intensities)), 6) if intensities else 0.0,
        "mean_confidence": round(float(np.mean(confidences)), 6) if confidences else 0.0,
        "region_count": total,
    }


def severity_rank(severity: str) -> int:
    """Numeric rank of a severity label (``low``=0 ... ``high``=2)."""
    return _SEVERITY_RANK.get(str(severity), -1)
