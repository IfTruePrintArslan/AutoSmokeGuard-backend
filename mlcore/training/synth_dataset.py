"""Synthetic vehicle-exhaust smoke segmentation dataset.

Why this exists
---------------
There is no openly downloadable, pixel-labelled dataset of vehicle exhaust
smoke.  Hand-labelling one was out of scope for this project, so instead we
*synthesise* supervision: real photographs supply the backgrounds (including
the exact geometry the inference pipeline sees -- the exhaust ROI under a
detected vehicle), and a physically-motivated plume model supplies the smoke
and its exact ground-truth alpha.

Realism levers
--------------
* **Backgrounds are real.**  Crops come from the COCO128 photo set.  Where a
  photo carries a COCO vehicle label (classes 2/3/5/7) the crop is the
  *exhaust ROI* under that vehicle -- the same lower-band, side-expanded,
  aspect-squashed region ``VehicleDetector.exhaust_roi`` hands the segmenter
  at inference time.  This removes the usual synthetic-to-real geometry gap.
* **Plumes are fractal.**  A 5-octave fractal-Brownian-motion field is shaped
  by an anisotropic cone that widens with distance, curved, sheared, then
  pushed through a turbulence displacement field.
* **Opacity is calibrated, not guessed.**  The plume's global opacity is
  solved so the resulting mask hits a sampled target coverage, which keeps the
  label statistics under control while still varying strongly per sample.
* **Hard negatives.**  Two fifths of the set has an all-zero mask, and most of
  those still contain something smoke-*like*: brown dust clouds, physically
  modelled cast shadows, real mined exhaust ROIs, or motion blur.  Without
  these the network learns "any soft grey blob is smoke".

Background images are split disjointly between ``train`` and ``val`` so the
validation score is not inflated by memorised backgrounds.

Why the negatives look the way they do (generator 2.0.0)
--------------------------------------------------------
Generator 1.4.0 trained a model whose real-world behaviour was measured in
``docs/REAL_DATA_EVALUATION.md``: **11.1% of real clean vehicles (14 of 126)
were reported as emitting, and all 14 were the vehicle's own cast shadow or a
dark low-texture surface.**  Diagnosing that against this file gave an answer
that was *not* the obvious one, so it is recorded here in full.

**What was not wrong.**  The old ``apply_shadow`` was already multiplicative
(``img * (1 - mask*k)``), and measuring its output confirms it behaved as a
shadow should: normalised local contrast ``std/mean`` inside the darkened
region came out at **1.066** of the original, i.e. texture and hue survived.
"The generator renders shadows as a flat grey blend that destroys texture"
is a reasonable hypothesis and it is false.

**What was wrong: the shadow negatives were far too weak to matter.**
Measured over 200 samples of each, against the real shadows the deployed
model actually fired on:

===========================  ==============  ==============  ===============
statistic                    1.4.0 negative  2.0.0 negative  real (measured)
===========================  ==============  ==============  ===============
luminance ratio in/out       0.763           0.515           0.16 - 0.77
   ... worst case            0.596           0.192           0.16
in-mask luminance            78.7            51.7            19.3 - 74.1
``std/mean`` ratio in/out    1.066           1.345           0.818 - 1.663
===========================  ==============  ==============  ===============

A 1.4.0 shadow negative bottomed out at 60% of the original brightness and
averaged an in-mask luminance of **78.7**.  Meanwhile ``SMOKE_PALETTE['black']``
plus :data:`MIN_SMOKE_CONTRAST` -- the colour search keeps whichever candidate
sits *furthest* from the local background, so on a bright road the winner is
almost always near-black -- produced *positives* averaging **64.9**, 95.4% of
them darker than their surround.

The two classes therefore did not overlap in brightness: **the shadow
negatives sat in a brighter band than the black smoke positives.**  A
luminance threshold around 70 separated them almost perfectly on the training
set; the network found it; and on real imagery -- where cast shadows reach
luminance 19 and routinely go *below* any plume -- that rule inverts and fires
on every sunlit vehicle.  The negatives were neither absent nor physically
wrong.  They simply never occupied the region of the space where the decision
actually gets made.

The fix is not to stop generating black smoke -- unburnt diesel really is
black, and it is the emission the product exists to catch.  The fix is to make
the negatives **as dark as the positives are**, so that brightness carries no
information and the network is forced onto a cue that genuinely distinguishes
the two.  That cue exists and is physical:

* A cast shadow is **multiplicative**: ``I -> I * (1 - k)``.  It scales the
  local mean and the local standard deviation together, so *normalised* local
  contrast ``std/mean`` is preserved.  Texture and hue survive.
* Smoke is **additive airlight**: ``I -> I*(1-a) + C*a``.  It pulls every
  pixel toward one colour, so ``std/mean`` collapses.

On real imagery that separation is clean: ``std/mean`` in/out measured
**0.584** for the one genuine tailpipe plume against **0.818 - 1.663** across
the eight inspected false positives.  :func:`apply_cast_shadow` reproduces the
shadow side of it (1.345 measured) and :func:`_blend` the smoke side (0.661
measured over 2.0.0's own positives), so for the first time the training set
contains the distinction the real decision turns on.  A third of positives are
additionally composited **on top of** a cast shadow, which puts dark-and-smoke
pixels and dark-and-background pixels in the same tile and makes a luminance
shortcut unavailable rather than merely unrewarding.

CLI::

    python -m mlcore.training.synth_dataset --train 2800 --val 700 --seed 1337
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..config import COCO128_IMAGES_DIR, COCO128_LABELS_DIR, SYNTH_DATASET_DIR

logger = logging.getLogger("asg.ml")

#: Bumped whenever the generator changes in a way that invalidates a cached set.
GENERATOR_VERSION = "2.0.0"

#: Output tile size (matches ``MLConfig.input_size``).
TILE = 256
# Alpha above which a pixel counts as smoke in the ground-truth mask.
#
# This number is the single most important knob in the whole generator.  A
# pixel blended at alpha `a` over a background differs from that background by
# roughly `a * |smoke_colour - background|` grey levels.  With the plume/
# background contrast floor below (MIN_SMOKE_CONTRAST) a cutoff of 0.35 puts
# the *label boundary* at ~20 grey levels of change -- comfortably above the
# sensor noise and JPEG artefacts we then add, so the boundary is something a
# network can actually see.  An earlier 0.15 cutoff produced boundaries around
# 7 grey levels, i.e. below the noise floor, which capped achievable Dice no
# matter how long the model trained.  It is also the more honest label: a 15%
# veil is not something a human annotator would outline as smoke.
MASK_ALPHA_CUTOFF = 0.35
#: COCO class ids that count as vehicles.
VEHICLE_CLASS_IDS = frozenset({2, 3, 5, 7})
# The ground-truth outline is taken from a *smoothed* copy of the alpha field.
#
# The composited image keeps every octave of fractal texture, but the label is
# the smooth contour enclosing the dense body of the plume -- which is exactly
# what a human annotator draws.  Thresholding the raw alpha instead produced a
# pixel-level fractal boundary whose exact wiggles are only recoverable by
# solving an alpha-matting problem, and that unpredictable component put a
# hard ceiling on achievable Dice without making the label any more useful.
#: Gaussian sigma (in pixels at a 256 px tile) used to smooth the label field.
LABEL_SMOOTH_SIGMA = 5.0
# Luminance below which a pixel counts as "dark" for the darkness/label
# balance audit.  80 is where the measured real false positives lived: the
# eight inspected in ``REAL_DATA_EVALUATION.md`` had in-mask luminance 19-74,
# while the one genuine real plume sat at 115.
DARK_LUMA = 80

# Fraction of samples that are negatives (all-zero mask).
#
# Raised from 0.25 (generator 1.4.0) to 0.34, and the *composition* matters
# more than the total.  Measured over 1600 generated tiles, counting coherent
# dark regions (luminance < 80, 8-connected, area > 2% of the tile) and asking
# how often such a region carries a smoke label:
#
#     generator        dark regions   labelled smoke   background per smoke
#     1.4.0                   3135              228                  11.95
#     2.0.0                   3260              148                  20.16
#
# P(smoke | coherent dark region) falls 0.0727 -> 0.0454.  That 1.6x is worth
# having but it is not the main event -- see the module docstring: the
# decisive change is that the dark negatives now reach the *same luminance*
# as the dark positives (51.7 vs 64.9 mean in-mask) instead of sitting in a
# separate, brighter band (78.7) that a threshold could cleanly split off.
# Raising the count without fixing the strength would not have worked.
#
# The total was first tried at 0.42.  That trained a model which fixed the
# shadow failure completely (real surveillance false positives 18.18% -> 0%)
# but went too quiet to be useful: its peak probability on the bundled
# `sample_bus_smoking.jpg` plume fell from 0.993 to 0.572, below the decision
# threshold, so a genuine plume went unreported.  0.34 keeps the negatives
# well above 1.4.0's 0.25 without pushing the prior that far toward silence.
# The lesson is worth recording: the fix is the negatives' *realism and
# strength*, not their headcount, and headcount overshoots quickly.
#
# :func:`build_split` recomputes the balance from the imagery it actually
# wrote and records it in the manifest under ``darkness_balance``, so this is
# checkable rather than asserted.
NEGATIVE_FRACTION = 0.34
#: Fraction of *positive* samples that also get a distractor pasted in.
POSITIVE_DISTRACTOR_FRACTION = 0.20
# Fraction of *positive* samples composited on top of a cast shadow, with the
# plume deliberately allowed to overlap it.  This is the single most important
# knob for breaking the darkness/label correlation: it produces pixels that are
# dark *and* labelled smoke and pixels that are dark *and* labelled background
# inside the same tile, so the network cannot settle for a luminance rule.
POSITIVE_ON_SHADOW_FRACTION = 0.34
#: Share of backgrounds reserved for the validation split.
VAL_BACKGROUND_FRACTION = 0.20
# Share of tiles pushed through an inference-like CLAHE.  Not 1.0: the
# product can be configured with ``enhance=False``, and the network should
# cope with both.  See :func:`clahe_jitter`.
CLAHE_AUGMENT_FRACTION = 0.55

#: Relative weights for how a background crop is sourced.
SOURCE_WEIGHTS = {"exhaust_roi": 0.52, "random_crop": 0.38, "procedural": 0.10}

# Negative-sample composition (must sum to 1.0).
#
#   cast_shadow  Physically-correct multiplicative cast shadow, perspective
#                quad with a penumbra, anchored at a vehicle contact line
#                inside the ROI.  The direct answer to the measured failure.
#   mined_real   A real exhaust-ROI (or real shadow-bearing) crop from COCO128,
#                unmodified, with an all-zero mask.  Real pixels with a
#                guaranteed-correct label -- the most valuable negative there
#                is, and the one the evaluation asked for first.
#   dust         Brown road dust: smoke-shaped but wrong colour.
#   shadow       The old soft-ellipse darkening.  Kept because it covers the
#                diffuse/overcast case the hard quad does not.
#   motion_blur  Smeared edges that read like haze.
#   plain        An untouched background, so "nothing here" is also taught.
NEGATIVE_KINDS = {
    "cast_shadow": 0.34,
    "mined_real": 0.22,
    "dust": 0.16,
    "shadow": 0.10,
    "motion_blur": 0.10,
    "plain": 0.08,
}


# --------------------------------------------------------------------------- #
# Noise / plume primitives
# --------------------------------------------------------------------------- #

def fbm(
    rng: np.random.Generator,
    size: int = TILE,
    octaves: int = 5,
    persistence: float = 0.55,
    base_freq: int = 4,
) -> np.ndarray:
    """Fractal-Brownian-motion value noise in ``[0, 1]``.

    Built by summing ``octaves`` layers of bicubically upsampled low-resolution
    white noise, each at double the frequency and ``persistence`` times the
    amplitude of the last.  This is the cheapest way to get the self-similar,
    billowing structure that makes a plume read as smoke rather than as a blob.

    Args:
        rng: Seeded generator (determinism comes from here).
        size: Output side length.
        octaves: Number of noise layers.
        persistence: Amplitude decay per octave.
        base_freq: Grid resolution of the first (coarsest) octave.

    Returns:
        ``(size, size)`` ``float32`` array normalised to ``[0, 1]``.
    """
    field = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0
    total = 0.0
    freq = int(base_freq)
    for _ in range(max(1, octaves)):
        grid = max(2, int(freq))
        low = rng.random((grid, grid), dtype=np.float32)
        field += amplitude * cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)
        total += amplitude
        amplitude *= persistence
        freq *= 2
    field /= max(total, 1e-6)
    lo, hi = float(field.min()), float(field.max())
    return ((field - lo) / (hi - lo)) if hi - lo > 1e-6 else np.zeros_like(field)


def turbulence_warp(field: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Displace *field* by a smooth noise vector field (``cv2.remap``)."""
    size = field.shape[0]
    dx = (fbm(rng, size, octaves=3, base_freq=3) - 0.5) * (2.0 * strength)
    dy = (fbm(rng, size, octaves=3, base_freq=3) - 0.5) * (2.0 * strength)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return cv2.remap(
        field,
        (xx + dx).astype(np.float32),
        (yy + dy).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def plume_shape(
    size: int,
    origin: tuple[float, float],
    angle: float,
    length: float,
    width0: float,
    spread: float,
    curl: float,
    shear: float,
) -> np.ndarray:
    """Anisotropic cone-shaped falloff describing where a plume can exist.

    The field is built in a rotated frame whose ``u`` axis follows the drift
    direction.  Density decays exponentially along ``u`` (densest at the pipe),
    is Gaussian across ``v`` with a width that grows linearly with distance
    (the cone), and is cut off sharply behind the nozzle.  ``curl`` bends the
    axis, ``shear`` skews the cross-section, both of which stop every plume
    looking like the same triangle.

    Args:
        size: Field side length.
        origin: ``(x, y)`` pixel position of the tailpipe.
        angle: Drift direction in radians (0 = +x).
        length: Characteristic plume length in pixels.
        width0: Half-width at the nozzle in pixels.
        spread: Cone widening per pixel travelled.
        curl: Lateral bend, in units of ``length``.
        shear: Cross-sectional skew factor.

    Returns:
        ``(size, size)`` ``float32`` field normalised so its maximum is 1.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx = xx - float(origin[0])
    dy = yy - float(origin[1])
    ca, sa = math.cos(angle), math.sin(angle)
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca

    length = max(float(length), 8.0)
    travelled = np.maximum(u, 0.0) / length
    v = v - curl * (travelled ** 2) * length * 0.35   # bend
    v = v - shear * travelled * length * 0.12         # skew

    width = width0 + spread * np.maximum(u, 0.0)
    core = np.exp(-1.6 * np.square(v / np.maximum(width, 1e-3)))
    axial = np.where(
        u >= 0.0,
        np.exp(-u / (0.55 * length)),
        np.exp(-np.square(u / (0.10 * length))),
    )
    shape = (core * axial).astype(np.float32)
    peak = float(shape.max())
    return shape / peak if peak > 1e-6 else shape


def make_plume_alpha(
    rng: np.random.Generator,
    size: int,
    origin: tuple[float, float],
    target_coverage: float,
    *,
    length_range: tuple[float, float] = (0.45, 1.10),
    opacity_range: tuple[float, float] = (0.38, 0.95),
    noise_floor: float = 0.30,
    min_coverage: float = 0.006,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Generate a plume alpha map whose thresholded area matches a target.

    The geometry and texture are sampled first, then the global opacity is
    *solved* so that ``(alpha > MASK_ALPHA_CUTOFF)`` covers approximately
    ``target_coverage`` of the tile.  Solving rather than sampling opacity keeps
    the label-area distribution controlled while the appearance still varies
    across the full opacity range (the solution is clipped to
    ``opacity_range``, so thin wispy and thick opaque plumes both occur).

    Args:
        rng: Seeded generator.
        size: Tile side length.
        origin: Tailpipe position in pixels.
        target_coverage: Desired mask area as a fraction of the tile.
        length_range: Plume length as a fraction of ``size``.
        opacity_range: Allowed peak alpha.  The floor was lowered from 0.55 to
            0.38 for generator 2.0.0 so the network has to find genuinely faint
            plumes: the one real tailpipe plume the project has ever measured
            scored only 0.4748, and a training set whose thinnest plume is
            still 55% opaque does not prepare a model for it.  The floor cannot
            go below :data:`MASK_ALPHA_CUTOFF`, or the label would be empty by
            construction.
        noise_floor: Minimum multiplier the fBm texture can apply, so the plume
            core never fully disappears.
        min_coverage: Labelled area below which the opacity is raised -- as
            far as ``opacity_range`` allows -- rather than emitting a positive
            with a two-pixel mask.  A faint plume is wanted; an unlearnable
            one is not.  When the plume's *geometry* is simply small the cap
            is reached and the sample stays small, which is correct; measured
            over the shipped set that happens twice in 2337 positives.

    Returns:
        ``(alpha, label, params)`` -- ``alpha`` is the ``float32`` compositing
        map in ``[0, 1]`` (full fractal detail) and ``label`` is the boolean
        ground-truth mask taken from the smoothed field.
    """
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    length = float(rng.uniform(*length_range)) * size
    width0 = float(rng.uniform(0.040, 0.110)) * size
    spread = float(rng.uniform(0.25, 0.70))
    curl = float(rng.uniform(-0.9, 0.9))
    shear = float(rng.uniform(-0.8, 0.8))

    shape = plume_shape(size, origin, angle, length, width0, spread, curl, shear)
    shape = np.power(shape, float(rng.uniform(0.6, 0.95)))  # broaden / tighten the cone

    texture = fbm(rng, size, octaves=5, persistence=float(rng.uniform(0.50, 0.66)),
                  base_freq=int(rng.integers(3, 6)))
    base = shape * (noise_floor + (1.0 - noise_floor) * texture)
    # A second, higher-frequency field frays the plume edge into wisps instead
    # of leaving the smooth teardrop the cone alone would produce.
    wisp = fbm(rng, size, octaves=3, persistence=0.62, base_freq=int(rng.integers(6, 11)))
    base = base * (0.82 + 0.18 * wisp)
    base = turbulence_warp(base, rng, strength=float(rng.uniform(7.0, 22.0)))
    # A generous final blur keeps the iso-contour that defines the label
    # smooth enough to be predictable, instead of a pixel-level fractal edge.
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=float(rng.uniform(1.8, 3.2)))

    peak = float(base.max())
    if peak <= 1e-6:
        empty = np.zeros((size, size), dtype=np.float32)
        return empty, empty.astype(bool), {"opacity": 0.0, "coverage": 0.0}
    base = base / peak

    # Sharpen the density profile.
    #
    # A raw fBm-shaped cone decays so gradually that the region where alpha
    # passes the label cutoff is tens of pixels wide, and a 10% error in the
    # network's alpha estimate moves the predicted outline a long way.  The
    # failure mode is exactly that: the model finds the plume but paints a
    # halo of faint fringe that the label excludes.  Remapping the density
    # through a smoothstep gives the plume a defined body with a soft but
    # bounded fringe -- which is also what dense exhaust actually looks like
    # near the pipe, as opposed to an infinitely diffuse veil.
    lo = float(rng.uniform(0.10, 0.22))
    hi = float(rng.uniform(0.46, 0.68))
    base = np.clip((base - lo) / max(hi - lo, 1e-3), 0.0, 1.0)
    base = (base * base * (3.0 - 2.0 * base)).astype(np.float32)

    # The label field is the same plume seen through an annotator's eye: a
    # smoothed version whose iso-contour is a clean outline, not a fractal.
    label_field = cv2.GaussianBlur(base, (0, 0), sigmaX=LABEL_SMOOTH_SIGMA * size / 256.0)
    label_peak = float(label_field.max())
    if label_peak > 1e-6:
        label_field = label_field / label_peak

    # Solve opacity so the labelled area hits the requested coverage.
    quantile = float(np.quantile(label_field, 1.0 - float(np.clip(target_coverage, 1e-4, 0.6))))
    opacity = MASK_ALPHA_CUTOFF / max(quantile, 1e-4)
    lo_op = max(float(opacity_range[0]), MASK_ALPHA_CUTOFF + 1e-3)
    opacity = float(np.clip(opacity, lo_op, float(opacity_range[1])))

    alpha = np.clip(base * opacity, 0.0, 1.0).astype(np.float32)
    label = (label_field * opacity) > MASK_ALPHA_CUTOFF
    coverage = float(label.mean())

    # Clipping the solved opacity at the (now much lower) floor can leave a
    # mask of a few dozen pixels, which is supervision the network cannot use
    # and which pollutes the Dice denominator.  Raise the opacity just enough
    # to reach `min_coverage`, still capped by the range.
    if coverage < float(min_coverage):
        needed = float(np.quantile(label_field, 1.0 - float(min_coverage)))
        bumped = float(np.clip(MASK_ALPHA_CUTOFF / max(needed, 1e-4), lo_op, float(opacity_range[1])))
        if bumped > opacity:
            opacity = bumped
            alpha = np.clip(base * opacity, 0.0, 1.0).astype(np.float32)
            label = (label_field * opacity) > MASK_ALPHA_CUTOFF
            coverage = float(label.mean())
    return alpha, label, {
        "opacity": round(opacity, 4),
        "coverage": round(coverage, 5),
        "angle": round(angle, 4),
        "length": round(length, 2),
    }


# --------------------------------------------------------------------------- #
# Smoke colour
# --------------------------------------------------------------------------- #

#: Emission type -> (BGR luminance range, per-channel BGR bias).
SMOKE_PALETTE: dict[str, tuple[tuple[int, int], tuple[int, int, int]]] = {
    # Unburnt diesel: sooty black.
    "black": ((18, 62), (4, 0, -4)),
    # Generic grey exhaust.
    "grey": ((88, 152), (6, 0, -5)),
    # Burning coolant / steam: near white.
    "white": ((176, 232), (2, 0, -2)),
    # Burning oil: characteristic blue-grey haze.
    "blue_grey": ((104, 172), (22, 2, -16)),
}


#: Minimum luminance separation between plume colour and the background it is
#: painted over.  Below this the "smoke" would be invisible and the mask would
#: be teaching the network to hallucinate.  Paired with
#: :data:`MASK_ALPHA_CUTOFF` this guarantees every labelled pixel differs from
#: its background by at least ~20 grey levels.
MIN_SMOKE_CONTRAST = 58.0


def sample_smoke_color(
    rng: np.random.Generator,
    size: int,
    background: np.ndarray,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Pick a plausible smoke colour that is actually visible on *background*.

    Contrast is measured against the background luminance **under the plume**
    (weighted by *weight*), not the whole tile -- black smoke over a bright road
    is fine, black smoke inside a dark wheel arch is not.  Candidates are drawn
    from the emission-type palette; if none clears
    :data:`MIN_SMOKE_CONTRAST` the winning value is pushed away from the local
    background until it does.

    Args:
        rng: Seeded generator.
        size: Tile side length.
        background: BGR ``uint8`` background the plume will be blended onto.
        weight: Optional ``[0, 1]`` map of where the plume will land.

    Returns:
        ``(colour_field, kind)`` where ``colour_field`` is an ``(H, W, 3)``
        ``float32`` BGR field with gentle low-frequency luminance variation.
    """
    luma = background.astype(np.float32).mean(axis=2)
    if weight is not None and float(weight.sum()) > 1e-3:
        bg_mean = float((luma * weight).sum() / weight.sum())
    else:
        bg_mean = float(luma.mean())

    kinds = list(SMOKE_PALETTE)
    probs = np.array([0.30, 0.28, 0.20, 0.22], dtype=np.float64)
    probs /= probs.sum()

    best: tuple[float, float, np.ndarray, str] | None = None
    for _ in range(6):
        kind = str(rng.choice(kinds, p=probs))
        (lo, hi), bias = SMOKE_PALETTE[kind]
        value = float(rng.uniform(lo, hi))
        bgr = np.array(
            [value + bias[0], value + bias[1], value + bias[2]], dtype=np.float32
        ) + rng.uniform(-7.0, 7.0, size=3).astype(np.float32)
        bgr = np.clip(bgr, 0.0, 255.0)
        contrast = abs(float(bgr.mean()) - bg_mean)
        if best is None or contrast > best[0]:
            best = (contrast, value, bgr, kind)
        if contrast >= MIN_SMOKE_CONTRAST:
            break

    assert best is not None
    contrast, value, bgr, kind = best
    if contrast < MIN_SMOKE_CONTRAST:
        # Push the colour to whichever side of the local background has room.
        direction = 1.0 if bg_mean <= 127.5 else -1.0
        target = float(np.clip(bg_mean + direction * MIN_SMOKE_CONTRAST, 8.0, 247.0))
        bgr = np.clip(bgr + (target - float(bgr.mean())), 0.0, 255.0)

    # Low-frequency luminance variation across the plume body.
    modulation = (fbm(rng, size, octaves=2, base_freq=2) - 0.5) * float(rng.uniform(10.0, 28.0))
    field = np.clip(bgr[None, None, :] + modulation[..., None], 0.0, 255.0).astype(np.float32)
    return field, kind


# --------------------------------------------------------------------------- #
# Backgrounds
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BackgroundSource:
    """One usable background: a photo plus any COCO vehicle boxes it carries."""

    path: Path
    vehicle_boxes: tuple[tuple[int, int, int, int], ...]  # (x, y, w, h) pixels


class BackgroundBank:
    """Indexes COCO128 photos and serves 256x256 crops.

    Crops come in three flavours:

    * ``exhaust_roi`` -- the lower band under a labelled vehicle, expanded and
      squashed to a square exactly like the inference ROI.
    * ``random_crop`` -- an arbitrary region of any photo, squashed by a random
      aspect factor so the network sees the same kind of distortion.
    * ``procedural`` -- an asphalt/sky gradient generated from scratch, so the
      generator still works when COCO128 is absent.
    """

    def __init__(
        self,
        images_dir: Path = COCO128_IMAGES_DIR,
        labels_dir: Path = COCO128_LABELS_DIR,
        split: str = "train",
        val_fraction: float = VAL_BACKGROUND_FRACTION,
        seed: int = 0,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.split = split
        self._cache: dict[Path, np.ndarray] = {}
        self.sources: list[BackgroundSource] = []
        self.vehicle_sources: list[BackgroundSource] = []
        self._index(val_fraction=val_fraction, seed=seed)

    # -- indexing ---------------------------------------------------------- #

    def _index(self, val_fraction: float, seed: int) -> None:
        if not self.images_dir.is_dir():
            logger.warning(
                "COCO128 images not found at %s -- every background will be procedural.",
                self.images_dir,
            )
            return

        paths = sorted(p for p in self.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not paths:
            logger.warning("No images under %s -- every background will be procedural.", self.images_dir)
            return

        # Deterministic, disjoint train/val background split.
        order = np.random.default_rng(seed).permutation(len(paths))
        cut = max(1, int(round(len(paths) * val_fraction)))
        val_idx = set(order[:cut].tolist())
        selected = [p for i, p in enumerate(paths) if (i in val_idx) == (self.split == "val")]
        if not selected:  # degenerate tiny pool
            selected = paths

        for path in selected:
            boxes = self._vehicle_boxes_for(path)
            source = BackgroundSource(path=path, vehicle_boxes=tuple(boxes))
            self.sources.append(source)
            if boxes:
                self.vehicle_sources.append(source)

        logger.info(
            "BackgroundBank[%s]: %d photos (%d with vehicle labels).",
            self.split,
            len(self.sources),
            len(self.vehicle_sources),
        )

    def _vehicle_boxes_for(self, image_path: Path) -> list[tuple[int, int, int, int]]:
        """Parse the YOLO label file and return vehicle boxes in pixels."""
        label_path = self.labels_dir / (image_path.stem + ".txt")
        if not label_path.is_file():
            return []
        image = self._read(image_path)
        if image is None:
            return []
        img_h, img_w = image.shape[:2]

        boxes: list[tuple[int, int, int, int]] = []
        try:
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                w = max(8, int(round(bw * img_w)))
                h = max(8, int(round(bh * img_h)))
                x = int(round(cx * img_w - w / 2.0))
                y = int(round(cy * img_h - h / 2.0))
                # Skip specks: they produce degenerate, uninformative ROIs.
                if w * h < 64 * 48:
                    continue
                boxes.append((x, y, w, h))
        except (OSError, ValueError):
            return []
        return boxes

    def _read(self, path: Path) -> np.ndarray | None:
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        self._cache[path] = image
        return image

    # -- sampling ---------------------------------------------------------- #

    def sample(self, rng: np.random.Generator, size: int = TILE) -> tuple[np.ndarray, str, tuple[float, float] | None]:
        """Draw one background crop.

        Returns:
            ``(crop_bgr, source_kind, pipe_hint)`` where ``pipe_hint`` is a
            suggested tailpipe position in crop coordinates (``None`` when the
            crop has no vehicle context).
        """
        kinds = list(SOURCE_WEIGHTS)
        weights = np.array([SOURCE_WEIGHTS[k] for k in kinds], dtype=np.float64)
        if not self.vehicle_sources:
            weights[kinds.index("exhaust_roi")] = 0.0
        if not self.sources:
            weights[:] = 0.0
            weights[kinds.index("procedural")] = 1.0
        weights = weights / weights.sum()
        kind = str(rng.choice(kinds, p=weights))

        if kind == "exhaust_roi":
            result = self._sample_exhaust_roi(rng, size)
            if result is not None:
                return result[0], "exhaust_roi", result[1]
            kind = "random_crop"
        if kind == "random_crop":
            result = self._sample_random_crop(rng, size)
            if result is not None:
                return result, "random_crop", None
        return procedural_background(rng, size), "procedural", None

    def _sample_exhaust_roi(
        self, rng: np.random.Generator, size: int
    ) -> tuple[np.ndarray, tuple[float, float]] | None:
        """Crop the exhaust ROI under a labelled vehicle, matching inference."""
        source = self.vehicle_sources[int(rng.integers(len(self.vehicle_sources)))]
        image = self._read(source.path)
        if image is None:
            return None
        img_h, img_w = image.shape[:2]
        vx, vy, vw, vh = source.vehicle_boxes[int(rng.integers(len(source.vehicle_boxes)))]

        # Same geometry as VehicleDetector.exhaust_roi, with jitter.
        band = max(8, int(vh * float(rng.uniform(0.38, 0.55))))
        pad_x = int(vw * float(rng.uniform(0.20, 0.42)))
        pad_down = int(band * float(rng.uniform(0.18, 0.45)))
        x0 = max(0, vx - pad_x)
        y0 = max(0, vy + vh - band)
        x1 = min(img_w, vx + vw + pad_x)
        y1 = min(img_h, vy + vh + pad_down)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None

        crop = cv2.resize(image[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)

        # Tailpipe: near a rear corner of the vehicle, i.e. left/right edge of
        # the ROI, a little below its vertical middle.
        roi_w, roi_h = x1 - x0, y1 - y0
        side = -1.0 if rng.random() < 0.5 else 1.0
        pipe_x_src = (vx + vw * (0.18 if side < 0 else 0.82)) - x0
        pipe_y_src = (vy + vh * float(rng.uniform(0.80, 0.98))) - y0
        pipe = (
            float(np.clip(pipe_x_src / roi_w * size, 0.05 * size, 0.95 * size)),
            float(np.clip(pipe_y_src / roi_h * size, 0.30 * size, 0.92 * size)),
        )
        return crop, pipe

    # -- mined real negatives ---------------------------------------------- #

    def sample_mined_negative(
        self, rng: np.random.Generator, size: int
    ) -> tuple[np.ndarray, str]:
        """A **real, unmodified** crop that provably contains no exhaust smoke.

        These are the most valuable negatives available to this project: real
        pixels carrying a guaranteed-correct all-zero label, as opposed to a
        procedural approximation of one.  ``docs/REAL_DATA_EVALUATION.md``
        recommendation #1 asks for exactly this.

        Two sources, because COCO128 only carries vehicle labels on a handful
        of photographs and an exhaust-ROI-only diet would memorise them:

        * **exhaust_roi** -- the real lower-band crop under a labelled vehicle.
          Correct geometry, correct content, tiny pool.
        * **shadow_rich** -- any crop from any COCO128 photo that scores highly
          for "large dark region against a bright one", i.e. a real cast
          shadow of *something*.  Large pool, and a real shadow is a real
          shadow whether a car or a lamp-post cast it.

        Only photographs belonging to this split's half of the photo-disjoint
        partition are ever used, so mined negatives cannot leak train imagery
        into the validation score.

        Returns:
            ``(crop_bgr, source_kind)``.
        """
        if self.vehicle_sources and rng.random() < 0.55:
            result = self._sample_exhaust_roi(rng, size)
            if result is not None:
                return result[0], "exhaust_roi"
        mined = self._sample_shadow_rich_crop(rng, size)
        if mined is not None:
            return mined, "shadow_rich"
        fallback = self._sample_random_crop(rng, size)
        if fallback is not None:
            return fallback, "random_crop"
        return procedural_background(rng, size), "procedural"

    @staticmethod
    def _shadow_score(crop: np.ndarray) -> float:
        """How much this crop looks like "a real shadow on a bright surface".

        Otsu splits the crop into a dark and a bright population.  A good
        hard negative has a *substantial* dark region (not a few dark pixels,
        not a uniformly dark frame) that is strongly separated from the bright
        one -- which is what a cast shadow on sunlit tarmac or concrete is.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = gray < threshold
        fraction = float(dark.mean())
        if fraction < 0.08 or fraction > 0.75:
            return 0.0
        separation = float(gray[~dark].mean()) - float(gray[dark].mean())
        return separation * min(fraction / 0.35, 1.0)

    def _sample_shadow_rich_crop(self, rng: np.random.Generator, size: int) -> np.ndarray | None:
        """Best of several random crops by :meth:`_shadow_score`."""
        best: np.ndarray | None = None
        best_score = -1.0
        for _ in range(6):
            candidate = self._sample_random_crop(rng, size)
            if candidate is None:
                continue
            score = self._shadow_score(candidate)
            if score > best_score:
                best, best_score = candidate, score
        return best

    def _sample_random_crop(self, rng: np.random.Generator, size: int) -> np.ndarray | None:
        """Arbitrary crop from any photo, squashed by a random aspect factor."""
        for _ in range(4):
            source = self.sources[int(rng.integers(len(self.sources)))]
            image = self._read(source.path)
            if image is None:
                continue
            img_h, img_w = image.shape[:2]
            side = int(min(img_h, img_w) * float(rng.uniform(0.35, 1.0)))
            side = max(48, min(side, img_h, img_w))
            aspect = float(rng.uniform(0.6, 1.7))
            cw = max(32, min(img_w, int(side * aspect)))
            ch = max(32, min(img_h, side))
            x0 = int(rng.integers(0, max(1, img_w - cw + 1)))
            y0 = int(rng.integers(0, max(1, img_h - ch + 1)))
            return cv2.resize(image[y0 : y0 + ch, x0 : x0 + cw], (size, size), interpolation=cv2.INTER_AREA)
        return None


def procedural_background(rng: np.random.Generator, size: int = TILE) -> np.ndarray:
    """Offline fallback background: asphalt, horizon, sky and clutter."""
    horizon = int(size * float(rng.uniform(0.15, 0.45)))
    canvas = np.zeros((size, size, 3), dtype=np.float32)

    sky = np.array([200, 190, 178], dtype=np.float32) + rng.uniform(-35, 35, 3).astype(np.float32)
    road_near = np.array([88, 88, 90], dtype=np.float32) + rng.uniform(-28, 28, 3).astype(np.float32)
    road_far = road_near * float(rng.uniform(1.12, 1.45))

    for y in range(size):
        if y < horizon:
            t = y / max(horizon, 1)
            canvas[y] = sky * (0.86 + 0.14 * t)
        else:
            t = (y - horizon) / max(size - horizon, 1)
            canvas[y] = road_far * (1.0 - t) + road_near * t

    grain = (fbm(rng, size, octaves=5, base_freq=6) - 0.5) * float(rng.uniform(18, 44))
    canvas += grain[..., None]

    # Lane markings / kerb clutter so it is not a flat gradient.
    for _ in range(int(rng.integers(0, 4))):
        y = int(rng.integers(horizon, size))
        thickness = int(rng.integers(2, 7))
        shade = float(rng.uniform(140, 225))
        cv2.line(canvas, (0, y), (size, y + int(rng.integers(-16, 16))), (shade, shade, shade), thickness)
    return np.clip(canvas, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Distractors and photometric realism
# --------------------------------------------------------------------------- #

def dust_alpha(rng: np.random.Generator, size: int, origin: tuple[float, float]) -> np.ndarray:
    """Alpha for a road-dust cloud: broader, softer and fainter than smoke."""
    alpha, _label, _params = make_plume_alpha(
        rng,
        size,
        origin,
        target_coverage=float(rng.uniform(0.06, 0.30)),
        length_range=(0.45, 1.05),
        opacity_range=(0.16, 0.34),
        noise_floor=0.55,
    )
    return cv2.GaussianBlur(alpha, (0, 0), sigmaX=float(rng.uniform(2.5, 5.0)))


def apply_dust(img: np.ndarray, rng: np.random.Generator, suppress: np.ndarray | None = None) -> np.ndarray:
    """Composite a brown-tinted dust cloud (a hard negative) onto *img*."""
    size = img.shape[0]
    origin = (float(rng.uniform(0.1, 0.9)) * size, float(rng.uniform(0.2, 0.9)) * size)
    alpha = dust_alpha(rng, size, origin)
    if suppress is not None:
        alpha = alpha * (1.0 - suppress)
    value = float(rng.uniform(120.0, 185.0))
    colour = np.array([value * 0.62, value * 0.80, value], dtype=np.float32)  # BGR, red-biased
    colour = np.clip(colour + rng.uniform(-10, 10, 3).astype(np.float32), 0, 255)
    return _blend(img, alpha, np.broadcast_to(colour, (size, size, 3)).astype(np.float32))


def apply_shadow(img: np.ndarray, rng: np.random.Generator, suppress: np.ndarray | None = None) -> np.ndarray:
    """Darken a soft elliptical region -- the classic under-vehicle shadow."""
    size = img.shape[0]
    mask = np.zeros((size, size), dtype=np.float32)
    cx = int(rng.uniform(0.15, 0.85) * size)
    cy = int(rng.uniform(0.35, 0.95) * size)
    axes = (int(rng.uniform(0.15, 0.45) * size), int(rng.uniform(0.07, 0.26) * size))
    cv2.ellipse(mask, (cx, cy), axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=float(rng.uniform(6.0, 18.0)))
    mask *= float(rng.uniform(0.35, 0.72))
    if suppress is not None:
        mask = mask * (1.0 - suppress)
    out = img.astype(np.float32) * (1.0 - mask[..., None] * float(rng.uniform(0.35, 0.65)))
    return np.clip(out, 0, 255).astype(np.uint8)


def cast_shadow_field(
    rng: np.random.Generator,
    size: int,
    *,
    contact_y: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Occlusion field for a vehicle's own cast shadow on the ground plane.

    The geometry is the one the exhaust ROI actually frames: the vehicle's
    bottom edge crosses the crop at a *contact line*, and its shadow is a
    perspective-skewed quadrilateral hanging off that line in the direction
    the sun pushes it.  Sampled independently:

    * **sun azimuth** via ``lean`` -- how far the far edge slides sideways;
    * **sun elevation** via ``depth`` -- how far the shadow reaches;
    * **penumbra** via the blur sigma -- a hard noon edge (sigma ~1 px) through
      to a soft overcast one (sigma ~20 px);
    * **contact band** -- with some probability, a near-opaque strip right at
      the contact line.  This is not a shadow at all but the *occluded cavity*
      under a bumper, and it is what the real false positives in
      ``docs/REAL_DATA_EVALUATION.md`` largely fired on: near-black, almost
      textureless, hard-topped.

    Args:
        rng: Seeded generator.
        size: Tile side length.
        contact_y: Row of the vehicle contact line; sampled when ``None``.

    Returns:
        ``(field, params)``.  ``field`` is ``float32`` in ``[0, 1]`` where 1
        means fully occluded.  It is an *occlusion* map, not an image: the
        caller multiplies by it, which is what makes the result physically a
        shadow rather than a grey blend.
    """
    y_contact = float(rng.uniform(0.28, 0.66)) * size if contact_y is None else float(contact_y)
    lean = float(rng.uniform(-1.0, 1.0))                  # sun azimuth
    depth = float(rng.uniform(0.16, 0.66)) * size         # sun elevation
    half_w = float(rng.uniform(0.24, 0.54)) * size
    taper = float(rng.uniform(0.55, 1.35))
    centre_x = float(rng.uniform(0.30, 0.70)) * size

    y_far = min(size - 1.0, y_contact + depth)
    quad = np.array(
        [
            [centre_x - half_w, y_contact],
            [centre_x + half_w, y_contact],
            [centre_x + half_w * taper + lean * depth, y_far],
            [centre_x - half_w * taper + lean * depth, y_far],
        ],
        dtype=np.float32,
    )
    field = np.zeros((size, size), dtype=np.float32)
    cv2.fillConvexPoly(field, np.round(quad).astype(np.int32), 1.0, lineType=cv2.LINE_AA)

    # A real shadow edge is not a straight line: the silhouette casting it is
    # ragged (wheels, exhaust, mud-flaps) and the ground is not flat.
    field = turbulence_warp(field, rng, strength=float(rng.uniform(2.0, 9.0)))

    # Penumbra.  Sun angular diameter plus surface roughness; hard at noon,
    # soft under cloud.
    penumbra = float(rng.uniform(1.0, 20.0)) * size / 256.0
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=penumbra)

    # Ambient skylight bounces back in with distance from the occluder, so the
    # umbra is deepest at the contact line and fades outward.
    yy = np.arange(size, dtype=np.float32)[:, None]
    falloff = np.exp(-np.maximum(yy - y_contact, 0.0) / max(depth * float(rng.uniform(0.45, 1.3)), 1.0))
    field = field * np.broadcast_to(falloff, field.shape)

    params = {
        "lean": round(lean, 3),
        "depth": round(depth / size, 3),
        "penumbra": round(penumbra, 2),
        "contact_y": round(y_contact / size, 3),
        "contact_band": 0.0,
    }

    # The occluded cavity under the bumper: hard-topped, near-total, shallow.
    if rng.random() < 0.55:
        band_h = max(2.0, float(rng.uniform(0.04, 0.16)) * size)
        band = np.zeros((size, size), dtype=np.float32)
        x0 = int(max(0, centre_x - half_w * float(rng.uniform(0.75, 1.05))))
        x1 = int(min(size, centre_x + half_w * float(rng.uniform(0.75, 1.05))))
        y0 = int(max(0, y_contact - band_h * float(rng.uniform(0.25, 0.9))))
        y1 = int(min(size, y_contact + band_h))
        if x1 > x0 and y1 > y0:
            band[y0:y1, x0:x1] = 1.0
            # Hard on top (the bumper line), softer underneath.
            band = cv2.GaussianBlur(band, (0, 0), sigmaX=max(0.6, 1.8 * size / 256.0))
            field = np.maximum(field, band * float(rng.uniform(0.85, 1.0)))
            params["contact_band"] = 1.0

    return np.clip(field, 0.0, 1.0).astype(np.float32), params


def apply_cast_shadow(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    field: np.ndarray | None = None,
    strength: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Darken *img* **multiplicatively** with a cast shadow.

    ``out = img * (1 - k * field)``, per channel.  This is the physically
    right operation and it is the whole point of the function: multiplication
    scales the local mean and the local standard deviation by the same factor,
    so *normalised* local contrast ``std/mean`` -- and therefore visible
    surface texture, and hue -- survives intact.  The alpha blend used for
    smoke (:func:`_blend`) does the opposite: it pulls every pixel toward one
    colour and collapses ``std/mean``.  That difference is the only reliable
    cue separating the two on real imagery, and a shadow rendered as a grey
    blend would teach the network precisely the wrong thing.

    ``k`` reaches 0.88 deliberately.  Generator 1.4.0's shadow negatives
    bottomed out around an effective 0.47 and so never got as dark as its own
    ``black`` smoke positives; the model could separate the classes on
    brightness alone, and duly did.  These shadows have to occupy the same
    luminance range as the dark plumes or they teach nothing.

    A small blue bias is applied because a real shadow is lit by sky rather
    than sun, so it is not a neutral darkening -- another cue that is free to
    add and that exhaust smoke does not share.

    Args:
        img: BGR ``uint8`` tile.
        rng: Seeded generator.
        field: Pre-computed occlusion field; generated when ``None``.
        strength: Peak occlusion ``k``; sampled when ``None``.

    Returns:
        ``(shadowed_bgr_uint8, field)`` -- the field is returned so a caller
        compositing a plume on top can record where the shadow went.
    """
    size = img.shape[0]
    if field is None:
        field, _params = cast_shadow_field(rng, size)
    k = float(rng.uniform(0.35, 0.88)) if strength is None else float(strength)

    # Skylight is blue, so the red channel loses proportionally more.
    tint = float(rng.uniform(0.0, 0.12))
    per_channel = np.array([k * (1.0 - tint), k, k * (1.0 + tint)], dtype=np.float32)  # BGR
    per_channel = np.clip(per_channel, 0.0, 0.97)

    out = img.astype(np.float32) * (1.0 - field[..., None] * per_channel[None, None, :])
    return np.clip(out, 0, 255).astype(np.uint8), field


def clahe_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the inference-time CLAHE so training sees the same statistics.

    ``mlcore.preprocess.enhance`` runs a bilateral filter plus CLAHE on the L
    channel of every frame before the ROI is cropped, so at inference the
    segmenter *never* sees raw pixels -- but generator 1.4.0's tiles were raw.
    CLAHE is also the single operation most likely to lift a shadow's internal
    contrast into the range the segmenter reacts to, which makes the skew
    specifically dangerous here (``REAL_DATA_EVALUATION.md`` §3.2, #8).

    This is an approximation, not an exact match: at inference CLAHE is
    computed over the whole frame and the ROI is cropped afterwards, so the
    tile grid and clip statistics differ.  It closes most of the gap for the
    cost of one filter.
    """
    try:
        denoised = cv2.bilateralFilter(img, d=5, sigmaColor=45, sigmaSpace=45)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(rng.uniform(1.4, 2.8)),
            tileGridSize=(int(rng.integers(6, 11)), int(rng.integers(6, 11))),
        )
        return cv2.cvtColor(cv2.merge((clahe.apply(l_chan), a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    except cv2.error:  # pragma: no cover - defensive
        return img


def apply_motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Directional blur: mimics a panning camera, which smears edges like haze."""
    length = int(rng.integers(7, 23))
    angle = float(rng.uniform(0.0, 180.0))
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    total = kernel.sum()
    if total <= 1e-6:
        return img
    return cv2.filter2D(img, -1, kernel / total)


def _blend(img: np.ndarray, alpha: np.ndarray, colour: np.ndarray) -> np.ndarray:
    """``img*(1-a) + colour*a`` with clipping, returning ``uint8``."""
    a = np.clip(alpha, 0.0, 1.0)[..., None].astype(np.float32)
    out = img.astype(np.float32) * (1.0 - a) + colour * a
    return np.clip(out, 0, 255).astype(np.uint8)


def photometric_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomise global brightness, contrast and colour temperature."""
    contrast = float(rng.uniform(0.80, 1.22))
    brightness = float(rng.uniform(-26.0, 26.0))
    out = img.astype(np.float32) * contrast + brightness
    temp = rng.uniform(-9.0, 9.0, size=3).astype(np.float32)  # mild white-balance drift
    out = out + temp[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def sensor_realism(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add sensor noise, optional defocus and JPEG-style block artefacts."""
    out = img.astype(np.float32)
    # Noise is kept well below the label-boundary contrast (see
    # MASK_ALPHA_CUTOFF) so augmentation never erases the supervision signal.
    out += rng.normal(0.0, float(rng.uniform(1.0, 4.0)), size=out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    if rng.random() < 0.45:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=float(rng.uniform(0.4, 1.2)))
    if rng.random() < 0.55:
        quality = int(rng.integers(58, 94))
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if decoded is not None:
                out = decoded
    return out


# --------------------------------------------------------------------------- #
# Sample generation
# --------------------------------------------------------------------------- #

def _default_pipe(rng: np.random.Generator, size: int) -> tuple[float, float]:
    """Plausible tailpipe position when the crop carries no vehicle context."""
    return (
        float(rng.uniform(0.12, 0.88)) * size,
        float(rng.uniform(0.35, 0.85)) * size,
    )


def generate_sample(
    rng: np.random.Generator,
    bank: BackgroundBank,
    size: int = TILE,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Produce one ``(image, mask, metadata)`` training triple.

    Args:
        rng: Seeded generator -- the *only* source of randomness.
        bank: Background provider for this split.
        size: Tile side length.

    Returns:
        ``(image_bgr_uint8, mask_uint8_0_or_255, metadata)``.
    """
    background, source_kind, pipe_hint = bank.sample(rng, size)
    meta: dict[str, Any] = {"source": source_kind}

    is_negative = bool(rng.random() < NEGATIVE_FRACTION)

    if is_negative:
        kinds = list(NEGATIVE_KINDS)
        probs = np.array([NEGATIVE_KINDS[k] for k in kinds], dtype=np.float64)
        probs /= probs.sum()
        neg_kind = str(rng.choice(kinds, p=probs))
        meta.update(label="negative", negative_kind=neg_kind)

        image = background
        if neg_kind == "cast_shadow":
            # Anchor the contact line on the vehicle's bottom edge when the
            # crop is a real exhaust ROI, so the shadow lands where a real one
            # would -- which is where every measured false positive was.
            contact = pipe_hint[1] if pipe_hint is not None else None
            if contact is not None:
                contact = float(np.clip(contact + rng.uniform(-0.08, 0.06) * size,
                                        0.20 * size, 0.72 * size))
            field, shadow_params = cast_shadow_field(rng, size, contact_y=contact)
            image, _field = apply_cast_shadow(image, rng, field=field)
            meta.update({f"shadow_{k}": v for k, v in shadow_params.items()})
        elif neg_kind == "mined_real":
            # Real pixels, untouched, guaranteed-correct all-zero label.
            image, mined_kind = bank.sample_mined_negative(rng, size)
            meta["source"] = source_kind = mined_kind
            meta["mined"] = True
        elif neg_kind == "dust":
            image = apply_dust(image, rng)
        elif neg_kind == "shadow":
            image = apply_shadow(image, rng)
        elif neg_kind == "motion_blur":
            image = apply_motion_blur(image, rng)
        mask = np.zeros((size, size), dtype=np.uint8)
    else:
        # A share of positives sit on top of a cast shadow, overlapping it on
        # purpose.  The tile then contains dark pixels labelled smoke *and*
        # dark pixels labelled background, which is what stops the network
        # from answering the question with luminance.
        on_shadow = bool(rng.random() < POSITIVE_ON_SHADOW_FRACTION)
        shadow_field: np.ndarray | None = None
        if on_shadow:
            contact = pipe_hint[1] if pipe_hint is not None else float(rng.uniform(0.30, 0.62)) * size
            contact = float(np.clip(contact + rng.uniform(-0.06, 0.06) * size,
                                    0.20 * size, 0.70 * size))
            shadow_field, shadow_params = cast_shadow_field(rng, size, contact_y=contact)
            background, _f = apply_cast_shadow(background, rng, field=shadow_field)
            meta.update(on_cast_shadow=True,
                        **{f"shadow_{k}": v for k, v in shadow_params.items()})

        pipe = pipe_hint if pipe_hint is not None else _default_pipe(rng, size)
        if shadow_field is not None:
            # Emit from inside the shadow so the plume and the shadow overlap
            # rather than sitting side by side.
            inside = np.argwhere(shadow_field > 0.45)
            if inside.size:
                py, px = inside[int(rng.integers(len(inside)))]
                pipe = (float(px), float(py))

        alpha, binary, params = make_plume_alpha(
            rng,
            size,
            pipe,
            target_coverage=float(rng.uniform(0.04, 0.26)),
        )
        colour, colour_kind = sample_smoke_color(rng, size, background, weight=alpha)
        image = _blend(background, alpha, colour)
        mask = (binary.astype(np.uint8)) * 255
        meta.update(label="positive", smoke_colour=colour_kind, **params)

        # Some positives also carry a distractor, placed so it cannot overlap
        # the labelled plume (otherwise we would be teaching contradictions).
        if rng.random() < POSITIVE_DISTRACTOR_FRACTION:
            suppress = cv2.dilate(binary.astype(np.float32), np.ones((9, 9), np.uint8))
            suppress = cv2.GaussianBlur(suppress, (0, 0), sigmaX=4.0)
            suppress = np.clip(suppress * 1.6, 0.0, 1.0)
            roll = rng.random()
            if roll < 0.40:
                image = apply_dust(image, rng, suppress=suppress)
                meta["distractor"] = "dust"
            elif roll < 0.75:
                # The hard multiplicative shadow, kept off the labelled plume.
                d_field, _p = cast_shadow_field(rng, size)
                d_field = d_field * (1.0 - suppress)
                image, _f = apply_cast_shadow(image, rng, field=d_field)
                meta["distractor"] = "cast_shadow"
            else:
                image = apply_shadow(image, rng, suppress=suppress)
                meta["distractor"] = "shadow"

    image = photometric_jitter(image, rng)
    # Close the train/serve gap: at inference every ROI has been through
    # preprocess.enhance() (bilateral + CLAHE).  See clahe_jitter().
    if rng.random() < CLAHE_AUGMENT_FRACTION:
        image = clahe_jitter(image, rng)
        meta["clahe"] = True
    image = sensor_realism(image, rng)

    if rng.random() < 0.5:  # horizontal flip keeps image and mask in sync
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])

    meta["mask_coverage"] = round(float((mask > 127).mean()), 5)
    return image, mask, meta


# --------------------------------------------------------------------------- #
# Dataset writing
# --------------------------------------------------------------------------- #

def _split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    return root / split / "images", root / split / "masks"


def _existing_count(root: Path, split: str) -> int:
    images_dir, masks_dir = _split_dirs(root, split)
    if not images_dir.is_dir() or not masks_dir.is_dir():
        return 0
    images = {p.stem for p in images_dir.glob("*.png")}
    masks = {p.stem for p in masks_dir.glob("*.png")}
    return len(images & masks)


def build_split(
    root: Path,
    split: str,
    count: int,
    seed: int,
    bank_seed: int,
    size: int = TILE,
    progress_every: int = 200,
) -> dict[str, Any]:
    """Generate and write *count* samples for one split.

    Args:
        root: Dataset root directory.
        split: ``"train"`` or ``"val"``.
        count: Number of samples to write.
        seed: Seed for this split's sample stream.
        bank_seed: Seed for the photo-level train/val partition.  Must be the
            **same** for both splits or the partition stops being disjoint.
        size: Tile side length.
        progress_every: Log cadence, ``0`` to silence.

    Returns:
        A stats dict recording positive/negative composition.
    """
    images_dir, masks_dir = _split_dirs(root, split)
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    split_id = int(zlib.crc32(split.encode("utf-8")))
    bank = BackgroundBank(split=split, seed=bank_seed)
    stats: dict[str, Any] = {
        "count": count,
        "positive": 0,
        "negative": 0,
        "negative_kinds": {},
        "sources": {},
        "smoke_colours": {},
        "background_photos": len(bank.sources),
        "background_photos_with_vehicles": len(bank.vehicle_sources),
        "mean_positive_coverage": 0.0,
        "positives_on_cast_shadow": 0,
    }
    coverages: list[float] = []
    index: list[dict[str, Any]] = []

    # Darkness/label balance -- the quantity generator 2.0.0 exists to fix.
    # A "shadow-like" region is dark in absolute terms; if the network can
    # separate the classes by luminance alone it will, so we measure whether
    # it still can.  See the module docstring.
    dark_smoke_px = 0.0
    dark_bg_px = 0.0
    total_smoke_px = 0.0
    total_px = 0.0
    shadowlike_positives = 0

    started = time.time()
    for i in range(count):
        # Per-sample stream so a partial regeneration is reproducible.
        rng = np.random.default_rng([seed, split_id, i])
        image, mask, meta = generate_sample(rng, bank, size=size)

        name = f"{i:06d}.png"
        cv2.imwrite(str(images_dir / name), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        cv2.imwrite(str(masks_dir / name), mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        labelled = mask > 127
        dark = gray < DARK_LUMA
        dark_smoke_px += float(np.count_nonzero(dark & labelled))
        dark_bg_px += float(np.count_nonzero(dark & ~labelled))
        total_smoke_px += float(np.count_nonzero(labelled))
        total_px += float(gray.size)

        if meta.get("label") == "positive":
            stats["positive"] += 1
            coverages.append(float(meta.get("mask_coverage", 0.0)))
            key = str(meta.get("smoke_colour", "?"))
            stats["smoke_colours"][key] = stats["smoke_colours"].get(key, 0) + 1
            if meta.get("on_cast_shadow"):
                stats["positives_on_cast_shadow"] += 1
            if labelled.any() and (~labelled).any():
                inside = float(gray[labelled].mean())
                outside = float(gray[~labelled].mean())
                if inside < DARK_LUMA and (inside - outside) < -25.0:
                    shadowlike_positives += 1
        else:
            stats["negative"] += 1
            key = str(meta.get("negative_kind", "?"))
            stats["negative_kinds"][key] = stats["negative_kinds"].get(key, 0) + 1
        src = str(meta.get("source", "?"))
        stats["sources"][src] = stats["sources"].get(src, 0) + 1
        index.append({"file": name, **meta})

        if progress_every and (i + 1) % progress_every == 0:
            logger.info("  %s: %d/%d samples (%.1fs)", split, i + 1, count, time.time() - started)

    stats["mean_positive_coverage"] = round(float(np.mean(coverages)), 5) if coverages else 0.0

    # How much does "this pixel is dark" tell you about "this pixel is smoke"?
    #
    # `lift` is P(smoke | dark) / P(smoke), at the *pixel* level.  Recorded for
    # transparency, but do not read much into it: measured on generator 1.4.0
    # it was already 0.667 (versus 0.658 here), because the marginal is
    # swamped by the enormous number of dark background pixels that were never
    # the problem.  The statistic that moved is region-level and lives in the
    # NEGATIVE_FRACTION comment; the statistic that mattered most is the
    # *luminance overlap* between dark negatives and dark positives, in the
    # module docstring.
    base_rate = total_smoke_px / max(total_px, 1.0)
    dark_total = dark_smoke_px + dark_bg_px
    p_smoke_given_dark = dark_smoke_px / max(dark_total, 1.0)
    stats["darkness_balance"] = {
        "dark_luma_threshold": DARK_LUMA,
        "p_smoke": round(base_rate, 6),
        "p_smoke_given_dark": round(p_smoke_given_dark, 6),
        "lift": round(p_smoke_given_dark / max(base_rate, 1e-9), 4),
        "shadowlike_positives": shadowlike_positives,
        "shadowlike_positive_fraction": round(shadowlike_positives / max(stats["positive"], 1), 4),
        "dark_background_px_per_dark_smoke_px": round(dark_bg_px / max(dark_smoke_px, 1.0), 3),
    }
    stats["seconds"] = round(time.time() - started, 2)
    (root / split / "index.json").write_text(json.dumps(index, indent=1))
    return stats


def build_dataset(
    root: Path = SYNTH_DATASET_DIR,
    train: int = 2800,
    val: int = 700,
    seed: int = 1337,
    size: int = TILE,
    force: bool = False,
) -> dict[str, Any]:
    """Build (or reuse) the whole synthetic dataset and write its manifest.

    Args:
        root: Output directory.
        train: Number of training samples.
        val: Number of validation samples.
        seed: Master seed -- the dataset is bit-for-bit reproducible from it.
        size: Tile side length.
        force: Regenerate even when the target counts already exist.

    Returns:
        The manifest dict that was written to ``manifest.json``.
    """
    root = Path(root)
    manifest_path = root / "manifest.json"

    if not force and _existing_count(root, "train") >= train and _existing_count(root, "val") >= val:
        if manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text())
                if existing.get("generator_version") == GENERATOR_VERSION and existing.get("seed") == seed:
                    logger.info("Dataset already present at %s (use --force to regenerate).", root)
                    return existing
            except (OSError, json.JSONDecodeError):
                pass

    root.mkdir(parents=True, exist_ok=True)
    logger.info("Generating synthetic smoke dataset -> %s (train=%d val=%d seed=%d)", root, train, val, seed)

    train_stats = build_split(root, "train", train, seed=seed, bank_seed=seed, size=size)
    val_stats = build_split(root, "val", val, seed=seed + 7919, bank_seed=seed, size=size)

    manifest = {
        "name": "smoke_synth",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "tile_size": size,
        "mask_alpha_cutoff": MASK_ALPHA_CUTOFF,
        "negative_fraction_target": NEGATIVE_FRACTION,
        "negative_kinds_target": dict(NEGATIVE_KINDS),
        "positive_on_shadow_fraction_target": POSITIVE_ON_SHADOW_FRACTION,
        "clahe_augment_fraction": CLAHE_AUGMENT_FRACTION,
        "background_source": str(COCO128_IMAGES_DIR),
        "background_split": {
            "strategy": "photo-disjoint",
            "val_fraction": VAL_BACKGROUND_FRACTION,
            "note": "train and val never share a source photograph",
        },
        "splits": {"train": train_stats, "val": val_stats},
        "totals": {
            "samples": train_stats["count"] + val_stats["count"],
            "positive": train_stats["positive"] + val_stats["positive"],
            "negative": train_stats["negative"] + val_stats["negative"],
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "Dataset ready: %d train / %d val (%d positive, %d negative).",
        train_stats["count"],
        val_stats["count"],
        manifest["totals"]["positive"],
        manifest["totals"]["negative"],
    )
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mlcore.training.synth_dataset",
        description="Generate the synthetic vehicle-smoke segmentation dataset.",
    )
    # 2800/700, up from 1600/400.  Generator 2.0.0's distribution is markedly
    # harder than 1.4.0's -- fainter plumes, cast-shadow negatives at full
    # strength, a third of positives sitting on top of a shadow -- and at the
    # old size the model plateaued 0.06 Dice lower.  Same seed, same
    # photo-disjoint split; only the sample count changed.
    parser.add_argument("--train", type=int, default=2800, help="number of training samples")
    parser.add_argument("--val", type=int, default=700, help="number of validation samples")
    parser.add_argument("--seed", type=int, default=1337, help="master RNG seed")
    parser.add_argument("--size", type=int, default=TILE, help="tile side length in pixels")
    parser.add_argument("--out", type=str, default=str(SYNTH_DATASET_DIR), help="output directory")
    parser.add_argument("--force", action="store_true", help="regenerate even if the dataset exists")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    manifest = build_dataset(
        root=Path(args.out),
        train=args.train,
        val=args.val,
        seed=args.seed,
        size=args.size,
        force=args.force,
    )
    print(json.dumps({k: manifest[k] for k in ("name", "generator_version", "seed", "totals")}, indent=2))
    print(json.dumps(manifest["splits"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
