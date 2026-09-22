#!/usr/bin/env python
"""Generate the demonstration media in ``backend/sample_media/``.

Four artifacts are produced, all derived from **real photographs** so that the
vehicle detector has genuine vehicles to find, with the smoke plumes
**synthesised** by the same fractal model that built the training set:

* ``sample_truck_smoking.mp4``  ~6 s, 640x480, 20 fps, animated plume
* ``sample_car_clean.mp4``      ~6 s, 640x480, 20 fps, no smoke
* ``sample_bus_smoking.jpg``    the real ``bus.jpg`` with an exhaust plume
* ``sample_street_clean.jpg``   an unmodified COCO128 street photo

Motion in the videos comes from a slow pan-and-zoom over the still photograph,
which is enough for YOLO to keep detecting the vehicle while the plume boils.

Run with::

    python tools/make_samples.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlcore.config import COCO128_IMAGES_DIR, DATASETS_DIR, MLConfig, SAMPLE_MEDIA_DIR  # noqa: E402
from mlcore.detector import VehicleDetector  # noqa: E402
from mlcore.training.synth_dataset import fbm, plume_shape  # noqa: E402

logger = logging.getLogger("asg.ml")

#: Canonical resolution the plume is simulated at before being warped into place.
CANON = 384
#: Where the tailpipe sits in the canonical frame (plume points along +x).
CANON_PIPE = (0.10 * CANON, 0.50 * CANON)
#: Plume length in the canonical frame.
CANON_LENGTH = 0.78 * CANON

VIDEO_SIZE = (640, 480)
VIDEO_FPS = 20
VIDEO_SECONDS = 6.0

#: Codec attempts, in order.  ``mp4v`` is the only fourcc that is reliably
#: available in stock OpenCV wheels on both macOS and Windows.
CODEC_CANDIDATES: tuple[tuple[str, str], ...] = (("mp4v", ".mp4"), ("avc1", ".mp4"), ("MJPG", ".avi"))


# --------------------------------------------------------------------------- #
# Animated plume
# --------------------------------------------------------------------------- #

def _fbm_rect(
    rng: np.random.Generator,
    height: int,
    width: int,
    octaves: int = 5,
    persistence: float = 0.56,
    base_freq: int = 4,
) -> np.ndarray:
    """Rectangular fractal-Brownian-motion field in ``[0, 1]``.

    The square :func:`mlcore.training.synth_dataset.fbm` cannot make the long
    strip needed to advect a plume over time, so this is its rectangular twin.
    """
    field = np.zeros((height, width), dtype=np.float32)
    amplitude, total, freq = 1.0, 0.0, int(base_freq)
    aspect = width / max(height, 1)
    for _ in range(max(1, octaves)):
        gh = max(2, int(freq))
        gw = max(2, int(round(freq * aspect)))
        low = rng.random((gh, gw), dtype=np.float32)
        field += amplitude * cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC)
        total += amplitude
        amplitude *= persistence
        freq *= 2
    field /= max(total, 1e-6)
    lo, hi = float(field.min()), float(field.max())
    return ((field - lo) / (hi - lo)) if hi - lo > 1e-6 else np.zeros_like(field)


class AnimatedPlume:
    """A fractal exhaust plume that can be sampled at any point in time.

    The plume is simulated once in a canonical frame (tailpipe on the left,
    drifting along +x) and then affine-warped into each output frame.  Motion
    comes from two independent effects:

    * **advection** -- a long noise strip scrolls past the cone, so texture
      appears to stream away from the pipe;
    * **turbulence** -- the displacement field used for the remap is a
      cosine-blend of two static noise fields, which makes the plume churn
      without any per-frame noise generation cost.
    """

    def __init__(self, rng: np.random.Generator, tiles: int = 3, noise_floor: float = 0.24) -> None:
        self.size = CANON
        self.tiles = int(tiles)
        self.noise_floor = float(noise_floor)

        self.shape = plume_shape(
            CANON,
            origin=CANON_PIPE,
            angle=0.0,
            length=CANON_LENGTH,
            width0=float(rng.uniform(0.035, 0.060)) * CANON,
            spread=float(rng.uniform(0.26, 0.42)),
            curl=float(rng.uniform(-0.45, 0.45)),
            shear=float(rng.uniform(-0.35, 0.35)),
        )
        self.shape = np.power(self.shape, float(rng.uniform(0.70, 0.90)))

        self.strip = _fbm_rect(rng, CANON, CANON * self.tiles, octaves=5, base_freq=4)
        self.wisp = _fbm_rect(rng, CANON, CANON * self.tiles, octaves=3, persistence=0.62, base_freq=10)

        self._warp_a = (fbm(rng, CANON, octaves=3, base_freq=3) - 0.5) * 2.0
        self._warp_b = (fbm(rng, CANON, octaves=3, base_freq=3) - 0.5) * 2.0
        self._warp_c = (fbm(rng, CANON, octaves=3, base_freq=3) - 0.5) * 2.0
        self._warp_d = (fbm(rng, CANON, octaves=3, base_freq=3) - 0.5) * 2.0
        self._grid_y, self._grid_x = np.mgrid[0:CANON, 0:CANON].astype(np.float32)
        self.turbulence = float(rng.uniform(8.0, 16.0))

    def _texture(self, phase: float) -> np.ndarray:
        """Scroll the noise strip so the plume texture streams away from the pipe."""
        span = CANON * (self.tiles - 1)
        offset = int(round((phase % 1.0) * span))
        window = self.strip[:, offset : offset + CANON]
        wisp = self.wisp[:, offset : offset + CANON]
        return window * (0.74 + 0.26 * wisp)

    def alpha(self, phase: float, opacity: float) -> np.ndarray:
        """Canonical-frame alpha map at animation *phase* (turns, ``[0, inf)``)."""
        texture = self._texture(phase)
        base = self.shape * (self.noise_floor + (1.0 - self.noise_floor) * texture)

        mix = 0.5 * (1.0 + math.cos(2.0 * math.pi * phase))
        dx = (self._warp_a * mix + self._warp_c * (1.0 - mix)) * self.turbulence
        dy = (self._warp_b * mix + self._warp_d * (1.0 - mix)) * self.turbulence
        base = cv2.remap(
            base,
            (self._grid_x + dx).astype(np.float32),
            (self._grid_y + dy).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        base = cv2.GaussianBlur(base, (0, 0), sigmaX=1.6)
        peak = float(base.max())
        if peak > 1e-6:
            base = base / peak
        return np.clip(base * float(opacity), 0.0, 1.0).astype(np.float32)

    def render(
        self,
        frame_shape: tuple[int, int],
        anchor: tuple[float, float],
        angle_deg: float,
        length_px: float,
        phase: float,
        opacity: float,
    ) -> np.ndarray:
        """Warp the canonical plume into a frame-sized alpha map.

        Args:
            frame_shape: ``(height, width)`` of the target frame.
            anchor: Tailpipe position in frame pixels.
            angle_deg: Drift direction, degrees clockwise from +x.
            length_px: Desired plume length in frame pixels.
            phase: Animation phase in turns.
            opacity: Peak alpha.

        Returns:
            ``(H, W)`` ``float32`` alpha in ``[0, 1]``.
        """
        height, width = int(frame_shape[0]), int(frame_shape[1])
        canonical = self.alpha(phase, opacity)
        scale = max(float(length_px) / CANON_LENGTH, 1e-3)
        matrix = cv2.getRotationMatrix2D(CANON_PIPE, -float(angle_deg), scale)
        matrix[0, 2] += float(anchor[0]) - CANON_PIPE[0]
        matrix[1, 2] += float(anchor[1]) - CANON_PIPE[1]
        return cv2.warpAffine(
            canonical, matrix, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        ).astype(np.float32)


#: Minimum luminance gap between the plume and the background it covers.
#: Matches the training generator's floor so the demo media looks like the
#: data the segmenter was trained on -- and so the smoke is actually visible.
MIN_DEMO_CONTRAST = 62.0


def pick_smoke_colour(
    frame: np.ndarray,
    alpha: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Choose a smoke colour that reads clearly against what it covers.

    Measures the background luminance *under the plume* and picks sooty black
    (unburnt diesel) over a bright background or pale grey/white (burning oil
    or coolant) over a dark one, always separated by at least
    :data:`MIN_DEMO_CONTRAST` grey levels.  A fixed dark grey looked like
    nothing at all when the plume happened to land on a tyre or a shadow.

    Args:
        frame: The background BGR frame.
        alpha: The plume alpha map.
        rng: Seeded generator for the within-band jitter.

    Returns:
        A BGR tuple.
    """
    weight = np.clip(alpha, 0.0, 1.0)
    luma = frame.astype(np.float32).mean(axis=2)
    total = float(weight.sum())
    background = float((luma * weight).sum() / total) if total > 1e-3 else float(luma.mean())

    if background >= 118.0:  # bright scene -> sooty black diesel smoke
        value = float(np.clip(background - MIN_DEMO_CONTRAST - rng.uniform(0.0, 26.0), 14.0, 96.0))
    else:  # dark scene -> pale oil/coolant smoke
        value = float(np.clip(background + MIN_DEMO_CONTRAST + rng.uniform(0.0, 26.0), 150.0, 238.0))
    return (value + 8.0, value + 2.0, value - 5.0)  # a touch of blue, like real exhaust


def composite_plume(
    frame: np.ndarray,
    alpha: np.ndarray,
    colour_bgr: Sequence[float],
    luminance_jitter: float = 0.0,
) -> np.ndarray:
    """Alpha-blend a smoke colour onto *frame*."""
    colour = np.array(colour_bgr, dtype=np.float32) + float(luminance_jitter)
    colour = np.clip(colour, 0.0, 255.0)
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    out = frame.astype(np.float32) * (1.0 - a) + colour[None, None, :] * a
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Pan / zoom camera
# --------------------------------------------------------------------------- #

def _ease(t: float) -> float:
    """Smoothstep, so the virtual camera accelerates and settles gently."""
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class PanZoomCamera:
    """Turns a still photograph into a moving shot of a fixed output size.

    Args:
        src_shape: ``(height, width)`` of the source photograph.
        out_size: ``(width, height)`` of the output frames.
        zoom: ``(start, end)`` window scale, 1.0 = the largest window that fits
            the output aspect ratio.
        pan: ``(start, end)`` centre offsets as a fraction of the slack space,
            each ``(dx, dy)`` in ``[-1, 1]``.
    """

    def __init__(
        self,
        src_shape: tuple[int, int],
        out_size: tuple[int, int] = VIDEO_SIZE,
        focus: tuple[float, float] | None = None,
        zoom: tuple[float, float] = (0.95, 0.74),
        pan: tuple[tuple[float, float], tuple[float, float]] = ((-0.30, 0.12), (0.34, -0.15)),
        pan_amplitude: float = 0.35,
    ) -> None:
        self.src_h, self.src_w = int(src_shape[0]), int(src_shape[1])
        self.out_w, self.out_h = int(out_size[0]), int(out_size[1])
        self.aspect = self.out_w / self.out_h
        self.base_w = min(self.src_w, self.src_h * self.aspect)
        self.base_h = self.base_w / self.aspect
        self.zoom = zoom
        self.pan = pan
        self.pan_amplitude = float(pan_amplitude)
        self.focus = focus if focus is not None else (self.src_w / 2.0, self.src_h / 2.0)

    def window(self, t: float) -> tuple[float, float, float, float]:
        """Crop window ``(x0, y0, w, h)`` in source pixels at normalised time *t*.

        The window is centred on :attr:`focus` (the subject vehicle) with a
        gentle drift on top, so zooming in never walks the subject -- or its
        tailpipe -- out of shot.
        """
        e = _ease(t)
        z = self.zoom[0] + (self.zoom[1] - self.zoom[0]) * e
        w = max(32.0, self.base_w * z)
        h = max(32.0, self.base_h * z)

        slack_x = max(0.0, (self.src_w - w) / 2.0)
        slack_y = max(0.0, (self.src_h - h) / 2.0)
        dx = self.pan[0][0] + (self.pan[1][0] - self.pan[0][0]) * e
        dy = self.pan[0][1] + (self.pan[1][1] - self.pan[0][1]) * e

        cx = self.focus[0] + dx * slack_x * self.pan_amplitude
        cy = self.focus[1] + dy * slack_y * self.pan_amplitude
        x0 = min(max(cx - w / 2.0, 0.0), max(0.0, self.src_w - w))
        y0 = min(max(cy - h / 2.0, 0.0), max(0.0, self.src_h - h))
        return x0, y0, w, h

    def frame(self, image: np.ndarray, t: float) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Render the shot at time *t*.

        Returns:
            ``(frame, (x0, y0, scale))`` -- the mapping from source pixels to
            output pixels is ``out = (src - (x0, y0)) * scale``.
        """
        x0, y0, w, h = self.window(t)
        xi, yi = int(round(x0)), int(round(y0))
        wi = max(2, min(int(round(w)), self.src_w - xi))
        hi = max(2, min(int(round(h)), self.src_h - yi))
        patch = image[yi : yi + hi, xi : xi + wi]
        frame = cv2.resize(patch, (self.out_w, self.out_h), interpolation=cv2.INTER_CUBIC)
        return frame, (float(xi), float(yi), self.out_w / float(wi))

    @staticmethod
    def map_box(box: dict[str, int], mapping: tuple[float, float, float]) -> dict[str, float]:
        """Map a source-space ``{'x','y','w','h'}`` box into output space."""
        x0, y0, scale = mapping
        return {
            "x": (box["x"] - x0) * scale,
            "y": (box["y"] - y0) * scale,
            "w": box["w"] * scale,
            "h": box["h"] * scale,
        }


# --------------------------------------------------------------------------- #
# Video writing with codec fallback
# --------------------------------------------------------------------------- #

def write_video_with_fallback(
    frames: Sequence[np.ndarray],
    stem: Path,
    fps: int = VIDEO_FPS,
) -> tuple[Path, str]:
    """Write *frames*, trying each codec until one produces a readable file.

    Args:
        frames: BGR frames, all the same size.
        stem: Output path **without** a suffix.
        fps: Frame rate.

    Returns:
        ``(path, fourcc)`` of the file that verified successfully.

    Raises:
        RuntimeError: Every codec failed.
    """
    if not frames:
        raise RuntimeError("Refusing to write an empty video.")
    height, width = frames[0].shape[:2]
    errors: list[str] = []

    for fourcc_name, suffix in CODEC_CANDIDATES:
        path = stem.with_suffix(suffix)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), float(fps), (width, height))
        if not writer.isOpened():
            writer.release()
            errors.append(f"{fourcc_name}: VideoWriter would not open")
            continue
        for frame in frames:
            writer.write(frame)
        writer.release()

        ok, detail = _verify_video(path, len(frames), width, height)
        if ok:
            logger.info("Wrote %s with fourcc %s (%d frames).", path.name, fourcc_name, len(frames))
            # Remove a stale file from a different-suffix attempt.
            for _, other_suffix in CODEC_CANDIDATES:
                other = stem.with_suffix(other_suffix)
                if other != path and other.is_file():
                    other.unlink()
            return path, fourcc_name

        errors.append(f"{fourcc_name}: {detail}")
        if path.is_file():
            path.unlink()

    raise RuntimeError("No usable video codec on this machine: " + "; ".join(errors))


def _verify_video(path: Path, expected_frames: int, width: int, height: int) -> tuple[bool, str]:
    """Reopen a written video and check it decodes with the expected geometry."""
    if not path.is_file() or path.stat().st_size < 1024:
        return False, "file missing or suspiciously small"
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False, "could not reopen"
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        decoded = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
    finally:
        cap.release()

    if (got_w, got_h) != (width, height):
        return False, f"geometry {got_w}x{got_h} != {width}x{height}"
    # Containers occasionally round the frame count; a couple of frames of
    # slack is normal, a large gap means the stream is broken.
    if abs(decoded - expected_frames) > 2:
        return False, f"decoded {decoded} frames, expected {expected_frames}"
    if reported and abs(reported - expected_frames) > 2:
        return False, f"header reports {reported} frames, expected {expected_frames}"
    return True, f"{decoded} frames"


# --------------------------------------------------------------------------- #
# Sample construction
# --------------------------------------------------------------------------- #

#: A plume is anchored just under the bumper, so the chosen vehicle needs at
#: least this much of the frame height below its box or the tailpipe -- and
#: therefore the whole plume -- would sit outside the picture.
MIN_BOTTOM_MARGIN = 0.08


def _pick_vehicle(
    detector: VehicleDetector,
    image: np.ndarray,
    prefer: Sequence[str],
    min_bottom_margin: float = MIN_BOTTOM_MARGIN,
) -> dict[str, Any] | None:
    """Return the best vehicle to hang a plume on.

    Prefers the requested classes, requires enough room below the box for the
    exhaust plume to be visible, and then takes the largest candidate.  Falls
    back to the largest detection of any class if nothing qualifies.
    """
    detections = detector.detect(image, conf=0.30)
    if not detections:
        return None
    height = image.shape[0]

    def has_room(d: dict[str, Any]) -> bool:
        bottom = d["bbox"]["y"] + d["bbox"]["h"]
        return (height - bottom) / max(height, 1) >= min_bottom_margin

    by_area = lambda d: d["bbox"]["w"] * d["bbox"]["h"]  # noqa: E731
    preferred = [d for d in detections if d["vehicle_type"] in prefer] or detections
    roomy = [d for d in preferred if has_room(d)]
    return max(roomy or preferred, key=by_area)


# Drift directions, in degrees clockwise from +x (so positive angles point
# *down* the image).  Exhaust leaves the pipe outward and sinks before it
# disperses, and -- critically -- the analysis pipeline only looks inside the
# exhaust ROI (the lower band of the vehicle box, expanded down and sideways).
# A plume drifting upward over the roof would be invisible to the very
# detector these samples exist to exercise, so the candidate set is restricted
# to outward-and-downward directions, ordered most-plausible first.
CANDIDATE_ANGLES_RIGHT: tuple[float, ...] = (30.0, 55.0, 80.0, 8.0)
CANDIDATE_ANGLES_LEFT: tuple[float, ...] = (150.0, 125.0, 100.0, 172.0)
#: Tailpipe position along the vehicle box, as a fraction of its width.
SIDE_LEFT, SIDE_RIGHT = 0.14, 0.86
#: A placement is accepted once the plume fills this much of the exhaust ROI.
#: Comfortably above the pipeline's 3% reporting floor.
TARGET_ROI_COVERAGE = 0.12


def _anchor_for(box: dict[str, float], side_fx: float, frame_shape: tuple[int, int]) -> tuple[float, float]:
    """Tailpipe position for a vehicle box, clamped inside the frame."""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    x = box["x"] + box["w"] * float(side_fx)
    y = box["y"] + box["h"] * 0.93
    return (
        float(min(max(x, 0.04 * width), 0.96 * width)),
        float(min(max(y, 0.12 * height), 0.90 * height)),
    )


def _roi_coverage(alpha: np.ndarray, roi: dict[str, int]) -> float:
    """Fraction of the exhaust ROI covered by the plume."""
    x0, y0 = int(roi["x"]), int(roi["y"])
    x1, y1 = x0 + int(roi["w"]), y0 + int(roi["h"])
    patch = alpha[y0:y1, x0:x1]
    return float((patch > 0.25).mean()) if patch.size else 0.0


def choose_placement(
    plume: "AnimatedPlume",
    frame_shape: tuple[int, int],
    box: dict[str, float],
    length_px: float,
    phase: float = 0.35,
    opacity: float = 0.85,
) -> tuple[float, float, float]:
    """Find a tailpipe side and drift angle that puts the plume in the ROI.

    Anchoring naively at "the rear" fails whenever the vehicle sits against
    the edge of the frame -- the plume drifts straight out of shot and the
    sample ends up with no visible smoke at all.  This walks a side/angle grid
    ordered by physical plausibility (the roomier side of the vehicle first,
    outward-and-downward angles first) and keeps the first placement that
    fills :data:`TARGET_ROI_COVERAGE` of the exhaust ROI, falling back to the
    best it saw.

    Args:
        plume: The plume to place.
        frame_shape: ``(height, width)`` of the target frame.
        box: Vehicle box in frame coordinates.
        length_px: Plume length in pixels.
        phase: Animation phase to probe with.
        opacity: Peak alpha to probe with.

    Returns:
        ``(side_fraction, angle_degrees, roi_coverage)``.
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    roi = VehicleDetector.exhaust_roi({k: int(round(v)) for k, v in box.items()}, (height, width))

    # Whichever side of the vehicle has more clear frame beside it goes first.
    space_left = box["x"]
    space_right = width - (box["x"] + box["w"])
    sides = (
        ((SIDE_LEFT, CANDIDATE_ANGLES_LEFT), (SIDE_RIGHT, CANDIDATE_ANGLES_RIGHT))
        if space_left >= space_right
        else ((SIDE_RIGHT, CANDIDATE_ANGLES_RIGHT), (SIDE_LEFT, CANDIDATE_ANGLES_LEFT))
    )

    best: tuple[float, float, float] = (sides[0][0], sides[0][1][0], -1.0)
    for side_fx, angles in sides:
        anchor = _anchor_for(box, side_fx, (height, width))
        for angle in angles:
            alpha = plume.render((height, width), anchor, angle, length_px, phase, opacity)
            coverage = _roi_coverage(alpha, roi)
            # Strictly-greater keeps the earlier (more plausible) candidate on
            # ties, but we still scan the whole grid: stopping at the first
            # "good enough" hit produced cramped plumes jammed into a frame
            # corner whenever the vehicle filled the shot.
            if coverage > best[2]:
                best = (side_fx, angle, coverage)
    if best[2] < TARGET_ROI_COVERAGE:
        logger.warning(
            "Best plume placement only fills %.1f%% of the exhaust ROI (target %.0f%%).",
            best[2] * 100.0, TARGET_ROI_COVERAGE * 100.0,
        )
    return best


def build_video_sample(
    source_image: Path,
    stem: Path,
    detector: VehicleDetector,
    with_smoke: bool,
    seed: int,
    seconds: float = VIDEO_SECONDS,
    fps: int = VIDEO_FPS,
) -> dict[str, Any]:
    """Render one pan-and-zoom clip, optionally with an animated exhaust plume."""
    image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read source photo {source_image}")

    rng = np.random.default_rng(seed)
    vehicle = _pick_vehicle(detector, image, prefer=("truck", "bus", "car"))
    if vehicle is None:
        raise RuntimeError(f"No vehicle detected in {source_image.name}; pick a different source photo.")

    focus = (
        vehicle["bbox"]["x"] + vehicle["bbox"]["w"] / 2.0,
        vehicle["bbox"]["y"] + vehicle["bbox"]["h"] * 0.60,
    )
    camera = PanZoomCamera(image.shape[:2], VIDEO_SIZE, focus=focus)
    plume = AnimatedPlume(rng) if with_smoke else None
    smoke_colour: tuple[float, float, float] = (70.0, 66.0, 60.0)

    total = int(round(seconds * fps))

    # Pick the tailpipe side and drift direction once, from the middle of the
    # shot, so the plume does not jump around between frames.
    side_fx, angle, coverage = (SIDE_RIGHT, CANDIDATE_ANGLES_RIGHT[0], 0.0)
    if plume is not None:
        probe_frame, probe_mapping = camera.frame(image, 0.5)
        probe_box = PanZoomCamera.map_box(vehicle["bbox"], probe_mapping)
        side_fx, angle, coverage = choose_placement(
            plume, probe_frame.shape[:2], probe_box, length_px=probe_box["w"] * 1.05
        )
        probe_alpha = plume.render(
            probe_frame.shape[:2], _anchor_for(probe_box, side_fx, probe_frame.shape[:2]),
            angle, probe_box["w"] * 1.05, phase=0.35, opacity=0.85,
        )
        smoke_colour = pick_smoke_colour(probe_frame, probe_alpha, rng)
        logger.info(
            "%s: plume placed at side=%.2f angle=%.0f deg (fills %.1f%% of the exhaust ROI).",
            stem.name, side_fx, angle, coverage * 100.0,
        )

    frames: list[np.ndarray] = []
    for index in range(total):
        t = index / max(total - 1, 1)
        frame, mapping = camera.frame(image, t)

        if plume is not None:
            box = PanZoomCamera.map_box(vehicle["bbox"], mapping)
            anchor = _anchor_for(box, side_fx, frame.shape[:2])
            # Gentle puffing: the plume swells and shrinks as the engine loads up.
            cycle = 0.5 * (1.0 + math.sin(2.0 * math.pi * (t * 2.0)))
            opacity = 0.76 + 0.19 * cycle
            length = box["w"] * (0.95 + 0.35 * cycle)
            alpha = plume.render(
                frame.shape[:2], anchor, angle, length,
                phase=t * 2.4, opacity=opacity,
            )
            frame = composite_plume(frame, alpha, smoke_colour, luminance_jitter=6.0 * (cycle - 0.5))

        # A touch of sensor noise so the clip does not look like a slideshow.
        noise = rng.normal(0.0, 2.2, size=frame.shape).astype(np.float32)
        frames.append(np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8))

    path, fourcc = write_video_with_fallback(frames, stem, fps=fps)

    # Sanity: does the detector still see a vehicle once the shot is moving?
    hits = sum(1 for i in (0, total // 3, total // 2, 2 * total // 3, total - 1) if detector.detect(frames[i]))
    logger.info("%s: vehicles detected in %d/5 probe frames.", path.name, hits)

    return {
        "path": str(path),
        "fourcc": fourcc,
        "frames": total,
        "fps": fps,
        "size": f"{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}",
        "source_photo": source_image.name,
        "seed_vehicle": vehicle["vehicle_type"],
        "detector_probe_hits": f"{hits}/5",
        "smoke": bool(with_smoke),
    }


def build_image_sample(
    source_image: Path,
    out_path: Path,
    detector: VehicleDetector,
    with_smoke: bool,
    seed: int,
) -> dict[str, Any]:
    """Write a still sample, optionally with a composited exhaust plume."""
    image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read source photo {source_image}")

    info: dict[str, Any] = {
        "path": str(out_path),
        "source_photo": source_image.name,
        "size": f"{image.shape[1]}x{image.shape[0]}",
        "smoke": bool(with_smoke),
    }

    if with_smoke:
        rng = np.random.default_rng(seed)
        vehicle = _pick_vehicle(detector, image, prefer=("bus", "truck", "car"))
        if vehicle is None:
            raise RuntimeError(f"No vehicle detected in {source_image.name}.")
        box = {k: float(v) for k, v in vehicle["bbox"].items()}
        plume = AnimatedPlume(rng)
        length_px = box["w"] * 1.05
        side_fx, angle, coverage = choose_placement(
            plume, image.shape[:2], box, length_px=length_px, phase=0.37, opacity=0.90
        )
        logger.info(
            "%s: plume placed at side=%.2f angle=%.0f deg (fills %.1f%% of the exhaust ROI).",
            out_path.name, side_fx, angle, coverage * 100.0,
        )
        alpha = plume.render(
            image.shape[:2], _anchor_for(box, side_fx, image.shape[:2]), angle,
            length_px=length_px, phase=0.37, opacity=0.90,
        )
        image = composite_plume(image, alpha, pick_smoke_colour(image, alpha, rng))
        info["seed_vehicle"] = vehicle["vehicle_type"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"Could not write {out_path}")

    detections = detector.detect(cv2.imread(str(out_path), cv2.IMREAD_COLOR))
    info["detected_vehicles"] = len(detections)
    logger.info("Wrote %s (%d vehicles detected).", out_path.name, len(detections))
    return info


README_TEMPLATE = """# AutoSmokeGuard sample media

Test fixtures for the analysis pipeline. Drop any of these into the uploader,
or run `python -m mlcore.selftest` to analyse all of them at once.

> **The vehicles are real; the smoke is not.**
> Every file below is derived from a real photograph (the COCO128 sample set,
> which ships with Ultralytics) so the YOLO vehicle detector has genuine
> vehicles to find. The exhaust plumes are **synthesised** by the same
> fractal-Brownian-motion plume model used to build the training set
> (`mlcore/training/synth_dataset.py`). No real vehicle in these files was
> actually emitting smoke. They exist to demonstrate and regression-test the
> pipeline, not to prove real-world accuracy.

| File | What it is | Expected analysis outcome |
| --- | --- | --- |
| `{truck_video}` | ~{seconds:.0f} s, {video_size}, {fps} fps. A slow pan-and-zoom over a real street photo of {truck_vehicle}s, with an animated exhaust plume composited at the rear of the largest vehicle. | Vehicles detected on most sampled frames; **at least one smoke region**, typically `moderate`/`high` severity. |
| `{car_video}` | ~{seconds:.0f} s, {video_size}, {fps} fps. Same construction over a real street photo with a car, **no smoke added**. | Vehicles detected; **no smoke regions** (a stray low-severity hit is tolerated, see the self-test tolerance). |
| `{bus_image}` | The real `bus.jpg` street photo with a synthesised plume composited at the rear of the bus. | One bus detected; **at least one smoke region**. |
| `{street_image}` | An unmodified real COCO128 street photo. Nothing has been added or removed. | Vehicles detected; **no smoke regions**. |

## Provenance

| Output | Source photograph |
| --- | --- |
{provenance}

## Regenerating

```bash
python tools/make_samples.py            # rebuild everything
python tools/make_samples.py --seed 7   # different plume shapes
```

Videos are written with the `{fourcc}` fourcc. The generator tries `mp4v`,
then `avc1`, then falls back to `MJPG` in an `.avi` container, verifying after
each attempt that the file reopens with the expected frame count.

_Generated by `tools/make_samples.py` on {generated_at}._
"""


def build_all(out_dir: Path = SAMPLE_MEDIA_DIR, seed: int = 20240921) -> dict[str, Any]:
    """Generate every sample file plus the folder README.

    Returns:
        A manifest describing what was written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = VehicleDetector(MLConfig())
    if not detector.available:
        raise RuntimeError(
            "YOLO weights are unavailable; cannot verify that the samples contain "
            "detectable vehicles. Check ml_assets/yolo11n.pt."
        )

    coco = Path(COCO128_IMAGES_DIR)
    sources = {
        "truck_video": coco / "000000000257.jpg",   # street scene, two trucks
        "car_video": coco / "000000000094.jpg",     # street scene, cars
        "bus_image": Path(DATASETS_DIR) / "bus.jpg",  # the classic Ultralytics bus photo
        "street_image": coco / "000000000471.jpg",  # school bus on an empty street
    }
    missing = [str(p) for p in sources.values() if not p.is_file()]
    if missing:
        raise RuntimeError("Missing source photographs: " + ", ".join(missing))

    started = time.time()
    results: dict[str, Any] = {}

    results["truck_video"] = build_video_sample(
        sources["truck_video"], out_dir / "sample_truck_smoking", detector, with_smoke=True, seed=seed
    )
    results["car_video"] = build_video_sample(
        sources["car_video"], out_dir / "sample_car_clean", detector, with_smoke=False, seed=seed + 11
    )
    results["bus_image"] = build_image_sample(
        sources["bus_image"], out_dir / "sample_bus_smoking.jpg", detector, with_smoke=True, seed=seed + 23
    )
    results["street_image"] = build_image_sample(
        sources["street_image"], out_dir / "sample_street_clean.jpg", detector, with_smoke=False, seed=seed + 37
    )

    provenance = "\n".join(
        f"| `{Path(results[key]['path']).name}` | `{results[key]['source_photo']}` |"
        for key in ("truck_video", "car_video", "bus_image", "street_image")
    )
    readme = README_TEMPLATE.format(
        truck_video=Path(results["truck_video"]["path"]).name,
        car_video=Path(results["car_video"]["path"]).name,
        bus_image=Path(results["bus_image"]["path"]).name,
        street_image=Path(results["street_image"]["path"]).name,
        truck_vehicle=results["truck_video"]["seed_vehicle"],
        seconds=VIDEO_SECONDS,
        video_size=f"{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}",
        fps=VIDEO_FPS,
        fourcc=results["truck_video"]["fourcc"],
        provenance=provenance,
        generated_at=time.strftime("%Y-%m-%d"),
    )
    (out_dir / "README.md").write_text(readme)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": seed,
        "elapsed_seconds": round(time.time() - started, 2),
        "files": results,
    }
    logger.info("Sample media written to %s in %.1fs.", out_dir, manifest["elapsed_seconds"])
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python tools/make_samples.py",
        description="Generate demonstration media in backend/sample_media/.",
    )
    parser.add_argument("--out", type=str, default=str(SAMPLE_MEDIA_DIR))
    parser.add_argument("--seed", type=int, default=20240921)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    manifest = build_all(Path(args.out), seed=args.seed)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
