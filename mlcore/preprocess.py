"""Media loading, frame iteration and image enhancement helpers.

Everything here works on OpenCV-native BGR ``uint8`` arrays.  Nothing in this
module ever holds a whole video in memory.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Iterator, NamedTuple

import cv2
import numpy as np

from .config import IMAGE_SUFFIXES, VIDEO_SUFFIXES

logger = logging.getLogger("asg.ml")

#: Below this side length CLAHE/bilateral filtering does more harm than good.
_MIN_ENHANCE_SIDE = 32
#: Assumed frame rate when a container reports a nonsensical value.
_FALLBACK_FPS = 25.0


class MediaError(ValueError):
    """Raised when a media file cannot be opened or decoded."""


class LetterboxInfo(NamedTuple):
    """Geometry needed to map letterboxed coordinates back to the original."""

    scale: float
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int

    def to_original(self, x: float, y: float) -> tuple[float, float]:
        """Map a point from letterboxed space back to original-image space."""
        return (x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale


def media_kind(path: str | os.PathLike[str]) -> str:
    """Classify *path* as ``"image"``, ``"video"`` or ``"unknown"`` by suffix."""
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return "unknown"


def load_image(path: str | os.PathLike[str]) -> np.ndarray:
    """Read an image from disk as a BGR ``uint8`` array.

    Args:
        path: Filesystem path to the image.

    Returns:
        ``(H, W, 3)`` BGR ``uint8`` array.

    Raises:
        MediaError: The file is missing, unreadable or not a decodable image.
    """
    p = Path(path)
    if not p.is_file():
        raise MediaError(f"Image not found: {p}")

    frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if frame is None:
        # Non-ASCII paths and some containers trip cv2.imread; retry via numpy.
        try:
            buf = np.fromfile(str(p), dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except OSError:
            frame = None
    if frame is None:
        raise MediaError(f"Could not decode image: {p}")

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(frame)


def probe_video(path: str | os.PathLike[str]) -> dict[str, float | int]:
    """Read container metadata without decoding the whole stream.

    Frame counts reported by OpenCV are frequently wrong or zero for
    variable-frame-rate files, so a zero/negative count is reported as ``0``
    and callers must treat it as "unknown".

    Args:
        path: Filesystem path to the video.

    Returns:
        ``{'fps', 'frame_count', 'duration', 'width', 'height'}``.

    Raises:
        MediaError: The container could not be opened.
    """
    p = Path(path)
    if not p.is_file():
        raise MediaError(f"Video not found: {p}")

    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise MediaError(f"Could not open video: {p}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0.1 or fps > 1000.0:
            logger.warning("Video %s reports implausible fps=%s; assuming %.1f.", p.name, fps, _FALLBACK_FPS)
            fps = _FALLBACK_FPS

        raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        frame_count = int(raw_count) if math.isfinite(raw_count) and raw_count > 0 else 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if frame_count else 0.0
    finally:
        cap.release()

    return {
        "fps": round(fps, 4),
        "frame_count": frame_count,
        "duration": round(duration, 3),
        "width": width,
        "height": height,
    }


def iter_frames(
    path: str | os.PathLike[str],
    sample_rate: int = 5,
    max_frames: int = 300,
    max_seconds: float | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Stream every ``sample_rate``-th frame of a video.

    The generator decodes sequentially (never ``CAP_PROP_POS_FRAMES`` seeking,
    which is unreliable on compressed streams) and yields only the frames it is
    asked for, so memory stays flat regardless of video length.

    Args:
        path: Filesystem path to the video.
        sample_rate: Keep one frame out of every ``sample_rate``.  Values < 1
            are clamped to 1.
        max_frames: Stop after yielding this many frames.
        max_seconds: Stop once the source timestamp passes this many seconds.

    Yields:
        ``(frame_index, timestamp_seconds, frame_bgr)`` where ``frame_index`` is
        the index in the *source* stream.

    Raises:
        MediaError: The container could not be opened.
    """
    p = Path(path)
    if not p.is_file():
        raise MediaError(f"Video not found: {p}")

    sample_rate = max(1, int(sample_rate))
    max_frames = max(1, int(max_frames))

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        raise MediaError(f"Could not open video: {p}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0.1 or fps > 1000.0:
        fps = _FALLBACK_FPS

    index = 0
    emitted = 0
    consecutive_failures = 0
    try:
        while emitted < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                # A handful of decoder hiccups mid-stream are survivable; a run
                # of them means we reached the end (or a broken file).
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    break
                index += 1
                continue
            consecutive_failures = 0

            timestamp = index / fps
            if max_seconds is not None and timestamp > max_seconds:
                break

            if index % sample_rate == 0:
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                yield index, round(timestamp, 4), frame
                emitted += 1

            index += 1
    finally:
        cap.release()


def enhance(frame: np.ndarray) -> np.ndarray:
    """Mildly denoise and locally equalise a frame.

    Applies a small bilateral filter (edge-preserving denoise) followed by
    CLAHE on the L channel of LAB.  This lifts faint smoke out of shadowed
    exhaust regions without washing out vehicle edges that YOLO relies on.

    Tiny images (either side < 32 px) are returned untouched -- the filters are
    meaningless at that scale and CLAHE's 8x8 tiling would fail.

    Args:
        frame: BGR ``uint8`` array.

    Returns:
        A new enhanced BGR ``uint8`` array, or the input unchanged when the
        frame is too small / not 3-channel / enhancement failed.
    """
    if frame is None or frame.size == 0:
        return frame
    if frame.ndim != 3 or frame.shape[2] != 3:
        return frame
    h, w = frame.shape[:2]
    if h < _MIN_ENHANCE_SIDE or w < _MIN_ENHANCE_SIDE:
        return frame

    try:
        denoised = cv2.bilateralFilter(frame, d=5, sigmaColor=45, sigmaSpace=45)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        tile = max(2, min(8, h // 16, w // 16))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
        lab = cv2.merge((clahe.apply(l_chan), a_chan, b_chan))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except cv2.error as exc:  # pragma: no cover - defensive
        logger.warning("enhance() failed (%s); passing frame through unchanged.", exc)
        return frame


def letterbox(
    frame: np.ndarray,
    size: int | tuple[int, int] = 256,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize *frame* into a fixed canvas preserving aspect ratio.

    Args:
        frame: BGR ``uint8`` array.
        size: Target square side, or an explicit ``(width, height)``.
        color: Padding colour in BGR.

    Returns:
        ``(canvas, info)`` where ``info`` carries the scale and padding needed
        to map coordinates back with :meth:`LetterboxInfo.to_original`.
    """
    target_w, target_h = (size, size) if isinstance(size, int) else size
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        raise MediaError("Cannot letterbox an empty frame.")

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    if frame.ndim == 2:
        canvas = np.full((target_h, target_w), color[0], dtype=frame.dtype)
    else:
        channels = frame.shape[2]
        fill = np.array(color[:channels], dtype=frame.dtype)
        canvas = np.empty((target_h, target_w, channels), dtype=frame.dtype)
        canvas[:] = fill
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, LetterboxInfo(scale=scale, pad_x=pad_x, pad_y=pad_y, orig_w=w, orig_h=h)


def crop(frame: np.ndarray, box: dict[str, int]) -> np.ndarray:
    """Return a clamped copy of the ``{'x','y','w','h'}`` region of *frame*."""
    h, w = frame.shape[:2]
    x0 = max(0, min(int(box["x"]), w - 1))
    y0 = max(0, min(int(box["y"]), h - 1))
    x1 = max(x0 + 1, min(int(box["x"]) + int(box["w"]), w))
    y1 = max(y0 + 1, min(int(box["y"]) + int(box["h"]), h))
    return np.ascontiguousarray(frame[y0:y1, x0:x1])
