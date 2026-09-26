"""Drawing vehicle boxes, smoke masks and severity chips onto frames.

Colours follow the AutoSmokeGuard product palette.  Remember OpenCV works in
**BGR**, so every constant here is stored pre-swapped.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

logger = logging.getLogger("asg.ml")

# --------------------------------------------------------------------------- #
# Palette (product hex -> OpenCV BGR)
# --------------------------------------------------------------------------- #
SEVERITY_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "low": (128, 222, 74),       # #4ade80
    "moderate": (36, 191, 251),  # #fbbf24
    "high": (113, 113, 248),     # #f87171
}
VEHICLE_COLOR_BGR: tuple[int, int, int] = (240, 232, 226)  # #e2e8f0 slate-200
PANEL_COLOR_BGR: tuple[int, int, int] = (26, 22, 15)       # #0f172a slate-900
TEXT_LIGHT_BGR: tuple[int, int, int] = (248, 250, 252)     # #f8fafc
TEXT_DARK_BGR: tuple[int, int, int] = (23, 23, 23)

#: Translucency of the smoke overlay, as required by the product spec.
MASK_ALPHA = 0.45

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _scale_for(frame: np.ndarray) -> float:
    """Font/line scale so annotations stay legible on 360p and on 4K alike."""
    shortest = min(frame.shape[:2])
    return float(np.clip(shortest / 640.0, 0.55, 2.0))


def _bbox_of(vehicle: Mapping[str, Any]) -> dict[str, int] | None:
    """Accept either ``bbox`` or ``bounding_box`` and normalise to ints."""
    box = vehicle.get("bbox") or vehicle.get("bounding_box")
    if not isinstance(box, Mapping):
        return None
    try:
        return {
            "x": int(box["x"]),
            "y": int(box["y"]),
            "w": max(1, int(box["w"])),
            "h": max(1, int(box["h"])),
        }
    except (KeyError, TypeError, ValueError):
        return None


def draw_text_with_backing(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.5,
    thickness: int = 1,
    fg: tuple[int, int, int] = TEXT_LIGHT_BGR,
    bg: tuple[int, int, int] = PANEL_COLOR_BGR,
    pad: int = 4,
    anchor: str = "bottom-left",
    bg_alpha: float = 1.0,
) -> tuple[int, int, int, int]:
    """Draw *text* on a filled backing rectangle so it is always readable.

    Args:
        frame: Target BGR image, modified in place.
        text: The string to render.
        origin: Anchor point in pixels.
        scale: ``cv2.putText`` font scale.
        thickness: Stroke thickness.
        fg: Text colour (BGR).
        bg: Backing rectangle colour (BGR).
        pad: Padding around the glyphs.
        anchor: ``"bottom-left"`` (default, *origin* is the text baseline's
            left end) or ``"top-left"`` (*origin* is the box's top-left).
        bg_alpha: Opacity of the backing rectangle.

    Returns:
        The drawn backing rectangle as ``(x0, y0, x1, y1)``.
    """
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    box_w = tw + 2 * pad
    box_h = th + baseline + 2 * pad

    if anchor == "top-left":
        x0, y0 = int(origin[0]), int(origin[1])
    else:
        x0, y0 = int(origin[0]), int(origin[1]) - box_h

    h, w = frame.shape[:2]
    x0 = max(0, min(x0, w - box_w)) if box_w <= w else 0
    y0 = max(0, min(y0, h - box_h)) if box_h <= h else 0
    x1, y1 = min(w, x0 + box_w), min(h, y0 + box_h)
    if x1 <= x0 or y1 <= y0:
        return (x0, y0, x1, y1)

    if bg_alpha >= 1.0:
        cv2.rectangle(frame, (x0, y0), (x1, y1), bg, thickness=-1)
    else:
        region = frame[y0:y1, x0:x1]
        tile = np.full_like(region, bg, dtype=np.uint8)
        cv2.addWeighted(tile, bg_alpha, region, 1.0 - bg_alpha, 0.0, dst=region)

    baseline_y = y0 + pad + th  # glyph baseline sits `th` below the top padding
    cv2.putText(frame, text, (x0 + pad, baseline_y), _FONT, scale, fg, thickness, cv2.LINE_AA)
    return (x0, y0, x1, y1)


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    roi: Mapping[str, int],
    color_bgr: tuple[int, int, int],
    alpha: float = MASK_ALPHA,
    threshold: float = 0.5,
    outline: bool = True,
) -> np.ndarray:
    """Blend a translucent, severity-tinted smoke mask onto *frame* in place.

    Args:
        frame: Target BGR image.
        mask: ``float`` probabilities in ``[0, 1]`` (or ``uint8`` 0-255) sized
            like the ROI; resized if it is not.
        roi: ``{'x','y','w','h'}`` placement of the mask in *frame*.
        color_bgr: Tint colour.
        alpha: Peak opacity of the overlay.
        threshold: Probabilities below this are not drawn.
        outline: Also stroke the plume contour for a crisp edge.

    Returns:
        *frame* (mutated).
    """
    if mask is None or mask.size == 0:
        return frame

    fh, fw = frame.shape[:2]
    x0 = max(0, int(roi.get("x", 0)))
    y0 = max(0, int(roi.get("y", 0)))
    x1 = min(fw, x0 + max(1, int(roi.get("w", 1))))
    y1 = min(fh, y0 + max(1, int(roi.get("h", 1))))
    if x1 <= x0 or y1 <= y0:
        return frame

    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.max() > 1.0 + 1e-6:
        m = m / 255.0
    target = (x1 - x0, y1 - y0)
    if (m.shape[1], m.shape[0]) != target:
        m = cv2.resize(m, target, interpolation=cv2.INTER_LINEAR)

    m = np.where(m >= float(threshold), m, 0.0).astype(np.float32)
    if not bool(np.any(m)):
        return frame

    region = frame[y0:y1, x0:x1].astype(np.float32)
    tint = np.empty_like(region)
    tint[:] = color_bgr
    weight = (m * float(alpha))[..., None]
    frame[y0:y1, x0:x1] = np.clip(region * (1.0 - weight) + tint * weight, 0, 255).astype(np.uint8)

    if outline:
        binary = (m >= float(threshold)).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            shifted = [c + np.array([[x0, y0]], dtype=c.dtype) for c in contours]
            cv2.drawContours(frame, shifted, -1, color_bgr, thickness=1, lineType=cv2.LINE_AA)
    return frame


def draw_legend(frame: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Draw the severity colour key in the bottom-right corner."""
    s = scale if scale is not None else _scale_for(frame)
    font_scale = 0.42 * s
    thickness = max(1, int(round(s)))
    pad = int(8 * s)
    swatch = int(12 * s)
    line_h = int(18 * s)

    entries = [("LOW", "low"), ("MODERATE", "moderate"), ("HIGH", "high")]
    widths = [cv2.getTextSize(label, _FONT, font_scale, thickness)[0][0] for label, _ in entries]
    title = "SMOKE SEVERITY"
    title_w = cv2.getTextSize(title, _FONT, font_scale, thickness)[0][0]

    box_w = max(max(widths) + swatch + 3 * pad, title_w + 2 * pad)
    box_h = line_h * (len(entries) + 1) + pad

    h, w = frame.shape[:2]
    x1, y1 = w - pad, h - pad
    x0, y0 = max(0, x1 - box_w), max(0, y1 - box_h)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return frame

    region = frame[y0:y1, x0:x1]
    panel = np.full_like(region, PANEL_COLOR_BGR, dtype=np.uint8)
    cv2.addWeighted(panel, 0.72, region, 0.28, 0.0, dst=region)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (90, 82, 70), 1, cv2.LINE_AA)

    cv2.putText(frame, title, (x0 + pad, y0 + line_h - int(4 * s)), _FONT, font_scale, TEXT_LIGHT_BGR, thickness, cv2.LINE_AA)
    for i, (label, key) in enumerate(entries, start=1):
        cy = y0 + line_h * i + line_h - int(6 * s)
        cv2.rectangle(
            frame,
            (x0 + pad, cy - swatch + int(2 * s)),
            (x0 + pad + swatch, cy + int(2 * s)),
            SEVERITY_COLORS_BGR[key],
            -1,
        )
        cv2.putText(frame, label, (x0 + pad + swatch + int(6 * s), cy + int(1 * s)), _FONT, font_scale, TEXT_LIGHT_BGR, thickness, cv2.LINE_AA)
    return frame


def draw(
    frame: np.ndarray,
    vehicles: Sequence[Mapping[str, Any]],
    *,
    frame_number: int | None = None,
    timestamp: float | None = None,
    stamp: str | None = None,
    legend: bool = True,
    mask_threshold: float = 0.5,
    copy: bool = True,
) -> np.ndarray:
    """Annotate a frame with vehicle boxes, smoke overlays and severity chips.

    Each entry of *vehicles* may carry a ``smoke`` mapping with ``mask``
    (``np.ndarray`` probabilities), ``roi`` (``{'x','y','w','h'}``),
    ``severity`` and ``intensity``.  Vehicles without ``smoke`` are drawn with
    the neutral box only.

    Args:
        frame: Source BGR image.
        vehicles: Detection dicts (``bbox``/``bounding_box``, ``vehicle_type``,
            ``confidence``, optional ``smoke``).
        frame_number: Shown in the top-left stamp.
        timestamp: Seconds into the media, shown in the top-left stamp.
        stamp: Explicit stamp text, overriding *frame_number*/*timestamp*.
        legend: Draw the severity colour key.
        mask_threshold: Probability cut-off for the overlay.
        copy: Annotate a copy (default) instead of mutating *frame*.

    Returns:
        The annotated BGR image.
    """
    if frame is None or frame.size == 0:
        return frame
    canvas = frame.copy() if copy else frame
    s = _scale_for(canvas)
    box_thickness = max(1, int(round(2 * s)))
    font_scale = 0.46 * s
    text_thickness = max(1, int(round(s)))

    # 1) Smoke overlays first, so boxes and text sit on top of them.
    for vehicle in vehicles or []:
        smoke = vehicle.get("smoke")
        if not isinstance(smoke, Mapping):
            continue
        mask = smoke.get("mask")
        roi = smoke.get("roi") or _bbox_of(vehicle)
        if mask is None or roi is None:
            continue
        color = SEVERITY_COLORS_BGR.get(str(smoke.get("severity", "low")), SEVERITY_COLORS_BGR["low"])
        overlay_mask(canvas, mask, roi, color, alpha=MASK_ALPHA, threshold=mask_threshold)

    # 2) Vehicle boxes, labels and severity chips.
    for vehicle in vehicles or []:
        box = _bbox_of(vehicle)
        if box is None:
            continue
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), VEHICLE_COLOR_BGR, box_thickness, cv2.LINE_AA)

        label = str(vehicle.get("vehicle_type", "vehicle"))
        conf = vehicle.get("confidence")
        text = f"{label} {float(conf) * 100:.0f}%" if conf is not None else label
        draw_text_with_backing(
            canvas,
            text,
            (x, max(0, y - int(2 * s))),
            scale=font_scale,
            thickness=text_thickness,
            fg=TEXT_DARK_BGR,
            bg=VEHICLE_COLOR_BGR,
            pad=int(4 * s),
            anchor="bottom-left",
        )

        smoke = vehicle.get("smoke")
        if isinstance(smoke, Mapping):
            severity = str(smoke.get("severity", "low"))
            color = SEVERITY_COLORS_BGR.get(severity, SEVERITY_COLORS_BGR["low"])
            intensity = smoke.get("intensity")
            chip = f"SMOKE {severity.upper()}"
            if intensity is not None:
                chip += f"  {float(intensity):.2f}"
            # Anchor the chip to the smoke ROI, not the vehicle box: on a wide
            # vehicle the plume can sit metres away from the box origin and a
            # chip stranded at the far corner is unreadable as a label.
            roi = smoke.get("roi") or box
            chip_x = int(roi.get("x", x))
            chip_y = min(canvas.shape[0] - 1, int(roi.get("y", y)) + int(roi.get("h", h)) + int(20 * s))
            draw_text_with_backing(
                canvas,
                chip,
                (chip_x, chip_y),
                scale=font_scale,
                thickness=text_thickness,
                fg=TEXT_DARK_BGR,
                bg=color,
                pad=int(4 * s),
                anchor="bottom-left",
            )

    # 3) Frame / timestamp stamp, top-left.
    if stamp is None and (frame_number is not None or timestamp is not None):
        parts: list[str] = []
        if frame_number is not None:
            parts.append(f"frame {int(frame_number)}")
        if timestamp is not None:
            parts.append(f"t={float(timestamp):.2f}s")
        stamp = "  |  ".join(parts)
    if stamp:
        draw_text_with_backing(
            canvas,
            stamp,
            (int(8 * s), int(8 * s)),
            scale=font_scale,
            thickness=text_thickness,
            fg=TEXT_LIGHT_BGR,
            bg=PANEL_COLOR_BGR,
            pad=int(5 * s),
            anchor="top-left",
            bg_alpha=0.78,
        )

    if legend:
        draw_legend(canvas, scale=s)
    return canvas


def save_frame(frame: np.ndarray, path: str | os.PathLike[str], quality: int = 90) -> str:
    """Write *frame* to *path*, creating parent directories.

    Args:
        frame: BGR image.
        path: Destination (``.jpg``/``.jpeg``/``.png``).
        quality: JPEG quality (ignored for PNG).

    Returns:
        The absolute path written.

    Raises:
        OSError: The encoder refused to write the file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if target.suffix.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif target.suffix.lower() == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    if not cv2.imwrite(str(target), frame, params):
        raise OSError(f"Failed to write image to {target}")
    return str(target.resolve())
