"""Vehicle detection built on an Ultralytics YOLO COCO model.

The detector is deliberately forgiving: a missing or corrupt weights file
degrades to "no detections" with a loud log line rather than taking the whole
request down.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_YOLO_WEIGHTS, MLConfig, device_string, get_device

logger = logging.getLogger("asg.ml")

#: COCO class id -> the label the product uses.
COCO_VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

#: Fraction of the vehicle box height, measured from the bottom, that can
#: plausibly contain a tailpipe.
EXHAUST_BAND_FRACTION = 0.45
#: How far the exhaust ROI is grown sideways, as a fraction of vehicle width.
EXHAUST_SIDE_EXPANSION = 0.30
#: How far the exhaust ROI is grown downward, as a fraction of the band height.
#: REVERTED from 1.00 to 0.30 (see measurement notes below). On the annotated
#: PoVSSeg test set (150 images, 138 with YOLO detections), raising to 1.00
#: improved median smoke coverage from 27.1% to 94.9% and density (coverage /
#: ROI area) from 4.29 to 10.25, both peaking at 1.00 across the range 0.30-1.50.
#: An upward-expansion term was measured and rejected: 81-93% of annotated smoke
#: lies below the vehicle box, trucks/buses show less above the roofline than
#: cars, and this is a ground-level plume, not a roof stack.
#:
#: BLOCKER: The PoVSSeg test set contains only images with smoke present, so it
#: cannot measure false positives. When 1.00 was applied to the live pipeline on
#: clean vehicles, sample_car_clean.mp4 regressed from 0/49 to 24/49 false smoke
#: detections (0% to 49%, against a 10% tolerance), failing selftest.
#: sample_truck_smoking.mp4 also fell from 29 confirmed regions to 23. The root
#: cause is segmenter precision (~0.15 on small plumes): a larger crop mostly adds
#: road surface for the model to over-paint. Re-apply 1.00 only after segmenter
#: precision is fixed, and re-measure false positives on clean vehicles at the
#: same time (not coverage alone).
EXHAUST_DOWN_EXPANSION = 0.30

# Ultralytics is slow to construct and holds GPU state, so one instance per
# weights path is shared process-wide.  Guarded because Django workers are
# threaded.
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_yolo(weights: str, device: str) -> Any | None:
    """Load (or fetch from cache) an Ultralytics model for *weights*.

    Returns ``None`` when the weights are missing or unreadable.
    """
    key = f"{os.path.abspath(weights)}|{device}"
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        path = Path(weights)
        if not path.is_file():
            logger.error("YOLO weights not found at %s -- detection disabled.", path)
            _MODEL_CACHE[key] = None
            return None

        try:
            # Imported lazily: ultralytics pulls in torchvision and is not free.
            os.environ.setdefault("YOLO_VERBOSE", "0")
            from ultralytics import YOLO  # noqa: PLC0415

            model = YOLO(str(path))
            model.to(device)
        except Exception as exc:  # noqa: BLE001 - never let a bad file kill a request
            logger.error("Failed to load YOLO weights %s: %s", path, exc)
            _MODEL_CACHE[key] = None
            return None

        logger.info("Loaded YOLO weights %s on %s.", path.name, device)
        _MODEL_CACHE[key] = model
        return model


def clear_model_cache() -> None:
    """Drop every cached YOLO model (used by tests and by hot-reload paths)."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


class VehicleDetector:
    """Thin wrapper around Ultralytics YOLO restricted to COCO vehicle classes."""

    def __init__(self, config: MLConfig | None = None) -> None:
        """Create a detector.

        Args:
            config: Pipeline configuration.  Defaults to :class:`MLConfig`
                defaults (bundled ``yolo11n.pt``, auto device).
        """
        self.config = config or MLConfig()
        self.weights = str(self.config.yolo_weights or DEFAULT_YOLO_WEIGHTS)
        self.device = device_string(get_device(self.config.device))
        self._model: Any | None = None
        self._loaded = False

    # -- model lifecycle --------------------------------------------------- #

    @property
    def available(self) -> bool:
        """``True`` when weights loaded successfully and detection can run."""
        return self._ensure_model() is not None

    def _ensure_model(self) -> Any | None:
        if not self._loaded:
            self._model = _load_yolo(self.weights, self.device)
            self._loaded = True
        return self._model

    def warmup(self, size: int = 640) -> bool:
        """Run one throwaway inference so the first real call is not slow.

        Returns:
            ``True`` if the warmup inference completed.
        """
        model = self._ensure_model()
        if model is None:
            return False
        try:
            blank = np.zeros((size, size, 3), dtype=np.uint8)
            model.predict(blank, verbose=False, device=self.device, conf=0.99)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLO warmup failed: %s", exc)
            return False

    # -- inference --------------------------------------------------------- #

    def detect(self, frame_bgr: np.ndarray, conf: float | None = None) -> list[dict[str, Any]]:
        """Detect vehicles in a single BGR frame.

        Args:
            frame_bgr: ``(H, W, 3)`` BGR ``uint8`` array.
            conf: Confidence floor; defaults to ``config.conf_threshold``.

        Returns:
            A list of ``{'bbox': {'x','y','w','h'}, 'vehicle_type': str,
            'confidence': float}`` sorted by descending confidence.  Empty when
            the model is unavailable or nothing matched -- this method never
            raises.
        """
        model = self._ensure_model()
        if model is None or frame_bgr is None or frame_bgr.size == 0:
            return []

        threshold = float(self.config.conf_threshold if conf is None else conf)
        try:
            results = model.predict(
                frame_bgr,
                verbose=False,
                device=self.device,
                conf=threshold,
                classes=sorted(COCO_VEHICLE_CLASSES),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLO inference failed on a frame: %s", exc)
            return []

        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        try:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
            confs = boxes.conf.detach().cpu().numpy().astype(float)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read YOLO boxes: %s", exc)
            return []

        frame_h, frame_w = frame_bgr.shape[:2]
        detections: list[dict[str, Any]] = []
        for (x1, y1, x2, y2), cls_id, score in zip(xyxy, cls_ids, confs):
            label = COCO_VEHICLE_CLASSES.get(int(cls_id))
            if label is None or score < threshold:
                continue
            x = int(max(0, min(round(float(x1)), frame_w - 1)))
            y = int(max(0, min(round(float(y1)), frame_h - 1)))
            w = int(max(1, min(round(float(x2)) - x, frame_w - x)))
            h = int(max(1, min(round(float(y2)) - y, frame_h - y)))
            detections.append(
                {
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                    "vehicle_type": label,
                    "confidence": round(float(score), 4),
                }
            )

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    # -- geometry ---------------------------------------------------------- #

    @staticmethod
    def exhaust_roi(bbox: dict[str, int], frame_shape: tuple[int, ...]) -> dict[str, int]:
        """Compute the region where exhaust smoke is expected for a vehicle.

        Tailpipes sit low and at the rear of a vehicle, and the plume drifts
        down and outward before dispersing.  The ROI is therefore the bottom
        :data:`EXHAUST_BAND_FRACTION` of the detection box, grown sideways by
        :data:`EXHAUST_SIDE_EXPANSION` of the vehicle width on each side and
        downward by :data:`EXHAUST_DOWN_EXPANSION` of the band height, then
        clamped to the frame.

        Args:
            bbox: Vehicle box as ``{'x','y','w','h'}``.
            frame_shape: The frame's ``.shape`` (``(H, W)`` or ``(H, W, C)``).

        Returns:
            The ROI as ``{'x','y','w','h'}``, guaranteed non-degenerate and
            inside the frame.
        """
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        vx, vy = int(bbox["x"]), int(bbox["y"])
        vw, vh = max(1, int(bbox["w"])), max(1, int(bbox["h"]))

        band_h = max(1, int(round(vh * EXHAUST_BAND_FRACTION)))
        top = vy + vh - band_h

        pad_x = int(round(vw * EXHAUST_SIDE_EXPANSION))
        pad_down = int(round(band_h * EXHAUST_DOWN_EXPANSION))

        x0 = max(0, vx - pad_x)
        y0 = max(0, top)
        x1 = min(frame_w, vx + vw + pad_x)
        y1 = min(frame_h, vy + vh + pad_down)

        # Clamp degenerate results (tiny boxes at the frame edge).
        if x1 <= x0:
            x0, x1 = max(0, min(vx, frame_w - 1)), min(frame_w, max(vx + 1, x0 + 1))
        if y1 <= y0:
            y0, y1 = max(0, min(top, frame_h - 1)), min(frame_h, max(top + 1, y0 + 1))

        return {"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0)}
