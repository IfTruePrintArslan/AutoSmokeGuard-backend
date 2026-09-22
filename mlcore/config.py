"""Configuration objects, device selection and asset paths for :mod:`mlcore`.

This module holds the single source of truth for where model weights live and
how the rest of the package picks a torch device.  It must stay free of any
web-framework imports so the ML layer can be unit tested standalone.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import torch

logger = logging.getLogger("asg.ml")

# --------------------------------------------------------------------------- #
# Asset locations
# --------------------------------------------------------------------------- #

#: ``<repo>/backend/mlcore``
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
#: ``<repo>/backend`` -- the directory that owns ``ml_assets`` and ``sample_media``.
BACKEND_ROOT: Path = PACKAGE_ROOT.parent

#: Overridable via ``ASG_ML_ASSETS`` so a deployment can mount weights elsewhere.
ML_ASSETS_DIR: Path = Path(os.environ.get("ASG_ML_ASSETS") or (BACKEND_ROOT / "ml_assets"))

DATASETS_DIR: Path = ML_ASSETS_DIR / "datasets"
COCO128_DIR: Path = DATASETS_DIR / "coco128"
COCO128_IMAGES_DIR: Path = COCO128_DIR / "images" / "train2017"
COCO128_LABELS_DIR: Path = COCO128_DIR / "labels" / "train2017"
SYNTH_DATASET_DIR: Path = DATASETS_DIR / "smoke_synth"

DEFAULT_YOLO_WEIGHTS: Path = ML_ASSETS_DIR / "yolo11n.pt"
DEFAULT_SEGMENTER_WEIGHTS: Path = ML_ASSETS_DIR / "smoke_unet.pt"
TRAINING_CURVE_PATH: Path = ML_ASSETS_DIR / "training_curve.png"
METRICS_PATH: Path = ML_ASSETS_DIR / "metrics.json"

SAMPLE_MEDIA_DIR: Path = BACKEND_ROOT / "sample_media"

IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
VIDEO_SUFFIXES: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #

def _cpu_forced() -> bool:
    """Return ``True`` when ``ASG_FORCE_CPU`` asks us to stay on the CPU."""
    return str(os.environ.get("ASG_FORCE_CPU", "")).strip().lower() in {"1", "true", "yes", "on"}


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the best available torch device.

    Preference order is ``cuda`` > ``mps`` > ``cpu``.  Setting the environment
    variable ``ASG_FORCE_CPU=1`` overrides everything and returns ``cpu`` --
    useful for CI and for debugging MPS-specific numerical issues.

    Args:
        prefer: Optional explicit device string (``"cuda"``, ``"mps"``,
            ``"cpu"``, ``"cuda:1"``, ...).  Falls back to auto-selection when
            the requested backend is unavailable.

    Returns:
        A :class:`torch.device`.
    """
    if _cpu_forced():
        return torch.device("cpu")

    if prefer:
        wanted = str(prefer).strip().lower()
        if wanted not in {"", "auto"}:
            backend = wanted.split(":", 1)[0]
            if backend == "cuda" and torch.cuda.is_available():
                return torch.device(wanted)
            if backend == "mps" and torch.backends.mps.is_available():
                return torch.device("mps")
            if backend == "cpu":
                return torch.device("cpu")
            logger.warning("Requested device %r is unavailable; falling back to auto-selection.", prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_string(device: torch.device | str) -> str:
    """Normalise a device to the short string Ultralytics and our API expect."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda":
        return f"cuda:{dev.index}" if dev.index is not None else "cuda:0"
    return dev.type


# --------------------------------------------------------------------------- #
# MLConfig
# --------------------------------------------------------------------------- #

@dataclass
class MLConfig:
    """Tunable knobs for the whole analysis pipeline.

    Every field is a plain scalar or string so the Django settings layer can
    persist and pass these around without adapters.  Use :meth:`from_dict` to
    build one from an untrusted mapping -- unknown keys are ignored on purpose
    so the web layer can hand its settings dict straight through.
    """

    #: Path to Ultralytics COCO detection weights.
    yolo_weights: str = str(DEFAULT_YOLO_WEIGHTS)
    #: Path to the trained :class:`~mlcore.model.SmokeUNet` checkpoint.
    segmenter_weights: str = str(DEFAULT_SEGMENTER_WEIGHTS)
    #: ``None``/``"auto"`` means auto-select; otherwise ``"cuda"``/``"mps"``/``"cpu"``.
    device: str | None = None
    #: Minimum YOLO confidence for a vehicle detection to be kept.
    conf_threshold: float = 0.35
    #: Probability threshold applied to the segmenter's sigmoid output.
    mask_threshold: float = 0.5
    #: Process every Nth video frame.
    frame_sample_rate: int = 5
    #: Hard cap on the number of *processed* frames for one video.
    max_frames: int = 300
    #: Hard cap on how much wall-clock video we will read.
    max_video_seconds: int = 300
    #: Density strictly below this is ``"low"``.
    severity_low_max: float = 0.33
    #: Density strictly below this (and >= ``severity_low_max``) is ``"moderate"``.
    severity_moderate_max: float = 0.66
    #: Square input resolution of the segmentation network.
    input_size: int = 256
    #: Apply CLAHE + bilateral denoise before detection.
    enhance: bool = True

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "MLConfig":
        """Build an :class:`MLConfig` from a mapping, ignoring unknown keys.

        Values are coerced to the declared field types where that is safe, so a
        settings dict sourced from JSON/env strings still produces a usable
        config.  A value that cannot be coerced is dropped with a warning
        rather than raising.
        """
        if not d:
            return cls()

        known = {f.name: f for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in d.items():
            field_def = known.get(key)
            if field_def is None:
                continue
            try:
                kwargs[key] = _coerce(field_def.name, field_def.type, value)
            except (TypeError, ValueError):
                logger.warning("Ignoring MLConfig key %r with uncoercible value %r.", key, value)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of this config."""
        return asdict(self)

    # -- derived ----------------------------------------------------------- #

    def torch_device(self) -> torch.device:
        """Resolve :attr:`device` into a concrete :class:`torch.device`."""
        return get_device(self.device)

    def resolved_device_string(self) -> str:
        """Short device name (``"mps"``, ``"cpu"``, ``"cuda:0"``)."""
        return device_string(self.torch_device())

    def validate(self) -> "MLConfig":
        """Clamp nonsensical values into a usable range and return ``self``."""
        self.conf_threshold = float(min(max(self.conf_threshold, 0.01), 0.99))
        self.mask_threshold = float(min(max(self.mask_threshold, 0.01), 0.99))
        self.frame_sample_rate = max(1, int(self.frame_sample_rate))
        self.max_frames = max(1, int(self.max_frames))
        self.max_video_seconds = max(1, int(self.max_video_seconds))
        self.input_size = max(64, int(self.input_size) // 32 * 32)
        lo = float(min(max(self.severity_low_max, 0.01), 0.98))
        mo = float(min(max(self.severity_moderate_max, lo + 0.01), 0.99))
        self.severity_low_max, self.severity_moderate_max = lo, mo
        return self


_TRUTHY = {"1", "true", "yes", "on", "t", "y"}
_FALSEY = {"0", "false", "no", "off", "f", "n"}


def _coerce(name: str, declared: Any, value: Any) -> Any:
    """Best-effort coercion of *value* to the type declared for a dataclass field."""
    # ``from __future__ import annotations`` makes field types strings.
    hint = declared if isinstance(declared, str) else getattr(declared, "__name__", str(declared))

    if "bool" in hint:
        if isinstance(value, str):
            low = value.strip().lower()
            if low in _TRUTHY:
                return True
            if low in _FALSEY:
                return False
            raise ValueError(value)
        return bool(value)
    if "int" in hint and "float" not in hint:
        return int(value)
    if "float" in hint:
        return float(value)
    if "str" in hint:
        if value is None:
            if "None" in hint:
                return None
            raise ValueError(f"{name} may not be None")
        return str(value)
    return value
