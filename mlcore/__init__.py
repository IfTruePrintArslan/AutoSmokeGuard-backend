"""AutoSmokeGuard machine-learning core.

A framework-agnostic package (no Django, no network I/O) that turns an image or
video of road traffic into structured vehicle-smoke-emission findings plus
annotated artifacts on disk.

Typical use from the web layer::

    from mlcore import MLConfig, analyze_media, warmup

    warmup()                                     # once, at worker startup
    cfg = MLConfig.from_dict(settings.ASG_ML)    # unknown keys are ignored
    result = analyze_media(upload_path, out_dir, cfg, progress_cb=report)

Command-line entry points:

* ``python -m mlcore.selftest`` -- end-to-end check over ``sample_media/``
* ``python -m mlcore.training.synth_dataset`` -- build the training set
* ``python -m mlcore.training.train_unet`` -- train the segmenter
* ``python -m mlcore.training.evaluate`` -- write ``ml_assets/metrics.json``
"""

from __future__ import annotations

import logging as _logging

# Must run before torch/torchvision/ultralytics are imported anywhere below.
from ._compat import ensure_stdlib as _ensure_stdlib

_ensure_stdlib()

# The library never configures handlers; the host application owns logging.
_logging.getLogger("asg.ml").addHandler(_logging.NullHandler())

from .config import (  # noqa: E402
    ML_ASSETS_DIR,
    SAMPLE_MEDIA_DIR,
    MLConfig,
    device_string,
    get_device,
)
from .detector import VehicleDetector  # noqa: E402
from .intensity import aggregate, classify_severity, extract_features, smoke_density  # noqa: E402
from .model import SmokeUNet  # noqa: E402
from .pipeline import MLPipelineError, analyze_media, warmup  # noqa: E402
from .segmenter import SmokeSegmenter  # noqa: E402

__version__ = "1.0.0"

__all__ = [
    "MLConfig",
    "MLPipelineError",
    "SmokeSegmenter",
    "SmokeUNet",
    "VehicleDetector",
    "ML_ASSETS_DIR",
    "SAMPLE_MEDIA_DIR",
    "__version__",
    "aggregate",
    "analyze_media",
    "classify_severity",
    "device_string",
    "extract_features",
    "get_device",
    "smoke_density",
    "warmup",
]
