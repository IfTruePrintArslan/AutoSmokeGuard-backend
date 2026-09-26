"""Training, dataset-synthesis and evaluation entry points for :mod:`mlcore`.

These modules are only needed to *produce* ``ml_assets/smoke_unet.pt``; the
inference package never imports them.
"""

from __future__ import annotations

__all__ = ["synth_dataset", "train_unet", "evaluate"]
