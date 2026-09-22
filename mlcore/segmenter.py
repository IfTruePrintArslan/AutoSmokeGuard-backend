"""Smoke segmentation: trained :class:`~mlcore.model.SmokeUNet` with a
classical-computer-vision fallback.

The fallback exists so the product still produces plausible, honest output on a
machine where the checkpoint has not been trained or shipped.  Results always
carry the :attr:`SmokeSegmenter.mode` that produced them so the UI and the
report can say which one was used.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from .config import DEFAULT_SEGMENTER_WEIGHTS, MLConfig, device_string, get_device
from .model import SmokeUNet

logger = logging.getLogger("asg.ml")

#: Normalisation applied to network input.  Must match ``training/train_unet.py``.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: Classical-fallback scores are deliberately discounted: the heuristic is a
#: stand-in, not a measurement, and downstream severity must not over-trust it.
CLASSICAL_CONFIDENCE_SCALE = 0.75
# Gate values for the classical fallback.
#
# These were chosen by sweeping the three thresholds over the bundled sample
# media (23 smoking exhaust ROIs, 14 clean ones) and taking the point that
# maximises recall while heavily penalising false alarms.  The best available
# operating point is **13% recall at a 7% false-alarm rate** -- the heuristic
# is a safety net, not a detector.  Hand-crafted colour/haze/texture cues
# genuinely cannot separate exhaust smoke from sunlit asphalt (both are
# achromatic, bright and smooth), which is the whole reason this project
# trains SmokeUNet.  The gates are therefore tuned to *under*-report: a missed
# plume in a degraded deployment is far less damaging than telling an operator
# a clean vehicle is polluting.
#: Absolute floor on the classical cue score before a pixel can be smoke.
CLASSICAL_MIN_SCORE = 0.40
#: A candidate covering more of the exhaust ROI than this is the road surface,
#: not a plume.
CLASSICAL_MAX_AREA_RATIO = 0.32
#: How much the candidate's mean cue score must exceed its surrounding ring's.
CLASSICAL_RING_MARGIN = 0.28

_CHECKPOINT_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


class SmokeSegmenter:
    """Produce a smoke probability map for an exhaust ROI.

    Attributes:
        mode: ``"unet"`` when the trained checkpoint is in use, ``"classical"``
            when the heuristic fallback is active.
    """

    def __init__(self, config: MLConfig | None = None) -> None:
        """Create a segmenter (the checkpoint is loaded lazily on first use)."""
        self.config = config or MLConfig()
        self.weights = str(self.config.segmenter_weights or DEFAULT_SEGMENTER_WEIGHTS)
        self.device = get_device(self.config.device)
        self.input_size = int(self.config.input_size or 256)
        self.mode: str = "classical"
        self._net: SmokeUNet | None = None
        self._loaded = False
        self._checkpoint_meta: dict[str, Any] = {}

    # -- lifecycle --------------------------------------------------------- #

    def resolved_mode(self) -> str:
        """Return :attr:`mode`, loading the checkpoint first if necessary.

        :attr:`mode` is only meaningful once the checkpoint has been resolved,
        and loading is lazy -- so a run that happened to detect no vehicles
        would otherwise report the initial ``"classical"`` value even with a
        perfectly good checkpoint on disk.
        """
        self._ensure_net()
        return self.mode

    @property
    def checkpoint_meta(self) -> dict[str, Any]:
        """Metadata recorded in the checkpoint (arch, metrics, trained_at)."""
        self._ensure_net()
        return dict(self._checkpoint_meta)

    def _ensure_net(self) -> SmokeUNet | None:
        if self._loaded:
            return self._net
        self._loaded = True

        key = f"{os.path.abspath(self.weights)}|{device_string(self.device)}"
        with _CACHE_LOCK:
            cached = _CHECKPOINT_CACHE.get(key)
            if cached is not None:
                self._net, self._checkpoint_meta = cached
                self.mode = "unet" if self._net is not None else "classical"
                return self._net

            net, meta = self._load_checkpoint()
            _CHECKPOINT_CACHE[key] = (net, meta)

        self._net, self._checkpoint_meta = net, meta
        self.mode = "unet" if net is not None else "classical"
        return self._net

    def _load_checkpoint(self) -> tuple[SmokeUNet | None, dict[str, Any]]:
        path = Path(self.weights)
        if not path.is_file():
            logger.warning(
                "Smoke segmenter checkpoint not found at %s -- falling back to the "
                "classical heuristic. Train one with "
                "`python -m mlcore.training.train_unet`.",
                path,
            )
            return None, {}

        # weights_only=True is mandatory here (security review finding ASG-03).
        #
        # A ``.pt`` file is a pickle. With weights_only=False, torch.load
        # executes whatever the pickle tells it to, so *reading* a checkpoint
        # is arbitrary code execution as the server user if that file is ever
        # attacker-controlled. The path is influenced by configuration
        # (ASG['SEGMENTER_WEIGHTS'], and for the detector a client-supplied
        # model name resolved inside ml_assets/), and the assets directory is
        # writable by anything that can write to the deployment — so this is
        # not a purely theoretical source.
        #
        # weights_only=True restricts the unpickler to tensors and plain
        # primitives, which is exactly what this checkpoint holds: a
        # state_dict plus str/int/float/dict/list metadata (arch, base,
        # input_size, params, metrics, epoch, trained_at, dataset_manifest,
        # history). Verified against the shipped ml_assets/smoke_unet.pt.
        try:
            blob = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not 500
            logger.warning(
                "Could not read segmenter checkpoint %s (%s); using classical fallback. "
                "If this is a legacy checkpoint containing pickled Python objects, "
                "re-export it with torch.save of a plain state_dict -- it will not be "
                "loaded unsafely.",
                path,
                exc,
            )
            return None, {}

        try:
            if isinstance(blob, dict) and "state_dict" in blob:
                state = blob["state_dict"]
                base = int(blob.get("base", 16))
                meta = {k: v for k, v in blob.items() if k != "state_dict"}
                if blob.get("input_size"):
                    self.input_size = int(blob["input_size"])
            else:
                state = blob
                base = 16
                meta = {}
            net = SmokeUNet(in_ch=3, base=base)
            net.load_state_dict(state)
            net.eval()
            net.to(self.device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Segmenter checkpoint %s is incompatible (%s); using classical fallback.", path, exc)
            return None, {}

        logger.info(
            "Loaded SmokeUNet checkpoint %s (base=%s, %s params) on %s.",
            path.name,
            base,
            f"{net.count_parameters():,}",
            device_string(self.device),
        )
        return net, meta

    def warmup(self) -> bool:
        """Force the checkpoint to load and run one dummy forward pass."""
        net = self._ensure_net()
        if net is None:
            return False
        try:
            dummy = torch.zeros(1, 3, self.input_size, self.input_size, device=self.device)
            with torch.inference_mode():
                net(dummy)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Segmenter warmup failed: %s", exc)
            return False

    # -- inference --------------------------------------------------------- #

    def segment(self, roi_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Segment smoke inside an exhaust ROI.

        Args:
            roi_bgr: ``(H, W, 3)`` BGR ``uint8`` crop.

        Returns:
            ``(mask, confidence)`` where ``mask`` is ``float32`` in ``[0, 1]``
            with the *same* ``(H, W)`` as *roi_bgr*, and ``confidence`` is the
            mean probability inside the thresholded region (``0.0`` when
            nothing was found).
        """
        if roi_bgr is None or roi_bgr.size == 0 or roi_bgr.ndim != 3:
            return np.zeros((1, 1), dtype=np.float32), 0.0

        net = self._ensure_net()
        if net is None:
            return self._segment_classical(roi_bgr)

        try:
            return self._segment_unet(net, roi_bgr)
        except Exception as exc:  # noqa: BLE001 - a bad ROI must not kill the run
            logger.warning("SmokeUNet inference failed (%s); falling back to classical for this ROI.", exc)
            return self._segment_classical(roi_bgr)

    def segment_many(self, rois: Sequence[np.ndarray]) -> list[tuple[np.ndarray, float]]:
        """Segment several ROIs from the same frame in one forward pass.

        Every ROI is resized to the network's square input anyway, so they can
        be stacked into a single batch.  On MPS (and CUDA) the dominant cost of
        a small network is the per-call host/device synchronisation, so folding
        a frame's vehicles into one call is roughly a 2-3x speed-up on
        multi-vehicle footage.

        Args:
            rois: BGR ``uint8`` crops.  Empty entries are tolerated.

        Returns:
            One ``(mask, confidence)`` per input, in the same order.
        """
        if not rois:
            return []

        net = self._ensure_net()
        valid = [i for i, roi in enumerate(rois) if roi is not None and roi.size and roi.ndim == 3]
        results: list[tuple[np.ndarray, float]] = [
            (np.zeros(roi.shape[:2] if roi is not None and roi.ndim >= 2 else (1, 1), dtype=np.float32), 0.0)
            for roi in rois
        ]
        if net is None:
            for i in valid:
                results[i] = self._segment_classical(rois[i])
            return results
        if not valid:
            return results

        try:
            batch = torch.stack([self._to_tensor(rois[i]) for i in valid]).to(self.device)
            with torch.inference_mode():
                probs = torch.sigmoid(net(batch))[:, 0].float().cpu().numpy()
        except Exception as exc:  # noqa: BLE001 - a bad batch must not kill the frame
            logger.warning("SmokeUNet batch inference failed (%s); falling back to classical.", exc)
            for i in valid:
                results[i] = self._segment_classical(rois[i])
            return results

        for prob, i in zip(probs, valid):
            results[i] = self._postprocess(prob, rois[i].shape[:2])
        return results

    def _to_tensor(self, roi_bgr: np.ndarray) -> torch.Tensor:
        """Resize, recolour and standardise one ROI into a CHW float tensor."""
        h, w = roi_bgr.shape[:2]
        size = self.input_size
        interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
        resized = cv2.resize(roi_bgr, (size, size), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))

    def _postprocess(self, prob: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, float]:
        """Resize a probability map back to the ROI and score its confidence."""
        h, w = int(shape[0]), int(shape[1])
        mask = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
        return mask, _mask_confidence(mask, float(self.config.mask_threshold))

    def _segment_unet(self, net: SmokeUNet, roi_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        tensor = self._to_tensor(roi_bgr)[None].to(self.device)
        with torch.inference_mode():
            prob = torch.sigmoid(net(tensor))[0, 0].float().cpu().numpy()
        return self._postprocess(prob, roi_bgr.shape[:2])

    # -- classical fallback ------------------------------------------------ #

    def _segment_classical(self, roi_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Heuristic smoke mask from colour, haze and texture cues.

        Three independent cues are combined **multiplicatively**, so a region
        only scores if it satisfies all of them:

        * **Achromaticity** -- exhaust smoke is grey/black/white, i.e. very low
          HSV saturation.  Painted bodywork, tail-lights and vegetation are not.
        * **Dark-channel haze prior** -- the classic single-image dehazing cue.
          A veil of smoke raises the per-pixel minimum across colour channels,
          because scattered light leaks into every channel.
        * **Texture loss** -- smoke *hides* the high-frequency structure behind
          it, so its local gradient energy falls below the region's own median.

        The third cue is what makes the heuristic usable at all.  Summing the
        cues instead of multiplying them let white lettering on a bus outscore
        the actual plume: paint is achromatic and bright, but it is *sharp*,
        and only a multiplicative combination lets the texture term veto it.

        Even so this path is weak by nature -- see the note on
        :data:`CLASSICAL_MIN_SCORE` for its measured recall.  Callers must
        check :attr:`SmokeSegmenter.mode` and present results accordingly.

        Args:
            roi_bgr: BGR ``uint8`` exhaust crop.

        Returns:
            ``(mask, confidence)`` as per :meth:`segment`.
        """
        h, w = roi_bgr.shape[:2]
        if h < 12 or w < 12:
            return np.zeros((h, w), dtype=np.float32), 0.0

        blurred = cv2.GaussianBlur(roi_bgr, (5, 5), 0)
        img = blurred.astype(np.float32)

        # 1) Achromaticity from HSV saturation.
        sat = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
        achromatic = np.clip(1.0 - sat / 70.0, 0.0, 1.0)

        # 2) Dark-channel prior: min over channels, min-filtered over a patch.
        patch = max(3, (min(h, w) // 14) | 1)
        dark_channel = cv2.erode(img.min(axis=2), np.ones((patch, patch), np.uint8))
        haze = np.clip((dark_channel - 45.0) / 120.0, 0.0, 1.0)

        # 3) Texture loss, measured against this region's own median energy.
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY).astype(np.float32)
        energy = cv2.blur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)), (13, 13))
        reference = float(np.median(energy)) + 1e-3
        texture_loss = np.clip(1.0 - energy / (2.0 * reference), 0.0, 1.0)

        # Weighted geometric mean.  Multiplying keeps the veto behaviour while
        # the root puts the result back on a [0, 1] scale, which matters
        # because the raw triple product lives around 0.02-0.15 and any
        # absolute threshold on it would be unreadable.
        weights = (0.8, 0.8, 1.0)
        score = np.power(
            np.power(achromatic, weights[0]) * np.power(haze, weights[1]) * np.power(texture_loss, weights[2]),
            1.0 / sum(weights),
        )
        score = cv2.GaussianBlur(score, (0, 0), sigmaX=max(1.5, min(h, w) / 90.0))

        # Relative gate: the plume must stand out against the region's own
        # background, so a uniformly hazy ROI is not declared to be all smoke.
        # The ceiling is the 99th percentile rather than the maximum -- a
        # single specular highlight was otherwise enough to rescale the real
        # plume down to nothing.
        baseline = float(np.percentile(score, 70))
        ceiling = float(np.percentile(score, 99))
        if ceiling < CLASSICAL_MIN_SCORE or ceiling - baseline < 0.05:
            return np.zeros((h, w), dtype=np.float32), 0.0

        mask = np.clip((score - baseline) / (ceiling - baseline), 0.0, 1.0)
        # ...and an absolute gate, so "least bad pixel in a boring ROI" loses.
        mask = mask * np.clip((score - CLASSICAL_MIN_SCORE) / 0.14, 0.0, 1.0)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = (mask > 0.40).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        binary = self._keep_localised_blob(binary, score, (h, w))
        if binary is None:
            return np.zeros((h, w), dtype=np.float32), 0.0

        mask = (mask * binary.astype(np.float32)).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0)

        confidence = _mask_confidence(mask, float(self.config.mask_threshold)) * CLASSICAL_CONFIDENCE_SCALE
        return np.clip(mask, 0.0, 1.0), round(confidence, 6)

    @staticmethod
    def _keep_localised_blob(
        binary: np.ndarray,
        score: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray | None:
        """Keep the candidate only if it behaves like a plume, not a surface.

        Colour/haze/texture cues alone cannot tell exhaust smoke from sunlit
        asphalt -- both are achromatic, bright (so the dark-channel prior
        fires) and smooth.  The distinguishing property is *locality*: a plume
        is a compact anomaly with ordinary surroundings, whereas a road
        surface fills the whole exhaust ROI and looks identical just outside
        the candidate region.

        Three tests are applied to the largest connected component:

        1. it must not cover more than :data:`CLASSICAL_MAX_AREA_RATIO` of the
           ROI (anything bigger is the background itself);
        2. it must hold the bulk of the thresholded pixels, rather than being
           one of many scattered specks;
        3. its mean cue score must beat the mean score of a ring around it by
           :data:`CLASSICAL_RING_MARGIN`.

        Args:
            binary: Thresholded candidate mask.
            score: The continuous cue score the mask came from.
            shape: ``(height, width)`` of the ROI.

        Returns:
            The cleaned single-component mask, or ``None`` to reject.
        """
        height, width = shape
        total = float(height * width)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num <= 1:
            return None

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(np.argmax(areas)) + 1
        area = float(areas.max())
        if area / total > CLASSICAL_MAX_AREA_RATIO or area < 24:
            return None
        if area / max(float(binary.sum()), 1.0) < 0.45:
            return None  # scattered speckle, not a plume

        blob = (labels == largest)
        radius = max(5, int(0.12 * min(height, width)) | 1)
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        ring = cv2.dilate(blob.astype(np.uint8), ring_kernel).astype(bool) & ~blob
        if ring.sum() < 24:
            return None
        if float(score[blob].mean()) - float(score[ring].mean()) < CLASSICAL_RING_MARGIN:
            return None
        return blob.astype(np.uint8)


def _mask_confidence(mask: np.ndarray, threshold: float) -> float:
    """Mean probability inside the thresholded region, ``0.0`` if it is empty."""
    selected = mask >= threshold
    if not bool(selected.any()):
        return 0.0
    return round(float(mask[selected].mean()), 6)


def clear_checkpoint_cache() -> None:
    """Drop cached checkpoints (used by tests and after retraining)."""
    with _CACHE_LOCK:
        _CHECKPOINT_CACHE.clear()
