"""Evaluate the trained :class:`~mlcore.model.SmokeUNet` and write metrics.

Recomputes every validation metric from the checkpoint (rather than trusting
the numbers the training loop cached), sweeps the decision threshold to find
the best-F1 operating point, and writes ``ml_assets/metrics.json``.

Run with::

    python -m mlcore.training.evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from ..config import (
    DEFAULT_SEGMENTER_WEIGHTS,
    METRICS_PATH,
    SYNTH_DATASET_DIR,
    device_string,
    get_device,
)
from ..model import SmokeUNet
from .train_unet import ConfusionAccumulator, SmokeSegDataset

logger = logging.getLogger("asg.ml")

#: Thresholds swept when looking for the best operating point.
THRESHOLD_SWEEP = tuple(round(0.30 + 0.05 * i, 2) for i in range(9))  # 0.30 .. 0.70


class EvaluationError(RuntimeError):
    """Raised when the checkpoint or the dataset needed for evaluation is absent."""


def load_checkpoint(path: Path, device: torch.device) -> tuple[SmokeUNet, dict[str, Any]]:
    """Load a ``SmokeUNet`` checkpoint into eval mode on *device*."""
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(
            f"Checkpoint not found: {path}. Train one with "
            "`python -m mlcore.training.train_unet`."
        )
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    base = int(blob.get("base", 16)) if isinstance(blob, dict) else 16
    meta = {k: v for k, v in blob.items() if k not in {"state_dict", "history"}} if isinstance(blob, dict) else {}

    model = SmokeUNet(in_ch=3, base=base)
    model.load_state_dict(state)
    model.eval().to(device)
    return model, meta


@torch.inference_mode()
def collect_probabilities(
    model: SmokeUNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run the model once and keep the probability maps for the whole split.

    Keeping probabilities lets the threshold sweep reuse a single forward pass
    instead of re-running the network nine times.
    """
    probs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for images, batch_targets in loader:
        logits = model(images.to(device))
        probs.append(torch.sigmoid(logits).float().cpu())
        targets.append(batch_targets.float().cpu())
    return probs, targets


def metrics_at(probs: Sequence[torch.Tensor], targets: Sequence[torch.Tensor], threshold: float) -> dict[str, float]:
    """Micro-averaged metrics for a given decision threshold."""
    accumulator = ConfusionAccumulator()
    for prob, target in zip(probs, targets):
        accumulator.update(prob, target, threshold)
    return accumulator.metrics()


def per_group_report(
    dataset: SmokeSegDataset,
    probs: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    threshold: float,
) -> dict[str, Any]:
    """Break the result down by sample kind (positive / each negative type).

    This is where the hard negatives earn their keep: it shows, explicitly, how
    often the model paints smoke onto dust, shadow or motion blur.
    """
    index_path = dataset.root / dataset.split / "index.json"
    if not index_path.is_file():
        return {}
    try:
        index = {entry["file"]: entry for entry in json.loads(index_path.read_text())}
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return {}

    flat_probs = torch.cat([p for p in probs], dim=0) if probs else torch.empty(0)
    flat_targets = torch.cat([t for t in targets], dim=0) if targets else torch.empty(0)
    if flat_probs.numel() == 0 or flat_probs.shape[0] != len(dataset.names):
        return {}

    groups: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0})
    for i, name in enumerate(dataset.names):
        entry = index.get(name, {})
        label = str(entry.get("label", "unknown"))
        group = label if label != "negative" else f"negative:{entry.get('negative_kind', '?')}"

        pred = (flat_probs[i] >= threshold).float()
        tgt = (flat_targets[i] >= 0.5).float()
        bucket = groups[group]
        bucket["n"] += 1.0
        bucket["tp"] += float((pred * tgt).sum())
        bucket["fp"] += float((pred * (1.0 - tgt)).sum())
        bucket["fn"] += float(((1.0 - pred) * tgt).sum())
        bucket["tn"] += float(((1.0 - pred) * (1.0 - tgt)).sum())

    report: dict[str, Any] = {}
    for group, bucket in sorted(groups.items()):
        total = max(bucket["tp"] + bucket["fp"] + bucket["fn"] + bucket["tn"], 1.0)
        entry: dict[str, Any] = {
            "samples": int(bucket["n"]),
            "pixel_accuracy": round((bucket["tp"] + bucket["tn"]) / total, 6),
            "false_positive_rate": round(bucket["fp"] / max(bucket["fp"] + bucket["tn"], 1.0), 6),
        }
        if bucket["tp"] + bucket["fn"] > 0:
            denominator = 2 * bucket["tp"] + bucket["fp"] + bucket["fn"]
            entry["dice"] = round(2 * bucket["tp"] / denominator, 6) if denominator else 0.0
            entry["iou"] = round(bucket["tp"] / max(bucket["tp"] + bucket["fp"] + bucket["fn"], 1.0), 6)
        report[group] = entry
    return report


def evaluate(
    checkpoint: Path = DEFAULT_SEGMENTER_WEIGHTS,
    dataset_root: Path = SYNTH_DATASET_DIR,
    out_path: Path = METRICS_PATH,
    batch_size: int = 16,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate the checkpoint on the validation split and write metrics.json.

    Args:
        checkpoint: Path to ``smoke_unet.pt``.
        dataset_root: Dataset root containing the ``val`` split.
        out_path: Where to write the metrics JSON.
        batch_size: Inference batch size.
        limit: Cap validation samples (debugging only).

    Returns:
        The metrics dict that was written.

    Raises:
        EvaluationError: Checkpoint or dataset split is missing.
    """
    started = time.time()
    device = get_device()
    model, meta = load_checkpoint(Path(checkpoint), device)

    try:
        dataset = SmokeSegDataset(Path(dataset_root), "val", augment=False, limit=limit, seed=0)
    except FileNotFoundError as exc:
        raise EvaluationError(str(exc)) from exc

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logger.info("Evaluating %s on %d validation samples (%s).", Path(checkpoint).name, len(dataset), device_string(device))

    probs, targets = collect_probabilities(model, loader, device)

    sweep: list[dict[str, float]] = []
    for threshold in THRESHOLD_SWEEP:
        scores = metrics_at(probs, targets, threshold)
        sweep.append({"threshold": threshold, **{k: scores[k] for k in ("iou", "dice", "pixel_accuracy", "precision", "recall", "f1")}})

    best = max(sweep, key=lambda row: (row["f1"], row["iou"]))
    best_threshold = float(best["threshold"])
    default_metrics = metrics_at(probs, targets, 0.5)
    best_metrics = metrics_at(probs, targets, best_threshold)

    manifest = meta.get("dataset_manifest") or {}
    dataset_info = {
        "name": manifest.get("name", "smoke_synth"),
        "root": str(Path(dataset_root)),
        "generator_version": manifest.get("generator_version"),
        "seed": manifest.get("seed"),
        "val_samples": len(dataset),
        "totals": manifest.get("totals"),
        "background_split": manifest.get("background_split"),
    }

    payload: dict[str, Any] = {
        "model": str(meta.get("arch", "SmokeUNet")),
        "params": int(meta.get("params", model.count_parameters())),
        "checkpoint": str(Path(checkpoint)),
        "trained_at": meta.get("trained_at"),
        "dataset": dataset_info,
        "device": device_string(device),
        "threshold": round(best_threshold, 2),
        "val": {k: best_metrics[k] for k in ("iou", "dice", "pixel_accuracy", "precision", "recall", "f1")},
        "val_at_0.5": {k: default_metrics[k] for k in ("iou", "dice", "pixel_accuracy", "precision", "recall", "f1")},
        "val_dice_per_image": best_metrics["dice_per_image"],
        "threshold_sweep": sweep,
        "per_group": per_group_report(dataset, probs, targets, best_threshold),
        "nfr_target": 0.80,
        "nfr_met": bool(best_metrics["dice"] >= 0.80 and best_metrics["pixel_accuracy"] >= 0.80),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "evaluation_seconds": round(time.time() - started, 2),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Wrote %s -- best threshold %.2f: dice %.4f, IoU %.4f, pixel acc %.4f.",
        out_path, best_threshold, best_metrics["dice"], best_metrics["iou"], best_metrics["pixel_accuracy"],
    )
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mlcore.training.evaluate",
        description="Evaluate the SmokeUNet checkpoint and write ml_assets/metrics.json.",
    )
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_SEGMENTER_WEIGHTS))
    parser.add_argument("--data", type=str, default=str(SYNTH_DATASET_DIR))
    parser.add_argument("--out", type=str, default=str(METRICS_PATH))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    try:
        payload = evaluate(
            checkpoint=Path(args.checkpoint),
            dataset_root=Path(args.data),
            out_path=Path(args.out),
            batch_size=args.batch,
            limit=args.limit,
        )
    except EvaluationError as exc:
        logger.error("%s", exc)
        return 2

    print(json.dumps(payload, indent=2))
    return 0 if payload["nfr_met"] else 3


if __name__ == "__main__":
    sys.exit(main())
