# `mlcore` — the AutoSmokeGuard ML layer

Technical reference for the package that implements UC-05 (vehicle detection)
and UC-06 (smoke segmentation, intensity, severity). This document is for a
reader who wants to understand or retrain the models, not just call them; for
the HTTP-facing behaviour see `backend/README.md`.

`mlcore` has no Django import at module scope — it is a standalone Python
package that a plain script can `import mlcore` and run. The web layer calls
exactly one function, `mlcore.analyze_media`, from `analysis/worker.py`.

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | `MLConfig` dataclass (every tunable knob), asset paths (`ML_ASSETS_DIR`, weight file locations), torch device selection (`cuda` > `mps` > `cpu`, `ASG_FORCE_CPU=1` override) |
| `_compat.py` | Import-time guard that re-binds the real stdlib `sysconfig` module if a same-named package shadows it before `torch`/`torchvision`/`ultralytics` import (see "The `sysconfig` collision" below) |
| `preprocess.py` | Image/video loading, frame iteration with sampling and time caps, CLAHE + bilateral-denoise frame enhancement |
| `detector.py` | `VehicleDetector` — thin wrapper over an Ultralytics YOLO model restricted to COCO classes `{2: car, 3: motorcycle, 5: bus, 7: truck}`; computes the exhaust ROI under each detection |
| `model.py` | `SmokeUNet` — the segmentation network architecture |
| `segmenter.py` | `SmokeSegmenter` — loads `SmokeUNet` checkpoints, runs batched inference, and falls back to a classical CV heuristic when no checkpoint is present |
| `intensity.py` | Turns a probability mask into `area_ratio` / `mean_opacity` / `edge_density` / `darkness` / `compactness` / `largest_blob_ratio` features, combines them into a single `smoke_density` score, classifies severity, and aggregates per-region results into a media-level verdict |
| `annotate.py` | Draws vehicle boxes, smoke overlays and severity chips onto frames for the PDF/UI artifacts |
| `pipeline.py` | `analyze_media` — the single entry point; orchestrates detection → segmentation → density gating → artifact writing → progress reporting for one image or video |
| `selftest.py` | `python -m mlcore.selftest` — runs the real pipeline against `backend/sample_media/`, asserts the NFR budgets and the clean/smoking expectations per file |
| `training/synth_dataset.py` | Procedural dataset generator |
| `training/train_unet.py` | Training loop, loss, dataset class, augmentation |
| `training/evaluate.py` | Threshold sweep, `metrics.json` writer, per-group breakdown |

## The dataset: procedurally synthesised smoke on real street photography

There is no public, pixel-labelled dataset of vehicle exhaust smoke, and
hand-labelling one was out of scope. `mlcore/training/synth_dataset.py`
(generator version `1.4.0`) instead *synthesises* supervision:

- **Backgrounds are real photographs.** Crops are drawn from the COCO128
  image set (`ml_assets/datasets/coco128`). 52% of crops are the *exhaust
  ROI* under a COCO-labelled vehicle — the exact same lower-band,
  side-expanded, aspect-squashed geometry `VehicleDetector.exhaust_roi`
  computes at inference time — so there is no synthetic-to-real geometry gap.
  38% are arbitrary aspect-squashed crops of any photo; 10% are a fully
  procedural asphalt/sky gradient (used when COCO128 is unavailable).
- **Plumes are fractal.** A 5-octave fractal-Brownian-motion field is shaped
  by an anisotropic cone (`plume_shape`) that widens with distance from the
  tailpipe, then bent (`curl`), skewed (`shear`) and pushed through a
  turbulence displacement field (`turbulence_warp`) so no two plumes are the
  same triangle.
- **Opacity is solved, not guessed.** For each sample, `make_plume_alpha`
  solves the global opacity so the thresholded mask hits a sampled target
  coverage (4–26% of the tile for positives), which keeps label-area
  statistics controlled while appearance still varies across the whole
  opacity range.
- **Colour is contrast-checked against the local background.** Four emission
  palettes (`black`, `grey`, `white`, `blue_grey`) are sampled and, if none
  clears a minimum luminance separation of 58 grey levels against the
  background *under the plume*, the colour is pushed until it does — an
  invisible label is never written.
- **Hard negatives are 25% of the set.** Within that quarter: 32% road dust
  (a translucent brown cloud using the same plume-alpha machinery but wider
  and fainter), 26% under-vehicle shadow (a soft dark ellipse), 20%
  directional motion blur, 22% a plain, untouched background. 20% of
  *positive* samples additionally carry a distractor (dust or shadow) placed
  so it cannot overlap the labelled plume. Without these, the network learns
  "any soft grey blob is smoke."
- **Backgrounds are photo-disjoint between train and val.** A background
  photo used in `train` is never used in `val` (`VAL_BACKGROUND_FRACTION =
  0.20`, partitioned once with its own seeded RNG), so the validation score
  cannot be inflated by memorised backgrounds.
- **Photometric and sensor realism** are applied last: brightness/contrast/
  white-balance jitter, Gaussian sensor noise, optional defocus blur, and a
  55% chance of a JPEG re-encode round-trip at a random quality (58–94).

Regenerate it with:

```bash
python -m mlcore.training.synth_dataset --train 1600 --val 400 --seed 1337
```

The bundled checkpoint was trained on the default sizes: 1600 train / 400 val
(2000 total, 1516 positive / 484 negative per `metrics.json`). The dataset
lives at `ml_assets/datasets/smoke_synth/`, is **~263 MB**, is **gitignored**,
and is bit-for-bit reproducible from `--seed 1337` (every sample is drawn from
`np.random.default_rng([seed, split_id, i])`, so a partial regeneration is
also reproducible).

## `SmokeUNet`

A compact encoder-decoder U-Net, defined in `mlcore/model.py`:

- 4 encoder stages (`DoubleConv` = `conv3x3 → BN → ReLU` twice) at widths
  `base, 2·base, 4·base, 8·base` with 2×2 max-pool downsampling, a
  `16·base` bottleneck, and 4 decoder stages using bilinear upsample +
  skip-concatenation back down to `base`, then a 1×1 conv head producing a
  single logit channel.
- `base=16` → **1,964,097 parameters** (confirmed in `ml_assets/metrics.json`,
  field `params`).
- Input: `(N, 3, 256, 256)`, ImageNet-normalised RGB. Output: `(N, 1, 256,
  256)` raw logits — callers apply `sigmoid` themselves.
- The output head is initialised with a RetinaNet-style prior
  (`_init_head_prior`, default `positive_rate=0.09`) so the untrained network
  starts near the dataset's true positive-pixel rate instead of ~50%
  everywhere, saving several epochs that would otherwise be spent undoing a
  bad initialisation.

## Training recipe

`python -m mlcore.training.train_unet` (defaults shown):

| Setting | Value |
|---|---|
| Epochs | up to 60, early-stopped after 10 epochs without a new best val Dice |
| Batch size | 16 |
| Optimizer | AdamW, `lr=1e-3`, `weight_decay=1e-4` |
| LR schedule | 3-epoch linear warmup, then cosine annealing to 2% of peak |
| Loss | `BCEDiceLoss` — `0.5 · BCEWithLogitsLoss + 0.5 · (1 − batch Dice)` |
| Gradient clipping | max norm 5.0 |
| Device | auto (`cuda` > `mps` > `cpu`) |
| Seed | 1337 |

The checkpoint on disk (`ml_assets/smoke_unet.pt`) was trained at
`2026-09-22T00:25:39+0500` on `mps`. Its best epoch is recorded inside the
checkpoint payload (`epoch`) alongside the full per-epoch history and the
dataset manifest it was trained against.

### The four fixes needed to reach the 0.80 NFR

The NFR (SRS non-functional requirement) is ML accuracy ≥ 80%. Four specific
bugs/design choices stood between an early version of this pipeline and that
number; all four are documented in code comments at the point they were
fixed, and are summarised here because they are the most instructive part of
the ML work.

1. **Batch-aggregated Dice instead of per-image Dice.**
   `train_unet.py`'s `BCEDiceLoss` computes the Dice term over the *whole
   batch's* pooled true/false positives, not as a mean of per-image Dice
   scores. A quarter of the dataset is deliberately empty (hard negatives).
   Per-image Dice on an empty mask is `smooth / (Σprobs + smooth)`: even a
   near-perfect prediction of `sigmoid(logit) ≈ 0.0025` summed over 65,536
   pixels evaluates to a Dice of ~0.006 — the *worst* possible score for what
   is functionally a correct prediction. Averaging that over many empty
   images before optimising drags every logit toward saturation and wrecks
   recall on the positives. Batch aggregation makes an empty mask contribute
   only its false positives to the denominator, which is both correct and
   proportionate, and is exactly the micro-Dice reported as the accuracy
   metric (`ConfusionAccumulator.metrics`, `train_unet.py`).

2. **Raising the mask-alpha cutoff above the augmentation noise floor.**
   `synth_dataset.py`'s `MASK_ALPHA_CUTOFF = 0.35` (previously 0.15). A pixel
   blended at alpha `a` over a background differs from that background by
   roughly `a · |smoke_colour − background|` grey levels. With the enforced
   58-grey-level colour/background contrast floor, a cutoff of 0.35 puts the
   label boundary at ~20 grey levels of change — comfortably above sensor
   noise and JPEG artefacts. The earlier 0.15 cutoff produced boundaries
   around 7 grey levels, below the noise floor the augmentation itself adds,
   which capped achievable Dice no matter how long the model trained.

3. **The smoothstep alpha remap.** A raw fractal-Brownian-motion cone decays
   so gradually that the region where alpha crosses the label cutoff is tens
   of pixels wide — a 10% error in the network's estimate then moves the
   predicted outline a long way, and the model tends to find the plume but
   paint an under-confident halo the label excludes. `make_plume_alpha`
   remaps the density through a smoothstep (`base = base² · (3 − 2·base)`,
   after clipping between two sampled percentile bounds) so the plume gets a
   defined body with a soft but *bounded* fringe — closer to how dense
   exhaust actually looks near the pipe than an infinitely diffuse veil.

4. **The augmentation bug that reflected the image but zero-filled the
   mask.** `train_unet.py`'s `SmokeSegDataset._augment` applies a random
   rotation/scale via `cv2.warpAffine`. Image and mask must use the *same*
   border-fill mode for the newly exposed corners; an earlier version
   reflected the image (`BORDER_REFLECT_101`) but left the mask's border mode
   at the OpenCV default (zero-fill). Any reflected smoke pixel in a rotated
   corner was then silently labelled background — a training signal telling
   the network "this looks like smoke, but it isn't." Both `cv2.warpAffine`
   calls now use `borderMode=cv2.BORDER_REFLECT_101`.

### Measured metrics (`ml_assets/metrics.json`, quoted exactly)

```
model: SmokeUNet, params: 1,964,097
trained_at: 2026-09-22T00:25:39+0500
device: mps
best operating threshold: 0.60 (selected by best F1 over a sweep of 0.30–0.70 in steps of 0.05)

val @ threshold 0.60:
  iou        0.673865
  dice       0.805161
  pixel_acc  0.970401
  precision  0.846444
  recall     0.767717
  f1         0.805161

val @ threshold 0.50 (the segmenter's runtime default):
  iou        0.673265
  dice       0.804732
  pixel_acc  0.970014
  precision  0.836091
  recall     0.775640
  f1         0.804732

val_dice_per_image: 0.750576   (reported for transparency; NOT the accuracy metric — see fix #1 above)

nfr_target: 0.80
nfr_met:    true
evaluation_seconds: 3.89
```

Per-group breakdown (400 validation samples: 302 positive, 98 negative split
across four hard-negative kinds):

| Group | Samples | Pixel accuracy | False-positive rate | Dice | IoU |
|---|---|---|---|---|---|
| positive | 302 | 0.964933 | 0.011804 | 0.822062 | 0.697882 |
| negative:dust | 35 | 0.995152 | 0.004848 | — | — |
| negative:motion_blur | 21 | 0.963327 | 0.036673 | — | — |
| negative:plain | 20 | 0.995856 | 0.004144 | — | — |
| negative:shadow | 22 | 0.989700 | 0.010300 | — | — |

Motion-blur negatives have the highest false-positive rate of the four
(3.7%) — directional blur is the hard-negative type closest in appearance to
a wispy real plume — but it is still an order of magnitude below the
positive-class miss rate. The full 9-point threshold sweep (0.30–0.70) is
stored in `metrics.json`'s `threshold_sweep` array; Dice is essentially flat
across it (0.8021–0.8052), while precision rises and recall falls
monotonically as the threshold increases — the operating point is a
precision/recall trade-off, not a sharp peak.

To reproduce these numbers: `python -m mlcore.training.evaluate` (recomputes
every metric from the checkpoint rather than trusting the training loop's
cached numbers, and overwrites `ml_assets/metrics.json`).

## The `largest_blob_ratio` gate

At inference, `pipeline.py` only reports a smoke region when **all three** of
these floors clear, on the segmenter's raw probability mask for that
vehicle's exhaust ROI:

| Floor | Constant | Value | Purpose |
|---|---|---|---|
| Total thresholded area | `MIN_SMOKE_AREA_RATIO` | 0.02 | Filters stray pixels every model produces on a hard crop |
| Largest connected-component area | `MIN_SMOKE_BLOB_RATIO` | 0.018 | The discriminating gate — see below |
| Combined density score | `MIN_SMOKE_DENSITY` | 0.15 | Rejects a plume that is large but transparent/textureless/pale |

`largest_blob_ratio` (computed in `intensity.extract_features`, "share of the
ROI taken up by the single biggest 8-connected component") is the gate that
does the real work. Measured over the bundled sample media, false
activations on clean vehicles are scattered speckle along the bumper line —
individually tiny, but summing to more total area than a genuine plume
occupies in the very large ROI of a bus. Real smoke is one coherent mass;
segmentation speckle is many small ones, and total area alone cannot tell
them apart.

The measured split that set the 0.018 floor (`pipeline.py`, comment on
`MIN_SMOKE_BLOB_RATIO`):

- Clean vehicles: largest blob ≤ **0.0169** of the ROI (49 ROIs measured).
- Smoking vehicles: largest blob from **0.0207** (the bus sample) up to
  **0.153** (the truck sample).

The floor sits at 0.018, between the two. **The margin on the bus sample is
modest: 0.0207 clears the 0.018 floor by only ~15%, and the floor itself
clears the clean-vehicle maximum by only ~6.5%.** This is a calibration
point measured against the current checkpoint and the current sample media —
not a law of nature — and it **must be re-measured if the segmenter is
retrained**, since a retrained network's probability calibration on the bus
crop specifically is not guaranteed to reproduce 0.0207.

As of this writing, running `python -m mlcore.selftest` against the four
bundled sample files (on `mps`) passes all checks with this gate in place:
`sample_bus_smoking.jpg` reports 1 smoke region (moderate), `sample_car_clean.mp4`
and `sample_street_clean.jpg` report 0, and `sample_truck_smoking.mp4` reports
26 smoke judgements across its sampled frames — all within the NFR budgets
(see "Performance" below).

## The classical fallback

When `ml_assets/smoke_unet.pt` is missing, unreadable, or fails to load,
`SmokeSegmenter` falls back to a hand-written computer-vision heuristic
(`segmenter.py`, `_segment_classical`) so the product degrades rather than
dies. It combines three cues **multiplicatively** (so a candidate must
satisfy all three, not just one):

- **Achromaticity** — exhaust smoke is low-saturation grey/black/white;
  painted bodywork and vegetation are not.
- **Dark-channel haze prior** — the classic single-image dehazing cue; a veil
  of smoke raises the per-pixel channel minimum.
- **Texture loss** — smoke hides high-frequency structure behind it, so local
  gradient energy falls below the region's own median. This is the cue that
  makes the heuristic usable at all: without it, sharp white lettering on a
  bus (achromatic and bright) outscores the actual plume.

The candidate is then also required to be a compact, localised blob rather
than filling the whole ROI (`CLASSICAL_MAX_AREA_RATIO = 0.32`) and to stand
out from its own surrounding ring by a fixed margin (`CLASSICAL_RING_MARGIN
= 0.28`) — a road surface looks identical just outside the candidate region,
a real plume does not.

**Measured performance: 13% recall at a 7% false-alarm rate**, found by
sweeping the three gate thresholds over the bundled sample media (23 smoking
exhaust ROIs, 14 clean ones) and picking the point that maximises recall
while heavily penalising false alarms. This is documented in code as a
"safety net, not a detector": hand-crafted colour/haze/texture cues genuinely
cannot separate exhaust smoke from sunlit asphalt (both are achromatic,
bright and smooth), which is the whole reason this project trains SmokeUNet.
The gates are deliberately tuned to *under*-report — a missed plume in a
degraded deployment is considered far less damaging than falsely flagging a
clean vehicle. Every result carries `SmokeSegmenter.mode` (`"unet"` or
`"classical"`) so the API, the UI and the PDF report can say which one
produced it (`AnalysisDetail.segmenter_mode` in the API contract); `mlcore.selftest`
also treats the smoke expectations as warnings rather than hard failures when
the classical fallback is in use.

## Performance (NFRs)

`mlcore/selftest.py` gates two NFRs directly against the real pipeline run
over `backend/sample_media/` (`MAX_IMAGE_SECONDS = 10.0`, `MIN_VIDEO_FPS =
5.0`). A run against the current checkpoint on this machine (`mps`) measured:

| Sample | Kind | Frames | Seconds | FPS | Verdict |
|---|---|---|---|---|---|
| `sample_bus_smoking.jpg` | image | 1 | 0.37 s | — | PASS (< 10 s) |
| `sample_street_clean.jpg` | image | 1 | 0.42 s | — | PASS (< 10 s) |
| `sample_car_clean.mp4` | video | 24 | 1.46 s | 16.5 | PASS (≥ 5 fps) |
| `sample_truck_smoking.mp4` | video | 24 | 1.21 s | 19.9 | PASS (≥ 5 fps) |

These numbers are for one machine and one run — reproduce with `python -m
mlcore.selftest`, or `python -m mlcore.selftest --json` for the full raw
result dicts.

## The `sysconfig` collision

`mlcore` is imported from inside a Django project whose app package is named
`system_config` specifically *because* a package literally named `sysconfig`
at the project root shadows the standard-library `sysconfig` module (see
`backend/README.md` for the full story). `torch._dynamo.config` calls
`sysconfig.get_config_var()` during import and raises `AttributeError` if
`sysconfig` resolves to anything but the real module — a failure that
cascades through `torchvision` and `ultralytics`. `mlcore/_compat.py`'s
`ensure_stdlib()` is a belt-and-braces runtime guard against the same class
of problem: if some other shadowing package ever reappears on `sys.path`
before `mlcore` is imported, it re-binds the real stdlib module directly from
the interpreter's stdlib directory rather than failing. It is a stopgap, not
a substitute for not naming a package `sysconfig`.

## Regenerating the dataset and retraining

```bash
# 1. Regenerate the synthetic dataset (deterministic from --seed; ~263 MB, gitignored)
python -m mlcore.training.synth_dataset --train 1600 --val 400 --seed 1337

# 2. Train (writes ml_assets/smoke_unet.pt and ml_assets/training_curve.png)
python -m mlcore.training.train_unet

# 3. Evaluate on the val split and refresh ml_assets/metrics.json
python -m mlcore.training.evaluate

# 4. Sanity-check the real pipeline end to end, including the two NFRs
python -m mlcore.selftest
```

A fast smoke run for iterating on the generator or the training loop itself:

```bash
python -m mlcore.training.train_unet --limit 200 --epochs 2
```

After retraining, re-run `mlcore.selftest` and re-check the
`largest_blob_ratio` floor in `pipeline.py` against the new checkpoint's
behaviour on `sample_bus_smoking.jpg` specifically — see "The
`largest_blob_ratio` gate" above.

## Model rollback

The current checkpoint on disk (`ml_assets/smoke_unet.pt`) has SHA256:

```
1d2b1de19102d8d66584bf7440839cfb04d7a8ffb449d5f1fdfce9c0692a354f
```

It can be replaced with the previous checkpoint (SHA256 `3cda307e550d3b9ea591f75194f31ca5eb7401d11b1df28540dcc9484266194a`, saved at git commit f7ed668) without any code changes, using:

```bash
git show f7ed668:ml_assets/smoke_unet.pt > ml_assets/smoke_unet.prev.pt
```

**Alternative: evaluate with a custom checkpoint without moving files.** `mlcore.config.MLConfig.segmenter_weights` accepts an absolute path override; pass it when instantiating the config to test a checkpoint from anywhere on disk, then save it to `ml_assets/smoke_unet.pt` only when validated.
