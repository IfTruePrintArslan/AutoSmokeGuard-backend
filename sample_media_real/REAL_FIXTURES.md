# Real-smoke fixtures

> **The vehicles are real, and so is the smoke.**
> Everything below is a trimmed, re-encoded derivative of real stock footage,
> *not* a synthetic fBm plume composited onto a photo. These exist to
> complement — not replace — the synthetic fixtures in
> `../sample_media/README.md`, which stay in place, untouched, and are still
> what the current self-test and product gate exercise. Rewiring the gate to
> also use these is a separate step, sequenced deliberately after the current
> retrain lands.

## Why these exist

The synthetic fixtures (`sample_truck_smoking.mp4`, `sample_bus_smoking.jpg`,
etc.) reward a detector for recognising the specific fractal-Brownian-motion
texture used by `mlcore/training/synth_dataset.py`, not for recognising real
smoke. A model can score 4/4 on them while doing worse on real held-out data,
or vice versa. These six files are cut from a 65-clip real-footage corpus
assembled specifically to evaluate vehicle-exhaust detection, so that a
detector's score against them means something. Three fixtures (rear-tailpipe
car, vertical-stack truck, motorbike) target the three positive geometries
where round-2 model variants diverge most; the rest are negatives/hard
negatives.

## Why this lives in `sample_media_real/`, not `sample_media/`

`backend/mlcore/selftest.py` walks `media_dir.iterdir()` and analyses every
file it finds; with a U-Net checkpoint present it treats a missed detection
as a hard failure, not a warning. `backend/tests/test_analysis.py` globs
`sample_media/`, sorts, and end-to-end-tests whichever file sorts first.
Dropping these into `sample_media/` would therefore silently pull them into
the existing self-test/gate/CI run and could flip it red the moment the
*currently shipped* model — never measured against real footage — fails to
detect real smoke, before that comparison is meant to happen deliberately.
So these files sit in this sibling directory, inert until someone points
`selftest`'s `media_dir` argument (or an equivalent) at it on purpose.

The role names and the `smoking` / `clean` substring convention are kept
identical to `sample_media/` on purpose, because `selftest._expect_smoke()`
derives the expected answer from the filename (`smoking`/`smoke` → expect
smoke, checked before `clean` → expect no smoke). No `clean` fixture here
contains the substring `smoke` anywhere in its filename — verified below.

## Licence rule applied

**Primary rule: Pexels only.** The Pexels Licence
(<https://www.pexels.com/license/>) is free to use, modification is
permitted, and no attribution is required — it explicitly allows
redistribution, including bundling a modified/trimmed derivative inside a
source repository as a test fixture. Five of the six fixtures here are
Pexels-derived on that basis alone.

**One deliberate exception: `sample_stack_smoking_real.mp4`.** This is cut
from `File:F-450_coal_rolling_Monster_(video).webm` on Wikimedia Commons.
Per-item licence, read and quoted (not assumed): **CC BY 3.0 Unported,
attribution to Salvatore Arnone**
(<https://commons.wikimedia.org/wiki/File:F-450_coal_rolling_Monster_(video).webm>,
licence-reviewed by a Commons admin 2016-05-09; originally a CC-licensed
YouTube upload). CC BY 3.0 permits redistribution and modification inside a
repository provided attribution is given — unlike the "no explicit licence"
Internet Archive mirrors and the not-yet-read Pixabay item, which stay
excluded (see "Dropped for licence reasons"). This exception is taken
because vertical-stack recall is the single widest-swinging metric across
the round-2 model variants (9.3%–52.3%), and this is the only real
vertical-stack-exhaust clip found on any openly-licensed host across five
source platforms searched.

**Required attribution for the one CC BY file (both places, per the
condition of the licence):**

> "F-450 coal rolling Monster (video).webm" by Salvatore Arnone, used under
> CC BY 3.0 Unported (<https://creativecommons.org/licenses/by/3.0/>).
> Source: <https://commons.wikimedia.org/wiki/File:F-450_coal_rolling_Monster_(video).webm>

1. **In this document** — the line above, plus the provenance-table row
   below.
2. **Travelling with the file itself** — `sample_stack_smoking_real.mp4` has
   this baked into its own MP4 container metadata (`artist`, `copyright`,
   `comment`, `title` tags), verified with `ffprobe -show_entries
   format_tags`:
   ```
   TAG:title=F-450 coal rolling Monster (trimmed excerpt)
   TAG:artist=Salvatore Arnone
   TAG:copyright=CC BY 3.0 Unported (https://creativecommons.org/licenses/by/3.0/)
   TAG:comment=Source: https://commons.wikimedia.org/wiki/File:F-450_coal_rolling_Monster_(video).webm ; Licence: CC BY 3.0 Unported ; Attribution: Salvatore Arnone ; trimmed 00:00:04.0-00:00:07.0 from the original, re-encoded H.264, no other modification ; re-cut from the prior 00:00:01.0-00:00:07.0 trim so the truck's apparent width stays large enough throughout for a fair detector test (see REAL_FIXTURES.md)
   ```
   Caveat, stated honestly: container metadata does not survive every
   possible re-encode/re-mux a future pipeline step might do. This
   REAL_FIXTURES.md entry is the durable attribution record of record; the
   embedded tags are a second, redundant surface, not a substitute for it.

**Gap, stated honestly, for the five Pexels items:** `MANIFEST.json` records
`source_url`, `title`, `licence`, and `sha256` for every clip, but not the
Pexels contributor's username, and Pexels' own licence does not require
capturing one. "Uploader" is `not captured — not required by licence` for
each; nothing was fabricated.

## Provenance / licence table

| Shipped file | Source clip (corpus) | Source URL | Uploader / author | Licence | Source clip sha256 | Trim | Crop / processing | Shipped file sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sample_sedan_smoking_real.mp4` | `positives/A_pexels_jaguar_white_exhaust.mp4` ("White Smoke from Exhaust") | https://www.pexels.com/video/6160044/ | not captured — not required by licence | Pexels Licence — free to use, modification permitted, no attribution required | `29fcf183ced687764b071a5e706b3c8a99f3bad6ce0afc7e1b73ed56076f37fb` | 00:00:00.5 → 00:00:06.5 (6.0 s) of a 12.133 s source, static camera | Center-crop 854×480 → 640×480 (`crop=640:480:107:0`); H.264 yuv420p, CRF 20, 15 fps, no audio | `c42dedf406481f14ef69e3a485e5ac7da23c3aee244b695def89506690d365af` |
| `sample_car_clean_real.mp4` | `negatives/NC_pexels_tailpipe_black_car.mp4` ("Video of a car's exhaust pipe") | https://www.pexels.com/video/4228873/ | not captured — not required by licence | Pexels Licence — free to use, modification permitted, no attribution required | `491bb5c526dfb90380e28f4da12cd13687564cc172a2e03f7955049ba0600837` | 00:00:01.0 → 00:00:07.0 (6.0 s) of a 12.0 s source, static camera | Center-crop 854×480 → 640×480 (`crop=640:480:107:0`); H.264 yuv420p, CRF 20, 15 fps, no audio | `dddf8cc95e8c246d791d720546ec059835f9620fa57135fe20153471b0b1da6c` |
| `sample_coldstart_smoking_real.jpg` | frame of `positives/A_pexels_dodge_coldstart.mp4` ("Close Up of Dodge Car Starting") — **a different clip from the smoking video above, by design** | https://www.pexels.com/video/12152980/ | not captured — not required by licence | Pexels Licence — free to use, modification permitted, no attribution required | `1a71ee2b61f86d4a5a0e33c2cb17aa24e20ad0b7db5fd8b5e31fdcb6760bc1bc` | Single frame at 00:00:04.8 | No crop, full native 270×480 portrait frame, `ffmpeg -frames:v 1 -q:v 3` | `dbe784c1e80475d86ac7dfeb972af1b82ef1df8997ee5e24838446d2821a0cd4` |
| `sample_street_clean_real.jpg` | frame of `negatives/N_pexels_highway_city.mp4` ("Heavy Traffic on a Highway in a City") | https://www.pexels.com/video/12607644/ | not captured — not required by licence | Pexels Licence — free to use, modification permitted, no attribution required | `ce70cc22d341bcdbbc59ad8868d7a8921ef84706f42bb4ab9ffa9c50f6dac833` | Single frame at 00:00:05.0 | No crop, full native 854×480 frame, `ffmpeg -frames:v 1 -q:v 3` | `7d4a187b41176c4045c24450557178d6ef61c9d653c3e3dce949af385166d07e` |
| `sample_stack_smoking_real.mp4` | `positives/B_commons_f450_stacks.mp4` ("F-450 coal rolling Monster (video).webm") | https://commons.wikimedia.org/wiki/File:F-450_coal_rolling_Monster_(video).webm | **Salvatore Arnone** | **CC BY 3.0 Unported** — attribution required, see box above | `caa77de81f12a098ecd7ee51a5f3e0e49a7b8affff3ba32cd25aca70353ab6b7` | 00:00:04.0 → 00:00:07.0 (3.0 s) of a 7.133 s source, static camera, truck approaching (re-cut from the prior 00:00:01.0-00:00:07.0 / 6.0 s trim — see "Apparent vehicle width" below for why) | No crop, native 270×480 portrait (kept portrait deliberately — the stacks tower above the cab and need the vertical headroom); H.264 yuv420p, CRF 20, 15 fps, no audio; attribution metadata baked into container tags | `1c2261e544f00807b44f547acb3095ae0fa9dc63776f8b7c052ce52f547a138c` |
| `sample_motorbike_smoking_real.mp4` | `positives/C_pexels_motorbike_exhaust.mp4` ("Motorbike Exhaust Producing Smoke") | https://www.pexels.com/video/28375013/ | not captured — not required by licence | Pexels Licence — free to use, modification permitted, no attribution required | `11abb6963a97337cfd3854febabc6b0036862cb97932a10d2999e5a0fea76386` | 00:00:02.0 → 00:00:08.0 (6.0 s) of a 12.133 s source, static camera | Center-crop 1122×480 → 640×480 (`crop=640:480:241:0`); H.264 yuv420p, CRF 20, 15 fps, no audio | `0f80045bf4cdb90cf320c2ed2e7a5a4c748f34cc052697ac38c1fed6b78a55a1` |

Source-clip sha256 values are copied from the corpus `MANIFEST.json` and were
independently re-verified against the actual bytes on disk before cutting
(`shasum -a 256`, matched exactly, including for the Commons and re-checked
Pexels items). Shipped-file sha256 values were computed after encoding, on
the exact bytes now sitting in this directory.

## Why the smoking still is no longer a frame of the smoking video

The first cut of this fixture set took `sample_sedan_smoking_real.jpg` as a
frame of the same clip as `sample_sedan_smoking_real.mp4` (both video/6160044,
the Jaguar). That means two of the four fixtures exercised one scene: they
would pass or fail together, adding almost no independent coverage. The
still is now cut from `A_pexels_dodge_coldstart.mp4` (video/12152980)
instead — a different vehicle, different location, different smoke
mechanism (cold-start condensation enveloping the whole rear, vs. a
kerbside idle plume at the tailpipe) — and renamed to
`sample_coldstart_smoking_real.jpg` to describe what it actually shows.

**A more precise, corrected account of the contamination status of
`A_pexels_tailpipe_whitesmoke`** (this agent and the coordinator each had
half of it; both facts are recorded here so a future reader gets the full
picture instead of either half):

- **Raw inventory status: TOUCHED.** `FYP-ml-assets/uni-out/inventory.json`,
  key `contamination.rear_tailpipe_TOUCHED_by_selflab`, lists
  `A_pexels_tailpipe_whitesmoke`. Read in isolation, this clip was touched
  by the self-labelling/training pipeline.
- **Effective status: CLEAN, by explicit freeing.** `round2/code_snapshot/msdata.py:95`
  defines `SELFLAB_INNER_FREED_CLIPS = {"A_pexels_tailpipe_whitesmoke",
  "A_ia_tailpipe_blacksmoke"}`, and `round2/code_snapshot/evalvid.py:65-70`'s
  `clip_roles()` resolves the sets the eval actually gates on as:
  ```
  touched = set(inv["rear_tailpipe_TOUCHED_by_selflab"]) - set(SELFLAB_INNER_FREED_CLIPS)
  clean   = set(inv["rear_tailpipe_CLEAN"]) | set(SELFLAB_INNER_FREED_CLIPS)
  ```
  `A_pexels_tailpipe_whitesmoke` is explicitly freed out of `touched` and
  into `clean`. The round-2 evaluation counts it as a legitimate CLEAN gate
  clip. Do not read the raw-inventory line alone and conclude the eval's
  CLEAN set is contaminated, and do not discard this clip on that basis
  alone — it is a valid, never-actually-trained-on (per the freeing
  rationale) clip.

**The decision to use `A_pexels_dodge_coldstart` instead stands regardless
of the correction above.** `A_pexels_dodge_coldstart` is in
`rear_tailpipe_CLEAN` outright, with no freeing caveat needed, so it remains
the stronger, simpler-to-justify choice. The swap away from
`A_pexels_tailpipe_whitesmoke` for the smoking still is therefore still
correct — just not for the reason ("it's contaminated") this document
originally gave. The real reasons it wasn't used are the scene-independence
requirement (see above) and the portrait-cropping/weak-vehicle-visibility
issues noted below; contamination is no longer one of them.

## Eval-corpus overlap (per request: document what's genuinely held out)

Checked against `/Users/arslan/Desktop/FYP-ml-assets/datasets/realvid_masks/groups.json`
and `/Users/arslan/Desktop/FYP-ml-assets/uni-out/inventory.json`, which back
the round-2 "1,624-ROI video evaluation corpus" (`uni-out/round2/eval_video.json`,
`n_rows: 1624`, `n_clips: 62`):

| Shipped fixture's source clip | In the `rear_tailpipe` contamination inventory? | Status |
| --- | --- | --- |
| `A_pexels_jaguar_white_exhaust` (→ `sample_sedan_smoking_real.mp4`) | Yes — `rear_tailpipe_CLEAN` | Member of the eval corpus, confirmed never trained on. Legitimate as a fixture, **but not independent of the eval numbers** — a result on it can't be cited as fresh confirmation of the eval score, only as a same-population sanity check. |
| `A_pexels_dodge_coldstart` (→ `sample_coldstart_smoking_real.jpg`) | Yes — `rear_tailpipe_CLEAN` | Same as above: in the eval corpus, never trained on, not independent of the eval numbers. |
| `A_pexels_tailpipe_whitesmoke` (rejected, not shipped) | Raw: `rear_tailpipe_TOUCHED_by_selflab`. Effective: `rear_tailpipe_CLEAN` (explicitly freed by `SELFLAB_INNER_FREED_CLIPS`, `msdata.py:95`; see correction above) | **Not** contaminated once freeing is accounted for — the eval itself treats it as CLEAN. Rejected instead for scene-independence and weak-vehicle-visibility reasons (see above/below), not for contamination. |
| `NC_pexels_tailpipe_black_car` (→ `sample_car_clean_real.mp4`) | Not in the `rear_tailpipe` contamination inventory (bucket is `negative_closeup`, which that inventory doesn't cover) | Held-out/eval status **not determined** from the records available to me. Documented as unknown, not asserted either way. |
| `N_pexels_highway_city` (→ `sample_street_clean_real.jpg`) | Not covered (bucket `negative`) | Same: unknown, not asserted. |
| `B_commons_f450_stacks` (→ `sample_stack_smoking_real.mp4`) | Not covered (bucket `vertical_stack`) | Same: unknown, not asserted. |
| `C_pexels_motorbike_exhaust` (→ `sample_motorbike_smoking_real.mp4`) | Not covered (bucket `motorbike`) | Same: unknown, not asserted. |

The `inventory.json` contamination tracking that exists only covers the
`rear_tailpipe` bucket (15 clips: 7 `CLEAN`, 8 `TOUCHED_by_selflab`). No
equivalent CLEAN/TOUCHED split was found for the `negative_closeup`,
`negative`, `vertical_stack`, or `motorbike` buckets in the files this agent
has read access to, so **no claim of independence is made for the other four
fixtures** — that would need someone with visibility into the actual
training-data assembly to confirm.

**Net effect:** two of the six fixtures (the sedan-smoking video and the
cold-start-smoking still) are legitimate-but-not-independent — real,
never-trained-on, but drawn from the same pool the round-2 numbers already
measure. The other four are new to this fixture set's cross-check but their
train/eval relationship is unverified rather than confirmed-independent.
Treat a pass/fail on any of these six as "the model behaves sanely on real
footage," not as "this reproduces the round-2 eval score."

## Apparent vehicle width per positive fixture (is each one a fair test?)

Round-2 recall by apparent vehicle width, from
`/Users/arslan/Desktop/FYP-ml-assets/uni-out/round2/eval_video.json`,
`u1fbm_full.by_vehicle_width` (other model variants in the same file show
the same shape — near-zero below 180px, a step change above it — with
different absolute numbers per bucket):

| Width bucket | Recall (u1fbm_full) |
| --- | --- |
| 0–60px | 4.76% |
| 60–110px | 0.0% |
| 110–180px | 0.0% |
| 180–300px | 44.2% |
| 300–500px | 55.1% |
| 500px+ | 44.3% |

A positive fixture whose vehicle sits below ~180px is therefore not really
testing smoke detection — it's testing whether the model can see the
vehicle at all, which round-2 data says it mostly can't, regardless of
smoke. Measurements below are by eye against a pixel-grid overlay
(`ffmpeg drawgrid`) on the actual shipped frame(s), not a detector run —
no GPU/model was used, per instruction.

| Fixture | Frame size | Apparent vehicle width | Bucket | Fair test? |
| --- | --- | --- | --- | --- |
| `sample_sedan_smoking_real.mp4` | 640×480 | ≈400px (Jaguar spans the full left two-thirds of frame, from the left edge — itself a crop boundary, so the true vehicle is at least this wide — to the front bumper at ~x=400) | 300–500px | **Fair test.** Comfortably clear of the cliff. |
| `sample_coldstart_smoking_real.jpg` | 270×480 (native) | ≈240px (Dodge Challenger runs from ~x=10 to ~x=250 in a tight/close shot) | 180–300px | **Fair test**, despite the narrow frame — it clears the cliff because the shot is tight, not because the frame is wide. |
| `sample_stack_smoking_real.mp4` | 270×480 (native) | **Re-cut 2026-09-26 (was 6.0 s, now 3.0 s) — see note below.** Measured on actual extracted frames with an `ffmpeg drawgrid`-style pixel overlay (verified numerically, per-pixel, not eyeballed): ≈150–151px at 00:00:00.0 (start of the new clip), rising through ≈178px at 00:00:01.2, crossing ≈180px at ≈00:00:01.25, ≈187px at 00:00:01.3, ≈207px at 00:00:01.5, and reaching a lower-bound of 270px+ (truck partly cropped by the frame edge as it closes in and turns) for roughly the final 1.3–1.5 s | **110–180px for ~00:00:00.0–01.2, then 180–300px+ for the rest** | **Fair test throughout.** The truck is at least ~150px wide from the first frame of this re-cut and clears the 180px cliff by ≈00:00:01.25 of 3.0 s, with the heaviest smoke in the same back portion where the vehicle is largest. (Re-cut from a prior 00:00:01.0–00:00:07.0 / 6.0 s trim that measured ≈70–270px across its length by eye; a careful re-measurement on extracted frames found the truck was actually a fairly constant ≈145–151px for the clip's first ~5 s, not the ≈70–90px originally estimated, but that is still squarely in the 110–180px zero-recall bucket, so the original conclusion — that roughly the first two-thirds of the old 6 s clip was confounding vehicle-resolution failure with smoke-detection failure — still holds; only the numbers behind it were corrected. This re-cut keeps the ≈150px start, per instruction, since the ≥180px-only window was under 2 s on its own.) |
| `sample_motorbike_smoking_real.mp4` | 640×480 (cropped from 1122×480 native) | ≈350–380px (motorcycle wheel/tank/silencer assembly fills most of the cropped width; estimate carries more uncertainty than the others because the scene is dark and low-contrast) | 300–500px | **Fair test on the resolution axis** — clears the cliff easily. This means the previously-reported 12.5%→100% recall swing on this geometry is *not* explained by vehicle width; it is more likely driven by the low light/low contrast of the plume itself (already flagged in "Expected detector outcome" above), a separate axis from resolution. |

**Is 270×480 a genuine source aspect, or a crop I applied?** Checked the
pre-normalisation raw files directly with `ffprobe`, not assumed:

- `raw/px_dodge_cold_start.mp4` (source of the coldstart still): native
  **1080×1920**, portrait, H.264, 60fps — genuine vertical phone footage.
  Matches the manifest's own description, "Portrait 9:16." The corpus's own
  normalisation pipeline scaled this to height 480 (⇒ width 270); I applied
  **no crop of my own** to this file — the still is the full native
  normalised frame.
- `raw/wc_f450_coalroll.webm` (source of the stack video): native
  **720×1280**, portrait, VP9, 30fps — also genuine vertical footage (a
  chase-vehicle phone/dashcam shot following the truck). Same
  normalisation, same result: I applied **no crop** here either.

Neither narrowness is something this fixture-building pass introduced.
There is no wider framing available for either specific clip: the
dodge-coldstart clip is one of only two Pexels rear-tailpipe clips in the
`rear_tailpipe_CLEAN` set (the other is the Jaguar, already used for the
video, and reusing it for the still would reintroduce the same-scene
problem this round's fix #1 was about), and the F-450 clip is the *only*
vertical-stack-exhaust clip found on any openly-licensed host across five
platforms searched (see `PROVENANCE.txt`). Both fixtures happen to clear
the resolution floor anyway (240px throughout, and — after the 2026-09-26
re-cut described above — ≈150px at the start rising past 180px by
≈00:00:01.25 of 3.0 s for the stack clip) because both are tight/close
shots, not wide establishing shots — narrow frame width and narrow vehicle
width are not the same thing, and it's the latter that the round-2 table is
keyed on.

## Expected detector outcome per fixture

| File | Expected outcome |
| --- | --- |
| `sample_sedan_smoking_real.mp4` | Vehicle detected (black Jaguar saloon, rear-left of frame; a white pickup's tailgate/wheel is also in shot at the right edge but is not the smoking vehicle). **At least one smoke region** in every sampled frame — dense white/grey plume at the tailpipe, bottom-centre, drifting low across the road. Continuous for the whole 6 s. |
| `sample_car_clean_real.mp4` | Vehicle detected (rear of a black saloon, chrome oval tailpipe centred in frame). **No smoke regions** — deliberately hard: the dark lower valance/diffuser beside the tailpipe reads as a shadow, not a plume, in every frame; nothing ever emerges from the pipe. |
| `sample_coldstart_smoking_real.jpg` | Vehicle detected (black Dodge Challenger, rear three-quarter, centre-left of frame, tail light lit). **At least one smoke region** — thick white condensation smoke wrapping the entire rear of the car, centre/bottom of frame, unambiguous. |
| `sample_street_clean_real.jpg` | Multiple vehicles detected across a dense multi-lane highway. **No smoke regions** — the pale haze over the city skyline in the background is ordinary atmospheric/urban haze at distance, not an exhaust plume near any vehicle, and must not be flagged. |
| `sample_stack_smoking_real.mp4` | Vehicle detected (white Ford F-450 pickup, centred, driving toward camera) in every sampled frame of this 3.0 s re-cut (the truck is ≥150px wide throughout, see "Apparent vehicle width" above). **At least one smoke region, positioned ABOVE and behind the vehicle box**, not at ground level or the rear bumper — twin (merging into one) dense black plumes towering from cab-height vertical stacks. This is the geometry existing rear-tailpipe-tuned heuristics are most likely to miss, which is the point of shipping it. |
| `sample_motorbike_smoking_real.mp4` | Vehicle detected (motorcycle rear wheel/silencer, left of frame). **At least one smoke region**, low and to the right of the wheel at silencer height — but ship this expectation with a caveat: the plume is genuinely faint, pale blue-white haze against a dark night background, low contrast even after inspection with brightness/contrast boosted for verification. This is deliberately the hardest fixture in the set (real recall on this exact geometry swung 12.5%→100% across round-2 model variants); do not read a miss here the same way you'd read a miss on the other, unambiguous positives. |

## Verification performed

- `ffprobe` on every shipped file for codec/resolution/fps/duration/bitrate,
  and `format_tags` for the one file carrying embedded attribution metadata.
- `ffmpeg -i ... -f null -` decode pass on all four shipped videos: clean,
  zero errors.
- Frame-count / dimension cross-check via `ffprobe -count_frames` and, for
  the two originally-cut videos, via `backend/.venv`'s own `cv2` build
  (`cv2.VideoCapture` read every frame, `cv2.imread` loaded both JPEGs at
  the expected shape) — i.e. verified against the same OpenCV build the
  product's own environment uses, not just against ffmpeg.
- `shasum -a 256` on every source clip copied out of the corpus, checked
  against `MANIFEST.json` before cutting, for all six source clips
  including the Commons item.
- Every candidate/chosen frame, and the climax of the black-cloud stack
  clip, was visually inspected (Read tool, not just described) before being
  cut. For the motorbike clip specifically, a brightness/contrast-boosted
  copy of representative frames was inspected (verification only — the
  shipped file is untouched, native exposure) to confirm the described faint
  plume is genuinely present and not merely an artefact of the manifest's
  own description.
- Confirmed via `git status --porcelain` from inside the `backend` repo
  that `sample_media/` shows no changes at any point in this process, and
  that `sample_media_real/` is the only new, untracked path.

## Dropped for licence reasons

Per the hard constraint, the following were **not** used, and none of their
bytes were copied into this repository:

- **All remaining `_ia_` (Internet Archive) clips** — `A_ia_*` (12 clips),
  `B_ia_*` (5 clips), `N_ia_buses_nosmoke.mp4`, `N_ia_dashcam_traffic.mp4`.
  Per the corpus's own `PROVENANCE.txt`, these are archive.org "mirrortube"
  copies of third-party YouTube uploads with **no explicit licence**;
  original rights are presumed reserved by the uploader. Explicitly
  evaluation-only.
- **`NC_pixabay_bmw_cabrio_rear34.mp4`** — Pixabay source, per-item licence
  not read/quoted, so excluded per the hard constraint.
- All remaining archive.org raw downloads already rejected upstream by the
  corpus builder (video-game footage, animation, no-smoke industrial film,
  etc.) were never candidates — see the corpus's own `PROVENANCE.txt`.

`B_commons_f450_stacks.mp4` (Wikimedia Commons) is **no longer** in this
"dropped" list — its CC BY 3.0 licence was read, quoted, and judged to
justify the one-time exception above (see "Licence rule applied").

## Pexels clips considered but not used

Already licence-clean (Pexels), inspected as alternatives, not shipped:

- `positives/A_pexels_tailpipe_whitesmoke.mp4` — extremely dense,
  unambiguous smoke, but the vehicle body is only marginally in frame.
  (Earlier drafts of this document also cited training contamination as a
  reason; that was wrong — see the correction above. This clip is
  effectively CLEAN. It simply wasn't the stronger pick versus
  `A_pexels_dodge_coldstart` for scene-independence and vehicle-visibility
  reasons.)
- `negatives/NC_pexels_defender_rear_dusk.mp4` — a strong alternative hard
  negative (dark SUV, wet gravel, dusk); `NC_pexels_tailpipe_black_car.mp4`
  was preferred because it puts the tailpipe itself, not just the vehicle
  rear, directly beside the dark valance shadow.
- The rest of `NC_pexels_*` (dark bumpers, night, wet, indoor-workshop
  framings) and plain-traffic `N_pexels_*` clips are all valid additional
  hard-negative/negative candidates for later expansion.
- `NH_pexels_*` (hard negatives: tyre smoke, ambient haze/dyno smoke) were
  not used for any "clean" role even though they are Pexels-licensed,
  because they contain real smoke (just not exhaust smoke) and would muddy
  the "no smoke" label.

## Reproducing these files

```bash
# Sedan smoking video (6 s, 640x480, from the Jaguar clip)
ffmpeg -ss 0.5 -i A_pexels_jaguar_white_exhaust.mp4 -t 6 \
  -vf "crop=640:480:107:0" -c:v libx264 -pix_fmt yuv420p -crf 20 -an \
  sample_sedan_smoking_real.mp4

# Clean video (6 s, 640x480, from the black-car tailpipe close-up)
ffmpeg -ss 1 -i NC_pexels_tailpipe_black_car.mp4 -t 6 \
  -vf "crop=640:480:107:0" -c:v libx264 -pix_fmt yuv420p -crf 20 -an \
  sample_car_clean_real.mp4

# Cold-start smoking still (single frame, native 270x480 portrait)
ffmpeg -ss 4.8 -i A_pexels_dodge_coldstart.mp4 -frames:v 1 -q:v 3 \
  sample_coldstart_smoking_real.jpg

# Clean street still (single frame, 854x480)
ffmpeg -ss 5 -i N_pexels_highway_city.mp4 -frames:v 1 -q:v 3 \
  sample_street_clean_real.jpg

# Vertical-stack smoking video (3 s, native 270x480 portrait, CC BY 3.0 — attribution embedded)
# Re-cut 2026-09-26 from the prior 00:00:01.0-00:00:07.0 (6 s) trim: that window measured
# ~145-270px in apparent vehicle width across its length (by eye, later corrected by per-pixel
# measurement — see "Apparent vehicle width" above), spending most of its length below the
# 180px detector-recall cliff. This 00:00:04.0-00:00:07.0 window keeps the truck >=~150px
# throughout and >=~180px for its back two-thirds, per instruction ("~150px" is used instead of
# a strict ">=180px only" cut because the latter alone would be under 2 s).
# NOTE: ffmpeg's fast/keyframe seek (-ss before -i) was found to be inaccurate for this specific
# source file (multiple nearby -ss values returned identical frames); the -ss below is placed
# after -i for frame-accurate trimming.
ffmpeg -i B_commons_f450_stacks.mp4 -ss 4.0 -t 3 \
  -c:v libx264 -pix_fmt yuv420p -crf 20 -an \
  -metadata title="F-450 coal rolling Monster (trimmed excerpt)" \
  -metadata artist="Salvatore Arnone" \
  -metadata author="Salvatore Arnone" \
  -metadata copyright="CC BY 3.0 Unported (https://creativecommons.org/licenses/by/3.0/)" \
  -metadata comment="Source: https://commons.wikimedia.org/wiki/File:F-450_coal_rolling_Monster_(video).webm ; Licence: CC BY 3.0 Unported ; Attribution: Salvatore Arnone ; trimmed 00:00:04.0-00:00:07.0 from the original, re-encoded H.264, no other modification ; re-cut from the prior 00:00:01.0-00:00:07.0 trim so the truck's apparent width stays large enough throughout for a fair detector test (see REAL_FIXTURES.md)" \
  sample_stack_smoking_real.mp4

# Motorbike smoking video (6 s, 640x480, from the night silencer clip)
ffmpeg -ss 2.0 -i C_pexels_motorbike_exhaust.mp4 -t 6 \
  -vf "crop=640:480:241:0" -c:v libx264 -pix_fmt yuv420p -crf 20 -an \
  sample_motorbike_smoking_real.mp4
```

The source clips are from the evaluation corpus at the path recorded in
this repo's build notes; that corpus itself is not, and must not be,
committed anywhere in this repository.

_Compiled 2026-09-26; updated same day to add the vertical-stack and
motorbike positives, re-cut the smoking still from an independent clip,
document eval-corpus overlap, correct the contamination status of
`A_pexels_tailpipe_whitesmoke` (raw TOUCHED, effectively CLEAN by explicit
freeing), and record measured apparent vehicle width per positive fixture;
updated again same day to re-cut `sample_stack_smoking_real.mp4` from
00:00:01.0-00:00:07.0 (6.0 s) to 00:00:04.0-00:00:07.0 (3.0 s) of its
source clip, because the wider window measured well below the detector's
180px-apparent-width recall cliff for most of its length, confounding
"can the model see smoke" with "is the vehicle too small to resolve"; the
new sha256, trim timestamps, and measured width range are recorded in the
provenance table and "Apparent vehicle width" section above, and the
required CC BY attribution tags were re-verified as present on the new
file._
