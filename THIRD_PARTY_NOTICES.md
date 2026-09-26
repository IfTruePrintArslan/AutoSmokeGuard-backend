# Third-party notices

This file records the third-party components shipped in or used by this
repository (AutoSmokeGuard backend), their licences, and where each is used.
It exists alongside `LICENSE` (the AGPL-3.0 licence text under which this
project itself is distributed) and does not replace it.

## Why this project is licensed AGPL-3.0

This repository bundles Ultralytics YOLO model weights and imports the
`ultralytics` Python package at runtime (see below). Ultralytics is
licensed AGPL-3.0-only. AGPL-3.0 is a copyleft licence: any work that
incorporates AGPL-3.0-licensed code, including by importing it as a library,
must itself be distributed under AGPL-3.0 (or a compatible licence) unless
the incorporator has obtained a separate commercial licence from the
copyright holder. **That is the reason this project as a whole is licensed
AGPL-3.0** (see `LICENSE`), not an independent choice made for its own sake.

Anyone redistributing this repository, or a modified version of it, or
running a modified version of it as a network service, must comply with the
AGPL-3.0 terms in `LICENSE` — most notably AGPL-3.0 §13, which requires
offering the complete corresponding source of the running (possibly
modified) version to every user who interacts with it over a network, not
only to those who receive a distributed copy. The alternative to AGPL
compliance is to obtain a commercial licence directly from Ultralytics
(see <https://www.ultralytics.com/license>); this project has not done so
and therefore relies on AGPL-3.0 compliance.

---

## Ultralytics YOLO — AGPL-3.0

- **What it is / where it's used:**
  - The `ultralytics` Python package, imported at runtime in
    `mlcore/detector.py:86` (`from ultralytics import YOLO`), and pinned in
    `requirements.txt` (`ultralytics==8.4.157`).
  - Stock, unmodified pretrained weight files shipped in this repository:
    `ml_assets/yolo11n.pt` and `ml_assets/yolov8n.pt` (vehicle detection).
  - Two packages installed automatically as part of the `ultralytics`
    dependency tree, also AGPL-3.0-only: `ultralytics-thop` (FLOPs/params
    counting utility) and `ultralytics-platform`. Neither is imported
    directly by this project's own code; they are transitive dependencies
    of `ultralytics` itself.
- **Licence:** AGPL-3.0-only (verified directly from each package's
  installed `dist-info/METADATA` in this project's `.venv`).
- **Upstream source:** <https://github.com/ultralytics/ultralytics>
- **Licence text:** <https://www.gnu.org/licenses/agpl-3.0.txt> (also
  reproduced verbatim in this repository's own `LICENSE` file).
- **Commercial alternative:** <https://www.ultralytics.com/license>

## smoke_unet.pt — not third-party

`ml_assets/smoke_unet.pt` is this project's own trained model (a U-Net
checkpoint for smoke segmentation), produced by this FYP group's own
training pipeline (`mlcore/training/`). It is not a third-party asset and
carries no separate third-party licence obligation. It is distributed as
part of this repository under the same AGPL-3.0 licence as the rest of the
project (see `LICENSE`).

## DejaVu Sans / DejaVu Sans Bold — Bitstream Vera licence

- **What it is / where it's used:** `reports/fonts/DejaVuSans.ttf` and
  `reports/fonts/DejaVuSans-Bold.ttf`, used by the PDF report renderer.
- **Licence:** Bitstream Vera licence (a permissive, MIT/BSD-like font
  licence with an additional trademark-notice-retention condition). The
  full licence text is already present in this repository at
  `reports/fonts/LICENSE_DEJAVU.txt` — see that file for the authoritative
  text; it is not reproduced here.
- **Upstream source:** <https://dejavu-fonts.github.io/>

## Real-footage fixtures — `sample_media_real/`

These fixtures are documented in full, including per-file provenance,
sha256 hashes, trims, and crops, in
`sample_media_real/REAL_FIXTURES.md`. The attributions below are carried
over verbatim from that file rather than re-derived, per its own
authoritative provenance table.

- **Five clips/stills are Pexels-derived** (`sample_sedan_smoking_real.mp4`,
  `sample_car_clean_real.mp4`, `sample_coldstart_smoking_real.jpg`,
  `sample_street_clean_real.jpg`, `sample_motorbike_smoking_real.mp4`):
  licensed under the **Pexels Licence** (<https://www.pexels.com/license/>),
  which is free to use, permits modification, and does not require
  attribution. Per-clip source URLs are listed in the provenance table in
  `sample_media_real/REAL_FIXTURES.md`.
- **One clip requires attribution:** `sample_stack_smoking_real.mp4` is cut
  from `File:F-450_coal_rolling_Monster_(video).webm` on Wikimedia Commons,
  licensed **CC BY 3.0 Unported**. Required attribution, reproduced here
  verbatim from `sample_media_real/REAL_FIXTURES.md`:

  > "F-450 coal rolling Monster (video).webm" by Salvatore Arnone, used under
  > CC BY 3.0 Unported (<https://creativecommons.org/licenses/by/3.0/>).
  > Source: <https://commons.wikimedia.org/wiki/File:F-450_coal_rolling_Monster_(video).webm>

  This attribution is also baked into the shipped file's own MP4 container
  metadata (`artist`/`copyright`/`comment`/`title` tags) — see
  `sample_media_real/REAL_FIXTURES.md` for the verified `ffprobe` output.

## Synthetic fixtures — `sample_media/`

- **What it is / where it's used:** `sample_media/sample_truck_smoking.mp4`,
  `sample_media/sample_car_clean.mp4`, `sample_media/sample_bus_smoking.jpg`,
  `sample_media/sample_street_clean.jpg`.
- **Provenance:** each file is derived from a real photograph in the
  **COCO128** sample image set, which ships bundled with the `ultralytics`
  package (source photographs `000000000257.jpg`, `000000000094.jpg`,
  `bus.jpg`, `000000000471.jpg` respectively — see
  `sample_media/README.md` for the exact mapping). The vehicles in these
  images are real photographs; any exhaust plume visible is **synthetic**,
  composited by this project's own fractal-Brownian-motion plume generator
  (`mlcore/training/synth_dataset.py`), not real smoke.
- **Licence:** COCO128 is a small sample subset of the COCO dataset
  distributed by Ultralytics for testing purposes; licence not verified
  independently by this project beyond Ultralytics' own distribution of it
  alongside the AGPL-3.0-licensed `ultralytics` package. Treated here as a
  test/demonstration fixture, not redistributed as a standalone dataset.

## Python dependencies — copyleft (individually listed)

The following direct or transitive dependency carries a copyleft licence
and is called out individually rather than folded into the general
paragraph below:

- **`psycopg` / `psycopg[binary]`** (PostgreSQL driver, `requirements.txt`,
  used when `DATABASE_URL` points at PostgreSQL) — **LGPL-3.0-only**
  (verified from the installed package's `dist-info/METADATA`,
  `License-Expression: LGPL-3.0-only`). LGPL is a weak-copyleft licence:
  unlike AGPL/GPL, linking against an LGPL library from separately
  distributed application code does not, by itself, require the
  application to also be LGPL/GPL-licensed, provided the LGPL component
  itself remains replaceable/re-linkable per LGPL §4-6. This project is
  AGPL-3.0 regardless, for the Ultralytics reason stated above; this entry
  exists for accuracy, not because psycopg is the reason.
  Upstream: <https://github.com/psycopg/psycopg>

## Python dependencies — permissive (general paragraph)

The remaining direct dependencies in `requirements.txt` and
`requirements-prod.txt` are distributed under permissive licences (MIT,
BSD, Apache-2.0, or an equivalent OSI-approved permissive licence) that do
not impose copyleft obligations on this project. These include, among
others: `Django`, `djangorestframework`,
`djangorestframework-simplejwt`, `django-cors-headers`, `django-filter`,
`drf-spectacular`, `whitenoise`, `python-dotenv` (web/API layer);
`torch`, `torchvision`, `opencv-python` (Apache-2.0, verified from
installed metadata), `numpy`, `pillow` (MIT-CMU, verified from installed
metadata) (ML/CV stack, aside from `ultralytics` itself, covered above);
`reportlab` (BSD licence, verified from installed metadata) (PDF
rendering); `gunicorn` (MIT, per upstream project documentation — **not**
independently verified against installed package metadata in this
repository's environment, since `gunicorn` is a production-only
dependency not installed in the local dev `.venv`); `pytest`,
`pytest-django`, `pytest-cov` (MIT, test tooling only, not distributed as
part of the running application).

No comprehensive transitive-dependency scan was performed beyond the
packages named above and their installed metadata where noted; this
section covers this project's own direct dependency list as declared in
`requirements.txt` / `requirements-prod.txt` (no `pyproject.toml` exists in
this repository at the time of writing).

---

_Compiled 2026-09-27. Licence facts for `ultralytics`, `ultralytics-thop`,
`ultralytics-platform`, `psycopg`, `psycopg-binary`, `reportlab`,
`pillow`, and `opencv-python` were verified directly against each
package's installed `dist-info/METADATA` in this repository's `.venv`,
not merely assumed from memory or PyPI page text. `gunicorn`'s licence
was not verified this way (see note above) and is stated on the basis of
its own public documentation only._
