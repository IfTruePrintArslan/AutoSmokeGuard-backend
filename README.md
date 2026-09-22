# AutoSmokeGuard — Backend

Django REST API for AutoSmokeGuard: upload traffic images/video, run a YOLO
vehicle-detection + U-Net smoke-segmentation pipeline, review per-vehicle
emission severity, and download a PDF report. This document covers the
backend in isolation; see the [root README](../README.md) for the one-command
launcher and a whole-system view, and [`API_CONTRACT.md`](../API_CONTRACT.md)
for the frozen wire format.

For the one-command launcher (`start.py` / `START.command` / `START.bat`),
see [`backend/scripts/README.md`](scripts/README.md) — it is not repeated
here.

## Architecture

```
config/          settings (single settings.py, env-driven), root urls, asgi/wsgi
common/          abstract models, pagination, permissions, validators, storage, the exception handler
accounts/        custom UUID user, JWT auth, register/login/logout/refresh/password-reset
uploads/         UploadedMedia, format/size/signature validation, upload/list/delete
analysis/        AnalysisResult/DetectedVehicle/SmokeRegion, the threaded job worker, status/history/dashboard
reports/         GeneratedReport, the ReportLab PDF builder, download
system_config/   SystemSetting singleton (app label "sysconfig" — see below)
health/          liveness probe
mlcore/          the ML layer — see mlcore/README.md
tools/           dev_runner.py (the launcher), seed.py, make_samples.py
tests/           TC-01..TC-15 plus supporting unit/integration tests
```

Each feature app owns its own `models.py`, `serializers.py`, `services.py`
(business logic) and `views.py` (thin HTTP layer), mounted flat under `/api/`
by `config/urls.py`. `common/` is imported by every other app and therefore
loads first in `INSTALLED_APPS`.

## Data model

All six SDS tables, plus the `SystemSetting` configuration singleton. Every
primary key is a UUID (see "why" in `accounts/models.py`'s docstring: no
sequential ID leaking user counts, no username in the product, and it lets
the API hand out the same identifier it stores).

| Table (`db_table`) | Model | Key fields | Relations |
|---|---|---|---|
| `accounts_user` | `User` | `user_id` (UUID PK), `email` (unique), `full_name`, `role` (`user`\|`admin`), `is_active`, `is_staff`, `created_at` | — |
| `accounts_password_reset_token` | `PasswordResetToken` | `token_id` (UUID PK), `token` (64-hex, unique), `expires_at`, `used` | FK → `User` |
| `uploads_media` | `UploadedMedia` | `media_id` (UUID PK), `filename`, `format`, `media_type`, `size_bytes`, `upload_timestamp`, `file`/`file_path`, `width`, `height`, `duration_seconds`, `checksum` | FK → `User` |
| `analysis_result` | `AnalysisResult` | `analysis_id` (UUID PK), `status`, `progress`, `stage`, `start_time`, `end_time`, `total_vehicles`, `total_smoke`, `frames_processed`, `avg_confidence`, `overall_severity`, `severity_counts` (JSON), `settings_snapshot` (JSON), `preview_path` | FK → `UploadedMedia`, FK → `User` (denormalised — see below) |
| `analysis_detected_vehicle` | `DetectedVehicle` | `vehicle_id` (UUID PK), `vehicle_type`, `bounding_box` (JSON), `confidence`, `frame_number`, `timestamp_seconds`, `crop_path` | FK → `AnalysisResult` |
| `analysis_smoke_region` | `SmokeRegion` | `smoke_id` (UUID PK), `mask_path`, `intensity`, `severity`, `confidence`, `area_ratio`, `opacity` | FK → `DetectedVehicle` |
| `reports_generated_report` | `GeneratedReport` | `report_id` (UUID PK), `report_path`, `generated_at`, `page_count`, `file_size_bytes` | O2O → `AnalysisResult` |
| `sysconfig_system_setting` | `SystemSetting` | singleton (`pk=1`): upload policy (`allowed_image_formats`, `allowed_video_formats`, `max_upload_mb`, `max_video_seconds`), model thresholds (`confidence_threshold`, `smoke_mask_threshold`), severity banding (`severity_low_max`, `severity_moderate_max`), `frame_sample_rate`, `auto_generate_pdf`, `updated_at`, `updated_by` | FK → `User` (`updated_by`, nullable) |

Two design decisions worth knowing about:

- **`AnalysisResult.user` is denormalised** from `media.user`. The history
  screen is the most-hit authenticated endpoint and filters by owner on every
  request; carrying the FK directly turns that into a single indexed scan
  instead of a join, and keeps `common.permissions.IsOwner` uniform across
  models.
- **There is no separate `Job` table.** `status`/`progress`/`stage` live
  directly on `AnalysisResult`; the `job_id` the API hands back for polling
  *is* the `analysis_id`. One row, one identifier.
- **`settings_snapshot` is frozen per run.** An admin can retune thresholds
  at any time (UC-09); a report generated last month must still be
  explainable in terms of the numbers it was actually produced with, so the
  effective configuration is copied onto the row before work starts.
  `settings_snapshot` also carries a reserved `_runtime` key
  (`analysis.services.RUNTIME_KEY`) holding output the frozen schema has no
  column for — device, segmenter mode, the annotated-frame list. The API
  strips it out of what it serves; only server-side code that reads the model
  directly sees it. Treat `settings_snapshot` as *not* pure configuration for
  that reason.

## API endpoints

The full request/response shapes are frozen in
[`API_CONTRACT.md`](../API_CONTRACT.md) and are machine-readable at
`GET /api/schema/` with a browsable UI at `GET /api/docs/`. Summary:

| Method & path | View | Auth |
|---|---|---|
| `POST /api/register` | `accounts.RegisterView` | public |
| `POST /api/login` | `accounts.LoginView` | public |
| `POST /api/refresh-token` | `accounts.RefreshTokenView` | public |
| `POST /api/logout` | `accounts.LogoutView` | required |
| `GET /api/me` | `accounts.MeView` | required |
| `POST /api/password-reset` | `accounts.PasswordResetRequestView` | public |
| `POST /api/password-reset/confirm` | `accounts.PasswordResetConfirmView` | public |
| `POST /api/upload` | `uploads.UploadView` | required |
| `GET /api/media` | `uploads.MediaListView` | required |
| `GET /api/media/{id}`, `DELETE /api/media/{id}` | `uploads.MediaDetailView` | required (owner) |
| `POST /api/analyze` | `analysis.AnalyzeView` | required |
| `GET /api/status/{job_id}` | `analysis.StatusView` | required (owner) |
| `GET /api/analysis`, `GET /api/history` | `analysis.AnalysisListView` / `HistoryView` | required |
| `GET /api/analysis/{id}`, `DELETE /api/analysis/{id}` | `analysis.AnalysisDetailView` | required (owner or admin for GET) |
| `POST /api/analysis/{id}/report` | `analysis.AnalysisReportView` | required (owner) |
| `GET /api/dashboard/stats` | `analysis.DashboardStatsView` | required |
| `GET /api/reports`, `GET /api/report/{id}`, `GET /api/download-report/{id}` | `reports.ReportListView` / `ReportDetailView` / `ReportDownloadView` | required (owner) |
| `GET/PATCH /api/settings` | `system_config.SystemSettingsView` | required (`PATCH` = admin only) |
| `GET /api/health/`, `GET /health/` (legacy alias) | `health.health_check` | public |

Permission classes (`common/permissions.py`): `IsAdminRole`, `IsOwner`,
`IsOwnerOrAdmin`, `IsAdminOrReadOnly`.

Every failure uses one envelope (`common/exceptions.py`,
`api_exception_handler`, wired via `REST_FRAMEWORK['EXCEPTION_HANDLER']`):

```json
{ "detail": "human readable", "code": "machine_code", "errors": { "field": ["msg"] } | null }
```

Django's own `ValidationError` is converted to DRF's so model-level and
serializer-level validation read identically to the client; anything DRF
does not recognise is logged with a full traceback under the `asg.api`
logger and returned as an opaque `500 server_error` so internals never leak.
Every list endpoint returns the same pagination envelope
(`common/pagination.py`, `StandardPagination`, page size 10):
`{count, page, pages, page_size, next, previous, results}`.

## The threaded job worker

`analysis/worker.py` is the most interesting part of the backend. The SDS
calls for a background job queue and names Celery; Celery needs a broker
(Redis/RabbitMQ) and a second long-running process, and this project has a
hard requirement that a grader can `git clone` and be running with one
command on a bare macOS or Windows laptop. So **the queue is the database and
the workers are a `concurrent.futures.ThreadPoolExecutor` inside the Django
process** (`analysis.worker.JobQueue`, module-level singleton `job_queue`,
default 2 threads, `ASG_WORKER_THREADS`). There is no `Job` table — `status`
lives directly on `AnalysisResult` and the `job_id` the API returns from
`POST /api/analyze` is the `analysis_id`.

**Claiming a job.** `run_analysis` claims its row with a conditional
`UPDATE analysis_result SET status='running' ... WHERE status='queued'`. That
`UPDATE` is the entire concurrency story: exactly one caller's statement
matches the predicate and gets rows-affected `1`; every other caller
(a retried submission, a queue replay, a double `enqueue`) sees `0` and
declines. This works without `SELECT ... FOR UPDATE`, which SQLite does not
support, and it is why `run_analysis` is safe to call twice with the same id.

**Progress throttling.** A 300-frame video would otherwise emit hundreds of
progress writes. `_ProgressReporter` throttles writes to one per 0.4 s
(`PROGRESS_WRITE_INTERVAL`) while always letting the terminal 100% through,
and skips a write entirely if the `(percent, stage)` pair is an exact repeat.
Writes exclude rows already in a terminal state, so a late callback cannot
resurrect a run already marked `failed`.

**Database-connection lifecycle.** Every thread Django touches gets its own
DB connection, and nothing closes it automatically for a thread that is not
a request. `_worker_entrypoint` — what a pool thread actually runs — always
closes the thread's connections in a `finally` (`_close_connections`), guarded
against closing inside an open transaction (which would poison it instead).
The *synchronous* path (worker disabled, or a test forcing inline execution)
deliberately does **not** close the connection, because there it belongs to
the caller (a request, or a pytest-django test transaction).

**Write contention under SQLite.** The worker thread and a web request can
write to the same tables at the same moment, and SQLite serialises writers
across the whole database file. `analysis/services.py`'s `retry_on_lock`
wraps every statement that can lose that race (creating the analysis row,
claiming it, saving its result, reading `SystemSetting`, deleting it) in up
to 6 attempts with jittered exponential back-off (`DB_RETRY_ATTEMPTS = 6`,
starting at `DB_RETRY_BACKOFF = 0.05`s, doubling each retry). Only genuine
"database/table is locked" errors are retried (`is_lock_error`); every other
`OperationalError` propagates untouched, so this can never mask a real bug by
silently repeating it.

**The process-wide inference lock, and why MPS made it necessary.** `mlcore`
caches one Ultralytics YOLO model and one `SmokeUNet` per (weights, device)
pair and shares them process-wide. Those caches are locked for *loading* but
not for *inference*, and neither torch nor Apple's Metal backend is safe to
drive from two threads at once. Two concurrent `analyze_media` calls on
Apple silicon do not just produce wrong numbers — they trip a hard Metal
assertion ("A command encoder is already encoding to this command buffer")
that `abort()`s the entire interpreter, killing every in-flight request and
every other queued job. This was reproduced and verified on this project's
own sample media. `analysis/worker.py` therefore serialises the forward pass
process-wide with `_INFERENCE_LOCK = threading.Lock()`, held around the
single call to `mlcore.analyze_media` inside `_execute`. The trade-off is
head-of-line blocking — a 90-second video delays a photo queued behind it —
which is the correct behaviour for a single-accelerator box and is exactly
the seam where a real broker plus one worker *process* per GPU would take
over. The thread pool still buys what it was added for even with the lock:
the HTTP request never blocks on inference, the queue is bounded and
ordered, and file I/O / database writes / PDF rendering for one job overlap
with another job's inference.

**Restart recovery.** An in-process queue cannot survive a process restart —
anything left `running` has no thread behind it, anything left `queued` has
no submission behind it. `recover_interrupted_jobs()` runs on a daemon thread
at boot (`start_bootstrap` → `_bootstrap`) and fails both categories
immediately with `error_message = "interrupted by server restart"` rather
than leaving a progress bar spinning forever. This assumes a single serving
process; a multi-process deployment would need the sweep scoped by a
process/heartbeat column, which is again the point at which the Celery seam
is the right answer. Bootstrap is skipped under pytest, for non-serving
management commands (migrate, spectacular, etc. — see
`NON_SERVING_COMMANDS`), and in the `runserver` autoreloader's parent
process, so torch is never loaded twice or raced against a cold migration.

**User-facing failure messages.** `_mark_failed` logs the full exception with
a traceback but stores only a whitelisted, human sentence
(`_SAFE_MESSAGES`/`user_safe_error`) on the row — the mapping is keyed by
*exception class name only*, never `str(exc)`, because exception text
routinely carries absolute paths and library internals that must not reach a
report or a UI.

## The ML pipeline

`mlcore.analyze_media` (called from `worker._execute`) runs, per frame:
detect vehicles with YOLO → compute each vehicle's exhaust ROI → segment
smoke in that ROI with `SmokeUNet` (or the classical fallback) → gate the
result against three density/area floors → score severity → write annotated
artifacts (crops, masks, up to 12 annotated frames, one preview image).
Progress is reported at 5 (loading), 15 (detecting), 40→72 (segmenting,
scaled across the frame loop), 90 (writing artifacts), 100 (done). See
[`mlcore/README.md`](mlcore/README.md) for the full technical detail —
architecture, dataset generation, training recipe, the four fixes that
reached the 0.80 NFR, the classical fallback's measured 13% recall, and the
`largest_blob_ratio` calibration gate.

## Configuration reference

### Environment variables (`backend/.env`, see `.env.example`)

| Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | dev-only fallback string | Django secret key |
| `DEBUG` | `True` | only the exact string `"True"` enables debug mode |
| `ALLOWED_HOSTS` | `127.0.0.1,localhost` | comma-separated |
| `CORS_ORIGINS` | Vite dev (5173) + preview (4173), both `localhost`/`127.0.0.1` | comma-separated allowed browser origins |
| `DATABASE_URL` | unset (→ SQLite at `backend/db.sqlite3`) | a `postgres://`/`postgresql://` URL switches to PostgreSQL |
| `DB_CONN_MAX_AGE` | `60` | PostgreSQL connection reuse, seconds |
| `ASG_WORKER_THREADS` | `2` | analysis worker pool size |
| `ASG_WORKER_ENABLED` | `True` | `False` accepts uploads but never processes them (used by CI/tests) |
| `ASG_MAX_UPLOAD_MB` | `512` | boot-time ceiling; live value is `SystemSetting.max_upload_mb` |
| `ASG_ML_ASSETS` | `backend/ml_assets` | override where `mlcore` looks for weights/datasets |
| `ASG_FORCE_CPU` | unset | `1`/`true`/`yes`/`on` forces `cpu`, skipping `cuda`/`mps` auto-detection |
| `ASG_WORKER_BOOTSTRAP` | unset | `False` disables the boot-time model warm-up and recovery sweep entirely |
| `SECURE_SSL_REDIRECT`, `SECURE_HSTS_SECONDS`, `SESSION_COOKIE_SECURE`, `CSRF_COOKIE_SECURE` | production defaults | only read when `DEBUG=False` |

### `SystemSetting` fields (UC-09, `GET`/`PATCH /api/settings`)

See the data-model table above for the full field list; `PATCH` requires
`role=admin` (else `403 admin_required`) and stamps `updated_at`/`updated_by`.
The response also includes a read-only `runtime` block: `device`
(`mps`/`cpu`/`cuda`), `segmenter_mode` (`unet`/`classical`),
`yolo_weights_present`, `segmenter_weights_present`, `worker_threads`, and
`model_metrics` (the `dice`/`iou`/`pixel_accuracy` from `ml_assets/metrics.json`,
or `null` if it has not been generated).

## The `sysconfig` → `system_config` naming decision

Django apps live at the project root, which Python puts first on `sys.path`.
A package literally named `sysconfig` there shadows the **standard-library**
`sysconfig` module, and `zoneinfo` imports it during Django start-up — so
naming the app directory `sysconfig` makes Django refuse to boot at all, with
`AttributeError: module 'sysconfig' has no attribute 'get_config_var'`
(the ML stack hits the same collision independently: `torch._dynamo.config`
calls `sysconfig.get_config_var()` on import, which is why `mlcore/_compat.py`
carries its own defensive guard — see `mlcore/README.md`).

The fix: the directory is `system_config`, while
`system_config.apps.SystemConfigConfig` pins `label = 'sysconfig'`. Everything
SDS-facing keeps the documented name — the app label, the
`sysconfig_system_setting` table, model references written as
`'sysconfig.SystemSetting'`, and management commands such as `manage.py
makemigrations sysconfig`. Only the Python **import path** differs
(`from system_config.models import SystemSetting`).

## Running tests

```bash
cd backend
source .venv/bin/activate        # or backend\.venv\Scripts\activate on Windows
pytest                           # full suite
pytest -m slow                   # includes test_real_pipeline_end_to_end (real YOLO + U-Net on sample_media)
pytest tests/test_analysis.py -k tc14   # one file / one test
pytest --cov=. --cov-report=term-missing   # coverage (pytest-cov is installed)
```

`tests/conftest.py` provides `api` (unauthenticated `APIClient`), `auth_client`
/ `admin_client` (JWT-authenticated, minted directly against SimpleJWT — see
`BUILD_PLAN.md` gauntlet item G6 for the open question of switching this to a
real `POST /api/login`), `sample_files` (real byte-accurate JPEG/PNG/MP4/AVI
payloads), and an autouse `tmp_media` fixture that points `MEDIA_ROOT` at a
throwaway per-test directory. `tests/test_*.py` map onto the project's TC-01
through TC-15 test-design IDs; see `docs/TRACEABILITY.md` for the exact
function names and pass/fail status.

`python -m mlcore.selftest` is a separate, non-pytest end-to-end check that
exercises the real pipeline against `backend/sample_media/` and asserts the
NFR budgets directly (see `mlcore/README.md`).
