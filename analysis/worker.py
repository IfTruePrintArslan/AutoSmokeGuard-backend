"""
The background job worker — an in-process thread pool backed by the database.

Why not Celery
--------------
The SDS asks for background workers and names Celery.  Celery needs a broker
(Redis or RabbitMQ) and a second long-running process.  This project has a
hard non-functional requirement that a grader on a clean macOS or Windows
laptop can ``git clone`` and be running with one command — no Redis, no
``celery -A``, no Docker.  A broker would be the single biggest source of
"works on my machine".

So the queue is the database and the workers are a
:class:`~concurrent.futures.ThreadPoolExecutor` inside the Django process.
``AnalysisResult.status`` *is* the job state, which is why there is no Job
table: the id the front end polls is the analysis id.

The Celery-shaped seam is deliberate.  :func:`run_analysis` is a plain
``(analysis_id) -> None`` function with no shared state, no closures over
request objects and no return value anyone depends on.  Swapping
``job_queue.enqueue`` for ``run_analysis.delay`` is a ten-line change if this
ever needs to scale past one box.

Four things this module is careful about
----------------------------------------
**Database connections.**  Every thread Django touches gets its own
connection, and nothing ever closes it for a thread that is not a request.
Leak them and SQLite runs out of handles and the test suite deadlocks on a
write lock.  Everything that runs in a pool thread therefore goes through
:func:`_worker_entrypoint`, which closes the thread's connections in a
``finally``.  The synchronous path deliberately does *not* close, because
there it is the caller's (a request's, or a test transaction's) connection.

**Idempotency.**  ``run_analysis`` claims its row with a conditional
``UPDATE ... WHERE status = 'queued'``.  That is a compare-and-set the
database performs atomically, so a double submission results in exactly one
execution — without ``SELECT FOR UPDATE``, which SQLite does not support.

**Write pressure.**  A 300-frame video would otherwise emit hundreds of
progress updates.  :class:`_ProgressReporter` throttles them to one write per
:data:`PROGRESS_WRITE_INTERVAL` seconds while always letting the terminal
100% through.  Statements that can still lose a race for SQLite's write lock
go through :func:`analysis.services.retry_on_lock`.

**Inference is single-file.**  ``mlcore`` shares one detector and one
segmenter across the whole process, and driving them from two threads at once
aborts the interpreter on Apple silicon.  :data:`_INFERENCE_LOCK` serialises
the forward pass; see the long comment on it for the full reasoning.
"""
import atexit
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_for_futures
from pathlib import Path

from django.conf import settings
from django.db import OperationalError, connections, transaction
from django.utils import timezone

from common.storage import analysis_artifact_dir, to_media_relative
from system_config.models import SystemSetting

from .models import (
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MODERATE,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_QUEUED,
    STATUS_RUNNING,
    TERMINAL_STATUSES,
    AnalysisResult,
    DetectedVehicle,
    SmokeRegion,
)
from .services import (
    RUNTIME_KEY,
    build_ml_config,
    is_lock_error,
    retry_on_lock,
    worst_severity,
)

logger = logging.getLogger('asg.worker')

#: Minimum wall-clock gap between two progress writes for one analysis.
PROGRESS_WRITE_INTERVAL = 0.4

# ---------------------------------------------------------------------------
# Why inference is serialised
#
# ``mlcore`` caches one Ultralytics model and one U-Net per (weights, device)
# pair and shares them process-wide (``mlcore.detector._MODEL_CACHE``,
# ``mlcore.segmenter._CHECKPOINT_CACHE``).  Those caches are locked for
# *loading* but not for *inference*, and neither torch modules nor the Metal
# backend are safe to drive from two threads at once.  Two concurrent
# ``analyze_media`` calls on an Apple-silicon machine do not merely give wrong
# numbers — they trip a Metal assertion
#
#   "A command encoder is already encoding to this command buffer"
#
# which is a hard ``abort()``: the entire Django process dies, taking every
# in-flight request and both queued jobs with it.  Verified on this project's
# own sample media before this lock existed.
#
# So the forward pass is serialised process-wide.  This costs very little
# real throughput — there is one GPU, and two pipelines sharing it would each
# run at roughly half speed anyway — while the thread pool still buys the
# things it was added for: the request never blocks, the queue is bounded and
# ordered, and file I/O, database writes and PDF rendering for one job still
# overlap with another job's inference.
#
# The trade-off is head-of-line blocking: a 90-second video delays a photo
# queued behind it.  That is the correct behaviour for a single-accelerator
# box, and it is the seam where a real broker plus one worker *process* per
# GPU would take over.
# ---------------------------------------------------------------------------
_INFERENCE_LOCK = threading.Lock()

#: ``AnalysisResult.stage`` is a 50-character column.
STAGE_MAX_LENGTH = 50

#: The only values ``AnalysisResult.overall_severity`` (a ``SEVERITY_CHOICES``
#: column) accepts.  ``mlcore`` uses a wider vocabulary internally (it reports
#: ``"none"`` for "no smoke was ever seen"); anything outside this set maps to
#: ``''``, the model's own "no verdict" value.
VALID_OVERALL_SEVERITIES = frozenset({SEVERITY_LOW, SEVERITY_MODERATE, SEVERITY_HIGH})

#: Marker written onto jobs that were in flight when the process died.
INTERRUPTED_MESSAGE = 'interrupted by server restart'

#: Management commands that must never spin up a worker or load torch.
#: ``migrate`` on a cold database would otherwise race the recovery sweep, and
#: ``spectacular``/``check`` would pay a multi-second import for nothing.
NON_SERVING_COMMANDS = frozenset({
    'check', 'collectstatic', 'compilemessages', 'createcachetable',
    'createsuperuser', 'dbshell', 'diffsettings', 'dumpdata', 'flush',
    'inspectdb', 'loaddata', 'makemessages', 'makemigrations', 'migrate',
    'sendtestemail', 'shell', 'showmigrations', 'spectacular', 'sqlflush',
    'sqlmigrate', 'sqlsequencereset', 'squashmigrations', 'startapp',
    'startproject', 'test', 'testserver',
})


# ---------------------------------------------------------------------------
# User-facing failure messages
#
# An error_message is rendered verbatim in the UI and embedded in the PDF, so
# it must never contain a traceback, a module name or a filesystem path.  The
# full detail goes to the log; the user gets a sentence they can act on.
# ---------------------------------------------------------------------------

_SAFE_MESSAGES = {
    'MLPipelineError': 'The media file could not be opened or decoded. '
                       'Please re-upload it and try again.',
    'MediaError': 'The media file could not be opened or decoded. '
                  'Please re-upload it and try again.',
    'FileNotFoundError': 'The uploaded file is no longer available on the '
                         'server. Please upload it again.',
    'PermissionError': 'The server could not read the uploaded file.',
    'MemoryError': 'The server ran out of memory while processing this file. '
                   'Try a shorter clip or a smaller image.',
    'ModuleNotFoundError': 'The analysis engine is not available on this '
                           'server. Please contact an administrator.',
    'ImportError': 'The analysis engine is not available on this server. '
                   'Please contact an administrator.',
    'OperationalError': 'The server was too busy to record this analysis. '
                        'Please try again in a moment.',
    'DatabaseError': 'The results of this analysis could not be saved. '
                     'Please try again.',
    'OSError': 'The server could not read or write the files needed for this '
               'analysis.',
}

DEFAULT_ERROR_MESSAGE = ('Analysis failed because of an unexpected server '
                         'error. Please try again, or contact an '
                         'administrator if it keeps happening.')


def user_safe_error(exc):
    """
    Map an exception to a sentence that is safe to show a user.

    Only the exception *class* is consulted — never ``str(exc)``, which
    routinely carries absolute paths ("cannot open /Users/ali/...") and
    library internals.  Unknown failures collapse to a single generic line.
    """
    for klass in type(exc).__mro__:
        message = _SAFE_MESSAGES.get(klass.__name__)
        if message:
            return message
    return DEFAULT_ERROR_MESSAGE




# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

class _ProgressReporter:
    """
    Throttled ``(percent, stage) -> UPDATE`` callback handed to the pipeline.

    Instances are per-run and are touched from exactly one thread, but the
    lock is kept anyway: ``mlcore`` is free to parallelise internally, and a
    torn read of ``_last_write`` would defeat the throttle silently.
    """

    def __init__(self, analysis_id, min_interval=PROGRESS_WRITE_INTERVAL):
        self._analysis_id = analysis_id
        self._min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._last_write = 0.0
        self._last_state = (None, None)
        self.writes = 0

    def __call__(self, percent, stage):
        """Persist progress, unless it is a duplicate or we wrote recently."""
        percent = max(0, min(100, int(percent or 0)))
        stage = str(stage or '')[:STAGE_MAX_LENGTH]
        now = time.monotonic()

        with self._lock:
            if (percent, stage) == self._last_state:
                return
            if percent < 100 and (now - self._last_write) < self._min_interval:
                return
            self._last_write = now
            self._last_state = (percent, stage)
            self.writes += 1

        # Outside the lock: never hold a mutex across a database round-trip.
        # ``exclude(TERMINAL)`` stops a late callback from resurrecting a run
        # that has already been marked failed.
        try:
            AnalysisResult.objects.filter(pk=self._analysis_id).exclude(
                status__in=TERMINAL_STATUSES,
            ).update(progress=percent, stage=stage)
        except OperationalError as exc:
            # Progress is advisory. Losing one update costs a stale progress
            # bar for 400 ms; failing the run over it would be indefensible.
            if not is_lock_error(exc):
                raise
            logger.debug('Dropped a progress update for %s: %s',
                         self._analysis_id, exc)


# ---------------------------------------------------------------------------
# The unit of work
# ---------------------------------------------------------------------------

def run_analysis(analysis_id):
    """
    Execute one analysis end to end. Never raises.

    Safe to call twice with the same id — the second call finds the row in
    ``running`` and returns immediately.  The caller owns the database
    connection lifecycle; use :func:`_worker_entrypoint` from a pool thread.
    """
    analysis = _claim(analysis_id)
    if analysis is None:
        return None

    try:
        _execute(analysis)
    except Exception as exc:                          # noqa: BLE001 - by design
        _mark_failed(analysis_id, exc)
        return None

    _maybe_generate_report(analysis)
    return analysis


def _claim(analysis_id):
    """
    Atomically move a ``queued`` row to ``running``, or decline the job.

    The conditional ``UPDATE`` is the whole concurrency story: whichever
    thread's statement matches the ``status='queued'`` predicate wins, and
    every other caller sees ``0`` rows affected.
    """
    claimed = retry_on_lock(
        lambda: AnalysisResult.objects.filter(
            pk=analysis_id, status=STATUS_QUEUED,
        ).update(
            status=STATUS_RUNNING,
            progress=0,
            stage='starting',
            error_message='',
            start_time=timezone.now(),
            end_time=None,
        ),
        f'claiming analysis {analysis_id}',
    )
    if not claimed:
        logger.info('Analysis %s is not queued; skipping duplicate submission.',
                    analysis_id)
        return None

    try:
        return retry_on_lock(
            lambda: AnalysisResult.objects.select_related('media')
            .get(pk=analysis_id),
            f'loading analysis {analysis_id}',
        )
    except AnalysisResult.DoesNotExist:               # pragma: no cover
        logger.warning('Analysis %s vanished between claim and load.',
                       analysis_id)
        return None


def _execute(analysis):
    """Run the pipeline and persist everything it produced."""
    from mlcore import analyze_media

    snapshot = analysis.settings_snapshot or {}
    config = build_ml_config(snapshot)

    media = analysis.media
    media_path = Path(settings.MEDIA_ROOT) / (media.file_path or media.file.name)
    output_dir = analysis_artifact_dir(analysis.analysis_id)

    logger.info('Analysis %s starting on %s', analysis.analysis_id,
                media.filename)
    started = time.monotonic()
    # One forward pass at a time, process-wide. See the comment on
    # _INFERENCE_LOCK — without it, two threads on MPS abort the process.
    with _INFERENCE_LOCK:
        waited = time.monotonic() - started
        if waited > 0.1:
            logger.info('Analysis %s waited %.2fs for the inference lock',
                        analysis.analysis_id, waited)
        result = analyze_media(
            str(media_path),
            str(output_dir),
            config,
            progress_cb=_ProgressReporter(analysis.analysis_id),
        )
    elapsed = time.monotonic() - started

    _persist(analysis, result or {}, output_dir)
    logger.info('Analysis %s finished in %.2fs (%s vehicles, %s smoke)',
                analysis.analysis_id, elapsed, analysis.total_vehicles,
                analysis.total_smoke)


def _persist(analysis, result, output_dir):
    """
    Write the pipeline's output to the database in one transaction.

    Two things happen here that are easy to miss:

    * **Path rebasing.**  ``mlcore`` returns every path relative to its own
      ``output_dir``; the database stores paths relative to ``MEDIA_ROOT``.
      Skipping the conversion produces URLs that 404 in a way that only shows
      up in the browser.
    * **Per-region severity re-banding.**  Each region's individual
      classification was made with the thresholds the pipeline was handed;
      the stored per-region severity (``SmokeRegion.severity``, and therefore
      ``severity_counts``, the distribution the donut chart reads) is
      re-derived from the *live* :class:`SystemSetting` so what the history
      screen filters on always matches the bands currently configured.
    * **``overall_severity`` is not re-derived.**  Unlike the per-region
      bands above, the single media-level verdict is taken verbatim from the
      pipeline's own :func:`mlcore.intensity.aggregate` result — the same
      worst-case-present rule the ML log and the PDF are built from — so
      there is exactly one implementation of "how bad was this, overall?"
      instead of two that can quietly disagree (see the module-level
      docstring of :mod:`mlcore.intensity`).  Because that verdict was
      computed against the *frozen* run-start thresholds while the counts
      above use the *live* ones, the two can legitimately diverge if an
      admin retunes the bands while a job is in flight; :func:`_persist` logs
      that as a warning rather than silently resolving it one way or the
      other.
    """
    setting = retry_on_lock(SystemSetting.get_solo, 'reading system settings')
    output_dir = Path(output_dir)

    def rebase(relative):
        """ML-relative path -> MEDIA_ROOT-relative path (``''`` when absent)."""
        if not relative:
            return ''
        return to_media_relative(output_dir / str(relative))

    vehicles = []
    smoke_regions = []
    counts = {'low': 0, 'moderate': 0, 'high': 0}
    confidences = []

    for record in result.get('vehicles') or []:
        vehicle = DetectedVehicle(
            analysis=analysis,
            vehicle_type=str(record.get('vehicle_type') or 'car')[:30],
            bounding_box=record.get('bounding_box') or {},
            confidence=float(record.get('confidence') or 0.0),
            frame_number=record.get('frame_number'),
            timestamp_seconds=record.get('timestamp_seconds'),
            crop_path=rebase(record.get('crop_path')),
        )
        vehicles.append(vehicle)
        confidences.append(vehicle.confidence)

        smoke = record.get('smoke')
        if not smoke:
            continue

        intensity = float(smoke.get('intensity') or 0.0)
        severity = setting.severity_for(intensity)
        counts[severity] += 1
        smoke_regions.append(SmokeRegion(
            vehicle=vehicle,
            mask_path=rebase(smoke.get('mask_path')),
            intensity=intensity,
            severity=severity,
            confidence=float(smoke.get('confidence') or 0.0),
            area_ratio=float(smoke.get('area_ratio') or 0.0),
            opacity=float(smoke.get('opacity') or 0.0),
        ))

    annotated_frames = [
        rebase(frame) for frame in (result.get('annotated_frames') or [])
    ]
    runtime = {
        'segmenter_mode': result.get('segmenter_mode') or '',
        'device': result.get('device') or '',
        'annotated_frames': [frame for frame in annotated_frames if frame],
        'elapsed_seconds': result.get('elapsed_seconds'),
        'mean_intensity': result.get('mean_intensity'),
        'media_meta': result.get('media_meta') or {},
    }

    reported_confidence = result.get('avg_confidence')
    if reported_confidence is None and confidences:
        reported_confidence = sum(confidences) / len(confidences)

    snapshot = dict(analysis.settings_snapshot or {})
    snapshot[RUNTIME_KEY] = runtime

    analysis.total_vehicles = len(vehicles)
    analysis.total_smoke = len(smoke_regions)
    analysis.frames_processed = int(result.get('frames_processed') or 0)
    analysis.avg_confidence = round(float(reported_confidence or 0.0), 6)
    analysis.severity_counts = counts

    # Single source of truth for the media-level verdict: whatever mlcore's
    # aggregate() decided (worst-case-present), not a second computation here.
    # See the docstring above for why this can differ from `counts`.
    pipeline_severity = result.get('overall_severity')
    if pipeline_severity not in VALID_OVERALL_SEVERITIES:
        pipeline_severity = ''  # mlcore's "none", or nothing usable at all.
    analysis.overall_severity = pipeline_severity

    # Cheap drift check, not a second implementation: if the live re-banded
    # counts disagree with the pipeline's own verdict, that is either a live
    # threshold change mid-run (expected, rare) or a real bug (not expected,
    # worth knowing about either way).
    rebanded_severity = worst_severity(counts)
    if rebanded_severity != analysis.overall_severity:
        logger.warning(
            "Analysis %s: pipeline overall_severity=%r disagrees with the "
            "live-rebanded worst_severity=%r; likely a SystemSetting "
            "threshold change while this run was in flight.",
            analysis.analysis_id, analysis.overall_severity, rebanded_severity,
        )

    analysis.preview_path = rebase(result.get('preview_path'))
    analysis.settings_snapshot = snapshot
    analysis.status = STATUS_DONE
    analysis.progress = 100
    analysis.stage = 'done'
    analysis.error_message = ''
    analysis.end_time = timezone.now()

    def write():
        """The whole result, atomically. Retried as a unit if it is locked."""
        with transaction.atomic():
            # Re-running an analysis in place must not duplicate its children.
            # SmokeRegion rows cascade off the vehicles.
            analysis.vehicles.all().delete()
            if vehicles:
                DetectedVehicle.objects.bulk_create(vehicles, batch_size=500)
            if smoke_regions:
                SmokeRegion.objects.bulk_create(smoke_regions, batch_size=500)
            analysis.save(update_fields=[
                'total_vehicles', 'total_smoke', 'frames_processed',
                'avg_confidence', 'severity_counts', 'overall_severity',
                'preview_path', 'settings_snapshot', 'status', 'progress',
                'stage', 'error_message', 'end_time',
            ])

    retry_on_lock(write, f'saving analysis {analysis.analysis_id}')


def _maybe_generate_report(analysis):
    """
    Produce the PDF when the frozen configuration asked for one.

    Called *after* the analysis is already marked ``done``, not before: the
    measurement is the product, the PDF is a rendering of it.  A reports app
    that is missing, broken or mid-rewrite must never turn a successful
    analysis into a failed one, so every exception is swallowed after being
    logged in full.
    """
    if not (analysis.settings_snapshot or {}).get('auto_generate_pdf'):
        return
    try:
        from reports.services import generate_report_for_analysis
        generate_report_for_analysis(analysis)
    except Exception:                                 # noqa: BLE001 - by design
        logger.exception('report generation failed; analysis %s still succeeds',
                         analysis.analysis_id)


def _mark_failed(analysis_id, exc):
    """Record a failure: full detail to the log, one safe sentence to the row."""
    logger.exception('Analysis %s failed', analysis_id)
    try:
        retry_on_lock(
            lambda: AnalysisResult.objects.filter(pk=analysis_id).update(
                status=STATUS_FAILED,
                stage='failed',
                error_message=user_safe_error(exc),
                end_time=timezone.now(),
            ),
            f'failing analysis {analysis_id}',
        )
    except Exception:                                 # noqa: BLE001
        # If even the failure cannot be recorded, the restart sweep will.
        logger.exception('Could not record the failure of analysis %s',
                         analysis_id)


# ---------------------------------------------------------------------------
# Thread plumbing
# ---------------------------------------------------------------------------

def _close_connections():
    """
    Hand this thread's database connections back.

    Guarded against being called inside an open transaction: Django's
    ``close()`` marks an in-flight atomic block as needing rollback, which
    would silently poison a request (or a pytest-django test transaction) that
    ran the job synchronously.
    """
    for connection in connections.all(initialized_only=True):
        try:
            if connection.in_atomic_block:
                continue
            connection.close()
        except Exception:                             # pragma: no cover
            logger.exception('Failed to close a worker database connection')


def _worker_entrypoint(analysis_id):
    """What a pool thread actually runs: the job, then connection cleanup."""
    try:
        run_analysis(analysis_id)
    except Exception:                                 # pragma: no cover
        logger.exception('Worker thread crashed on analysis %s', analysis_id)
    finally:
        _close_connections()


def worker_enabled():
    """
    Whether jobs run on the thread pool (``True``) or inline (``False``).

    Read from settings on every call rather than cached at import: the test
    suite flips ``ASG['WORKER_ENABLED']`` per test to make execution
    deterministic, and ``--check`` runs want the inline path too.
    """
    return bool(getattr(settings, 'ASG', {}).get('WORKER_ENABLED', True))


def worker_threads():
    """Configured pool size, never below one."""
    return max(1, int(getattr(settings, 'ASG', {}).get('WORKER_THREADS', 2) or 1))


class JobQueue:
    """
    The process-wide submission point for analysis jobs.

    One instance, created at import (:data:`job_queue`).  The executor itself
    is built lazily on first use so that importing this module — which
    ``apps.py`` does at startup, and which ``manage.py check`` therefore does
    too — never spawns threads on its own.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._executor = None
        self._futures = {}
        self._closed = False

    # -- submission -----------------------------------------------------

    def enqueue(self, analysis_id):
        """
        Mark ``analysis_id`` queued and arrange for it to run.

        Returns ``True`` when the job was accepted.  ``False`` means the row
        was already past ``queued`` (someone beat us to it) or the pool is
        shutting down — both are normal, neither is an error.

        When the worker is disabled the job runs *synchronously, on this
        thread*, before returning.  That is what makes the test suite
        deterministic and what lets a ``--check`` boot prove the pipeline
        end to end without a second thread.
        """
        analysis_id = str(analysis_id)
        accepted = retry_on_lock(
            lambda: AnalysisResult.objects.filter(
                pk=analysis_id, status__in=(STATUS_PENDING, STATUS_QUEUED),
            ).update(status=STATUS_QUEUED, progress=0, stage='queued',
                     error_message=''),
            f'queueing analysis {analysis_id}',
        )
        if not accepted:
            logger.info('Analysis %s is no longer pending; not enqueued.',
                        analysis_id)
            return False

        if not worker_enabled():
            logger.debug('Worker disabled — running analysis %s inline.',
                         analysis_id)
            run_analysis(analysis_id)
            return True

        with self._lock:
            if self._closed:
                logger.warning('Queue is shutting down; analysis %s stays '
                               'queued.', analysis_id)
                return False
            executor = self._ensure_executor()
            future = executor.submit(_worker_entrypoint, analysis_id)
            self._futures[analysis_id] = future

        future.add_done_callback(
            lambda finished, key=analysis_id: self._forget(key),
        )
        return True

    # -- lifecycle ------------------------------------------------------

    def _ensure_executor(self):
        """Create the pool on first use. Caller must hold ``self._lock``."""
        if self._executor is None:
            size = worker_threads()
            logger.info('Starting analysis worker pool with %d thread(s)', size)
            self._executor = ThreadPoolExecutor(
                max_workers=size, thread_name_prefix='asg-worker',
            )
        return self._executor

    def _forget(self, analysis_id):
        """Drop a finished future so the map cannot grow without bound."""
        with self._lock:
            self._futures.pop(analysis_id, None)

    @property
    def pending(self):
        """How many submitted jobs have not finished yet."""
        with self._lock:
            return len(self._futures)

    def wait(self, timeout=None):
        """
        Block until every submitted job has finished. Returns ``True`` if so.

        Exists for tests and for a clean shutdown; nothing in the request path
        should ever wait on a job.
        """
        with self._lock:
            futures = list(self._futures.values())
        if not futures:
            return True
        done, not_done = wait_for_futures(futures, timeout=timeout)
        return not not_done

    def shutdown(self, wait=True, timeout=None):
        """
        Stop accepting work and let in-flight jobs finish.

        Queued-but-not-started jobs are cancelled rather than run: their rows
        stay ``queued`` and the next boot's recovery sweep resolves them.
        """
        with self._lock:
            if self._closed and self._executor is None:
                return
            self._closed = True
            executor, self._executor = self._executor, None
            self._futures.clear()
        if executor is not None:
            logger.info('Shutting down the analysis worker pool')
            executor.shutdown(wait=wait, cancel_futures=True)

    def reset(self):
        """
        Tear the pool down and allow a fresh one — test support only.

        Lets a test change ``ASG['WORKER_THREADS']`` and get a pool that
        actually honours it, which a module-level singleton otherwise makes
        impossible.
        """
        self.shutdown(wait=True)
        with self._lock:
            self._closed = False


#: The one queue. Import this, do not instantiate :class:`JobQueue`.
job_queue = JobQueue()


def graceful_shutdown():
    """Drain the pool at interpreter exit so no job is killed mid-write."""
    job_queue.shutdown(wait=True)


atexit.register(graceful_shutdown)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def should_bootstrap():
    """
    Whether this process should warm the models and sweep stale jobs.

    Skipped for management commands that are not serving traffic, under
    pytest, when the worker is disabled, and in the ``runserver``
    autoreloader's *parent* process (which forks and would otherwise load
    torch twice).  ``ASG_WORKER_BOOTSTRAP=False`` forces it off entirely.
    """
    if os.environ.get('ASG_WORKER_BOOTSTRAP', '') == 'False':
        return False
    if not worker_enabled():
        return False
    if 'pytest' in sys.modules or 'PYTEST_CURRENT_TEST' in os.environ:
        return False

    argv = sys.argv[1:]
    command = argv[0] if argv else ''
    if command in NON_SERVING_COMMANDS:
        return False
    if command == 'runserver' and os.environ.get('RUN_MAIN') != 'true':
        # The reloader's supervising process; the child does the real work.
        return False
    return True


def recover_interrupted_jobs():
    """
    Resolve jobs that were in flight when the process last died.

    An in-process queue cannot survive a restart: anything left ``running``
    has no thread behind it, and anything left ``queued`` has no submission
    behind it.  Both are dead ends, so both are failed with an honest message
    instead of spinning a progress bar forever.

    This assumes a single serving process, which is the deployment this worker
    is designed for.  Behind a multi-process server the sweep would have to be
    scoped by a process/heartbeat column — the point at which the Celery seam
    is the right answer instead.
    """
    stale = AnalysisResult.objects.filter(
        status__in=(STATUS_RUNNING, STATUS_QUEUED),
    )
    count = stale.update(
        status=STATUS_FAILED,
        stage='failed',
        error_message=INTERRUPTED_MESSAGE,
        end_time=timezone.now(),
    )
    if count:
        logger.warning('Marked %d interrupted analysis job(s) as failed', count)
    return count


def _bootstrap():
    """Recovery sweep then model warm-up; runs on a daemon thread at boot."""
    try:
        recover_interrupted_jobs()
    except Exception:                                 # noqa: BLE001
        logger.exception('Could not sweep interrupted analysis jobs')

    try:
        from mlcore import warmup
        # Warm-up populates the same shared model caches and runs a real
        # forward pass, so it takes the same lock a job would.
        with _INFERENCE_LOCK:
            info = warmup()
        logger.info('ML runtime warm: %s', info)
    except Exception:                                 # noqa: BLE001
        # A machine without the ML extras must still serve auth, history and
        # report downloads; only new analyses would fail, and they fail with a
        # clean message via user_safe_error().
        logger.exception('ML warm-up failed; analyses will fail until fixed')
    finally:
        _close_connections()


def start_bootstrap():
    """Spawn the boot thread. Returns it, or ``None`` when not applicable."""
    if not should_bootstrap():
        return None
    thread = threading.Thread(
        target=_bootstrap, name='asg-worker-bootstrap', daemon=True,
    )
    thread.start()
    return thread
