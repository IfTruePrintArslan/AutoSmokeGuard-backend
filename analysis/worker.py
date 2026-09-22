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

More than one serving process
-----------------------------
The queue is per *process*, but the deployment is not: ``docker/entrypoint.sh``
ships ``gunicorn --workers 2``, so two independent Python processes each run
their own pool, their own ``AppConfig.ready()`` and their own boot-time
recovery sweep against one shared database.  Everything in this module that
touches a row another process might own is therefore scoped by an ownership
stamp — ``(host, pid, boot)`` written into
``settings_snapshot['_runtime']['owner']`` atomically with the status
transition that creates the obligation.  See the "Job ownership" block below;
it is what stops one worker's restart from failing another worker's live
analysis, and what stops that analysis from later resurrecting the row it was
told it had lost.

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
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_for_futures
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.db import OperationalError, connections, transaction
from django.utils import timezone

from common.storage import analysis_artifact_dir, to_media_relative
from system_config.models import SystemSetting

from .models import (
    REPORT_FAILED,
    REPORT_GENERATING,
    REPORT_READY,
    REPORT_SKIPPED,
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
    set_report_status,
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

#: ``argv[0]`` basenames that mean "this process was launched as a Django
#: management command".  Anything else — gunicorn, uwsgi, daphne, uvicorn,
#: mod_wsgi's embedded interpreter — is a real server and always bootstraps.
MANAGEMENT_ENTRYPOINTS = frozenset({
    'manage.py', 'django-admin', 'django-admin.py',
})

#: Key inside ``settings_snapshot[RUNTIME_KEY]`` recording which process is
#: responsible for a non-terminal row.  See :func:`process_owner`.
OWNER_KEY = 'owner'

#: Default age, in seconds, past which a non-terminal job owned by a process
#: this host cannot ask about (i.e. one on a *different* host) is presumed
#: dead.  Deliberately generous — the only cost of waiting is a progress bar
#: that keeps spinning, while the cost of being wrong is killing a live
#: analysis.  Override with ``ASG['STALE_JOB_SECONDS']``.
DEFAULT_STALE_JOB_SECONDS = 6 * 60 * 60


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
# Job ownership
#
# Why a row has to know who is running it (review finding ASG-R06)
# ----------------------------------------------------------------
# The recovery sweep used to be a single unscoped statement::
#
#     AnalysisResult.objects.filter(status__in=('running', 'queued')) \
#                           .update(status='failed', ...)
#
# which is correct for exactly one deployment shape: a single serving
# process.  The shipped production default is not that shape —
# ``docker/entrypoint.sh`` runs ``gunicorn --workers 2``, so there are two
# independent Python processes, each with its own ``AppConfig.ready()``, its
# own bootstrap thread and its own thread pool.  If gunicorn's arbiter
# replaces one worker (``--timeout 300`` expiry, ``SIGHUP`` reload, OOM
# kill), the replacement booted and blanket-failed *the other worker's live
# analysis*: the user's page flipped to "Analysis failed" while the run was
# still going, and minutes later the run finished and wrote ``done`` back
# over the top, resurrecting a row the user had already been told was dead.
#
# The fix is ownership.  A row is stamped with the process that is
# responsible for it, atomically, in the same statement that moves it into
# ``queued`` or ``running``, so there is no window in which a live row has no
# owner.  The sweep then only resolves rows whose owner is demonstrably gone,
# and ``_persist`` refuses to write a result for a row it no longer owns.
#
# The stamp lives in ``settings_snapshot[RUNTIME_KEY]['owner']``.  That JSON
# column is already the documented home for per-run runtime data that the
# frozen schema has no column for (``analysis.services.RUNTIME_KEY``), and
# the detail serializer already strips underscore-prefixed keys back out — so
# this needs no migration and changes no API response.
#
# Identity is ``(host, pid, boot)``:
#
# * **host** — ``socket.gethostname()``.  A pid is only meaningful on the
#   machine that issued it, and this project supports PostgreSQL, so two
#   containers can legitimately share one database.
# * **pid** — what makes the liveness probe possible.
# * **boot** — a UUID minted per *process*, not per import.  Pids are reused:
#   after a crash a new process can be handed the dead one's pid, and without
#   ``boot`` it could not tell "my own row" from "the row of the process I
#   replaced".  It is cached against ``os.getpid()`` so a ``fork`` (gunicorn
#   ``--preload``, which this project does not use today but might) gives the
#   child a fresh identity instead of silently inheriting its parent's.
# ---------------------------------------------------------------------------

#: ``{pid: boot uuid}`` — one entry, replaced whenever the pid changes.
_boot_ids = {}
_boot_id_lock = threading.Lock()


def _boot_id():
    """A UUID identifying this OS process, stable for its whole lifetime."""
    pid = os.getpid()
    with _boot_id_lock:
        cached = _boot_ids.get(pid)
        if cached is None:
            cached = uuid.uuid4().hex
            # A fork makes the parent's entry meaningless in the child.
            _boot_ids.clear()
            _boot_ids[pid] = cached
        return cached


def process_owner():
    """This process's ownership stamp, as a JSON-serialisable dict."""
    return {
        'host': socket.gethostname(),
        'pid': os.getpid(),
        'boot': _boot_id(),
        'at': timezone.now().isoformat(),
    }


def owner_of(snapshot):
    """The owner dict recorded in ``snapshot``, or ``None``."""
    runtime = (snapshot or {}).get(RUNTIME_KEY)
    if not isinstance(runtime, dict):
        return None
    owner = runtime.get(OWNER_KEY)
    return owner if isinstance(owner, dict) else None


def snapshot_with_owner(snapshot):
    """
    Return a copy of ``snapshot`` carrying this process's ownership stamp.

    Copies rather than mutates: the caller's dict is usually the one attached
    to a live model instance, and a JSONField that is mutated in place does
    not reliably round-trip through ``update_fields``.
    """
    merged = dict(snapshot or {})
    runtime = dict(merged.get(RUNTIME_KEY) or {})
    runtime[OWNER_KEY] = process_owner()
    merged[RUNTIME_KEY] = runtime
    return merged


def snapshot_without_owner(snapshot):
    """Return a copy of ``snapshot`` with the ownership stamp removed."""
    merged = dict(snapshot or {})
    runtime = merged.get(RUNTIME_KEY)
    if isinstance(runtime, dict) and OWNER_KEY in runtime:
        runtime = dict(runtime)
        runtime.pop(OWNER_KEY, None)
        merged[RUNTIME_KEY] = runtime
    return merged


def is_owned_by_this_process(snapshot):
    """True when ``snapshot``'s owner stamp is this exact process."""
    owner = owner_of(snapshot)
    return bool(owner) and owner.get('boot') == _boot_id()


def process_is_alive(pid):
    """
    ``True`` / ``False`` / ``None`` (unknowable) for "is ``pid`` running?".

    POSIX uses ``os.kill(pid, 0)``, the standard side-effect-free liveness
    probe: it performs the permission checks and then does nothing.

    **Never on Windows.**  CPython's ``os.kill`` on Windows maps every signal
    other than ``CTRL_C_EVENT``/``CTRL_BREAK_EVENT`` onto
    ``TerminateProcess(handle, sig)`` — so ``os.kill(pid, 0)`` there does not
    ask whether the process is alive, it *kills* it.  Using the POSIX idiom
    unguarded would turn this recovery sweep into a tool that murders the
    sibling worker it was trying to protect.  There is no dependency-free
    Windows equivalent, so this returns ``None`` and
    :func:`_recovery_verdict` decides what to do with "unknown".
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if pid == os.getpid():
        return True
    if os.name != 'posix':
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists; it just belongs to another user.
        return True
    except OSError:                                   # pragma: no cover
        return None
    return True


def stale_job_seconds():
    """How long a job owned by an unreachable host may sit before it is failed."""
    configured = getattr(settings, 'ASG', {}).get(
        'STALE_JOB_SECONDS', DEFAULT_STALE_JOB_SECONDS,
    )
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = DEFAULT_STALE_JOB_SECONDS
    return max(60, configured)


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
        persisted = _execute(analysis)
    except Exception as exc:                          # noqa: BLE001 - by design
        _mark_failed(analysis_id, exc)
        return None

    if not persisted:
        # The row belongs to somebody else now (see _persist's ownership
        # gate). Generating a PDF from a result we were not allowed to store
        # would attach a report to a row whose contents say something else.
        return None

    _maybe_generate_report(analysis)
    return analysis


def _claim(analysis_id):
    """
    Atomically move a ``queued`` row to ``running``, or decline the job.

    The conditional ``UPDATE`` is the whole concurrency story: whichever
    thread's statement matches the ``status='queued'`` predicate wins, and
    every other caller sees ``0`` rows affected.

    The ownership stamp goes into that *same* statement (review finding
    ASG-R06).  Writing it afterwards would leave a window — however short —
    in which a genuinely running row looks unowned, and an unowned
    non-terminal row is exactly what :func:`recover_interrupted_jobs` treats
    as an orphan.  Reading the snapshot first costs one extra ``SELECT`` per
    job, which is nothing next to the run it precedes.
    """
    row = retry_on_lock(
        lambda: AnalysisResult.objects.filter(pk=analysis_id)
        .values('settings_snapshot').first(),
        f'reading the snapshot of analysis {analysis_id}',
    )
    if row is None:
        logger.warning('Analysis %s does not exist; nothing to claim.',
                       analysis_id)
        return None
    owned_snapshot = snapshot_with_owner(row['settings_snapshot'])

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
            settings_snapshot=owned_snapshot,
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
    """
    Run the pipeline and persist everything it produced.

    Returns ``True`` when the result was written, ``False`` when
    :func:`_persist` declined it because the row is no longer this process's
    to write (review finding ASG-R06).
    """
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

    if not _persist(analysis, result or {}, output_dir):
        # _persist already logged, at ERROR, exactly why it declined.  Return
        # instead of raising: raising would route into _mark_failed, which
        # would stamp this row a second time — and the whole point of
        # declining was that this process no longer has any business writing
        # to it.
        logger.warning(
            'Analysis %s ran to completion in %.2fs but its result was '
            'discarded; the row is no longer owned by this worker.',
            analysis.analysis_id, elapsed,
        )
        return False

    logger.info('Analysis %s finished in %.2fs (%s vehicles, %s smoke)',
                analysis.analysis_id, elapsed, analysis.total_vehicles,
                analysis.total_smoke)
    return True


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
    # Replacing _runtime wholesale intentionally drops the ownership stamp
    # (review finding ASG-R06): the row is about to become terminal, and a
    # terminal row has no owner — nothing is running it, and leaving a stale
    # (host, pid) behind would only invite a future sweep to reason about a
    # process that has nothing to do with it.
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
            # -- Ownership gate (review finding ASG-R06) -------------------
            #
            # Between the claim and here, minutes of inference have gone by.
            # Another process's boot sweep may have decided this row was
            # abandoned and failed it; an administrator may have deleted it.
            # Writing ``done`` unconditionally at that point *resurrects* a
            # row the user has already been told is dead — the second half of
            # the gunicorn multi-worker bug, and the half that makes it
            # silent rather than merely wrong.
            #
            # Who wins: the worker that did the work, but only while it still
            # owns the row.  Ownership is the tie-breaker rather than "last
            # writer wins" because ownership is the thing that was true when
            # the work started and the thing whose loss means somebody else
            # has already acted on this row.  Losing it is not recoverable
            # from here, so the result is discarded and the loss is logged at
            # ERROR — the artefacts stay on disk for forensics rather than
            # being silently reconciled one way or the other.
            current = (
                AnalysisResult.objects
                .filter(pk=analysis.analysis_id)
                .values('status', 'settings_snapshot')
                .first()
            )
            if current is None:
                logger.error(
                    'Analysis %s no longer exists; discarding the result of '
                    'a run that had already completed.',
                    analysis.analysis_id,
                )
                return False
            if current['status'] != STATUS_RUNNING:
                logger.error(
                    'Refusing to save the result of analysis %s: the row is '
                    'now status=%r, not %r. Another process resolved it '
                    'while this run was in flight; the result is being '
                    'discarded rather than resurrecting the row.',
                    analysis.analysis_id, current['status'], STATUS_RUNNING,
                )
                return False
            if not is_owned_by_this_process(current['settings_snapshot']):
                logger.error(
                    'Refusing to save the result of analysis %s: it is now '
                    'owned by %r, not by this process (pid %s). The result '
                    'is being discarded rather than overwriting whatever the '
                    'current owner is doing.',
                    analysis.analysis_id,
                    owner_of(current['settings_snapshot']), os.getpid(),
                )
                return False

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
            return True

    return bool(retry_on_lock(write, f'saving analysis {analysis.analysis_id}'))


def _maybe_generate_report(analysis):
    """
    Produce the PDF when the frozen configuration asked for one.

    Called *after* the analysis is already marked ``done``, not before: the
    measurement is the product, the PDF is a rendering of it.  A reports app
    that is missing, broken or mid-rewrite must never turn a successful
    analysis into a failed one, so every exception is swallowed after being
    logged in full.

    Swallowed, but no longer *silent*.  Each branch stamps
    ``AnalysisResult.report_status`` on the way through — ``skipped`` when the
    frozen configuration never asked for a PDF, ``generating`` while one is
    being made, then ``ready`` or ``failed``.  Without that, "the report is
    still being written" and "the report died four seconds ago" are the same
    observation from outside (a ``done`` analysis with no ``report``), and the
    UI could only tell them apart by waiting out the server's whole 30-second
    budget on the off-chance.

    The ``failed`` stamp is deliberately written here as well as inside
    ``reports.services.generate_report_for_analysis``.  The seam cannot record
    what it never reached: the import above is itself inside the ``try``, and
    an unimportable (or stubbed, or half-rewritten) reports app is exactly the
    kind of breakage this hook was built to absorb.  One state this row must
    never be abandoned in is ``generating``.
    """
    if not (analysis.settings_snapshot or {}).get('auto_generate_pdf'):
        set_report_status(analysis, REPORT_SKIPPED)
        return

    set_report_status(analysis, REPORT_GENERATING)
    try:
        from reports.services import generate_report_for_analysis
        generate_report_for_analysis(analysis)
    except Exception:                                 # noqa: BLE001 - by design
        logger.exception('report generation failed; analysis %s still succeeds',
                         analysis.analysis_id)
        set_report_status(analysis, REPORT_FAILED)
    else:
        set_report_status(analysis, REPORT_READY)


def _mark_failed(analysis_id, exc):
    """
    Record a failure: full detail to the log, one safe sentence to the row.

    Scoped to ``status='running'`` (review finding ASG-R06).  A row that
    another process has already resolved must not have its ``error_message``
    rewritten by a worker that has lost it — the caller sees whatever the
    current owner decided, and the real cause is in the log either way.

    ``report_status`` is resolved in the same statement.  A run that failed
    never reaches :func:`_maybe_generate_report`, so the column would
    otherwise be abandoned at its ``pending`` default — a row promising a PDF
    that nothing will ever render, and which the report endpoint would refuse
    to make on request (it requires ``status='done'``).  ``skipped`` is the
    honest reading: no report applies to this run.
    """
    logger.exception('Analysis %s failed', analysis_id)
    try:
        retry_on_lock(
            lambda: AnalysisResult.objects.filter(
                pk=analysis_id, status=STATUS_RUNNING,
            ).update(
                status=STATUS_FAILED,
                stage='failed',
                error_message=user_safe_error(exc),
                end_time=timezone.now(),
                report_status=REPORT_SKIPPED,
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

        The row is stamped with this process's ownership in the same
        statement that queues it (review finding ASG-R06).  A ``queued`` row
        lives in *this* process's ``ThreadPoolExecutor`` and nowhere else, so
        a sibling gunicorn worker rebooting must be able to see that the job
        belongs to someone else and leave it alone — and must equally be able
        to see, when this process dies, that nothing is left to run it.
        """
        analysis_id = str(analysis_id)
        row = retry_on_lock(
            lambda: AnalysisResult.objects.filter(pk=analysis_id)
            .values('settings_snapshot').first(),
            f'reading the snapshot of analysis {analysis_id}',
        )
        if row is None:
            logger.info('Analysis %s does not exist; not enqueued.',
                        analysis_id)
            return False
        owned_snapshot = snapshot_with_owner(row['settings_snapshot'])

        accepted = retry_on_lock(
            lambda: AnalysisResult.objects.filter(
                pk=analysis_id, status__in=(STATUS_PENDING, STATUS_QUEUED),
            ).update(status=STATUS_QUEUED, progress=0, stage='queued',
                     error_message='', settings_snapshot=owned_snapshot),
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

def is_serving_context(argv=None, environ=None):
    """
    Whether ``argv``/``environ`` describe a process that will serve traffic.

    Pure and injectable so every deployment shape this project supports can be
    asserted in a unit test rather than discovered in production — which is
    exactly how the bug below escaped.

    Why not ``RUN_MAIN`` alone (robustness finding §D)
    --------------------------------------------------
    The previous rule was ``command == 'runserver' and RUN_MAIN != 'true' ->
    don't bootstrap``.  ``RUN_MAIN`` is set by Django's autoreloader when it
    re-execs itself, so under plain ``runserver`` it correctly picks the child
    out of the parent/child pair.  But ``runserver --noreload`` never starts
    an autoreloader at all, so ``RUN_MAIN`` is *never set* — and the rule then
    returned ``False`` unconditionally, for the entire life of the process.
    Both the crash-recovery sweep and the ML warm-up were silently skipped, so
    a job killed mid-flight stayed ``running`` forever and the UI spun with
    no process behind it.

    The reloader is therefore detected from ``argv`` (where the decision
    actually lives) and ``RUN_MAIN`` is consulted only when there really is a
    parent/child pair to choose between:

    ==========================================  ==========================
    Invocation                                  Result
    ==========================================  ==========================
    ``manage.py runserver`` (parent)            ``False`` — child does it
    ``manage.py runserver`` (child, RUN_MAIN)   ``True``
    ``manage.py runserver --noreload``          ``True``  <- was ``False``
    ``gunicorn config.wsgi:application``        ``True``
    ``uwsgi --module config.wsgi``              ``True``
    ``manage.py migrate`` / ``test`` / ...      ``False``
    ==========================================  ==========================
    """
    argv = list(sys.argv if argv is None else argv)
    environ = os.environ if environ is None else environ

    entrypoint = os.path.basename(argv[0]) if argv and argv[0] else ''
    args = argv[1:]
    command = args[0] if args else ''

    if entrypoint not in MANAGEMENT_ENTRYPOINTS:
        # gunicorn, uWSGI, daphne, uvicorn, mod_wsgi, or any other WSGI/ASGI
        # server importing config.wsgi directly.  One process per worker, no
        # autoreloader, nothing to de-duplicate against: always bootstrap.
        return True

    if command in NON_SERVING_COMMANDS:
        return False

    if command == 'runserver':
        if '--noreload' in args:
            # Single process, no re-exec, no RUN_MAIN — this *is* the server.
            return True
        return environ.get('RUN_MAIN') == 'true'

    # Some other management command.  Historically these bootstrapped (the
    # project ships custom serving-adjacent commands), so the permissive
    # default is kept rather than changed as a side effect of this fix.
    return True


def should_bootstrap():
    """
    Whether this process should warm the models and sweep stale jobs.

    Skipped when the worker is disabled, under pytest, and for anything
    :func:`is_serving_context` does not recognise as a serving process.
    ``ASG_WORKER_BOOTSTRAP=False`` forces it off entirely.
    """
    if os.environ.get('ASG_WORKER_BOOTSTRAP', '') == 'False':
        return False
    if not worker_enabled():
        return False
    if 'pytest' in sys.modules or 'PYTEST_CURRENT_TEST' in os.environ:
        return False
    return is_serving_context()


def _recovery_verdict(row, stale_cutoff):
    """
    Decide whether the boot sweep may fail ``row``.  ``(bool, reason)``.

    The default is **no**: a live analysis that is wrongly failed is a
    user-visible data-integrity bug, while a dead one that survives a sweep
    is only a progress bar that keeps spinning until the next restart.  Every
    "yes" below therefore needs positive evidence that nothing is behind the
    row.
    """
    owner = owner_of(row['settings_snapshot'])

    if not owner:
        # No process ever took responsibility for it.  Both transitions into
        # a non-terminal status (``enqueue`` -> queued, ``_claim`` ->
        # running) write the stamp in the *same* statement as the status, so
        # there is no window where a live row looks like this: an unowned
        # row is a row from before this fix shipped, or one a fixture or an
        # administrator created by hand.  Either way nothing is running it.
        return True, 'no process ever claimed it'

    host = owner.get('host')
    pid = owner.get('pid')

    if owner.get('boot') == _boot_id():
        # This very process owns it.  Cannot happen during the boot sweep
        # (we have not claimed anything yet); if it ever does, the job is
        # live and ours.
        return False, 'this process owns it'

    if host != socket.gethostname():
        # Another machine's pid means nothing here, and it may well be
        # running the job right now (PostgreSQL deployments share one
        # database across containers).  Only age can settle it.
        started = row.get('start_time') or row.get('created_at')
        if started is not None and started < stale_cutoff:
            return True, f'owned by host {host!r} and untouched since {started}'
        return False, f'owned by another host ({host!r})'

    alive = process_is_alive(pid)
    if alive is True:
        return False, f'its owner (pid {pid}) is still running on this host'
    if alive is False:
        return True, f'its owning process (pid {pid}) is gone'

    # Liveness is unknowable: a non-POSIX host (see process_is_alive — we
    # must not use os.kill on Windows).  Windows has no multi-process
    # serving story for this project (gunicorn is POSIX-only; the supported
    # shape there is a single `manage.py runserver`), so a *different* pid on
    # this host is necessarily a process that has already exited.  Revisit
    # this line if a multi-process Windows server is ever supported.
    return True, (f'pid {pid} is not this process and liveness cannot be '
                  f'probed on {os.name!r}')


def recover_interrupted_jobs():
    """
    Resolve jobs that were in flight when *their own* process died.

    An in-process queue does not survive a restart: a row left ``running``
    has no thread behind it and a row left ``queued`` has no submission
    behind it, so both would otherwise spin a progress bar forever.  They are
    failed with an honest message instead.

    Scoped by ownership, not blanket (review finding ASG-R06)
    ---------------------------------------------------------
    This used to be one unscoped ``UPDATE`` over every ``running``/``queued``
    row, on the assumption — stated in this docstring — of a single serving
    process.  The shipped production default is ``gunicorn --workers 2``
    (``docker/entrypoint.sh``), which is two processes, so the assumption was
    false: when the arbiter replaced one worker, the replacement's boot sweep
    failed the *other* worker's live analysis, and that analysis then wrote
    ``done`` back over the top when it finished.

    Now every non-terminal row carries the ``(host, pid, boot)`` of the
    process responsible for it (see :func:`process_owner`), stamped
    atomically with the status transition, and this sweep only resolves a row
    when it can show that process is gone.  A row owned by a live sibling is
    left strictly alone.  :func:`_recovery_verdict` holds the rules and the
    reasoning for each.

    Returns the number of rows failed.
    """
    now = timezone.now()
    stale_cutoff = now - timedelta(seconds=stale_job_seconds())

    candidates = list(
        AnalysisResult.objects
        .filter(status__in=(STATUS_RUNNING, STATUS_QUEUED))
        .values('analysis_id', 'status', 'settings_snapshot', 'start_time',
                'created_at')
    )

    failed = 0
    left_alone = 0
    for row in candidates:
        resolve, reason = _recovery_verdict(row, stale_cutoff)
        if not resolve:
            left_alone += 1
            logger.info('Recovery sweep left analysis %s (%s) alone: %s',
                        row['analysis_id'], row['status'], reason)
            continue

        # Re-assert the status predicate: between the SELECT above and this
        # UPDATE the owning process may have finished the job itself.
        failed += retry_on_lock(
            lambda pk=row['analysis_id']: AnalysisResult.objects.filter(
                pk=pk, status__in=(STATUS_RUNNING, STATUS_QUEUED),
            ).update(
                status=STATUS_FAILED,
                stage='failed',
                error_message=INTERRUPTED_MESSAGE,
                end_time=now,
                # Same reasoning as _mark_failed: an interrupted run is never
                # getting a PDF, so the column must not be left promising one.
                report_status=REPORT_SKIPPED,
            ),
            f'failing interrupted analysis {row["analysis_id"]}',
        )
        logger.info('Recovery sweep failed analysis %s (%s): %s',
                    row['analysis_id'], row['status'], reason)

    if failed:
        logger.warning('Marked %d interrupted analysis job(s) as failed',
                       failed)
    if left_alone:
        logger.info('Recovery sweep left %d in-flight analysis job(s) owned '
                    'by other live processes untouched', left_alone)
    return failed


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


#: Set the first time :func:`start_bootstrap` actually starts the thread.
#:
#: ``AppConfig.ready()`` is not contractually once-per-process: calling
#: ``django.setup()`` again, re-populating the app registry, or importing the
#: project from a script that has already been set up will all fire it a
#: second time.  Two bootstraps would mean two ``recover_interrupted_jobs()``
#: sweeps (the second one able to fail a job the first one's warm-up just let
#: through) and two concurrent torch loads.  The flag is process-local, which
#: is the right scope: under the autoreloader the parent and child are
#: separate processes and are separated by ``RUN_MAIN`` in
#: :func:`is_serving_context` instead, and a reload re-execs the child, which
#: *should* sweep again.
_bootstrap_started = False
_bootstrap_lock = threading.Lock()


def start_bootstrap():
    """
    Spawn the boot thread. Returns it, or ``None`` when not applicable.

    Idempotent: only the first call in a process starts anything.
    """
    global _bootstrap_started

    if not should_bootstrap():
        return None

    with _bootstrap_lock:
        if _bootstrap_started:
            logger.debug('Analysis worker bootstrap already ran in this '
                         'process; skipping')
            return None
        _bootstrap_started = True

    thread = threading.Thread(
        target=_bootstrap, name='asg-worker-bootstrap', daemon=True,
    )
    thread.start()
    return thread


def reset_bootstrap_state():
    """Forget that bootstrap ran — test support only, never called at runtime."""
    global _bootstrap_started

    with _bootstrap_lock:
        _bootstrap_started = False
