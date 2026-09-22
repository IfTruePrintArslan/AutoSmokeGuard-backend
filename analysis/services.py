"""
Business logic that sits between the analysis HTTP layer and the ML core.

Everything in here is deliberately framework-light: views call it, the worker
thread calls it, and a management command could call it.  Keeping the rules in
one module means "what does the system actually do when you press Analyse?"
has a single answer instead of being smeared across a view, a serializer and a
thread.

Four responsibilities live here:

1. **Freezing configuration** (:func:`build_settings_snapshot`).  An admin can
   retune thresholds at any time (UC-09); a run must stay explainable, so the
   effective configuration is copied onto the row before any work starts.
   Client-supplied overrides are *clamped*, never trusted — a browser sending
   ``confidence_threshold: 99`` gets 0.95, not a 400 and not a broken run.

2. **Translating that snapshot into an** :class:`mlcore.MLConfig`.  The two
   vocabularies differ on purpose: the product speaks
   ``confidence_threshold`` / ``smoke_mask_threshold`` (what an inspector
   understands), the ML package speaks ``conf_threshold`` / ``mask_threshold``.
   ``MLConfig.from_dict`` silently drops keys it does not know, so handing it
   the raw snapshot would quietly run every analysis at the library defaults.
   :func:`build_ml_config` does the mapping explicitly for exactly that reason.

3. **Admission control** (:func:`start_analysis`).  One in-flight run per
   upload, ownership enforced by the caller, row created *then* enqueued.

4. **Dashboard aggregation** (:func:`dashboard_stats`).  A handful of grouped
   queries plus in-Python zero-filling — never a loop that issues one query
   per day.

``mlcore`` is imported lazily inside functions throughout.  Importing torch at
module scope would add seconds to ``manage.py`` startup, would break
``manage.py check`` on a machine without the ML extras installed, and would
make the test suite impossible to monkeypatch.
"""
import logging
import random
import time
from datetime import timedelta

from django.conf import settings
from django.db import OperationalError, transaction
from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate
from django.utils import timezone

from system_config.models import SystemSetting

from .models import (
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MODERATE,
    STATUS_PENDING,
    STATUS_QUEUED,
    STATUS_RUNNING,
    AnalysisResult,
)

logger = logging.getLogger('asg.analysis')

# ---------------------------------------------------------------------------
# Clamping ranges
#
# These are the *hard* bounds the server will accept from a client, and they
# are intentionally narrower than what the models technically tolerate.  A
# confidence threshold of 0.0 turns the detector into a noise generator and a
# frame sample rate of 0 is a division by zero three layers down; neither is a
# useful thing for a user to ask for, so both are silently corrected.
# ---------------------------------------------------------------------------

CONFIDENCE_MIN = 0.05
CONFIDENCE_MAX = 0.95
FRAME_SAMPLE_RATE_MIN = 1
FRAME_SAMPLE_RATE_MAX = 60
SENSITIVITY_MIN = 0
SENSITIVITY_MAX = 100
MASK_THRESHOLD_MIN = 0.05
MASK_THRESHOLD_MAX = 0.95

#: How the 0-100 "Smoke sensitivity" slider maps onto the segmenter's mask
#: threshold: ``0.25`` at sensitivity 0 ("Low"), rising to ``0.75`` at 100
#: ("Strict").  Ascending, because a higher threshold keeps only the pixels
#: the model is most confident about and is therefore the *stricter* run —
#: see :func:`build_settings_snapshot` and review finding F21.  The whole
#: range sits inside ``[MASK_THRESHOLD_MIN, MASK_THRESHOLD_MAX]``, so no
#: valid slider position is ever clamped.
MASK_THRESHOLD_AT_ZERO_SENSITIVITY = 0.25
MASK_THRESHOLD_SENSITIVITY_SPAN = 0.50

#: Statuses that mean "this upload is already being worked on".
ACTIVE_STATUSES = (STATUS_PENDING, STATUS_QUEUED, STATUS_RUNNING)

#: Severity ordering, weakest first — used to pick an overall verdict.
SEVERITY_ORDER = (SEVERITY_LOW, SEVERITY_MODERATE, SEVERITY_HIGH)

#: Product palette for the dashboard donut. Kept here (not in the front end)
#: so the PDF report and the web UI cannot drift apart.
SEVERITY_PALETTE = {
    SEVERITY_LOW: '#4ade80',
    SEVERITY_MODERATE: '#fbbf24',
    SEVERITY_HIGH: '#f87171',
}

SEVERITY_LABELS = {
    SEVERITY_LOW: 'Low',
    SEVERITY_MODERATE: 'Moderate',
    SEVERITY_HIGH: 'High',
}

#: Number of points in every dashboard sparkline.
SPARKLINE_POINTS = 14

#: Bounds on ``?days=`` so a client cannot ask us to group ten years by day.
DASHBOARD_DAYS_MIN = 1
DASHBOARD_DAYS_MAX = 365
DASHBOARD_DAYS_DEFAULT = 30

#: How many rows the dashboard's "recent" strip shows.
RECENT_ANALYSES_LIMIT = 5

#: Reserved key inside ``settings_snapshot`` holding *outputs* of the run that
#: the frozen schema has no column for (annotated frame list, device,
#: segmenter mode).  Stripped back out by the detail serializer so the
#: ``settings_snapshot`` the API exposes stays pure configuration.
RUNTIME_KEY = '_runtime'


#: How many times a statement that lost the race for SQLite's write lock is
#: retried, and the initial back-off in seconds (doubling, with jitter).
DB_RETRY_ATTEMPTS = 6
DB_RETRY_BACKOFF = 0.05


class ReportsUnavailable(RuntimeError):
    """The reports app could not be imported or is not wired up yet."""


class AnalysisInFlight(RuntimeError):
    """
    Admission refused: this upload already has a non-terminal run.

    Raised from inside :func:`start_analysis`'s transaction, which is the
    only place the answer can be trusted — the pre-flight
    :func:`active_analysis_for` the view does is a courtesy, not a guarantee
    (review finding F20b).  Carries the offending row so the view can put its
    id in the 409 body without going back to the database.
    """

    def __init__(self, analysis):
        self.analysis = analysis
        super().__init__(
            f'Media {analysis.media_id} already has analysis '
            f'{analysis.analysis_id} in flight (status={analysis.status!r}).'
        )


# ---------------------------------------------------------------------------
# Database resilience
#
# This app is the only part of the system where a *background thread* and a
# *web request* write to the same tables at the same time, and the default
# deployment is a single SQLite file.  SQLite serialises writers across the
# whole database, so "POST /api/analyze inserts a row" and "worker thread
# saves a result" genuinely collide.  The sqlite3 driver's five-second busy
# timeout absorbs most of that, but not all of it — and not at all for the
# ``SQLITE_LOCKED`` a shared-cache connection raises, which is exactly what
# Django's in-memory test database uses.
#
# Losing a user's analysis to a 40-millisecond lock would be indefensible, so
# the handful of statements that can race are retried with jittered back-off.
# Only lock errors are retried; every other database error propagates
# untouched, so this can never mask a real bug by silently repeating it.
# ---------------------------------------------------------------------------

def is_lock_error(exc):
    """True for the "database/table is locked" family of SQLite errors."""
    return 'lock' in str(exc).lower()


def retry_on_lock(operation, description, attempts=DB_RETRY_ATTEMPTS):
    """Run ``operation``, retrying while the database write lock is held."""
    delay = DB_RETRY_BACKOFF
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except OperationalError as exc:
            if attempt == attempts or not is_lock_error(exc):
                raise
            wait = delay + random.uniform(0, delay)
            logger.warning('%s lost a race for the database lock '
                           '(attempt %d/%d); retrying in %.0f ms',
                           description, attempt, attempts, wait * 1000)
            time.sleep(wait)
            delay *= 2
    return None                                       # pragma: no cover


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def clamp_float(value, low, high, default=None):
    """Coerce ``value`` to a float inside ``[low, high]``; ``default`` if junk."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:            # NaN — the only value that fails self-equality.
        return default
    return min(max(number, low), high)


def clamp_int(value, low, high, default=None):
    """Coerce ``value`` to an int inside ``[low, high]``; ``default`` if junk."""
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    return min(max(number, low), high)


def worst_severity(counts):
    """
    The strongest severity present in a ``{severity: count}`` histogram.

    Returns ``''`` (the model's "no verdict" value) when nothing was detected,
    which is what a clean image should read as — not ``"low"``, which would
    imply we found something mild.
    """
    for level in reversed(SEVERITY_ORDER):
        if int(counts.get(level) or 0) > 0:
            return level
    return ''


# ---------------------------------------------------------------------------
# Configuration freezing
# ---------------------------------------------------------------------------

def _resolve_model_weights(name):
    """
    Turn a client-supplied ``model`` name into a weights path, or ``None``.

    Security: the client picks a *file name*, never a path.  The candidate is
    reduced to its basename, must end in ``.pt``, and must already exist
    directly inside ``ASG['ML_ASSETS_DIR']``.  Anything else — an absolute
    path, a traversal, a name we do not ship — is ignored and the run uses the
    configured default.  This is the only place a user's input is allowed
    anywhere near the filesystem in this app.
    """
    if not name:
        return None

    from pathlib import Path

    candidate = Path(str(name)).name
    if not candidate.endswith('.pt'):
        candidate = f'{candidate}.pt'

    assets_dir = Path(settings.ASG['ML_ASSETS_DIR'])
    weights = assets_dir / candidate
    try:
        weights.relative_to(assets_dir)
    except ValueError:                                   # pragma: no cover
        return None
    if not weights.is_file():
        logger.info('Ignoring unknown model %r; using the configured default.',
                    name)
        return None
    return str(weights)


def build_settings_snapshot(setting=None, overrides=None):
    """
    Freeze the effective configuration for one run.

    Starts from :meth:`SystemSetting.as_snapshot`, layers the (already
    validated, still untrusted) client ``overrides`` on top, and clamps
    everything into a range the pipeline can actually survive.

    ``sensitivity`` deserves a note.  The API contract exposes a 0-100
    "Smoke sensitivity" slider but no client-settable
    ``smoke_mask_threshold`` — :class:`AnalysisSettingsSerializer` has no
    such field and would silently drop one — so the server derives the
    threshold the pipeline actually understands from the raw slider value.
    :func:`build_settings_snapshot` is the *only* place that derivation
    happens; keep it that way, or the two ends of the control can drift
    apart again.

    The two ascend **together**: a higher ``sensitivity`` means a *stricter*
    run.  ``smoke_mask_threshold`` binarises the segmenter's probability
    map, so raising it keeps only the pixels the model is most confident
    about — the same direction as the ``confidence_threshold`` the client
    sends alongside it.  The slider's right-hand "Strict" end is therefore
    the strictest configuration the API allows.

    This was inverted until review finding F21: the mapping was
    ``smoke_mask_threshold = 1 - s/100`` and this docstring asserted the two
    values "are inverses", which made the end of the slider *labelled*
    ``Strict`` the loosest, highest-false-positive setting on offer — on a
    run whose PDF is explicitly framed as enforcement evidence.  An
    inspector dragging toward "Strict" was getting the noisiest possible
    analysis.  The client half (ascending ``confidence_threshold``
    0.15 -> 0.75, raw un-inverted ``sensitivity``) has already landed; this
    is the server half.

    The mapping is now ``0.25 + (s/100) * 0.50`` — 0.25 at Low, 0.50 at
    Balanced, 0.75 at Strict — which sits wholly inside
    ``[MASK_THRESHOLD_MIN, MASK_THRESHOLD_MAX]``, so a valid slider position
    is never silently corrected by the clamp.  The clamp stays anyway, for
    the same reason every other one here does: ``sensitivity`` is untrusted
    input and the band is the pipeline's survivable range, not a formatting
    detail.  Both values are recorded in the snapshot so a report can
    explain the run either way round.
    """
    setting = setting or SystemSetting.get_solo()
    overrides = overrides or {}

    snapshot = dict(setting.as_snapshot())
    snapshot['auto_generate_pdf'] = bool(setting.auto_generate_pdf)

    if 'confidence_threshold' in overrides:
        snapshot['confidence_threshold'] = clamp_float(
            overrides['confidence_threshold'], CONFIDENCE_MIN, CONFIDENCE_MAX,
            default=snapshot['confidence_threshold'],
        )

    if 'frame_sample_rate' in overrides:
        snapshot['frame_sample_rate'] = clamp_int(
            overrides['frame_sample_rate'],
            FRAME_SAMPLE_RATE_MIN, FRAME_SAMPLE_RATE_MAX,
            default=snapshot['frame_sample_rate'],
        )

    if 'sensitivity' in overrides:
        sensitivity = clamp_int(overrides['sensitivity'],
                                SENSITIVITY_MIN, SENSITIVITY_MAX)
        if sensitivity is not None:
            snapshot['sensitivity'] = sensitivity
            # Ascending with sensitivity: higher slider = stricter run.
            # See the docstring (review finding F21) before touching the
            # direction of this line -- it is half of a control whose other
            # half lives in the front end.
            snapshot['smoke_mask_threshold'] = clamp_float(
                MASK_THRESHOLD_AT_ZERO_SENSITIVITY
                + (sensitivity / 100.0) * MASK_THRESHOLD_SENSITIVITY_SPAN,
                MASK_THRESHOLD_MIN, MASK_THRESHOLD_MAX,
                default=snapshot['smoke_mask_threshold'],
            )

    if 'auto_generate_pdf' in overrides:
        snapshot['auto_generate_pdf'] = bool(overrides['auto_generate_pdf'])

    weights = _resolve_model_weights(overrides.get('model'))
    if weights:
        snapshot['model'] = str(overrides['model'])
        snapshot['yolo_weights'] = weights

    # Belt and braces: clamp the server-side values too.  An admin can write
    # anything into SystemSetting through the Django admin, and a bad row must
    # not be able to wedge the worker.
    snapshot['confidence_threshold'] = clamp_float(
        snapshot.get('confidence_threshold'), CONFIDENCE_MIN, CONFIDENCE_MAX,
        default=0.35,
    )
    snapshot['smoke_mask_threshold'] = clamp_float(
        snapshot.get('smoke_mask_threshold'),
        MASK_THRESHOLD_MIN, MASK_THRESHOLD_MAX, default=0.5,
    )
    snapshot['frame_sample_rate'] = clamp_int(
        snapshot.get('frame_sample_rate'),
        FRAME_SAMPLE_RATE_MIN, FRAME_SAMPLE_RATE_MAX, default=5,
    )
    return snapshot


def build_ml_config(snapshot):
    """
    Translate a frozen ``settings_snapshot`` into an :class:`mlcore.MLConfig`.

    The mapping is written out by hand rather than passed through
    ``MLConfig.from_dict(snapshot)`` because the two key sets only partially
    overlap; ``from_dict`` ignores unknown keys, so the shorthand would throw
    away the confidence and mask thresholds without a word.
    """
    from mlcore import MLConfig

    snapshot = snapshot or {}
    kwargs = {
        'yolo_weights': snapshot.get('yolo_weights')
                        or str(settings.ASG['YOLO_WEIGHTS']),
        'segmenter_weights': str(settings.ASG['SEGMENTER_WEIGHTS']),
        'conf_threshold': snapshot.get('confidence_threshold', 0.35),
        'mask_threshold': snapshot.get('smoke_mask_threshold', 0.5),
        'frame_sample_rate': snapshot.get('frame_sample_rate', 5),
        'max_video_seconds': snapshot.get('max_video_seconds', 300),
        'severity_low_max': snapshot.get('severity_low_max', 0.33),
        'severity_moderate_max': snapshot.get('severity_moderate_max', 0.66),
    }
    return MLConfig.from_dict({k: v for k, v in kwargs.items() if v is not None})


# ---------------------------------------------------------------------------
# Admission control
# ---------------------------------------------------------------------------

def active_analysis_for(media):
    """The queued/running/pending analysis for ``media``, or ``None``."""
    return retry_on_lock(
        lambda: AnalysisResult.objects
        .filter(media=media, status__in=ACTIVE_STATUSES)
        .order_by('-created_at')
        .first(),
        'checking for an analysis already in flight',
    )


def _serialise_admissions_for(media):
    """
    Make "is a run already in flight for this upload?" a question only one
    caller at a time can be answering (review finding F20b).

    Must be called as the first statement inside the ``atomic()`` block that
    then does the check and the insert.  What provides the mutual exclusion
    differs by backend, and both are real database guarantees rather than a
    narrower check window:

    **PostgreSQL** — ``SELECT ... FOR UPDATE`` on the parent ``UploadedMedia``
    row.  The second request blocks on the row lock until the first commits.
    Django runs at the backend default isolation level, ``READ COMMITTED``,
    under which every statement takes a fresh snapshot; so once the loser is
    unblocked its *next* statement — the in-flight check below — sees the
    winner's committed ``AnalysisResult`` and refuses.  (A deployment that
    forces ``REPEATABLE READ`` would keep the pre-lock snapshot and this
    would degrade to the original race; nothing here sets that, and the
    project's ``DATABASES`` config does not expose it.)

    **SQLite** — there is no row locking, and ``select_for_update()`` raises
    ``NotSupportedError`` rather than silently doing nothing, so it is
    skipped.  It is also unnecessary: ``config.settings`` pins
    ``OPTIONS['transaction_mode'] = 'IMMEDIATE'``, so *entering* the
    enclosing ``atomic()`` issues ``BEGIN IMMEDIATE`` and takes SQLite's
    database-wide write lock up front.  The second request therefore blocks
    at ``BEGIN`` (for up to the 20 s ``busy_timeout``, with
    :func:`retry_on_lock` behind that), and when it finally begins it reads a
    post-commit snapshot containing the winner's row.

    Neither path depends on a schema change, so this needs no migration and
    no partial unique index.
    """
    from django.db import connection

    if not connection.features.has_select_for_update:
        return

    from uploads.models import UploadedMedia

    UploadedMedia.objects.select_for_update().filter(pk=media.pk).first()


def start_analysis(user, media, overrides=None, setting=None):
    """
    Admit ``media`` for analysis, create the row, and hand it to the worker.

    Returns the row.  Ownership is the caller's problem (the view has already
    404'd a foreign ``media_id``).  Admission is *not*: the one-in-flight-run
    rule is enforced here, inside the same transaction as the insert, and a
    refusal is raised as :class:`AnalysisInFlight` carrying the row that is
    already going.  A caller that checks :func:`active_analysis_for` first
    still gets the cheap common-case answer, but two clicks 50 ms apart used
    to sail past that check and start the same upload twice — two artefact
    trees, two reports, two history rows, and the promised 409 never fired
    (review finding F20b).

    The row is created and committed *before* the job is enqueued: a worker
    thread has its own database connection and can only see committed data.
    For the same reason this must not be called from inside an ``atomic()``
    block when the threaded worker is enabled — the ``atomic()`` below would
    then be a savepoint of the caller's transaction and the worker would
    chase a row that is not visible to it yet.
    """
    from . import worker

    snapshot = build_settings_snapshot(setting=setting, overrides=overrides)

    def admit():
        with transaction.atomic():
            _serialise_admissions_for(media)
            in_flight = (
                AnalysisResult.objects
                .filter(media=media, status__in=ACTIVE_STATUSES)
                .order_by('-created_at')
                .first()
            )
            if in_flight is not None:
                raise AnalysisInFlight(in_flight)
            return AnalysisResult.objects.create(
                media=media,
                user=user,
                status=STATUS_PENDING,
                stage='waiting for a worker',
                settings_snapshot=snapshot,
            )

    analysis = retry_on_lock(admit, 'creating the analysis row')
    logger.info('Analysis %s created for media %s by %s',
                analysis.analysis_id, media.media_id, user.pk)

    worker.job_queue.enqueue(analysis.analysis_id)
    analysis.refresh_from_db()
    return analysis


def set_report_status(analysis, value):
    """
    Record where report generation for ``analysis`` has got to.

    The one writer of ``AnalysisResult.report_status``.  Every caller that
    starts, finishes or abandons a render goes through here, so the vocabulary
    is enforced in a single place and the transitions read as a sequence
    rather than as scattered column assignments.

    A targeted ``UPDATE`` rather than ``analysis.save()``, for the same reason
    the worker's progress reporter uses one: the caller is usually holding a
    fully-loaded row it fetched before a render that took seconds, and saving
    that instance would push its whole stale in-memory copy over whatever the
    current owner has written since.  ``exclude()`` makes a repeated
    transition free of writes — the worker and the reports seam both announce
    ``generating`` for the same render, which is the normal case, not an error.

    **Never raises.**  This is bookkeeping *about* report generation, not the
    generation itself.  If the write fails, the caller's own outcome — a
    rendered PDF, or the exception that stopped one — is still the truth that
    matters, and a client that gets no signal falls back to the wall-clock
    budget it has always had.  Crucially, a failure here must not replace the
    exception a failed render is in the middle of raising: that exception is
    the thing worth knowing.

    Args:
        analysis: The ``AnalysisResult`` whose report state is being recorded.
            Its in-memory ``report_status`` is updated to match, so a
            serializer handed this same instance does not describe a state two
            transitions old.
        value: One of ``analysis.models.REPORT_STATUS_CHOICES``.

    Returns:
        ``True`` when the stored value actually changed.
    """
    try:
        changed = retry_on_lock(
            lambda: AnalysisResult.objects.filter(pk=analysis.pk)
            .exclude(report_status=value)
            .update(report_status=value),
            f'recording report_status={value!r} for analysis {analysis.pk}',
        )
    except Exception:                       # noqa: BLE001 - advisory by design
        logger.exception('Could not record report_status=%r for analysis %s',
                         value, analysis.pk)
        return False

    analysis.report_status = value
    return bool(changed)


def generate_report(analysis):
    """
    Re-render the PDF for ``analysis`` via the reports app.

    Always passes ``force=True``.  The only caller is ``POST /api/analysis/
    {id}/report``, whose contract is "generate / regenerate" and which answers
    ``201 Created``: without the flag ``generate_report_for_analysis``
    short-circuits on the existing row and the endpoint hands back the *old*
    ``generated_at`` and the *old* bytes while claiming to have created
    something (review finding F3).  A caller that only wants "a report, if one
    does not exist yet" — the worker's post-run hook — calls
    ``reports.services.generate_report_for_analysis`` directly.

    Imported lazily and by name so this app has no import-time dependency on
    a sibling that may not have landed yet.  Raises :class:`ReportsUnavailable`
    when the reports service is missing, which the view turns into a 503; any
    other exception is the reports app's own failure and propagates.
    """
    try:
        from reports.services import generate_report_for_analysis
    except Exception as exc:                # noqa: BLE001 - ImportError or worse
        raise ReportsUnavailable(str(exc)) from exc
    return generate_report_for_analysis(analysis, force=True)


# ---------------------------------------------------------------------------
# Dashboard aggregation
# ---------------------------------------------------------------------------

def _day_label(day):
    """``date(2026, 9, 1)`` -> ``'Sep 1'`` — portable across macOS/Windows."""
    return f'{day:%b} {day.day}'


def _pct_delta(current, previous):
    """
    Percentage change between two windows, rounded to one decimal.

    A previous window of zero has no defined percentage change; reporting
    ``+100%`` for "we went from nothing to something" is the convention every
    dashboard in the world uses, and ``0.0`` for "still nothing" is the only
    honest answer.
    """
    current = float(current or 0)
    previous = float(previous or 0)
    if previous == 0:
        return 0.0 if current == 0 else 100.0
    return round(((current - previous) / previous) * 100.0, 1)


def _integer_percentages(counts, total):
    """
    Split ``counts`` into integer percentages that sum to exactly 100.

    Naive rounding of 1/3, 1/3, 1/3 gives 33+33+33 = 99 and a donut chart with
    a visible gap.  The largest-remainder (Hamilton) method hands the leftover
    points to the buckets with the biggest fractional parts, so the slices
    always add up.
    """
    if total <= 0:
        return {key: 0 for key in counts}

    exact = {key: (value * 100.0) / total for key, value in counts.items()}
    floors = {key: int(value) for key, value in exact.items()}
    remainder = 100 - sum(floors.values())

    ranked = sorted(exact, key=lambda key: (exact[key] - floors[key], counts[key]),
                    reverse=True)
    for key in ranked[:max(0, remainder)]:
        floors[key] += 1
    return floors


def _window_totals(queryset, start, end):
    """Counts and averages for one time window, in a single query."""
    window = queryset.filter(created_at__gte=start, created_at__lt=end)
    aggregated = window.aggregate(
        analyses=Count('pk'),
        high_severity=Count('pk', filter=Q(overall_severity=SEVERITY_HIGH)),
        avg_confidence=Avg('avg_confidence', filter=Q(avg_confidence__gt=0)),
    )
    return {
        'analyses': int(aggregated['analyses'] or 0),
        'high_severity': int(aggregated['high_severity'] or 0),
        'avg_confidence': round(float(aggregated['avg_confidence'] or 0.0), 4),
    }


def dashboard_stats(user, days=DASHBOARD_DAYS_DEFAULT):
    """
    Everything the dashboard screen renders, for one user, in ~9 queries.

    Scoped to ``user`` even for administrators: "my dashboard" being the same
    thing for everybody is far easier to reason about (and to test) than a
    number that silently changes meaning when you are promoted.  Admin-wide
    reporting, if it is ever needed, belongs on its own endpoint.

    Empty accounts get zeros and empty lists — never ``null``, never seeded
    demo data.
    """
    days = clamp_int(days, DASHBOARD_DAYS_MIN, DASHBOARD_DAYS_MAX,
                     default=DASHBOARD_DAYS_DEFAULT)

    now = timezone.now()
    today = timezone.localdate(now)
    window_start_day = today - timedelta(days=days - 1)
    # One query has to cover both the requested window and the fixed-length
    # sparkline, so group over whichever reaches further back.
    series_start_day = min(window_start_day,
                           today - timedelta(days=SPARKLINE_POINTS - 1))

    start = _start_of_day(series_start_day)
    window_start = _start_of_day(window_start_day)
    window_end = _start_of_day(today + timedelta(days=1))
    previous_start = window_start - timedelta(days=days)

    owned = AnalysisResult.objects.filter(user=user)

    # -- per-day analysis buckets (1 query) ------------------------------
    per_day = {}
    daily = (
        owned.filter(created_at__gte=start, created_at__lt=window_end)
        .annotate(day=TruncDate('created_at'))
        .values('day')
        .annotate(
            total=Count('pk'),
            high=Count('pk', filter=Q(overall_severity=SEVERITY_HIGH)),
            confidence=Avg('avg_confidence', filter=Q(avg_confidence__gt=0)),
        )
        .order_by('day')
    )
    for bucket in daily:
        per_day[bucket['day']] = bucket

    # -- per-day report buckets (1 query) --------------------------------
    per_day_reports = {}
    daily_reports = (
        owned.filter(report__isnull=False,
                     report__generated_at__gte=start,
                     report__generated_at__lt=window_end)
        .annotate(day=TruncDate('report__generated_at'))
        .values('day')
        .annotate(total=Count('pk'))
        .order_by('day')
    )
    for bucket in daily_reports:
        per_day_reports[bucket['day']] = int(bucket['total'] or 0)

    # -- window totals and their predecessors (4 queries) ----------------
    current = _window_totals(owned, window_start, window_end)
    previous = _window_totals(owned, previous_start, window_start)
    current['reports'] = _count_reports(owned, window_start, window_end)
    previous_reports = _count_reports(owned, previous_start, window_start)

    totals = {
        'analyses': current['analyses'],
        'high_severity': current['high_severity'],
        'reports': current['reports'],
        'avg_confidence': current['avg_confidence'],
        'analyses_delta_pct': _pct_delta(current['analyses'],
                                         previous['analyses']),
        'high_delta_pct': _pct_delta(current['high_severity'],
                                     previous['high_severity']),
        'reports_delta_pct': _pct_delta(current['reports'], previous_reports),
        'confidence_delta_pct': _pct_delta(current['avg_confidence'],
                                           previous['avg_confidence']),
    }

    # -- dense, zero-filled series ---------------------------------------
    detections_over_time = []
    for offset in range(days):
        day = window_start_day + timedelta(days=offset)
        bucket = per_day.get(day)
        detections_over_time.append({
            'date': day.isoformat(),
            'label': _day_label(day),
            'detections': int(bucket['total']) if bucket else 0,
            'high': int(bucket['high']) if bucket else 0,
        })

    sparklines = {'analyses': [], 'high': [], 'reports': [], 'confidence': []}
    for offset in range(SPARKLINE_POINTS):
        day = today - timedelta(days=SPARKLINE_POINTS - 1 - offset)
        bucket = per_day.get(day)
        sparklines['analyses'].append(int(bucket['total']) if bucket else 0)
        sparklines['high'].append(int(bucket['high']) if bucket else 0)
        sparklines['reports'].append(per_day_reports.get(day, 0))
        sparklines['confidence'].append(
            round(float(bucket['confidence']), 4)
            if bucket and bucket['confidence'] else 0.0
        )

    # -- severity distribution (1 query) ---------------------------------
    severity_counts = {level: 0 for level in SEVERITY_ORDER}
    breakdown = (
        owned.filter(created_at__gte=window_start, created_at__lt=window_end)
        .values('overall_severity')
        .annotate(total=Count('pk'))
    )
    for bucket in breakdown:
        level = bucket['overall_severity']
        if level in severity_counts:
            severity_counts[level] = int(bucket['total'] or 0)

    graded_total = sum(severity_counts.values())
    percentages = _integer_percentages(severity_counts, graded_total)
    severity_distribution = [
        {
            'key': level,
            'label': SEVERITY_LABELS[level],
            'value': severity_counts[level],
            'pct': percentages[level],
            'color': SEVERITY_PALETTE[level],
        }
        for level in SEVERITY_ORDER
    ]

    # -- recent rows and the live queue (2 queries + prefetch) -----------
    recent_analyses = list(
        owned.select_related('media', 'report')
        .order_by('-created_at')[:RECENT_ANALYSES_LIMIT]
    )
    processing_queue = [
        {
            'analysis_id': str(row['analysis_id']),
            'filename': row['media__filename'],
            'status': row['status'],
            'progress': int(row['progress'] or 0),
            'stage': row['stage'] or '',
        }
        for row in owned.filter(status__in=ACTIVE_STATUSES)
        .order_by('created_at')
        .values('analysis_id', 'media__filename', 'status', 'progress', 'stage')
    ]

    return {
        'totals': totals,
        'sparklines': sparklines,
        'detections_over_time': detections_over_time,
        'severity_distribution': severity_distribution,
        'recent_analyses': recent_analyses,
        'processing_queue': processing_queue,
    }


def _count_reports(queryset, start, end):
    """Reports generated for this user's analyses inside a window."""
    return queryset.filter(
        report__isnull=False,
        report__generated_at__gte=start,
        report__generated_at__lt=end,
    ).count()


def _start_of_day(day):
    """Midnight at the start of ``day``, in the active timezone."""
    from datetime import datetime, time as time_of_day

    naive = datetime.combine(day, time_of_day.min)
    if settings.USE_TZ:
        return timezone.make_aware(naive, timezone.get_current_timezone())
    return naive


# ---------------------------------------------------------------------------
# Deletion
#
# An analysis owns files in two separate trees:
#
#   MEDIA_ROOT/analyses/<analysis_id>/   preview, frames, crops, masks
#   MEDIA_ROOT/reports/<report_id>.pdf   the generated report
#
# and the only pointer to the second one is the ``GeneratedReport`` row that
# the ``AnalysisResult`` cascade destroys.  Deleting the row first and then
# calling ``delete_analysis_artifacts`` — which is what this module used to
# do — therefore reclaimed the first tree and orphaned the PDF forever
# (review finding F6).  The functions below resolve *everything* an analysis
# owns while the rows are still there; ``common.storage.delete_paths`` does
# the reclaiming afterwards.
# ---------------------------------------------------------------------------

def artifact_paths_for_analyses(analysis_ids):
    """
    Every path on disk owned by these analyses.

    One artefact directory per analysis plus one PDF per generated report.
    Directories and files are returned in a single list on purpose —
    :func:`common.storage.delete_paths` dispatches per entry, so no caller
    has to keep them apart.

    The reports app is consulted through a lazy import, matching
    :func:`generate_report`: a deployment where ``reports`` cannot be
    imported still deletes its artefact trees, and says so in the log rather
    than failing the delete.
    """
    from common.storage import analysis_artifact_dir

    ids = [analysis_id for analysis_id in (analysis_ids or ()) if analysis_id]
    if not ids:
        return []

    paths = [analysis_artifact_dir(analysis_id, create=False)
             for analysis_id in ids]

    try:
        from reports.services import report_paths_for_analyses
    except Exception:                       # noqa: BLE001 - ImportError or worse
        logger.warning(
            'reports app unavailable; %d analysis PDF(s) may be left behind '
            'on disk.', len(ids), exc_info=True,
        )
    else:
        paths.extend(report_paths_for_analyses(ids))

    return paths


def artifact_paths_for_analysis(analysis):
    """Everything one analysis owns on disk. See :func:`artifact_paths_for_analyses`."""
    return artifact_paths_for_analyses([analysis.analysis_id])


def artifact_paths_for_media(media):
    """
    Everything ``media``'s analyses own on disk.

    For the ``DELETE /api/media/{id}`` path, whose cascade
    (``UploadedMedia`` -> ``AnalysisResult`` -> ``GeneratedReport``) wipes
    out the rows that were the only record of those files.  Usage::

        leftovers = artifact_paths_for_media(media)   # before the delete
        media.delete()                                # cascade
        delete_paths(leftovers)                       # after it commits

    Deliberately *excludes* the upload's own file: that one is a Django
    ``FileField``, and its owner already deletes it through the storage API.
    """
    ids = list(
        AnalysisResult.objects.filter(media=media)
        .values_list('analysis_id', flat=True)
    )
    return artifact_paths_for_analyses(ids)


def delete_analysis(analysis):
    """
    Remove an analysis row and every file it owns — artefacts *and* its PDF.

    Order matters twice over.  The paths are resolved first, because the
    ``GeneratedReport`` row is the only thing that knows where the PDF is.
    The row goes next, inside a transaction.  The files go last, after the
    commit: if the reclaim fails we would rather leak a few megabytes than
    leave a live row pointing at bytes that are already gone, and a rollback
    must never be able to destroy data that is still referenced.
    """
    from common.storage import delete_paths

    analysis_id = analysis.analysis_id
    leftovers = artifact_paths_for_analysis(analysis)

    def drop():
        with transaction.atomic():
            analysis.delete()

    retry_on_lock(drop, f'deleting analysis {analysis_id}')
    reclaimed = delete_paths(leftovers)
    logger.info('Analysis %s deleted; reclaimed %d of %d path(s) on disk',
                analysis_id, reclaimed, len(leftovers))
