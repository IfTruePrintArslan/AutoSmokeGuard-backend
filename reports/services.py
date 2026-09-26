"""
The public seam other apps use to obtain a PDF for a finished analysis.

``generate_report_for_analysis`` is the one function outside code is allowed
to call.  Two callers exist today:

* the analysis worker, which calls it *lazily* right after a run finishes
  (``analysis.status`` flips to ``'done'``), wrapped in its own try/except so
  a rendering problem never fails the analysis itself;
* the ``POST /api/analysis/{id}/report`` endpoint (owned by the analysis
  app), which calls it again with ``force=True`` to regenerate on demand.

Both callers may run on a worker thread rather than inside a request, so this
module is deliberately request-free: no ``HttpRequest``, no thread-local
state, nothing that assumes it is running inside DRF.  It opens exactly the
file it writes and closes it before returning (via :mod:`reports.pdf`,
through ``reportlab``'s own file handling), so nothing lingers open across
calls.

Concurrency (review finding F11).  Both entry points can fire at the same
instant for the same report — a double-clicked Download against a row whose
file has gone missing, a request-thread regeneration racing the worker's
post-run hook — and they would then render into the *same* path.  Two
defences, at different layers:

* :func:`reports.pdf.build_report` renders to a unique temporary file beside
  the target and ``os.replace()``s it into place, so no reader ever sees a
  partially written PDF and ``stat()`` can never measure one.  That is the
  correctness guarantee, and it holds across processes.
* :func:`_analysis_report_lock` below serialises renders for one analysis
  *within* a process, so the duplicated work is avoided rather than merely
  made safe — and, crucially, so two first-time generators cannot both
  INSERT and blow up the ``OneToOne``.  A cross-process duplicate is caught
  as an ``IntegrityError`` and resolved by adopting the winner's row.

NFR (UC-07): a report must be produced within 30 seconds of the analysis
completing.  We do not enforce that here — a slow render is still a valid
render — but we log the elapsed time on every call so a regression is visible
in the logs rather than only in a flaky test.

That budget used to be the *only* thing a client had to go on: with no report
on the wire and no other signal, the UI inferred "still rendering" from how
long ago the analysis finished, and therefore could not tell a render that was
half done from one that had died in the first two seconds.  So every path
through :func:`generate_report_for_analysis` now records where it got to on
``AnalysisResult.report_status`` (via ``analysis.services.set_report_status``)
— ``generating`` before the render, ``ready`` after it lands, and ``failed``
before any exception is allowed to leave.  The 30-second budget survives as
the client's fallback for a server too old to send the field, not as its
primary evidence.
"""
import logging
import threading
import time
from contextlib import contextmanager

from django.db import IntegrityError, transaction
from django.utils import timezone

from analysis.models import (
    REPORT_FAILED,
    REPORT_GENERATING,
    REPORT_READY,
    STATUS_DONE,
)
from analysis.services import set_report_status
from common.storage import (
    delete_paths,
    from_media_relative,
    report_path,
    to_media_relative,
)

from .models import GeneratedReport
from .pdf import build_report

logger = logging.getLogger('asg.reports')

#: Target wall-clock budget for one report, per the UC-07 non-functional
#: requirement.  Exceeding it does not raise — it is logged as a warning so
#: the regression shows up without turning a slow machine into a hard failure.
REPORT_BUDGET_SECONDS = 30

# ---------------------------------------------------------------------------
# Per-analysis rendering locks
#
# ``{analysis_id: [lock, interested_thread_count]}``, guarded by the registry
# lock below.  Refcounted rather than left to grow: a long-lived process that
# generates thousands of reports must not accumulate a lock object per id.
# ``threading.Lock`` is not weak-referenceable, so a ``WeakValueDictionary``
# is not an option here.
#
# Keyed on the **analysis**, not on the report.  The report id is the obvious
# choice and it is the wrong one: for a first-time generation the row does
# not exist yet, so each caller mints its own ``uuid4`` and two callers
# would take two different locks — which is exactly the collision the worker
# hits in production, where ``_maybe_generate_report`` starts rendering the
# automatic PDF at the instant the status flips to ``done`` and the UI, which
# is polling for precisely that, fires ``POST /api/analysis/{id}/report``.
# Both read "no report yet", both INSERT, and the loser hits
# ``UNIQUE constraint failed: reports_generated_report.analysis_id``.
# "One report per analysis" is the invariant, so the analysis is the unit of
# exclusion.
#
# Deliberately *not* one global lock: two different analyses have no reason
# to wait for one another, and a global one would serialise the whole report
# surface behind the slowest render in flight.
# ---------------------------------------------------------------------------

_LOCK_REGISTRY = {}
_LOCK_REGISTRY_GUARD = threading.Lock()


@contextmanager
def _analysis_report_lock(analysis_id):
    """Hold the rendering lock for one analysis's report for the duration."""
    key = str(analysis_id)

    with _LOCK_REGISTRY_GUARD:
        entry = _LOCK_REGISTRY.get(key)
        if entry is None:
            entry = _LOCK_REGISTRY[key] = [threading.Lock(), 0]
        entry[1] += 1
        lock = entry[0]

    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        with _LOCK_REGISTRY_GUARD:
            entry[1] -= 1
            if entry[1] <= 0:
                _LOCK_REGISTRY.pop(key, None)


class ReportNotReady(Exception):
    """
    Raised when a PDF is requested for an analysis that has not finished.

    Callers with access to a DRF response (the reports views, the analysis
    app's generate endpoint) are expected to catch this and translate it into
    the project's ``409 report_not_ready`` envelope; a bare worker-thread
    caller can just let it propagate into its own error log.
    """


# ---------------------------------------------------------------------------
# Reclaiming PDFs (review finding F6)
#
# ``GeneratedReport`` is the only pointer to a file in
# ``MEDIA_ROOT/reports/``.  Every cascade that destroys those rows — deleting
# an analysis, deleting the upload an analysis belongs to — therefore has to
# resolve the paths *before* the delete and reclaim them afterwards.  These
# two helpers are that seam.  They are deliberately side-effect free, so a
# caller can take the list while the rows still exist and do the filesystem
# work only once the delete has actually committed: a rolled-back transaction
# must not leave a live row pointing at a file we already unlinked.
# ---------------------------------------------------------------------------

def report_file_path(report):
    """
    Absolute path of one report's PDF, or ``None`` if it cannot be trusted.

    Prefers the stored ``report_path`` column and falls back to the
    canonical ``reports/<report_id>.pdf`` for a row saved before the file
    was written.  A column that escapes ``MEDIA_ROOT`` yields ``None``
    rather than a path: the caller's very next move is ``unlink``, and an
    arbitrary-file-delete is a far worse outcome than a leaked PDF.
    """
    from django.core.exceptions import SuspiciousFileOperation

    if report.report_path:
        try:
            return from_media_relative(report.report_path)
        except SuspiciousFileOperation:
            logger.error(
                'Report %s has a report_path outside MEDIA_ROOT (%r); '
                'refusing to reclaim it.',
                report.report_id, report.report_path,
            )
            return None
    return report_path(report.report_id, create=False)


def report_paths_for_analyses(analysis_ids):
    """
    Absolute PDF paths for every report attached to ``analysis_ids``.

    Call it while the rows still exist — after the cascade there is nothing
    left to read the paths off.
    """
    ids = [analysis_id for analysis_id in (analysis_ids or ()) if analysis_id]
    if not ids:
        return []

    rows = GeneratedReport.objects.filter(analysis_id__in=ids).only(
        'report_id', 'report_path',
    )
    return [path for path in (report_file_path(row) for row in rows)
            if path is not None]


def generate_report_for_analysis(analysis, force=False):
    """
    Produce (or fetch) the PDF report for one completed analysis.

    Args:
        analysis: An ``analysis.models.AnalysisResult`` instance. Must be
            fully loaded (this function never re-fetches it), and its
            ``status`` must be ``'done'``.
        force: When ``False`` (the default) and a report already exists, that
            existing row is returned unchanged — no re-render, no file
            rewrite.  When ``True`` the PDF is always rebuilt and the
            existing ``GeneratedReport`` row (same ``report_id``, hence the
            same on-disk path) is overwritten in place, so there is never more
            than one PDF per analysis.  ``generated_at`` moves forward on a
            forced rebuild: it records when the bytes currently on disk were
            produced, not when the row was first created.

    Returns:
        The ``GeneratedReport`` row for this analysis, freshly saved.

    Raises:
        ReportNotReady: ``analysis.status != 'done'``.  Nothing is written to
            disk or to the database in this case — including
            ``report_status``, which stays where it was: nothing was
            attempted, so nothing failed.

    Report state.  Every other path records itself on
    ``AnalysisResult.report_status``: ``generating`` before the render,
    ``ready`` once the row is saved (or once a racing process's row is
    adopted), ``failed`` before any exception leaves this function.  A caller
    that swallows the exception — the worker's post-run hook does, by design —
    therefore no longer swallows the *fact* of it.

    Concurrency (review finding F11). Three layers, because two threads
    genuinely do arrive here at the same instant — the worker's
    ``_maybe_generate_report`` fires the moment a run flips to ``done``, and
    the UI is polling for exactly that moment so it can offer "Generate
    report":

    1. :func:`_analysis_report_lock` serialises everything below per
       *analysis* inside this process, and the existing-row check is
       repeated inside it, so the loser adopts the winner's row instead of
       minting a second one.
    2. :func:`reports.pdf.build_report` renders to a temporary file and
       ``os.replace()``s it into place, so no reader ever sees a partial
       PDF and ``stat()`` can never measure one.
    3. An ``IntegrityError`` on the ``OneToOne`` is caught and resolved by
       adopting the winner, which covers the one case an in-process lock
       cannot: a second gunicorn *process*.
    """
    if analysis.status != STATUS_DONE:
        raise ReportNotReady(
            f'Analysis {analysis.analysis_id} is not ready for a report '
            f'yet (status={analysis.status!r}).'
        )

    # Fast path, outside the lock: an existing report and no force means
    # there is nothing to render and therefore nothing to serialise.
    existing = GeneratedReport.objects.filter(analysis=analysis).first()
    if existing is not None and not force:
        # Also heals a row whose column predates this field (or was written
        # by a build that did not have it): a PDF that demonstrably exists is
        # 'ready', whatever the column happens to say.
        set_report_status(analysis, REPORT_READY)
        return existing

    with _analysis_report_lock(analysis.analysis_id):
        # Re-read under the lock. The thread we just waited behind may have
        # created the row — or re-rendered it, which satisfies a force=True
        # caller that arrived at the same instant just as well.
        existing = GeneratedReport.objects.filter(analysis=analysis).first()
        if existing is not None and not force:
            set_report_status(analysis, REPORT_READY)
            return existing

        # Re-using the existing row (rather than deleting + recreating) keeps
        # the report_id — and therefore the on-disk path from
        # common.storage.report_path — stable across a forced regeneration,
        # so a previously shared download link keeps working.
        report = (existing if existing is not None
                  else GeneratedReport(analysis=analysis))
        target = report_path(report.report_id)

        # Announced *before* the first expensive call, so a client that polls
        # mid-render reads 'generating' rather than a stale 'pending'.
        set_report_status(analysis, REPORT_GENERATING)

        started = time.monotonic()
        try:
            page_count = build_report(analysis, report.report_id, target)
            elapsed = time.monotonic() - started

            # Stat *after* build_report's os.replace(), and still under the
            # lock: the name now refers to the finished file this call made.
            report.report_path = to_media_relative(target)
            report.page_count = page_count
            report.file_size_bytes = target.stat().st_size
            # generated_at is auto_now_add, which fires on INSERT only — on a
            # re-render the row would otherwise keep advertising the moment
            # the *first* PDF was made while carrying the bytes of the newest
            # one.  Setting it by hand is honoured (Django's
            # DateTimeField.pre_save only overrides the value when add=True),
            # and needs no migration.
            report.generated_at = timezone.now()

            try:
                # atomic() so the constraint violation is confined to a
                # savepoint: an IntegrityError poisons the enclosing
                # transaction, and the very next query below (finding the
                # winner) would otherwise raise TransactionManagementError.
                with transaction.atomic():
                    report.save()
            except IntegrityError:
                winner = (GeneratedReport.objects
                          .filter(analysis=analysis).first())
                if winner is None:                    # pragma: no cover
                    raise
                logger.info(
                    'Another process created report %s for analysis %s while '
                    'we were rendering %s; adopting theirs and discarding '
                    'ours.',
                    winner.report_id, analysis.analysis_id, report.report_id,
                )
                # Nothing in the database points at what we just rendered.
                delete_paths([target])
                set_report_status(analysis, REPORT_READY)
                return winner
        except Exception:
            # The render died.  Leaving the row on 'generating' would be the
            # exact untruth this field exists to stop: the client would keep
            # showing a disabled "Preparing report…" for the rest of the
            # 30-second budget, waiting on work that is already over.  The
            # exception itself is re-raised untouched — the worker's hook
            # still swallows it so a bad PDF cannot fail a good analysis, but
            # it can no longer do so invisibly.
            set_report_status(analysis, REPORT_FAILED)
            raise

        set_report_status(analysis, REPORT_READY)

        log = logger.warning if elapsed > REPORT_BUDGET_SECONDS else logger.info
        log(
            'Generated report %s for analysis %s in %.2fs (budget %ss): '
            '%d page(s), %d byte(s).',
            report.report_id, analysis.analysis_id, elapsed,
            REPORT_BUDGET_SECONDS, page_count, report.file_size_bytes,
        )

        return report
