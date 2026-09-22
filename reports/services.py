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

NFR (UC-07): a report must be produced within 30 seconds of the analysis
completing.  We do not enforce that here — a slow render is still a valid
render — but we log the elapsed time on every call so a regression is visible
in the logs rather than only in a flaky test.
"""
import logging
import time

from analysis.models import STATUS_DONE
from common.storage import report_path, to_media_relative

from .models import GeneratedReport
from .pdf import build_report

logger = logging.getLogger('asg.reports')

#: Target wall-clock budget for one report, per the UC-07 non-functional
#: requirement.  Exceeding it does not raise — it is logged as a warning so
#: the regression shows up without turning a slow machine into a hard failure.
REPORT_BUDGET_SECONDS = 30


class ReportNotReady(Exception):
    """
    Raised when a PDF is requested for an analysis that has not finished.

    Callers with access to a DRF response (the reports views, the analysis
    app's generate endpoint) are expected to catch this and translate it into
    the project's ``409 report_not_ready`` envelope; a bare worker-thread
    caller can just let it propagate into its own error log.
    """


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
            than one PDF per analysis.

    Returns:
        The ``GeneratedReport`` row for this analysis, freshly saved.

    Raises:
        ReportNotReady: ``analysis.status != 'done'``.  Nothing is written to
            disk or to the database in this case.
    """
    if analysis.status != STATUS_DONE:
        raise ReportNotReady(
            f'Analysis {analysis.analysis_id} is not ready for a report '
            f'yet (status={analysis.status!r}).'
        )

    existing = GeneratedReport.objects.filter(analysis=analysis).first()
    if existing is not None and not force:
        return existing

    # Re-using the existing row (rather than deleting + recreating) keeps the
    # report_id — and therefore the on-disk path from common.storage.report_
    # path — stable across a forced regeneration, so a previously shared
    # download link keeps working.
    report = existing if existing is not None else GeneratedReport(analysis=analysis)
    target = report_path(report.report_id)

    started = time.monotonic()
    page_count = build_report(analysis, report.report_id, target)
    elapsed = time.monotonic() - started

    report.report_path = to_media_relative(target)
    report.page_count = page_count
    report.file_size_bytes = target.stat().st_size
    report.save()

    log = logger.warning if elapsed > REPORT_BUDGET_SECONDS else logger.info
    log(
        'Generated report %s for analysis %s in %.2fs (budget %ss): '
        '%d page(s), %d byte(s).',
        report.report_id, analysis.analysis_id, elapsed,
        REPORT_BUDGET_SECONDS, page_count, report.file_size_bytes,
    )

    return report
