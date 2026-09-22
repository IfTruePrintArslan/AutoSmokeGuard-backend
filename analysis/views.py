"""
The analysis HTTP surface: start a run, poll it, read it, list it, chart it.

Design notes that are not obvious from the route table
------------------------------------------------------

**A job is an analysis.**  There is no Job table and no separate job id — the
``job_id`` the front end polls *is* the ``analysis_id``.  One row, one
identifier, and "how far along is my upload?" is an indexed primary-key read
rather than a join.

**``POST /api/analyze`` returns immediately.**  It writes a row, hands the id
to the worker and answers 202.  It never waits for inference: a 90-second
video would otherwise hold a WSGI thread open past every sensible proxy
timeout.  The only exception is the deliberately synchronous test/``--check``
mode (``ASG['WORKER_ENABLED'] = False``), where the run completes inline and
the 202 therefore reports the real, already-terminal status.

**Ownership is enforced in the queryset, not in a permission class.**  A
foreign id must be indistinguishable from a missing one, so the base queryset
is filtered by owner and a mismatch falls out as a 404.  Returning 403 would
confirm that somebody else's analysis exists.  Administrators may read and
delete any single object (support work) but their *lists* and their dashboard
stay scoped to their own rows, so "my numbers" never silently changes meaning
when an account is promoted.

**Every response body is contract-shaped**, including failures: the standard
``{detail, code, errors}`` envelope from :mod:`common.exceptions`.  The two
409s add an ``analysis_id`` alongside the envelope so the UI can navigate
straight to the run that is already in flight instead of showing a dead end.
"""
import logging

from django.db.models import Prefetch
from django.shortcuts import get_object_or_404
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, OpenApiResponse, extend_schema
from rest_framework import status as http_status
from rest_framework.generics import ListAPIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django_filters.rest_framework import DjangoFilterBackend

from common.exceptions import error_response
from common.pagination import StandardPagination
from system_config.models import SystemSetting
from uploads.models import UploadedMedia

from . import services
from .filters import ALLOWED_ORDERING, AnalysisFilterSet
from .models import STATUS_DONE, AnalysisResult, SmokeRegion
from .serializers import (
    AnalysisConflictSerializer,
    AnalysisDetailSerializer,
    AnalysisRowSerializer,
    AnalyzeAcceptedSerializer,
    AnalyzeRequestSerializer,
    DashboardStatsSerializer,
    ErrorEnvelopeSerializer,
    StatusSerializer,
)

logger = logging.getLogger('asg.analysis')

ANALYSIS_NOT_FOUND = 'No analysis with that id.'
MEDIA_NOT_FOUND = 'No media with that id.'


# ---------------------------------------------------------------------------
# Shared query helpers
# ---------------------------------------------------------------------------

def _is_admin(user):
    """True for an authenticated administrator."""
    return bool(user and user.is_authenticated
                and (getattr(user, 'is_admin', False) or user.is_staff))


def owned_analyses(user, allow_admin=False):
    """
    Base queryset for one requester.

    ``allow_admin`` widens it to every row for administrators; it is passed
    only by the single-object endpoints, never by a list or the dashboard.
    """
    queryset = AnalysisResult.objects.all()
    if allow_admin and _is_admin(user):
        return queryset
    return queryset.filter(user=user)


def row_queryset(user, allow_admin=False):
    """List/row queryset with the two joins ``AnalysisRowSerializer`` needs."""
    return owned_analyses(user, allow_admin).select_related('media', 'report')


def detail_queryset(user, allow_admin=False):
    """
    Detail queryset: two joins plus the whole detection tree in two queries.

    ``Prefetch`` with an explicit ordering keeps the vehicle list stable
    between requests — the model's default ordering is
    ``(frame_number, -confidence)``, and pinning it here documents that the
    API depends on it.
    """
    return (
        owned_analyses(user, allow_admin)
        .select_related('media', 'user', 'report')
        .prefetch_related(
            Prefetch(
                'vehicles__smoke_regions',
                queryset=SmokeRegion.objects.order_by('-intensity'),
            ),
            'vehicles',
        )
    )


def conflict_response(detail, analysis):
    """
    409 envelope that also names the run already in flight.

    A superset of the standard error envelope: the extra keys let the UI
    offer "view the run that is already going" instead of a dead end.
    """
    return Response(
        {
            'detail': detail,
            'code': 'analysis_in_progress',
            'errors': None,
            'analysis_id': str(analysis.analysis_id),
            'job_id': str(analysis.analysis_id),
            'status': analysis.status,
        },
        status=http_status.HTTP_409_CONFLICT,
    )


def report_payload(report):
    """
    Build the contract's ``ReportObj`` from a ``GeneratedReport``.

    Assembled here rather than imported from ``reports.serializers`` on
    purpose: this module must keep importing cleanly whether or not the
    reports app has landed, and the shape is frozen in the API contract
    anyway.  The download path is likewise the frozen one.
    """
    return {
        'report_id': str(report.report_id),
        'analysis_id': str(report.analysis_id),
        'generated_at': report.generated_at,
        'page_count': report.page_count,
        'file_size_bytes': report.file_size_bytes,
        'download_url': f'/api/download-report/{report.report_id}',
    }


# ---------------------------------------------------------------------------
# POST /api/analyze
# ---------------------------------------------------------------------------

class AnalyzeView(APIView):
    """Accept an upload for analysis and hand it to the worker pool."""

    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary='Start an analysis',
        description=(
            'Queues the vehicle-detection and smoke-segmentation pipeline for '
            'an uploaded file and returns immediately with a job id — which is '
            'the analysis id. Poll `GET /api/status/{job_id}` for progress.\n\n'
            'Optional per-run `settings` are **clamped**, not rejected: values '
            'outside the safe range are corrected to the nearest allowed one '
            'and the effective configuration is frozen onto the row as '
            '`settings_snapshot`.'
        ),
        request=AnalyzeRequestSerializer,
        responses={
            202: AnalyzeAcceptedSerializer,
            400: OpenApiResponse(ErrorEnvelopeSerializer,
                                 'Malformed body.'),
            404: OpenApiResponse(ErrorEnvelopeSerializer,
                                 'No such media, or it belongs to someone '
                                 'else.'),
            409: OpenApiResponse(AnalysisConflictSerializer,
                                 'That upload already has a run in flight.'),
        },
        tags=['Analysis'],
    )
    def post(self, request):
        """Validate, admit, enqueue — then answer 202 without blocking."""
        payload = AnalyzeRequestSerializer(data=request.data)
        payload.is_valid(raise_exception=True)
        data = payload.validated_data

        # Lock-tolerant: this read shares a SQLite file with two worker
        # threads that are writing results. See services.retry_on_lock.
        media = services.retry_on_lock(
            lambda: UploadedMedia.objects.filter(
                media_id=data['media_id'], user=request.user,
            ).first(),
            'loading the requested media',
        )
        if media is None:
            # Deliberately the same answer for "does not exist" and "not
            # yours" — otherwise this endpoint enumerates other people's ids.
            return error_response(MEDIA_NOT_FOUND, 'media_not_found',
                                  http_status.HTTP_404_NOT_FOUND)

        in_flight = services.active_analysis_for(media)
        if in_flight is not None:
            return conflict_response(
                'This file is already being analysed.', in_flight,
            )

        analysis = services.start_analysis(
            user=request.user,
            media=media,
            overrides=data.get('settings') or {},
            setting=SystemSetting.get_solo(),
        )

        body = {
            'job_id': str(analysis.analysis_id),
            'analysis_id': str(analysis.analysis_id),
            # The row's *real* state. With the threaded worker this is always
            # 'queued' or 'running'; only the synchronous test/--check mode
            # can already be terminal by the time we answer.
            'status': analysis.status,
            'message': (
                'Analysis queued. Poll /api/status/'
                f'{analysis.analysis_id} for progress.'
            ),
        }
        return Response(body, status=http_status.HTTP_202_ACCEPTED)


# ---------------------------------------------------------------------------
# GET /api/status/{job_id}
# ---------------------------------------------------------------------------

#: Exactly the columns ``StatusSerializer`` needs, and nothing else.
STATUS_COLUMNS = (
    'analysis_id', 'status', 'progress', 'stage', 'start_time', 'end_time',
    'error_message', 'total_vehicles', 'total_smoke', 'overall_severity',
    'report__report_id',
)


class StatusView(APIView):
    """Poll one job. Designed to be called once a second, so kept tiny."""

    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary='Poll analysis progress',
        description=(
            'Cheap status read for a running job. `job_id` is the '
            '`analysis_id`. Returns `report_id` as soon as a PDF exists.'
        ),
        responses={
            200: StatusSerializer,
            404: OpenApiResponse(ErrorEnvelopeSerializer, 'No such job.'),
        },
        tags=['Analysis'],
    )
    def get(self, request, job_id):
        """One ``SELECT`` of eleven columns, joined to the report id."""
        row = (
            owned_analyses(request.user, allow_admin=True)
            .filter(pk=job_id)
            .values(*STATUS_COLUMNS)
            .first()
        )
        if row is None:
            return error_response(ANALYSIS_NOT_FOUND, 'analysis_not_found',
                                  http_status.HTTP_404_NOT_FOUND)

        body = {
            'job_id': row['analysis_id'],
            'analysis_id': row['analysis_id'],
            'status': row['status'],
            'progress': row['progress'],
            'stage': row['stage'] or '',
            'started_at': row['start_time'],
            'ended_at': row['end_time'],
            'error_message': row['error_message'] or '',
            'report_id': row['report__report_id'],
            'total_vehicles': row['total_vehicles'],
            'total_smoke': row['total_smoke'],
            'overall_severity': row['overall_severity'] or None,
        }
        return Response(StatusSerializer(body).data)


# ---------------------------------------------------------------------------
# GET /api/analysis  and  GET /api/history
# ---------------------------------------------------------------------------

class AnalysisListView(ListAPIView):
    """
    The user's analyses, newest first.

    ``GET /api/analysis`` and ``GET /api/history`` are the same query with the
    same filter set; the contract names them separately because they back two
    different screens, and keeping both routes means either can grow its own
    behaviour later without a breaking change.
    """

    serializer_class = AnalysisRowSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = StandardPagination
    # Never used at runtime — ``get_queryset`` always wins. It exists so
    # drf-spectacular can introspect the model without instantiating a
    # request, which it does with an AnonymousUser that our owner filter
    # cannot accept.
    queryset = AnalysisResult.objects.none()
    # Only DjangoFilterBackend: the project-wide defaults also install DRF's
    # SearchFilter and OrderingFilter, which would give two competing
    # (differently spelled) ways to sort and an unwhitelisted `?search=`.
    filter_backends = [DjangoFilterBackend]
    filterset_class = AnalysisFilterSet

    def get_queryset(self):
        """Own rows only — history is never widened for administrators."""
        return row_queryset(self.request.user).order_by('-created_at')

    @extend_schema(
        summary='List analyses',
        description='Paginated analysis rows for the authenticated user.',
        parameters=[
            OpenApiParameter('status', OpenApiTypes.STR,
                             description='pending | queued | running | done | '
                                         'failed'),
            OpenApiParameter('page', OpenApiTypes.INT),
            OpenApiParameter('page_size', OpenApiTypes.INT,
                             description='1-100, default 10.'),
        ],
        responses={200: AnalysisRowSerializer(many=True)},
        tags=['Analysis'],
    )
    def get(self, request, *args, **kwargs):
        """Paginated list of the caller's analyses."""
        return super().get(request, *args, **kwargs)


class HistoryView(AnalysisListView):
    """The history screen (UC-08): the same rows with the full filter set."""

    @extend_schema(
        summary='Search analysis history',
        description=(
            'Paginated analysis rows with the full UC-08 filter set. All '
            'parameters are optional and combine with AND.'
        ),
        parameters=[
            OpenApiParameter('severity', OpenApiTypes.STR,
                             description='low | moderate | high'),
            OpenApiParameter('vehicle_type', OpenApiTypes.STR,
                             description='car | motorcycle | bus | truck — '
                                         'keeps runs containing at least one.'),
            OpenApiParameter('status', OpenApiTypes.STR,
                             description='pending | queued | running | done | '
                                         'failed'),
            OpenApiParameter('date_from', OpenApiTypes.DATE,
                             description='Inclusive, YYYY-MM-DD.'),
            OpenApiParameter('date_to', OpenApiTypes.DATE,
                             description='Inclusive, YYYY-MM-DD.'),
            OpenApiParameter('search', OpenApiTypes.STR,
                             description='Substring of the source filename.'),
            OpenApiParameter('ordering', OpenApiTypes.STR,
                             enum=list(ALLOWED_ORDERING)),
            OpenApiParameter('page', OpenApiTypes.INT),
            OpenApiParameter('page_size', OpenApiTypes.INT),
        ],
        responses={200: AnalysisRowSerializer(many=True)},
        tags=['Analysis'],
    )
    def get(self, request, *args, **kwargs):
        """Filtered, ordered, paginated history."""
        return super().get(request, *args, **kwargs)


# ---------------------------------------------------------------------------
# GET / DELETE /api/analysis/{analysis_id}
# ---------------------------------------------------------------------------

class AnalysisDetailView(APIView):
    """Read or discard one complete analysis."""

    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary='Get an analysis',
        description=(
            'The full result: summary, frozen configuration, annotated frame '
            'URLs and every detected vehicle with its smoke region.'
        ),
        responses={
            200: AnalysisDetailSerializer,
            404: OpenApiResponse(ErrorEnvelopeSerializer, 'No such analysis.'),
        },
        tags=['Analysis'],
    )
    def get(self, request, analysis_id):
        """Detail view, four queries regardless of how many vehicles there are."""
        analysis = get_object_or_404(
            detail_queryset(request.user, allow_admin=True), pk=analysis_id,
        )
        serializer = AnalysisDetailSerializer(
            analysis, context={'request': request},
        )
        return Response(serializer.data)

    @extend_schema(
        summary='Delete an analysis',
        description=(
            'Removes the row and its entire artefact directory (preview, '
            'annotated frames, crops, masks). Refused while the run is still '
            'in flight — cancel is not the same thing as delete, and deleting '
            'the row out from under a worker thread would strand its files.'
        ),
        responses={
            204: OpenApiResponse(description='Deleted.'),
            404: OpenApiResponse(ErrorEnvelopeSerializer, 'No such analysis.'),
            409: OpenApiResponse(AnalysisConflictSerializer,
                                 'The run is still in flight.'),
        },
        tags=['Analysis'],
    )
    def delete(self, request, analysis_id):
        """Delete a finished analysis and everything it wrote to disk."""
        analysis = get_object_or_404(
            owned_analyses(request.user, allow_admin=True), pk=analysis_id,
        )
        if not analysis.is_terminal:
            return conflict_response(
                'This analysis is still running; wait for it to finish before '
                'deleting it.',
                analysis,
            )
        services.delete_analysis(analysis)
        return Response(status=http_status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# POST /api/analysis/{analysis_id}/report
# ---------------------------------------------------------------------------

class AnalysisReportView(APIView):
    """Generate (or regenerate) the PDF for a finished analysis."""

    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary='Generate a report',
        description=(
            'Renders the PDF emission report for a completed analysis, '
            'replacing any previous one. Delegates to the reports app.'
        ),
        request=None,
        responses={
            201: OpenApiResponse(OpenApiTypes.OBJECT, 'The generated report.'),
            404: OpenApiResponse(ErrorEnvelopeSerializer, 'No such analysis.'),
            409: OpenApiResponse(ErrorEnvelopeSerializer,
                                 'The analysis has not finished.'),
            503: OpenApiResponse(ErrorEnvelopeSerializer,
                                 'The reports service is unavailable.'),
        },
        tags=['Analysis'],
    )
    def post(self, request, analysis_id):
        """Delegate to ``reports.services`` behind a lazy import."""
        analysis = get_object_or_404(
            owned_analyses(request.user, allow_admin=True)
            .select_related('media', 'user'),
            pk=analysis_id,
        )
        if analysis.status != STATUS_DONE:
            return error_response(
                'A report can only be generated for a completed analysis.',
                'report_not_ready', http_status.HTTP_409_CONFLICT,
            )

        try:
            report = services.generate_report(analysis)
        except services.ReportsUnavailable:
            logger.warning('Report generation requested but reports app is '
                           'not importable')
            return error_response(
                'Report generation is not available on this server yet.',
                'reports_unavailable',
                http_status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        if report is None:                            # pragma: no cover
            return error_response(
                'The report could not be generated.', 'report_failed',
                http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return Response(report_payload(report),
                        status=http_status.HTTP_201_CREATED)


# ---------------------------------------------------------------------------
# GET /api/dashboard/stats
# ---------------------------------------------------------------------------

class DashboardStatsView(APIView):
    """Everything the dashboard renders, in one round trip."""

    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary='Dashboard statistics',
        description=(
            'Totals with period-over-period deltas, 14-point sparklines, a '
            'dense zero-filled activity series, the severity distribution, '
            'the five most recent runs and everything currently in the '
            'queue — all scoped to the authenticated user.\n\n'
            'A brand-new account gets zeros and empty arrays, never seeded '
            'demo data.'
        ),
        parameters=[
            OpenApiParameter(
                'days', OpenApiTypes.INT,
                description=(
                    f'Window length in days '
                    f'({services.DASHBOARD_DAYS_MIN}-'
                    f'{services.DASHBOARD_DAYS_MAX}, default '
                    f'{services.DASHBOARD_DAYS_DEFAULT}). Out-of-range values '
                    f'are clamped.'
                ),
            ),
        ],
        responses={200: DashboardStatsSerializer},
        tags=['Dashboard'],
    )
    def get(self, request):
        """Aggregate the caller's rows over the requested window."""
        stats = services.dashboard_stats(
            request.user, days=request.query_params.get('days'),
        )
        serializer = DashboardStatsSerializer(
            stats, context={'request': request},
        )
        return Response(serializer.data)


__all__ = [
    'AnalysisDetailView',
    'AnalysisListView',
    'AnalysisReportView',
    'AnalyzeView',
    'DashboardStatsView',
    'HistoryView',
    'StatusView',
    'owned_analyses',
    'report_payload',
]
