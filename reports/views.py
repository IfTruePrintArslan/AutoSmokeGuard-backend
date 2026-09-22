"""
Views for the reports API: fetch, list and download generated PDFs.

Ownership is enforced by scoping every queryset to the requesting user (or,
for an administrator, to everyone) rather than by an object-level permission
check — a mismatched owner and an unknown id then look identical to the
client (both 404 ``report_not_found``), which is the point: the API must not
leak whether a given report id exists at all to someone who cannot see it.
"""
import logging

from django.core.exceptions import SuspiciousFileOperation
from django.http import FileResponse
from django.utils import timezone
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import generics, permissions
from rest_framework.response import Response
from rest_framework.views import APIView

from common.exceptions import error_response
from common.pagination import StandardPagination
from common.storage import from_media_relative

from .serializers import ReportDetailSerializer, ReportListSerializer
from .services import ReportNotReady, generate_report_for_analysis
from .models import GeneratedReport

logger = logging.getLogger('asg.reports')

_ERROR_ENVELOPE = {
    'type': 'object',
    'properties': {
        'detail': {'type': 'string'},
        'code': {'type': 'string'},
        'errors': {'type': 'object', 'nullable': True},
    },
}


def _visible_reports(user):
    """
    Reports *user* is allowed to see: their own, or — for an administrator —
    every report in the system.

    ``select_related('analysis', 'analysis__media')`` backs both the
    ownership filter below and ``ReportListSerializer``'s nested ``analysis``
    summary (filename, severity, vehicle/smoke totals) — without it, every
    row in ``GET /api/reports`` would cost two extra queries, an N+1 the list
    endpoint's query count must stay flat against (see
    ``tests/test_reports.py``).
    """
    queryset = GeneratedReport.objects.select_related(
        'analysis', 'analysis__user', 'analysis__media',
    )
    if getattr(user, 'role', None) == 'admin' or user.is_staff:
        return queryset
    return queryset.filter(analysis__user=user)


def _get_visible_report(user, report_id):
    """The report *user* may see with this id, or ``None``."""
    return _visible_reports(user).filter(report_id=report_id).first()


class ReportDetailView(APIView):
    """``GET /api/report/{report_id}`` — full report, nested analysis included."""

    permission_classes = [permissions.IsAuthenticated]

    @extend_schema(
        summary='Retrieve a generated report',
        description=(
            'Full report JSON, including the nested analysis detail. Only '
            'the owning user or an administrator may fetch it; anyone else, '
            'and unknown ids, get a 404 so existence is never leaked.'
        ),
        responses={
            200: ReportDetailSerializer,
            404: OpenApiTypes.OBJECT,
        },
    )
    def get(self, request, report_id):
        report = _get_visible_report(request.user, report_id)
        if report is None:
            return error_response('Report not found.', 'report_not_found', 404)
        serializer = ReportDetailSerializer(report, context={'request': request})
        return Response(serializer.data)


class ReportListView(generics.ListAPIView):
    """``GET /api/reports`` — paginated, without the nested analysis detail."""

    serializer_class = ReportListSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardPagination

    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            return GeneratedReport.objects.none()
        return _visible_reports(self.request.user)

    @extend_schema(
        summary='List generated reports',
        description='Paginated list of the caller\'s own reports (all reports, for an administrator).',
        parameters=[
            OpenApiParameter('page', OpenApiTypes.INT, description='Page number, 1-based.'),
            OpenApiParameter('page_size', OpenApiTypes.INT, description='Items per page, capped server-side.'),
        ],
        responses={200: ReportListSerializer},
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)


class ReportDownloadView(APIView):
    """``GET /api/download-report/{report_id}`` — streams the PDF file."""

    permission_classes = [permissions.IsAuthenticated]

    @extend_schema(
        summary='Download a generated report PDF',
        description=(
            'Streams the report as an attachment. If the database row exists '
            'but the PDF has gone missing from disk, it is regenerated on the '
            'fly rather than failing the request.'
        ),
        responses={
            200: OpenApiTypes.BINARY,
            404: OpenApiTypes.OBJECT,
            409: OpenApiTypes.OBJECT,
        },
    )
    def get(self, request, report_id):
        report = _get_visible_report(request.user, report_id)
        if report is None:
            return error_response('Report not found.', 'report_not_found', 404)

        # from_media_relative confines the stored path to MEDIA_ROOT and
        # raises if it escapes (finding ASG-07). A row whose report_path has
        # been tampered with must read as "no such report", not as an
        # arbitrary file read and not as a 500.
        try:
            absolute = (
                from_media_relative(report.report_path)
                if report.report_path else None
            )
        except SuspiciousFileOperation:
            logger.error(
                'Report %s has a report_path outside MEDIA_ROOT (%r); '
                'refusing to serve it.',
                report.report_id, report.report_path,
            )
            return error_response('Report not found.', 'report_not_found', 404)

        if absolute is None or not absolute.is_file():
            logger.info(
                'Report %s has no file on disk; regenerating before download.',
                report.report_id,
            )
            try:
                report = generate_report_for_analysis(report.analysis, force=True)
            except ReportNotReady:
                return error_response('Report not ready yet.', 'report_not_ready', 409)
            absolute = from_media_relative(report.report_path)

        stamp = timezone.localtime(report.generated_at).strftime('%Y%m%d')
        filename = f'autosmokeguard-report-{str(report.report_id)[:8]}-{stamp}.pdf'

        response = FileResponse(
            open(absolute, 'rb'),
            as_attachment=True,
            filename=filename,
            content_type='application/pdf',
        )
        return response
