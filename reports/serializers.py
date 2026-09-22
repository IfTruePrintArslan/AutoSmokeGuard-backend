"""
Serializers for the reports API.

Two shapes, matching the frozen contract's ``ReportObj``:

* :class:`ReportListSerializer` — the compact form used by ``GET
  /api/reports``, with a *slim* nested ``analysis`` summary (source filename,
  overall severity, vehicle/smoke totals) rather than the full detail, so the
  list endpoint stays cheap.
* :class:`ReportDetailSerializer` — replaces that slim summary with the full
  nested ``AnalysisDetail`` for ``GET /api/report/{report_id}``.

The full nested analysis (detail endpoint only) is sourced from
``analysis.serializers
.AnalysisDetailSerializer``, imported lazily inside the method rather than at
module scope: the analysis app's serializers are being written concurrently
by a sibling agent, and importing them at import time would make this whole
app fail to load until that lands. If the import (or the serializer itself)
raises, a minimal inline dict built straight from the ORM is used instead, so
this endpoint — and the tests in ``tests/test_reports.py`` — work regardless
of the other app's state.
"""
import logging

from drf_spectacular.utils import extend_schema_field
from rest_framework import serializers

from common.storage import media_url

from .models import GeneratedReport

logger = logging.getLogger('asg.reports')


class _ReportRowMediaSerializer(serializers.Serializer):
    """
    Just enough of the source media for a ``GET /api/reports`` row to show a
    filename — deliberately not the full ``analysis.serializers
    .MediaBriefSerializer`` shape (no ``url``), to keep the list endpoint's
    payload slim.
    """

    media_id = serializers.UUIDField(read_only=True)
    filename = serializers.CharField(read_only=True)
    media_type = serializers.CharField(read_only=True)


class _ReportRowAnalysisSerializer(serializers.Serializer):
    """
    Slim analysis summary nested into every ``GET /api/reports`` row, so the
    Reports table can render the source filename and overall severity
    without a second round trip per row. Deliberately not the full
    ``AnalysisDetail`` — see :class:`ReportDetailSerializer` for that.
    """

    analysis_id = serializers.UUIDField(read_only=True)
    overall_severity = serializers.SerializerMethodField()
    total_vehicles = serializers.IntegerField(read_only=True)
    total_smoke = serializers.IntegerField(read_only=True)
    media = _ReportRowMediaSerializer(read_only=True)

    def get_overall_severity(self, analysis) -> str | None:
        """Severity band, or ``null`` when nothing was detected."""
        return analysis.overall_severity or None


class ReportListSerializer(serializers.ModelSerializer):
    """
    Compact ``ReportObj`` shape: a slim nested ``analysis`` summary, not the
    full ``AnalysisDetail`` (that stays on :class:`ReportDetailSerializer`
    only, to keep the list endpoint cheap).
    """

    analysis_id = serializers.UUIDField(read_only=True)
    analysis = _ReportRowAnalysisSerializer(read_only=True)
    download_url = serializers.SerializerMethodField()

    class Meta:
        model = GeneratedReport
        fields = (
            'report_id', 'analysis_id', 'generated_at',
            'page_count', 'file_size_bytes', 'download_url', 'analysis',
        )
        read_only_fields = fields

    def get_download_url(self, obj) -> str:
        """Absolute-path URL of ``GET /api/download-report/{report_id}``."""
        return f'/api/download-report/{obj.report_id}'


def _fallback_analysis_detail(analysis):
    """
    Minimal, dependency-free stand-in for ``analysis.serializers
    .AnalysisDetailSerializer``.

    # TODO(reports): remove this fallback once the analysis app's own
    # serializer is guaranteed to be importable; this only covers the fields
    # reports/tests.py and the frozen AnalysisDetail contract need.
    """
    media = analysis.media
    vehicles = []
    for vehicle in analysis.vehicles.prefetch_related('smoke_regions').all():
        smoke = vehicle.smoke_regions.first()
        vehicles.append({
            'vehicle_id': str(vehicle.vehicle_id),
            'vehicle_type': vehicle.vehicle_type,
            'bounding_box': vehicle.bounding_box,
            'confidence': vehicle.confidence,
            'frame_number': vehicle.frame_number,
            'timestamp_seconds': vehicle.timestamp_seconds,
            'crop_path': media_url(vehicle.crop_path) if vehicle.crop_path else None,
            'smoke': None if smoke is None else {
                'smoke_id': str(smoke.smoke_id),
                'mask_path': media_url(smoke.mask_path) if smoke.mask_path else None,
                'intensity': smoke.intensity,
                'severity': smoke.severity,
                'confidence': smoke.confidence,
                'area_ratio': smoke.area_ratio,
                'opacity': smoke.opacity,
            },
        })

    return {
        'analysis_id': str(analysis.analysis_id),
        'media': {
            'media_id': str(media.media_id),
            'filename': media.filename,
            'media_type': media.media_type,
            'url': media_url(media.file_path) if media.file_path else None,
        },
        'status': analysis.status,
        'progress': analysis.progress,
        'created_at': analysis.created_at,
        'start_time': analysis.start_time,
        'end_time': analysis.end_time,
        'duration_seconds': analysis.duration_seconds,
        'total_vehicles': analysis.total_vehicles,
        'total_smoke': analysis.total_smoke,
        'avg_confidence': analysis.avg_confidence,
        'overall_severity': analysis.overall_severity or None,
        'severity_counts': analysis.severity_counts,
        'preview_url': media_url(analysis.preview_path) if analysis.preview_path else None,
        'report': None,
        'settings_snapshot': analysis.settings_snapshot,
        'frames_processed': analysis.frames_processed,
        'error_message': analysis.error_message,
        'annotated_frames': [],
        'vehicles': vehicles,
        'segmenter_mode': None,
        'device': None,
    }


class ReportDetailSerializer(ReportListSerializer):
    """
    Full ``ReportObj`` shape: the slim ``analysis`` summary from
    :class:`ReportListSerializer` is replaced by the full nested
    ``AnalysisDetail``. ``Meta`` (and its ``fields``) is inherited as-is —
    ``analysis`` is already a listed field, just re-declared below.
    """

    analysis = serializers.SerializerMethodField()

    @extend_schema_field(serializers.DictField())
    def get_analysis(self, obj):
        try:
            from analysis.serializers import AnalysisDetailSerializer
        except ImportError:
            return _fallback_analysis_detail(obj.analysis)

        try:
            return AnalysisDetailSerializer(obj.analysis, context=self.context).data
        except Exception:
            logger.warning(
                'analysis.serializers.AnalysisDetailSerializer failed for '
                'analysis %s; falling back to the inline shape.',
                obj.analysis_id, exc_info=True,
            )
            return _fallback_analysis_detail(obj.analysis)
