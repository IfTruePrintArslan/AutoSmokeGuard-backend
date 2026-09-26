"""
Serializers for the reports API.

Two shapes, matching the frozen contract's ``ReportObj``:

* :class:`ReportListSerializer` — the compact form used by ``GET
  /api/reports``, with a *slim* nested ``analysis`` summary (source filename,
  overall severity, vehicle/smoke totals) rather than the full detail, so the
  list endpoint stays cheap.
* :class:`ReportDetailSerializer` — replaces that slim summary with the full
  nested ``AnalysisDetail`` for ``GET /api/report/{report_id}``.

The full nested analysis (detail endpoint only) is
``analysis.serializers.AnalysisDetailSerializer`` — the real one, imported at
module scope, with nothing behind it.

That used to be a lazy import wrapped in ``except Exception:``, falling back
to a minimal inline dict, because the analysis app's serializers were being
written concurrently and might not have existed yet.  They exist.  Keeping
the guard had become actively harmful (review finding F10): the ``except``
caught *serialization* failures too, so one malformed ``bounding_box`` turned
``GET /api/report/{id}`` into a permanent 200 carrying a quietly different
shape — ``annotated_frames: []``, ``segmenter_mode: null``, ``device: null``
where the real serializer emits ``""`` — traced only by a single WARNING.
A wrong answer that looks right is worse than an error, especially on the
endpoint the report-review screen renders from, so a genuine serialization
failure now surfaces as a real error with a traceback.
"""
from drf_spectacular.utils import extend_schema_field
from rest_framework import serializers

from analysis.serializers import AnalysisDetailSerializer

from .models import GeneratedReport


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
        """
        The full ``AnalysisDetail``, from the analysis app's own serializer.

        No ``try``/``except``.  Anything this raises is a real defect in the
        data or in that serializer, and the caller is entitled to hear about
        it — see the module docstring and review finding F10 for why the
        fallback that used to sit here was removed rather than tightened.
        """
        return AnalysisDetailSerializer(obj.analysis, context=self.context).data
