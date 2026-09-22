"""
Serializers for the analysis API — the frozen wire shapes.

Three conventions run through this module.

**Stored paths are relative; the wire carries URLs.**  Every ``*_path`` column
holds a ``MEDIA_ROOT``-relative string so the media tree can be moved to
another volume or an object store without rewriting rows.  The client can do
nothing useful with that, so every one of them is passed through
:func:`common.storage.media_url` on the way out.  The contract keeps the
``crop_path`` / ``mask_path`` field *names* even though the values are URLs,
so those names are preserved here rather than "corrected".

**The row shape is the unit of reuse.**  ``AnalysisRowSerializer`` is what
``/api/analysis``, ``/api/history`` and the dashboard's ``recent_analyses``
all emit; ``AnalysisDetailSerializer`` extends it rather than redefining it,
so the two can never drift.

**Runtime facts hide inside ``settings_snapshot``.**  The frozen schema has no
column for ``segmenter_mode``, ``device`` or the annotated-frame list, and
adding one would mean a migration this agent is not allowed to write.  The
worker stashes them under a reserved ``_runtime`` key inside the snapshot
JSON; :class:`AnalysisDetailSerializer` lifts them to the top level and
removes the key, so the ``settings_snapshot`` the API exposes stays pure
configuration.
"""
from django.core.exceptions import ObjectDoesNotExist
from drf_spectacular.utils import extend_schema_field
from rest_framework import serializers

from common.storage import media_url

from .models import (
    SEVERITY_CHOICES,
    STATUS_CHOICES,
    VEHICLE_TYPE_CHOICES,
    AnalysisResult,
    DetectedVehicle,
    SmokeRegion,
)
from .services import RUNTIME_KEY

# ---------------------------------------------------------------------------
# Leaf shapes
# ---------------------------------------------------------------------------


class MediaBriefSerializer(serializers.Serializer):
    """The four media fields an analysis row needs to render a list item."""

    media_id = serializers.UUIDField(read_only=True)
    filename = serializers.CharField(read_only=True)
    media_type = serializers.CharField(read_only=True)
    url = serializers.SerializerMethodField()

    def get_url(self, media) -> str | None:
        """Public URL of the original upload."""
        return media_url(media.file_path or getattr(media.file, 'name', ''))


class ReportBriefSerializer(serializers.Serializer):
    """Just enough of a report for a row to link to it."""

    report_id = serializers.UUIDField(read_only=True)
    generated_at = serializers.DateTimeField(read_only=True)


class BoundingBoxSerializer(serializers.Serializer):
    """
    Pixel box in source-image coordinates.

    Every component defaults to ``0`` so a row whose ``bounding_box`` JSON is
    empty or partial — a half-written record from an older pipeline build —
    renders as a degenerate box instead of blowing up the whole response.
    """

    x = serializers.IntegerField(default=0)
    y = serializers.IntegerField(default=0)
    w = serializers.IntegerField(default=0)
    h = serializers.IntegerField(default=0)


class SeverityCountsSerializer(serializers.Serializer):
    """Histogram of smoke regions by severity band."""

    low = serializers.IntegerField(default=0)
    moderate = serializers.IntegerField(default=0)
    high = serializers.IntegerField(default=0)


# ---------------------------------------------------------------------------
# Detection hierarchy
# ---------------------------------------------------------------------------


class SmokeRegionSerializer(serializers.ModelSerializer):
    """One segmented smoke plume hanging off a detected vehicle."""

    mask_path = serializers.SerializerMethodField()

    class Meta:
        model = SmokeRegion
        fields = ('smoke_id', 'mask_path', 'intensity', 'severity',
                  'confidence', 'area_ratio', 'opacity')
        read_only_fields = fields

    def get_mask_path(self, region) -> str | None:
        """URL of the binary mask PNG, or ``null`` when none was written."""
        return media_url(region.mask_path)


class DetectedVehicleSerializer(serializers.ModelSerializer):
    """
    One vehicle the detector found, with its smoke region if it had one.

    The contract exposes ``smoke`` as a single object or ``null`` even though
    the schema models it one-to-many.  The pipeline produces at most one
    region per vehicle (it segments a single exhaust ROI), so the first one is
    *the* one; keeping the relation one-to-many leaves room for per-plume
    tracking later without a migration.

    ``smoke_regions`` is read through ``list(...)`` so a prefetched cache is
    used as-is.  Calling ``.first()`` would issue a fresh query per vehicle
    and reintroduce the N+1 the detail view's ``prefetch_related`` removes.
    """

    bounding_box = BoundingBoxSerializer(read_only=True)
    crop_path = serializers.SerializerMethodField()
    smoke = serializers.SerializerMethodField()

    class Meta:
        model = DetectedVehicle
        fields = ('vehicle_id', 'vehicle_type', 'bounding_box', 'confidence',
                  'frame_number', 'timestamp_seconds', 'crop_path', 'smoke')
        read_only_fields = fields

    def get_crop_path(self, vehicle) -> str | None:
        """URL of the cropped vehicle JPEG, or ``null``."""
        return media_url(vehicle.crop_path)

    @extend_schema_field(SmokeRegionSerializer(allow_null=True))
    def get_smoke(self, vehicle):
        """The vehicle's smoke region, or ``null`` when it was clean."""
        regions = list(vehicle.smoke_regions.all())
        if not regions:
            return None
        return SmokeRegionSerializer(regions[0], context=self.context).data


# ---------------------------------------------------------------------------
# Analysis rows
# ---------------------------------------------------------------------------


class AnalysisRowSerializer(serializers.ModelSerializer):
    """
    The list shape: one analysis summarised for a table or a card.

    Requires ``select_related('media', 'report')`` on the queryset — without
    it every row costs two extra queries, which is precisely the N+1 the
    history endpoint's performance budget forbids.
    """

    media = MediaBriefSerializer(read_only=True)
    duration_seconds = serializers.FloatField(read_only=True, allow_null=True)
    severity_counts = SeverityCountsSerializer(read_only=True)
    overall_severity = serializers.SerializerMethodField()
    preview_url = serializers.SerializerMethodField()
    report = serializers.SerializerMethodField()

    class Meta:
        model = AnalysisResult
        fields = ('analysis_id', 'media', 'status', 'progress', 'created_at',
                  'start_time', 'end_time', 'duration_seconds',
                  'total_vehicles', 'total_smoke', 'avg_confidence',
                  'overall_severity', 'severity_counts', 'preview_url',
                  'report')
        read_only_fields = fields

    def get_overall_severity(self, analysis) -> str | None:
        """Severity band, or ``null`` when nothing was detected."""
        return analysis.overall_severity or None

    def get_preview_url(self, analysis) -> str | None:
        """URL of the annotated preview frame, or ``null``."""
        return media_url(analysis.preview_path)

    @staticmethod
    def _report_of(analysis):
        """The related report, or ``None`` when no PDF has been produced."""
        try:
            return analysis.report
        except ObjectDoesNotExist:
            return None

    @extend_schema_field(ReportBriefSerializer(allow_null=True))
    def get_report(self, analysis):
        """``{report_id, generated_at}`` once a PDF exists, else ``null``."""
        report = self._report_of(analysis)
        if report is None:
            return None
        return ReportBriefSerializer(report).data


class AnalysisDetailSerializer(AnalysisRowSerializer):
    """
    The full result: the row plus every detection and the run's provenance.

    Requires ``select_related('media', 'user', 'report')`` and
    ``prefetch_related('vehicles__smoke_regions')``.
    """

    settings_snapshot = serializers.SerializerMethodField()
    annotated_frames = serializers.SerializerMethodField()
    vehicles = DetectedVehicleSerializer(many=True, read_only=True)
    segmenter_mode = serializers.SerializerMethodField()
    device = serializers.SerializerMethodField()

    class Meta(AnalysisRowSerializer.Meta):
        fields = AnalysisRowSerializer.Meta.fields + (
            'settings_snapshot', 'frames_processed', 'error_message',
            'annotated_frames', 'vehicles', 'segmenter_mode', 'device',
        )
        read_only_fields = fields

    @staticmethod
    def _runtime(analysis):
        """The worker's stashed runtime facts, or an empty mapping."""
        snapshot = analysis.settings_snapshot or {}
        runtime = snapshot.get(RUNTIME_KEY)
        return runtime if isinstance(runtime, dict) else {}

    @extend_schema_field(serializers.DictField())
    def get_settings_snapshot(self, analysis):
        """The frozen configuration, with the runtime stash removed."""
        snapshot = dict(analysis.settings_snapshot or {})
        snapshot.pop(RUNTIME_KEY, None)
        return snapshot

    @extend_schema_field(serializers.ListField(child=serializers.CharField()))
    def get_annotated_frames(self, analysis):
        """URLs of the annotated frames the pipeline kept."""
        frames = self._runtime(analysis).get('annotated_frames') or []
        return [url for url in (media_url(frame) for frame in frames) if url]

    def get_segmenter_mode(self, analysis) -> str:
        """``'unet'`` or ``'classical'`` — which segmenter actually ran."""
        return self._runtime(analysis).get('segmenter_mode') or ''

    def get_device(self, analysis) -> str:
        """``'mps'``, ``'cuda:0'`` or ``'cpu'`` — where inference ran."""
        return self._runtime(analysis).get('device') or ''


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------


class StatusSerializer(serializers.Serializer):
    """
    The polling shape. Built from a ``.values()`` dict, never a model.

    The front end hits this every second while a run is in flight, so it must
    stay a single indexed read of a handful of columns: no related objects, no
    properties, no serializer method fields that lazy-load.
    """

    job_id = serializers.UUIDField(read_only=True)
    analysis_id = serializers.UUIDField(read_only=True)
    status = serializers.ChoiceField(choices=STATUS_CHOICES, read_only=True)
    progress = serializers.IntegerField(read_only=True)
    stage = serializers.CharField(read_only=True, allow_blank=True)
    started_at = serializers.DateTimeField(read_only=True, allow_null=True)
    ended_at = serializers.DateTimeField(read_only=True, allow_null=True)
    error_message = serializers.CharField(read_only=True, allow_blank=True)
    report_id = serializers.UUIDField(read_only=True, allow_null=True)
    total_vehicles = serializers.IntegerField(read_only=True)
    total_smoke = serializers.IntegerField(read_only=True)
    # CharField, not ChoiceField: see SeveritySliceSerializer for why a second
    # field name carrying the severity vocabulary confuses enum naming in the
    # generated OpenAPI document. Values are still low|moderate|high|null.
    overall_severity = serializers.CharField(read_only=True, allow_null=True)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class AnalysisSettingsSerializer(serializers.Serializer):
    """
    Per-run overrides a client may send with ``POST /api/analyze``.

    Deliberately **unbounded at the boundary**: no ``min_value`` /
    ``max_value``.  A slider that overshoots, a stale front end or a curl typo
    should not cost the user a 400 — the server clamps the value into a range
    the pipeline can survive (see
    :func:`analysis.services.build_settings_snapshot`) and records what it
    actually used in ``settings_snapshot``.  Only genuinely un-coercible input
    (``"abc"`` for a float) is rejected.
    """

    confidence_threshold = serializers.FloatField(required=False)
    frame_sample_rate = serializers.IntegerField(required=False)
    sensitivity = serializers.IntegerField(required=False)
    auto_generate_pdf = serializers.BooleanField(required=False)
    model = serializers.CharField(required=False, allow_blank=True,
                                  max_length=100)


class AnalyzeRequestSerializer(serializers.Serializer):
    """Body of ``POST /api/analyze``."""

    media_id = serializers.UUIDField()
    settings = AnalysisSettingsSerializer(required=False)


class AnalyzeAcceptedSerializer(serializers.Serializer):
    """202 body of ``POST /api/analyze``."""

    job_id = serializers.UUIDField(read_only=True)
    analysis_id = serializers.UUIDField(read_only=True)
    status = serializers.ChoiceField(choices=STATUS_CHOICES, read_only=True)
    message = serializers.CharField(read_only=True)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class DashboardTotalsSerializer(serializers.Serializer):
    """Headline counters and their change versus the preceding window."""

    analyses = serializers.IntegerField()
    high_severity = serializers.IntegerField()
    reports = serializers.IntegerField()
    avg_confidence = serializers.FloatField()
    analyses_delta_pct = serializers.FloatField()
    high_delta_pct = serializers.FloatField()
    reports_delta_pct = serializers.FloatField()
    confidence_delta_pct = serializers.FloatField()


class DashboardSparklinesSerializer(serializers.Serializer):
    """Fourteen daily points per metric, oldest first."""

    analyses = serializers.ListField(child=serializers.IntegerField())
    high = serializers.ListField(child=serializers.IntegerField())
    reports = serializers.ListField(child=serializers.IntegerField())
    confidence = serializers.ListField(child=serializers.FloatField())


class DetectionsOverTimePointSerializer(serializers.Serializer):
    """One day of the dense, zero-filled activity series."""

    date = serializers.DateField()
    label = serializers.CharField()
    detections = serializers.IntegerField()
    high = serializers.IntegerField()


class SeveritySliceSerializer(serializers.Serializer):
    """
    One wedge of the severity donut, palette colour included.

    ``key`` is a plain ``CharField`` rather than a ``ChoiceField`` even though
    its values are the severity vocabulary: a third field name carrying the
    same choice set makes drf-spectacular emit ambiguous enum component names,
    and the fix for that (``ENUM_NAME_OVERRIDES``) lives in settings, which
    this app does not own.
    """

    key = serializers.CharField()
    label = serializers.CharField()
    value = serializers.IntegerField()
    pct = serializers.IntegerField()
    color = serializers.CharField()


class ProcessingQueueItemSerializer(serializers.Serializer):
    """One job the user currently has in flight."""

    analysis_id = serializers.UUIDField()
    filename = serializers.CharField()
    status = serializers.ChoiceField(choices=STATUS_CHOICES)
    progress = serializers.IntegerField()
    stage = serializers.CharField(allow_blank=True)


class DashboardStatsSerializer(serializers.Serializer):
    """The whole dashboard payload, assembled by ``services.dashboard_stats``."""

    totals = DashboardTotalsSerializer()
    sparklines = DashboardSparklinesSerializer()
    detections_over_time = DetectionsOverTimePointSerializer(many=True)
    severity_distribution = SeveritySliceSerializer(many=True)
    recent_analyses = AnalysisRowSerializer(many=True)
    processing_queue = ProcessingQueueItemSerializer(many=True)


# ---------------------------------------------------------------------------
# Documentation-only shapes
# ---------------------------------------------------------------------------


class ErrorEnvelopeSerializer(serializers.Serializer):
    """The project-wide failure envelope, declared so the docs show it."""

    detail = serializers.CharField()
    code = serializers.CharField()
    errors = serializers.DictField(allow_null=True, required=False)


class AnalysisConflictSerializer(ErrorEnvelopeSerializer):
    """409 body that also names the run already in flight."""

    analysis_id = serializers.UUIDField()
    job_id = serializers.UUIDField()
    status = serializers.ChoiceField(choices=STATUS_CHOICES)


VEHICLE_TYPE_VALUES = tuple(value for value, _label in VEHICLE_TYPE_CHOICES)
SEVERITY_VALUES = tuple(value for value, _label in SEVERITY_CHOICES)
STATUS_VALUES = tuple(value for value, _label in STATUS_CHOICES)
