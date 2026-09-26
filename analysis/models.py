"""
The output side of the pipeline: what the models saw, and how bad it was.

Three tables, one per level of the detection hierarchy::

    AnalysisResult   one run of the pipeline over one upload
      └─ DetectedVehicle   one YOLO detection inside that run
           └─ SmokeRegion  one U-Net smoke mask attached to that vehicle

Deliberate design decisions
---------------------------

**No separate Job table.**  A job *is* an analysis.  ``status``, ``progress``
and ``stage`` live directly on ``AnalysisResult``, so the ``jobId`` the API
hands the front end for polling is simply the ``analysis_id``.  One row, one
identifier, no join to answer "how far along is my upload?".

**``user`` is denormalised onto ``AnalysisResult``.**  It is reachable via
``media.user``, but the history screen is the most-hit authenticated endpoint
in the product and it filters by owner on every request.  Carrying the FK here
turns that into a single indexed scan instead of a join, and it keeps the
ownership check in :class:`common.permissions.IsOwner` uniform across models.

**``settings_snapshot`` is frozen per run.**  Admins can change thresholds at
any time (UC-09); a report generated in March must still explain the numbers
it was produced with, so the effective configuration is copied into the row.

**``settings_snapshot`` is not purely configuration.**  It doubles as a
side-channel for a handful of *run outputs* that this schema has no column
for, under the reserved ``_runtime`` key.  Anything reading the snapshot off
the model — ``reports/`` does — sees them; the analysis API strips them back
out before serving.  The full list and the reasoning are in the comment
immediately above the field.  (That note is deliberately a comment rather
than an addition to the field's ``help_text``: ``help_text`` is part of a
field's deconstruction, so editing it would force a no-op ``AlterField``
migration for a documentation change.)

Views and serializers are owned by the analysis API agent; this module is the
schema only.
"""
import uuid

from django.conf import settings
from django.db import models

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

STATUS_PENDING = 'pending'
STATUS_QUEUED = 'queued'
STATUS_RUNNING = 'running'
STATUS_DONE = 'done'
STATUS_FAILED = 'failed'

STATUS_CHOICES = (
    (STATUS_PENDING, 'Pending'),
    (STATUS_QUEUED, 'Queued'),
    (STATUS_RUNNING, 'Running'),
    (STATUS_DONE, 'Done'),
    (STATUS_FAILED, 'Failed'),
)

#: Statuses from which no further transition happens.
TERMINAL_STATUSES = frozenset({STATUS_DONE, STATUS_FAILED})

SEVERITY_LOW = 'low'
SEVERITY_MODERATE = 'moderate'
SEVERITY_HIGH = 'high'

SEVERITY_CHOICES = (
    (SEVERITY_LOW, 'Low'),
    (SEVERITY_MODERATE, 'Moderate'),
    (SEVERITY_HIGH, 'High'),
)

# ---------------------------------------------------------------------------
# Report-generation vocabulary
#
# "Is the server still writing my PDF?" was, until this field existed, a
# question the client answered with a clock: a terminal analysis carrying no
# ``report`` yet was assumed to be mid-render until the server's own
# ``reports.services.REPORT_BUDGET_SECONDS`` (30s) had elapsed.  That guess is
# right in the common case and wrong in the one that matters — a render that
# dies two seconds in left the user watching a disabled "Preparing report…"
# for the other twenty-eight, for work that was already dead.  The server
# knows exactly which it is, so it says so.
#
# Same register as ``STATUS_CHOICES`` above: lowercase, one word, closed set.
# ``pending`` and ``failed`` are deliberately spelled the same as the analysis
# statuses they parallel; ``generating`` is *not* spelled ``running`` so that
# a log line or a schema enum can never be read as the pipeline's own state.
# ---------------------------------------------------------------------------

REPORT_PENDING = 'pending'
REPORT_GENERATING = 'generating'
REPORT_READY = 'ready'
REPORT_FAILED = 'failed'
REPORT_SKIPPED = 'skipped'

REPORT_STATUS_CHOICES = (
    (REPORT_PENDING, 'Pending'),
    (REPORT_GENERATING, 'Generating'),
    (REPORT_READY, 'Ready'),
    (REPORT_FAILED, 'Failed'),
    (REPORT_SKIPPED, 'Skipped'),
)

#: Report states nothing moves out of on its own.  ``pending`` and
#: ``generating`` are the only two that still promise a client something.
TERMINAL_REPORT_STATUSES = frozenset(
    {REPORT_READY, REPORT_FAILED, REPORT_SKIPPED},
)

VEHICLE_TYPE_CHOICES = (
    ('car', 'Car'),
    ('truck', 'Truck'),
    ('bus', 'Bus'),
    ('motorcycle', 'Motorcycle'),
)


def default_severity_counts():
    """Zeroed severity histogram — the default for ``severity_counts``."""
    return {SEVERITY_LOW: 0, SEVERITY_MODERATE: 0, SEVERITY_HIGH: 0}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AnalysisResult(models.Model):
    """
    One pipeline run over one uploaded file — job record *and* summary.

    Lifecycle: ``pending`` (row created) -> ``queued`` (accepted by the worker
    pool) -> ``running`` (frames being processed, ``progress``/``stage``
    updating) -> ``done`` or ``failed``.

    .. warning::
       ``settings_snapshot`` is **not** pure configuration.  The worker also
       stores run outputs under a reserved ``_runtime`` key; read the comment
       above that field before consuming the JSON.
    """

    analysis_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name='analysis ID',
    )
    media = models.ForeignKey(
        'uploads.UploadedMedia',
        related_name='analyses',
        on_delete=models.CASCADE,
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name='analyses',
        on_delete=models.CASCADE,
        help_text='Denormalised from media.user for fast history queries.',
    )

    # -- job state ----------------------------------------------------------
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
        db_index=True,
    )
    progress = models.PositiveSmallIntegerField(
        default=0,
        help_text='Completion percentage, 0-100.',
    )
    stage = models.CharField(
        max_length=50,
        blank=True,
        help_text="Human-readable step, e.g. 'segmenting frames'.",
    )
    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)

    # -- report generation --------------------------------------------------
    #
    # The sub-job that runs *after* this one: rendering the PDF.  It is a
    # separate state because it has a separate lifetime — an analysis is
    # ``done`` for a window during which its report is still being written,
    # and it can stay ``done`` forever with a report that failed to render.
    # Both look identical from ``status`` alone, which is precisely why the
    # client used to have to guess between them with a stopwatch.
    #
    # Written only by targeted ``UPDATE`` (``analysis.services.
    # set_report_status``), never by ``AnalysisResult.save()``: the worker
    # holds a stale in-memory row while it renders.
    report_status = models.CharField(
        max_length=20,
        choices=REPORT_STATUS_CHOICES,
        default=REPORT_PENDING,
        help_text='State of the PDF report for this run.',
    )

    # -- aggregate results --------------------------------------------------
    total_vehicles = models.IntegerField(default=0)
    total_smoke = models.IntegerField(
        default=0,
        help_text='Number of smoke regions detected across all vehicles.',
    )
    frames_processed = models.IntegerField(default=0)
    avg_confidence = models.FloatField(default=0.0)
    overall_severity = models.CharField(
        max_length=10,
        choices=SEVERITY_CHOICES,
        blank=True,
    )
    severity_counts = models.JSONField(
        default=default_severity_counts,
        help_text='{"low": n, "moderate": n, "high": n}',
    )

    # -- provenance ---------------------------------------------------------
    #
    # Reserved key: '_runtime'
    # -----------------------
    # Besides the frozen configuration, the worker stashes a dict of run
    # *outputs* under the reserved '_runtime' key
    # (``analysis.services.RUNTIME_KEY``), because this table has no columns
    # for them and adding six more to a schema that is already frozen by
    # migration 0001 was not worth it:
    #
    #     _runtime = {
    #         'segmenter_mode':   'unet' | 'classical',
    #         'device':           'cpu' | 'mps' | 'cuda',
    #         'annotated_frames': [<MEDIA_ROOT-relative path>, ...],
    #         'elapsed_seconds':  float | None,
    #         'mean_intensity':   float | None,
    #         'media_meta':       {...},
    #     }
    #
    # Consequences, in order of how likely they are to bite:
    #
    # * The analysis detail serializer pops '_runtime' and promotes its
    #   contents to top-level response fields, so the ``settings_snapshot``
    #   the API returns *is* pure configuration. The one stored here is not.
    # * ``reports/`` reads this field straight off the model and therefore
    #   sees the reserved key. Anything that iterates the snapshot to render
    #   "the settings this run used" must skip keys starting with '_'.
    # * It is an output, not an input: never feed a stored snapshot back into
    #   ``services.build_settings_snapshot`` or into ``MLConfig`` without
    #   stripping it first.
    settings_snapshot = models.JSONField(
        default=dict,
        help_text='Thresholds and sampling rate in force when this ran.',
    )
    preview_path = models.CharField(
        max_length=500,
        blank=True,
        help_text='Annotated preview image, relative to MEDIA_ROOT.',
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'analysis_result'
        ordering = ('-created_at',)
        indexes = [
            models.Index(fields=['user', 'created_at'],
                         name='analysis_user_created_idx'),
            models.Index(fields=['status'], name='analysis_status_idx'),
        ]
        verbose_name = 'analysis result'
        verbose_name_plural = 'analysis results'

    def __str__(self):
        return f'{self.analysis_id} [{self.status}]'

    # -- convenience --------------------------------------------------------

    @property
    def duration_seconds(self):
        """
        Wall-clock processing time, or ``None`` while the run is incomplete.

        Reported to the user as "analysed in 12.4 s" and used to sanity-check
        the non-functional requirement on throughput.
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def is_terminal(self):
        """True once the run has finished, successfully or not."""
        return self.status in TERMINAL_STATUSES

    @property
    def is_active(self):
        """True while the run is still waiting or in flight."""
        return not self.is_terminal


class DetectedVehicle(models.Model):
    """
    One vehicle located by the detector inside an analysis.

    For a still image ``frame_number``/``timestamp_seconds`` are null; for a
    video they place the detection in the clip so the report can cite
    "truck at 00:07".
    """

    vehicle_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name='vehicle ID',
    )
    analysis = models.ForeignKey(
        AnalysisResult,
        related_name='vehicles',
        on_delete=models.CASCADE,
    )

    vehicle_type = models.CharField(max_length=30, choices=VEHICLE_TYPE_CHOICES)
    bounding_box = models.JSONField(
        default=dict,
        help_text='Pixel box in source coordinates: {"x":, "y":, "w":, "h":}.',
    )
    confidence = models.FloatField(help_text='Detector confidence, 0-1.')

    frame_number = models.IntegerField(null=True, blank=True)
    timestamp_seconds = models.FloatField(null=True, blank=True)

    crop_path = models.CharField(
        max_length=500,
        blank=True,
        help_text='Cropped vehicle image, relative to MEDIA_ROOT.',
    )

    class Meta:
        db_table = 'analysis_detected_vehicle'
        ordering = ('frame_number', '-confidence')
        indexes = [
            models.Index(fields=['analysis', 'frame_number'],
                         name='vehicle_analysis_frame_idx'),
        ]
        verbose_name = 'detected vehicle'
        verbose_name_plural = 'detected vehicles'

    def __str__(self):
        return f'{self.vehicle_type} @ {self.confidence:.2f}'

    @property
    def has_smoke(self):
        """True when at least one smoke region was segmented on this vehicle."""
        return self.smoke_regions.exists()


class SmokeRegion(models.Model):
    """
    One segmented exhaust-smoke region belonging to a detected vehicle.

    ``intensity`` is the scalar the severity banding is derived from; the
    thresholds that turned it into ``severity`` are the ones recorded in the
    parent analysis's ``settings_snapshot``, so historical reports stay
    explainable after an admin retunes the system.
    """

    smoke_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name='smoke ID',
    )
    vehicle = models.ForeignKey(
        DetectedVehicle,
        related_name='smoke_regions',
        on_delete=models.CASCADE,
    )

    mask_path = models.CharField(
        max_length=500,
        blank=True,
        help_text='Binary segmentation mask, relative to MEDIA_ROOT.',
    )
    intensity = models.FloatField(help_text='Normalised smoke intensity, 0-1.')
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    confidence = models.FloatField(help_text='Segmenter confidence, 0-1.')
    area_ratio = models.FloatField(
        default=0.0,
        help_text='Mask area as a fraction of the vehicle bounding box.',
    )
    opacity = models.FloatField(
        default=0.0,
        help_text='Mean opacity of the mask, 0-1.',
    )

    class Meta:
        db_table = 'analysis_smoke_region'
        ordering = ('-intensity',)
        indexes = [
            models.Index(fields=['vehicle', 'severity'],
                         name='smoke_vehicle_severity_idx'),
        ]
        verbose_name = 'smoke region'
        verbose_name_plural = 'smoke regions'

    def __str__(self):
        return f'{self.severity} ({self.intensity:.2f})'
