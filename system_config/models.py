"""
Runtime system configuration (UC-09).

Everything an administrator is allowed to retune without a redeploy lives in
one row: accepted upload formats and size caps, the detector/segmenter
thresholds, the severity banding, the video frame-sampling rate, and whether a
PDF is produced automatically when a run completes.

Why a singleton table rather than settings or env vars
------------------------------------------------------
The values have to be editable from the admin UI at runtime, auditable
(``updated_at`` / ``updated_by``), and readable by the ML worker in another
thread.  A one-row table gives all three for free, whereas Django settings are
process-local and frozen at import.

``config.settings.ASG`` still holds the *boot* defaults — the values used
before an administrator has ever touched the screen, and the ones
:meth:`SystemSetting.get_solo` seeds the row with.

Views and serializers are owned by the sysconfig API agent; this module is the
schema only.
"""
from django.conf import settings
from django.db import models

#: There is exactly one configuration row, and this is its primary key.
SINGLETON_PK = 1


def _split_formats(value):
    """Parse a stored 'jpg,jpeg,png' string into a clean lower-case list."""
    return [
        item.strip().lstrip('.').lower()
        for item in (value or '').split(',')
        if item.strip()
    ]


class SystemSetting(models.Model):
    """
    The one-and-only system configuration row.

    Never instantiate this directly — call :meth:`get_solo`, which
    get-or-creates ``pk=1``.  :meth:`save` pins the primary key and
    :meth:`delete` is a no-op, so even a mistaken ``SystemSetting().save()`` or
    a stray admin click cannot produce a second row or leave the system with
    none.
    """

    id = models.PositiveSmallIntegerField(
        primary_key=True,
        editable=False,
        verbose_name='ID',
    )

    # -- upload policy ------------------------------------------------------
    allowed_image_formats = models.CharField(
        max_length=100,
        default='jpg,jpeg,png',
        help_text='Comma-separated image extensions, no dots.',
    )
    allowed_video_formats = models.CharField(
        max_length=100,
        default='mp4,avi,mov',
        help_text='Comma-separated video extensions, no dots.',
    )
    max_upload_mb = models.IntegerField(
        default=512,
        help_text='Maximum size of a single upload, in megabytes.',
    )
    max_video_seconds = models.IntegerField(
        default=300,
        help_text='Longest video clip accepted, in seconds.',
    )

    # -- model thresholds ---------------------------------------------------
    confidence_threshold = models.FloatField(
        default=0.35,
        help_text='Minimum detector confidence for a vehicle to count (0-1).',
    )
    smoke_mask_threshold = models.FloatField(
        default=0.5,
        help_text='Probability cut-off that binarises the smoke mask (0-1).',
    )

    # -- severity banding ---------------------------------------------------
    severity_low_max = models.FloatField(
        default=0.33,
        help_text='Intensity at or below this is "low".',
    )
    severity_moderate_max = models.FloatField(
        default=0.66,
        help_text='Intensity at or below this is "moderate"; above is "high".',
    )

    # -- pipeline behaviour -------------------------------------------------
    frame_sample_rate = models.IntegerField(
        default=5,
        help_text='Analyse every Nth frame of a video.',
    )
    auto_generate_pdf = models.BooleanField(
        default=True,
        help_text='Produce a PDF report automatically when a run completes.',
    )

    # -- audit --------------------------------------------------------------
    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='system_setting_updates',
    )

    class Meta:
        db_table = 'sysconfig_system_setting'
        verbose_name = 'system configuration'
        verbose_name_plural = 'system configuration'

    def __str__(self):
        return 'System configuration'

    # -- singleton enforcement ---------------------------------------------

    def save(self, **kwargs):
        """
        Pin the primary key to 1 so a second row can never be created.

        ``force_insert`` is dropped as well: ``objects.create(...)`` passes it,
        and honouring it would turn "save the config again" into an
        ``IntegrityError`` instead of an update.
        """
        self.pk = SINGLETON_PK
        kwargs.pop('force_insert', None)
        super().save(**kwargs)

    def delete(self, *args, **kwargs):
        """
        Refuse deletion — the rest of the system assumes this row exists.

        Returning the ``(0, {})`` tuple Django's ``delete()`` normally returns
        keeps callers (and the admin) working without a special case.
        """
        return 0, {}

    @classmethod
    def get_solo(cls):
        """
        Fetch the configuration row, creating it with defaults if absent.

        This is the only supported way to read the configuration; it is safe
        to call on a fresh database, from a request, or from a worker thread.
        """
        instance, _created = cls.objects.get_or_create(pk=SINGLETON_PK)
        return instance

    # -- parsed views of the stored values ----------------------------------

    @property
    def image_formats(self):
        """``allowed_image_formats`` as a list, e.g. ``['jpg','jpeg','png']``."""
        return _split_formats(self.allowed_image_formats)

    @property
    def video_formats(self):
        """``allowed_video_formats`` as a list, e.g. ``['mp4','avi','mov']``."""
        return _split_formats(self.allowed_video_formats)

    @property
    def all_formats(self):
        """Every accepted extension, images first, de-duplicated in order."""
        seen = {}
        for extension in self.image_formats + self.video_formats:
            seen[extension] = None
        return list(seen)

    @property
    def max_upload_bytes(self):
        """``max_upload_mb`` expressed in bytes, ready for the validator."""
        return int(self.max_upload_mb) * 1024 * 1024

    def severity_for(self, intensity):
        """
        Band a smoke intensity (0-1) into ``low`` / ``moderate`` / ``high``
        using the currently configured cut-offs.
        """
        value = float(intensity or 0.0)
        if value <= self.severity_low_max:
            return 'low'
        if value <= self.severity_moderate_max:
            return 'moderate'
        return 'high'

    def as_snapshot(self):
        """
        The subset of this configuration worth freezing onto an analysis run
        (``AnalysisResult.settings_snapshot``) so old reports stay explainable
        after the thresholds are retuned.
        """
        return {
            'confidence_threshold': self.confidence_threshold,
            'smoke_mask_threshold': self.smoke_mask_threshold,
            'severity_low_max': self.severity_low_max,
            'severity_moderate_max': self.severity_moderate_max,
            'frame_sample_rate': self.frame_sample_rate,
            'max_video_seconds': self.max_video_seconds,
        }
