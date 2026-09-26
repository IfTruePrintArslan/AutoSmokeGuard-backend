"""
Uploaded source media — the input side of the pipeline.

One ``UploadedMedia`` row is the record of a single file a user handed us: a
traffic photo or a dash-cam/CCTV clip.  It is deliberately separate from
``analysis.AnalysisResult`` because the same upload can be analysed more than
once (different confidence thresholds, a re-run after the model is retrained),
and because the file must survive an analysis being deleted.

Views and serializers are owned by the uploads API agent; this module is the
schema only.
"""
import uuid

from django.conf import settings
from django.db import models

from common.storage import user_media_path

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

MEDIA_TYPE_IMAGE = 'image'
MEDIA_TYPE_VIDEO = 'video'

MEDIA_TYPE_CHOICES = (
    (MEDIA_TYPE_IMAGE, 'Image'),
    (MEDIA_TYPE_VIDEO, 'Video'),
)

FORMAT_CHOICES = (
    ('jpg', 'JPEG'),
    ('png', 'PNG'),
    ('mp4', 'MP4'),
    ('avi', 'AVI'),
    ('mov', 'QuickTime'),
)


class UploadedMedia(models.Model):
    """
    A single file uploaded for emission analysis.

    Path handling is split in two on purpose:

    * ``file`` is the real ``FileField``.  Django owns it, so storage backends,
      ``.url``, and deletion all behave normally.  The upload is renamed to a
      UUID under ``uploads/<user_id>/`` by
      :func:`common.storage.user_media_path` — the client-supplied name never
      touches the filesystem.
    * ``file_path`` mirrors the SDS ``file_path`` column.  It is kept in sync
      with ``file.name`` on save so reports, the ML worker and any future
      out-of-band tooling can read the location straight from the row without
      needing the Django storage API.
    """

    media_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name='media ID',
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name='media',
        on_delete=models.CASCADE,
    )

    filename = models.CharField(
        max_length=255,
        help_text='Original client-supplied filename, for display only.',
    )
    format = models.CharField(max_length=10, choices=FORMAT_CHOICES)
    media_type = models.CharField(max_length=10, choices=MEDIA_TYPE_CHOICES)
    size_bytes = models.BigIntegerField()
    upload_timestamp = models.DateTimeField(auto_now_add=True)

    file = models.FileField(upload_to=user_media_path, max_length=500)
    file_path = models.CharField(
        max_length=500,
        blank=True,
        help_text='Storage-relative path; mirrors file.name.',
    )

    # Probed once at upload time so the UI and the pipeline never have to
    # re-open the file just to show dimensions or a duration.
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    duration_seconds = models.FloatField(
        null=True,
        blank=True,
        help_text='Videos only; null for stills.',
    )

    checksum = models.CharField(
        max_length=64,
        blank=True,
        db_index=True,
        help_text='SHA-256 of the file contents; identifies re-uploads.',
    )

    class Meta:
        db_table = 'uploads_media'
        ordering = ('-upload_timestamp',)
        indexes = [
            models.Index(fields=['user', 'upload_timestamp'],
                         name='media_user_uploaded_idx'),
        ]
        verbose_name = 'uploaded media'
        verbose_name_plural = 'uploaded media'

    def __str__(self):
        return f'{self.filename} ({self.media_type})'

    def save(self, **kwargs):
        """
        Persist the row, then reconcile ``file_path`` with the stored name.

        The final storage name is only known *after* ``FileField.pre_save`` has
        run (it is the point at which collisions are resolved), which happens
        inside ``super().save()``.  So the mirror column is written in a second,
        narrowly-scoped ``update_fields`` save — and only when it actually
        changed, so ordinary updates stay single-query.
        """
        super().save(**kwargs)

        stored_name = self.file.name if self.file else ''
        if stored_name and self.file_path != stored_name:
            self.file_path = stored_name
            super().save(update_fields=['file_path'])

    # -- convenience --------------------------------------------------------

    @property
    def is_video(self):
        """True when this upload is a clip rather than a still image."""
        return self.media_type == MEDIA_TYPE_VIDEO

    @property
    def size_mb(self):
        """File size in megabytes, rounded for display."""
        return round((self.size_bytes or 0) / (1024 * 1024), 2)

    @property
    def resolution(self):
        """``'1920x1080'``, or ``None`` when the media was never probed."""
        if self.width and self.height:
            return f'{self.width}x{self.height}'
        return None
