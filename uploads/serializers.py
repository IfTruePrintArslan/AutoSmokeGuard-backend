"""
Serializers for the uploads API.

Two shapes, matching the API contract's split between what a client sends and
what it gets back:

* :class:`UploadSerializer` — the write side.  A single ``file`` field; DRF's
  ``FileField`` is enough since format/size validation is real business logic
  (it depends on the live ``SystemSetting`` policy) and lives in
  :mod:`uploads.services`, not here.
* :class:`UploadedMediaSerializer` — the read side (``MediaObj`` in the
  contract), used by every response: the upload itself, the list, and the
  detail view.
"""
from django.apps import apps
from drf_spectacular.utils import extend_schema_field
from rest_framework import serializers

from common.storage import media_url

from .models import UploadedMedia


class UploadSerializer(serializers.Serializer):
    """
    Write-side serializer for ``POST /api/upload``.

    Only used to describe the multipart request body to drf-spectacular and
    to give the view a typed object to validate against; the field is read
    directly off ``request.FILES`` beforehand so the ``file`` / ``media``
    alias can be resolved before validation runs.
    """

    file = serializers.FileField(
        help_text="The image or video to upload. Field name 'file' "
                  "('media' is also accepted as an alias).",
    )


class LatestAnalysisSerializer(serializers.Serializer):
    """Shape of ``MediaObj.latest_analysis`` — documentation only, never bound."""

    analysis_id = serializers.UUIDField()
    status = serializers.CharField()
    progress = serializers.IntegerField()


class UploadedMediaSerializer(serializers.ModelSerializer):
    """Read-side serializer — the contract's ``MediaObj``."""

    url = serializers.SerializerMethodField()
    latest_analysis = serializers.SerializerMethodField()

    class Meta:
        model = UploadedMedia
        fields = (
            'media_id', 'filename', 'format', 'media_type', 'size_bytes',
            'upload_timestamp', 'width', 'height', 'duration_seconds',
            'url', 'latest_analysis',
        )
        read_only_fields = fields

    @extend_schema_field(serializers.URLField(allow_null=True))
    def get_url(self, obj):
        """Public URL for the stored file, or ``None`` if never saved."""
        return media_url(obj.file_path or (obj.file.name if obj.file else None))

    @extend_schema_field(LatestAnalysisSerializer(allow_null=True))
    def get_latest_analysis(self, obj):
        """
        The most recently created analysis for this upload, or ``None``.

        Views that list or retrieve media attach a prefetched, already
        ``-created_at``-ordered list to ``obj._analyses_ordered`` (see
        ``uploads.views``) so this never issues a per-row query. Falling back
        to a direct query keeps this serializer correct even when used on its
        own — e.g. straight after ``POST /api/upload``, where there cannot be
        an analysis yet but the shape must still be right.

        The analysis model is imported lazily via ``django.apps.apps`` to
        avoid an import-time cycle with the analysis app, which is being
        built concurrently.
        """
        prefetched = getattr(obj, '_analyses_ordered', None)
        if prefetched is not None:
            latest = prefetched[0] if prefetched else None
        else:
            AnalysisResult = apps.get_model('analysis', 'AnalysisResult')
            latest = (
                AnalysisResult.objects
                .filter(media=obj)
                .order_by('-created_at')
                .first()
            )

        if latest is None:
            return None

        return {
            'analysis_id': latest.analysis_id,
            'status': latest.status,
            'progress': latest.progress,
        }
