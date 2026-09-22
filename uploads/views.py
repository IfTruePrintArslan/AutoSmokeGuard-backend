"""
Views for the uploads API: create, list, retrieve, delete.

Kept intentionally thin — every rule that can fail (format policy, size
ceiling, decode integrity, de-duplication) lives in :mod:`uploads.services`;
these views parse the request, enforce *who can see what* (ownership /
admin-sees-all), and translate results into the contract's response shapes.
"""
import logging
import time

from django.apps import apps
from django.db.models import Prefetch
from drf_spectacular.utils import OpenApiParameter, OpenApiTypes, extend_schema
from rest_framework import generics, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from common.exceptions import error_response
from system_config.models import SystemSetting

from .models import MEDIA_TYPE_CHOICES, UploadedMedia
from .serializers import UploadedMediaSerializer, UploadSerializer
from .services import UploadError, store_upload

logger = logging.getLogger('asg.uploads')

#: Statuses that block deleting the parent media (see ``AnalysisResult`` in
#: the analysis app — mirrored here as plain strings, not imported, so this
#: module never has to import that app at load time).
_ANALYSIS_BLOCKING_STATUSES = ('queued', 'running')

_VALID_MEDIA_TYPES = {choice for choice, _label in MEDIA_TYPE_CHOICES}


def _is_admin(user):
    """True for an authenticated administrator (role or Django staff flag)."""
    return bool(user and user.is_authenticated and getattr(user, 'is_admin', False))


def _owned_queryset(user):
    """Base queryset for ``user``: everything for an admin, own rows only otherwise."""
    if _is_admin(user):
        return UploadedMedia.objects.all()
    return UploadedMedia.objects.filter(user=user)


def _with_latest_analysis(queryset):
    """
    Attach each row's analyses (newest first) to ``_analyses_ordered``.

    A single extra query for the whole page/detail, rather than one query per
    row — the N+1 the contract calls out.  The analysis model is looked up
    lazily through the app registry so this module never imports
    ``analysis.models`` directly, avoiding an import cycle with the app
    another agent is actively editing.
    """
    AnalysisResult = apps.get_model('analysis', 'AnalysisResult')
    return queryset.prefetch_related(
        Prefetch(
            'analyses',
            queryset=AnalysisResult.objects.order_by('-created_at'),
            to_attr='_analyses_ordered',
        ),
    )


# ---------------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------------


class UploadView(APIView):
    """``POST /api/upload`` — accept one image or video for later analysis."""

    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        summary='Upload media',
        description=(
            "Upload a single image or video for emission analysis. Accepts "
            "the multipart field 'file' ('media' is accepted as an alias). "
            "The owner is always the authenticated caller — a client-"
            "supplied user id is ignored. Returns 201 for a new file, or "
            "200 with `deduplicated: true` when the same bytes were already "
            "uploaded by this user."
        ),
        request={'multipart/form-data': UploadSerializer},
        responses={
            201: UploadedMediaSerializer,
            200: UploadedMediaSerializer,
            400: OpenApiTypes.OBJECT,
        },
    )
    def post(self, request, *args, **kwargs):
        django_file = request.FILES.get('file') or request.FILES.get('media')
        if django_file is None:
            logger.warning('upload rejected user=%s code=no_file', request.user.pk)
            return error_response(
                'No file was supplied.', 'no_file', status.HTTP_400_BAD_REQUEST,
            )

        settings_row = SystemSetting.get_solo()
        started = time.perf_counter()
        try:
            media, created = store_upload(request.user, django_file, settings_row)
        except UploadError as exc:
            return error_response(exc.message, exc.code, exc.status_code)

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            'upload request handled user=%s media_id=%s created=%s in %.1f ms',
            request.user.pk, media.media_id, created, elapsed_ms,
        )

        payload = UploadedMediaSerializer(media, context={'request': request}).data
        if created:
            return Response(payload, status=status.HTTP_201_CREATED)

        payload = dict(payload)
        payload['deduplicated'] = True
        return Response(payload, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# GET /api/media
# ---------------------------------------------------------------------------


class MediaListView(generics.ListAPIView):
    """``GET /api/media`` — the caller's uploads (or everyone's, for an admin)."""

    serializer_class = UploadedMediaSerializer

    @extend_schema(
        summary='List media',
        description="The caller's own uploads, newest first; an admin sees "
                    "every user's uploads.",
        parameters=[
            OpenApiParameter('page', OpenApiTypes.INT, description='Page number.'),
            OpenApiParameter('page_size', OpenApiTypes.INT,
                              description='Results per page (max 100).'),
            OpenApiParameter('media_type', OpenApiTypes.STR,
                              enum=sorted(_VALID_MEDIA_TYPES),
                              description='Filter by image or video.'),
        ],
        responses={200: UploadedMediaSerializer},
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            return UploadedMedia.objects.none()

        queryset = _owned_queryset(self.request.user)

        media_type = self.request.query_params.get('media_type')
        if media_type in _VALID_MEDIA_TYPES:
            queryset = queryset.filter(media_type=media_type)

        queryset = queryset.select_related('user')
        return _with_latest_analysis(queryset).order_by('-upload_timestamp')


# ---------------------------------------------------------------------------
# GET / DELETE /api/media/{media_id}
# ---------------------------------------------------------------------------


class MediaDetailView(generics.RetrieveDestroyAPIView):
    """``GET`` / ``DELETE /api/media/{media_id}``."""

    serializer_class = UploadedMediaSerializer
    lookup_field = 'media_id'
    lookup_url_kwarg = 'media_id'

    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            return UploadedMedia.objects.none()

        queryset = _owned_queryset(self.request.user).select_related('user')
        return _with_latest_analysis(queryset)

    @extend_schema(summary='Get media detail', responses={200: UploadedMediaSerializer})
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)

    @extend_schema(
        summary='Delete media',
        responses={204: None, 409: OpenApiTypes.OBJECT},
    )
    def delete(self, request, *args, **kwargs):
        return super().delete(request, *args, **kwargs)

    def destroy(self, request, *args, **kwargs):
        media = self.get_object()

        AnalysisResult = apps.get_model('analysis', 'AnalysisResult')
        blocked = AnalysisResult.objects.filter(
            media=media, status__in=_ANALYSIS_BLOCKING_STATUSES,
        ).exists()
        if blocked:
            return error_response(
                'This media has an analysis in progress and cannot be '
                'deleted.',
                'analysis_in_progress',
                status.HTTP_409_CONFLICT,
            )

        media_id = media.media_id
        file_field = media.file
        media.delete()

        if file_field:
            try:
                if file_field.storage.exists(file_field.name):
                    file_field.delete(save=False)
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.warning('Could not remove file for deleted media %s', media_id)

        logger.info('media deleted user=%s media_id=%s', request.user.pk, media_id)
        return Response(status=status.HTTP_204_NO_CONTENT)
