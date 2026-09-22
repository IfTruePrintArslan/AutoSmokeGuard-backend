"""
URL routes for the uploads API, mounted under ``/api/`` by ``config.urls``.

Paths are the exact, flat strings the API contract freezes (``/api/upload``,
``/api/media``, ``/api/media/{media_id}``) — no trailing slashes, so a client
following the contract literally is never redirected.
"""
from django.urls import path

from .views import MediaDetailView, MediaListView, UploadView

urlpatterns = [
    path('upload', UploadView.as_view(), name='media-upload'),
    path('media', MediaListView.as_view(), name='media-list'),
    path('media/<uuid:media_id>', MediaDetailView.as_view(), name='media-detail'),
]
