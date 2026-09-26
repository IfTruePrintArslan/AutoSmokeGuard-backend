"""
URL routes for the system-configuration API, mounted under ``/api/`` by
``config.urls``.

One path, the exact contract string (``/api/settings``, no trailing slash) —
``GET``/``PATCH`` are dispatched by the same view since there is only ever
one configuration object.
"""
from django.urls import path

from .views import SystemSettingsView

urlpatterns = [
    path('settings', SystemSettingsView.as_view(), name='system-settings'),
]
