"""
Health routes.

Mounted twice by ``config.urls`` — at ``/api/health/`` (canonical) and at
``/health/`` (the Sprint-1 path, kept alive for existing deployment probes).
"""
from django.urls import path

from .views import health_check

urlpatterns = [
    path('', health_check, name='health-check'),
]
