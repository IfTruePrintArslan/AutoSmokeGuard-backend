"""
URL routes for the reports API, mounted under ``/api/`` by ``config.urls``.

Paths match the frozen API contract exactly — flat, verb-specific, and
without a trailing slash — matching the convention every other app in this
project uses (see ``accounts/urls.py``).
"""
from django.urls import path

from .views import ReportDetailView, ReportDownloadView, ReportListView

urlpatterns = [
    path('reports', ReportListView.as_view(), name='report-list'),
    path('report/<uuid:report_id>', ReportDetailView.as_view(), name='report-detail'),
    path('download-report/<uuid:report_id>', ReportDownloadView.as_view(), name='report-download'),
]
