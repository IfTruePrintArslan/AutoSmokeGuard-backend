"""
URL routes for the analysis API, mounted under ``/api/`` by ``config.urls``.

The paths are written exactly as the frozen API contract spells them, which
means **no trailing slashes**.  Django's ``APPEND_SLASH`` only ever adds a
slash, never removes one, so registering the canonical no-slash form is what
makes ``/api/analyze`` work without a redirect — and a redirect would be
fatal here, because browsers drop the ``Authorization`` header and the body
of a POST when they follow a 301.

``status/<job_id>`` deliberately uses the ``uuid`` converter rather than
``str``: a malformed id then 404s at the routing layer instead of reaching
the database as a bad query, and the view never has to parse anything.
"""
from django.urls import path

from .views import (
    AnalysisDetailView,
    AnalysisListView,
    AnalysisReportView,
    AnalyzeView,
    DashboardStatsView,
    HistoryView,
    StatusView,
)

urlpatterns = [
    # -- running a job --------------------------------------------------
    path('analyze', AnalyzeView.as_view(), name='analysis-analyze'),
    path('status/<uuid:job_id>', StatusView.as_view(), name='analysis-status'),

    # -- results --------------------------------------------------------
    path('analysis', AnalysisListView.as_view(), name='analysis-list'),
    path('analysis/<uuid:analysis_id>', AnalysisDetailView.as_view(),
         name='analysis-detail'),
    path('analysis/<uuid:analysis_id>/report', AnalysisReportView.as_view(),
         name='analysis-report'),

    # -- screens --------------------------------------------------------
    path('history', HistoryView.as_view(), name='analysis-history'),
    path('dashboard/stats', DashboardStatsView.as_view(),
         name='dashboard-stats'),
]
