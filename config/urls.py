"""
Root URL configuration for the AutoSmokeGuard backend.

Everything the front end talks to lives under ``/api/``.  Each feature app owns
its own ``urls.py`` and is mounted flat under that prefix (the apps namespace
their own routes, e.g. ``auth/login``, ``uploads/``, ``analysis/<id>/``), which
keeps this file a one-line-per-app table of contents.

Error handlers
--------------
``handler404`` / ``handler500`` below keep the ``{detail, code, errors}``
envelope intact for failures that happen *outside* a DRF view, where
``common.exceptions.api_exception_handler`` is never consulted — most
importantly a malformed value in a ``<uuid:...>`` path converter, which fails
at the URL resolver before any view runs (robustness finding §B).  They are
scoped to ``/api/``; ``/admin/``, ``/media/`` and everything else keep
Django's normal HTML behaviour, which is what a browser fetching those
actually wants.

Reference: https://docs.djangoproject.com/en/6.0/topics/http/urls/
Reference: https://docs.djangoproject.com/en/6.0/ref/urls/#handler404
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path
from django.views import defaults as default_views
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

#: Requests under this prefix must always be answered with the JSON envelope.
API_PREFIX = '/api/'

urlpatterns = [
    path('admin/', admin.site.urls),

    # ---------------------------------------------------------------------
    # API documentation — machine-readable schema plus a browsable Swagger UI.
    # Both are AllowAny (see SPECTACULAR_SETTINGS / the view kwargs below) so
    # the front-end team can read them without minting a token first.
    # ---------------------------------------------------------------------
    path('api/schema/', SpectacularAPIView.as_view(
        permission_classes=[], authentication_classes=[],
    ), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(
        url_name='schema', permission_classes=[], authentication_classes=[],
    ), name='swagger-ui'),

    # ---------------------------------------------------------------------
    # Health probe.  '/health/' is the legacy Sprint-1 path and is registered
    # first so that reverse('health-check') resolves to the canonical
    # '/api/health/' form used by the front end and by the deployment probes.
    # ---------------------------------------------------------------------
    path('health/', include('health.urls')),
    path('api/health/', include('health.urls')),

    # ---------------------------------------------------------------------
    # Feature apps.  These modules are owned by other agents; the foundation
    # layer only guarantees that they exist and import cleanly.
    # ---------------------------------------------------------------------
    path('api/', include('accounts.urls')),
    path('api/', include('uploads.urls')),
    path('api/', include('analysis.urls')),
    path('api/', include('reports.urls')),
    path('api/', include('system_config.urls')),
]

# In development Django itself serves MEDIA_ROOT (uploaded media, annotated
# previews, smoke masks, generated PDFs).  In production this is the web
# server's / object store's job and the helper below returns an empty list.
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


# ---------------------------------------------------------------------------
# Error handlers
#
# Django only consults these when ``DEBUG`` is off — with ``DEBUG=True`` a 404
# becomes ``django.views.debug.technical_404_response`` and a 500 becomes the
# technical traceback page, neither of which goes anywhere near ``handler404``
# / ``handler500`` (see ``django.core.handlers.exception.response_for_
# exception``).  ``common.middleware.ApiErrorEnvelopeMiddleware`` is what
# covers the DEBUG case for 404s, including the URL-pattern listing that the
# technical page would otherwise show an unauthenticated caller.
# ---------------------------------------------------------------------------

def _is_api_request(request):
    """True for a request the JSON envelope contract applies to."""
    return str(getattr(request, 'path', '')).startswith(API_PREFIX)


def api_not_found(request, exception=None):
    """``handler404`` — the envelope for ``/api/``, Django's page elsewhere."""
    if not _is_api_request(request):
        return default_views.page_not_found(request, exception)
    return JsonResponse(
        {'detail': 'Not found.', 'code': 'not_found', 'errors': None},
        status=404,
    )


def api_server_error(request):
    """``handler500`` — the envelope for ``/api/``, Django's page elsewhere."""
    if not _is_api_request(request):
        return default_views.server_error(request)
    # Deliberately identical to common.exceptions.SERVER_ERROR_BODY: a caller
    # must not be able to tell whether a fault was caught by DRF or escaped
    # it, and nothing about the failure may leak.
    return JsonResponse(
        {'detail': 'Internal Server Error', 'code': 'server_error',
         'errors': None},
        status=500,
    )


handler404 = 'config.urls.api_not_found'
handler500 = 'config.urls.api_server_error'
