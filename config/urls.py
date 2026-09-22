"""
Root URL configuration for the AutoSmokeGuard backend.

Everything the front end talks to lives under ``/api/``.  Each feature app owns
its own ``urls.py`` and is mounted flat under that prefix (the apps namespace
their own routes, e.g. ``auth/login``, ``uploads/``, ``analysis/<id>/``), which
keeps this file a one-line-per-app table of contents.

Reference: https://docs.djangoproject.com/en/6.0/topics/http/urls/
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

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
