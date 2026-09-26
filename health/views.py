"""
Liveness / readiness probe.

Sprint 1 shipped a bare ``{"status": "ok"}``.  That is enough to prove the WSGI
worker is breathing but says nothing about whether the service can actually do
its job, so the payload now also reports the three dependencies that can fail
independently of Django itself:

* **database** — a real ``SELECT 1``, not just "the settings parsed".
* **ml_assets** — are the YOLO and U-Net weight files present on disk?  A
  deployment that forgot to mount the model volume looks perfectly healthy
  until the first upload; this catches it at boot.
* **worker** — is background analysis enabled in this process?

The endpoint is deliberately unauthenticated and unthrottled: container
orchestrators, uptime monitors and the front end's "backend reachable?" banner
all need it, and none of them can carry a JWT.  It leaks nothing beyond
booleans and a version string.
"""
from django.conf import settings
from django.db import connections
from django.utils import timezone
from drf_spectacular.utils import extend_schema
from rest_framework import status as http_status
from rest_framework.decorators import (
    api_view,
    authentication_classes,
    permission_classes,
    throttle_classes,
)
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

SERVICE_NAME = 'autosmokeguard-backend'


def _database_status():
    """Return 'ok' if the default database answers a trivial query."""
    try:
        with connections['default'].cursor() as cursor:
            cursor.execute('SELECT 1')
            cursor.fetchone()
        return 'ok'
    except Exception:
        # Any driver/connection error at all means "not ready". The detail is
        # already on the ERROR log via django.db; the probe stays opaque.
        return 'error'


def _ml_asset_status():
    """Report which model weight files are actually present on disk."""
    asg = getattr(settings, 'ASG', {})
    yolo = asg.get('YOLO_WEIGHTS')
    segmenter = asg.get('SEGMENTER_WEIGHTS')
    return {
        'yolo': bool(yolo) and yolo.exists(),
        'segmenter': bool(segmenter) and segmenter.exists(),
    }


@extend_schema(
    summary='Service health',
    description='Unauthenticated liveness/readiness probe. Returns 200 when '
                'the service and its database are usable, 503 otherwise.',
    auth=[],
    responses={200: dict, 503: dict},
)
@api_view(['GET'])
@authentication_classes([])
@permission_classes([AllowAny])
@throttle_classes([])
def health_check(request):
    """
    Liveness probe — confirms the service is up, reachable and wired together.

    Answers HTTP 200 with ``status: 'ok'`` when everything required to serve
    traffic is available, or HTTP 503 with ``status: 'degraded'`` when the
    database is unreachable.  Missing ML weights are reported but do **not**
    fail the probe: the API (auth, history, report download) still works
    without them, only new analyses would not.
    """
    database = _database_status()
    asg = getattr(settings, 'ASG', {})

    payload = {
        'status': 'ok' if database == 'ok' else 'degraded',
        'service': SERVICE_NAME,
        'version': settings.SPECTACULAR_SETTINGS.get('VERSION', '1.0.0'),
        'database': database,
        'ml_assets': _ml_asset_status(),
        'worker': 'enabled' if asg.get('WORKER_ENABLED') else 'disabled',
        'time': timezone.now().isoformat(),
    }

    code = (
        http_status.HTTP_200_OK if database == 'ok'
        else http_status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return Response(payload, status=code)
