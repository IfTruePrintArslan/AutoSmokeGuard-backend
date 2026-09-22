"""
Request-level instrumentation and media-tree access control.

``RequestLogMiddleware`` is the last entry in ``MIDDLEWARE`` so the duration it
measures covers the entire downstream stack (CORS, sessions, auth, the view,
and every renderer).  Output goes to the ``asg.request`` logger, which
``config.settings.LOGGING`` sends to both the console and the rotating file at
``backend/logs/backend.log``.

``MediaGuardMiddleware`` sits in front of the ``/media/`` tree; see its
docstring for what it does and, just as importantly, what it deliberately does
not try to do.
"""
import logging
import posixpath
import time
from urllib.parse import unquote

from django.http import HttpResponseNotFound

logger = logging.getLogger('asg.request')
media_logger = logging.getLogger('asg.media')

# Paths that would otherwise flood the log with no diagnostic value.
IGNORED_PREFIXES = ('/static/', '/media/', '/favicon.ico')


class RequestLogMiddleware:
    """
    Log ``METHOD path -> status (duration_ms)`` for every non-static request.

    The log level tracks the response class so that ``grep WARNING`` finds the
    4xx traffic and ``grep ERROR`` finds the 5xx traffic:

    * 2xx / 3xx -> INFO
    * 4xx       -> WARNING
    * 5xx       -> ERROR
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.startswith(IGNORED_PREFIXES):
            return self.get_response(request)

        started = time.perf_counter()
        response = self.get_response(request)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)

        # Stamp the header too — the front end surfaces it in the dev console
        # when chasing slow analysis polling.
        response['X-Response-Time-ms'] = str(duration_ms)

        status_code = getattr(response, 'status_code', 0)
        if status_code >= 500:
            level = logging.ERROR
        elif status_code >= 400:
            level = logging.WARNING
        else:
            level = logging.INFO

        logger.log(
            level,
            '%s %s -> %s (%.2f ms) user=%s',
            request.method,
            request.get_full_path(),
            status_code,
            duration_ms,
            self._user_label(request),
        )
        return response

    @staticmethod
    def _user_label(request):
        """
        Identify the caller without touching the DB.

        AuthenticationMiddleware attaches a lazy ``request.user``; DRF's JWT
        authentication runs later still (inside the view), so for token traffic
        this is usually 'anonymous'.  That is fine — it is a log line, not an
        audit trail, and evaluating it must never raise.
        """
        user = getattr(request, 'user', None)
        try:
            if user is not None and user.is_authenticated:
                return getattr(user, 'email', None) or str(user.pk)
        except Exception:  # pragma: no cover - defensive; logging must not 500
            return 'unknown'
        return 'anonymous'


class MediaGuardMiddleware:
    """
    Access control for the ``/media/`` tree (security review finding ASG-04).

    The problem
    -----------
    Everything the pipeline writes for a user lives under ``MEDIA_ROOT`` and is
    served as a static file: by Django while ``DEBUG`` is on
    (``config.urls`` appends ``static(MEDIA_URL, ...)``), and by nginx in the
    container (``location ^~ /media/ { alias /media/; }``).  Neither path
    performs any authorization at all, so possession of the URL *is* the
    credential.  Verified live during the review: an unauthenticated ``GET``
    of another user's ``/media/uploads/<user_id>/<uuid>.jpg`` returns 200 with
    the footage, and ``/media/reports/<report_id>.pdf`` returns the full PDF.

    What this middleware does
    -------------------------
    1. **Blocks the ``reports/`` subtree outright.**  Generated PDFs are the
       most sensitive artefact in the system — they carry the whole analysis,
       the source filename and the operator's details — and *nothing* needs to
       fetch them as a static file.  Both the front end and the API contract
       use the authenticated ``GET /api/download-report/{report_id}``
       endpoint, which scopes to the owner and 404s otherwise.  Closing the
       static route therefore removes an unauthenticated copy of the data with
       no functional cost.
    2. **Rejects traversal and absolute-path tricks** in the media URL before
       any file handler sees them, including percent-encoded forms.
    3. **Logs** every blocked attempt to ``asg.media`` so the access-control
       failure is visible to an incident responder.

    What it deliberately does not do
    --------------------------------
    It does not authenticate ``uploads/`` or ``analyses/``.  Those are
    referenced by ``<img src=...>`` in the SPA, which cannot attach a Bearer
    header, so gating them needs signed expiring URLs or a cookie-scoped
    handler — a design change well beyond a review fix, and one that has to be
    made in the serializers that emit the URLs.  They remain capability URLs
    (unguessable UUIDv4 paths); see ``docs/SECURITY_REVIEW.md`` for the
    residual-risk statement and the production fix.

    It also only covers the Django-served path.  When nginx serves
    ``/media/`` directly the request never reaches Django, so the production
    deployment needs the equivalent nginx rule — documented in the review.
    """

    #: Media subtrees that must never be served as plain static files.
    BLOCKED_PREFIXES = ('reports/',)

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        from django.conf import settings

        media_url = getattr(settings, 'MEDIA_URL', '/media/') or '/media/'
        if not media_url.endswith('/'):
            media_url += '/'

        path = request.path
        if path.startswith(media_url):
            relative = path[len(media_url):]
            denial = self._denial_reason(relative)
            if denial is not None:
                media_logger.warning(
                    'Blocked %s %s from %s (%s)',
                    request.method, path,
                    request.META.get('REMOTE_ADDR', '?'), denial,
                )
                # 404, not 403: a 403 would confirm the path exists.
                return HttpResponseNotFound()

        return self.get_response(request)

    @classmethod
    def _denial_reason(cls, relative_path):
        """
        Why ``relative_path`` may not be served, or ``None`` when it may be.

        ``relative_path`` is the portion after ``MEDIA_URL``.  It is decoded
        once and normalised before the checks, so ``..%2f``, ``%2e%2e/`` and a
        bare ``../`` are all treated the same way.
        """
        decoded = unquote(relative_path or '')

        # Reject a path that escapes (or tries to escape) the media root.
        if decoded.startswith('/') or decoded.startswith('\\'):
            return 'absolute path'
        if '\x00' in decoded:
            return 'null byte'

        normalised = posixpath.normpath(decoded.replace('\\', '/'))
        if normalised == '..' or normalised.startswith('../'):
            return 'path traversal'

        lowered = normalised.lstrip('./').lower()
        for prefix in cls.BLOCKED_PREFIXES:
            if lowered == prefix.rstrip('/') or lowered.startswith(prefix):
                return f'{prefix} is not publicly served'

        return None
