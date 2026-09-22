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

``ApiErrorEnvelopeMiddleware`` is the last line of defence for the project's
one-error-shape promise: it rewrites an HTML 404 under ``/api/`` into the
standard JSON envelope (robustness finding §B).
"""
import json
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


class ApiErrorEnvelopeMiddleware:
    """
    Guarantee the JSON envelope for ``/api/`` 404s (finding ASG-R02 / §B).

    The problem
    -----------
    Six routes use Django's ``<uuid:...>`` path converter
    (``/api/status/<job_id>``, ``/api/media/<media_id>``,
    ``/api/analysis/<analysis_id>``, ``/api/analysis/<analysis_id>/report``,
    ``/api/report/<report_id>``, ``/api/download-report/<report_id>``).  A
    malformed id fails at the *URL resolver*, so no DRF view ever runs and
    ``common.exceptions.api_exception_handler`` — which DRF only calls from
    inside ``APIView.dispatch`` — never sees it.  The caller gets Django's
    HTML 404 instead of ``{detail, code, errors}``, breaking the contract
    every other failure in the project honours.

    Why this is not just ``handler404``
    -----------------------------------
    ``config.urls.handler404`` fixes it — but only with ``DEBUG=False``.
    ``django.core.handlers.exception.response_for_exception`` short-circuits
    to ``django.views.debug.technical_404_response`` whenever ``DEBUG`` is on,
    and that page lists *every registered URL pattern in the project* to an
    unauthenticated caller.  ``DEBUG=True`` is the default for a fresh clone
    (no ``SECRET_KEY`` set), so leaving it unhandled means the common case
    both violates the contract and discloses the routing table.  Rewriting
    the response covers both settings with one mechanism.

    What it deliberately does not touch
    -----------------------------------
    * **Anything outside ``/api/``** — ``/admin/``, ``/media/`` and ``/static/``
      are consumed by a browser, not by the SPA's error renderer, and Django's
      HTML 404 is the right answer there.
    * **Responses DRF produced** (identified by ``accepted_renderer``, which
      DRF sets on every ``Response`` it renders).  A DRF 404 is already the
      envelope, and converting a ``BrowsableAPIRenderer`` page to JSON would
      break the browsable API for no gain.
    * **Anything already JSON**, and anything streaming.
    * **5xx.**  ``handler500`` owns those.  Rewriting them here would destroy
      the ``DEBUG=True`` traceback page, which is a debugging tool, not a
      contract violation — DRF already catches every exception raised inside
      a view, so a 500 that reaches this middleware is by definition outside
      the API surface.

    Ordering
    --------
    Registered directly after ``MediaGuardMiddleware``, i.e. near the top of
    ``MIDDLEWARE``, so its response phase runs *last*: ``CommonMiddleware``
    has already had its chance to turn a 404 into an ``APPEND_SLASH``
    redirect (``/api/health`` -> ``/api/health/``), and ``CorsMiddleware``
    has already attached its headers.  The response is mutated in place
    rather than replaced so none of those headers — nor
    ``X-Response-Time-ms`` — is lost.
    """

    #: Only requests under this prefix are subject to the envelope contract.
    API_PREFIX = '/api/'

    #: The body written in place of the HTML.  Matches what DRF's handler
    #: emits for a 404 so the front end cannot tell the two apart.
    NOT_FOUND_BODY = {'detail': 'Not found.', 'code': 'not_found',
                      'errors': None}

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if self._needs_envelope(request, response):
            self._rewrite(response, self.NOT_FOUND_BODY)
        return response

    def _needs_envelope(self, request, response):
        """True when ``response`` breaks the envelope contract for ``request``."""
        if not str(getattr(request, 'path', '')).startswith(self.API_PREFIX):
            return False
        if getattr(response, 'status_code', None) != 404:
            return False
        if getattr(response, 'streaming', False):
            return False
        # DRF rendered it: already the envelope (or the browsable API).
        if getattr(response, 'accepted_renderer', None) is not None:
            return False
        content_type = (response.headers.get('Content-Type') or '').lower()
        return not content_type.startswith('application/json')

    @staticmethod
    def _rewrite(response, body):
        """Replace the payload in place, preserving every existing header."""
        payload = json.dumps(body).encode('utf-8')
        try:
            response.content = payload
        except Exception:  # pragma: no cover - non-content response class
            return
        response.headers['Content-Type'] = 'application/json'
        if response.has_header('Content-Length'):
            response.headers['Content-Length'] = str(len(payload))
