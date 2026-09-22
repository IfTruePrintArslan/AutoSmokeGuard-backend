"""
One error shape for the whole API.

Every failure — a serializer rejection, a 401, a 404, a throttle, or an
unhandled crash — leaves the backend as the same JSON envelope::

    {
      "detail": "Human readable summary.",
      "code":   "machine_readable_code",
      "errors": {"field": ["message", ...]} | null
    }

The front end therefore only ever writes one error renderer.  Wired up through
``REST_FRAMEWORK['EXCEPTION_HANDLER']`` in ``config.settings``.
"""
import logging

from django.core.exceptions import ValidationError as DjangoValidationError
from django.http import Http404
from rest_framework import exceptions as drf_exceptions
from rest_framework import status
from rest_framework.response import Response
from rest_framework.serializers import as_serializer_error
from rest_framework.views import exception_handler as drf_exception_handler

logger = logging.getLogger('asg.api')

# Fallback code per status class, used only when the exception itself carries
# nothing more specific.
_STATUS_CODES = {
    status.HTTP_400_BAD_REQUEST: 'bad_request',
    status.HTTP_401_UNAUTHORIZED: 'not_authenticated',
    status.HTTP_403_FORBIDDEN: 'permission_denied',
    status.HTTP_404_NOT_FOUND: 'not_found',
    status.HTTP_405_METHOD_NOT_ALLOWED: 'method_not_allowed',
    status.HTTP_406_NOT_ACCEPTABLE: 'not_acceptable',
    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: 'payload_too_large',
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: 'unsupported_media_type',
    status.HTTP_429_TOO_MANY_REQUESTS: 'throttled',
}

SERVER_ERROR_BODY = {
    'detail': 'Internal Server Error',
    'code': 'server_error',
    'errors': None,
}

# Generic summary used when the payload is a map of per-field messages and the
# exception gave us no single sentence to show.
VALIDATION_DETAIL = 'Validation failed.'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_message_list(value):
    """
    Flatten an arbitrarily nested DRF error payload into a list of strings.

    DRF nests errors to mirror serializer structure (lists for many=True,
    dicts for nested serializers).  The envelope promises ``{field: [str]}``,
    so nested keys are folded into the message text as ``"child: message"``.
    """
    if isinstance(value, (list, tuple)):
        messages = []
        for item in value:
            messages.extend(_as_message_list(item))
        return messages
    if isinstance(value, dict):
        messages = []
        for key, nested in value.items():
            messages.extend(f'{key}: {msg}' for msg in _as_message_list(nested))
        return messages
    return [str(value)]


def _error_code(exc, status_code):
    """Pick the most specific machine-readable code available."""
    if isinstance(exc, drf_exceptions.ValidationError):
        return 'validation_error'

    # ErrorDetail (a str subclass) carries the code DRF raised it with.
    detail = getattr(exc, 'detail', None)
    code = getattr(detail, 'code', None)
    if isinstance(code, str) and code:
        return code

    code = getattr(exc, 'default_code', None)
    if isinstance(code, str) and code:
        return code

    return _STATUS_CODES.get(status_code, 'error')


def _split_payload(data):
    """
    Turn DRF's ``response.data`` into ``(detail, errors)``.

    ``detail`` is always a single human-readable sentence; ``errors`` is either
    a ``{field: [messages]}`` map or ``None`` when the failure is not
    field-specific.
    """
    if isinstance(data, dict):
        # The plain ``{'detail': '...'}`` shape used by every non-validation
        # APIException.  Anything alongside 'detail' is treated as field data.
        if 'detail' in data and len(data) == 1:
            return str(data['detail']), None

        errors = {
            str(field): _as_message_list(messages)
            for field, messages in data.items()
        }
        detail = str(data['detail']) if 'detail' in data else VALIDATION_DETAIL
        return detail, errors or None

    if isinstance(data, (list, tuple)):
        messages = _as_message_list(data)
        detail = messages[0] if messages else VALIDATION_DETAIL
        return detail, {'non_field_errors': messages} if messages else None

    return str(data), None


def error_response(detail, code, status_code=status.HTTP_400_BAD_REQUEST,
                   errors=None):
    """
    Build the standard envelope by hand.

    Handy inside a view when you want to fail with a specific code without
    raising (``return error_response('Upload too large.', 'file_too_large',
    413)``).
    """
    return Response(
        {'detail': detail, 'code': code, 'errors': errors},
        status=status_code,
    )


# ---------------------------------------------------------------------------
# The handler itself
# ---------------------------------------------------------------------------

def api_exception_handler(exc, context):
    """
    Project-wide DRF exception handler.

    1. Django's own ``ValidationError`` (raised by ``full_clean()`` or a model
       field validator) is converted to DRF's so model-level validation reads
       the same as serializer-level validation to the client.
    2. ``Http404`` and Django's ``PermissionDenied`` are converted by DRF's
       built-in handler, which we delegate to for status-code selection and
       for the transaction rollback it performs.
    3. Anything DRF does not recognise is a bug: it is logged with a full
       traceback under the ``asg.api`` logger and reported as an opaque 500 so
       internals never leak to the client.
    """
    if isinstance(exc, DjangoValidationError):
        exc = drf_exceptions.ValidationError(as_serializer_error(exc))

    response = drf_exception_handler(exc, context)

    if response is None:
        # Unhandled — a genuine server fault (or a bare Http404 raised outside
        # DRF's view machinery, which DRF has already normalised above).
        if isinstance(exc, Http404):
            return Response(
                {'detail': 'Not found.', 'code': 'not_found', 'errors': None},
                status=status.HTTP_404_NOT_FOUND,
            )

        view = context.get('view') if isinstance(context, dict) else None
        request = context.get('request') if isinstance(context, dict) else None
        logger.exception(
            'Unhandled exception in %s while serving %s %s',
            getattr(view, '__class__', type(view)).__name__ if view else 'API',
            getattr(request, 'method', '?'),
            getattr(request, 'path', '?'),
        )
        return Response(
            dict(SERVER_ERROR_BODY),
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    detail, errors = _split_payload(response.data)
    response.data = {
        'detail': detail,
        'code': _error_code(exc, response.status_code),
        'errors': errors,
    }
    return response
