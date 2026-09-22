"""
JWT authentication that honours a password change (security review ASG-08).

The problem
-----------
SimpleJWT access tokens are stateless bearer credentials: once signed they are
valid until their own ``exp``, and nothing on the server is consulted beyond
"is the signature good and is the user still active?".  ``accounts.services.
revoke_all_refresh_tokens`` blacklists every *refresh* token when a password is
reset, which stops the session being renewed — but the access token already in
the attacker's hands keeps working for the remainder of its 30-minute lifetime.

The review measured exactly that:

    before reset: old access -> HTTP 200
    password-reset/confirm   -> HTTP 200
    AFTER reset: old REFRESH -> HTTP 401   (correctly revoked)
    AFTER reset: old ACCESS  -> HTTP 200   <-- the finding

A user who resets their password *because* they think they are compromised
reasonably expects "log everyone out now".

The fix
-------
``accounts.User.set_password`` stamps ``password_changed_at``.  This
authentication class compares it against the token's ``iat`` (issued-at) claim
and refuses anything minted before the change.  It is the minimum viable amount
of server-side state — one nullable timestamp per user, read on a row the
authenticator already had to fetch, so it costs no extra query.

Deliberately *not* a cache denylist: this deployment runs the per-process
``LocMemCache``, so a denylist would be invisible to sibling gunicorn workers
and would silently fail open — the worst possible failure mode for a revocation
control.

The error code matters
----------------------
Rejections raise ``AuthenticationFailed`` with ``code='token_not_valid'``.
That exact string is load-bearing: the front end's single-flight refresh logic
(``frontend/src/lib/api.js``) keys off it to decide "try the refresh endpoint
once, then sign out", and the refresh will legitimately fail too (the refresh
chain was blacklisted by the same reset), so the user lands on the login screen
instead of a dead-end error.  Any other code would leave the SPA stuck.
"""
import logging
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone

from drf_spectacular.contrib.rest_framework_simplejwt import SimpleJWTScheme
# NOTE: DRF's AuthenticationFailed, deliberately, and not SimpleJWT's
# same-named subclass.  SimpleJWT's mixes in ``DetailDictMixin``, which turns
# the detail into a ``{"detail": ..., "code": ...}`` *dict*;
# ``common.exceptions`` then looks for the code on ``exc.detail`` (a dict has
# no ``.code``), falls back to ``default_code`` — ``'authentication_failed'``
# — and folds the dict into the envelope's ``errors`` map.  DRF's plain
# version produces an ``ErrorDetail`` that carries the code, which is what
# yields ``{"detail": ..., "code": "token_not_valid", "errors": null}``.
from rest_framework.exceptions import AuthenticationFailed
from rest_framework_simplejwt.authentication import JWTAuthentication

logger = logging.getLogger('asg.accounts')

#: Claim carrying the token's issue time, in whole seconds since the epoch.
IAT_CLAIM = 'iat'

#: Slack allowed between ``iat`` and ``password_changed_at``.
#:
#: Two independent reasons this cannot be zero:
#:
#: 1. ``iat`` is an integer number of seconds (RFC 7519 NumericDate), while
#:    ``password_changed_at`` is a microsecond-precision timestamp.  A token
#:    minted at 12:00:00.900 carries ``iat = 12:00:00``, so it would appear to
#:    predate a password change recorded at 12:00:00.800 — and the login
#:    immediately after a reset would 401 on its own brand-new token.
#: 2. A multi-process (or multi-host) deployment can have a small clock skew
#:    between the process that mints the token and the one that wrote the row.
#:
#: Two seconds covers both while leaving the control meaningful: the window in
#: which a pre-reset token survives is two seconds, not thirty minutes.
LEEWAY = timedelta(seconds=2)

#: Shown to the client. Deliberately generic — it must not distinguish
#: "your password changed" from any other invalid token, and the SPA renders
#: the code, not this sentence.
_REJECTED_DETAIL = 'Token is invalid or expired.'


class PasswordChangeAwareJWTAuthentication(JWTAuthentication):
    """
    ``JWTAuthentication`` plus a check that the token predates no password
    change.

    Wired up as ``REST_FRAMEWORK['DEFAULT_AUTHENTICATION_CLASSES']`` in
    ``config.settings``, so it covers every authenticated endpoint at once
    rather than being something each view has to remember to opt into.
    """

    def get_user(self, validated_token):
        """
        Resolve the token's user, then reject a token issued before the
        user's last password change.

        ``get_user`` is the right seam: the parent has already verified the
        signature, the expiry and the token type, and has just fetched the
        user row this check needs — so the check is free, and it runs for
        every caller of ``authenticate()`` without duplicating any of that.
        """
        user = super().get_user(validated_token)
        self.enforce_password_change(user, validated_token)
        return user

    # -- the check ----------------------------------------------------------

    def enforce_password_change(self, user, validated_token):
        """
        Raise ``AuthenticationFailed`` if ``validated_token`` predates
        ``user.password_changed_at``.

        Two "do nothing" cases, both intentional:

        * ``password_changed_at is None`` — every account that existed before
          the column was added (migration
          ``accounts/0002_user_password_changed_at``) has NULL here, and NULL
          means "no password change has ever been recorded", not "reject
          everything".  Backfilling it to ``now()`` in the migration would
          have signed every live user out at deploy time for no security
          benefit, so the column ships null and starts being populated the
          first time each user's password is set.
        * the user model has no such attribute at all — keeps this class
          usable against ``TokenUser``/a swapped user model without blowing
          up with an ``AttributeError`` that would surface as a 500.
        """
        changed_at = getattr(user, 'password_changed_at', None)
        if changed_at is None:
            return

        issued_at = self._issued_at(validated_token)

        if issued_at is None:
            # The token carries no usable ``iat``, so it cannot be shown to
            # postdate the password change. SimpleJWT always sets ``iat`` on
            # tokens it mints, so this is either a hand-crafted token or one
            # from a much older version of the library — fail closed, not
            # open. ``_reject`` always raises; the ``return`` is unreachable
            # and is here only so the control flow reads unambiguously.
            self._reject(user, 'the token carries no usable iat claim')
            return

        if issued_at < changed_at - LEEWAY:
            self._reject(
                user,
                f'issued {issued_at.isoformat()}, password changed '
                f'{changed_at.isoformat()}',
            )

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _issued_at(validated_token):
        """The token's ``iat`` as an aware datetime, or ``None``."""
        raw = validated_token.get(IAT_CLAIM)
        if raw is None:
            return None
        try:
            return datetime.fromtimestamp(int(raw), tz=dt_timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            # A non-numeric or absurd ``iat`` is a malformed token.
            return None

    @staticmethod
    def _reject(user, reason):
        """Log for incident response, then fail with the contract's code."""
        logger.warning(
            'Rejected an access token that predates a password change for '
            '%s (%s)', getattr(user, 'email', user), reason,
        )
        raise AuthenticationFailed(_REJECTED_DETAIL, code='token_not_valid')


# ---------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------

class PasswordChangeAwareJWTScheme(SimpleJWTScheme):
    """
    Teach drf-spectacular that the class above is still bearer-JWT auth.

    Extensions are matched on the *exact* class path and
    ``match_subclasses`` is off by default, so swapping
    ``DEFAULT_AUTHENTICATION_CLASSES`` to a subclass made spectacular emit
    ``W001: could not resolve authenticator`` for all sixteen authenticated
    views and silently drop the ``security`` block from the generated
    schema. That turned ``manage.py check --deploy`` from clean into sixteen
    warnings and would have shipped a schema that told clients the API needs
    no credentials.

    Registration is by ``__init_subclass__`` on ``OpenApiAuthenticationExtension``,
    which fires when this module is imported — and this module is imported
    the moment DRF resolves ``DEFAULT_AUTHENTICATION_CLASSES``, i.e. before
    any schema can be generated. Keeping it in the same file as the class it
    describes is therefore what guarantees the ordering.

    ``name = 'jwtAuth'`` is inherited unchanged, so the component name in the
    published schema is byte-identical to before — no client regenerates.
    """

    target_class = 'accounts.authentication.PasswordChangeAwareJWTAuthentication'
