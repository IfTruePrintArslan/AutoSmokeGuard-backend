"""
Business logic for the accounts API.

Kept separate from ``accounts.views`` so the token/reset mechanics — minting
a JWT pair, issuing and consuming a password-reset token, revoking every
refresh token a user holds — can be read (and reused) without any DRF
request/response plumbing in the way.  Views stay thin: parse input, call
one of these, shape the response.
"""
import hashlib
import logging

from rest_framework_simplejwt.token_blacklist.models import (
    BlacklistedToken,
    OutstandingToken,
)
from rest_framework_simplejwt.tokens import RefreshToken

from .models import PasswordResetToken

logger = logging.getLogger('asg.accounts')


# ---------------------------------------------------------------------------
# Client metadata
# ---------------------------------------------------------------------------

def get_client_ip(request):
    """
    Best-effort caller address for the audit log.

    Prefers the first hop recorded in ``X-Forwarded-For`` (the address a
    reverse proxy/load balancer saw) and falls back to the socket peer
    address Django itself observed.
    """
    forwarded = request.META.get('HTTP_X_FORWARDED_FOR')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', '')


# ---------------------------------------------------------------------------
# JWT issuance / revocation
# ---------------------------------------------------------------------------

def issue_tokens(user):
    """
    Mint a fresh access/refresh pair for ``user``.

    Used by register and login so the client walks away authenticated in
    the same round trip that created/verified the account — no separate
    login call needed straight after signing up.
    """
    refresh = RefreshToken.for_user(user)
    return {
        'access': str(refresh.access_token),
        'refresh': str(refresh),
    }


def revoke_all_refresh_tokens(user):
    """
    Blacklist every outstanding refresh token belonging to ``user``.

    Called after a password change: a stolen, or merely still-cached, old
    refresh token must not be able to keep a session alive once the
    password it was issued under is no longer valid.

    This is only *half* of "log everyone out", and on its own it was the
    review's finding ASG-08.  Blacklisting is a lookup the *refresh* endpoint
    performs; an access token is never looked up anywhere, so one already in
    flight stayed valid until its own ``exp`` — up to 30 more minutes of full
    API access for the attacker the reset was meant to evict.

    The other half is ``accounts.User.password_changed_at``, stamped by
    ``User.set_password`` and enforced by
    ``accounts.authentication.PasswordChangeAwareJWTAuthentication``.  Both
    run in ``accounts.views.PasswordResetConfirmView``; changing one without
    the other reopens the hole, so they are cross-referenced here on purpose.
    """
    outstanding = OutstandingToken.objects.filter(user=user)
    revoked = 0
    for token in outstanding:
        _, created = BlacklistedToken.objects.get_or_create(token=token)
        if created:
            revoked += 1
    logger.info('Revoked %s outstanding refresh token(s) for %s',
               revoked, user.email)


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------

def create_reset_token(user):
    """
    Issue a new password-reset token for ``user``.

    Any tokens still outstanding for this user are invalidated first, so a
    user who requests several resets in a row only ever has one live link —
    a stale, previously issued token cannot be used after a newer one has
    been minted.

    Logging (security review finding ASG-05)
    ----------------------------------------
    The token itself is **never** written to the log.  A password-reset token
    is a live, single-use account-takeover credential for a full hour; writing
    it to ``backend/logs/backend.log`` put a working credential for every
    account that ever requested a reset into a plain file that outlives the
    token's TTL, is rotated rather than purged, and is routinely shipped to
    log aggregators and attached to bug reports.

    What is logged instead is a *reference*: the token row's UUID primary key
    plus a truncated SHA-256 of the secret.  That is enough to correlate "a
    reset was issued" with "that reset was consumed" when investigating an
    incident, and cannot be used to reset anybody's password.

    There is no mail server in this deployment; the delivery channel is the
    ``debug_token`` field the view returns while ``DEBUG`` is on (see
    ``accounts.views.PasswordResetRequestView``), not the log.
    """
    PasswordResetToken.objects.filter(user=user, used=False).update(used=True)
    token = PasswordResetToken.objects.create(user=user)
    logger.info(
        'Password reset token issued for %s (token_id=%s ref=%s)',
        user.email, token.token_id, token_reference(token.token),
    )
    return token


def token_reference(raw_token):
    """
    A short, non-reversible handle for a secret, safe to write to a log.

    Twelve hex characters of SHA-256 — enough to tie two log lines together
    when reconstructing an incident, far too little to recover the 256-bit
    token it refers to.
    """
    if not raw_token:
        return 'none'
    return hashlib.sha256(str(raw_token).encode('utf-8')).hexdigest()[:12]


def consume_reset_token(raw_token):
    """
    Validate ``raw_token`` and mark it used, returning the owning user.

    Raises ``PasswordResetToken.DoesNotExist`` for anything that is not a
    live, unused token within its TTL — unknown, expired and already-used
    tokens are all indistinguishable to the caller by design (the view
    answers all three with the same ``code="invalid_reset_token"``).
    """
    reset_token = PasswordResetToken.objects.select_related('user').get(
        token=raw_token,
    )

    if not reset_token.is_valid():
        # Logged as a reference only — never the token (finding ASG-05).
        logger.warning(
            'Rejected an expired or already-used password reset token '
            '(token_id=%s ref=%s)',
            reset_token.token_id, token_reference(raw_token),
        )
        raise PasswordResetToken.DoesNotExist(
            'Password reset token is expired or already used.',
        )

    reset_token.mark_used()
    logger.info(
        'Password reset token consumed for %s (token_id=%s ref=%s)',
        reset_token.user.email, reset_token.token_id,
        token_reference(raw_token),
    )
    return reset_token.user
