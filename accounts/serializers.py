"""
Serializers for the accounts API.

Kept deliberately thin: field-level validation (formats, password strength,
matching a model's writable surface) lives here.  Anything that needs a
side-effect tied directly to the security model — checking whether an e-mail
is already taken, authenticating a login attempt, resolving a password-reset
token — is done in ``accounts.views`` instead, because
``common.exceptions.api_exception_handler`` collapses *every*
``rest_framework.exceptions.ValidationError`` down to ``code=
"validation_error"``, and several of those flows need the specific machine
code the API contract promises (``email_exists``, ``invalid_credentials``,
``invalid_reset_token``, ``token_not_valid``).  Raising those through a
serializer would silently lose the code, so the views build those particular
responses by hand with ``common.exceptions.error_response`` instead.
"""
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from .models import User


# ---------------------------------------------------------------------------
# Read / update model
# ---------------------------------------------------------------------------

class UserSerializer(serializers.ModelSerializer):
    """
    The public ``UserObj`` shape from the API contract.

    Used to render ``GET /api/me`` (and the ``user`` key on register/login),
    and — via ``partial=True`` — to validate ``PATCH /api/me``.  Every field
    but ``full_name`` is marked read-only, so an attempt to smuggle
    ``role: "admin"`` (or ``email``, ``user_id``, ``created_at``) through the
    patch body is silently dropped rather than erroring: DRF never even
    looks at a read-only field's incoming value.
    """

    class Meta:
        model = User
        fields = ('user_id', 'email', 'full_name', 'role', 'created_at')
        read_only_fields = ('user_id', 'email', 'role', 'created_at')


# ---------------------------------------------------------------------------
# Register / login
# ---------------------------------------------------------------------------

class RegisterSerializer(serializers.Serializer):
    """
    ``POST /api/register`` input.

    ``role`` is deliberately not a field here — self-registration is always
    ``role="user"``, so there is nothing for a client-supplied role to do
    but be ignored, which not declaring the field guarantees.  The
    duplicate-email check itself lives in the view (see the module
    docstring) so it can answer with ``code="email_exists"``.
    """

    full_name = serializers.CharField(
        max_length=150, required=False, allow_blank=True, default='',
    )
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True, trim_whitespace=False)

    def validate_email(self, value):
        """Case-fold so ``Ali@X.com`` and ``ali@x.com`` collide on purpose."""
        return value.strip().lower()

    def validate_password(self, value):
        """Run every configured ``AUTH_PASSWORD_VALIDATORS`` check."""
        validate_password(value)
        return value


class LoginSerializer(serializers.Serializer):
    """``POST /api/login`` input — authentication itself happens in the view."""

    email = serializers.EmailField()
    password = serializers.CharField(write_only=True, trim_whitespace=False)


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

class RefreshSerializer(serializers.Serializer):
    """``POST /api/refresh-token`` input."""

    refresh = serializers.CharField()


class LogoutSerializer(serializers.Serializer):
    """
    ``POST /api/logout`` input.

    ``refresh`` is optional at this layer on purpose: signing out must
    return 200 even for a missing or garbage token (see the view), so
    nothing here should be able to turn that into a 400.
    """

    refresh = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, default='',
    )


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------

class PasswordResetRequestSerializer(serializers.Serializer):
    """``POST /api/password-reset`` input."""

    email = serializers.EmailField()

    def validate_email(self, value):
        return value.strip().lower()


class PasswordResetConfirmSerializer(serializers.Serializer):
    """``POST /api/password-reset/confirm`` input."""

    token = serializers.CharField()
    password = serializers.CharField(write_only=True, trim_whitespace=False)

    def validate_password(self, value):
        validate_password(value)
        return value
