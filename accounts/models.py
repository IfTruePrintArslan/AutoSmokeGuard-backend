"""
Identity for AutoSmokeGuard.

The project replaces Django's default ``auth.User`` with a UUID-keyed, e-mail
addressed model (``AUTH_USER_MODEL = 'accounts.User'``).  Three reasons:

* the SDS schema keys every table on a descriptive UUID (``user_id``,
  ``media_id``, ``analysis_id``, ...), and sequential integer ids would leak
  how many users the system has and make ids guessable;
* there is no username in the product — people sign in with an e-mail;
* the two-value ``role`` column drives the admin-only configuration screens
  (UC-09) without dragging in Django's group/permission UI.

Views, serializers and URL routing for these models are owned by the accounts
API agent; this module is the schema only.
"""
import secrets
import uuid
from datetime import timedelta

from django.conf import settings
from django.contrib.auth.models import (
    AbstractBaseUser,
    BaseUserManager,
    PermissionsMixin,
)
from django.db import models
from django.utils import timezone

# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

ROLE_USER = 'user'
ROLE_ADMIN = 'admin'

ROLE_CHOICES = (
    (ROLE_USER, 'User'),
    (ROLE_ADMIN, 'Administrator'),
)

#: How long a password-reset link stays usable.
PASSWORD_RESET_TTL = timedelta(hours=1)


def default_reset_expiry():
    """Expiry timestamp for a freshly minted password-reset token."""
    return timezone.now() + PASSWORD_RESET_TTL


def generate_reset_token():
    """
    A 64-character, cryptographically random hex token.

    ``secrets.token_hex(32)`` gives 32 bytes / 256 bits of entropy, which is
    far beyond brute-forcing inside the one-hour window above.
    """
    return secrets.token_hex(32)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class UserManager(BaseUserManager):
    """
    Manager for the e-mail-addressed :class:`User` model.

    ``use_in_migrations`` is on so data migrations and ``createsuperuser`` can
    both go through the same normalisation and hashing path.
    """

    use_in_migrations = True

    def create_user(self, email, password=None, **extra_fields):
        """
        Create and save a regular user.

        The e-mail is normalised (domain lower-cased by Django, then the whole
        address lower-cased here so ``Ali@X.com`` and ``ali@x.com`` cannot
        become two accounts) and the password is always run through the
        configured hasher — a ``None`` password produces an unusable hash
        rather than an empty one, which is what the seeding script and any
        future SSO flow need.
        """
        if not email or not str(email).strip():
            raise ValueError('Users must have an email address.')

        email = self.normalize_email(str(email).strip()).lower()

        extra_fields.setdefault('role', ROLE_USER)
        extra_fields.setdefault('is_active', True)
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)

        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        """Create a staff + superuser account with the ``admin`` role."""
        extra_fields.setdefault('role', ROLE_ADMIN)
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self.create_user(email, password, **extra_fields)

    def get_by_natural_key(self, username):
        """Case-insensitive login lookup (the natural key is the e-mail)."""
        return self.get(**{f'{self.model.USERNAME_FIELD}__iexact': username})


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class User(AbstractBaseUser, PermissionsMixin):
    """
    An AutoSmokeGuard account.

    ``AbstractBaseUser`` contributes ``password`` and ``last_login`` plus the
    hashing helpers; ``PermissionsMixin`` contributes ``is_superuser``,
    ``groups`` and ``user_permissions`` so the Django admin works unchanged.
    Everything else is declared here to match the SDS ``users`` table.
    """

    user_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name='user ID',
    )
    email = models.EmailField(
        unique=True,
        db_index=True,
        help_text='Used as the login identifier.',
    )
    full_name = models.CharField(max_length=150, blank=True)
    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default=ROLE_USER,
        db_index=True,
        help_text="'admin' unlocks the system-configuration screens.",
    )
    is_active = models.BooleanField(
        default=True,
        help_text='Deactivate instead of deleting to preserve analysis history.',
    )
    is_staff = models.BooleanField(
        default=False,
        help_text='Grants access to the Django admin site.',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    password_changed_at = models.DateTimeField(
        null=True,
        blank=True,
        editable=False,
        help_text=(
            'When this password was last set. Access tokens issued before '
            'this moment are refused — see accounts.authentication.'
        ),
    )

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = UserManager()

    class Meta:
        db_table = 'accounts_user'
        ordering = ('-created_at',)
        verbose_name = 'user'
        verbose_name_plural = 'users'

    def __str__(self):
        return self.email

    # -- credentials --------------------------------------------------------

    def set_password(self, raw_password):
        """
        Hash and store a new password, and stamp ``password_changed_at``.

        Security review finding ASG-08
        ------------------------------
        Blacklisting refresh tokens (``accounts.services.revoke_all_refresh_
        tokens``) severs the *renewal* chain, but an access token that was
        already minted is a self-contained, stateless bearer credential: it
        keeps working until its own ``exp``, which is 30 minutes away.  A
        user who resets their password *because* they believe they are
        compromised therefore leaves the attacker with full API access for
        up to half an hour.

        The fix needs one bit of per-user server-side state that a stateless
        token can be compared against, and this is it.  Every access token
        carries an ``iat`` (issued-at) claim; ``accounts.authentication.
        PasswordChangeAwareJWTAuthentication`` refuses any token whose
        ``iat`` predates this column.

        Why the write lives *here* rather than in the reset view
        -------------------------------------------------------
        ``set_password`` is the single choke point every password change in
        the project already goes through — ``UserManager.create_user`` (and
        therefore ``createsuperuser`` and the seed script), ``manage.py
        changepassword``, and ``accounts.views.PasswordResetConfirmView``.
        Stamping it here means a future code path that changes a password
        cannot forget to invalidate the old tokens; scattering the write
        across callers would guarantee that it eventually is forgotten.

        Callers that save with ``update_fields`` must include
        ``'password_changed_at'`` alongside ``'password'`` for the stamp to
        reach the database.  The one place that matters is the reset-confirm
        view, which does exactly that.

        Note on hash upgrades: ``AbstractBaseUser.check_password`` re-calls
        ``set_password`` when the stored hash needs re-encoding (e.g. after
        ``PASSWORD_HASHERS`` or the PBKDF2 iteration count changes) and then
        saves with ``update_fields=['password']``.  The new stamp therefore
        stays in memory and is *not* persisted, so a routine rehash does not
        mass-invalidate live sessions.  That is the behaviour we want.
        """
        super().set_password(raw_password)
        self.password_changed_at = timezone.now()

    # -- convenience --------------------------------------------------------

    @property
    def is_admin(self):
        """True for administrators, by either the role column or staff flag."""
        return self.role == ROLE_ADMIN or self.is_staff

    def get_full_name(self):
        """Django admin hook — fall back to the e-mail when no name is set."""
        return self.full_name or self.email

    def get_short_name(self):
        """Django admin hook — first word of the name, else the local part."""
        if self.full_name:
            return self.full_name.split()[0]
        return self.email.split('@')[0]


class PasswordResetToken(models.Model):
    """
    A single-use, time-limited credential for the "forgot password" flow.

    The token is a random 256-bit hex string rather than anything derived from
    the user, so it cannot be forged from a known e-mail address.  It is
    invalidated two ways: ``used`` is flipped the moment a password is
    actually changed, and ``expires_at`` caps the window at one hour even if
    the mail is never opened.
    """

    token_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name='token ID',
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name='password_reset_tokens',
        on_delete=models.CASCADE,
    )
    token = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
        default=generate_reset_token,
        editable=False,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(default=default_reset_expiry)
    used = models.BooleanField(default=False)

    class Meta:
        db_table = 'accounts_password_reset_token'
        ordering = ('-created_at',)
        indexes = [
            models.Index(fields=['user', 'used'],
                         name='pwreset_user_used_idx'),
        ]
        verbose_name = 'password reset token'
        verbose_name_plural = 'password reset tokens'

    def __str__(self):
        state = 'used' if self.used else ('valid' if self.is_valid() else 'expired')
        return f'{self.user_id} ({state})'

    def is_valid(self):
        """True only while the token is unused and inside its TTL."""
        return not self.used and timezone.now() < self.expires_at

    def mark_used(self):
        """Consume the token. Idempotent; safe to call more than once."""
        if not self.used:
            self.used = True
            self.save(update_fields=['used'])
