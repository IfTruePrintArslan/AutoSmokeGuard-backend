"""
Views for the accounts API: register, login, token refresh, logout, the
caller's own profile, and the "forgot password" flow.

Every view is a plain ``APIView`` (not a viewset) because the contract's
paths are flat and verb-specific rather than a resourceful collection.
Views stay thin — validation lives in ``accounts.serializers``, the token
and password-reset mechanics live in ``accounts.services`` — except for the
handful of branches that need one of the contract's specific machine codes
(``email_exists``, ``invalid_credentials``, ``account_disabled``,
``invalid_reset_token``, ``token_not_valid``).  Those are built by hand with
``common.exceptions.error_response`` because the project-wide exception
handler collapses any ``rest_framework.exceptions.ValidationError`` to the
generic ``code="validation_error"``, which would otherwise swallow the
specific code the frontend keys its behaviour off.
"""
import logging

from django.conf import settings
from django.contrib.auth.models import update_last_login
from drf_spectacular.utils import OpenApiResponse, extend_schema
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.serializers import TokenRefreshSerializer
from rest_framework_simplejwt.tokens import RefreshToken

from common.exceptions import error_response

from .models import ROLE_USER, PasswordResetToken, User
from .serializers import (
    LoginSerializer,
    LogoutSerializer,
    PasswordResetConfirmSerializer,
    PasswordResetRequestSerializer,
    RefreshSerializer,
    RegisterSerializer,
    UserSerializer,
)
from .services import (
    consume_reset_token,
    create_reset_token,
    get_client_ip,
    issue_tokens,
    revoke_all_refresh_tokens,
)

logger = logging.getLogger('asg.accounts')

#: Always the same regardless of whether the account exists — no enumeration.
_RESET_REQUESTED_DETAIL = (
    'If an account exists for that email, a password reset link has been sent.'
)


# ---------------------------------------------------------------------------
# Throttles
#
# Each hardcodes its own ``rate`` as a class attribute rather than reading
# ``DEFAULT_THROTTLE_RATES[scope]`` from settings: ``SimpleRateThrottle``
# only consults that dict when the class does not already carry a ``rate``,
# so the limit can live right next to the endpoint it protects instead of a
# shared settings mapping this agent does not own.
# ---------------------------------------------------------------------------

class RegisterRateThrottle(AnonRateThrottle):
    """20 registrations per hour per IP — blunts scripted account creation."""

    scope = 'accounts_register'
    rate = '20/hour'


class LoginRateThrottle(AnonRateThrottle):
    """10 attempts per minute per IP — blunts credential stuffing."""

    scope = 'accounts_login'
    rate = '10/min'


class PasswordResetRateThrottle(AnonRateThrottle):
    """
    5 requests per hour per IP.

    The endpoint itself never reveals whether an address has an account, so
    this rate limit is the main defence against it being hammered as an
    enumeration or mail-bombing oracle.
    """

    scope = 'accounts_password_reset'
    rate = '5/hour'


# ---------------------------------------------------------------------------
# Register / login
# ---------------------------------------------------------------------------

class RegisterView(APIView):
    """Create an account and log the caller straight in."""

    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [RegisterRateThrottle]

    @extend_schema(
        summary='Register a new account',
        request=RegisterSerializer,
        responses={201: OpenApiResponse(description='Account created.')},
        tags=['Auth'],
    )
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data['email']
        if User.objects.filter(email__iexact=email).exists():
            return error_response(
                'An account with this email already exists.',
                'email_exists',
                status.HTTP_400_BAD_REQUEST,
                errors={'email': ['An account with this email already exists.']},
            )

        user = User.objects.create_user(
            email=email,
            password=serializer.validated_data['password'],
            full_name=serializer.validated_data.get('full_name', ''),
            role=ROLE_USER,
        )
        tokens = issue_tokens(user)
        logger.info('Registered new account %s', user.email)
        return Response(
            {'user': UserSerializer(user).data, **tokens},
            status=status.HTTP_201_CREATED,
        )


class LoginView(APIView):
    """Authenticate by e-mail/password and issue a fresh token pair."""

    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [LoginRateThrottle]

    @extend_schema(
        summary='Log in',
        request=LoginSerializer,
        responses={200: OpenApiResponse(description='Authenticated.')},
        tags=['Auth'],
    )
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data['email'].strip().lower()
        password = serializer.validated_data['password']
        ip = get_client_ip(request)

        # Looked up (and the password checked) by hand rather than through
        # django.contrib.auth.authenticate(): the stock ModelBackend refuses
        # an inactive user before password checking ever happens, which
        # would make "wrong password" and "disabled account" indistinguishable
        # from here — and the contract wants them reported differently.
        user = User.objects.filter(email__iexact=email).first()

        # Constant-ish work regardless of whether the account exists
        # (security review finding ASG-06).
        #
        # The two failure bodies below are already byte-identical, but the
        # *timing* was not: a missing account short-circuited before any
        # hashing, while a real one paid for a full PBKDF2 verification.
        # Measured on this machine that was ~0.5 ms vs ~97 ms — a ~200x gap
        # that turns this endpoint into a reliable "does this address have an
        # account?" oracle despite the identical responses.
        #
        # Hashing the supplied password against a throw-away user object on
        # the miss path spends the same PBKDF2 budget and closes the gap.
        # This is exactly what django.contrib.auth's own ModelBackend does
        # for the same reason; we cannot call it (see above), so the
        # mitigation is reproduced here.
        if user is None:
            User().set_password(password)
            authenticated = False
        else:
            authenticated = user.check_password(password)

        if not authenticated:
            logger.warning('Failed login attempt for %s from %s', email, ip)
            return error_response(
                'Invalid credentials.', 'invalid_credentials',
                status.HTTP_401_UNAUTHORIZED,
            )

        if not user.is_active:
            logger.warning('Login blocked for disabled account %s from %s',
                           email, ip)
            return error_response(
                'This account has been disabled.', 'account_disabled',
                status.HTTP_401_UNAUTHORIZED,
            )

        # No django.contrib.auth.login() call anywhere in this flow — this
        # API is stateless JWT only, so a successful login must never leave
        # a session behind.
        update_last_login(None, user)
        tokens = issue_tokens(user)
        logger.info('Successful login for %s from %s', email, ip)
        return Response(
            {'user': UserSerializer(user).data, **tokens},
            status=status.HTTP_200_OK,
        )


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

class RefreshTokenView(APIView):
    """
    Exchange a refresh token for a new access token.

    Delegates the actual rotation/blacklist mechanics to SimpleJWT's own
    ``TokenRefreshSerializer`` — it already knows how to honour
    ``ROTATE_REFRESH_TOKENS`` / ``BLACKLIST_AFTER_ROTATION`` and to re-check
    the owning user is still active — and only reshapes the result: adds
    ``access_expires_in``, and turns a rejected token into the project's
    standard error envelope with ``code="token_not_valid"`` (SimpleJWT's own
    ``TokenError`` is a bare ``Exception``, not a DRF one, so it is caught
    here explicitly rather than left for the shared exception handler).
    """

    permission_classes = [AllowAny]
    authentication_classes = []

    @extend_schema(
        summary='Refresh an access token',
        request=RefreshSerializer,
        responses={200: OpenApiResponse(description='New access token issued.')},
        tags=['Auth'],
    )
    def post(self, request):
        input_serializer = RefreshSerializer(data=request.data)
        input_serializer.is_valid(raise_exception=True)

        jwt_serializer = TokenRefreshSerializer(data=input_serializer.validated_data)
        try:
            jwt_serializer.is_valid(raise_exception=True)
        except TokenError as exc:
            logger.info('Refresh token rejected: %s', exc)
            return error_response(
                'Token is invalid or expired.', 'token_not_valid',
                status.HTTP_401_UNAUTHORIZED,
            )

        data = dict(jwt_serializer.validated_data)
        data['access_expires_in'] = int(
            settings.SIMPLE_JWT['ACCESS_TOKEN_LIFETIME'].total_seconds(),
        )
        return Response(data, status=status.HTTP_200_OK)


class LogoutView(APIView):
    """
    Blacklist the caller's refresh token.

    Always answers 200 — a missing, malformed or already-blacklisted token
    is swallowed rather than raised, because a sign-out action must never
    leave the client stuck on an error.
    """

    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary='Log out',
        request=LogoutSerializer,
        responses={200: OpenApiResponse(description='Signed out.')},
        tags=['Auth'],
    )
    def post(self, request):
        serializer = LogoutSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        raw = serializer.validated_data.get('refresh')

        if raw:
            try:
                RefreshToken(raw).blacklist()
            except TokenError:
                pass  # already blacklisted / malformed — never fail logout

        logger.info('User %s signed out', request.user.email)
        return Response({'detail': 'Signed out.'}, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

class MeView(APIView):
    """Read and partially update the authenticated user's own profile."""

    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary='Get the current user',
        responses={200: UserSerializer},
        tags=['Auth'],
    )
    def get(self, request):
        return Response(UserSerializer(request.user).data)

    @extend_schema(
        summary='Update the current user',
        description='Only `full_name` is writable; every other key is ignored.',
        request=UserSerializer,
        responses={200: UserSerializer},
        tags=['Auth'],
    )
    def patch(self, request):
        serializer = UserSerializer(request.user, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(UserSerializer(request.user).data)


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------

class PasswordResetRequestView(APIView):
    """
    Start the "forgot password" flow.

    Always returns 200 with the exact same body whether or not the address
    belongs to an account. When it does, a token is minted and — because
    this deployment has no mail server — logged; in ``DEBUG`` the token is
    also handed back in the body so the UI can offer a working link without
    a real inbox.
    """

    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [PasswordResetRateThrottle]

    @extend_schema(
        summary='Request a password reset',
        request=PasswordResetRequestSerializer,
        responses={200: OpenApiResponse(description='Reset requested.')},
        tags=['Auth'],
    )
    def post(self, request):
        serializer = PasswordResetRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        email = serializer.validated_data['email']
        ip = get_client_ip(request)

        user = User.objects.filter(email__iexact=email).first()
        body = {'detail': _RESET_REQUESTED_DETAIL}

        if user is not None:
            token = create_reset_token(user)
            if settings.DEBUG:
                body['debug_token'] = token.token
            logger.info('Password reset requested for %s from %s', email, ip)
        else:
            logger.info('Password reset requested for unknown email %s from %s',
                        email, ip)

        return Response(body, status=status.HTTP_200_OK)


class PasswordResetConfirmView(APIView):
    """Complete the "forgot password" flow and kill every existing session."""

    permission_classes = [AllowAny]
    authentication_classes = []

    @extend_schema(
        summary='Confirm a password reset',
        request=PasswordResetConfirmSerializer,
        responses={200: OpenApiResponse(description='Password updated.')},
        tags=['Auth'],
    )
    def post(self, request):
        serializer = PasswordResetConfirmSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            user = consume_reset_token(serializer.validated_data['token'])
        except PasswordResetToken.DoesNotExist:
            return error_response(
                'That reset link is invalid or has expired.',
                'invalid_reset_token', status.HTTP_400_BAD_REQUEST,
            )

        # ``User.set_password`` also stamps ``password_changed_at``; the stamp
        # has to be named in ``update_fields`` or it never reaches the
        # database.  Between the two of them:
        #   * ``revoke_all_refresh_tokens`` cuts the renewal chain, and
        #   * ``password_changed_at`` invalidates every access token already
        #     issued (``accounts.authentication``), which blacklisting alone
        #     could not do — security review finding ASG-08.
        user.set_password(serializer.validated_data['password'])
        user.save(update_fields=['password', 'password_changed_at'])
        revoke_all_refresh_tokens(user)
        logger.info('Password reset completed for %s', user.email)
        return Response({'detail': 'Password updated.'}, status=status.HTTP_200_OK)
