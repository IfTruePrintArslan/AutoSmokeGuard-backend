"""
Tests for the accounts API: register, login, token refresh, logout, the
caller's own profile, and the "forgot password" flow.

Test names carry the formal test-design id they map to (``test_tc01_...``
etc.) so the traceability back to the acceptance criteria is visible from
the test list alone; everything else covers a specific behaviour called out
in the accounts API's build brief that does not have its own TC number.
"""
import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework_simplejwt.tokens import AccessToken

from accounts.models import PasswordResetToken

from .conftest import DEFAULT_TEST_PASSWORD

#: Keys that must never appear anywhere in a response body.
_FORBIDDEN_KEYS = {'password', 'password_hash', 'is_staff', 'is_superuser'}


def _assert_no_forbidden_keys(data):
    """Recursively assert nothing in ``data`` leaks a sensitive field."""
    if isinstance(data, dict):
        leaked = _FORBIDDEN_KEYS & set(data)
        assert not leaked, f'leaked keys {leaked} in {data}'
        for value in data.values():
            _assert_no_forbidden_keys(value)
    elif isinstance(data, list):
        for item in data:
            _assert_no_forbidden_keys(item)


# ---------------------------------------------------------------------------
# Throttle isolation
#
# Login/register/password-reset are rate-limited per IP via Django's default
# cache, which — unlike the database — is not reset between tests by
# pytest-django's transaction rollback. Without this, a test earlier in the
# run (or an earlier run of this module) could leave the shared 127.0.0.1
# counters primed and intermittently 429 an unrelated test.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_throttle_cache():
    from django.core.cache import cache
    cache.clear()
    yield
    cache.clear()


def _register(api, email, password=DEFAULT_TEST_PASSWORD, full_name='Test User'):
    return api.post('/api/register', {
        'full_name': full_name, 'email': email, 'password': password,
    }, format='json')


def _login(api, email, password=DEFAULT_TEST_PASSWORD):
    return api.post('/api/login', {'email': email, 'password': password},
                    format='json')


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestRegister:
    """TC-01 / TC-02 and the surrounding register behaviour."""

    def test_tc01_register_success(self, api):
        response = _register(api, 'ada@example.com', full_name='Ada Lovelace')

        assert response.status_code == 201
        body = response.data
        assert body['user']['email'] == 'ada@example.com'
        assert body['user']['full_name'] == 'Ada Lovelace'
        assert body['user']['role'] == 'user'
        assert body['user']['user_id']
        assert body['access'] and body['refresh']
        _assert_no_forbidden_keys(body)

        user = get_user_model().objects.get(email='ada@example.com')
        assert user.password != DEFAULT_TEST_PASSWORD
        assert user.check_password(DEFAULT_TEST_PASSWORD) is True

    def test_tc01_ignores_client_supplied_role(self, api):
        response = api.post('/api/register', {
            'full_name': 'Sneaky', 'email': 'sneaky@example.com',
            'password': DEFAULT_TEST_PASSWORD, 'role': 'admin',
        }, format='json')

        assert response.status_code == 201
        assert response.data['user']['role'] == 'user'

    def test_tc02_duplicate_email_rejected(self, api, user_factory):
        user_factory(email='taken@example.com')

        response = _register(api, 'taken@example.com')

        assert response.status_code == 400
        assert response.data['code'] == 'email_exists'
        assert 'already exists' in response.data['errors']['email'][0]
        assert get_user_model().objects.filter(
            email__iexact='taken@example.com').count() == 1

    def test_tc02_duplicate_email_is_case_insensitive(self, api, user_factory):
        user_factory(email='user@x.com')

        response = _register(api, 'USER@x.com')

        assert response.status_code == 400
        assert response.data['code'] == 'email_exists'
        assert get_user_model().objects.filter(
            email__iexact='user@x.com').count() == 1

    def test_register_weak_password_returns_field_errors(self, api):
        response = _register(api, 'weak@example.com', password='123')

        assert response.status_code == 400
        assert response.data['errors']['password']
        assert not get_user_model().objects.filter(email='weak@example.com').exists()


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestLogin:
    """TC-03 / TC-04 and the surrounding login behaviour."""

    def test_tc03_wrong_password_rejected(self, api, user_factory):
        user_factory(email='bob@example.com', password=DEFAULT_TEST_PASSWORD)

        response = _login(api, 'bob@example.com', 'not-the-password')

        assert response.status_code == 401
        assert response.data['code'] == 'invalid_credentials'
        assert response.data['detail'] == 'Invalid credentials.'

    def test_tc03_unknown_and_wrong_password_are_identical(self, api, user_factory):
        user_factory(email='bob2@example.com', password=DEFAULT_TEST_PASSWORD)

        wrong_password = _login(api, 'bob2@example.com', 'nope-nope-nope')
        unknown_email = _login(api, 'ghost@example.com', 'nope-nope-nope')

        assert wrong_password.status_code == unknown_email.status_code == 401
        assert wrong_password.data == unknown_email.data

    def test_tc04_valid_login_issues_a_working_access_token(self, api, user_factory):
        user = user_factory(email='carol@example.com', password=DEFAULT_TEST_PASSWORD)

        response = _login(api, 'carol@example.com')

        assert response.status_code == 200
        body = response.data
        assert body['user']['email'] == 'carol@example.com'
        _assert_no_forbidden_keys(body)

        access = AccessToken(body['access'])
        assert access['user_id'] == str(user.user_id)

        api.credentials(HTTP_AUTHORIZATION=f'Bearer {body["access"]}')
        me = api.get('/api/me')
        assert me.status_code == 200
        assert me.data['email'] == 'carol@example.com'

    def test_login_updates_last_login(self, api, user_factory):
        user = user_factory(email='dana@example.com', password=DEFAULT_TEST_PASSWORD)
        assert user.last_login is None

        assert _login(api, 'dana@example.com').status_code == 200

        user.refresh_from_db()
        assert user.last_login is not None

    def test_login_rejects_disabled_account(self, api, user_factory):
        user_factory(email='closed@example.com', password=DEFAULT_TEST_PASSWORD,
                    is_active=False)

        response = _login(api, 'closed@example.com')

        assert response.status_code == 401
        assert response.data['code'] == 'account_disabled'

    def test_login_is_case_insensitive_on_email(self, api, user_factory):
        user_factory(email='eve@example.com', password=DEFAULT_TEST_PASSWORD)

        response = _login(api, 'EVE@Example.com')

        assert response.status_code == 200

    def test_login_never_creates_a_session(self, api, user_factory):
        user_factory(email='frank@example.com', password=DEFAULT_TEST_PASSWORD)

        response = _login(api, 'frank@example.com')

        assert response.status_code == 200
        assert 'sessionid' not in response.cookies


# ---------------------------------------------------------------------------
# Protected endpoints / bad tokens
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestAuthProtection:
    """TC-05: protected endpoints reject the unauthenticated and the fake."""

    def test_tc05_unauthenticated_request_is_rejected(self, api):
        response = api.get('/api/me')

        assert response.status_code == 401

    def test_tc05_garbage_bearer_token_is_rejected(self, api):
        api.credentials(HTTP_AUTHORIZATION='Bearer not-a-real-token')

        response = api.get('/api/me')

        assert response.status_code == 401
        assert response.data['code'] == 'token_not_valid'


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestRefreshToken:

    def test_refresh_happy_path(self, api, user_factory):
        user_factory(email='gina@example.com', password=DEFAULT_TEST_PASSWORD)
        login = _login(api, 'gina@example.com')
        old_refresh = login.data['refresh']

        response = api.post('/api/refresh-token', {'refresh': old_refresh},
                            format='json')

        assert response.status_code == 200
        assert response.data['access']
        assert response.data['refresh']
        assert response.data['refresh'] != old_refresh
        assert response.data['access_expires_in'] == 30 * 60

    def test_blacklisted_refresh_is_rejected(self, api, user_factory):
        user_factory(email='hank@example.com', password=DEFAULT_TEST_PASSWORD)
        login = _login(api, 'hank@example.com')
        refresh = login.data['refresh']

        first = api.post('/api/refresh-token', {'refresh': refresh}, format='json')
        assert first.status_code == 200

        # ROTATE_REFRESH_TOKENS + BLACKLIST_AFTER_ROTATION blacklist the
        # original token the moment it is used once.
        second = api.post('/api/refresh-token', {'refresh': refresh}, format='json')

        assert second.status_code == 401
        assert second.data['code'] == 'token_not_valid'

    def test_garbage_refresh_is_rejected(self, api):
        response = api.post('/api/refresh-token', {'refresh': 'garbage'},
                            format='json')

        assert response.status_code == 401
        assert response.data['code'] == 'token_not_valid'


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestLogout:

    def test_logout_blacklists_the_refresh_token(self, api, user_factory):
        user_factory(email='iris@example.com', password=DEFAULT_TEST_PASSWORD)
        login = _login(api, 'iris@example.com')
        access, refresh = login.data['access'], login.data['refresh']
        api.credentials(HTTP_AUTHORIZATION=f'Bearer {access}')

        response = api.post('/api/logout', {'refresh': refresh}, format='json')

        assert response.status_code == 200
        assert response.data == {'detail': 'Signed out.'}

        reuse = api.post('/api/refresh-token', {'refresh': refresh}, format='json')
        assert reuse.status_code == 401
        assert reuse.data['code'] == 'token_not_valid'

    def test_logout_twice_still_returns_200(self, api, user_factory):
        user_factory(email='jack@example.com', password=DEFAULT_TEST_PASSWORD)
        login = _login(api, 'jack@example.com')
        access, refresh = login.data['access'], login.data['refresh']
        api.credentials(HTTP_AUTHORIZATION=f'Bearer {access}')

        first = api.post('/api/logout', {'refresh': refresh}, format='json')
        second = api.post('/api/logout', {'refresh': refresh}, format='json')

        assert first.status_code == 200
        assert second.status_code == 200

    def test_logout_requires_authentication(self, api):
        response = api.post('/api/logout', {'refresh': 'whatever'}, format='json')

        assert response.status_code == 401

    def test_logout_with_malformed_refresh_still_returns_200(self, auth_client):
        response = auth_client.post('/api/logout', {'refresh': 'not-a-token'},
                                    format='json')

        assert response.status_code == 200

    def test_logout_with_missing_refresh_still_returns_200(self, auth_client):
        response = auth_client.post('/api/logout', {}, format='json')

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestPasswordReset:

    def test_unknown_email_returns_the_same_200_as_a_known_one(
            self, api, user_factory, settings):
        settings.DEBUG = True
        user_factory(email='known@example.com', password=DEFAULT_TEST_PASSWORD)

        known = api.post('/api/password-reset', {'email': 'known@example.com'},
                         format='json')
        unknown = api.post('/api/password-reset', {'email': 'ghost@example.com'},
                           format='json')

        assert known.status_code == unknown.status_code == 200
        assert known.data['detail'] == unknown.data['detail']
        assert 'debug_token' in known.data
        assert 'debug_token' not in unknown.data

    def test_debug_token_hidden_when_debug_is_false(self, api, user_factory, settings):
        settings.DEBUG = False
        user_factory(email='nodebug@example.com', password=DEFAULT_TEST_PASSWORD)

        response = api.post('/api/password-reset', {'email': 'nodebug@example.com'},
                            format='json')

        assert response.status_code == 200
        assert 'debug_token' not in response.data

    def test_request_confirm_then_login_with_new_password(
            self, api, user_factory, settings):
        settings.DEBUG = True
        user_factory(email='kim@example.com', password=DEFAULT_TEST_PASSWORD)

        old_login = _login(api, 'kim@example.com')
        old_refresh = old_login.data['refresh']

        requested = api.post('/api/password-reset', {'email': 'kim@example.com'},
                             format='json')
        token = requested.data['debug_token']

        new_password = 'NewSecur3@Passw0rd'
        confirm = api.post('/api/password-reset/confirm', {
            'token': token, 'password': new_password,
        }, format='json')

        assert confirm.status_code == 200
        assert confirm.data == {'detail': 'Password updated.'}

        assert _login(api, 'kim@example.com', DEFAULT_TEST_PASSWORD).status_code == 401
        assert _login(api, 'kim@example.com', new_password).status_code == 200

        # a password change must kill existing sessions
        old_refresh_attempt = api.post('/api/refresh-token',
                                       {'refresh': old_refresh}, format='json')
        assert old_refresh_attempt.status_code == 401
        assert old_refresh_attempt.data['code'] == 'token_not_valid'

    def test_reused_token_fails(self, api, user_factory, settings):
        settings.DEBUG = True
        user_factory(email='liam@example.com', password=DEFAULT_TEST_PASSWORD)

        requested = api.post('/api/password-reset', {'email': 'liam@example.com'},
                             format='json')
        token = requested.data['debug_token']

        first = api.post('/api/password-reset/confirm', {
            'token': token, 'password': 'FirstNew@Passw0rd',
        }, format='json')
        assert first.status_code == 200

        second = api.post('/api/password-reset/confirm', {
            'token': token, 'password': 'SecondNew@Passw0rd',
        }, format='json')

        assert second.status_code == 400
        assert second.data['code'] == 'invalid_reset_token'

    def test_expired_token_fails(self, api, user_factory):
        user = user_factory(email='mia@example.com', password=DEFAULT_TEST_PASSWORD)
        expired = PasswordResetToken.objects.create(
            user=user, expires_at=timezone.now() - timezone.timedelta(minutes=1),
        )

        response = api.post('/api/password-reset/confirm', {
            'token': expired.token, 'password': 'Whatever@Passw0rd1',
        }, format='json')

        assert response.status_code == 400
        assert response.data['code'] == 'invalid_reset_token'

    def test_unknown_token_fails(self, api):
        response = api.post('/api/password-reset/confirm', {
            'token': 'x' * 64, 'password': 'Whatever@Passw0rd1',
        }, format='json')

        assert response.status_code == 400
        assert response.data['code'] == 'invalid_reset_token'

    def test_confirm_validates_password_strength(self, api, user_factory, settings):
        settings.DEBUG = True
        user_factory(email='noah@example.com', password=DEFAULT_TEST_PASSWORD)

        requested = api.post('/api/password-reset', {'email': 'noah@example.com'},
                             format='json')
        token = requested.data['debug_token']

        response = api.post('/api/password-reset/confirm', {
            'token': token, 'password': '123',
        }, format='json')

        assert response.status_code == 400
        assert response.data['errors']['password']

    def test_password_reset_request_is_throttled(self, api, user_factory):
        user_factory(email='olive@example.com', password=DEFAULT_TEST_PASSWORD)

        responses = [
            api.post('/api/password-reset', {'email': 'olive@example.com'},
                    format='json')
            for _ in range(6)
        ]

        assert [r.status_code for r in responses[:5]] == [200] * 5
        assert responses[5].status_code == 429


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestMeProfile:

    def test_get_me_returns_the_documented_shape(self, auth_client):
        response = auth_client.get('/api/me')

        assert response.status_code == 200
        assert set(response.data) == {
            'user_id', 'email', 'full_name', 'role', 'created_at',
        }

    def test_patch_updates_full_name_and_ignores_everything_else(self, auth_client):
        original_email = auth_client.user.email

        response = auth_client.patch('/api/me', {
            'full_name': 'New Name', 'role': 'admin',
            'email': 'takeover@evil.example.com',
        }, format='json')

        assert response.status_code == 200
        assert response.data['full_name'] == 'New Name'
        assert response.data['role'] == 'user'
        assert response.data['email'] == original_email

        auth_client.user.refresh_from_db()
        assert auth_client.user.full_name == 'New Name'
        assert auth_client.user.role == 'user'
        assert auth_client.user.email == original_email


# ---------------------------------------------------------------------------
# No endpoint ever leaks a password / staff flag
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestNoSensitiveFieldLeaks:

    def test_register_login_me_and_refresh_never_leak_sensitive_fields(self, api):
        register = _register(api, 'privacy@example.com')
        _assert_no_forbidden_keys(register.data)

        login = _login(api, 'privacy@example.com')
        _assert_no_forbidden_keys(login.data)

        api.credentials(HTTP_AUTHORIZATION=f'Bearer {login.data["access"]}')

        me = api.get('/api/me')
        _assert_no_forbidden_keys(me.data)

        patch = api.patch('/api/me', {'full_name': 'Still Private'}, format='json')
        _assert_no_forbidden_keys(patch.data)

        refresh = api.post('/api/refresh-token', {'refresh': login.data['refresh']},
                           format='json')
        _assert_no_forbidden_keys(refresh.data)
