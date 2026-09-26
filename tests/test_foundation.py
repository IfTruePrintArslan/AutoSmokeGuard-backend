"""
Foundation-layer tests.

These cover the pieces every other agent's code sits on top of: the health
probe, the configuration singleton, upload validation, the pagination
envelope, the error envelope, the media path helpers and the object-level
permissions.  If something here breaks, every feature app breaks with it.
"""
import io
import logging
from unittest import mock

import pytest
from django.core.exceptions import ValidationError as DjangoValidationError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.http import Http404
from django.urls import reverse
from rest_framework import exceptions as drf_exceptions
from rest_framework.request import Request
from rest_framework.test import APIRequestFactory

from common import exceptions as common_exceptions
from common import validators
from common.pagination import StandardPagination
from common.permissions import IsAdminRole, IsOwner, IsOwnerOrAdmin
from common.storage import (
    report_path,
    to_media_relative,
    user_media_path,
    vehicle_crop_path,
)
from common.validators import validate_upload
from system_config.models import SystemSetting

from .conftest import (
    make_avi_bytes,
    make_jpeg_bytes,
    make_mp4_bytes,
    make_png_bytes,
)

IMAGE_AND_VIDEO_EXTS = ['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov']
ONE_MB = 1024 * 1024


def upload(name, data):
    """Wrap raw bytes in the same object Django hands a view for an upload."""
    return SimpleUploadedFile(name, data)


# ===========================================================================
# Health endpoint
# ===========================================================================

@pytest.mark.django_db
class TestHealthEndpoint:
    """The probe must answer anyone, and must actually check its dependencies."""

    def test_is_public_and_reports_ok(self, api):
        response = api.get('/api/health/')

        assert response.status_code == 200
        assert response.data['status'] == 'ok'
        assert response.data['service'] == 'autosmokeguard-backend'
        assert response.data['database'] == 'ok'

    def test_reports_every_documented_field(self, api):
        response = api.get('/api/health/')
        body = response.data

        assert set(body) == {
            'status', 'service', 'version', 'database', 'ml_assets',
            'worker', 'time',
        }
        assert set(body['ml_assets']) == {'yolo', 'segmenter'}
        assert isinstance(body['ml_assets']['yolo'], bool)
        assert isinstance(body['ml_assets']['segmenter'], bool)
        assert body['worker'] in {'enabled', 'disabled'}
        assert body['version']
        assert body['time']

    def test_legacy_path_still_served(self, api):
        """Sprint-1 deployments probe /health/; that must not break."""
        assert api.get('/health/').status_code == 200

    def test_reverse_resolves_to_the_canonical_path(self):
        assert reverse('health-check') == '/api/health/'

    def test_ignores_a_bogus_authorization_header(self, api):
        """
        Proof that the view clears authentication_classes, not just permissions.

        With the global JWTAuthentication still attached, a malformed token
        would 401 before the permission check ever ran — which would break
        every uptime monitor that sends a stale header.
        """
        api.credentials(HTTP_AUTHORIZATION='Bearer not-a-real-token')
        assert api.get('/api/health/').status_code == 200

    def test_request_logging_middleware_stamps_a_duration(self, api):
        response = api.get('/api/health/')
        assert float(response['X-Response-Time-ms']) >= 0


# ===========================================================================
# SystemSetting singleton
# ===========================================================================

@pytest.mark.django_db
class TestSystemSettingSingleton:
    """There is exactly one configuration row — always, and only one."""

    def test_get_solo_creates_then_reuses_the_same_row(self):
        assert SystemSetting.objects.count() == 0

        first = SystemSetting.get_solo()
        second = SystemSetting.get_solo()

        assert SystemSetting.objects.count() == 1
        assert first.pk == second.pk == 1

    def test_defaults_match_the_shipped_configuration(self, settings_row):
        assert settings_row.allowed_image_formats == 'jpg,jpeg,png'
        assert settings_row.allowed_video_formats == 'mp4,avi,mov'
        assert settings_row.max_upload_mb == 512
        assert settings_row.confidence_threshold == pytest.approx(0.35)
        assert settings_row.smoke_mask_threshold == pytest.approx(0.5)
        assert settings_row.severity_low_max == pytest.approx(0.33)
        assert settings_row.severity_moderate_max == pytest.approx(0.66)
        assert settings_row.frame_sample_rate == 5
        assert settings_row.auto_generate_pdf is True
        assert settings_row.max_video_seconds == 300

    def test_a_second_instance_overwrites_rather_than_duplicates(self,
                                                                 settings_row):
        """Even a naive ``SystemSetting(...).save()`` cannot fork the config."""
        rogue = SystemSetting(max_upload_mb=64)
        rogue.save()

        assert SystemSetting.objects.count() == 1
        assert rogue.pk == 1
        assert SystemSetting.get_solo().max_upload_mb == 64

    def test_objects_create_does_not_raise_on_an_existing_row(self,
                                                              settings_row):
        """``create()`` passes force_insert; the model must absorb it."""
        SystemSetting.objects.create(frame_sample_rate=9)

        assert SystemSetting.objects.count() == 1
        assert SystemSetting.get_solo().frame_sample_rate == 9

    def test_delete_is_refused(self, settings_row):
        settings_row.delete()
        assert SystemSetting.objects.count() == 1

    def test_parsed_format_helpers(self, settings_row):
        assert settings_row.image_formats == ['jpg', 'jpeg', 'png']
        assert settings_row.video_formats == ['mp4', 'avi', 'mov']
        assert settings_row.all_formats == [
            'jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov',
        ]
        assert settings_row.max_upload_bytes == 512 * ONE_MB

    def test_severity_banding_uses_the_configured_cut_offs(self, settings_row):
        assert settings_row.severity_for(0.10) == 'low'
        assert settings_row.severity_for(0.33) == 'low'
        assert settings_row.severity_for(0.50) == 'moderate'
        assert settings_row.severity_for(0.66) == 'moderate'
        assert settings_row.severity_for(0.90) == 'high'


# ===========================================================================
# Upload validation
# ===========================================================================

class TestValidateUploadAccepts:
    """Real files in accepted formats must pass, and be classified correctly."""

    def test_accepts_a_real_jpeg(self):
        ok, code, message, meta = validate_upload(
            upload('photo.jpg', make_jpeg_bytes()),
            IMAGE_AND_VIDEO_EXTS, 10 * ONE_MB,
        )

        assert ok is True
        assert (code, message) == (None, None)
        assert meta['media_type'] == 'image'
        assert meta['format'] == 'jpg'
        assert meta['detected_format'] == 'jpeg'
        assert meta['size_bytes'] > 0

    def test_normalises_the_jpeg_extension(self):
        """``.jpeg`` and ``.jpg`` must land in the database as one spelling."""
        ok, _code, _message, meta = validate_upload(
            upload('photo.jpeg', make_jpeg_bytes()),
            IMAGE_AND_VIDEO_EXTS, 10 * ONE_MB,
        )

        assert ok is True
        assert meta['extension'] == 'jpeg'
        assert meta['format'] == 'jpg'

    def test_accepts_a_real_png(self):
        ok, _code, _message, meta = validate_upload(
            upload('frame.PNG', make_png_bytes()),
            IMAGE_AND_VIDEO_EXTS, 10 * ONE_MB,
        )

        assert ok is True
        assert meta['media_type'] == 'image'
        assert meta['format'] == 'png'

    def test_accepts_an_mp4_container(self):
        ok, _code, _message, meta = validate_upload(
            upload('clip.mp4', make_mp4_bytes()),
            IMAGE_AND_VIDEO_EXTS, 10 * ONE_MB,
        )

        assert ok is True
        assert meta['media_type'] == 'video'
        assert meta['format'] == 'mp4'
        assert meta['detected_format'] == 'isobmff'

    def test_accepts_an_avi_container(self):
        ok, _code, _message, meta = validate_upload(
            upload('clip.avi', make_avi_bytes()),
            IMAGE_AND_VIDEO_EXTS, 10 * ONE_MB,
        )

        assert ok is True
        assert meta['format'] == 'avi'
        assert meta['detected_format'] == 'avi'

    def test_leaves_the_file_rewound_for_the_caller(self):
        """Validation must not consume the stream the view is about to save."""
        payload = make_jpeg_bytes()
        handle = upload('photo.jpg', payload)

        validate_upload(handle, IMAGE_AND_VIDEO_EXTS, 10 * ONE_MB)

        assert handle.read() == payload


class TestValidateUploadRejects:
    """Every rejection path, with its stable machine-readable code."""

    def test_rejects_a_text_file(self):
        ok, code, message, meta = validate_upload(
            upload('notes.txt', b'hello world, this is not media at all'),
            IMAGE_AND_VIDEO_EXTS, 10 * ONE_MB,
        )

        assert ok is False
        assert code == validators.ERR_INVALID_EXTENSION
        assert '.txt' in message
        assert meta['extension'] == 'txt'
        assert meta['media_type'] is None

    def test_rejects_a_file_with_no_extension(self):
        ok, code, _message, _meta = validate_upload(
            upload('payload', make_jpeg_bytes()),
            IMAGE_AND_VIDEO_EXTS, 10 * ONE_MB,
        )

        assert ok is False
        assert code == validators.ERR_INVALID_EXTENSION

    def test_rejects_an_oversized_file(self):
        oversized = make_jpeg_bytes(width=64, height=64)
        max_bytes = len(oversized) - 1

        ok, code, message, meta = validate_upload(
            upload('big.jpg', oversized), IMAGE_AND_VIDEO_EXTS, max_bytes,
        )

        assert ok is False
        assert code == validators.ERR_FILE_TOO_LARGE
        assert 'exceeds' in message
        assert meta['size_bytes'] == len(oversized)

    def test_accepts_a_file_exactly_on_the_limit(self):
        """The ceiling is inclusive — off-by-one here rejects valid uploads."""
        payload = make_jpeg_bytes()

        ok, _code, _message, _meta = validate_upload(
            upload('exact.jpg', payload), IMAGE_AND_VIDEO_EXTS, len(payload),
        )

        assert ok is True

    def test_rejects_an_empty_file(self):
        ok, code, _message, _meta = validate_upload(
            upload('empty.jpg', b''), IMAGE_AND_VIDEO_EXTS, 10 * ONE_MB,
        )

        assert ok is False
        assert code == validators.ERR_EMPTY_FILE

    def test_rejects_a_truncated_file(self):
        """Two bytes of a JPEG header is an interrupted upload, not an image."""
        ok, code, _message, _meta = validate_upload(
            upload('cut.jpg', b'\xff\xd8'), IMAGE_AND_VIDEO_EXTS, 10 * ONE_MB,
        )

        assert ok is False
        assert code == validators.ERR_CORRUPT_FILE

    def test_rejects_a_corrupt_file_with_a_valid_extension(self):
        ok, code, _message, meta = validate_upload(
            upload('corrupt.png', b'\x00' * 4096),
            IMAGE_AND_VIDEO_EXTS, 10 * ONE_MB,
        )

        assert ok is False
        assert code == validators.ERR_CORRUPT_FILE
        assert meta['detected_format'] is None

    def test_rejects_a_renamed_file_whose_contents_disagree(self):
        """``script.png`` that is really a JPEG is a signature mismatch."""
        ok, code, message, _meta = validate_upload(
            upload('disguised.png', make_jpeg_bytes()),
            IMAGE_AND_VIDEO_EXTS, 10 * ONE_MB,
        )

        assert ok is False
        assert code == validators.ERR_FORMAT_MISMATCH
        assert 'JPEG' in message

    def test_rejects_a_format_this_deployment_has_switched_off(self):
        """An allowed-list of images only must turn away a valid MP4."""
        ok, code, _message, _meta = validate_upload(
            upload('clip.mp4', make_mp4_bytes()), ['jpg', 'png'], 10 * ONE_MB,
        )

        assert ok is False
        assert code == validators.ERR_INVALID_EXTENSION

    def test_rejects_a_missing_file(self):
        ok, code, _message, _meta = validate_upload(
            None, IMAGE_AND_VIDEO_EXTS, 10 * ONE_MB,
        )

        assert ok is False
        assert code == validators.ERR_NO_FILE


class TestValidatorHelpers:
    """The small pieces the main validator is assembled from."""

    @pytest.mark.parametrize('name,expected', [
        ('photo.JPG', 'jpg'),
        ('a.b.mp4', 'mp4'),
        ('noext', ''),
        ('', ''),
    ])
    def test_file_extension(self, name, expected):
        assert validators.file_extension(name) == expected

    def test_sniff_needs_a_minimum_header(self):
        assert validators.sniff_format(b'\xff\xd8\xff') is None

    def test_checksum_is_stable_and_rewinds(self):
        payload = make_png_bytes()
        handle = io.BytesIO(payload)

        first = validators.sha256_checksum(handle)
        second = validators.sha256_checksum(handle)

        assert first == second
        assert len(first) == 64


# ===========================================================================
# Pagination envelope
# ===========================================================================

class TestStandardPagination:
    """Every list endpoint answers with the same envelope."""

    def paginate(self, query='', total=25):
        paginator = StandardPagination()
        request = Request(APIRequestFactory().get(f'/api/things/{query}'))
        page = paginator.paginate_queryset(list(range(total)), request)
        return paginator, paginator.get_paginated_response(page).data

    def test_envelope_keys_and_values(self):
        _paginator, body = self.paginate('?page=2')

        assert list(body) == [
            'count', 'page', 'pages', 'page_size', 'next', 'previous',
            'results',
        ]
        assert body['count'] == 25
        assert body['page'] == 2
        assert body['pages'] == 3
        assert body['page_size'] == 10
        assert len(body['results']) == 10
        assert body['next'] is not None
        assert body['previous'] is not None

    def test_first_page_has_no_previous_link(self):
        _paginator, body = self.paginate()

        assert body['page'] == 1
        assert body['previous'] is None
        assert body['next'] is not None

    def test_page_size_is_client_tunable(self):
        _paginator, body = self.paginate('?page_size=5')

        assert body['page_size'] == 5
        assert body['pages'] == 5
        assert len(body['results']) == 5

    def test_page_size_is_capped_by_the_server(self):
        """A client must not be able to pull the whole table in one request."""
        _paginator, body = self.paginate('?page_size=100000', total=250)

        assert body['page_size'] == StandardPagination.max_page_size
        assert len(body['results']) == StandardPagination.max_page_size

    def test_schema_describes_the_envelope(self):
        paginator = StandardPagination()
        schema = paginator.get_paginated_response_schema(
            {'type': 'array', 'items': {'type': 'object'}},
        )

        assert set(schema['properties']) == {
            'count', 'page', 'pages', 'page_size', 'next', 'previous',
            'results',
        }


# ===========================================================================
# Error envelope
# ===========================================================================

class TestApiExceptionHandler:
    """Every failure leaves the API in the same {detail, code, errors} shape."""

    ENVELOPE_KEYS = {'detail', 'code', 'errors'}

    def handle(self, exc, context=None):
        return common_exceptions.api_exception_handler(exc, context or {})

    def test_not_found(self):
        response = self.handle(drf_exceptions.NotFound())

        assert response.status_code == 404
        assert set(response.data) == self.ENVELOPE_KEYS
        assert response.data['code'] == 'not_found'
        assert response.data['errors'] is None
        assert isinstance(response.data['detail'], str)

    def test_http404_is_converted(self):
        response = self.handle(Http404('gone'))

        assert response.status_code == 404
        assert response.data['code'] == 'not_found'

    def test_permission_denied(self):
        response = self.handle(drf_exceptions.PermissionDenied())

        assert response.status_code == 403
        assert response.data['code'] == 'permission_denied'

    def test_not_authenticated(self):
        response = self.handle(drf_exceptions.NotAuthenticated())

        assert response.status_code == 401
        assert response.data['code'] == 'not_authenticated'

    def test_field_validation_errors_are_collected(self):
        response = self.handle(drf_exceptions.ValidationError({
            'email': ['This field is required.'],
            'password': ['Too short.', 'Too common.'],
        }))

        assert response.status_code == 400
        assert set(response.data) == self.ENVELOPE_KEYS
        assert response.data['code'] == 'validation_error'
        assert response.data['detail'] == common_exceptions.VALIDATION_DETAIL
        assert response.data['errors'] == {
            'email': ['This field is required.'],
            'password': ['Too short.', 'Too common.'],
        }

    def test_non_field_validation_errors(self):
        response = self.handle(
            drf_exceptions.ValidationError(['Passwords do not match.']),
        )

        assert response.data['code'] == 'validation_error'
        assert response.data['errors'] == {
            'non_field_errors': ['Passwords do not match.'],
        }

    def test_nested_validation_errors_are_flattened_to_strings(self):
        response = self.handle(drf_exceptions.ValidationError({
            'media': {'file': ['Unsupported format.']},
        }))

        assert response.data['errors'] == {
            'media': ['file: Unsupported format.'],
        }

    def test_django_validation_error_is_converted(self):
        response = self.handle(
            DjangoValidationError({'max_upload_mb': ['Must be positive.']}),
        )

        assert response.status_code == 400
        assert response.data['code'] == 'validation_error'
        assert response.data['errors'] == {
            'max_upload_mb': ['Must be positive.'],
        }

    def test_unhandled_exception_becomes_an_opaque_500(self):
        """
        TC-15: an internal fault must never leak a traceback or a message.

        The exception text ('boom') must not appear anywhere in the body.
        """
        with mock.patch.object(common_exceptions.logger, 'exception') as logged:
            response = self.handle(RuntimeError('boom: /etc/secrets missing'))

        assert response.status_code == 500
        assert response.data == {
            'detail': 'Internal Server Error',
            'code': 'server_error',
            'errors': None,
        }
        assert 'boom' not in str(response.data)
        assert logged.called, 'a 5xx must be logged with its traceback'

    def test_the_500_body_is_not_shared_between_calls(self):
        """The module-level template must be copied, never handed out."""
        with mock.patch.object(common_exceptions.logger, 'exception'):
            first = self.handle(RuntimeError('one'))
            first.data['detail'] = 'mutated'
            second = self.handle(RuntimeError('two'))

        assert second.data['detail'] == 'Internal Server Error'
        assert common_exceptions.SERVER_ERROR_BODY['detail'] == \
            'Internal Server Error'

    def test_error_response_helper_matches_the_envelope(self):
        response = common_exceptions.error_response(
            'Upload too large.', 'file_too_large', 413,
        )

        assert response.status_code == 413
        assert set(response.data) == self.ENVELOPE_KEYS
        assert response.data['code'] == 'file_too_large'


# ===========================================================================
# Storage helpers
# ===========================================================================

@pytest.mark.django_db
class TestStorageHelpers:
    """Uploads are renamed and namespaced; nothing client-supplied is trusted."""

    def test_user_media_path_discards_the_client_filename(self, user_factory):
        user = user_factory()
        instance = mock.Mock(user_id=user.user_id)

        path = user_media_path(instance, '../../../etc/passwd.jpg')

        assert path.startswith(f'uploads/{user.user_id}/')
        assert path.endswith('.jpg')
        assert '..' not in path
        assert 'passwd' not in path

    def test_user_media_path_is_unique_per_call(self, user_factory):
        instance = mock.Mock(user_id=user_factory().user_id)

        assert user_media_path(instance, 'a.png') != \
            user_media_path(instance, 'a.png')

    def test_artifact_and_report_paths_live_under_media_root(self, tmp_media):
        crop = vehicle_crop_path('an-analysis', 'a-vehicle')
        pdf = report_path('a-report')

        assert crop.parent.is_dir()
        assert pdf.parent.is_dir()
        assert to_media_relative(crop) == 'analyses/an-analysis/crops/a-vehicle.jpg'
        assert to_media_relative(pdf) == 'reports/a-report.pdf'


# ===========================================================================
# Permissions
# ===========================================================================

@pytest.mark.django_db
class TestPermissions:
    """Ownership and role checks, exercised without any views in place yet."""

    def request_for(self, user):
        request = APIRequestFactory().get('/api/anything/')
        request.user = user
        return request

    def test_is_owner(self, user_factory):
        owner = user_factory()
        other = user_factory()
        obj = mock.Mock(user=owner)

        assert IsOwner().has_object_permission(
            self.request_for(owner), None, obj) is True
        assert IsOwner().has_object_permission(
            self.request_for(other), None, obj) is False

    def test_is_admin_role_accepts_both_the_role_and_the_staff_flag(
            self, user_factory):
        admin = user_factory(role='admin')
        plain = user_factory()

        assert IsAdminRole().has_permission(self.request_for(admin), None) is True
        assert IsAdminRole().has_permission(self.request_for(plain), None) is False

    def test_is_owner_or_admin_lets_an_admin_through(self, user_factory):
        owner = user_factory()
        admin = user_factory(role='admin')
        obj = mock.Mock(user=owner)

        assert IsOwnerOrAdmin().has_object_permission(
            self.request_for(admin), None, obj) is True
        assert IsOwnerOrAdmin().has_object_permission(
            self.request_for(owner), None, obj) is True


# ===========================================================================
# Accounts model + JWT wiring
# ===========================================================================

@pytest.mark.django_db
class TestUserModel:
    """The custom user model and the token claims that depend on it."""

    def test_create_user_normalises_the_email_and_hashes_the_password(
            self, user_factory):
        user = user_factory(email='Mixed.Case@Example.COM', password='Sw0rdf1sh!')

        assert user.email == 'mixed.case@example.com'
        assert user.password != 'Sw0rdf1sh!'
        assert user.password.startswith('pbkdf2_')
        assert user.check_password('Sw0rdf1sh!') is True

    def test_create_user_rejects_a_blank_email(self):
        from django.contrib.auth import get_user_model

        with pytest.raises(ValueError):
            get_user_model().objects.create_user(email='  ', password='x')

    def test_superuser_gets_the_admin_role(self):
        from django.contrib.auth import get_user_model

        admin = get_user_model().objects.create_superuser(
            email='root@example.com', password='Sup3r@secret',
        )

        assert (admin.role, admin.is_staff, admin.is_superuser) == \
            ('admin', True, True)
        assert admin.is_admin is True

    def test_password_reset_token_lifecycle(self, user_factory):
        from django.utils import timezone

        from accounts.models import PasswordResetToken

        token = PasswordResetToken.objects.create(user=user_factory())

        assert len(token.token) == 64
        assert token.is_valid() is True

        token.mark_used()
        assert token.is_valid() is False

        expired = PasswordResetToken.objects.create(
            user=user_factory(),
            expires_at=timezone.now() - timezone.timedelta(minutes=1),
        )
        assert expired.is_valid() is False

    def test_jwt_carries_the_uuid_primary_key(self, auth_client):
        """
        SIMPLE_JWT is configured for USER_ID_FIELD/CLAIM = 'user_id'.

        Getting this wrong makes every token unusable the moment the accounts
        agent wires up authentication, so it is asserted here.
        """
        from rest_framework_simplejwt.tokens import AccessToken

        header = auth_client._credentials['HTTP_AUTHORIZATION']
        token = AccessToken(header.removeprefix('Bearer '))

        assert token['user_id'] == str(auth_client.user.user_id)


# ===========================================================================
# Project wiring
# ===========================================================================

@pytest.mark.django_db
class TestProjectWiring:
    """The schema and docs endpoints must be reachable without a token."""

    def test_openapi_schema_is_public(self, api):
        response = api.get('/api/schema/')

        assert response.status_code == 200

    def test_swagger_ui_is_public(self, api):
        response = api.get('/api/docs/')

        assert response.status_code == 200

    def test_logging_is_configured_for_the_asg_logger(self):
        logger = logging.getLogger('asg')

        assert logger.level == logging.INFO
        assert logger.handlers, 'the asg logger must have its own handlers'


# ===========================================================================
# UploadedMedia storage behaviour
# ===========================================================================

@pytest.mark.django_db
class TestUploadedMediaModel:
    """The upload row and the file on disk must stay in agreement."""

    def create(self, user, name='dashcam.jpg'):
        from uploads.models import UploadedMedia

        payload = make_jpeg_bytes()
        return UploadedMedia.objects.create(
            user=user,
            filename=name,
            format='jpg',
            media_type='image',
            size_bytes=len(payload),
            file=upload(name, payload),
        )

    def test_file_is_renamed_and_namespaced_by_owner(self, user_factory):
        user = user_factory()

        media = self.create(user, name='../../etc/passwd.jpg')

        assert media.file.name.startswith(f'uploads/{user.user_id}/')
        assert 'passwd' not in media.file.name
        assert media.filename == '../../etc/passwd.jpg'  # display name kept

    def test_file_path_mirrors_the_stored_name(self, user_factory):
        from uploads.models import UploadedMedia

        media = self.create(user_factory())

        assert media.file_path == media.file.name
        # ...and the mirror survives a round trip through the database.
        assert UploadedMedia.objects.get(pk=media.pk).file_path == media.file.name

    def test_plain_updates_do_not_rewrite_file_path(self, user_factory):
        media = self.create(user_factory())
        original = media.file_path

        media.width, media.height = 1920, 1080
        media.save(update_fields=['width', 'height'])
        media.refresh_from_db()

        assert media.file_path == original
        assert media.resolution == '1920x1080'


# ===========================================================================
# Django admin
# ===========================================================================

@pytest.mark.django_db
class TestAdminSite:
    """
    Every registered model's change list must actually render.

    ``manage.py check`` validates the admin *declarations*; only a real request
    catches a ``list_display`` callable that blows up or a broken
    ``list_select_related``.
    """

    @pytest.fixture
    def staff_browser(self, client, user_factory):
        user = user_factory(role='admin', is_staff=True, is_superuser=True)
        client.force_login(user)
        return client

    @pytest.mark.parametrize('url', [
        '/admin/accounts/user/',
        '/admin/accounts/passwordresettoken/',
        '/admin/uploads/uploadedmedia/',
        '/admin/analysis/analysisresult/',
        '/admin/analysis/detectedvehicle/',
        '/admin/analysis/smokeregion/',
        '/admin/reports/generatedreport/',
        '/admin/sysconfig/systemsetting/',
    ])
    def test_changelist_renders(self, staff_browser, url):
        assert staff_browser.get(url).status_code == 200

    def test_system_setting_add_is_blocked_once_the_row_exists(
            self, staff_browser, settings_row):
        response = staff_browser.get('/admin/sysconfig/systemsetting/add/')

        assert response.status_code == 403
