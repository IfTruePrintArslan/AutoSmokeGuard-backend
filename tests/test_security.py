"""
Security regression tests (OWASP Top 10 review, ``docs/SECURITY_REVIEW.md``).

Every test here pins down something the review either *found broken and
fixed* or *found working and must keep working*.  Test classes are named for
the OWASP category they cover and each finding-specific test carries its
``ASG-nn`` id in the docstring so a failure points straight at the section of
the review that explains it.

The centrepiece is :class:`TestA01AccessControlMatrix`, which walks the full
cross-tenant matrix: for all five object types a second user must get a
**404** — never a 403, never a 200 — because a 403 confirms the object
exists and that is itself a leak.
"""
import importlib
import logging
import os
import struct
import zlib
from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone

from analysis.models import STATUS_DONE, AnalysisResult, default_severity_counts
from common.middleware import MediaGuardMiddleware
from common.validators import (
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    check_image_dimensions,
)
from reports.services import generate_report_for_analysis
from uploads.models import UploadedMedia

from .conftest import authenticate, make_jpeg_bytes


# ---------------------------------------------------------------------------
# Builders — real ORM rows owned by a specific user.
# ---------------------------------------------------------------------------

def _make_media(user, filename='evidence.jpg'):
    """A real ``UploadedMedia`` row with a real (small) JPEG on disk."""
    data = make_jpeg_bytes()
    return UploadedMedia.objects.create(
        user=user, filename=filename, format='jpg', media_type='image',
        size_bytes=len(data),
        file=SimpleUploadedFile(filename, data, content_type='image/jpeg'),
        width=32, height=24,
    )


def _make_analysis(user, media=None):
    """A completed ``AnalysisResult`` belonging to ``user``."""
    media = media or _make_media(user)
    now = timezone.now()
    return AnalysisResult.objects.create(
        media=media, user=user, status=STATUS_DONE, progress=100, stage='',
        start_time=now, end_time=now, total_vehicles=0, total_smoke=0,
        frames_processed=1, avg_confidence=0.0, overall_severity='low',
        severity_counts=default_severity_counts(),
        settings_snapshot={
            'confidence_threshold': 0.35, 'smoke_mask_threshold': 0.5,
            'severity_low_max': 0.33, 'severity_moderate_max': 0.66,
            'frame_sample_rate': 5, 'max_video_seconds': 300,
        },
    )


def _png_with_declared_size(width, height):
    """
    A structurally valid PNG whose IHDR declares ``width`` x ``height``.

    The IDAT really does carry that many (zero) scanlines, so this is a
    genuine decompression bomb rather than a header-only forgery: anything
    that actually decodes it pays the full memory cost.
    """
    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', crc)

    compressor = zlib.compressobj(9)
    body = b''
    scanline = b'\x00' * (width + 1)          # filter byte + zeroed row
    for _ in range(height):
        body += compressor.compress(scanline)
    body += compressor.flush()

    ihdr = struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0)
    return (
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', ihdr)
        + chunk(b'IDAT', body)
        + chunk(b'IEND', b'')
    )


# ---------------------------------------------------------------------------
# A01 — Broken Access Control
# ---------------------------------------------------------------------------

class TestA01AccessControlMatrix:
    """
    The cross-tenant matrix: user B must not reach any of user A's objects.

    Five object types, every verb the contract exposes on them.  The assertion
    is always ``404`` — a ``403`` would be an information leak in its own
    right, so it is failed explicitly rather than tolerated.
    """

    @pytest.fixture
    def tenants(self, db, user_factory):
        """Two users; A owns a media, an analysis and a report. B owns nothing."""
        from rest_framework.test import APIClient

        alice = user_factory(email='alice-sec@example.com')
        bob = user_factory(email='bob-sec@example.com')

        media = _make_media(alice)
        analysis = _make_analysis(alice, media)
        report = generate_report_for_analysis(analysis)

        return {
            'client_b': authenticate(APIClient(), bob),
            'client_a': authenticate(APIClient(), alice),
            'media': media,
            'analysis': analysis,
            'report': report,
        }

    @staticmethod
    def _assert_not_found(response, what):
        assert response.status_code != 403, (
            f'{what} answered 403 — that confirms the object exists to a user '
            f'who may not see it. It must be 404.'
        )
        assert response.status_code == 404, (
            f'{what} answered {response.status_code}, expected 404'
        )

    def test_media_detail_is_404_for_other_user(self, tenants):
        """Object type 1/5: uploaded media."""
        response = tenants['client_b'].get(
            f"/api/media/{tenants['media'].media_id}")
        self._assert_not_found(response, 'GET /api/media/{id}')

    def test_media_delete_is_404_and_leaves_the_row(self, tenants):
        """Object type 1/5: a foreign delete must not destroy the row either."""
        response = tenants['client_b'].delete(
            f"/api/media/{tenants['media'].media_id}")
        self._assert_not_found(response, 'DELETE /api/media/{id}')
        assert UploadedMedia.objects.filter(
            media_id=tenants['media'].media_id).exists()

    def test_analysis_detail_is_404_for_other_user(self, tenants):
        """Object type 2/5: the analysis result."""
        response = tenants['client_b'].get(
            f"/api/analysis/{tenants['analysis'].analysis_id}")
        self._assert_not_found(response, 'GET /api/analysis/{id}')

    def test_analysis_delete_is_404_and_leaves_the_row(self, tenants):
        """Object type 2/5: a foreign delete must not destroy the analysis."""
        response = tenants['client_b'].delete(
            f"/api/analysis/{tenants['analysis'].analysis_id}")
        self._assert_not_found(response, 'DELETE /api/analysis/{id}')
        assert AnalysisResult.objects.filter(
            analysis_id=tenants['analysis'].analysis_id).exists()

    def test_status_is_404_for_other_user(self, tenants):
        """Object type 3/5: the job status poll."""
        response = tenants['client_b'].get(
            f"/api/status/{tenants['analysis'].analysis_id}")
        self._assert_not_found(response, 'GET /api/status/{job_id}')

    def test_report_json_is_404_for_other_user(self, tenants):
        """Object type 4/5: the report JSON."""
        response = tenants['client_b'].get(
            f"/api/report/{tenants['report'].report_id}")
        self._assert_not_found(response, 'GET /api/report/{id}')

    def test_report_pdf_download_is_404_for_other_user(self, tenants):
        """Object type 5/5: the report PDF stream."""
        response = tenants['client_b'].get(
            f"/api/download-report/{tenants['report'].report_id}")
        self._assert_not_found(response, 'GET /api/download-report/{id}')

    def test_report_generation_on_foreign_analysis_is_404(self, tenants):
        """Writing is scoped too, not just reading."""
        response = tenants['client_b'].post(
            f"/api/analysis/{tenants['analysis'].analysis_id}/report")
        self._assert_not_found(response, 'POST /api/analysis/{id}/report')

    def test_analyze_with_foreign_media_is_404(self, tenants):
        """A foreign media id must not be analysable by a non-owner."""
        response = tenants['client_b'].post(
            '/api/analyze',
            {'media_id': str(tenants['media'].media_id)},
            format='json',
        )
        self._assert_not_found(response, 'POST /api/analyze')

    def test_owner_can_still_reach_every_object(self, tenants):
        """The matrix must fail closed, not fail shut: A still has access."""
        client_a = tenants['client_a']
        assert client_a.get(
            f"/api/media/{tenants['media'].media_id}").status_code == 200
        assert client_a.get(
            f"/api/analysis/{tenants['analysis'].analysis_id}").status_code == 200
        assert client_a.get(
            f"/api/status/{tenants['analysis'].analysis_id}").status_code == 200
        assert client_a.get(
            f"/api/report/{tenants['report'].report_id}").status_code == 200
        assert client_a.get(
            f"/api/download-report/{tenants['report'].report_id}"
        ).status_code == 200

    def test_listings_never_include_another_users_rows(self, tenants):
        """The collection endpoints are scoped, not just the detail ones."""
        client_b = tenants['client_b']

        media_ids = {row['media_id']
                     for row in client_b.get('/api/media').data['results']}
        assert str(tenants['media'].media_id) not in media_ids

        report_ids = {row['report_id']
                      for row in client_b.get('/api/reports').data['results']}
        assert str(tenants['report'].report_id) not in report_ids

        analysis_ids = {row['analysis_id']
                        for row in client_b.get('/api/history').data['results']}
        assert str(tenants['analysis'].analysis_id) not in analysis_ids


class TestA01PrivilegeEscalation:
    """Server-side values must never be settable from a request body."""

    def test_register_ignores_client_supplied_role_and_ids(self, api, db):
        """ASG: self-registration is always role='user'."""
        response = api.post(
            '/api/register',
            {
                'email': 'escalate@example.com',
                'password': 'Testpass@12345',
                'role': 'admin',
                'is_staff': True,
                'is_superuser': True,
                'user_id': '00000000-0000-0000-0000-000000000001',
            },
            format='json',
        )
        assert response.status_code == 201
        user = response.data['user']
        assert user['role'] == 'user'
        assert user['user_id'] != '00000000-0000-0000-0000-000000000001'

        from django.contrib.auth import get_user_model
        row = get_user_model().objects.get(email='escalate@example.com')
        assert row.is_staff is False
        assert row.is_superuser is False

    def test_patch_me_cannot_change_role_or_email(self, auth_client):
        """Only ``full_name`` is writable on the profile."""
        original_email = auth_client.user.email
        response = auth_client.patch(
            '/api/me',
            {'role': 'admin', 'email': 'root@example.com',
             'full_name': 'Renamed', 'is_staff': True},
            format='json',
        )
        assert response.status_code == 200
        assert response.data['role'] == 'user'
        assert response.data['email'] == original_email
        assert response.data['full_name'] == 'Renamed'

        auth_client.user.refresh_from_db()
        assert auth_client.user.is_staff is False

    def test_upload_ignores_client_supplied_owner(self, auth_client, user_factory):
        """A spoofed ``user_id`` form field must not reassign ownership."""
        victim = user_factory(email='victim@example.com')
        data = make_jpeg_bytes()
        response = auth_client.post(
            '/api/upload',
            {
                'file': SimpleUploadedFile('x.jpg', data,
                                           content_type='image/jpeg'),
                'user_id': str(victim.user_id),
                'user': str(victim.user_id),
            },
            format='multipart',
        )
        assert response.status_code == 201
        media = UploadedMedia.objects.get(media_id=response.data['media_id'])
        assert media.user_id == auth_client.user.user_id
        # The on-disk path is namespaced by the *authenticated* owner.
        assert str(auth_client.user.user_id) in media.file.name
        assert str(victim.user_id) not in media.file.name

    def test_non_admin_cannot_patch_settings(self, auth_client):
        """``PATCH /api/settings`` is admin-only and says so with 403."""
        response = auth_client.patch(
            '/api/settings', {'max_upload_mb': 99999}, format='json')
        assert response.status_code == 403
        assert response.data['code'] == 'admin_required'

    def test_admin_can_patch_settings(self, admin_client):
        """The other half of the gate: a real admin is allowed through."""
        response = admin_client.patch(
            '/api/settings', {'max_upload_mb': 256}, format='json')
        assert response.status_code == 200
        assert response.data['max_upload_mb'] == 256

    def test_anonymous_is_rejected_from_every_object_endpoint(self, api, db,
                                                              user_factory):
        """No object endpoint is reachable without authentication."""
        owner = user_factory()
        media = _make_media(owner)
        analysis = _make_analysis(owner, media)
        report = generate_report_for_analysis(analysis)

        for url in (
            f'/api/media/{media.media_id}',
            f'/api/analysis/{analysis.analysis_id}',
            f'/api/status/{analysis.analysis_id}',
            f'/api/report/{report.report_id}',
            f'/api/download-report/{report.report_id}',
            '/api/media', '/api/reports', '/api/history', '/api/settings',
        ):
            assert api.get(url).status_code == 401, f'{url} was reachable'


class TestA01MediaGuard:
    """
    ``/media/`` tree hardening (finding ASG-04).

    The PDF report tree must not be reachable as a static file — it has an
    authenticated endpoint — and no media URL may traverse out of the root.
    """

    @pytest.mark.parametrize('relative', [
        'reports/abc.pdf',
        'reports/',
        'REPORTS/abc.pdf',
        'reports/nested/deep.pdf',
        './reports/abc.pdf',
    ])
    def test_reports_subtree_is_blocked(self, relative):
        assert MediaGuardMiddleware._denial_reason(relative) is not None

    @pytest.mark.parametrize('relative', [
        '../db.sqlite3',
        '../../etc/passwd',
        '..%2fdb.sqlite3',
        '%2e%2e/db.sqlite3',
        '/etc/passwd',
        '..\\db.sqlite3',
        'uploads/\x00.jpg',
    ])
    def test_traversal_is_blocked(self, relative):
        assert MediaGuardMiddleware._denial_reason(relative) is not None

    @pytest.mark.parametrize('relative', [
        'uploads/1234/abcd.jpg',
        'analyses/5678/preview.jpg',
        'analyses/5678/crops/veh.jpg',
        'analyses/5678/masks/smoke.png',
    ])
    def test_legitimate_media_is_still_served(self, relative):
        """The guard must not break the SPA's <img> sources."""
        assert MediaGuardMiddleware._denial_reason(relative) is None

    def test_report_pdf_is_not_reachable_over_http(self, client, db,
                                                   user_factory, settings):
        """End to end: the static route 404s while the API route works."""
        settings.DEBUG = True
        owner = user_factory()
        analysis = _make_analysis(owner)
        report = generate_report_for_analysis(analysis)

        response = client.get(f'/media/{report.report_path}')
        assert response.status_code == 404

    def test_owner_can_download_the_pdf_through_the_api(self, db, user_factory):
        """The supported path still returns the bytes."""
        from rest_framework.test import APIClient

        owner = user_factory()
        analysis = _make_analysis(owner)
        report = generate_report_for_analysis(analysis)
        client = authenticate(APIClient(), owner)

        response = client.get(f'/api/download-report/{report.report_id}')
        assert response.status_code == 200
        assert response['Content-Type'] == 'application/pdf'


# ---------------------------------------------------------------------------
# A02 — Cryptographic Failures
# ---------------------------------------------------------------------------

class TestA02Cryptography:
    """Password storage, secret-key handling and what reaches the logs."""

    def test_passwords_are_pbkdf2_hashed_never_stored_plain(self, db,
                                                            user_factory):
        """ASG: PBKDF2 is configured *and* actually used."""
        password = 'Testpass@12345'
        user = user_factory(password=password)
        user.refresh_from_db()

        assert user.password.startswith('pbkdf2_sha256$')
        assert password not in user.password
        assert user.check_password(password)
        assert not user.check_password('wrong')

    def test_login_response_never_echoes_the_password(self, api, user_factory):
        response = api.post(
            '/api/login',
            {'email': user_factory(email='echo@example.com').email,
             'password': 'Testpass@12345'},
            format='json',
        )
        assert response.status_code == 200
        assert 'password' not in str(response.data).lower()

    def test_password_reset_token_is_never_written_to_the_log(
        self, api, user_factory, caplog, settings,
    ):
        """
        ASG-05: the reset token is an account-takeover credential.

        Only a non-reversible reference may be logged — never the secret.
        """
        settings.DEBUG = True
        user = user_factory(email='logsafe@example.com')

        with caplog.at_level(logging.INFO, logger='asg.accounts'):
            response = api.post('/api/password-reset',
                                {'email': user.email}, format='json')

        assert response.status_code == 200
        token = response.data['debug_token']
        assert len(token) == 64

        logged = '\n'.join(record.getMessage() for record in caplog.records)
        assert token not in logged, 'the live reset token reached the log'
        # ...but the flow is still auditable.
        assert 'Password reset token issued' in logged
        assert 'ref=' in logged

    def test_token_reference_is_one_way(self):
        """The logged handle must not be reversible to the token."""
        from accounts.services import token_reference

        token = 'a' * 64
        reference = token_reference(token)
        assert reference != token
        assert reference not in token
        assert len(reference) == 12
        assert token_reference(token) == reference       # stable
        assert token_reference('b' * 64) != reference    # distinguishing

    def test_secret_key_fallback_is_refused_when_debug_is_off(self,
                                                              monkeypatch):
        """
        ASG-01: booting on the public dev key with DEBUG=False is an
        authentication bypass, because SECRET_KEY is the JWT signing key.
        The settings module must refuse to import rather than fail open.
        """
        from django.core.exceptions import ImproperlyConfigured

        monkeypatch.delenv('SECRET_KEY', raising=False)
        monkeypatch.setenv('DEBUG', 'False')
        # A .env file next to manage.py would re-seed SECRET_KEY on import.
        monkeypatch.setattr('dotenv.load_dotenv', lambda *a, **k: False)

        spec = importlib.util.spec_from_file_location(
            '_settings_under_test',
            Path(__file__).resolve().parent.parent / 'config' / 'settings.py',
        )
        module = importlib.util.module_from_spec(spec)
        with pytest.raises(ImproperlyConfigured, match='SECRET_KEY'):
            spec.loader.exec_module(module)

    def test_secret_key_from_env_is_accepted_when_debug_is_off(self,
                                                               monkeypatch):
        """The guard must not block a correctly configured deployment."""
        monkeypatch.setenv('SECRET_KEY', 'x' * 64)
        monkeypatch.setenv('DEBUG', 'False')
        monkeypatch.setattr('dotenv.load_dotenv', lambda *a, **k: False)

        spec = importlib.util.spec_from_file_location(
            '_settings_under_test_ok',
            Path(__file__).resolve().parent.parent / 'config' / 'settings.py',
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.SECRET_KEY == 'x' * 64
        assert module.SIMPLE_JWT['SIGNING_KEY'] == 'x' * 64
        assert module.DEBUG is False

    def test_jwt_signing_key_tracks_secret_key(self, settings):
        """A forged token is only impossible while these two agree."""
        assert settings.SIMPLE_JWT['SIGNING_KEY'] == settings.SECRET_KEY
        assert settings.SIMPLE_JWT['ALGORITHM'] == 'HS256'

    def test_a_token_signed_with_the_wrong_key_is_rejected(self, api, db,
                                                           user_factory):
        """The signature is actually verified, not just parsed."""
        import jwt as pyjwt

        user = user_factory()
        forged = pyjwt.encode(
            {'token_type': 'access', 'exp': 9999999999, 'iat': 1,
             'jti': 'deadbeef', 'user_id': str(user.user_id)},
            'not-the-signing-key', algorithm='HS256',
        )
        api.credentials(HTTP_AUTHORIZATION=f'Bearer {forged}')
        assert api.get('/api/me').status_code == 401


# ---------------------------------------------------------------------------
# A03 — Injection
# ---------------------------------------------------------------------------

class TestA03Injection:
    """The ORM is used throughout; these pin the parameters that reach it."""

    @pytest.mark.parametrize('ordering', [
        'password', 'user__password', 'user__email',
        'id; DROP TABLE accounts_user--', "created_at'); DELETE FROM x--",
        '../../etc/passwd',
    ])
    def test_ordering_is_whitelisted(self, auth_client, ordering):
        """An unlisted sort key is a 400, not a silently honoured column."""
        response = auth_client.get('/api/history', {'ordering': ordering})
        assert response.status_code == 400

    @pytest.mark.parametrize('ordering', [
        'created_at', '-created_at', 'total_vehicles', '-total_vehicles',
        'avg_confidence', '-avg_confidence',
    ])
    def test_documented_ordering_values_are_accepted(self, auth_client,
                                                     ordering):
        response = auth_client.get('/api/history', {'ordering': ordering})
        assert response.status_code == 200

    @pytest.mark.parametrize('payload', [
        "' OR 1=1--",
        "%' UNION SELECT password FROM accounts_user--",
        "'; DROP TABLE uploads_media;--",
        '%',
        '_',
    ])
    def test_search_is_parameterised_and_matches_nothing(self, auth_client,
                                                         user_factory,
                                                         payload):
        """
        SQL metacharacters are data, not syntax — and the LIKE wildcards
        ``%``/``_`` are escaped, so they cannot broaden the match either.
        """
        _make_analysis(auth_client.user)

        response = auth_client.get('/api/history', {'search': payload})
        assert response.status_code == 200
        assert response.data['count'] == 0

        # The table is still there.
        assert AnalysisResult.objects.count() == 1

    def test_search_still_finds_a_genuine_substring(self, auth_client):
        """The filter has to actually work, or the test above proves nothing."""
        media = _make_media(auth_client.user, filename='junction-cam-7.jpg')
        _make_analysis(auth_client.user, media)

        response = auth_client.get('/api/history', {'search': 'junction'})
        assert response.status_code == 200
        assert response.data['count'] == 1

    @pytest.mark.parametrize('param,value', [
        ('status', 'bogus'),
        ('severity', '../../etc/passwd'),
        ('vehicle_type', "' OR '1'='1"),
    ])
    def test_choice_filters_reject_unknown_values(self, auth_client, param,
                                                  value):
        response = auth_client.get('/api/history', {param: value})
        assert response.status_code == 400

    @pytest.mark.parametrize('model_name', [
        '../../../../etc/passwd',
        '/etc/passwd',
        '../../db.sqlite3',
        '....//....//etc/passwd',
        '../ml_assets/../../../etc/hosts',
        'nonexistent-model',
    ])
    def test_analyze_model_setting_cannot_escape_the_assets_dir(
        self, auth_client, model_name, settings,
    ):
        """
        The ``model`` override is reduced to a basename that must already
        exist inside ``ML_ASSETS_DIR``; anything else is ignored entirely
        and the run falls back to the configured default.
        """
        settings.ASG = {**settings.ASG, 'WORKER_ENABLED': False}
        media = _make_media(auth_client.user)

        response = auth_client.post(
            '/api/analyze',
            {'media_id': str(media.media_id),
             'settings': {'model': model_name}},
            format='json',
        )
        assert response.status_code == 202

        analysis = AnalysisResult.objects.get(
            analysis_id=response.data['analysis_id'])
        snapshot = analysis.settings_snapshot
        assert 'yolo_weights' not in snapshot, (
            f'{model_name!r} resolved to a weights path: '
            f'{snapshot.get("yolo_weights")!r}'
        )
        assert 'model' not in snapshot

    def test_resolve_model_weights_rejects_traversal_directly(self):
        """Unit-level check of the resolver itself."""
        from analysis.services import _resolve_model_weights

        for candidate in ('../../../etc/passwd', '/etc/passwd',
                          '../../db.sqlite3', 'no-such-model.pt'):
            assert _resolve_model_weights(candidate) is None

    @pytest.mark.parametrize('stored', [
        '/etc/passwd',
        '../../etc/passwd',
        'reports/../../../etc/passwd',
        '../db.sqlite3',
    ])
    def test_from_media_relative_refuses_to_escape_media_root(self, stored,
                                                              tmp_media):
        """
        ASG-07: ``MEDIA_ROOT / value`` is not containment.

        pathlib discards the left operand when the right is absolute, so a
        tampered ``*_path`` column would otherwise resolve to any file on the
        box and be handed straight to ``open()``.
        """
        from django.core.exceptions import SuspiciousFileOperation

        from common.storage import from_media_relative

        with pytest.raises(SuspiciousFileOperation):
            from_media_relative(stored)

    @pytest.mark.parametrize('stored', [
        'reports/abc.pdf',
        'uploads/uid/file.jpg',
        'analyses/aid/preview.jpg',
        'analyses/aid/crops/v.jpg',
    ])
    def test_from_media_relative_accepts_legitimate_paths(self, stored,
                                                          tmp_media):
        from common.storage import from_media_relative

        resolved = from_media_relative(stored)
        assert str(resolved).startswith(str(Path(tmp_media).resolve()))

    def test_download_of_a_tampered_report_path_is_404_not_a_file_read(
        self, db, user_factory,
    ):
        """A poisoned ``report_path`` must not become an arbitrary file read."""
        from rest_framework.test import APIClient
        from reports.models import GeneratedReport

        owner = user_factory()
        analysis = _make_analysis(owner)
        report = generate_report_for_analysis(analysis)

        GeneratedReport.objects.filter(report_id=report.report_id).update(
            report_path='../../../../etc/passwd',
        )

        client = authenticate(APIClient(), owner)
        response = client.get(f'/api/download-report/{report.report_id}')
        assert response.status_code == 404
        assert response.data['code'] == 'report_not_found'

    def test_no_dynamically_built_sql_in_request_serving_code(self):
        """
        Any raw-SQL call must take a *literal* string.

        Parsed with ``ast`` rather than grepped, because the interesting
        question is not "is there a ``cursor.execute``" — the health probe
        has a perfectly safe ``cursor.execute('SELECT 1')`` — but "is the
        SQL built out of anything that a request could influence".  An
        f-string, a ``%`` format, a ``.format()`` call or a concatenation in
        that position is the bug; a bare constant is not.
        """
        import ast

        backend = Path(__file__).resolve().parent.parent
        raw_sql_calls = {'raw', 'extra', 'execute', 'executemany',
                         'RawSQL', 'RawQuery'}
        offenders = []

        for app in ('accounts', 'uploads', 'analysis', 'reports',
                    'system_config', 'common', 'health'):
            for path in (backend / app).rglob('*.py'):
                if 'migrations' in path.parts:
                    continue
                tree = ast.parse(path.read_text(encoding='utf-8'))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    name = (node.func.attr if isinstance(node.func, ast.Attribute)
                            else getattr(node.func, 'id', None))
                    if name not in raw_sql_calls or not node.args:
                        continue
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        continue    # a literal query — safe by construction
                    offenders.append(
                        f'{path.relative_to(backend)}:{node.lineno} '
                        f'{name}() with a non-literal query'
                    )

        assert not offenders, f'dynamically built SQL: {offenders}'


# ---------------------------------------------------------------------------
# A04 — Insecure Design (the upload pipeline)
# ---------------------------------------------------------------------------

class TestA04UploadPipeline:
    """
    Decompression bombs and undecodable media (finding ASG-02).

    A malicious file reaches Pillow, OpenCV and a PyTorch model, so the gate
    in front of them has to hold: reject cleanly with a contract error code,
    never 500, and never leave the rejected file or its row behind.
    """

    def test_decompression_bomb_is_rejected_cleanly(self, auth_client):
        """
        ASG-02: a 50000x50000 PNG is ~2.4 MB on disk and ~2.3 GiB decoded.

        It used to pass the size check, reach ``Image.open``, raise
        ``DecompressionBombError`` (which subclasses plain ``Exception`` and
        so escaped the decode guards), and 500 — stranding the file on disk.
        """
        bomb = _png_with_declared_size(50_000, 50_000)
        assert len(bomb) < 5 * 1024 * 1024, 'bomb must be small on disk'

        before_rows = UploadedMedia.objects.count()
        response = auth_client.post(
            '/api/upload',
            {'file': SimpleUploadedFile('bomb.png', bomb,
                                        content_type='image/png')},
            format='multipart',
        )

        assert response.status_code == 400, (
            f'expected a clean rejection, got {response.status_code}'
        )
        assert response.data['code'] == 'file_too_large'
        assert UploadedMedia.objects.count() == before_rows, (
            'the rejected upload left an orphaned row behind'
        )

    def test_bomb_in_pillows_warning_band_is_also_rejected(self, auth_client):
        """
        Between 1x and 2x its limit Pillow only *warns* and decodes anyway.

        60 MPix is above this project's 50 MPix policy but below Pillow's own
        100 MPix error threshold, so only our explicit check stops it.
        """
        image = _png_with_declared_size(8_000, 7_500)

        before_rows = UploadedMedia.objects.count()
        response = auth_client.post(
            '/api/upload',
            {'file': SimpleUploadedFile('mid.png', image,
                                        content_type='image/png')},
            format='multipart',
        )

        assert response.status_code == 400
        assert response.data['code'] == 'file_too_large'
        assert UploadedMedia.objects.count() == before_rows

    def test_no_orphan_file_is_left_on_disk_after_a_rejected_upload(
        self, auth_client, tmp_media,
    ):
        """The file must be removed too, not just the row."""
        bomb = _png_with_declared_size(50_000, 50_000)
        before = {p for p in Path(tmp_media).rglob('*') if p.is_file()}

        auth_client.post(
            '/api/upload',
            {'file': SimpleUploadedFile('bomb.png', bomb,
                                        content_type='image/png')},
            format='multipart',
        )

        after = {p for p in Path(tmp_media).rglob('*') if p.is_file()}
        assert after == before, f'orphaned file(s) left behind: {after - before}'

    def test_a_normal_image_still_uploads(self, auth_client):
        """The guard must not reject legitimate media."""
        response = auth_client.post(
            '/api/upload',
            {'file': SimpleUploadedFile('ok.jpg', make_jpeg_bytes(),
                                        content_type='image/jpeg')},
            format='multipart',
        )
        assert response.status_code == 201
        assert response.data['width'] == 32
        assert response.data['height'] == 24

    def test_zero_frame_video_is_rejected_not_crashed(self, auth_client):
        """A structurally valid container with no decodable frames."""
        header = b'ftypisom' + struct.pack('>I', 512) + b'isomiso2avc1mp41'
        payload = (struct.pack('>I', len(header) + 4) + header
                   + struct.pack('>I', 8) + b'free' + b'\x00' * 64)

        before_rows = UploadedMedia.objects.count()
        response = auth_client.post(
            '/api/upload',
            {'file': SimpleUploadedFile('zero.mp4', payload,
                                        content_type='video/mp4')},
            format='multipart',
        )

        assert response.status_code == 400
        assert response.data['code'] == 'corrupt_file'
        assert UploadedMedia.objects.count() == before_rows

    @pytest.mark.parametrize('width,height,ok', [
        (1920, 1080, True),
        (32, 24, True),
        (MAX_IMAGE_DIMENSION, 10, True),
        (MAX_IMAGE_DIMENSION + 1, 10, False),
        (10, MAX_IMAGE_DIMENSION + 1, False),
        (50_000, 50_000, False),
        (0, 100, False),
        (-5, 100, False),
    ])
    def test_check_image_dimensions_policy(self, width, height, ok):
        allowed, code, _message = check_image_dimensions(width, height)
        assert allowed is ok
        if not ok:
            assert code is not None

    def test_pixel_budget_is_bounded(self):
        """A sanity bound on the policy constant itself."""
        assert 0 < MAX_IMAGE_PIXELS <= 100_000_000
        assert 0 < MAX_IMAGE_DIMENSION <= 50_000


# ---------------------------------------------------------------------------
# A05 — Security Misconfiguration
# ---------------------------------------------------------------------------

class TestA05Configuration:
    """CORS, headers, and the shape of a 5xx."""

    def test_cors_is_an_allowlist_not_a_wildcard(self, settings):
        assert getattr(settings, 'CORS_ALLOW_ALL_ORIGINS', False) is False
        assert getattr(settings, 'CORS_ORIGIN_ALLOW_ALL', False) is False
        assert settings.CORS_ALLOWED_ORIGINS
        assert '*' not in settings.CORS_ALLOWED_ORIGINS

    def test_foreign_origin_gets_no_allow_origin_header(self, client):
        response = client.options(
            '/api/health/',
            HTTP_ORIGIN='http://evil.example.com',
            HTTP_ACCESS_CONTROL_REQUEST_METHOD='GET',
        )
        assert 'Access-Control-Allow-Origin' not in response

    def test_allowed_origin_is_echoed(self, client):
        response = client.options(
            '/api/health/',
            HTTP_ORIGIN='http://localhost:5173',
            HTTP_ACCESS_CONTROL_REQUEST_METHOD='GET',
        )
        assert response['Access-Control-Allow-Origin'] == 'http://localhost:5173'

    def test_security_headers_are_present(self, client):
        response = client.get('/api/health/')
        assert response['X-Content-Type-Options'] == 'nosniff'
        assert response['X-Frame-Options'] == 'DENY'

    def test_hardening_settings_are_declared(self, settings):
        assert settings.SECURE_CONTENT_TYPE_NOSNIFF is True
        assert settings.X_FRAME_OPTIONS == 'DENY'

    def test_unhandled_exception_returns_a_safe_envelope(self, auth_client,
                                                         monkeypatch):
        """
        A 5xx must never carry a traceback, a file path or an exception
        message — just the opaque envelope.
        """
        def explode(*args, **kwargs):
            raise RuntimeError(
                'boom: /Users/secret/path/settings.py line 42 '
                'password=hunter2'
            )

        monkeypatch.setattr('analysis.services.dashboard_stats', explode)

        response = auth_client.get('/api/dashboard/stats')
        assert response.status_code == 500
        assert response.data == {
            'detail': 'Internal Server Error',
            'code': 'server_error',
            'errors': None,
        }
        body = str(response.data)
        assert 'boom' not in body
        assert 'Traceback' not in body
        assert 'settings.py' not in body
        assert 'hunter2' not in body

    def test_404_body_is_the_standard_envelope(self, auth_client):
        response = auth_client.get(
            '/api/report/00000000-0000-0000-0000-000000000000')
        assert response.status_code == 404
        assert set(response.data) == {'detail', 'code', 'errors'}


# ---------------------------------------------------------------------------
# A07 — Identification and Authentication Failures
# ---------------------------------------------------------------------------

class TestA07Authentication:
    """Enumeration, token lifetimes, rotation and revocation."""

    def test_login_does_not_enumerate_users_by_response(self, api,
                                                        user_factory):
        user_factory(email='real@example.com')

        missing = api.post(
            '/api/login',
            {'email': 'ghost@example.com', 'password': 'Testpass@12345'},
            format='json', REMOTE_ADDR='10.99.0.1',
        )
        wrong = api.post(
            '/api/login',
            {'email': 'real@example.com', 'password': 'WrongPass@9999'},
            format='json', REMOTE_ADDR='10.99.0.2',
        )

        assert missing.status_code == wrong.status_code == 401
        assert missing.data == wrong.data

    def test_login_hashes_a_password_even_when_the_user_is_missing(
        self, api, user_factory, monkeypatch,
    ):
        """
        ASG-06: the identical bodies above were undermined by timing — the
        miss path used to skip PBKDF2 entirely (~0.5 ms vs ~97 ms, a ~200x
        tell).  Asserting on wall-clock would be flaky, so this asserts on
        the mechanism: a hash must be computed on *both* paths.
        """
        calls = []
        from django.contrib.auth.hashers import PBKDF2PasswordHasher

        original = PBKDF2PasswordHasher.encode

        def counting_encode(self, password, salt, iterations=None):
            calls.append(password)
            return original(self, password, salt, iterations)

        monkeypatch.setattr(PBKDF2PasswordHasher, 'encode', counting_encode)

        api.post(
            '/api/login',
            {'email': 'definitely-not-here@example.com',
             'password': 'Testpass@12345'},
            format='json', REMOTE_ADDR='10.99.1.1',
        )
        assert calls, (
            'no password hash was computed for a non-existent account — '
            'the timing oracle is back'
        )

    def test_password_reset_does_not_enumerate_users(self, api, user_factory,
                                                     settings):
        settings.DEBUG = False
        user_factory(email='known-a@example.com')

        known = api.post('/api/password-reset',
                         {'email': 'known-a@example.com'}, format='json',
                         REMOTE_ADDR='10.98.0.1')
        unknown = api.post('/api/password-reset',
                           {'email': 'ghost-a@example.com'}, format='json',
                           REMOTE_ADDR='10.98.0.2')

        assert known.status_code == unknown.status_code == 200
        assert known.data == unknown.data

    def test_login_throttle_fires(self, api, user_factory):
        user_factory(email='throttled@example.com')
        codes = [
            api.post('/api/login',
                     {'email': 'throttled@example.com', 'password': 'nope'},
                     format='json', REMOTE_ADDR='10.97.0.1').status_code
            for _ in range(12)
        ]
        assert 429 in codes, 'the login throttle never fired'

    def test_password_reset_throttle_fires(self, api, user_factory):
        user_factory(email='rst@example.com')
        codes = [
            api.post('/api/password-reset', {'email': 'rst@example.com'},
                     format='json', REMOTE_ADDR='10.96.0.1').status_code
            for _ in range(8)
        ]
        assert 429 in codes, 'the password-reset throttle never fired'

    def test_register_throttle_fires(self, api, db):
        codes = [
            api.post('/api/register',
                     {'email': f'bulk{i}@example.com',
                      'password': 'Testpass@12345'},
                     format='json', REMOTE_ADDR='10.95.0.1').status_code
            for i in range(24)
        ]
        assert 429 in codes, 'the register throttle never fired'

    def test_token_lifetimes_are_bounded(self, settings):
        from datetime import timedelta

        assert settings.SIMPLE_JWT['ACCESS_TOKEN_LIFETIME'] <= timedelta(hours=1)
        assert settings.SIMPLE_JWT['REFRESH_TOKEN_LIFETIME'] <= timedelta(days=30)
        assert settings.SIMPLE_JWT['ROTATE_REFRESH_TOKENS'] is True
        assert settings.SIMPLE_JWT['BLACKLIST_AFTER_ROTATION'] is True

    def test_rotated_refresh_token_cannot_be_replayed(self, api, user_factory):
        user = user_factory(email='rotate@example.com')
        login = api.post('/api/login',
                         {'email': user.email, 'password': 'Testpass@12345'},
                         format='json', REMOTE_ADDR='10.94.0.1')
        original_refresh = login.data['refresh']

        first = api.post('/api/refresh-token',
                         {'refresh': original_refresh}, format='json')
        assert first.status_code == 200

        replay = api.post('/api/refresh-token',
                          {'refresh': original_refresh}, format='json')
        assert replay.status_code == 401
        assert replay.data['code'] == 'token_not_valid'

    def test_password_reset_revokes_every_refresh_token(self, api,
                                                        user_factory,
                                                        settings):
        """A reset must sever the refresh chain, so sessions cannot renew."""
        settings.DEBUG = True
        user = user_factory(email='revoke@example.com')
        login = api.post('/api/login',
                         {'email': user.email, 'password': 'Testpass@12345'},
                         format='json', REMOTE_ADDR='10.93.0.1')
        old_refresh = login.data['refresh']

        token = api.post('/api/password-reset', {'email': user.email},
                         format='json',
                         REMOTE_ADDR='10.93.0.2').data['debug_token']
        confirmed = api.post(
            '/api/password-reset/confirm',
            {'token': token, 'password': 'Rotated@98765'}, format='json',
        )
        assert confirmed.status_code == 200

        replay = api.post('/api/refresh-token', {'refresh': old_refresh},
                          format='json')
        assert replay.status_code == 401

    def test_reset_token_is_single_use(self, api, user_factory, settings):
        settings.DEBUG = True
        user = user_factory(email='once@example.com')
        token = api.post('/api/password-reset', {'email': user.email},
                         format='json',
                         REMOTE_ADDR='10.92.0.1').data['debug_token']

        first = api.post('/api/password-reset/confirm',
                         {'token': token, 'password': 'Firstuse@12345'},
                         format='json')
        assert first.status_code == 200

        second = api.post('/api/password-reset/confirm',
                          {'token': token, 'password': 'Seconduse@12345'},
                          format='json')
        assert second.status_code == 400
        assert second.data['code'] == 'invalid_reset_token'

    def test_debug_token_is_absent_when_debug_is_off(self, api, user_factory,
                                                     settings):
        """The dev-only delivery channel must close in production."""
        settings.DEBUG = False
        user = user_factory(email='nodebug-sec@example.com')
        response = api.post('/api/password-reset', {'email': user.email},
                            format='json', REMOTE_ADDR='10.91.0.1')
        assert response.status_code == 200
        assert 'debug_token' not in response.data

    def test_disabled_account_cannot_log_in(self, api, user_factory):
        user = user_factory(email='disabled@example.com')
        user.is_active = False
        user.save(update_fields=['is_active'])

        response = api.post('/api/login',
                            {'email': user.email,
                             'password': 'Testpass@12345'},
                            format='json', REMOTE_ADDR='10.90.0.1')
        assert response.status_code == 401
        assert response.data['code'] == 'account_disabled'


class TestA07PasswordChangeInvalidatesAccessTokens:
    """
    ASG-08 — a password reset must kill the access token too, not just the
    refresh chain.

    The review measured the gap live: after a successful
    ``POST /api/password-reset/confirm`` the old *refresh* token correctly
    401'd, but the old *access* token still returned ``200`` on ``/api/me``
    for the remainder of its 30-minute lifetime.  The fix is
    ``accounts.User.password_changed_at`` (migration
    ``accounts/0002_user_password_changed_at``) enforced by
    ``accounts.authentication.PasswordChangeAwareJWTAuthentication``.

    These tests pin the whole contract: the rejection, the exact error code
    the SPA keys off, the two edge cases that would turn the control into an
    outage (NULL for pre-existing accounts, and second-granularity ``iat``
    against a microsecond timestamp), and the wiring that makes it apply to
    every endpoint at once.
    """

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _token_issued_at(user, issued_at):
        """
        A genuine, correctly signed access token whose ``iat`` is ``issued_at``.

        Everything else about the token is real — it is minted by SimpleJWT
        with the live signing key — so these tests exercise the new check and
        not some parallel code path.
        """
        from rest_framework_simplejwt.tokens import AccessToken

        token = AccessToken.for_user(user)
        token.set_iat(at_time=issued_at)
        return str(token)

    @staticmethod
    def _advance_clock(monkeypatch, delta):
        """
        Make ``django.utils.timezone.now()`` report ``delta`` into the future.

        Needed because the real elapsed time between "log in" and "reset the
        password" inside a test is milliseconds, which sits inside the
        deliberate two-second leeway (``accounts.authentication.LEEWAY``).
        Advancing Django's clock — and *not* SimpleJWT's, which reads the
        stdlib directly — reproduces the scenario that actually matters: a
        token minted ten minutes before the reset.
        """
        real_now = timezone.now
        monkeypatch.setattr(
            'django.utils.timezone.now', lambda: real_now() + delta,
        )

    def _reset_password(self, api, user, new_password, settings,
                        source_ip='10.89.0.1'):
        """Drive the real two-step reset flow; returns the new password."""
        settings.DEBUG = True
        requested = api.post('/api/password-reset', {'email': user.email},
                             format='json', REMOTE_ADDR=source_ip)
        assert requested.status_code == 200
        confirmed = api.post(
            '/api/password-reset/confirm',
            {'token': requested.data['debug_token'],
             'password': new_password},
            format='json',
        )
        assert confirmed.status_code == 200
        return new_password

    # -- the finding --------------------------------------------------------

    def test_access_token_issued_before_a_reset_is_rejected(
        self, api, user_factory, settings, monkeypatch,
    ):
        """
        ASG-08, the finding itself: the review's
        ``AFTER reset: old ACCESS -> HTTP 200`` must now be a 401.
        """
        from datetime import timedelta

        user = user_factory(email='asg08-stale@example.com')
        login = api.post('/api/login',
                         {'email': user.email, 'password': 'Testpass@12345'},
                         format='json', REMOTE_ADDR='10.89.1.1')
        assert login.status_code == 200
        old_access = login.data['access']

        # Establish that the token genuinely worked *before* the reset, so
        # the 401 below cannot be blamed on a broken token.
        api.credentials(HTTP_AUTHORIZATION=f'Bearer {old_access}')
        assert api.get('/api/me').status_code == 200
        api.credentials()

        self._advance_clock(monkeypatch, timedelta(minutes=10))
        self._reset_password(api, user, 'Rotated@24680', settings,
                             source_ip='10.89.1.2')

        api.credentials(HTTP_AUTHORIZATION=f'Bearer {old_access}')
        response = api.get('/api/me')

        assert response.status_code == 401, (
            'an access token minted before the password reset is still '
            'being accepted — ASG-08 has regressed'
        )

    def test_the_rejection_uses_the_token_not_valid_code(
        self, api, user_factory, settings, monkeypatch,
    ):
        """
        The code is load-bearing, not cosmetic.

        ``frontend/src/lib/api.js`` keys its single-flight
        refresh-then-sign-out logic off ``code == 'token_not_valid'``.  Any
        other code (DRF's default ``authentication_failed``, say) leaves the
        SPA on a dead-end error instead of returning the user to the login
        screen.
        """
        from datetime import timedelta

        user = user_factory(email='asg08-code@example.com')
        login = api.post('/api/login',
                         {'email': user.email, 'password': 'Testpass@12345'},
                         format='json', REMOTE_ADDR='10.89.2.1')
        old_access = login.data['access']

        self._advance_clock(monkeypatch, timedelta(minutes=10))
        self._reset_password(api, user, 'Rotated@13579', settings,
                             source_ip='10.89.2.2')

        api.credentials(HTTP_AUTHORIZATION=f'Bearer {old_access}')
        response = api.get('/api/me')

        assert response.status_code == 401
        assert response.data['code'] == 'token_not_valid'
        # ...and still the project's standard envelope, not SimpleJWT's
        # nested {"detail": ..., "messages": [...]} shape.
        assert set(response.data) == {'detail', 'code', 'errors'}
        assert response.data['errors'] is None

    def test_a_token_issued_after_the_change_is_accepted(self, api,
                                                         user_factory):
        """The control must not be indiscriminate: newer tokens still work."""
        from datetime import timedelta

        from accounts.models import User

        user = user_factory(email='asg08-fresh@example.com')
        # A password change an hour ago; the token below is minted now.
        User.objects.filter(pk=user.pk).update(
            password_changed_at=timezone.now() - timedelta(hours=1),
        )

        login = api.post('/api/login',
                         {'email': user.email, 'password': 'Testpass@12345'},
                         format='json', REMOTE_ADDR='10.89.3.1')
        assert login.status_code == 200

        api.credentials(HTTP_AUTHORIZATION=f'Bearer {login.data["access"]}')
        response = api.get('/api/me')

        assert response.status_code == 200
        assert response.data['email'] == user.email

    def test_login_immediately_after_a_reset_works(self, api, user_factory,
                                                   settings):
        """
        The clock-granularity edge case — and the one that would be an outage.

        ``iat`` is a whole number of seconds (RFC 7519 NumericDate) while
        ``password_changed_at`` keeps microseconds, so a token minted at
        12:00:00.900 carries ``iat = 12:00:00`` and *looks* older than a
        change recorded at 12:00:00.800.  Without
        ``accounts.authentication.LEEWAY`` the very next login after a reset
        would 401 on its own brand-new token, locking the user out of the
        account they had just recovered.  Nothing is mocked here on purpose:
        this runs at real speed, where the reset and the re-login land inside
        the same second.
        """
        user = user_factory(email='asg08-relogin@example.com')
        new_password = self._reset_password(api, user, 'Recovered@11223',
                                            settings, source_ip='10.89.4.1')

        login = api.post('/api/login',
                         {'email': user.email, 'password': new_password},
                         format='json', REMOTE_ADDR='10.89.4.2')
        assert login.status_code == 200, (
            'could not log in with the new password straight after a reset'
        )

        api.credentials(HTTP_AUTHORIZATION=f'Bearer {login.data["access"]}')
        me = api.get('/api/me')
        assert me.status_code == 200, (
            'the token issued by the post-reset login was rejected — the '
            'iat/password_changed_at comparison has no leeway'
        )
        assert me.data['email'] == user.email

    def test_leeway_tolerates_a_token_minted_in_the_same_second(
        self, api, user_factory,
    ):
        """The same edge case pinned directly, without relying on test speed."""
        from datetime import timedelta

        from accounts.authentication import LEEWAY
        from accounts.models import User

        assert LEEWAY > timedelta(0), (
            'a zero leeway cannot absorb the truncation of iat to whole seconds'
        )

        user = user_factory(email='asg08-leeway@example.com')
        changed_at = timezone.now()
        User.objects.filter(pk=user.pk).update(password_changed_at=changed_at)

        # Inside the leeway -> accepted; well outside it -> rejected.
        inside = self._token_issued_at(user, changed_at - LEEWAY / 2)
        outside = self._token_issued_at(user, changed_at - timedelta(minutes=5))

        api.credentials(HTTP_AUTHORIZATION=f'Bearer {inside}')
        assert api.get('/api/me').status_code == 200

        api.credentials(HTTP_AUTHORIZATION=f'Bearer {outside}')
        rejected = api.get('/api/me')
        assert rejected.status_code == 401
        assert rejected.data['code'] == 'token_not_valid'

    # -- the other edge case: rows that predate the migration ---------------

    def test_a_user_with_no_recorded_password_change_authenticates_normally(
        self, api, user_factory,
    ):
        """
        NULL means "no restriction", never "reject everything".

        Every account that existed before migration 0002 has NULL here — the
        migration deliberately does not backfill (see its docstring), because
        doing so would have signed the whole user base out at deploy time.
        If NULL were treated as "changed at the epoch" this would be a total
        outage, so it is pinned with a token whose ``iat`` is a year old.
        """
        from datetime import timedelta

        from accounts.models import User

        user = user_factory(email='asg08-legacy@example.com')
        # Reproduce a pre-migration row exactly.
        User.objects.filter(pk=user.pk).update(password_changed_at=None)
        user.refresh_from_db()
        assert user.password_changed_at is None

        login = api.post('/api/login',
                         {'email': user.email, 'password': 'Testpass@12345'},
                         format='json', REMOTE_ADDR='10.89.5.1')
        assert login.status_code == 200
        api.credentials(HTTP_AUTHORIZATION=f'Bearer {login.data["access"]}')
        assert api.get('/api/me').status_code == 200

        # Logging in reads but never writes the column, so it is still NULL —
        # and even a very old token must be honoured.
        user.refresh_from_db()
        assert user.password_changed_at is None

        ancient = self._token_issued_at(user,
                                        timezone.now() - timedelta(days=365))
        api.credentials(HTTP_AUTHORIZATION=f'Bearer {ancient}')
        assert api.get('/api/me').status_code == 200, (
            'a NULL password_changed_at must not lock an existing user out'
        )

    # -- the mechanism ------------------------------------------------------

    def test_set_password_is_the_single_place_the_stamp_is_written(
        self, db, user_factory,
    ):
        """
        The write lives on the model, not scattered across callers.

        ``set_password`` is the choke point every password change in the
        project already goes through (``create_user``, and therefore
        ``createsuperuser`` and the seed script; ``manage.py changepassword``;
        ``PasswordResetConfirmView``), so a future flow cannot forget to
        invalidate the old tokens.
        """
        from accounts.models import User

        # create_user -> set_password -> stamped and persisted.
        user = user_factory(email='asg08-stamp@example.com')
        user.refresh_from_db()
        first = user.password_changed_at
        assert first is not None, 'create_user did not stamp password_changed_at'

        # A later set_password moves it forward.
        user.set_password('Another@55555')
        user.save(update_fields=['password', 'password_changed_at'])
        user.refresh_from_db()
        assert user.password_changed_at > first

        # createsuperuser goes through the same path.
        boss = User.objects.create_superuser(
            email='asg08-root@example.com', password='Rootpass@12345',
        )
        boss.refresh_from_db()
        assert boss.password_changed_at is not None

    def test_password_reset_confirm_persists_the_stamp(self, api,
                                                       user_factory, settings):
        """
        ``save(update_fields=...)`` must name the column.

        This is the failure mode that would silently un-fix ASG-08: the model
        stamps the attribute in memory, the view saves only ``['password']``,
        and the database never learns about it.
        """
        user = user_factory(email='asg08-persist@example.com')
        assert user.password_changed_at is not None
        before = user.password_changed_at

        self._reset_password(api, user, 'Persisted@44556', settings,
                             source_ip='10.89.6.1')

        user.refresh_from_db()
        assert user.password_changed_at > before, (
            'the reset did not persist password_changed_at — check the '
            'update_fields list in PasswordResetConfirmView'
        )

    def test_the_guard_is_the_default_authentication_class(self, settings):
        """
        Wired globally, so no endpoint added later can quietly skip it.

        Asserting on the resolved class rather than the string catches a
        rename or a move that leaves the setting pointing at something that
        still imports but no longer performs the check.
        """
        from django.utils.module_loading import import_string
        from rest_framework_simplejwt.authentication import JWTAuthentication

        from accounts.authentication import (
            PasswordChangeAwareJWTAuthentication,
        )

        configured = settings.REST_FRAMEWORK['DEFAULT_AUTHENTICATION_CLASSES']
        assert len(configured) == 1
        resolved = import_string(configured[0])

        assert resolved is PasswordChangeAwareJWTAuthentication
        assert issubclass(resolved, JWTAuthentication)

    def test_the_published_schema_still_declares_bearer_jwt_auth(self, api):
        """
        Swapping the default authentication class must not blank the
        schema's `security` block.

        drf-spectacular matches its authentication extensions on the exact
        class path (``match_subclasses`` is off), so pointing
        DEFAULT_AUTHENTICATION_CLASSES at a subclass made it warn
        "could not resolve authenticator" for every authenticated view and
        drop `security` from the output — publishing a schema that tells
        clients the API needs no credentials, and turning
        `manage.py check --deploy` from clean into sixteen warnings.
        ``PasswordChangeAwareJWTScheme`` in accounts/authentication.py fixes
        it; this is the test that notices if it is ever removed.
        """
        import json

        response = api.get('/api/schema/?format=json')
        assert response.status_code == 200
        schema = json.loads(response.content)

        schemes = schema['components']['securitySchemes']
        assert 'jwtAuth' in schemes, (
            'the bearer-JWT security scheme vanished from the OpenAPI schema'
        )
        assert schemes['jwtAuth']['scheme'] == 'bearer'
        assert schemes['jwtAuth']['bearerFormat'] == 'JWT'

        secured = [
            path for path, ops in schema['paths'].items()
            for op in ops.values()
            if isinstance(op, dict)
            and any('jwtAuth' in entry for entry in op.get('security', []))
        ]
        assert secured, (
            'no operation in the schema requires jwtAuth any more'
        )

    def test_a_token_with_no_iat_claim_fails_closed(self, api, user_factory):
        """
        A token that cannot prove when it was issued cannot be shown to
        postdate the password change, so it is refused rather than waved
        through.  SimpleJWT always sets ``iat``, so nothing legitimate is
        affected.
        """
        from rest_framework_simplejwt.tokens import AccessToken

        user = user_factory(email='asg08-noiat@example.com')
        assert user.password_changed_at is not None

        token = AccessToken.for_user(user)
        del token.payload['iat']
        api.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')

        response = api.get('/api/me')
        assert response.status_code == 401
        assert response.data['code'] == 'token_not_valid'


# ---------------------------------------------------------------------------
# A08 — Software and Data Integrity
# ---------------------------------------------------------------------------

class TestA08Integrity:
    """Deserialisation of model checkpoints (finding ASG-03)."""

    def test_segmenter_loads_checkpoints_with_weights_only(self):
        """
        ``torch.load`` without ``weights_only=True`` is arbitrary code
        execution on a malicious checkpoint.  Asserted against the parsed
        source — not a grep, so the explanatory comment that names the unsafe
        value does not satisfy or defeat the check.
        """
        import ast

        source = (Path(__file__).resolve().parent.parent
                  / 'mlcore' / 'segmenter.py').read_text(encoding='utf-8')
        tree = ast.parse(source)

        loads = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else getattr(node.func, 'id', None))
            if name == 'load':
                loads.append(node)

        assert loads, 'no torch.load call found in segmenter.py'
        for call in loads:
            flag = next((kw.value for kw in call.keywords
                         if kw.arg == 'weights_only'), None)
            assert flag is not None, (
                f'torch.load at line {call.lineno} does not pass weights_only'
            )
            assert isinstance(flag, ast.Constant) and flag.value is True, (
                f'torch.load at line {call.lineno} passes '
                f'weights_only={ast.dump(flag)} — must be True'
            )

    def test_no_request_serving_module_loads_pickles_unsafely(self):
        """Nothing on the request path may call torch.load unsafely."""
        backend = Path(__file__).resolve().parent.parent
        offenders = []
        for app in ('accounts', 'uploads', 'analysis', 'reports',
                    'system_config', 'common', 'health'):
            for path in (backend / app).rglob('*.py'):
                source = path.read_text(encoding='utf-8')
                if 'weights_only=False' in source or 'pickle.load' in source:
                    offenders.append(str(path.relative_to(backend)))
        assert not offenders, f'unsafe deserialisation: {offenders}'

    def test_malicious_checkpoint_does_not_execute(self, tmp_path):
        """
        End to end: a pickle payload that would run a command must be
        refused by the loader the segmenter now uses.
        """
        torch = pytest.importorskip('torch')
        marker = tmp_path / 'pwned.txt'
        checkpoint = tmp_path / 'evil.pt'

        class Evil:
            def __reduce__(self):
                return (os.system, (f'touch {marker}',))

        torch.save({'state_dict': {}, 'base': 16, 'note': Evil()}, checkpoint)

        with pytest.raises(Exception):
            torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert not marker.exists(), 'the payload executed under weights_only=True'


# ---------------------------------------------------------------------------
# A09 — Security Logging
# ---------------------------------------------------------------------------

class TestA09Logging:
    """Enough to investigate an incident, nothing sensitive in the file."""

    def test_failed_login_is_logged_with_the_source_address(self, api,
                                                            user_factory,
                                                            caplog):
        user_factory(email='audit@example.com')
        with caplog.at_level(logging.WARNING, logger='asg.accounts'):
            api.post('/api/login',
                     {'email': 'audit@example.com', 'password': 'wrong'},
                     format='json', REMOTE_ADDR='203.0.113.7')

        logged = '\n'.join(r.getMessage() for r in caplog.records)
        assert 'Failed login attempt' in logged
        assert '203.0.113.7' in logged

    def test_successful_login_is_logged_with_the_source_address(self, api,
                                                                user_factory,
                                                                caplog):
        user = user_factory(email='audit-ok@example.com')
        with caplog.at_level(logging.INFO, logger='asg.accounts'):
            api.post('/api/login',
                     {'email': user.email, 'password': 'Testpass@12345'},
                     format='json', REMOTE_ADDR='203.0.113.8')

        logged = '\n'.join(r.getMessage() for r in caplog.records)
        assert 'Successful login' in logged
        assert '203.0.113.8' in logged

    def test_password_is_never_logged_on_a_failed_login(self, api,
                                                        user_factory, caplog):
        user_factory(email='nolog@example.com')
        secret = 'SuperSecret@98765'
        with caplog.at_level(logging.DEBUG):
            api.post('/api/login',
                     {'email': 'nolog@example.com', 'password': secret},
                     format='json', REMOTE_ADDR='203.0.113.9')

        logged = '\n'.join(r.getMessage() for r in caplog.records)
        assert secret not in logged

    def test_admin_config_change_is_audited(self, admin_client, caplog):
        with caplog.at_level(logging.INFO, logger='asg.sysconfig'):
            response = admin_client.patch(
                '/api/settings', {'confidence_threshold': 0.42},
                format='json')
        assert response.status_code == 200

        logged = '\n'.join(r.getMessage() for r in caplog.records)
        assert 'confidence_threshold' in logged

    def test_rejected_admin_change_is_logged(self, auth_client, caplog):
        with caplog.at_level(logging.WARNING, logger='asg.sysconfig'):
            auth_client.patch('/api/settings', {'max_upload_mb': 1},
                              format='json')
        logged = '\n'.join(r.getMessage() for r in caplog.records)
        assert 'Non-admin' in logged

    def test_blocked_media_request_is_logged(self, client, caplog, settings):
        settings.DEBUG = True
        with caplog.at_level(logging.WARNING, logger='asg.media'):
            client.get('/media/reports/whatever.pdf')
        logged = '\n'.join(r.getMessage() for r in caplog.records)
        assert 'Blocked' in logged


# ---------------------------------------------------------------------------
# A10 — Server-Side Request Forgery
# ---------------------------------------------------------------------------

class TestA10SSRF:
    """No user input may reach a network call at runtime."""

    def test_no_outbound_http_in_request_serving_code(self):
        backend = Path(__file__).resolve().parent.parent
        needles = ('requests.get', 'requests.post', 'urllib.request.urlopen',
                   'urlopen(', 'httpx.', 'aiohttp.')
        offenders = []
        for app in ('accounts', 'uploads', 'analysis', 'reports',
                    'system_config', 'common', 'health'):
            for path in (backend / app).rglob('*.py'):
                source = path.read_text(encoding='utf-8')
                for needle in needles:
                    if needle in source:
                        offenders.append(
                            f'{path.relative_to(backend)}: {needle}')
        assert not offenders, f'outbound network call on the request path: {offenders}'
