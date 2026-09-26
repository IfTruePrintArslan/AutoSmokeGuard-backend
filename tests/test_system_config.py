"""
Tests for the system-configuration API (UC-09): ``GET``/``PATCH
/api/settings`` over the :class:`~system_config.models.SystemSetting`
singleton.

Grouped by concern: read access and authorisation, per-field and
cross-field validation, the CSV<->list format translation, partial-update
semantics, the computed ``runtime`` block (including its two documented
failure modes — no ``metrics.json``, no ``mlcore``), and the singleton
guarantee surviving repeated writes.
"""
import json
import sys

import pytest
from django.conf import settings as django_settings

from system_config.models import SystemSetting

SETTINGS_URL = '/api/settings'

#: Every top-level key the contract's ``SettingsObj`` promises.
CONTRACT_KEYS = {
    'allowed_image_formats', 'allowed_video_formats', 'max_upload_mb',
    'confidence_threshold', 'smoke_mask_threshold', 'severity_low_max',
    'severity_moderate_max', 'frame_sample_rate', 'auto_generate_pdf',
    'max_video_seconds', 'updated_at', 'updated_by', 'runtime',
}
RUNTIME_KEYS = {
    'device', 'segmenter_mode', 'yolo_weights_present',
    'segmenter_weights_present', 'worker_threads', 'model_metrics',
}


# ---------------------------------------------------------------------------
# Read access
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestGet:

    def test_normal_user_gets_full_object_with_list_formats(
            self, auth_client, settings_row):
        response = auth_client.get(SETTINGS_URL)

        assert response.status_code == 200
        data = response.data
        assert CONTRACT_KEYS <= set(data)
        assert data['allowed_image_formats'] == ['jpg', 'jpeg', 'png']
        assert data['allowed_video_formats'] == ['mp4', 'avi', 'mov']
        assert isinstance(data['allowed_image_formats'], list)
        assert isinstance(data['allowed_video_formats'], list)
        assert RUNTIME_KEYS == set(data['runtime'])

    def test_anonymous_is_rejected(self, api):
        response = api.get(SETTINGS_URL)

        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Write authorisation
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestPatchAuthorisation:

    def test_normal_user_gets_403_admin_required(self, auth_client, settings_row):
        response = auth_client.patch(
            SETTINGS_URL, {'max_upload_mb': 100}, format='json')

        assert response.status_code == 403
        assert response.data['code'] == 'admin_required'

    def test_anonymous_is_rejected(self, api):
        response = api.patch(SETTINGS_URL, {'max_upload_mb': 100}, format='json')

        assert response.status_code == 401

    def test_admin_updates_value_and_stamps_audit_columns(
            self, admin_client, settings_row):
        response = admin_client.patch(
            SETTINGS_URL, {'max_upload_mb': 256}, format='json')

        assert response.status_code == 200
        assert CONTRACT_KEYS <= set(response.data)
        assert response.data['max_upload_mb'] == 256
        assert response.data['updated_by'] == admin_client.user.email

        settings_row.refresh_from_db()
        assert settings_row.max_upload_mb == 256
        assert settings_row.updated_by_id == admin_client.user.user_id


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestValidation:

    def test_severity_low_max_must_be_below_moderate_when_both_sent(
            self, admin_client, settings_row):
        response = admin_client.patch(SETTINGS_URL, {
            'severity_low_max': 0.9, 'severity_moderate_max': 0.5,
        }, format='json')

        assert response.status_code == 400
        assert 'severity_low_max' in response.data['errors']

    def test_severity_low_max_checked_against_existing_moderate(
            self, admin_client, settings_row):
        # default severity_moderate_max is 0.66; sending only the low side
        # must still be compared against it.
        response = admin_client.patch(
            SETTINGS_URL, {'severity_low_max': 0.9}, format='json')

        assert response.status_code == 400
        assert 'severity_low_max' in response.data['errors']

    @pytest.mark.parametrize('field,value', [
        ('confidence_threshold', 1.5),
        ('confidence_threshold', -0.01),
        ('smoke_mask_threshold', 1.01),
        ('severity_low_max', 2),
        ('severity_moderate_max', -1),
    ])
    def test_thresholds_out_of_range(
            self, admin_client, settings_row, field, value):
        response = admin_client.patch(SETTINGS_URL, {field: value}, format='json')

        assert response.status_code == 400
        assert field in response.data['errors']

    def test_max_upload_mb_zero_rejected(self, admin_client, settings_row):
        response = admin_client.patch(
            SETTINGS_URL, {'max_upload_mb': 0}, format='json')

        assert response.status_code == 400
        assert 'max_upload_mb' in response.data['errors']

    def test_frame_sample_rate_zero_rejected(self, admin_client, settings_row):
        response = admin_client.patch(
            SETTINGS_URL, {'frame_sample_rate': 0}, format='json')

        assert response.status_code == 400
        assert 'frame_sample_rate' in response.data['errors']

    def test_max_video_seconds_too_large_rejected(self, admin_client, settings_row):
        response = admin_client.patch(
            SETTINGS_URL, {'max_video_seconds': 99999}, format='json')

        assert response.status_code == 400
        assert 'max_video_seconds' in response.data['errors']


# ---------------------------------------------------------------------------
# Format list <-> CSV translation
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestFormatNormalisation:

    def test_messy_input_is_normalised_and_deduplicated(
            self, admin_client, settings_row):
        response = admin_client.patch(SETTINGS_URL, {
            'allowed_image_formats': ['  .JPG ', 'png', 'png'],
        }, format='json')

        assert response.status_code == 200
        assert response.data['allowed_image_formats'] == ['jpg', 'png']

        settings_row.refresh_from_db()
        assert settings_row.allowed_image_formats == 'jpg,png'

    def test_empty_list_rejected(self, admin_client, settings_row):
        response = admin_client.patch(
            SETTINGS_URL, {'allowed_image_formats': []}, format='json')

        assert response.status_code == 400
        assert 'allowed_image_formats' in response.data['errors']

    def test_unsupported_format_names_the_supported_list(
            self, admin_client, settings_row):
        response = admin_client.patch(
            SETTINGS_URL, {'allowed_image_formats': ['exe']}, format='json')

        assert response.status_code == 400
        message = ' '.join(response.data['errors']['allowed_image_formats'])
        assert 'exe' in message
        for supported in ('jpg', 'jpeg', 'png', 'bmp', 'webp'):
            assert supported in message

    def test_token_with_dot_slash_or_space_rejected(
            self, admin_client, settings_row):
        response = admin_client.patch(
            SETTINGS_URL, {'allowed_video_formats': ['mp/4']}, format='json')

        assert response.status_code == 400
        assert 'allowed_video_formats' in response.data['errors']

    def test_non_list_non_string_rejected(self, admin_client, settings_row):
        response = admin_client.patch(
            SETTINGS_URL, {'allowed_image_formats': 42}, format='json')

        assert response.status_code == 400
        assert 'allowed_image_formats' in response.data['errors']


# ---------------------------------------------------------------------------
# Partial update semantics
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestPartialUpdate:

    def test_untouched_fields_keep_their_values(self, admin_client, settings_row):
        original_video_formats = settings_row.allowed_video_formats
        original_mask_threshold = settings_row.smoke_mask_threshold

        response = admin_client.patch(
            SETTINGS_URL, {'max_upload_mb': 200}, format='json')

        assert response.status_code == 200
        settings_row.refresh_from_db()
        assert settings_row.max_upload_mb == 200
        assert settings_row.allowed_video_formats == original_video_formats
        assert settings_row.smoke_mask_threshold == pytest.approx(original_mask_threshold)


# ---------------------------------------------------------------------------
# The computed `runtime` block
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestRuntimeBlock:

    def test_model_metrics_null_when_metrics_file_absent(
            self, auth_client, settings_row, monkeypatch, tmp_path):
        monkeypatch.setitem(django_settings.ASG, 'ML_ASSETS_DIR', tmp_path)

        response = auth_client.get(SETTINGS_URL)

        assert response.status_code == 200
        assert response.data['runtime']['model_metrics'] is None

    def test_model_metrics_parsed_from_metrics_json(
            self, auth_client, settings_row, monkeypatch, tmp_path):
        (tmp_path / 'metrics.json').write_text(json.dumps({
            'val': {
                'dice': 0.876543, 'iou': 0.7912, 'pixel_accuracy': 0.941,
                'precision': 0.9, 'recall': 0.85, 'f1': 0.87,
            },
            'val_at_0.5': {'dice': 0.5},
        }))
        monkeypatch.setitem(django_settings.ASG, 'ML_ASSETS_DIR', tmp_path)

        response = auth_client.get(SETTINGS_URL)

        assert response.status_code == 200
        assert response.data['runtime']['model_metrics'] == {
            'dice': 0.876543, 'iou': 0.7912, 'pixel_accuracy': 0.941,
        }

    def test_model_metrics_null_when_file_is_unparseable(
            self, auth_client, settings_row, monkeypatch, tmp_path):
        (tmp_path / 'metrics.json').write_text('{not valid json')
        monkeypatch.setitem(django_settings.ASG, 'ML_ASSETS_DIR', tmp_path)

        response = auth_client.get(SETTINGS_URL)

        assert response.status_code == 200
        assert response.data['runtime']['model_metrics'] is None

    def test_runtime_survives_a_missing_mlcore(
            self, auth_client, settings_row, monkeypatch):
        monkeypatch.setitem(sys.modules, 'mlcore', None)

        response = auth_client.get(SETTINGS_URL)

        assert response.status_code == 200
        runtime = response.data['runtime']
        assert runtime['device'] == 'unknown'
        assert runtime['segmenter_mode'] == 'unknown'


# ---------------------------------------------------------------------------
# Singleton guarantee
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestSingletonRoundTrip:

    def test_two_patches_still_a_single_row(self, admin_client, settings_row):
        first = admin_client.patch(
            SETTINGS_URL, {'max_upload_mb': 111}, format='json')
        second = admin_client.patch(
            SETTINGS_URL, {'max_upload_mb': 222}, format='json')

        assert first.status_code == 200
        assert second.status_code == 200
        assert SystemSetting.objects.count() == 1
        assert SystemSetting.objects.get().max_upload_mb == 222
