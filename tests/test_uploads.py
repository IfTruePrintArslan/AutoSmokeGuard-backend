"""
Uploads API tests.

Test names carry the project's test-design IDs where one exists (``TC-06``
rejects a bad file, ``TC-07`` accepts a good one); the rest cover the
behaviour requirements from the media section of the API contract:
UC-01/UC-04 (upload), UC-08-adjacent list/detail/delete, UC-09 (runtime
policy changes take effect immediately), de-duplication, ownership isolation,
pagination, and the 3-second acknowledgement NFR.
"""
import os
import time

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient

from analysis.models import STATUS_QUEUED, AnalysisResult
from uploads.models import UploadedMedia

from .conftest import (
    VIDEO_DURATION_SECONDS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    authenticate,
    make_jpeg_bytes,
    make_png_bytes,
)

ONE_MB = 1024 * 1024


def upload_file(auth_client, filename, data, content_type='application/octet-stream'):
    """POST ``data`` to ``/api/upload`` as ``filename`` through ``auth_client``."""
    return auth_client.post(
        '/api/upload',
        {'file': SimpleUploadedFile(filename, data, content_type=content_type)},
        format='multipart',
    )


# ===========================================================================
# TC-06 — rejected uploads
# ===========================================================================

@pytest.mark.django_db
class TestTC06RejectedUploads:
    """Files that must never reach the decoder are rejected with a specific code."""

    def test_disallowed_extension_is_rejected(self, auth_client):
        response = upload_file(auth_client, 'notes.txt', b'just some text', 'text/plain')

        assert response.status_code == 400
        assert response.data['code'] == 'invalid_format'
        assert "'.txt'" in response.data['detail']
        assert 'jpg' in response.data['detail']

    def test_text_content_disguised_as_jpeg_is_rejected(self, auth_client):
        """A `.jpg` whose bytes are plain text fails the signature/decode gate."""
        response = upload_file(
            auth_client, 'fake.jpg', b'this is not a jpeg, just text padding' * 4,
            'image/jpeg',
        )

        assert response.status_code == 400
        assert response.data['code'] == 'corrupt_file'
        assert response.data['detail']


# ===========================================================================
# TC-07 — accepted uploads
# ===========================================================================

@pytest.mark.django_db
class TestTC07AcceptedUploads:
    """Genuine JPEG, PNG and MP4 uploads succeed and are probed correctly."""

    def test_valid_jpeg_is_accepted(self, auth_client, sample_files):
        response = upload_file(auth_client, 'photo.jpg', sample_files['jpg'], 'image/jpeg')

        assert response.status_code == 201
        body = response.data
        assert body['media_id']
        assert body['format'] == 'jpg'
        assert body['media_type'] == 'image'
        assert body['width'] == 32
        assert body['height'] == 24

        media = UploadedMedia.objects.get(media_id=body['media_id'])
        assert media.user == auth_client.user
        assert os.path.exists(media.file.path)
        assert f'uploads/{auth_client.user.user_id}/' in media.file.name

    def test_valid_png_is_accepted(self, auth_client, sample_files):
        response = upload_file(auth_client, 'photo.png', sample_files['png'], 'image/png')

        assert response.status_code == 201
        body = response.data
        assert body['format'] == 'png'
        assert body['media_type'] == 'image'
        assert body['width'] == 32
        assert body['height'] == 24

        media = UploadedMedia.objects.get(media_id=body['media_id'])
        assert os.path.exists(media.file.path)

    def test_valid_mp4_is_accepted(self, auth_client, sample_files):
        """
        ``sample_files['mp4']`` is a genuinely decodable clip, so the probe
        step runs for real: the width, height and duration asserted below are
        what OpenCV read back out of the stored file, not a stubbed answer.
        """
        response = upload_file(auth_client, 'clip.mp4', sample_files['mp4'], 'video/mp4')

        assert response.status_code == 201
        body = response.data
        assert body['format'] == 'mp4'
        assert body['media_type'] == 'video'
        assert body['width'] == VIDEO_WIDTH
        assert body['height'] == VIDEO_HEIGHT
        assert body['duration_seconds'] == pytest.approx(
            VIDEO_DURATION_SECONDS, abs=0.05)

        media = UploadedMedia.objects.get(media_id=body['media_id'])
        assert os.path.exists(media.file.path)

    def test_valid_avi_is_accepted(self, auth_client, sample_files):
        """The other accepted video container decodes for real too."""
        response = upload_file(auth_client, 'clip.avi', sample_files['avi'], 'video/x-msvideo')

        assert response.status_code == 201
        body = response.data
        assert body['format'] == 'avi'
        assert body['media_type'] == 'video'
        assert body['width'] == VIDEO_WIDTH
        assert body['height'] == VIDEO_HEIGHT
        assert body['duration_seconds'] == pytest.approx(
            VIDEO_DURATION_SECONDS, abs=0.05)


# ===========================================================================
# Size, emptiness, presence, and authentication gates
# ===========================================================================

@pytest.mark.django_db
class TestUploadGates:

    def test_oversized_upload_is_rejected(self, auth_client, sample_files, settings_row):
        settings_row.max_upload_mb = 1
        settings_row.save()

        oversized = sample_files['jpg'] + (b'\x00' * (2 * ONE_MB))
        response = upload_file(auth_client, 'big.jpg', oversized, 'image/jpeg')

        assert response.status_code == 400
        assert response.data['code'] == 'file_too_large'
        assert '1 MB' in response.data['detail']

    def test_empty_file_is_rejected(self, auth_client):
        response = upload_file(auth_client, 'empty.jpg', b'', 'image/jpeg')

        assert response.status_code == 400
        assert response.data['code'] == 'empty_file'

    def test_missing_file_field_is_rejected(self, auth_client):
        response = auth_client.post('/api/upload', {}, format='multipart')

        assert response.status_code == 400
        assert response.data['code'] == 'no_file'

    def test_unauthenticated_upload_is_rejected(self, api, sample_files):
        response = api.post(
            '/api/upload',
            {'file': SimpleUploadedFile('photo.jpg', sample_files['jpg'], content_type='image/jpeg')},
            format='multipart',
        )

        assert response.status_code == 401

    def test_video_longer_than_limit_is_rejected(self, auth_client, sample_files, settings_row):
        """
        The limit is measured against a real decode, not a stubbed duration.

        The fixture clip runs for ``VIDEO_DURATION_SECONDS``; the ceiling is
        set just under it so OpenCV itself supplies the number the gate
        rejects on.
        """
        settings_row.max_video_seconds = 1
        settings_row.save()
        assert VIDEO_DURATION_SECONDS > 1, 'fixture clip is too short to test the limit'

        response = upload_file(auth_client, 'toolong.mp4', sample_files['mp4'], 'video/mp4')

        assert response.status_code == 400
        assert response.data['code'] == 'video_too_long'
        assert 'the limit is 1s' in response.data['detail']
        # The row created before the probe ran must be rolled back — a
        # rejected upload leaves nothing behind.
        assert not UploadedMedia.objects.filter(user=auth_client.user).exists()


# ===========================================================================
# UC-09 — runtime policy changes take effect immediately
# ===========================================================================

@pytest.mark.django_db
class TestRuntimePolicy:

    def test_excluding_png_from_settings_rejects_a_png_upload(self, auth_client, sample_files, settings_row):
        settings_row.allowed_image_formats = 'jpg,jpeg'
        settings_row.save()

        response = upload_file(auth_client, 'photo.png', sample_files['png'], 'image/png')

        assert response.status_code == 400
        assert response.data['code'] == 'invalid_format'


# ===========================================================================
# De-duplication
# ===========================================================================

@pytest.mark.django_db
class TestDeduplication:

    def test_duplicate_bytes_reuse_the_existing_row(self, auth_client, sample_files):
        first = upload_file(auth_client, 'a.jpg', sample_files['jpg'], 'image/jpeg')
        assert first.status_code == 201
        assert not first.data.get('deduplicated')
        media_id = first.data['media_id']

        second = upload_file(auth_client, 'a-copy.jpg', sample_files['jpg'], 'image/jpeg')

        assert second.status_code == 200
        assert second.data['media_id'] == media_id
        assert second.data['deduplicated'] is True
        assert UploadedMedia.objects.filter(user=auth_client.user).count() == 1


# ===========================================================================
# List / detail / delete — ownership, isolation, admin visibility
# ===========================================================================

@pytest.mark.django_db
class TestListDetailDelete:

    def _two_users_with_media(self, user_factory, sample_files):
        user_a = user_factory()
        user_b = user_factory()
        client_a = authenticate(APIClient(), user_a)
        client_b = authenticate(APIClient(), user_b)

        resp_a = upload_file(client_a, 'a.jpg', sample_files['jpg'], 'image/jpeg')
        resp_b = upload_file(client_b, 'b.png', sample_files['png'], 'image/png')
        assert resp_a.status_code == 201
        assert resp_b.status_code == 201

        return client_a, client_b, resp_a.data, resp_b.data

    def test_list_is_isolated_per_owner(self, user_factory, sample_files):
        client_a, client_b, media_a, media_b = self._two_users_with_media(
            user_factory, sample_files,
        )

        list_a = client_a.get('/api/media')
        assert list_a.status_code == 200
        ids_a = {row['media_id'] for row in list_a.data['results']}
        assert ids_a == {media_a['media_id']}

        list_b = client_b.get('/api/media')
        ids_b = {row['media_id'] for row in list_b.data['results']}
        assert ids_b == {media_b['media_id']}

    def test_admin_sees_every_users_media(self, user_factory, sample_files, admin_client):
        _client_a, _client_b, media_a, media_b = self._two_users_with_media(
            user_factory, sample_files,
        )

        listing = admin_client.get('/api/media')
        assert listing.status_code == 200
        ids = {row['media_id'] for row in listing.data['results']}
        assert {media_a['media_id'], media_b['media_id']} <= ids

    def test_detail_for_another_users_media_is_404(self, user_factory, sample_files):
        client_a, client_b, media_a, _media_b = self._two_users_with_media(
            user_factory, sample_files,
        )

        response = client_b.get(f"/api/media/{media_a['media_id']}")
        assert response.status_code == 404

    def test_delete_another_users_media_is_404(self, user_factory, sample_files):
        client_a, client_b, media_a, _media_b = self._two_users_with_media(
            user_factory, sample_files,
        )

        response = client_b.delete(f"/api/media/{media_a['media_id']}")
        assert response.status_code == 404
        assert UploadedMedia.objects.filter(media_id=media_a['media_id']).exists()

    def test_delete_own_media_removes_row_and_file(self, auth_client, sample_files):
        created = upload_file(auth_client, 'a.jpg', sample_files['jpg'], 'image/jpeg')
        assert created.status_code == 201
        media = UploadedMedia.objects.get(media_id=created.data['media_id'])
        file_path = media.file.path
        assert os.path.exists(file_path)

        response = auth_client.delete(f"/api/media/{created.data['media_id']}")

        assert response.status_code == 204
        assert not UploadedMedia.objects.filter(media_id=created.data['media_id']).exists()
        assert not os.path.exists(file_path)

    def test_delete_refused_while_analysis_in_progress(self, auth_client, sample_files):
        created = upload_file(auth_client, 'a.jpg', sample_files['jpg'], 'image/jpeg')
        assert created.status_code == 201
        media = UploadedMedia.objects.get(media_id=created.data['media_id'])
        AnalysisResult.objects.create(
            media=media, user=auth_client.user, status=STATUS_QUEUED,
        )

        response = auth_client.delete(f"/api/media/{created.data['media_id']}")

        assert response.status_code == 409
        assert response.data['code'] == 'analysis_in_progress'
        assert UploadedMedia.objects.filter(media_id=media.media_id).exists()


# ===========================================================================
# Pagination envelope
# ===========================================================================

@pytest.mark.django_db
class TestPagination:

    def test_envelope_keys_and_count_across_two_pages(self, auth_client):
        for index in range(15):
            colour = (index % 256, (index * 7) % 256, (index * 13) % 256)
            media = UploadedMedia(
                user=auth_client.user,
                filename=f'img-{index}.jpg',
                format='jpg',
                media_type='image',
                size_bytes=100,
                checksum=f'checksum-{index}',
                width=10,
                height=10,
            )
            media.file.save(
                f'img-{index}.jpg',
                SimpleUploadedFile(f'img-{index}.jpg', make_jpeg_bytes(colour=colour)),
                save=False,
            )
            media.save()

        page_one = auth_client.get('/api/media', {'page': 1, 'page_size': 10})
        assert page_one.status_code == 200
        body = page_one.data
        assert set(body) == {
            'count', 'page', 'pages', 'page_size', 'next', 'previous', 'results',
        }
        assert body['count'] == 15
        assert body['page'] == 1
        assert body['pages'] == 2
        assert body['page_size'] == 10
        assert len(body['results']) == 10

        page_two = auth_client.get('/api/media', {'page': 2, 'page_size': 10})
        assert page_two.status_code == 200
        assert page_two.data['count'] == 15
        assert len(page_two.data['results']) == 5


# ===========================================================================
# NFR — acknowledge within 3 seconds
# ===========================================================================

@pytest.mark.django_db
class TestUploadPerformance:

    def test_five_megabyte_upload_completes_under_three_seconds(self, auth_client):
        base = make_png_bytes(width=64, height=48)
        # Padding after the IEND chunk is ignored by decoders, same as
        # trailing bytes after a JPEG's EOI marker — the image stays valid
        # while the payload grows to the size this NFR needs to exercise.
        padded = base + (b'\x00' * (5 * ONE_MB - len(base)))

        started = time.perf_counter()
        response = upload_file(auth_client, 'big.png', padded, 'image/png')
        elapsed = time.perf_counter() - started

        assert response.status_code == 201
        assert elapsed < 3.0, f'upload took {elapsed:.2f}s, over the 3s NFR budget'
