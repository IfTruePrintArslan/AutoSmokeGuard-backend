"""
Shared pytest fixtures for the AutoSmokeGuard backend test suite.

The fixtures here are the contract the other API agents build their tests on:
grab ``auth_client`` and you have a DRF client that has genuinely logged in
through ``POST /api/login``; grab ``sample_files`` and you have real,
decodable JPEG/PNG/MP4/AVI payloads to POST at the upload endpoint.

Note that ``admin_client`` intentionally **shadows** pytest-django's built-in
fixture of the same name.  The built-in returns a plain Django test ``Client``
logged in via a session against ``django.contrib.auth``'s default user model;
this project uses JWT and a custom user, so the session-based version would be
useless here.
"""
import io
import itertools
import tempfile
import uuid
from pathlib import Path

import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework.test import APIClient

from accounts.models import ROLE_ADMIN, ROLE_USER
from system_config.models import SystemSetting

#: Password used for every generated test account. Satisfies the configured
#: validators (length >= 8, not numeric, not a common password).
DEFAULT_TEST_PASSWORD = 'Testpass@12345'


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def tmp_media(settings, tmp_path):
    """
    Point ``MEDIA_ROOT`` at a throw-away directory for every single test.

    Autouse and function-scoped on purpose: upload and report tests write real
    files, and without this they would accumulate in the developer's working
    copy and leak state between tests.  ``tmp_path`` is unique per test, so
    tests can also run in parallel safely.
    """
    media_root = tmp_path / 'media'
    media_root.mkdir(parents=True, exist_ok=True)
    settings.MEDIA_ROOT = str(media_root)
    return media_root


# ---------------------------------------------------------------------------
# Clients and users
# ---------------------------------------------------------------------------

@pytest.fixture
def api():
    """An unauthenticated DRF API client."""
    return APIClient()


@pytest.fixture
def user_factory(db):
    """
    Callable that creates users on demand.

    ::

        user = user_factory()                        # ordinary user
        boss = user_factory(role='admin')            # administrator
        ana  = user_factory(email='ana@example.com', full_name='Ana')

    E-mails are unique per call unless one is given explicitly, so a test can
    create as many users as it likes without collisions.
    """
    User = get_user_model()

    def _make(email=None, password=DEFAULT_TEST_PASSWORD, role=ROLE_USER,
              **extra):
        email = email or f'user-{uuid.uuid4().hex[:12]}@example.com'
        if role == ROLE_ADMIN:
            extra.setdefault('is_staff', True)
        return User.objects.create_user(
            email=email, password=password, role=role, **extra,
        )

    return _make


#: Source addresses handed to the login endpoint, one per call.
#:
#: ``POST /api/login`` is rate-limited to 10 attempts per minute *per IP*
#: (``accounts.views.LoginRateThrottle``) and DRF keeps those counters in the
#: process-wide cache, which pytest-django's transaction rollback does not
#: reset.  A suite with hundreds of authenticated tests would trip the limit
#: within seconds and start 429-ing unrelated tests.  Giving every fixture
#: login its own client address puts each one in its own throttle bucket,
#: which is both realistic (these are different clients) and side-effect free
#: — unlike clearing the shared cache, which would quietly destroy the
#: throttle state a test under way may be asserting on.
_login_source_ips = (f'10.{(n >> 16) & 0xFF}.{(n >> 8) & 0xFF}.{n & 0xFF}'
                     for n in itertools.count(1))


def authenticate(client, user, password=DEFAULT_TEST_PASSWORD):
    """
    Log ``user`` in through the real endpoint and attach the Bearer token.

    This deliberately goes through ``POST /api/login`` rather than minting a
    token with ``RefreshToken.for_user``.  Every authenticated test in the
    suite therefore exercises the production authentication path end to end —
    credential lookup, password check, active-account check, ``last_login``
    update and token issuance — instead of trusting that the shortcut and the
    endpoint agree.  A regression in login now fails hundreds of tests rather
    than none.

    The public surface is unchanged: the client comes back with its
    ``Authorization: Bearer ...`` header set and a ``.user`` attribute.
    """
    response = client.post(
        reverse('accounts-login'),
        {'email': user.email, 'password': password},
        format='json',
        REMOTE_ADDR=next(_login_source_ips),
    )
    assert response.status_code == 200, (
        f'login failed for {user.email}: {response.status_code} '
        f'{getattr(response, "data", response.content)}'
    )
    client.credentials(HTTP_AUTHORIZATION=f'Bearer {response.data["access"]}')
    client.user = user
    return client


@pytest.fixture
def auth_client(db, user_factory):
    """A DRF client authenticated as an ordinary user (``client.user``)."""
    return authenticate(APIClient(), user_factory())


@pytest.fixture
def admin_client(db, user_factory):
    """A DRF client authenticated as an administrator (``client.user``)."""
    return authenticate(APIClient(), user_factory(role=ROLE_ADMIN))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@pytest.fixture
def settings_row(db):
    """The ``SystemSetting`` singleton, created with defaults if absent."""
    return SystemSetting.get_solo()


# ---------------------------------------------------------------------------
# Sample media
# ---------------------------------------------------------------------------

def make_jpeg_bytes(width=32, height=24, colour=(120, 120, 120)):
    """A genuine, decodable JPEG — encoded by Pillow, not hand-faked."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new('RGB', (width, height), colour).save(buffer, format='JPEG')
    return buffer.getvalue()


def make_png_bytes(width=32, height=24, colour=(40, 160, 90)):
    """A genuine, decodable PNG — encoded by Pillow, not hand-faked."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new('RGB', (width, height), colour).save(buffer, format='PNG')
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Video fixtures
#
# These are genuinely decodable clips, not container stubs.  The previous
# fixtures were a hand-written ``ftyp`` box (and a RIFF header) with zero
# filler: they passed the magic-number sniff in ``common.validators`` but
# OpenCV could not open them, so every test that reached
# ``uploads.services.probe_video`` had to monkeypatch the probe away.  That
# left the one code path most likely to break on a real user's file — "does
# this actually decode?" — completely untested.
#
# ``cv2.VideoWriter`` encodes a real clip with no extra dependency: OpenCV is
# already required by ``mlcore``.  Eight 32x24 frames is ~1 KB for MP4 and
# ~7 KB for AVI, and encoding takes single-digit milliseconds.  The bytes are
# cached at module scope so the whole session pays for it once.
#
# ``VIDEO_FPS`` is deliberately low so the clip has a *usable duration*
# (8 frames / 4 fps = 2.0 s) while staying tiny: the "video too long" test
# needs a file that really is longer than a configurable limit.
# ---------------------------------------------------------------------------

#: Geometry and length of every generated test clip.
VIDEO_FRAMES = 8
VIDEO_WIDTH = 32
VIDEO_HEIGHT = 24
VIDEO_FPS = 4.0
VIDEO_DURATION_SECONDS = VIDEO_FRAMES / VIDEO_FPS      # 2.0

#: ``{extension: bytes}``, populated lazily by :func:`_encoded_video`.
_VIDEO_CACHE = {}


def _encoded_video(extension, fourcc):
    """
    Encode (once per session) a real ``extension`` clip and return its bytes.

    ``cv2.VideoWriter`` only writes to a path, so the clip is built in a
    temporary directory and slurped into memory; the cache means that happens
    a single time no matter how many tests ask for it.
    """
    if extension in _VIDEO_CACHE:
        return _VIDEO_CACHE[extension]

    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as workdir:
        path = Path(workdir) / f'fixture.{extension}'
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), VIDEO_FPS,
            (VIDEO_WIDTH, VIDEO_HEIGHT),
        )
        if not writer.isOpened():
            raise RuntimeError(
                f'OpenCV could not open a {fourcc!r} writer for .{extension}; '
                'the video fixtures need a working opencv-python build.'
            )
        try:
            for index in range(VIDEO_FRAMES):
                # A visibly changing frame, so a decoder that silently returns
                # the same buffer every time would be caught by a pixel check.
                frame = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), 30,
                                dtype=np.uint8)
                frame[:, :, index % 3] = 30 + (index * 25) % 220
                writer.write(frame)
        finally:
            writer.release()
        _VIDEO_CACHE[extension] = path.read_bytes()

    return _VIDEO_CACHE[extension]


def make_mp4_bytes():
    """
    A real, decodable MP4: 8 frames of 32x24 at 4 fps, MPEG-4 Part 2.

    ``mp4v`` rather than ``avc1`` because it is present in every
    ``opencv-python`` wheel, while H.264 encoding depends on how the local
    build was linked.  The container is still ISO-BMFF, so
    ``common.validators.sniff_format`` classifies it as ``'isobmff'`` exactly
    as a phone-recorded clip would be.
    """
    return _encoded_video('mp4', 'mp4v')


def make_avi_bytes():
    """
    A real, decodable AVI: the same 8 frames in a RIFF container, Motion JPEG.

    ``MJPG`` is the AVI codec that ships in every OpenCV wheel.
    """
    return _encoded_video('avi', 'MJPG')


@pytest.fixture
def sample_files():
    """
    Byte payloads for each accepted format, keyed by extension.

    ``sample_files['jpg']`` -> ``bytes``.  Handy for building
    ``SimpleUploadedFile`` instances in upload tests.

    Every entry is a real file that its decoder can actually open: Pillow
    encodes the stills, OpenCV encodes the clips.  Nothing here is a
    hand-faked header, so a test may rely on the upload pipeline genuinely
    probing width, height and duration.
    """
    return {
        'jpg': make_jpeg_bytes(),
        'png': make_png_bytes(),
        'mp4': make_mp4_bytes(),
        'avi': make_avi_bytes(),
    }
