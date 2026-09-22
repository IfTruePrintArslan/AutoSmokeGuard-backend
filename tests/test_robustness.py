"""
Regression tests for the hostile-input campaign in ``docs/ROBUSTNESS_REPORT.md``
plus the multi-worker defect the concurrent code review raised.

Every test here maps onto one numbered defect, and every one of them was
confirmed **red against the pre-fix code** before the fix was restored — a
regression test that passes without the fix protects nothing.  The mapping:

===========  =======================================================  ===========
Test prefix  Defect                                                   Report §
===========  =======================================================  ===========
``defect1``  >5 MB JSON body -> unhandled 500                         §A
``defect2``  bootstrap never runs under ``runserver --noreload``      §D
``defect3``  read-only ``MEDIA_ROOT`` -> unhandled 500 on upload      §E
``defect4``  malformed ``<uuid:...>`` -> HTML 404, not the envelope   §B
``defect5``  ``fps <= 0`` skips the ``max_video_seconds`` cap         §C
``defect6``  gunicorn boot sweep destroys a sibling's live analysis   review
===========  =======================================================  ===========
"""
import errno
import json
import math
import os
import socket
import stat
import uuid
from datetime import timedelta

import pytest
from django.core.exceptions import RequestDataTooBig, TooManyFieldsSent
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.http.multipartparser import MultiPartParserError
from django.test import RequestFactory
from django.urls import get_resolver, reverse
from django.utils import timezone

from analysis import worker
from analysis.models import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    AnalysisResult,
)
from analysis.services import RUNTIME_KEY
from common.exceptions import api_exception_handler, request_limit_failure
from uploads.models import UploadedMedia
from uploads.services import (
    FALLBACK_VIDEO_FPS,
    UploadError,
    probe_video,
    store_upload,
)

ANALYZE_URL_NAME = 'analysis-analyze'
UPLOAD_URL_NAME = 'media-upload'


# ---------------------------------------------------------------------------
# Local helpers — deliberately self-contained so this file does not import
# from another test module.
# ---------------------------------------------------------------------------

def envelope_keys(body):
    """The key set every error response in this project promises."""
    return set(body or {})


def make_media(user, sample_files, extension='mp4', media_type='video'):
    """A real ``UploadedMedia`` row with real bytes under ``MEDIA_ROOT``."""
    content = sample_files[extension]
    media = UploadedMedia(
        user=user,
        filename=f'clip.{extension}',
        format=extension,
        media_type=media_type,
        size_bytes=len(content),
        width=1280,
        height=720,
    )
    media.file.save(f'clip.{extension}', ContentFile(content), save=False)
    media.save()
    return media


def make_analysis(user, media, **overrides):
    """Seed an ``AnalysisResult`` straight into the database."""
    fields = {
        'media': media,
        'user': user,
        'status': STATUS_RUNNING,
        'progress': 10,
        'stage': 'detecting',
        'settings_snapshot': {'confidence_threshold': 0.35},
        'start_time': timezone.now(),
        'end_time': None,
    }
    fields.update(overrides)
    return AnalysisResult.objects.create(**fields)


def owner_stamp(host=None, pid=None, boot=None):
    """An ownership stamp shaped exactly like :func:`worker.process_owner`."""
    return {
        'host': socket.gethostname() if host is None else host,
        'pid': os.getpid() if pid is None else pid,
        'boot': uuid.uuid4().hex if boot is None else boot,
        'at': timezone.now().isoformat(),
    }


def snapshot_owned_by(owner):
    """A ``settings_snapshot`` carrying ``owner`` (or nothing, for ``None``)."""
    snapshot = {'confidence_threshold': 0.35}
    if owner is not None:
        snapshot[RUNTIME_KEY] = {'owner': owner}
    return snapshot


def a_dead_pid():
    """
    A pid that is guaranteed not to be running.

    ``os.fork`` + immediate ``_exit`` + ``waitpid`` reaps the child, so the
    pid is genuinely gone rather than merely unlikely to exist.
    """
    pid = os.fork()
    if pid == 0:                                      # pragma: no cover - child
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


class FakeCapture:
    """A ``cv2.VideoCapture`` stand-in with fully controlled metadata."""

    def __init__(self, props, opened=True):
        self._props = props
        self._opened = opened
        self.released = False

    def isOpened(self):                               # noqa: N802 - cv2's name
        return self._opened

    def get(self, prop):
        return self._props.get(prop, 0.0)

    def release(self):
        self.released = True


@pytest.fixture
def fake_capture(monkeypatch):
    """
    Patch ``cv2.VideoCapture`` so a container's reported metadata can be faked.

    The robustness report could not get any real encoder to emit a container
    that OpenCV reports as ``fps == 0`` (§C, "Verification method"), so the
    only way to exercise that branch is to control what the demuxer says.
    Everything else in :func:`probe_video` — including the code under test —
    is the real, unmodified function.
    """
    import cv2

    holder = {}

    def _install(frame_count, fps, width=640, height=480, opened=True):
        capture = FakeCapture({
            cv2.CAP_PROP_FRAME_COUNT: frame_count,
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
        }, opened=opened)
        holder['capture'] = capture
        monkeypatch.setattr(cv2, 'VideoCapture', lambda *a, **kw: capture)
        return capture

    _install.holder = holder
    return _install


# ===========================================================================
# DEFECT 1 (§A) — a request over Django's body limits must not be a 500
# ===========================================================================

@pytest.mark.django_db
def test_defect1_oversized_json_body_returns_413_not_500(auth_client, settings):
    """
    A JSON body over ``DATA_UPLOAD_MAX_MEMORY_SIZE`` is the caller's mistake.

    Django raises ``RequestDataTooBig`` from ``HttpRequest.body``; because a
    DRF view touches ``request.data`` *inside* ``APIView.dispatch``, DRF
    catches it and hands it to this project's handler rather than letting
    Django's outer middleware turn it into a 400.  The handler did not
    recognise the type, so it became an opaque 500 on every JSON endpoint.
    """
    settings.DATA_UPLOAD_MAX_MEMORY_SIZE = 1024

    response = auth_client.post(
        reverse(ANALYZE_URL_NAME),
        {'media_id': str(uuid.uuid4()), 'settings': {'padding': 'A' * 4096}},
        format='json',
    )

    assert response.status_code == 413
    body = response.json()
    assert body['code'] == 'request_too_large'
    assert body['errors'] is None
    assert envelope_keys(body) == {'detail', 'code', 'errors'}
    # The message must stay actionable and must not echo the payload back.
    assert 'AAAA' not in body['detail']


@pytest.mark.django_db
def test_defect1_too_many_form_fields_returns_413_not_500(auth_client, settings):
    """``TooManyFieldsSent`` is the same class of fault and took the same path."""
    settings.DATA_UPLOAD_MAX_NUMBER_FIELDS = 5

    response = auth_client.post(
        reverse(ANALYZE_URL_NAME),
        data='&'.join(f'field{n}=1' for n in range(50)),
        content_type='application/x-www-form-urlencoded',
    )

    assert response.status_code == 413
    body = response.json()
    assert body['code'] == 'too_many_fields'
    assert envelope_keys(body) == {'detail', 'code', 'errors'}


@pytest.mark.parametrize(('exception', 'code', 'status_code'), [
    (RequestDataTooBig('too big'), 'request_too_large', 413),
    (TooManyFieldsSent('too many'), 'too_many_fields', 413),
    # 400, not 413: a malformed multipart body is unparseable, not oversized.
    # Telling the client to shrink a request whose size was never the problem
    # would send them chasing the wrong fix.  Matches Django's own
    # response_for_exception and DRF's ParseError.
    (MultiPartParserError('bad boundary'), 'malformed_multipart', 400),
])
def test_defect1_body_limit_exceptions_map_to_the_envelope(
        exception, code, status_code):
    """Every Django body-limit exception has an explicit, non-500 mapping."""
    classified = request_limit_failure(exception)
    assert classified is not None, f'{type(exception).__name__} is unclassified'
    assert classified[0] == code
    assert classified[2] == status_code

    factory = RequestFactory()
    response = api_exception_handler(
        exception, {'request': factory.post('/api/analyze'), 'view': None},
    )
    assert response is not None
    assert response.status_code == status_code
    assert response.data['code'] == code
    assert envelope_keys(response.data) == {'detail', 'code', 'errors'}


def test_defect1_an_ordinary_exception_is_still_a_500():
    """The new branch must not swallow genuine server faults."""
    assert request_limit_failure(RuntimeError('boom')) is None

    factory = RequestFactory()
    response = api_exception_handler(
        RuntimeError('boom'),
        {'request': factory.post('/api/analyze'), 'view': None},
    )
    assert response.status_code == 500
    assert response.data['code'] == 'server_error'


# ===========================================================================
# DEFECT 2 (§D) — bootstrap must run in every real serving context
# ===========================================================================

def test_defect2_bootstrap_runs_under_runserver_noreload():
    """
    The reported bug, exactly.

    ``--noreload`` starts no autoreloader, so ``RUN_MAIN`` is never set —
    and the old ``command == 'runserver' and RUN_MAIN != 'true'`` rule then
    returned ``False`` for the whole life of the process, silently disabling
    both the crash-recovery sweep and the ML warm-up.
    """
    assert worker.is_serving_context(
        argv=['manage.py', 'runserver', '--noreload', '127.0.0.1:8777'],
        environ={},
    ) is True


def test_defect2_bootstrap_runs_in_the_reloader_child_only():
    """With the reloader on, exactly one of the two processes bootstraps."""
    argv = ['manage.py', 'runserver', '127.0.0.1:8777']

    # The supervising parent: forks the child, must not load torch itself.
    assert worker.is_serving_context(argv=argv, environ={}) is False
    # The child the autoreloader re-execs, which is the one serving traffic.
    assert worker.is_serving_context(
        argv=argv, environ={'RUN_MAIN': 'true'}) is True


@pytest.mark.parametrize('argv', [
    ['/app/.venv/bin/gunicorn', 'config.wsgi:application', '--workers', '2'],
    ['uwsgi', '--module', 'config.wsgi:application'],
    ['/usr/local/bin/daphne', 'config.asgi:application'],
    [''],                       # mod_wsgi / an embedded interpreter
])
def test_defect2_bootstrap_runs_under_a_real_wsgi_server(argv):
    """No WSGI/ASGI server sets RUN_MAIN; all of them serve traffic."""
    assert worker.is_serving_context(argv=argv, environ={}) is True


@pytest.mark.parametrize('command', [
    'migrate', 'makemigrations', 'collectstatic', 'shell', 'test',
    'spectacular', 'check',
])
def test_defect2_bootstrap_skipped_for_non_serving_commands(command):
    """A management command must never sweep job rows or import torch."""
    assert worker.is_serving_context(
        argv=['manage.py', command], environ={}) is False
    # ...not even when somebody passes --noreload to it.
    assert worker.is_serving_context(
        argv=['manage.py', command, '--noreload'], environ={}) is False


def test_defect2_bootstrap_stays_suppressed_under_pytest():
    """The guard that keeps the suite from loading torch is still in force."""
    assert worker.should_bootstrap() is False
    assert worker.start_bootstrap() is None


def test_defect2_start_bootstrap_runs_once_per_process(monkeypatch):
    """
    ``ready()`` firing twice in one process must not start two boot threads.

    Two sweeps means the second can fail a job the first just let through,
    and two warm-ups means two concurrent torch loads.
    """
    calls = []
    monkeypatch.setattr(worker, 'should_bootstrap', lambda: True)
    monkeypatch.setattr(worker, '_bootstrap', lambda: calls.append(1))

    worker.reset_bootstrap_state()
    try:
        first = worker.start_bootstrap()
        second = worker.start_bootstrap()
        assert first is not None, 'the first call must start the boot thread'
        assert second is None, 'the second call must be a no-op'
        first.join(timeout=5)
        assert calls == [1]
    finally:
        worker.reset_bootstrap_state()


# ===========================================================================
# DEFECT 3 (§E) — a filesystem that refuses the write is a 503, not a 500
# ===========================================================================

@pytest.fixture
def readonly_media_root(tmp_media):
    """Make ``MEDIA_ROOT`` unwritable for one test, then hand it back."""
    original = stat.S_IMODE(os.stat(tmp_media).st_mode)
    os.chmod(tmp_media, 0o500)                        # r-x------
    try:
        yield tmp_media
    finally:
        os.chmod(tmp_media, original)


@pytest.mark.skipif(os.geteuid() == 0,
                    reason='root ignores the write bit, so chmod proves nothing')
@pytest.mark.django_db
def test_defect3_readonly_media_root_returns_503_not_500(
        auth_client, settings_row, sample_files, readonly_media_root):
    """
    The reported bug: ``PermissionError`` escaped ``store_upload`` as a 500.

    The equivalent fault *during an analysis* was already handled (the worker
    maps it to a safe sentence via ``_SAFE_MESSAGES``); the synchronous
    upload path had no handling at all.
    """
    upload = SimpleUploadedFile('photo.jpg', sample_files['jpg'],
                                content_type='image/jpeg')

    response = auth_client.post(reverse(UPLOAD_URL_NAME), {'file': upload},
                                format='multipart')

    assert response.status_code == 503
    body = response.json()
    assert body['code'] == 'storage_unavailable'
    assert envelope_keys(body) == {'detail', 'code', 'errors'}

    # The message must not leak the filesystem layout to the caller.
    detail = body['detail']
    assert str(readonly_media_root) not in detail
    assert 'uploads/' not in detail
    assert 'Errno' not in detail

    # And nothing may survive the failure.
    assert not UploadedMedia.objects.exists()


@pytest.mark.django_db
def test_defect3_storage_failure_leaves_no_orphan_row_or_file(
        user_factory, settings_row, sample_files, tmp_media, monkeypatch):
    """
    A write that dies *part-way* leaves neither a row nor a truncated file.

    ``FileSystemStorage._save`` opens the destination with ``O_CREAT`` and
    then streams into it, and cleans up nothing if a ``write()`` raises — so
    a full disk (``ENOSPC``) genuinely leaves a partial file that no row
    references.  This simulates exactly that.
    """
    from django.core.files.storage import FileSystemStorage

    written = []
    real_save = FileSystemStorage._save

    def half_write_then_enospc(self, name, content):
        full_path = self.path(name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as handle:
            handle.write(b'partially written')
        written.append(full_path)
        raise OSError(errno.ENOSPC, 'No space left on device', full_path)

    monkeypatch.setattr(FileSystemStorage, '_save', half_write_then_enospc)

    user = user_factory()
    upload = SimpleUploadedFile('photo.jpg', sample_files['jpg'],
                                content_type='image/jpeg')

    with pytest.raises(UploadError) as raised:
        store_upload(user, upload, settings_row)

    assert raised.value.code == 'storage_unavailable'
    assert raised.value.status_code == 503

    monkeypatch.setattr(FileSystemStorage, '_save', real_save)
    assert written, 'the simulation did not actually create a partial file'
    assert not os.path.exists(written[0]), (
        f'partial file survived the failure: {written[0]}'
    )
    assert not UploadedMedia.objects.exists()


@pytest.mark.django_db
def test_defect3_a_healthy_media_root_still_uploads(
        auth_client, settings_row, sample_files):
    """The 503 path must not have broken the happy path."""
    upload = SimpleUploadedFile('photo.jpg', sample_files['jpg'],
                                content_type='image/jpeg')
    response = auth_client.post(reverse(UPLOAD_URL_NAME), {'file': upload},
                                format='multipart')
    assert response.status_code == 201
    assert UploadedMedia.objects.count() == 1


# ===========================================================================
# DEFECT 4 (§B) — every /api/ failure is the JSON envelope, HTML never
# ===========================================================================

#: The six routes that use Django's ``<uuid:...>`` path converter, so a
#: malformed id fails at the resolver before any DRF view runs.
MALFORMED_UUID_PATHS = [
    '/api/status/not-a-uuid',
    '/api/media/not-a-uuid',
    '/api/analysis/not-a-uuid',
    '/api/analysis/not-a-uuid/report',
    '/api/report/not-a-uuid',
    '/api/download-report/not-a-uuid',
]


@pytest.mark.parametrize('path', MALFORMED_UUID_PATHS)
@pytest.mark.django_db
def test_defect4_malformed_uuid_returns_the_json_envelope(auth_client, path):
    """All six ``<uuid:...>`` routes answer with ``{detail, code, errors}``."""
    response = auth_client.get(path)

    assert response.status_code == 404
    assert response['Content-Type'].startswith('application/json'), (
        f'{path} answered with {response["Content-Type"]!r}, not JSON'
    )
    body = response.json()
    assert body['code'] == 'not_found'
    assert envelope_keys(body) == {'detail', 'code', 'errors'}


@pytest.mark.django_db
def test_defect4_debug_404_no_longer_lists_the_url_patterns(auth_client,
                                                            settings):
    """
    With ``DEBUG=True`` the technical 404 page used to publish the routing
    table to an unauthenticated caller.

    ``handler404`` cannot fix this on its own — ``response_for_exception``
    short-circuits to ``django.views.debug.technical_404_response`` whenever
    ``DEBUG`` is on and never consults the handler.  This is the test that
    pins ``common.middleware.ApiErrorEnvelopeMiddleware`` in place.
    """
    settings.DEBUG = True

    response = auth_client.get('/api/status/not-a-uuid')

    assert response.status_code == 404
    assert response['Content-Type'].startswith('application/json')
    text = response.content.decode()
    assert 'Using the URLconf' not in text
    for leaked in ('download-report', 'dashboard/stats', 'api/schema'):
        assert leaked not in text, f'{leaked!r} is still disclosed under DEBUG'


@pytest.mark.django_db
def test_defect4_non_api_404_keeps_djangos_html_page(client, settings):
    """``/admin/``, ``/media/`` and friends are browser surfaces; leave them."""
    settings.DEBUG = False
    response = client.get('/definitely-not-an-api-path')

    assert response.status_code == 404
    assert not response['Content-Type'].startswith('application/json')


@pytest.mark.django_db
def test_defect4_append_slash_redirect_survives_the_envelope_middleware(client):
    """
    ``/api/health`` must still 301 to ``/api/health/``.

    ``CommonMiddleware`` turns a 404 into the ``APPEND_SLASH`` redirect during
    its own response phase.  The envelope middleware is registered above it
    precisely so that has already happened by the time the envelope
    middleware looks — if the ordering is ever changed, this test fails.
    """
    response = client.get('/api/health')
    assert response.status_code == 301
    assert response['Location'].endswith('/api/health/')


@pytest.mark.django_db
def test_defect4_a_drf_404_is_left_untouched(auth_client):
    """A well-formed but unknown id still goes through DRF, not the rewrite."""
    response = auth_client.get(f'/api/media/{uuid.uuid4()}')

    assert response.status_code == 404
    body = response.json()
    assert body['code'] == 'not_found'
    assert envelope_keys(body) == {'detail', 'code', 'errors'}


def test_defect4_handler404_is_wired_and_scoped_to_api():
    """``handler404`` itself (the ``DEBUG=False`` path) returns the envelope."""
    handler = get_resolver().resolve_error_handler(404)
    factory = RequestFactory()

    api = handler(factory.get('/api/status/not-a-uuid'), exception=None)
    assert api.status_code == 404
    assert api['Content-Type'].startswith('application/json')
    body = json.loads(api.content)
    assert body['code'] == 'not_found'
    assert envelope_keys(body) == {'detail', 'code', 'errors'}

    html = handler(factory.get('/somewhere-else'), exception=None)
    assert html.status_code == 404
    assert not html['Content-Type'].startswith('application/json')


def test_defect4_handler500_is_wired_and_scoped_to_api():
    """``handler500`` returns the envelope for ``/api/`` and leaks nothing."""
    handler = get_resolver().resolve_error_handler(500)
    factory = RequestFactory()

    api = handler(factory.get('/api/analysis'))
    assert api.status_code == 500
    assert api['Content-Type'].startswith('application/json')
    body = json.loads(api.content)
    assert body == {'detail': 'Internal Server Error', 'code': 'server_error',
                    'errors': None}

    html = handler(factory.get('/somewhere-else'))
    assert html.status_code == 500
    assert not html['Content-Type'].startswith('application/json')


# ===========================================================================
# DEFECT 5 (§C) — an unusable frame rate must not skip the duration cap
# ===========================================================================

def test_defect5_unusable_fps_still_enforces_the_duration_cap(fake_capture):
    """
    The reported bug: ``fps <= 0`` set ``duration = None`` and the cap was
    guarded on ``duration is not None``, so it was skipped outright.

    999,999 frames is ~9 hours at any plausible rate; a 300 s cap must
    refuse it whatever the container claims about timing.
    """
    fake_capture(frame_count=999_999, fps=0.0)

    with pytest.raises(UploadError) as raised:
        probe_video('/tmp/does-not-need-to-exist.mp4', max_seconds=300)

    assert raised.value.code == 'video_too_long'
    assert '999999' in raised.value.message


def test_defect5_unusable_fps_accepts_a_clip_inside_the_frame_budget(
        fake_capture):
    """
    The cap is a ceiling, not a ban: an unusable fps is not itself a
    rejection reason.

    ``corrupt_file`` was the other candidate fix and was not taken — OpenCV
    has already proved it can open the container and count its frames, so a
    missing frame-rate atom alone is not evidence of corruption.
    """
    budget = int(300 * FALLBACK_VIDEO_FPS)
    fake_capture(frame_count=budget - 1, fps=0.0)

    width, height, duration = probe_video(
        '/tmp/does-not-need-to-exist.mp4', max_seconds=300,
    )

    assert (width, height) == (640, 480)
    # Unknown, not guessed: the frame budget is good enough to enforce a
    # ceiling but must not be persisted as if it had been measured.
    assert duration is None


@pytest.mark.parametrize(('fps', 'frame_count', 'expected_seconds'), [
    (0.5, 100, 200.0),        # a time-lapse
    (23.976, 2000, 83.42),    # NTSC film
    (119.88, 6000, 50.05),    # a high-frame-rate capture
])
def test_defect5_odd_but_valid_frame_rates_are_not_rejected(
        fake_capture, fps, frame_count, expected_seconds):
    """A weird-but-real frame rate takes the exact arithmetic, as before."""
    fake_capture(frame_count=frame_count, fps=fps)

    _width, _height, duration = probe_video('/tmp/x.mp4', max_seconds=300)

    assert duration == pytest.approx(expected_seconds, rel=1e-3)


@pytest.mark.parametrize('frame_count', [0.0, -1.0, -999.0])
def test_defect5_zero_or_negative_frame_count_is_corrupt_file(fake_capture,
                                                              frame_count):
    """A container claiming no frames — or negative frames — is corrupt."""
    fake_capture(frame_count=frame_count, fps=30.0)

    with pytest.raises(UploadError) as raised:
        probe_video('/tmp/x.mp4', max_seconds=300)

    assert raised.value.code == 'corrupt_file'


@pytest.mark.parametrize(('frame_count', 'fps'), [
    (math.nan, 30.0),
    (math.inf, 30.0),
    (1000.0, math.nan),
    (1000.0, math.inf),
])
def test_defect5_non_finite_metadata_never_skips_the_cap(fake_capture,
                                                         frame_count, fps):
    """
    ``nan`` defeats every naive comparison, so it must be folded away first.

    ``nan <= 0`` is ``False`` and ``nan > max_seconds`` is ``False`` too, so
    an unguarded ``nan`` sails through both the frame-count check and the
    duration cap.
    """
    fake_capture(frame_count=frame_count, fps=fps)

    if math.isfinite(frame_count):
        # Only the frame rate is broken: the frame budget decides, and 1000
        # frames is well inside a 300 s budget.
        _w, _h, duration = probe_video('/tmp/x.mp4', max_seconds=300)
        assert duration is None
    else:
        with pytest.raises(UploadError) as raised:
            probe_video('/tmp/x.mp4', max_seconds=300)
        assert raised.value.code == 'corrupt_file'


@pytest.mark.django_db
def test_defect5_a_real_clip_still_uploads_with_a_measured_duration(
        auth_client, settings_row, sample_files):
    """End-to-end guard: the fixture clip's real duration is still measured."""
    upload = SimpleUploadedFile('clip.mp4', sample_files['mp4'],
                                content_type='video/mp4')
    response = auth_client.post(reverse(UPLOAD_URL_NAME), {'file': upload},
                                format='multipart')

    assert response.status_code == 201
    assert response.json()['duration_seconds'] == pytest.approx(2.0, abs=0.2)


# ===========================================================================
# DEFECT 6 (review) — the boot sweep must not destroy a sibling's live work
# ===========================================================================

@pytest.mark.django_db
def test_defect6_boot_sweep_leaves_a_live_sibling_workers_job_alone(
        auth_client, sample_files):
    """
    The reported bug, exactly.

    ``gunicorn --workers 2`` (the shipped default in
    ``docker/entrypoint.sh``) means two processes, each running the boot
    sweep.  When the arbiter replaces one worker, the replacement used to
    blanket-fail every ``running`` row — including the *other* worker's live
    analysis.
    """
    user = auth_client.user
    media = make_media(user, sample_files)
    # Owned by a different process that is demonstrably alive: this test
    # process, under a boot id that is not ours.
    live_sibling = make_analysis(
        user, media, status=STATUS_RUNNING,
        settings_snapshot=snapshot_owned_by(owner_stamp(pid=os.getpid())),
    )

    assert worker.recover_interrupted_jobs() == 0

    live_sibling.refresh_from_db()
    assert live_sibling.status == STATUS_RUNNING
    assert live_sibling.error_message == ''


@pytest.mark.django_db
def test_defect6_boot_sweep_fails_a_job_whose_owner_is_gone(auth_client,
                                                            sample_files):
    """A row whose owning process has exited is genuinely abandoned."""
    user = auth_client.user
    media = make_media(user, sample_files)
    dead = make_analysis(
        user, media, status=STATUS_RUNNING,
        settings_snapshot=snapshot_owned_by(owner_stamp(pid=a_dead_pid())),
    )

    assert worker.recover_interrupted_jobs() == 1

    dead.refresh_from_db()
    assert dead.status == STATUS_FAILED
    assert dead.error_message == worker.INTERRUPTED_MESSAGE
    assert dead.end_time is not None


@pytest.mark.django_db
def test_defect6_boot_sweep_fails_an_unowned_job(auth_client, sample_files):
    """
    No stamp means no process ever took responsibility.

    Both transitions into a non-terminal status write the owner in the *same*
    statement as the status, so a live row can never look like this.
    """
    user = auth_client.user
    media = make_media(user, sample_files)
    orphan = make_analysis(user, media, status=STATUS_QUEUED,
                           settings_snapshot=snapshot_owned_by(None))

    assert worker.recover_interrupted_jobs() == 1

    orphan.refresh_from_db()
    assert orphan.status == STATUS_FAILED


@pytest.mark.django_db
def test_defect6_boot_sweep_leaves_another_hosts_job_alone_until_it_is_stale(
        auth_client, sample_files, settings):
    """
    A pid means nothing on another machine, so only age can settle it.

    Two containers sharing one PostgreSQL database is a supported shape, and
    the row may well be running right now on the other one.
    """
    settings.ASG = {**settings.ASG, 'STALE_JOB_SECONDS': 3600}
    user = auth_client.user
    media = make_media(user, sample_files)

    fresh = make_analysis(
        user, media, status=STATUS_RUNNING,
        start_time=timezone.now(),
        settings_snapshot=snapshot_owned_by(owner_stamp(host='other-box')),
    )
    assert worker.recover_interrupted_jobs() == 0
    fresh.refresh_from_db()
    assert fresh.status == STATUS_RUNNING

    AnalysisResult.objects.filter(pk=fresh.pk).update(
        start_time=timezone.now() - timedelta(hours=2))
    assert worker.recover_interrupted_jobs() == 1
    fresh.refresh_from_db()
    assert fresh.status == STATUS_FAILED


@pytest.mark.django_db
def test_defect6_claim_stamps_ownership_in_the_same_statement(auth_client,
                                                              sample_files):
    """
    The window between "row is running" and "row has an owner" must be zero.

    An unowned non-terminal row is treated as an orphan, so a claim that
    stamped afterwards would be a race against any sibling booting in between.
    """
    user = auth_client.user
    media = make_media(user, sample_files)
    analysis = make_analysis(user, media, status=STATUS_QUEUED,
                             settings_snapshot=snapshot_owned_by(None))

    claimed = worker._claim(analysis.analysis_id)

    assert claimed is not None
    analysis.refresh_from_db()
    assert analysis.status == STATUS_RUNNING
    assert worker.is_owned_by_this_process(analysis.settings_snapshot)
    # ...and the configuration the run was frozen with is still intact.
    assert analysis.settings_snapshot['confidence_threshold'] == 0.35


@pytest.mark.django_db
def test_defect6_enqueue_stamps_ownership(auth_client, sample_files, settings):
    """A ``queued`` row lives in one process's pool, so it is owned too."""
    settings.ASG = {**settings.ASG, 'WORKER_ENABLED': False}
    user = auth_client.user
    media = make_media(user, sample_files)
    analysis = make_analysis(user, media, status='pending',
                             settings_snapshot=snapshot_owned_by(None))

    # WORKER_ENABLED=False runs it inline; stub the run so only the queueing
    # half is under test.
    original = worker.run_analysis
    worker.run_analysis = lambda analysis_id: None
    try:
        assert worker.job_queue.enqueue(analysis.analysis_id) is True
    finally:
        worker.run_analysis = original

    analysis.refresh_from_db()
    assert analysis.status == STATUS_QUEUED
    assert worker.is_owned_by_this_process(analysis.settings_snapshot)


@pytest.mark.django_db
def test_defect6_persist_refuses_to_resurrect_an_externally_failed_row(
        auth_client, sample_files, settings_row, tmp_path):
    """
    The second half of the bug: the run finishing and writing ``done`` back.

    After another process has failed the row, the user has already been told
    the analysis is dead.  Silently flipping it back to ``done`` minutes
    later is worse than either outcome on its own.
    """
    user = auth_client.user
    media = make_media(user, sample_files)
    analysis = make_analysis(user, media, status=STATUS_QUEUED,
                             settings_snapshot=snapshot_owned_by(None))
    claimed = worker._claim(analysis.analysis_id)
    assert claimed is not None

    # A sibling's boot sweep decides this row is abandoned.
    AnalysisResult.objects.filter(pk=analysis.analysis_id).update(
        status=STATUS_FAILED, stage='failed',
        error_message=worker.INTERRUPTED_MESSAGE, end_time=timezone.now(),
    )

    persisted = worker._persist(claimed, {'vehicles': []}, tmp_path)

    assert persisted is False
    analysis.refresh_from_db()
    assert analysis.status == STATUS_FAILED
    assert analysis.error_message == worker.INTERRUPTED_MESSAGE


@pytest.mark.django_db
def test_defect6_persist_refuses_when_another_process_owns_the_row(
        auth_client, sample_files, settings_row, tmp_path):
    """Still ``running``, but taken over: the current owner wins, not us."""
    user = auth_client.user
    media = make_media(user, sample_files)
    analysis = make_analysis(user, media, status=STATUS_QUEUED,
                             settings_snapshot=snapshot_owned_by(None))
    claimed = worker._claim(analysis.analysis_id)
    assert claimed is not None

    AnalysisResult.objects.filter(pk=analysis.analysis_id).update(
        settings_snapshot=snapshot_owned_by(owner_stamp()),
    )

    assert worker._persist(claimed, {'vehicles': []}, tmp_path) is False

    analysis.refresh_from_db()
    assert analysis.status == STATUS_RUNNING


@pytest.mark.django_db
def test_defect6_persist_still_writes_the_result_it_owns(
        auth_client, sample_files, settings_row, tmp_path):
    """The guard must not break the ordinary, overwhelmingly common case."""
    user = auth_client.user
    media = make_media(user, sample_files)
    analysis = make_analysis(user, media, status=STATUS_QUEUED,
                             settings_snapshot=snapshot_owned_by(None))
    claimed = worker._claim(analysis.analysis_id)

    assert worker._persist(claimed, {'vehicles': []}, tmp_path) is True

    analysis.refresh_from_db()
    assert analysis.status == STATUS_DONE
    assert analysis.progress == 100
    # A terminal row has no owner: nothing is running it any more.
    assert worker.owner_of(analysis.settings_snapshot) is None


def test_defect6_never_probes_liveness_with_os_kill_off_posix(monkeypatch):
    """
    ``os.kill(pid, 0)`` on Windows calls ``TerminateProcess`` — it kills the
    process it is asked about.  Guard the guard.
    """
    def explode(*args, **kwargs):                     # pragma: no cover
        raise AssertionError('os.kill must never be called off POSIX')

    monkeypatch.setattr(os, 'name', 'nt')
    monkeypatch.setattr(os, 'kill', explode)

    # A pid that is not ours: liveness is unknowable, never probed.
    assert worker.process_is_alive(os.getpid() + 1) is None
    # Our own pid needs no probe at all.
    assert worker.process_is_alive(os.getpid()) is True


def test_defect6_process_identity_is_stable_and_scoped():
    """The ownership stamp is well formed and recognises itself."""
    owner = worker.process_owner()
    assert owner['host'] == socket.gethostname()
    assert owner['pid'] == os.getpid()
    assert worker.process_owner()['boot'] == owner['boot']

    snapshot = worker.snapshot_with_owner({'confidence_threshold': 0.4})
    assert snapshot['confidence_threshold'] == 0.4
    assert worker.is_owned_by_this_process(snapshot) is True
    assert worker.is_owned_by_this_process(
        snapshot_owned_by(owner_stamp())) is False
    assert worker.is_owned_by_this_process({}) is False
    assert worker.owner_of(worker.snapshot_without_owner(snapshot)) is None


# ===========================================================================
# DEFECT 7 (review) — DELETE /api/media must reclaim artefacts and the PDF
# ===========================================================================

@pytest.mark.django_db
def test_defect7_deleting_media_reclaims_artefacts_and_the_report_pdf(
        auth_client, sample_files):
    """
    The reported bug: the cascade drops the only pointers to the files.

    ``DELETE /api/media/{id}`` cascade-deletes ``UploadedMedia ->
    AnalysisResult -> GeneratedReport``, and those rows were the only record
    of ``MEDIA_ROOT/analyses/<analysis_id>/`` and
    ``MEDIA_ROOT/reports/<report_id>.pdf``.  The upload's own file was
    removed, everything the pipeline produced from it was not — so
    ``MEDIA_ROOT`` grew monotonically, and a report PDF survived as an
    unreferenced copy of data the user had asked to delete.

    The sibling endpoint ``DELETE /api/analysis/{id}`` already reclaimed
    correctly; this pins the media path to the same standard.
    """
    from common.storage import (
        analysis_artifact_dir,
        report_path,
        to_media_relative,
    )
    from reports.models import GeneratedReport

    user = auth_client.user
    media = make_media(user, sample_files)
    analysis = make_analysis(user, media, status=STATUS_DONE, progress=100,
                             stage='done', end_time=timezone.now())

    # Real artefacts, laid out exactly as the pipeline writes them.
    artefact_dir = analysis_artifact_dir(analysis.analysis_id)
    preview = artefact_dir / 'preview.jpg'
    preview.write_bytes(sample_files['jpg'])
    crop = analysis_artifact_dir(analysis.analysis_id, 'crops') / 'v0.jpg'
    crop.write_bytes(sample_files['jpg'])
    mask = analysis_artifact_dir(analysis.analysis_id, 'masks') / 'm0.png'
    mask.write_bytes(sample_files['png'])

    # ...and a real report row whose PDF is on disk, with report_path set
    # exactly as reports.services writes it.
    report = GeneratedReport.objects.create(analysis=analysis)
    pdf = report_path(report.report_id)
    pdf.write_bytes(b'%PDF-1.4\nnot a real render, but a real file\n')
    report.report_path = to_media_relative(pdf)
    report.file_size_bytes = pdf.stat().st_size
    report.save(update_fields=['report_path', 'file_size_bytes'])

    upload_path = media.file.path
    assert preview.exists() and crop.exists() and mask.exists() and pdf.exists()

    response = auth_client.delete(
        reverse('media-detail', args=[media.media_id]))

    assert response.status_code == 204
    assert not UploadedMedia.objects.filter(pk=media.media_id).exists()
    assert not AnalysisResult.objects.filter(pk=analysis.analysis_id).exists()
    assert not GeneratedReport.objects.filter(pk=report.report_id).exists()

    assert not os.path.exists(upload_path), 'the upload itself survived'
    assert not artefact_dir.exists(), (
        f'the artefact tree survived the delete: {artefact_dir}'
    )
    assert not pdf.exists(), f'the report PDF survived the delete: {pdf}'


@pytest.mark.django_db
def test_defect7_deleting_media_with_no_analyses_still_works(auth_client,
                                                              sample_files):
    """The reclaim must not disturb the ordinary "just an upload" case."""
    media = make_media(auth_client.user, sample_files)
    upload_path = media.file.path

    response = auth_client.delete(
        reverse('media-detail', args=[media.media_id]))

    assert response.status_code == 204
    assert not UploadedMedia.objects.filter(pk=media.media_id).exists()
    assert not os.path.exists(upload_path)


@pytest.mark.django_db
def test_defect7_a_blocked_delete_reclaims_nothing(auth_client, sample_files):
    """
    A 409 must leave every file alone.

    The paths are resolved before the delete, so a delete that never happens
    must not reach the reclaim step at all.
    """
    from common.storage import analysis_artifact_dir

    user = auth_client.user
    media = make_media(user, sample_files)
    analysis = make_analysis(user, media, status=STATUS_RUNNING)
    artefact_dir = analysis_artifact_dir(analysis.analysis_id)
    (artefact_dir / 'preview.jpg').write_bytes(sample_files['jpg'])

    response = auth_client.delete(
        reverse('media-detail', args=[media.media_id]))

    assert response.status_code == 409
    assert response.json()['code'] == 'analysis_in_progress'
    assert artefact_dir.exists()
    assert (artefact_dir / 'preview.jpg').exists()
    assert os.path.exists(media.file.path)


# ===========================================================================
# DEFECT 8 (review) — the documented duplicate-upload contract
# ===========================================================================

@pytest.mark.django_db
def test_defect8_deduplicated_key_is_absent_on_a_new_upload(
        auth_client, settings_row, sample_files):
    """
    ``API_CONTRACT.md``: 201 has **no** ``deduplicated`` key at all; the
    200 re-upload has ``deduplicated: true``.

    Pinned here as well as in ``test_review_fixes.py`` because
    ``uploads/views.py`` is the file this remediation pass edits, and
    "helpfully" emitting ``deduplicated: false`` on the 201 path would be an
    easy and silent way to break a documented contract.
    """
    payload = sample_files['jpg']

    first = auth_client.post(reverse(UPLOAD_URL_NAME),
                             {'file': SimpleUploadedFile('dup.jpg', payload,
                                                         'image/jpeg')},
                             format='multipart')
    assert first.status_code == 201
    assert 'deduplicated' not in first.json()

    second = auth_client.post(reverse(UPLOAD_URL_NAME),
                              {'file': SimpleUploadedFile('dup.jpg', payload,
                                                          'image/jpeg')},
                              format='multipart')
    assert second.status_code == 200
    assert second.json()['deduplicated'] is True
    assert second.json()['media_id'] == first.json()['media_id']

    listed = auth_client.get(reverse('media-list')).json()
    assert all('deduplicated' not in row for row in listed['results'])
