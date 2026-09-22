"""
Regression tests for the ten defects raised by the independent code review.

One section per finding, named ``test_fNN_*`` so a failure names the defect
it re-opens.  Every test in here was written against the *unfixed* code
first and confirmed red before the corresponding fix landed; several of them
are deliberately awkward (real threads, real PDFs, real WebP bytes) because
the cheap version of the same assertion is exactly what let the defect
through in the first place — see ``test_f3_*`` for the clearest example,
where the test being replaced stubbed the collaborator and could therefore
only ever check the response *shape*, which a stale row satisfies too.

Conventions
-----------
* ``tmp_media`` (autouse, from ``conftest``) points ``MEDIA_ROOT`` at a
  per-test temporary directory, so the file assertions here are real reads
  of real bytes and still leave nothing behind.
* Tests that need two requests to genuinely overlap use
  ``django_db(transaction=True)`` plus a ``threading.Barrier``: worker
  threads have their own connections and can only see *committed* rows, so
  the default test transaction would make the race untestable.
* Nothing here monkeypatches the thing under test.  Where a seam is patched
  (``build_settings_snapshot``, ``count_pdf_pages``) it is to widen a timing
  window or to observe the filesystem mid-flight, never to supply the answer
  the assertion then checks.
"""
import io
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import connections
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APIClient

from analysis import services, worker
from analysis.models import (
    STATUS_DONE,
    STATUS_PENDING,
    STATUS_RUNNING,
    AnalysisResult,
    DetectedVehicle,
)
from common import validators
from common.storage import (
    analysis_artifact_dir,
    delete_paths,
    from_media_relative,
)
from reports.models import GeneratedReport
from reports.services import generate_report_for_analysis
from uploads.models import UploadedMedia

from .conftest import DEFAULT_TEST_PASSWORD, authenticate, make_jpeg_bytes

User = get_user_model()

#: Name of the environment variable that points this suite at the contract.
CONTRACT_ENV_VAR = 'ASG_CONTRACT_PATH'

#: The root of *this* repository (backend/), i.e. the checkout root in CI.
BACKEND_ROOT = Path(__file__).resolve().parents[1]


def contract_candidates():
    """
    Every place ``API_CONTRACT.md`` may legitimately be, most specific first.

    The contract is owned by the **umbrella** repository, not this one:
    ``backend/`` is its own git repo, so a standalone checkout — which is how
    CI clones it and how the README tells collaborators to work — has no
    sibling umbrella tree at all.  The previous ``parents[2] /
    'API_CONTRACT.md'`` only ever resolved because the author's ``backend/``
    happens to sit inside ``FYP/``; on a runner it was a hard
    ``FileNotFoundError``.

    Deleting the guard was not an option (it is the only thing that catches
    the frozen contract drifting away from what the server actually returns),
    and neither was an unconditional skip (green, and verifying nothing).  So
    the file is *searched for*, and when it genuinely is not there the test
    says so in as many words — see :func:`read_contract_or_skip`.
    """
    candidates = []

    override = os.environ.get(CONTRACT_ENV_VAR, '').strip()
    if override:
        supplied = Path(override).expanduser()
        # Accept either the file itself or the directory that contains it,
        # so `ASG_CONTRACT_PATH=$GITHUB_WORKSPACE/_umbrella` also works.
        candidates += [supplied, supplied / 'API_CONTRACT.md']

    candidates += [
        # Vendored into this repo (if it is ever moved/copied here).
        BACKEND_ROOT / 'API_CONTRACT.md',
        # A sibling checkout made by CI (see .github/workflows/ci.yml).
        BACKEND_ROOT / '_umbrella' / 'API_CONTRACT.md',
        # The umbrella working copy: FYP/backend -> FYP/API_CONTRACT.md.
        BACKEND_ROOT.parent / 'API_CONTRACT.md',
        BACKEND_ROOT.parent / '_umbrella' / 'API_CONTRACT.md',
    ]
    return candidates


def read_contract_or_skip():
    """
    The text of ``API_CONTRACT.md``, or an explicit *missing input* skip.

    The skip message names the file, every path that was tried and the
    environment variable that would supply it, so a skipped run reads as
    "this guard did not run and here is how to make it run" rather than as a
    pass.
    """
    tried = contract_candidates()
    for candidate in tried:
        if candidate.is_file():
            return candidate.read_text(encoding='utf-8')

    pytest.skip(
        'MISSING INPUT (this is not a pass): API_CONTRACT.md was not found, '
        'so the contract-drift guard below could not run. The contract lives '
        'in the umbrella repository (IfTruePrintArslan/AutoSmokeGuard), not '
        'in this one, so a standalone backend checkout does not contain it. '
        'Looked in: ' + '; '.join(str(path) for path in tried) + '. Set '
        + CONTRACT_ENV_VAR + ' to the file (or to the directory holding it) '
        'to make this guard run.'
    )


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def inline_worker(settings):
    """
    Keep the threaded worker out of every test in this module.

    Replaces the whole ``ASG`` dict rather than mutating it, so pytest-django
    restores it cleanly (a nested mutation leaks between tests).
    """
    settings.ASG = {**settings.ASG, 'WORKER_ENABLED': False}
    return settings


def make_media(user, filename='clip.jpg'):
    """A real ``UploadedMedia`` row with real, decodable bytes on disk."""
    content = make_jpeg_bytes()
    media = UploadedMedia(
        user=user, filename=filename, format='jpg', media_type='image',
        size_bytes=len(content), width=32, height=24,
    )
    media.file.save(filename, ContentFile(content), save=False)
    media.save()
    return media


def make_done_analysis(user, media=None, **overrides):
    """A finished ``AnalysisResult``, ready for a report."""
    now = timezone.now()
    fields = {
        'media': media or make_media(user),
        'user': user,
        'status': STATUS_DONE,
        'progress': 100,
        'total_vehicles': 0,
        'total_smoke': 0,
        'frames_processed': 1,
        'avg_confidence': 0.0,
        'overall_severity': '',
        'severity_counts': {'low': 0, 'moderate': 0, 'high': 0},
        'settings_snapshot': {'confidence_threshold': 0.35},
        'start_time': now,
        'end_time': now,
    }
    fields.update(overrides)
    return AnalysisResult.objects.create(**fields)


def run_concurrently(target, count=2, timeout=30):
    """
    Run ``target(index)`` in ``count`` threads that start at the same instant.

    Returns ``[(index, result, exception), ...]`` in completion-agnostic
    index order.  Each thread closes its own database connections on the way
    out — pytest-django's ``transaction=True`` teardown truncates tables, and
    a connection left open by a dead thread blocks that on SQLite.
    """
    barrier = threading.Barrier(count)
    outcomes = [None] * count

    def runner(index):
        try:
            barrier.wait(timeout=timeout)
            outcomes[index] = (index, target(index), None)
        except Exception as exc:                        # noqa: BLE001
            outcomes[index] = (index, None, exc)
        finally:
            for conn in connections.all():
                conn.close()

    threads = [threading.Thread(target=runner, args=(i,), daemon=True)
               for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=timeout)
        assert not thread.is_alive(), 'a concurrency test thread hung'

    return outcomes


# ===========================================================================
# F3 — POST /api/analysis/{id}/report must regenerate, not return the stale row
#
# The endpoint-level test lives in tests/test_analysis.py
# (test_report_endpoint_really_regenerates), replacing the stubbed one that
# cemented the bug. What is checked here is the seam underneath it: the
# analysis app's delegate must ask for a *forced* rebuild, because that flag
# is the entire difference between "201 Created" being true and being a lie.
# ===========================================================================

@pytest.mark.django_db
def test_f3_generate_report_delegate_forces_a_rebuild(auth_client, monkeypatch):
    """``services.generate_report`` passes ``force=True`` to the reports app."""
    analysis = make_done_analysis(auth_client.user)
    seen = {}

    import reports.services as reports_services

    def spy(target, force=False):
        seen['force'] = force
        return 'sentinel'

    monkeypatch.setattr(reports_services, 'generate_report_for_analysis', spy)

    assert services.generate_report(analysis) == 'sentinel'
    assert seen['force'] is True, (
        'the analysis app asked for a report without force=True, so an '
        'existing row short-circuits the render and the endpoint answers '
        '201 with stale bytes'
    )


@pytest.mark.django_db
def test_f3_forced_regeneration_moves_generated_at(auth_client):
    """
    ``generated_at`` records when the bytes on disk were made.

    It is an ``auto_now_add`` column, which only fires on INSERT — so a
    re-render would otherwise keep advertising the moment the *first* PDF was
    produced while carrying the newest one's contents.
    """
    analysis = make_done_analysis(auth_client.user)
    first = generate_report_for_analysis(analysis)
    first_stamp = first.generated_at

    second = generate_report_for_analysis(analysis, force=True)

    assert second.report_id == first.report_id
    assert second.generated_at > first_stamp
    assert GeneratedReport.objects.get(pk=first.report_id).generated_at > first_stamp


# ===========================================================================
# F6 — deleting an analysis (or its media) must reclaim every file it owns
# ===========================================================================

@pytest.mark.django_db
def test_f6_deleting_an_analysis_reclaims_its_pdf(auth_client):
    """
    ``DELETE /api/analysis/{id}`` removes the artefact tree *and* the PDF.

    The PDF lives in ``MEDIA_ROOT/reports/``, not under the analysis's own
    directory, and the ``GeneratedReport`` row the cascade destroys was the
    only pointer to it.
    """
    analysis = make_done_analysis(auth_client.user)
    report = generate_report_for_analysis(analysis)

    pdf = from_media_relative(report.report_path)
    artefacts = analysis_artifact_dir(analysis.analysis_id)
    (artefacts / 'preview.jpg').write_bytes(make_jpeg_bytes())

    assert pdf.is_file() and artefacts.is_dir()

    response = auth_client.delete(
        reverse('analysis-detail', args=[analysis.analysis_id]))

    assert response.status_code == 204
    assert not artefacts.exists(), 'artefact tree left behind'
    assert not pdf.exists(), (
        f'{pdf.name} orphaned: the GeneratedReport row is gone, so nothing '
        'in the system knows this file exists any more'
    )


@pytest.mark.django_db
def test_f6_media_cleanup_plan_covers_artefacts_and_pdfs(auth_client):
    """
    ``artifact_paths_for_media`` is what the media-delete path must call.

    It has to be resolvable *before* the cascade runs, and it has to include
    both trees for every analysis of that upload.
    """
    media = make_media(auth_client.user)
    first = make_done_analysis(auth_client.user, media=media)
    second = make_done_analysis(auth_client.user, media=media)
    report = generate_report_for_analysis(first)

    planned = set(services.artifact_paths_for_media(media))

    assert analysis_artifact_dir(first.analysis_id, create=False) in planned
    assert analysis_artifact_dir(second.analysis_id, create=False) in planned
    assert from_media_relative(report.report_path) in planned

    # And the plan genuinely reclaims: this is the exact two-step the
    # uploads view has to perform.
    pdf = from_media_relative(report.report_path)
    assert pdf.is_file()
    media.delete()
    delete_paths(planned)
    assert not pdf.exists()


@pytest.mark.django_db
def test_f6_delete_paths_refuses_to_escape_media_root(tmp_path, settings):
    """A tampered ``report_path`` must never become an arbitrary file delete."""
    outsider = tmp_path / 'not-media.txt'
    outsider.write_text('precious')

    inside = Path(settings.MEDIA_ROOT) / 'reports' / 'x.pdf'
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_bytes(b'%PDF-1.4\n')

    removed = delete_paths([outsider, inside])

    assert outsider.exists(), 'deleted a path outside MEDIA_ROOT'
    assert not inside.exists()
    assert removed == 1


@pytest.mark.django_db
def test_f6_delete_paths_tolerates_absent_entries(settings):
    """A half-reclaimed tree from an earlier crash is not an error."""
    missing = Path(settings.MEDIA_ROOT) / 'reports' / f'{uuid.uuid4()}.pdf'
    assert delete_paths([missing, None]) == 0


# ===========================================================================
# F8 — the settings whitelist and the uploader's policy are one list
# ===========================================================================

def _webp_bytes():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new('RGB', (32, 24), (10, 200, 30)).save(buffer, format='WEBP')
    return buffer.getvalue()


def _bmp_bytes():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new('RGB', (32, 24), (10, 200, 30)).save(buffer, format='BMP')
    return buffer.getvalue()


def _matroska_bytes(extension, fourcc):
    import tempfile

    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as workdir:
        path = Path(workdir) / f'clip.{extension}'
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), 4.0, (32, 24),
        )
        if not writer.isOpened():                     # pragma: no cover
            pytest.skip(f'this OpenCV build cannot write .{extension}')
        try:
            for index in range(8):
                frame = np.full((24, 32, 3), 30, dtype=np.uint8)
                frame[:, :, index % 3] = 30 + (index * 25) % 220
                writer.write(frame)
        finally:
            writer.release()
        return path.read_bytes()


def test_f8_settings_whitelist_is_derived_from_the_uploader_policy():
    """
    The admin-facing whitelist *is* the uploader's policy table.

    Two hand-maintained lists is how a format became enable-able and
    un-uploadable at the same time; this asserts there is only one.
    """
    from system_config import serializers as sysconfig_serializers

    policy = validators._EXTENSION_POLICY

    assert set(sysconfig_serializers.SUPPORTED_IMAGE_FORMATS) == {
        extension for extension, spec in policy.items() if spec[0] == 'image'
    }
    assert set(sysconfig_serializers.SUPPORTED_VIDEO_FORMATS) == {
        extension for extension, spec in policy.items() if spec[0] == 'video'
    }

    # And nothing an admin can enable is unknown to the uploader.
    enable_able = (set(sysconfig_serializers.SUPPORTED_IMAGE_FORMATS)
                   | set(sysconfig_serializers.SUPPORTED_VIDEO_FORMATS))
    assert enable_able <= set(policy)


@pytest.mark.parametrize('extension, payload_factory, sniff', [
    ('webp', _webp_bytes, 'webp'),
    ('bmp', _bmp_bytes, 'bmp'),
    ('mkv', lambda: _matroska_bytes('mkv', 'MJPG'), 'matroska'),
    ('webm', lambda: _matroska_bytes('webm', 'VP80'), 'matroska'),
])
def test_f8_new_formats_are_sniffed_from_real_bytes(extension,
                                                    payload_factory, sniff):
    """Signatures are checked against files their real encoder produced."""
    header = payload_factory()[:validators.HEADER_BYTES]
    assert validators.sniff_format(header) == sniff
    assert extension in validators._EXTENSION_POLICY
    assert sniff in validators._EXTENSION_POLICY[extension][2]


@pytest.mark.django_db
def test_f8_an_admin_enabled_format_actually_uploads(admin_client, settings_row):
    """
    Enable ``webp`` through the settings API, then upload a real ``.webp``.

    This is the end-to-end shape of the defect: the PATCH succeeded with 200
    and the upload that it authorised was then rejected — with a message
    that named ``webp`` among the accepted formats.
    """
    patch = admin_client.patch(
        reverse('system-settings'),
        {'allowed_image_formats': ['jpg', 'jpeg', 'png', 'webp']},
        format='json',
    )
    assert patch.status_code == 200
    assert 'webp' in patch.json()['allowed_image_formats']

    upload = admin_client.post(
        reverse('media-upload'),
        {'file': SimpleUploadedFile('sample.webp', _webp_bytes(),
                                    content_type='image/webp')},
        format='multipart',
    )

    assert upload.status_code == 201, upload.json()
    body = upload.json()
    assert body['format'] == 'webp'
    assert body['media_type'] == 'image'
    assert (body['width'], body['height']) == (32, 24)


@pytest.mark.django_db
def test_f8_an_admin_enabled_video_format_actually_uploads(admin_client,
                                                           settings_row):
    """The same, for the video half of the whitelist."""
    patch = admin_client.patch(
        reverse('system-settings'),
        {'allowed_video_formats': ['mp4', 'avi', 'mov', 'mkv']},
        format='json',
    )
    assert patch.status_code == 200

    upload = admin_client.post(
        reverse('media-upload'),
        {'file': SimpleUploadedFile('sample.mkv',
                                    _matroska_bytes('mkv', 'MJPG'),
                                    content_type='video/x-matroska')},
        format='multipart',
    )

    assert upload.status_code == 201, upload.json()
    assert upload.json()['format'] == 'mkv'
    assert upload.json()['media_type'] == 'video'


def test_f8_a_rejection_never_lists_the_rejected_format_as_allowed():
    """
    The "Allowed formats:" list only ever names formats that would work.

    Both halves of the old ``or`` produced the same message, so a format that
    was enabled but undecodable was reported back to the user as allowed.
    """
    payload = SimpleUploadedFile('sample.tiff', b'II*\x00' + b'\x00' * 40)

    ok, code, message, _meta = validators.validate_upload(
        payload, ['jpg', 'png', 'tiff'], 10_000_000,
    )

    assert not ok
    assert code == validators.ERR_INVALID_EXTENSION
    allowed_clause = message.split(':', 1)[1]
    assert 'tiff' not in allowed_clause, message


# ===========================================================================
# F10 — the report detail endpoint has one shape, not two
# ===========================================================================

@pytest.mark.django_db
def test_f10_a_serializer_failure_is_not_answered_with_200(auth_client):
    """
    A malformed ``bounding_box`` must not become a 200 with a hollow body.

    The fallback turned any exception from ``AnalysisDetailSerializer`` into
    a permanently different response — ``annotated_frames: []``,
    ``segmenter_mode: null``, ``device: null`` — logged once as a WARNING and
    otherwise invisible. Silently answering a different question is worse
    than failing.
    """
    analysis = make_done_analysis(auth_client.user)
    DetectedVehicle.objects.create(
        analysis=analysis, vehicle_type='car',
        bounding_box={'x': 'not-a-number', 'y': 0, 'w': 0, 'h': 0},
        confidence=0.9, frame_number=1, timestamp_seconds=0.0,
    )
    report = generate_report_for_analysis(analysis)

    response = auth_client.get(reverse('report-detail', args=[report.report_id]))

    assert response.status_code != 200, (
        'a broken row was served as a successful response carrying a '
        f'silently different shape: {response.json()}'
    )
    assert response.status_code == 500
    assert response.json()['code'] == 'server_error'


@pytest.mark.django_db
def test_f10_the_fallback_shape_is_gone(auth_client):
    """
    One builder for the nested detail, and ``device`` is always a string.

    The two paths disagreed on type — the real serializer emits ``""``, the
    fallback emitted ``null`` — so a client could not even rely on
    ``typeof device``.
    """
    from reports import serializers as report_serializers

    assert not hasattr(report_serializers, '_fallback_analysis_detail')

    analysis = make_done_analysis(auth_client.user)
    report = generate_report_for_analysis(analysis)

    response = auth_client.get(reverse('report-detail', args=[report.report_id]))
    assert response.status_code == 200

    nested = response.json()['analysis']
    assert nested['device'] == ''
    assert nested['segmenter_mode'] == ''

    detail = auth_client.get(
        reverse('analysis-detail', args=[analysis.analysis_id])).json()
    assert nested['device'] == detail['device']
    assert nested['segmenter_mode'] == detail['segmenter_mode']


@pytest.mark.django_db
def test_f10_annotated_frames_survive_the_nested_serialisation(auth_client):
    """
    The nested detail is the *real* one, so the frame list is populated.

    The fallback hardcoded ``annotated_frames: []``, which is also what an
    analysis with no frames legitimately returns — the two were
    indistinguishable from the outside.
    """
    analysis = make_done_analysis(auth_client.user)
    analysis.settings_snapshot = {
        **analysis.settings_snapshot,
        services.RUNTIME_KEY: {
            'annotated_frames': [f'analyses/{analysis.analysis_id}/frames/0001.jpg'],
            'segmenter_mode': 'unet',
            'device': 'cpu',
        },
    }
    analysis.save(update_fields=['settings_snapshot'])
    report = generate_report_for_analysis(analysis)

    nested = auth_client.get(
        reverse('report-detail', args=[report.report_id])).json()['analysis']

    assert nested['annotated_frames'] == [
        f'/media/analyses/{analysis.analysis_id}/frames/0001.jpg']
    assert nested['segmenter_mode'] == 'unet'
    assert nested['device'] == 'cpu'


# ===========================================================================
# F11 — concurrent PDF generation is atomic and de-duplicated
# ===========================================================================

@pytest.mark.django_db
def test_f11_the_render_never_writes_to_the_live_path(auth_client, monkeypatch):
    """
    reportlab is handed a temporary path, never the published one.

    Asserted from inside the render: ``count_pdf_pages`` runs after
    ``doc.build`` and before the rename, so at that instant the live file
    must still hold the previous bytes untouched. That is what makes a
    concurrent download safe, and it is exactly what handing reportlab the
    final path destroys.
    """
    import reports.pdf as pdf_module

    analysis = make_done_analysis(auth_client.user)
    report = generate_report_for_analysis(analysis)
    live = from_media_relative(report.report_path)

    sentinel = b'%PDF-1.4\n% previous complete file\n%%EOF\n'
    live.write_bytes(sentinel)

    observed = {}
    real_count = pdf_module.count_pdf_pages

    def observing_count(path):
        observed['rendered_to'] = Path(path)
        observed['live_bytes_mid_render'] = live.read_bytes()
        return real_count(path)

    monkeypatch.setattr(pdf_module, 'count_pdf_pages', observing_count)

    generate_report_for_analysis(analysis, force=True)

    assert observed['rendered_to'] != live
    assert observed['rendered_to'].parent == live.parent, (
        'the temporary file must share a directory with the target, or the '
        'move is a copy and not an atomic rename'
    )
    assert observed['live_bytes_mid_render'] == sentinel, (
        'the live PDF was being overwritten in place; a reader downloading '
        'it at that moment gets a truncated file'
    )
    assert live.read_bytes() != sentinel
    assert live.read_bytes().startswith(b'%PDF')


@pytest.mark.django_db
def test_f11_reported_size_matches_the_finished_file(auth_client):
    """``file_size_bytes`` is stat'ed after the rename, so it is the real size."""
    analysis = make_done_analysis(auth_client.user)
    report = generate_report_for_analysis(analysis, force=True)
    live = from_media_relative(report.report_path)

    assert report.file_size_bytes == live.stat().st_size
    assert live.read_bytes().rstrip().endswith(b'%%EOF')


@pytest.mark.skipif(
    os.name != 'posix',
    reason=(
        'POSIX permission bits only: Windows has no mode word to assert on — '
        'os.stat there synthesises 0o666/0o444 from the read-only attribute, '
        'so this check would compare 0o666 against 0o644 and fail without '
        'anything being wrong. The behaviour under test (a published PDF that '
        'nginx can read) is a POSIX deployment concern; the ubuntu leg of the '
        'CI matrix runs this for real.'
    ),
)
@pytest.mark.django_db
def test_f11_the_published_pdf_is_world_readable(auth_client):
    """
    Rendering via a temp file must not change the published file's mode.

    ``mkstemp`` creates ``0600`` and a rename carries the mode with it, so
    the atomic-write fix would otherwise hand nginx — which serves
    ``MEDIA_ROOT`` directly in the shipped deployment — a PDF it cannot
    read. Django's ``FILE_UPLOAD_PERMISSIONS`` is the reference.
    """
    import stat

    from django.conf import settings as django_settings

    expected = getattr(django_settings, 'FILE_UPLOAD_PERMISSIONS', None) or 0o644

    analysis = make_done_analysis(auth_client.user)
    report = generate_report_for_analysis(analysis, force=True)
    live = from_media_relative(report.report_path)

    mode = stat.S_IMODE(live.stat().st_mode)
    assert mode == expected, f'published PDF is {mode:#o}, expected {expected:#o}'


@pytest.mark.django_db
def test_f11_a_failed_render_leaves_no_partial_file(auth_client, monkeypatch):
    """A blown-up render must not litter the reports directory."""
    import reports.pdf as pdf_module

    analysis = make_done_analysis(auth_client.user)
    report = generate_report_for_analysis(analysis)
    live = from_media_relative(report.report_path)
    good_bytes = live.read_bytes()

    def explode(path):
        raise RuntimeError('render blew up')

    monkeypatch.setattr(pdf_module, 'count_pdf_pages', explode)

    with pytest.raises(RuntimeError):
        generate_report_for_analysis(analysis, force=True)

    leftovers = [p.name for p in live.parent.iterdir() if 'partial' in p.name]
    assert leftovers == [], leftovers
    assert live.read_bytes() == good_bytes, 'the published PDF was damaged'


@pytest.mark.django_db(transaction=True)
def test_f11_concurrent_first_time_generation_creates_one_report(monkeypatch):
    """
    Two threads generating an analysis's *first* report produce one row.

    Found by running the fixed code against a live server: the worker's
    ``_maybe_generate_report`` starts the automatic PDF the instant the run
    flips to ``done``, and the UI — which is polling for exactly that — fires
    ``POST /api/analysis/{id}/report`` at the same moment. Both saw "no
    report yet"; the loser hit ``UNIQUE constraint failed:
    reports_generated_report.analysis_id`` and the endpoint answered 500.

    Locking on ``report_id`` does not help here, because a row that does not
    exist yet has no shared id to lock on — each caller mints its own
    ``uuid4``. The unit of exclusion has to be the analysis.
    """
    import reports.services as reports_services

    owner = User.objects.create_user(
        email=f'report-race-{uuid.uuid4().hex[:8]}@example.com',
        password=DEFAULT_TEST_PASSWORD,
    )
    analysis = make_done_analysis(owner)
    reports_dir = Path(analysis_artifact_dir(
        analysis.analysis_id, create=False).parent.parent) / 'reports'

    real_build = reports_services.build_report

    def slow_build(target_analysis, report_id, output_path):
        # Widen the window between "is there a report?" and the INSERT.
        time.sleep(0.15)
        return real_build(target_analysis, report_id, output_path)

    monkeypatch.setattr(reports_services, 'build_report', slow_build)

    def generate(_index):
        fresh = AnalysisResult.objects.select_related('media').get(
            pk=analysis.analysis_id)
        return generate_report_for_analysis(fresh)

    try:
        outcomes = run_concurrently(generate, count=2)

        failures = [exc for _i, _r, exc in outcomes if exc is not None]
        assert failures == [], f'{type(failures[0]).__name__}: {failures[0]}'

        ids = {report.report_id for _i, report, _e in outcomes}
        assert len(ids) == 1, f'two reports minted for one analysis: {ids}'
        assert GeneratedReport.objects.filter(analysis=analysis).count() == 1

        # The loser's render must not be left on disk with nothing pointing
        # at it.
        on_disk = sorted(p.name for p in reports_dir.iterdir()) \
            if reports_dir.is_dir() else []
        assert on_disk == [f'{ids.pop()}.pdf'], on_disk
    finally:
        GeneratedReport.objects.filter(analysis=analysis).delete()
        AnalysisResult.objects.filter(pk=analysis.analysis_id).delete()
        UploadedMedia.objects.filter(user=owner).delete()
        User.objects.filter(pk=owner.pk).delete()


@pytest.mark.django_db(transaction=True)
def test_f11_two_threads_regenerating_one_report_do_not_overlap(auth_client,
                                                               monkeypatch):
    """
    The per-report lock means the work happens once at a time.

    ``build_report`` is replaced with a recorder that holds for 100 ms: with
    no lock the two renders interleave and ``peak`` reaches 2.
    """
    import reports.services as reports_services

    analysis = make_done_analysis(auth_client.user)
    generate_report_for_analysis(analysis)

    state = {'active': 0, 'peak': 0}
    guard = threading.Lock()
    real_build = reports_services.build_report

    def recording_build(target_analysis, report_id, output_path):
        with guard:
            state['active'] += 1
            state['peak'] = max(state['peak'], state['active'])
        try:
            time.sleep(0.1)
            return real_build(target_analysis, report_id, output_path)
        finally:
            with guard:
                state['active'] -= 1

    monkeypatch.setattr(reports_services, 'build_report', recording_build)

    def regenerate(_index):
        fresh = AnalysisResult.objects.select_related('media').get(
            pk=analysis.analysis_id)
        return generate_report_for_analysis(fresh, force=True)

    outcomes = run_concurrently(regenerate, count=2)

    assert [exc for _i, _r, exc in outcomes if exc is not None] == []
    assert state['peak'] == 1, (
        f'{state["peak"]} renders of the same report ran at once; both were '
        'writing the same path'
    )
    assert GeneratedReport.objects.filter(analysis=analysis).count() == 1


# ===========================================================================
# F16 — the contract documents 200 + deduplicated for a duplicate upload
# ===========================================================================

def test_f16_the_contract_documents_the_deduplicated_response():
    """The frozen contract has to describe what the server actually does."""
    contract = read_contract_or_skip()

    assert 'deduplicated' in contract, (
        'API_CONTRACT.md still freezes POST /api/upload at "201 MediaObj" '
        'and says nothing about the 200 + deduplicated duplicate response'
    )
    upload_section = contract.split('## Media', 1)[1].split('---', 1)[0]
    assert 'deduplicated' in upload_section
    assert '200' in upload_section


@pytest.mark.django_db
def test_f16_the_server_matches_the_documented_deduplication(auth_client):
    """201 without the key the first time, 200 with it the second."""
    payload = make_jpeg_bytes(colour=(7, 9, 11))

    first = auth_client.post(
        reverse('media-upload'),
        {'file': SimpleUploadedFile('same.jpg', payload,
                                    content_type='image/jpeg')},
        format='multipart',
    )
    second = auth_client.post(
        reverse('media-upload'),
        {'file': SimpleUploadedFile('same.jpg', payload,
                                    content_type='image/jpeg')},
        format='multipart',
    )

    assert first.status_code == 201
    assert 'deduplicated' not in first.json()

    assert second.status_code == 200
    assert second.json()['deduplicated'] is True
    assert second.json()['media_id'] == first.json()['media_id']
    assert UploadedMedia.objects.filter(user=auth_client.user).count() == 1


# ===========================================================================
# F17 — POST /api/analyze always answers the contract's "queued"
# ===========================================================================

@pytest.mark.django_db
def test_f17_analyze_answers_queued_even_when_the_worker_already_claimed_it(
        auth_client, monkeypatch):
    """
    The worker claiming the job between INSERT and ``refresh_from_db`` must
    not change the response body.

    ``enqueue`` is replaced with one that flips the row to ``running``
    immediately — which is precisely what the real worker pool does, only
    reliably.
    """
    media = make_media(auth_client.user)

    def claiming_enqueue(analysis_id):
        AnalysisResult.objects.filter(pk=analysis_id).update(
            status=STATUS_RUNNING, stage='segmenting frames')

    monkeypatch.setattr(worker.job_queue, 'enqueue', claiming_enqueue)

    response = auth_client.post(
        reverse('analysis-analyze'), {'media_id': str(media.media_id)},
        format='json',
    )

    assert response.status_code == 202
    assert response.json()['status'] == 'queued', (
        'the frozen contract value was not honoured; the live value belongs '
        'on GET /api/status/{job_id}'
    )
    # The row itself is untouched — this is a wire-format decision only.
    assert AnalysisResult.objects.get(
        pk=response.json()['analysis_id']).status == STATUS_RUNNING


@pytest.mark.django_db
def test_f17_a_terminal_row_still_reports_its_real_status(auth_client,
                                                          monkeypatch):
    """
    The synchronous mode's already-finished run is not flattened to "queued".

    ``ASG['WORKER_ENABLED'] = False`` runs the job inline, so by the time the
    202 is written the analysis really has finished; reporting "queued" then
    would be the opposite lie.
    """
    media = make_media(auth_client.user)

    def finishing_enqueue(analysis_id):
        AnalysisResult.objects.filter(pk=analysis_id).update(
            status=STATUS_DONE, progress=100)

    monkeypatch.setattr(worker.job_queue, 'enqueue', finishing_enqueue)

    response = auth_client.post(
        reverse('analysis-analyze'), {'media_id': str(media.media_id)},
        format='json',
    )

    assert response.status_code == 202
    assert response.json()['status'] == STATUS_DONE


# ===========================================================================
# F18 — one builder of ReportObj
# ===========================================================================

@pytest.mark.django_db
def test_f18_the_report_endpoints_emit_one_shape(auth_client):
    """
    ``POST /api/analysis/{id}/report`` and ``GET /api/reports`` agree.

    The analysis app used to hand-build a six-key copy that omitted
    ``analysis``; the reports serializer emits seven. Two builders of one
    contract type is a drift waiting to happen, and it had already happened.
    """
    analysis = make_done_analysis(auth_client.user)

    created = auth_client.post(
        reverse('analysis-report', args=[analysis.analysis_id]))
    assert created.status_code == 201
    body = created.json()

    listed = auth_client.get(reverse('report-list')).json()['results']
    row = next(r for r in listed if r['report_id'] == body['report_id'])

    assert set(body) == set(row), (
        f'POST emits {sorted(set(body) ^ set(row))} differently from the '
        'list endpoint'
    )
    assert set(body) == {
        'report_id', 'analysis_id', 'generated_at', 'page_count',
        'file_size_bytes', 'download_url', 'analysis',
    }
    assert body['analysis']['analysis_id'] == str(analysis.analysis_id)
    assert body['analysis']['media']['media_id'] == str(analysis.media_id)
    assert body == row


@pytest.mark.django_db
def test_f18_report_payload_is_the_reports_serializer(auth_client):
    """No hand-rolled dict remains in the analysis views."""
    from analysis import views as analysis_views
    from reports.serializers import ReportListSerializer

    analysis = make_done_analysis(auth_client.user)
    report = generate_report_for_analysis(analysis)

    assert (analysis_views.report_payload(report)
            == ReportListSerializer(report).data)


# ===========================================================================
# F19 — refreshing a token whose account is gone is a 401, not a 500
# ===========================================================================

@pytest.mark.django_db
def test_f19_refresh_after_the_account_was_deleted_is_401(api, user_factory):
    """
    A deleted account is routine; the SPA's refresh interceptor handles 401.

    SimpleJWT's ``TokenRefreshSerializer`` resolves the ``user_id`` claim
    with ``objects.get(...)`` and lets ``User.DoesNotExist`` escape — which
    reached the global handler as an unrecognised exception and produced a
    500 with a logged traceback.
    """
    user = user_factory()
    login = api.post(
        reverse('accounts-login'),
        {'email': user.email, 'password': DEFAULT_TEST_PASSWORD},
        format='json', REMOTE_ADDR='10.99.0.1',
    )
    assert login.status_code == 200
    refresh_token = login.json()['refresh']

    user.delete()

    response = api.post(reverse('accounts-refresh-token'),
                        {'refresh': refresh_token}, format='json')

    assert response.status_code == 401, response.json()
    assert response.json()['code'] == 'token_not_valid'


@pytest.mark.django_db
def test_f19_a_live_account_still_refreshes(api, user_factory):
    """The narrow catch must not swallow the happy path."""
    user = user_factory()
    login = api.post(
        reverse('accounts-login'),
        {'email': user.email, 'password': DEFAULT_TEST_PASSWORD},
        format='json', REMOTE_ADDR='10.99.0.2',
    )
    response = api.post(reverse('accounts-refresh-token'),
                        {'refresh': login.json()['refresh']}, format='json')

    assert response.status_code == 200
    assert response.json()['access']
    assert response.json()['access_expires_in'] > 0


# ===========================================================================
# F20a — a double-submitted signup is a 400, not a 500
# ===========================================================================

@pytest.mark.django_db(transaction=True)
def test_f20a_simultaneous_signups_yield_one_201_and_one_400(settings):
    """
    Two identical registrations, genuinely at once.

    Both clear the ``.exists()`` probe — the window between it and the INSERT
    is a whole PBKDF2 hash wide — and the loser hits the UNIQUE index. That
    used to be an uncaught ``IntegrityError`` and therefore a 500, where the
    contract promises ``400 email_exists``.
    """
    settings.REST_FRAMEWORK = {
        **settings.REST_FRAMEWORK,
        'DEFAULT_THROTTLE_CLASSES': [],
    }
    email = f'race-{uuid.uuid4().hex[:10]}@example.com'
    payload = {'email': email, 'password': DEFAULT_TEST_PASSWORD,
               'full_name': 'Race Condition'}

    def register(index):
        client = APIClient()
        response = client.post(reverse('accounts-register'), payload,
                               format='json',
                               REMOTE_ADDR=f'10.50.0.{index + 1}')
        return response.status_code, response.json()

    outcomes = run_concurrently(register, count=2)
    assert [exc for _i, _r, exc in outcomes if exc is not None] == []

    codes = sorted(result[0] for _i, result, _e in outcomes)
    bodies = {result[0]: result[1] for _i, result, _e in outcomes}

    assert 500 not in codes, bodies
    assert codes == [201, 400], bodies
    assert bodies[400]['code'] == 'email_exists'
    assert bodies[400]['errors']['email']
    assert User.objects.filter(email=email).count() == 1


@pytest.mark.django_db
def test_f20a_the_ordinary_duplicate_signup_is_unchanged(api, user_factory):
    """The fast path still answers 400 without touching the constraint."""
    existing = user_factory()

    response = api.post(
        reverse('accounts-register'),
        {'email': existing.email, 'password': DEFAULT_TEST_PASSWORD,
         'full_name': 'Copycat'},
        format='json', REMOTE_ADDR='10.51.0.1',
    )

    assert response.status_code == 400
    assert response.json()['code'] == 'email_exists'
    assert User.objects.filter(email=existing.email).count() == 1


# ===========================================================================
# F20b — one in-flight analysis per upload, under genuine concurrency
# ===========================================================================

@pytest.mark.django_db
def test_f20b_start_analysis_refuses_a_second_run(auth_client):
    """
    Admission control lives in ``start_analysis``, inside its transaction.

    The view's pre-check is a courtesy; this is the guarantee, and it has to
    hold even when the caller skipped the courtesy.
    """
    media = make_media(auth_client.user)
    first = AnalysisResult.objects.create(
        media=media, user=auth_client.user, status=STATUS_PENDING,
        settings_snapshot={},
    )

    with pytest.raises(services.AnalysisInFlight) as refused:
        services.start_analysis(user=auth_client.user, media=media)

    assert refused.value.analysis.analysis_id == first.analysis_id
    assert AnalysisResult.objects.filter(media=media).count() == 1


@pytest.mark.django_db(transaction=True)
def test_f20b_two_simultaneous_analyze_requests_start_one_run(monkeypatch):
    """
    Two clicks 50 ms apart must produce one run and one 409.

    The check-then-act window is widened deterministically by slowing the
    snapshot build — that work genuinely happens between the view's
    ``active_analysis_for`` probe and the INSERT, and on a loaded box it is
    not fast. The fix closes the window at the database, so the delay makes
    no difference to the outcome; before the fix it makes the second insert
    a certainty rather than a coin flip.
    """
    owner = User.objects.create_user(
        email=f'analyze-{uuid.uuid4().hex[:10]}@example.com',
        password=DEFAULT_TEST_PASSWORD,
    )
    media = make_media(owner)

    real_snapshot = services.build_settings_snapshot

    def slow_snapshot(*args, **kwargs):
        result = real_snapshot(*args, **kwargs)
        time.sleep(0.15)
        return result

    monkeypatch.setattr(services, 'build_settings_snapshot', slow_snapshot)
    monkeypatch.setattr(worker.job_queue, 'enqueue', lambda analysis_id: None)

    def analyze(index):
        client = authenticate(APIClient(), owner)
        response = client.post(
            reverse('analysis-analyze'), {'media_id': str(media.media_id)},
            format='json',
        )
        return response.status_code, response.json()

    try:
        outcomes = run_concurrently(analyze, count=2)
        assert [exc for _i, _r, exc in outcomes if exc is not None] == []

        codes = sorted(result[0] for _i, result, _e in outcomes)
        bodies = {result[0]: result[1] for _i, result, _e in outcomes}

        assert codes == [202, 409], bodies
        assert bodies[409]['code'] == 'analysis_in_progress'
        assert bodies[409]['analysis_id'] == bodies[202]['analysis_id']
        assert AnalysisResult.objects.filter(media=media).count() == 1, (
            'the same upload was admitted twice: two artefact trees, two '
            'reports and two history rows'
        )
    finally:
        AnalysisResult.objects.filter(media=media).delete()
        UploadedMedia.objects.filter(pk=media.pk).delete()
        User.objects.filter(pk=owner.pk).delete()


# ===========================================================================
# F21 — the sensitivity slider is a strictness dial, and it ascends
# ===========================================================================

#: The slider's three labelled stops, left to right, as the web client sends
#: them: raw ``sensitivity`` un-inverted, ``confidence_threshold`` ascending
#: across 0.15 -> 0.75.
SLIDER_STOPS = [
    ('Low', 0, 0.15),
    ('Balanced', 50, 0.45),
    ('Strict', 100, 0.75),
]


@pytest.mark.django_db
def test_f21_higher_sensitivity_freezes_stricter_thresholds(auth_client):
    """
    Dragging toward "Strict" must make the run stricter, not noisier.

    The slider is labelled ``Low | Balanced | Strict`` left to right, and the
    PDF it ends up in is framed as enforcement evidence. The server mapped
    ``smoke_mask_threshold = 1 - sensitivity/100``, so the right-hand
    "Strict" stop produced the *loosest*, highest-false-positive
    configuration the API allows — an inspector asking for rigour got the
    noisiest possible analysis.

    Asserted as monotonicity across the whole slider rather than against
    three magic numbers: the direction is the contract, and a test pinned to
    literals would have to be rewritten (and could be quietly re-inverted)
    the next time the band is retuned.
    """
    media = make_media(auth_client.user)
    frozen = []

    for label, sensitivity, confidence in SLIDER_STOPS:
        analysis = services.start_analysis(
            user=auth_client.user, media=media,
            overrides={'sensitivity': sensitivity,
                       'confidence_threshold': confidence},
        )
        snapshot = analysis.settings_snapshot
        frozen.append((label, sensitivity, snapshot))
        # One run at a time per upload (F20b), so clear it before the next.
        AnalysisResult.objects.filter(pk=analysis.analysis_id).delete()

    masks = [s['smoke_mask_threshold'] for _l, _v, s in frozen]
    confidences = [s['confidence_threshold'] for _l, _v, s in frozen]
    recorded = [s['sensitivity'] for _l, _v, s in frozen]

    assert recorded == [0, 50, 100], 'the raw slider value must be frozen as sent'

    assert masks == sorted(masks) and len(set(masks)) == 3, (
        f'smoke_mask_threshold must ascend with sensitivity, got {masks} for '
        f'sensitivity {recorded} — "Strict" is the loosest setting on offer'
    )
    assert confidences == sorted(confidences) and len(set(confidences)) == 3, (
        f'confidence_threshold must ascend with sensitivity, got {confidences}'
    )

    # Both halves of the control point the same way at the labelled ends.
    low, _mid, strict = frozen
    assert strict[2]['smoke_mask_threshold'] > low[2]['smoke_mask_threshold']
    assert strict[2]['confidence_threshold'] > low[2]['confidence_threshold']


@pytest.mark.django_db
def test_f21_the_whole_slider_stays_off_the_clamp(auth_client):
    """
    No valid slider position may be silently corrected.

    The old mapping ran 1.00 down to 0.00 and so hit ``MASK_THRESHOLD_MAX``
    (0.95) at one end and ``MASK_THRESHOLD_MIN`` (0.05) at the other: the
    extremes of the control were being quietly clamped, which is its own
    small lie about what the run used. ``0.25 -> 0.75`` sits wholly inside
    the band.
    """
    media = make_media(auth_client.user)
    seen = {}

    for sensitivity in range(0, 101, 10):
        snapshot = services.build_settings_snapshot(
            overrides={'sensitivity': sensitivity})
        mask = snapshot['smoke_mask_threshold']
        assert services.MASK_THRESHOLD_MIN < mask < services.MASK_THRESHOLD_MAX, (
            f'sensitivity {sensitivity} landed on the clamp at {mask}')
        seen[sensitivity] = mask

    assert seen[0] == pytest.approx(services.MASK_THRESHOLD_AT_ZERO_SENSITIVITY)
    assert seen[100] == pytest.approx(
        services.MASK_THRESHOLD_AT_ZERO_SENSITIVITY
        + services.MASK_THRESHOLD_SENSITIVITY_SPAN)
    assert list(seen.values()) == sorted(seen.values())
    assert len(set(seen.values())) == len(seen), 'the mapping must be injective'


@pytest.mark.django_db
def test_f21_the_mask_threshold_reaches_the_pipeline_ascending(auth_client,
                                                               monkeypatch):
    """
    The corrected direction survives the snapshot -> MLConfig translation.

    ``build_ml_config`` renames the keys, and a mapping that ascended in the
    snapshot but was re-inverted on the way to ``mlcore`` would be the same
    defect one layer down.
    """
    import types

    captured = {}

    class FakeMLConfig:
        @classmethod
        def from_dict(cls, values):
            # Mirrors the real constructor seam: build_ml_config hands the
            # mapping to MLConfig.from_dict, not to MLConfig(**kwargs).
            captured.clear()
            captured.update(values)
            return cls()

    module = types.ModuleType('mlcore')
    module.MLConfig = FakeMLConfig
    monkeypatch.setitem(sys.modules, 'mlcore', module)

    thresholds = []
    for sensitivity in (10, 90):
        snapshot = services.build_settings_snapshot(
            overrides={'sensitivity': sensitivity})
        services.build_ml_config(snapshot)
        thresholds.append(captured['mask_threshold'])

    assert thresholds[1] > thresholds[0], (
        f'mask_threshold reached mlcore as {thresholds} for sensitivity '
        '(10, 90) — the inversion survived the translation'
    )


def test_f21_there_is_exactly_one_derivation_from_sensitivity():
    """
    Only ``build_settings_snapshot`` may turn ``sensitivity`` into a
    threshold.

    A second derivation site is how the two ends of this control drifted in
    the first place, and it is the kind of thing that reappears when someone
    needs the value somewhere else and copies four lines.
    """
    import inspect
    import re

    import analysis.services as services_module

    source = inspect.getsource(services_module)
    # Lines that both mention sensitivity and do arithmetic on it.
    derivations = [
        line.strip() for line in source.splitlines()
        if re.search(r'sensitivity\s*/\s*100', line)
    ]
    assert len(derivations) == 1, derivations
    assert 'MASK_THRESHOLD_SENSITIVITY_SPAN' in derivations[0]
    assert '1.0 -' not in derivations[0] and '1 -' not in derivations[0], (
        f'the inverted mapping is back: {derivations[0]}')
