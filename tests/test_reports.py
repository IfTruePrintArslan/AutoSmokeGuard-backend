"""
Tests for the reports API (UC-07): ``reports.services.generate_report_for_
analysis`` plus the ``GET /api/report/{id}``, ``GET /api/download-report/
{id}`` and ``GET /api/reports`` endpoints.

Fixtures build an ``AnalysisResult`` (with ``DetectedVehicle``/``SmokeRegion``
children) directly through the ORM — the ML pipeline is out of scope here,
only the report built from its *output* is under test. Test names carry the
project's TC ids where the task brief assigns one (``TC-10``..``TC-13``); the
rest cover force-regeneration semantics, ownership, edge cases and the UC-07
30-second NFR.
"""
import base64
import re
import time
import uuid
import zlib
from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone
from PIL import Image

from analysis.models import (
    REPORT_FAILED,
    REPORT_GENERATING,
    REPORT_PENDING,
    REPORT_READY,
    STATUS_DONE,
    STATUS_RUNNING,
    AnalysisResult,
    DetectedVehicle,
    SmokeRegion,
    default_severity_counts,
)
from common.storage import analysis_artifact_dir, from_media_relative
from reports import services as reports_services
from reports.models import GeneratedReport
from reports.services import ReportNotReady, generate_report_for_analysis
from uploads.models import UploadedMedia

from .conftest import make_jpeg_bytes, make_mp4_bytes

REPORT_URL = '/api/report/{}'
DOWNLOAD_URL = '/api/download-report/{}'
LIST_URL = '/api/reports'

VEHICLE_TYPES = ('car', 'truck', 'bus', 'motorcycle')
SEVERITIES = ('low', 'moderate', 'high')


# ---------------------------------------------------------------------------
# A minimal, dependency-free PDF text extractor.
#
# The rest of this file's own words are "no pypdf installed, so this asserts
# on raw bytes" (see TestPdfStructure below) -- true, and still the reason
# every other test here never looks past page/byte counts. Proving that a
# *character* survives the renderer needs more than that, and pulling in a
# PDF-parsing dependency just for a test is not a reason to change what
# reports/pdf.py takes as a hard runtime dependency.
#
# So this walks exactly what ``reportlab`` is observed to emit: objects
# delimited by ``obj``/``endobj``, streams filtered through some combination
# of ``/ASCII85Decode`` and ``/FlateDecode`` (both handled by the stdlib --
# ``base64``/``zlib``), literal strings on ``Tj``/``TJ`` operators, and two
# kinds of font: the built-in Helvetica/Times faces (``/WinAnsiEncoding``,
# decodable with the stdlib ``cp1252`` codec, which is a superset-compatible
# match) and the embedded DejaVu TrueType faces (an explicit ``/ToUnicode``
# CMap, parsed directly off its ``beginbfchar``/``endbfchar`` pairs -- the
# same mechanism a real PDF viewer's "copy text" or a tool like ``pdftotext``
# ultimately relies on). It is not a general PDF parser and does not need to
# be: it only has to read back what this one module writes.
# ---------------------------------------------------------------------------

_PDF_OBJECT_RE = re.compile(rb'(\d+)\s+0\s+obj(.*?)endobj', re.S)
_FONT_NAME_RE = re.compile(rb'/Name\s*/([A-Za-z0-9+]+)')
_TOUNICODE_REF_RE = re.compile(rb'/ToUnicode\s+(\d+)\s+0\s+R')
_BFCHAR_RE = re.compile(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>')
_TEXT_TOKEN_RE = re.compile(
    rb'/(?P<font>[A-Za-z0-9+]+)\s+[-\d.]+\s+Tf'
    rb'|\((?P<lit>(?:[^()\\]|\\.)*)\)\s*Tj'
    rb'|\[(?P<arr>(?:[^\[\]]|\\.)*)\]\s*TJ',
    re.S,
)


def _pdf_split_object(body):
    """``(header, raw_stream_bytes_or_None)`` for one ``obj ... endobj`` body."""
    idx = body.find(b'stream')
    if idx == -1:
        return body, None
    header = body[:idx]
    rest = body[idx + len(b'stream'):]
    if rest[:2] == b'\r\n':
        rest = rest[2:]
    elif rest[:1] == b'\n':
        rest = rest[1:]
    end = rest.rfind(b'endstream')
    return header, (rest[:end] if end != -1 else rest)


def _pdf_decode_stream(header, raw):
    if raw is None:
        return None
    raw = raw.rstrip(b'\r\n')
    if b'ASCII85Decode' in header:
        if raw.endswith(b'~>'):
            raw = raw[:-2]
        try:
            raw = base64.a85decode(raw, adobe=False)
        except ValueError:
            return None
    if b'FlateDecode' in header:
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            return None
    return raw


def _pdf_decode_literal(lit):
    """A PDF literal string's bytes (``\\ddd``/``\\n`` etc. escapes undone)."""
    out = bytearray()
    i = 0
    while i < len(lit):
        byte = lit[i]
        if byte == 0x5c and i + 1 < len(lit):  # backslash
            octal = re.match(rb'^[0-7]{1,3}', lit[i + 1:i + 4])
            if octal:
                out.append(int(octal.group(0), 8) & 0xFF)
                i += 1 + len(octal.group(0))
                continue
            out.append({0x6e: 0x0a, 0x72: 0x0d, 0x74: 0x09,
                        0x62: 0x08, 0x66: 0x0c}.get(lit[i + 1], lit[i + 1]))
            i += 2
            continue
        out.append(byte)
        i += 1
    return bytes(out)


def extract_pdf_text(pdf_bytes):
    """
    Reconstruct the visible text of a ``reports.pdf``-generated PDF, in the
    logical (drawing) order the renderer emitted it -- i.e. *not* re-ordered
    for right-to-left display the way a viewer would. Good enough to prove a
    value survived the renderer; see the section banner above for scope.
    """
    objects = {}
    for match in _PDF_OBJECT_RE.finditer(pdf_bytes):
        header, raw = _pdf_split_object(match.group(2))
        objects[int(match.group(1))] = (header, _pdf_decode_stream(header, raw))

    font_tables = {}
    for header, dec in objects.values():
        if b'/Type /Font' not in header and b'/Type/Font' not in header:
            continue
        name_match = _FONT_NAME_RE.search(header)
        if not name_match:
            continue
        name = name_match.group(1).decode('ascii')
        touc_match = _TOUNICODE_REF_RE.search(header)
        if touc_match:
            cmap_bytes = objects.get(int(touc_match.group(1)), (None, None))[1]
            table = {}
            if cmap_bytes:
                for code_hex, uni_hex in _BFCHAR_RE.findall(cmap_bytes):
                    codepoint = int(uni_hex, 16)
                    if codepoint:
                        table[int(code_hex, 16)] = codepoint
            font_tables[name] = table
        elif b'WinAnsiEncoding' in header:
            font_tables[name] = {
                code: ord(bytes([code]).decode('cp1252', errors='replace'))
                for code in range(256)
            }

    chars = []
    for _num, (_header, dec) in sorted(objects.items()):  # deterministic order
        if dec is None or b'Tf' not in dec or (b'Tj' not in dec and b'TJ' not in dec):
            continue
        current_font = None
        for token in _TEXT_TOKEN_RE.finditer(dec):
            if token.group('font') is not None:
                current_font = token.group('font').decode('ascii')
                continue
            table = font_tables.get(current_font, {})
            literals = (
                [token.group('lit')] if token.group('lit') is not None else
                [m.group(0)[1:-1] for m in re.finditer(rb'\((?:[^()\\]|\\.)*\)', token.group('arr'))]
            )
            for lit in literals:
                for byte in _pdf_decode_literal(lit):
                    codepoint = table.get(byte)
                    if codepoint:
                        chars.append(chr(codepoint))
        chars.append('\n')
    return ''.join(chars)


# ---------------------------------------------------------------------------
# Fixture builders — real ORM rows, no ML pipeline involved.
# ---------------------------------------------------------------------------

def _make_media(user, filename='clip.mp4', kind='video'):
    """A genuine ``UploadedMedia`` row, with a real (small) file on disk."""
    if kind == 'video':
        data = make_mp4_bytes()
        upload = SimpleUploadedFile(filename, data, content_type='video/mp4')
        return UploadedMedia.objects.create(
            user=user, filename=filename, format='mp4', media_type='video',
            size_bytes=len(data), file=upload, width=1280, height=720,
            duration_seconds=12.5,
        )
    data = make_jpeg_bytes()
    upload = SimpleUploadedFile(filename, data, content_type='image/jpeg')
    return UploadedMedia.objects.create(
        user=user, filename=filename, format='jpg', media_type='image',
        size_bytes=len(data), file=upload, width=1280, height=720,
    )


def _make_analysis(user, media=None, status=STATUS_DONE, **overrides):
    """A bare ``AnalysisResult`` — no vehicles yet, see :func:`_populate_vehicles`."""
    media = media or _make_media(user)
    now = timezone.now()
    defaults = dict(
        media=media, user=user, status=status,
        progress=100 if status == STATUS_DONE else 40,
        stage='' if status == STATUS_DONE else 'segmenting frames',
        start_time=now, end_time=now if status == STATUS_DONE else None,
        total_vehicles=0, total_smoke=0, frames_processed=1,
        avg_confidence=0.0, overall_severity='',
        severity_counts=default_severity_counts(),
        settings_snapshot={
            'confidence_threshold': 0.35, 'smoke_mask_threshold': 0.5,
            'severity_low_max': 0.33, 'severity_moderate_max': 0.66,
            'frame_sample_rate': 5, 'max_video_seconds': 300,
        },
    )
    defaults.update(overrides)
    return AnalysisResult.objects.create(**defaults)


def _add_vehicle(analysis, vehicle_type='car', confidence=0.9, frame_number=1,
                  timestamp=1.0, smoke=None):
    vehicle = DetectedVehicle.objects.create(
        analysis=analysis, vehicle_type=vehicle_type,
        bounding_box={'x': 10, 'y': 10, 'w': 100, 'h': 80},
        confidence=confidence, frame_number=frame_number,
        timestamp_seconds=timestamp,
    )
    if smoke is not None:
        SmokeRegion.objects.create(vehicle=vehicle, **smoke)
    return vehicle


def _populate_vehicles(analysis, count, with_smoke=True):
    """Add *count* vehicles (roughly half smoking, cycling through severities)."""
    counts = {'low': 0, 'moderate': 0, 'high': 0}
    for i in range(count):
        smoke = None
        if with_smoke and i % 2 == 0:
            severity = SEVERITIES[i % 3]
            smoke = dict(
                intensity=round(0.2 + (i % 3) * 0.3, 3), severity=severity,
                confidence=0.8, area_ratio=round(0.1 + 0.01 * i, 3),
                opacity=0.5, mask_path='',
            )
            counts[severity] += 1
        _add_vehicle(
            analysis, vehicle_type=VEHICLE_TYPES[i % len(VEHICLE_TYPES)],
            confidence=round(0.5 + (i % 5) * 0.09, 3),
            frame_number=i, timestamp=float(i), smoke=smoke,
        )

    analysis.total_vehicles = count
    analysis.total_smoke = sum(counts.values())
    analysis.severity_counts = counts
    if counts['high']:
        analysis.overall_severity = 'high'
    elif counts['moderate']:
        analysis.overall_severity = 'moderate'
    elif counts['low']:
        analysis.overall_severity = 'low'
    analysis.avg_confidence = 0.8
    analysis.frames_processed = max(count, 1)
    analysis.save()
    return counts


def _write_frame(analysis, frame_number, size=(640, 480), colour=(80, 120, 160)):
    """A real, decodable annotated frame under ``analyses/<id>/frames/``."""
    frames_dir = analysis_artifact_dir(analysis.analysis_id, 'frames')
    path = frames_dir / f'frame_{frame_number:06d}.jpg'
    Image.new('RGB', size, colour).save(path, format='JPEG')
    return path


# ===========================================================================
# TC-10 — a report for an analysis that is not done yet
# ===========================================================================

@pytest.mark.django_db
class TestTC10ReportNotReady:

    def test_generate_report_raises_for_running_analysis(self, auth_client):
        analysis = _make_analysis(auth_client.user, status=STATUS_RUNNING, end_time=None)

        with pytest.raises(ReportNotReady) as exc_info:
            generate_report_for_analysis(analysis)

        assert 'not ready' in str(exc_info.value).lower()
        assert not GeneratedReport.objects.filter(analysis=analysis).exists()

    def test_download_surfaces_409_report_not_ready(self, auth_client):
        """
        The download endpoint regenerates a report whose file has gone
        missing from disk. If the underlying analysis has since regressed out
        of 'done' (a race in principle, exercised directly here), that
        regeneration attempt must surface the same 409 envelope, not a 500.
        """
        analysis = _make_analysis(auth_client.user, status=STATUS_DONE)
        report = generate_report_for_analysis(analysis)
        from_media_relative(report.report_path).unlink()
        AnalysisResult.objects.filter(pk=analysis.pk).update(
            status=STATUS_RUNNING, end_time=None,
        )

        response = auth_client.get(DOWNLOAD_URL.format(report.report_id))

        assert response.status_code == 409
        assert response.data['code'] == 'report_not_ready'
        assert 'not ready' in response.data['detail'].lower()


# ===========================================================================
# TC-11 — generate_report_for_analysis produces a real file and row
# ===========================================================================

@pytest.mark.django_db
class TestTC11GenerateReport:

    def test_produces_file_on_disk_and_a_populated_row(self, auth_client):
        analysis = _make_analysis(auth_client.user)
        _populate_vehicles(analysis, 5)

        report = generate_report_for_analysis(analysis)

        assert isinstance(report, GeneratedReport)
        assert report.analysis_id == analysis.analysis_id
        assert report.page_count > 0
        assert report.file_size_bytes > 0

        absolute = from_media_relative(report.report_path)
        assert absolute.is_file()
        assert absolute.stat().st_size == report.file_size_bytes
        assert absolute.read_bytes()[:4] == b'%PDF'


# ===========================================================================
# TC-12 — GET /api/download-report/{id}
# ===========================================================================

@pytest.mark.django_db
class TestTC12Download:

    def test_download_ok(self, auth_client):
        analysis = _make_analysis(auth_client.user)
        _populate_vehicles(analysis, 3)
        report = generate_report_for_analysis(analysis)

        response = auth_client.get(DOWNLOAD_URL.format(report.report_id))

        assert response.status_code == 200
        assert response['Content-Type'] == 'application/pdf'
        assert 'attachment' in response['Content-Disposition']
        assert f'{str(report.report_id)[:8]}' in response['Content-Disposition']

        body = b''.join(response.streaming_content)
        assert body[:4] == b'%PDF'


# ===========================================================================
# TC-13 — unknown report id
# ===========================================================================

@pytest.mark.django_db
class TestTC13UnknownReport:

    def test_unknown_id_json_404(self, auth_client):
        response = auth_client.get(REPORT_URL.format(uuid.uuid4()))

        assert response.status_code == 404
        assert response.data['code'] == 'report_not_found'

    def test_unknown_id_download_404(self, auth_client):
        response = auth_client.get(DOWNLOAD_URL.format(uuid.uuid4()))

        assert response.status_code == 404
        assert response.data['code'] == 'report_not_found'


# ===========================================================================
# Ownership — another user's report is invisible; an admin can see everything.
# ===========================================================================

@pytest.mark.django_db
class TestOwnership:

    def test_other_users_report_is_404_for_json_and_download(self, auth_client, user_factory):
        owner = user_factory()
        analysis = _make_analysis(owner)
        report = generate_report_for_analysis(analysis)

        json_response = auth_client.get(REPORT_URL.format(report.report_id))
        download_response = auth_client.get(DOWNLOAD_URL.format(report.report_id))

        assert json_response.status_code == 404
        assert json_response.data['code'] == 'report_not_found'
        assert download_response.status_code == 404
        assert download_response.data['code'] == 'report_not_found'

    def test_admin_can_fetch_another_users_report(self, admin_client, user_factory):
        owner = user_factory()
        analysis = _make_analysis(owner)
        _populate_vehicles(analysis, 2)
        report = generate_report_for_analysis(analysis)

        response = admin_client.get(REPORT_URL.format(report.report_id))

        assert response.status_code == 200
        assert response.data['report_id'] == str(report.report_id)
        assert response.data['analysis']['analysis_id'] == str(analysis.analysis_id)

    def test_admin_can_download_another_users_report(self, admin_client, user_factory):
        owner = user_factory()
        analysis = _make_analysis(owner)
        report = generate_report_for_analysis(analysis)

        response = admin_client.get(DOWNLOAD_URL.format(report.report_id))

        assert response.status_code == 200


# ===========================================================================
# force=True / force=False semantics
# ===========================================================================

@pytest.mark.django_db
class TestForceRegeneration:

    def test_force_false_returns_existing_row_unchanged(self, auth_client):
        analysis = _make_analysis(auth_client.user)
        first = generate_report_for_analysis(analysis)
        first_path = from_media_relative(first.report_path)
        first_mtime = first_path.stat().st_mtime

        second = generate_report_for_analysis(analysis, force=False)

        assert second.report_id == first.report_id
        assert first_path.stat().st_mtime == first_mtime

    def test_force_true_regenerates_and_replaces_the_file(self, auth_client):
        analysis = _make_analysis(auth_client.user)
        first = generate_report_for_analysis(analysis)
        first_bytes = from_media_relative(first.report_path).read_bytes()

        _populate_vehicles(analysis, 4)
        second = generate_report_for_analysis(analysis, force=True)
        second_bytes = from_media_relative(second.report_path).read_bytes()

        assert second.report_id == first.report_id  # same row, same on-disk path
        assert second_bytes != first_bytes
        assert GeneratedReport.objects.filter(analysis=analysis).count() == 1


# ===========================================================================
# Edge cases that must not crash
# ===========================================================================

@pytest.mark.django_db
class TestEdgeCases:

    def test_zero_vehicles_still_produces_a_valid_pdf(self, auth_client):
        analysis = _make_analysis(auth_client.user)  # no vehicles added

        report = generate_report_for_analysis(analysis)
        raw = from_media_relative(report.report_path).read_bytes()

        assert raw.startswith(b'%PDF-')
        assert raw.rstrip().endswith(b'%%EOF')
        assert report.page_count >= 1

    def test_vehicles_without_smoke_do_not_crash(self, auth_client):
        analysis = _make_analysis(auth_client.user)
        _populate_vehicles(analysis, 4, with_smoke=False)

        report = generate_report_for_analysis(analysis)

        assert report.page_count > 0
        assert analysis.total_smoke == 0

    def test_missing_annotated_frames_directory_is_skipped(self, auth_client):
        analysis = _make_analysis(auth_client.user)
        _populate_vehicles(analysis, 3)
        # No analyses/<id>/frames/ directory is ever created on disk.

        report = generate_report_for_analysis(analysis)

        assert report.page_count > 0

    def test_corrupt_frame_file_is_skipped_gracefully(self, auth_client):
        analysis = _make_analysis(auth_client.user)
        _populate_vehicles(analysis, 2)
        frames_dir = analysis_artifact_dir(analysis.analysis_id, 'frames')
        (frames_dir / 'frame_000001.jpg').write_bytes(b'not actually a jpeg')
        _write_frame(analysis, 2)  # one genuine frame alongside the corrupt one

        report = generate_report_for_analysis(analysis)

        assert report.page_count > 0

    def test_missing_preview_image_is_skipped(self, auth_client):
        analysis = _make_analysis(auth_client.user, preview_path='analyses/does-not-exist/preview.jpg')
        _populate_vehicles(analysis, 1)

        report = generate_report_for_analysis(analysis)

        assert report.page_count > 0

    def test_extremely_long_filename_does_not_crash(self, auth_client):
        long_name = ('x' * 300) + '.jpg'
        media = _make_media(auth_client.user, filename=long_name, kind='image')
        analysis = _make_analysis(auth_client.user, media=media)
        _populate_vehicles(analysis, 1)

        report = generate_report_for_analysis(analysis)

        assert report.page_count > 0

    def test_prefers_the_worker_stashed_runtime_metadata(self, auth_client):
        """
        When the analysis worker has stashed device/segmenter_mode/annotated
        frames under settings_snapshot['_runtime'] (see analysis.services
        .RUNTIME_KEY), the report should use that authoritative, per-run data
        rather than guessing the current render-time device.
        """
        media = _make_media(auth_client.user, kind='video')
        analysis = _make_analysis(auth_client.user, media=media)
        _populate_vehicles(analysis, 2)
        frame_path = _write_frame(analysis, 0)
        relative_frame = str(
            Path('analyses') / str(analysis.analysis_id) / 'frames' / frame_path.name
        )
        snapshot = dict(analysis.settings_snapshot)
        snapshot['_runtime'] = {
            'device': 'mps', 'segmenter_mode': 'unet',
            'annotated_frames': [relative_frame],
        }
        analysis.settings_snapshot = snapshot
        analysis.save(update_fields=['settings_snapshot'])

        report = generate_report_for_analysis(analysis)

        assert report.page_count > 0

    def test_deleted_pdf_is_regenerated_on_download(self, auth_client):
        analysis = _make_analysis(auth_client.user)
        _populate_vehicles(analysis, 2)
        report = generate_report_for_analysis(analysis)
        absolute = from_media_relative(report.report_path)
        absolute.unlink()
        assert not absolute.is_file()

        response = auth_client.get(DOWNLOAD_URL.format(report.report_id))

        assert response.status_code == 200
        assert absolute.is_file()
        body = b''.join(response.streaming_content)
        assert body[:4] == b'%PDF'


# ===========================================================================
# NFR — a report must be produced within 30 seconds (UC-07).
# ===========================================================================

@pytest.mark.django_db
class TestNFRTiming:

    def test_thirty_vehicles_six_frames_within_budget(self, auth_client):
        media = _make_media(auth_client.user, kind='video')
        analysis = _make_analysis(auth_client.user, media=media)
        _populate_vehicles(analysis, 30)
        for i in range(6):
            _write_frame(analysis, i)

        started = time.monotonic()
        report = generate_report_for_analysis(analysis)
        elapsed = time.monotonic() - started

        print(f'\n[NFR] report for 30 vehicles / 6 frames took {elapsed:.2f}s '
              f'({report.page_count} pages, {report.file_size_bytes} bytes)')

        assert elapsed < 30
        assert report.page_count > 0


# ===========================================================================
# Structural parse-back — no pypdf installed, so this asserts on raw bytes.
# ===========================================================================

@pytest.mark.django_db
class TestPdfStructure:

    def test_page_count_matches_raw_page_object_count(self, auth_client):
        analysis = _make_analysis(auth_client.user)
        _populate_vehicles(analysis, 60)  # enough rows to force a page break

        report = generate_report_for_analysis(analysis)
        raw = from_media_relative(report.report_path).read_bytes()

        assert raw.startswith(b'%PDF-')
        assert b'/Type /Page' in raw
        assert raw.rstrip().endswith(b'%%EOF')

        # '/Type /Page' is a substring of the single '/Type /Pages' container
        # object too, so it is subtracted out to get the real page count.
        page_objects = raw.count(b'/Type /Page') - raw.count(b'/Type /Pages')
        assert page_objects == report.page_count
        assert report.page_count > 1  # 60 detection rows must have paginated


# ===========================================================================
# The 200-row detection cap — never exercised by any test above, all of
# which stay at <=60 vehicles.
# ===========================================================================

@pytest.mark.django_db
class TestDetectionRowCap:

    def test_over_200_vehicles_are_capped_with_a_correct_remainder_message(self, auth_client):
        analysis = _make_analysis(auth_client.user)
        _populate_vehicles(analysis, 215)  # 15 over the MAX_DETECTION_ROWS cap

        report = generate_report_for_analysis(analysis)
        text = extract_pdf_text(from_media_relative(report.report_path).read_bytes())

        assert '…and 15 more detection(s), not shown.' in text
        # Exactly one remainder note -- not one per page the table spills onto.
        assert text.count('not shown.') == 1
        assert report.page_count > 1

    def test_exactly_200_vehicles_shows_no_remainder_message(self, auth_client):
        """No-change guard: the cap's edge is '> 200', not '>= 200'."""
        analysis = _make_analysis(auth_client.user)
        _populate_vehicles(analysis, 200)

        report = generate_report_for_analysis(analysis)
        text = extract_pdf_text(from_media_relative(report.report_path).read_bytes())

        assert 'not shown.' not in text


# ===========================================================================
# Fix 1 — non-Latin-1 free text (filename, analyst name/email) must survive
# the renderer, never render as silent ``.notdef`` boxes.
# ===========================================================================

@pytest.mark.django_db
class TestNonLatin1FreeText:

    def test_urdu_filename_and_analyst_name_survive_into_the_pdf(
            self, user_factory):
        """
        The concrete, realistic case the fix exists for: a Pakistani user
        uploading a file named in Urdu, and an Urdu analyst name. Both are
        genuinely user-supplied free text (media.filename, user.full_name),
        drawn in the embedded DejaVu Unicode font -- extracting the PDF's
        own text proves the characters made it in, not merely that nothing
        crashed.
        """
        user = user_factory(full_name='محمد علی')
        media = _make_media(user, filename='درخواست فائل.mp4', kind='video')
        analysis = _make_analysis(user, media=media)
        _populate_vehicles(analysis, 2)

        report = generate_report_for_analysis(analysis)
        text = extract_pdf_text(from_media_relative(report.report_path).read_bytes())

        assert 'درخواست فائل.mp4' in text
        assert 'محمد علی' in text
        assert '�' not in text  # nothing here was outside the embedded font

    def test_characters_outside_the_embedded_font_become_a_visible_marker(
            self, auth_client, caplog):
        """
        DejaVu Sans's own coverage stops short of CJK ideographs (confirmed
        with a filename containing 日本語): each such character must become
        a visible U+FFFD marker -- never a silent ``.notdef`` box -- and the
        field it came from must be named in a logged warning.
        """
        media = _make_media(auth_client.user, filename='日本語.mp4', kind='video')
        analysis = _make_analysis(auth_client.user, media=media)
        _populate_vehicles(analysis, 1)

        report = generate_report_for_analysis(analysis)
        text = extract_pdf_text(from_media_relative(report.report_path).read_bytes())

        assert '���.mp4' in text  # one marker per unsupported character
        assert '日本語' not in text
        assert 'filename' in caplog.text
        assert '日本語' in caplog.text  # the field's actual value is named, not hidden


# ===========================================================================
# Fix 2 — the severity chip must not dress up 'none' or unrecognised junk
# as a real low/moderate/high severity band.
# ===========================================================================

@pytest.mark.django_db
class TestSeverityChipRobustness:

    def test_overall_severity_none_renders_as_na_like_the_empty_case(self, auth_client):
        """
        ``AnalysisResult.overall_severity`` genuinely stores the literal
        string ``'none'`` (the pipeline's internal sentinel for "no smoke
        detected" -- see mlcore/pipeline.py), distinct from ``''`` at the
        database level even though both mean the same thing to a reader.
        Both must render identically as the existing neutral N/A chip, not
        as a look-alike 'NONE' severity band.
        """
        analysis = _make_analysis(auth_client.user, overall_severity='none')

        report = generate_report_for_analysis(analysis)
        text = extract_pdf_text(from_media_relative(report.report_path).read_bytes())

        assert 'N/A' in text
        assert 'NONE' not in text

    def test_unrecognised_overall_severity_is_flagged_not_legitimised(
            self, auth_client, caplog):
        """
        Any value outside {'', 'none', 'low', 'moderate', 'high'} is not a
        real severity band -- rendering it upper-cased in the normal chip
        style would dress up a bad value (e.g. a typo'd 'critical') as one
        the reader could mistake for real output. It must render distinctly
        and be logged instead.
        """
        analysis = _make_analysis(auth_client.user, overall_severity='critical')

        report = generate_report_for_analysis(analysis)
        text = extract_pdf_text(from_media_relative(report.report_path).read_bytes())

        assert 'UNKNOWN' in text
        assert 'CRITICAL' not in text
        assert 'critical' in caplog.text


# ===========================================================================
# GET /api/reports — pagination and per-user scoping
# ===========================================================================

@pytest.mark.django_db
class TestListEndpoint:

    def test_paginates_and_only_shows_the_callers_reports(self, auth_client, user_factory):
        other = user_factory()
        mine_ids = set()
        for i in range(3):
            media = _make_media(auth_client.user, filename=f'mine-{i}.jpg', kind='image')
            analysis = _make_analysis(auth_client.user, media=media)
            report = generate_report_for_analysis(analysis)
            mine_ids.add(str(report.report_id))

        other_analysis = _make_analysis(other)
        generate_report_for_analysis(other_analysis)

        response = auth_client.get(LIST_URL, {'page': 1, 'page_size': 2})

        assert response.status_code == 200
        body = response.data
        assert {'count', 'page', 'pages', 'page_size', 'next', 'previous', 'results'} <= set(body)
        assert body['count'] == 3
        assert body['page_size'] == 2
        assert len(body['results']) == 2
        for row in body['results']:
            assert row['report_id'] in mine_ids
            # Slim nested analysis summary, not the full AnalysisDetail.
            assert set(row['analysis']) == {
                'analysis_id', 'overall_severity', 'total_vehicles',
                'total_smoke', 'media',
            }
            assert 'vehicles' not in row['analysis']
            assert 'settings_snapshot' not in row['analysis']

        second_page = auth_client.get(LIST_URL, {'page': 2, 'page_size': 2})
        assert second_page.status_code == 200
        assert len(second_page.data['results']) == 1

    def test_row_analysis_summary_has_filename_and_severity(self, auth_client):
        """
        The fields the Reports screen actually renders: source filename and
        overall severity, both of which only exist inside the nested
        analysis — this is the G16 fix, replacing the old bare ReportObj
        that forced the frontend to fall back to '—' placeholders.
        """
        media = _make_media(auth_client.user, filename='smoking-truck.mp4', kind='video')
        analysis = _make_analysis(auth_client.user, media=media)
        _populate_vehicles(analysis, 5)
        report = generate_report_for_analysis(analysis)

        response = auth_client.get(LIST_URL)

        assert response.status_code == 200
        row = next(r for r in response.data['results'] if r['report_id'] == str(report.report_id))
        assert row['analysis']['analysis_id'] == str(analysis.analysis_id)
        assert row['analysis']['overall_severity'] == analysis.overall_severity
        assert row['analysis']['total_vehicles'] == analysis.total_vehicles
        assert row['analysis']['total_smoke'] == analysis.total_smoke
        assert row['analysis']['media'] == {
            'media_id': str(media.media_id),
            'filename': 'smoking-truck.mp4',
            'media_type': 'video',
        }

    def test_row_analysis_summary_severity_is_null_not_empty_string(self, auth_client):
        """
        `overall_severity` is stored as `''` on the model until anything is
        detected; the wire value must be `null`, matching the severity
        vocabulary (`"low"|"moderate"|"high"`), not an empty string.
        """
        analysis = _make_analysis(auth_client.user)  # no vehicles -> overall_severity == ''
        assert analysis.overall_severity == ''
        report = generate_report_for_analysis(analysis)

        response = auth_client.get(LIST_URL)

        row = next(r for r in response.data['results'] if r['report_id'] == str(report.report_id))
        assert row['analysis']['overall_severity'] is None

    def test_list_endpoint_analysis_summary_has_no_n_plus_one(self, auth_client, django_assert_num_queries):
        """
        The nested `analysis` summary must not reintroduce an N+1: listing
        one report and listing several must cost the exact same number of
        queries, proving `_visible_reports`'s `select_related('analysis',
        'analysis__media')` — not a per-row lookup — is what backs the
        nested filename/severity/media fields.
        """
        def make_report(i):
            media = _make_media(auth_client.user, filename=f'row-{i}.jpg', kind='image')
            analysis = _make_analysis(auth_client.user, media=media)
            _populate_vehicles(analysis, 2)
            return generate_report_for_analysis(analysis)

        make_report(0)
        with django_assert_num_queries(3):
            one_row_response = auth_client.get(LIST_URL, {'page_size': 50})
        assert one_row_response.status_code == 200
        assert one_row_response.data['count'] == 1

        for i in range(1, 5):
            make_report(i)

        with django_assert_num_queries(3):
            many_rows_response = auth_client.get(LIST_URL, {'page_size': 50})
        assert many_rows_response.status_code == 200
        assert many_rows_response.data['count'] == 5


# ===========================================================================
# report_status — the server saying outright whether a PDF is still coming
# ===========================================================================

@pytest.mark.django_db
class TestReportStatusTransitions:
    """
    ``AnalysisResult.report_status`` as written by this seam.

    The field exists because ``report: null`` on a ``done`` analysis is
    ambiguous: the PDF may be half rendered, or its render may have died in
    the first two seconds.  A client cannot tell those apart from outside, and
    the one it used to assume — "still rendering, wait out the 30-second
    budget" — is wrong in exactly the case that hurts, leaving the user
    watching a spinner for work that was already over.

    Every assertion below reads the column back **out of the database** rather
    than off the in-memory instance: it is the stored row a polling client
    sees, and an instance attribute that agrees with a column that was never
    written would pass while proving nothing.
    """

    @staticmethod
    def _stored(analysis):
        return AnalysisResult.objects.values_list(
            'report_status', flat=True).get(pk=analysis.pk)

    @staticmethod
    def _explode(*_args, **_kwargs):
        raise RuntimeError('reportlab fell over')

    def test_an_analysis_starts_out_promising_nothing_more_than_pending(
            self, auth_client):
        """
        No-change guard (passes with and without the transitions below).

        Pins the column's starting point, which every transition test is
        measured against: 'nothing has been attempted yet' must be the
        default, so an unattempted render can never read as a finished one.
        """
        analysis = _make_analysis(auth_client.user)

        assert self._stored(analysis) == REPORT_PENDING

    def test_generating_is_readable_from_another_connection_mid_render(
            self, auth_client, monkeypatch):
        """
        The in-flight state is announced *before* the expensive call, not
        after it — a client that polls during the render must see
        ``generating``, which is the only value that positively promises a
        PDF is on its way.
        """
        analysis = _make_analysis(auth_client.user)
        real_build = reports_services.build_report
        observed = []

        def spy(analysis_arg, report_id, target):
            observed.append(self._stored(analysis_arg))
            return real_build(analysis_arg, report_id, target)

        monkeypatch.setattr(reports_services, 'build_report', spy)

        generate_report_for_analysis(analysis)

        assert observed == [REPORT_GENERATING], (
            'report_status was not committed as "generating" before the '
            f'render started; a mid-render poll would have read {observed}'
        )
        assert self._stored(analysis) == REPORT_READY

    def test_a_render_that_raises_lands_in_failed_and_is_never_left_generating(
            self, auth_client, monkeypatch):
        """
        The whole reason this field exists.

        A render that dies must say so.  Left on ``generating`` it is
        indistinguishable from one still in progress, and the client — which
        cannot see the exception — would keep a disabled "Preparing report…"
        on screen for the remainder of the server's 30-second budget, for work
        that ended seconds ago.
        """
        analysis = _make_analysis(auth_client.user)
        monkeypatch.setattr(reports_services, 'build_report', self._explode)

        with pytest.raises(RuntimeError, match='reportlab fell over'):
            generate_report_for_analysis(analysis)

        stored = self._stored(analysis)
        assert stored == REPORT_FAILED, (
            f'a failed render left report_status={stored!r}; the one state it '
            'must never be abandoned in is "generating"'
        )
        assert stored != REPORT_GENERATING
        assert not GeneratedReport.objects.filter(analysis=analysis).exists()

    def test_the_exception_still_reaches_the_caller_untouched(
            self, auth_client, monkeypatch):
        """
        Recording the failure must not swallow it, nor replace it.

        The worker's hook catches this so a bad PDF cannot fail a good
        analysis; ``POST /api/analysis/{id}/report`` lets it surface.  Both
        depend on the original exception arriving intact.

        No-change guard (passes with and without the transitions): it exists
        because the obvious way to record a failure — a bare ``except`` that
        does bookkeeping — is one raise away from replacing the exception the
        caller needed with a database error about the exception.
        """
        analysis = _make_analysis(auth_client.user)
        monkeypatch.setattr(reports_services, 'build_report', self._explode)

        with pytest.raises(RuntimeError) as exc_info:
            generate_report_for_analysis(analysis, force=True)

        assert str(exc_info.value) == 'reportlab fell over'
        assert exc_info.value.__class__ is RuntimeError

    def test_a_failed_regeneration_says_failed_beside_a_surviving_report(
            self, auth_client, monkeypatch):
        """
        ``ready`` implies a report exists; the converse does not hold.

        A forced re-render writes to a temporary file and only replaces the
        live PDF on success, so a failure leaves the previous one downloadable
        — but it is still a failure, and saying ``ready`` because an *older*
        PDF happens to be on disk would re-hide precisely what this field was
        added to surface.
        """
        analysis = _make_analysis(auth_client.user)
        report = generate_report_for_analysis(analysis)
        assert self._stored(analysis) == REPORT_READY
        bytes_before = from_media_relative(report.report_path).read_bytes()

        monkeypatch.setattr(reports_services, 'build_report', self._explode)
        with pytest.raises(RuntimeError):
            generate_report_for_analysis(analysis, force=True)

        assert self._stored(analysis) == REPORT_FAILED
        assert from_media_relative(report.report_path).read_bytes() == bytes_before

    def test_an_analysis_that_is_not_done_leaves_the_column_alone(
            self, auth_client):
        """
        ``ReportNotReady`` is a precondition, not a failure.

        Nothing was attempted, so nothing failed: marking this ``failed``
        would tell a client watching a *running* analysis that its report is
        never coming, while the run that will produce it is still going.

        No-change guard (passes with and without the transitions): it is the
        boundary of the new writes, not one of them.
        """
        analysis = _make_analysis(auth_client.user, status=STATUS_RUNNING,
                                  end_time=None)

        with pytest.raises(ReportNotReady):
            generate_report_for_analysis(analysis)

        assert self._stored(analysis) == REPORT_PENDING

    def test_the_fast_path_heals_a_row_written_before_this_field_existed(
            self, auth_client):
        """
        A PDF that demonstrably exists is ``ready``, whatever the column says.

        Rows created by an earlier build carry the ``pending`` default beside
        a perfectly good report; the cheap existing-report path corrects them
        on the way past rather than leaving the API describing a render that
        finished long ago as one that has not started.
        """
        analysis = _make_analysis(auth_client.user)
        generate_report_for_analysis(analysis)
        AnalysisResult.objects.filter(pk=analysis.pk).update(
            report_status=REPORT_PENDING)

        generate_report_for_analysis(analysis)      # force=False: no re-render

        assert self._stored(analysis) == REPORT_READY
