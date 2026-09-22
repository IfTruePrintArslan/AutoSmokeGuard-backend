"""
Non-functional-requirement tests that can run inside the normal CI suite:
NFR-2 (history query wall clock) and NFR-3 (upload acknowledgement wall
clock).

``docs/TRACEABILITY.md`` records both budgets as "not measured" before this
file existed. There is already a ``django_assert_num_queries`` test
(``tests/test_analysis.py::test_history_has_no_n_plus_one``) that locks the
*query count* for the history endpoint; the tests below add the wall-clock
dimension the NFR is actually stated in, against a realistic row volume
seeded with bulk ORM creation rather than through the API.

NFR-1 ("10 simultaneous users") is deliberately **not** here. It needs real
threads making real HTTP requests, with their own JWTs, against an
already-running server — that is what ``tools/benchmark.py`` is for. A
Django test client call is single-threaded and shares one request/response
cycle with the test process, so it cannot exercise genuine concurrency or
the loopback network stack the way the SRS's "10 simultaneous users" implies.
"""
import logging
import random
import time
from datetime import timedelta

import pytest
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APIClient

from analysis.models import VEHICLE_TYPE_CHOICES, AnalysisResult, DetectedVehicle
from uploads.models import UploadedMedia

from .conftest import authenticate, make_jpeg_bytes, make_png_bytes

logger = logging.getLogger('asg.tests.nfr')

ONE_MB = 1024 * 1024

# ---------------------------------------------------------------------------
# Budgets, from docs/TRACEABILITY.md
# ---------------------------------------------------------------------------

#: NFR-2 — "history query response < 2 seconds".
HISTORY_BUDGET_SECONDS = 2.0
#: NFR-3 — "upload acknowledgement < 3 seconds".
UPLOAD_ACK_BUDGET_SECONDS = 3.0

#: "a realistic volume" — the task sets this floor explicitly.
SEED_ROW_COUNT = 2000
#: Distinct uploads the seeded rows point at; re-runs (a re-analysed upload)
#: are the realistic case, not one row per media.
SEED_MEDIA_POOL_SIZE = 25


# ---------------------------------------------------------------------------
# NFR-2 — seeding
# ---------------------------------------------------------------------------

def _seed_media_pool(user, size=SEED_MEDIA_POOL_SIZE):
    """``size`` tiny real JPEGs owned by ``user`` — not the row under test."""
    media = []
    for index in range(size):
        content = make_jpeg_bytes(width=48, height=36,
                                  colour=(index % 255, 90, 140))
        row = UploadedMedia(
            user=user, filename=f'nfr2-seed-{index:04d}.jpg', format='jpg',
            media_type='image', size_bytes=len(content),
        )
        row.file.save(f'nfr2-seed-{index:04d}.jpg', ContentFile(content),
                      save=False)
        row.save()
        media.append(row)
    return media


def _seed_history_rows(user, media_pool, count=SEED_ROW_COUNT):
    """
    Bulk-create ``count`` :class:`AnalysisResult` rows (with vehicles
    attached) for ``user`` — never through the API.

    ``created_at`` is spread over the last 400 days with a follow-up
    ``bulk_update`` (rather than left at ``auto_now_add``'s "now") so
    ``date_from``/``date_to`` have something real to filter, the same way a
    year of real usage would.
    """
    severities = ('low', 'moderate', 'high', '')
    vehicle_types = tuple(choice for choice, _label in VEHICLE_TYPE_CHOICES)
    now = timezone.now()

    rows = []
    for _ in range(count):
        severity = random.choice(severities)
        counts = {'low': 0, 'moderate': 0, 'high': 0}
        if severity:
            counts[severity] = random.randint(1, 4)
        rows.append(AnalysisResult(
            media=random.choice(media_pool),
            user=user,
            status='done',
            progress=100,
            total_vehicles=random.randint(1, 5),
            total_smoke=counts[severity] if severity else 0,
            frames_processed=random.randint(1, 60),
            avg_confidence=round(random.uniform(0.5, 0.98), 4),
            overall_severity=severity,
            severity_counts=counts,
            settings_snapshot={},
            start_time=now,
            end_time=now,
        ))
    AnalysisResult.objects.bulk_create(rows, batch_size=500)

    for row in rows:
        row.created_at = now - timedelta(
            days=random.uniform(0, 400), minutes=random.uniform(0, 1440))
    AnalysisResult.objects.bulk_update(rows, ['created_at'], batch_size=500)

    vehicles = []
    for row in rows:
        for _ in range(random.randint(1, 3)):
            vehicles.append(DetectedVehicle(
                analysis=row,
                vehicle_type=random.choice(vehicle_types),
                bounding_box={'x': 0, 'y': 0, 'w': 10, 'h': 10},
                confidence=round(random.uniform(0.5, 0.99), 4),
            ))
    DetectedVehicle.objects.bulk_create(vehicles, batch_size=500)

    return rows


def _time_get(client, url, params=None):
    """``(elapsed_seconds, response)`` for one ``GET``."""
    started = time.perf_counter()
    response = client.get(url, params or {})
    elapsed = time.perf_counter() - started
    return elapsed, response


# ---------------------------------------------------------------------------
# NFR-2 — history query wall clock
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_nfr2_history_query_stays_under_the_two_second_budget(user_factory):
    """
    ``GET /api/history`` answers within the NFR-2 budget at a realistic row
    count: unfiltered, with each filter kind, and on the last page.

    2,000+ rows (with vehicles attached, for the ``vehicle_type`` filter) are
    seeded with bulk ORM creation — the existing
    ``test_history_has_no_n_plus_one`` test already locks the *query count*
    at three; this locks the *wall clock* the NFR is actually phrased in.
    """
    user = user_factory()
    client = authenticate(APIClient(), user)

    media_pool = _seed_media_pool(user)
    _seed_history_rows(user, media_pool)
    assert AnalysisResult.objects.filter(user=user).count() >= SEED_ROW_COUNT

    url = reverse('analysis-history')
    thirty_days_ago = (timezone.now() - timedelta(days=30)).date().isoformat()
    today = timezone.now().date().isoformat()

    cases = (
        ('unfiltered', {'page_size': 50}),
        ('status=done', {'status': 'done', 'page_size': 50}),
        ('severity=high', {'severity': 'high', 'page_size': 50}),
        ('vehicle_type=truck', {'vehicle_type': 'truck', 'page_size': 50}),
        ('date range (30d)', {'date_from': thirty_days_ago, 'date_to': today,
                              'page_size': 50}),
        ('search', {'search': 'nfr2-seed-000', 'page_size': 50}),
    )

    measured = {}
    for label, params in cases:
        elapsed, response = _time_get(client, url, params)
        assert response.status_code == 200, (
            f'{label}: {response.status_code} {response.data}')
        measured[label] = elapsed
        logger.info('NFR-2 %-18s %.3fs (budget %.1fs, %d result(s))',
                    label, elapsed, HISTORY_BUDGET_SECONDS,
                    response.data['count'])
        assert elapsed < HISTORY_BUDGET_SECONDS, (
            f'{label} took {elapsed:.2f}s, over the '
            f'{HISTORY_BUDGET_SECONDS:.0f}s NFR-2 budget')

    _, first_page = _time_get(client, url, {'page_size': 50})
    last_page_number = first_page.data['pages']
    elapsed, last_page = _time_get(
        client, url, {'page_size': 50, 'page': last_page_number})
    assert last_page.status_code == 200
    measured['last page'] = elapsed
    logger.info('NFR-2 last page (%d/%d) %.3fs (budget %.1fs)',
                last_page_number, last_page_number, elapsed,
                HISTORY_BUDGET_SECONDS)
    assert elapsed < HISTORY_BUDGET_SECONDS, (
        f'last page took {elapsed:.2f}s, over the '
        f'{HISTORY_BUDGET_SECONDS:.0f}s NFR-2 budget')

    logger.info('NFR-2 PASS — all %d case(s) under %.1fs: %s',
                len(measured), HISTORY_BUDGET_SECONDS,
                {k: round(v, 3) for k, v in measured.items()})


# ---------------------------------------------------------------------------
# NFR-3 — upload acknowledgement wall clock
# ---------------------------------------------------------------------------

def _padded_png(total_bytes):
    """A real, decodable PNG padded with trailing bytes to ``total_bytes``.

    Trailing bytes after the ``IEND`` chunk are ignored by every decoder,
    same trick ``tests/test_uploads.py`` uses for its 5 MB case.
    """
    base = make_png_bytes(width=64, height=48)
    if len(base) >= total_bytes:
        return base
    return base + (b'\x00' * (total_bytes - len(base)))


@pytest.mark.django_db
def test_nfr3_upload_acknowledgement_stays_under_the_three_second_budget(
        auth_client):
    """
    ``POST /api/upload`` acknowledges within the NFR-3 budget for a small
    image, a ~5 MB file and a ~50 MB file.

    ``tests/test_uploads.py::TestUploadPerformance`` already asserts the
    5 MB case without printing the number; this measures and logs all three
    sizes the NFR itself calls out explicitly.
    """
    cases = (
        ('small image', make_jpeg_bytes(width=64, height=48), 'jpg', 'image/jpeg'),
        ('~5 MB file', _padded_png(5 * ONE_MB), 'png', 'image/png'),
        ('~50 MB file', _padded_png(50 * ONE_MB), 'png', 'image/png'),
    )

    measured = {}
    for label, data, extension, content_type in cases:
        upload = SimpleUploadedFile(
            f'{label.replace(" ", "-").replace("~", "")}.{extension}', data,
            content_type=content_type,
        )
        started = time.perf_counter()
        response = auth_client.post(
            '/api/upload', {'file': upload}, format='multipart',
        )
        elapsed = time.perf_counter() - started

        assert response.status_code in (200, 201), (
            f'{label}: {response.status_code} {response.data}')
        measured[label] = elapsed
        logger.info('NFR-3 %-14s %7.2f MB  %.3fs (budget %.1fs)',
                    label, len(data) / ONE_MB, elapsed,
                    UPLOAD_ACK_BUDGET_SECONDS)
        assert elapsed < UPLOAD_ACK_BUDGET_SECONDS, (
            f'{label} upload took {elapsed:.2f}s, over the '
            f'{UPLOAD_ACK_BUDGET_SECONDS:.0f}s NFR-3 budget')

    logger.info('NFR-3 PASS — all %d case(s) under %.1fs: %s',
                len(measured), UPLOAD_ACK_BUDGET_SECONDS,
                {k: round(v, 3) for k, v in measured.items()})
