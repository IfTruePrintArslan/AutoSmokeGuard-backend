#!/usr/bin/env python
"""
Measure the three non-functional requirements ``docs/TRACEABILITY.md``
records as "not measured": NFR-1 (10 simultaneous users), NFR-2 (history
query < 2 s) and NFR-3 (upload acknowledgement < 3 s).

This script starts nothing on its own — it talks to an **already-running**
server over real HTTP (``requests``), the same way a browser would. Run it
like this::

    python manage.py runserver 8000      # in one terminal
    python tools/benchmark.py            # in another

    python tools/benchmark.py --api-port 8001
    python tools/benchmark.py --cleanup  # remove everything this script made

What each NFR measures and why
-------------------------------
**NFR-1** spins up ten real threads, each with its own registered user and
JWT, each carrying out the realistic session the SRS implies: log in, upload
an image, start an analysis, poll status to completion, list history, fetch
the dashboard, download the PDF. Per-endpoint p50/p95/max latency and the
total wall clock are reported. "Without noticeable performance degradation"
is operationalised as: **zero errors**, and **p95 latency for the cheap read
endpoints** (status poll, history, dashboard, report download) **stays at or
under 2 seconds** even while ten sessions — including whatever real
inference that provokes — are in flight together. Upload/analyze/register
are reported too but are not gated on that bound: they are dominated by real
file I/O or the ML pipeline's own runtime, which is a separate, already
self-tested budget (see ``mlcore.selftest``), not what "simultaneous users"
is asking about.

**NFR-2** seeds a realistic row volume directly through the ORM — bulk
creation, never through the API — then times ``GET /api/history``
unfiltered, with each filter, and on the last page, against the frozen 2 s
budget.

**NFR-3** measures (and prints) the upload acknowledgement time for a small
image, a ~5 MB file and a ~50 MB file against the frozen 3 s budget.

Seeded/created data and ``--cleanup``
--------------------------------------
Every account this script creates uses the ``@nfr-benchmark.invalid`` email
domain (RFC 2606 reserved — guaranteed to never collide with a real
address), so ``--cleanup`` can find and remove exactly this script's own
accounts, uploads, analyses, reports and files — and nothing else — leaving
the demo database exactly as it was.

Never fabricates a number: every reported figure is a real measured
duration from a real HTTP round trip (or a real bulk-ORM insert for
seeding). A step that could not be attempted at all (for example, an
analysis that never finished in time to download its report) is reported as
an error, not as a synthetic 0.00s sample.
"""
import argparse
import logging
import os
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import requests

# --------------------------------------------------------------------------- #
# Django bootstrap
#
# This script runs as __main__, not through manage.py, so it puts the
# project root on sys.path and configures Django itself -- exactly like
# tools/seed.py.
#
# ASG_WORKER_BOOTSTRAP is forced off *before* django.setup() runs. Without
# this, importing Django here would start the same daemon thread the live
# server's own process uses to sweep "interrupted" jobs and warm up the ML
# models (see analysis.worker.should_bootstrap / _bootstrap). Running that
# sweep from a *second* process while the server we are benchmarking may
# have real jobs mid-flight -- which NFR-1 deliberately puts in flight --
# would wrongly mark them "interrupted" and fail them out from under it.
# This process only ever needs plain ORM access for seeding and cleanup.
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
os.environ.setdefault('ASG_WORKER_BOOTSTRAP', 'False')

import django  # noqa: E402  (must follow the sys.path / env setup above)

django.setup()

from django.contrib.auth import get_user_model  # noqa: E402
from django.core.files.base import ContentFile  # noqa: E402
from django.db import transaction  # noqa: E402
from django.utils import timezone  # noqa: E402
from PIL import Image  # noqa: E402

from analysis.models import AnalysisResult, DetectedVehicle, VEHICLE_TYPE_CHOICES  # noqa: E402
from common.storage import delete_analysis_artifacts, report_path  # noqa: E402
from uploads.models import UploadedMedia  # noqa: E402

logger = logging.getLogger('asg.benchmark')

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8000

#: RFC 2606 reserved TLD: guaranteed to never resolve or collide with a real
#: account, and trivially identifiable for --cleanup.
BENCHMARK_EMAIL_DOMAIN = 'nfr-benchmark.invalid'
BENCHMARK_PASSWORD = 'NfrBench@12345'

# -- NFR-1: 10 simultaneous users -------------------------------------------
NFR1_USER_COUNT = 10
NFR1_READ_P95_BUDGET_SECONDS = 2.0
#: The endpoints the "no noticeable degradation" bound is measured against;
#: see the module docstring for why upload/analyze/register are excluded.
NFR1_READ_ENDPOINTS = ('status', 'history', 'dashboard', 'report_download')
NFR1_POLL_INTERVAL_SECONDS = 0.5
NFR1_POLL_TIMEOUT_SECONDS = 240.0

# -- NFR-2: history query < 2s ----------------------------------------------
NFR2_SEED_EMAIL = f'nfr2-seed@{BENCHMARK_EMAIL_DOMAIN}'
NFR2_SEED_COUNT = 2000
NFR2_MEDIA_POOL_SIZE = 25
NFR2_BUDGET_SECONDS = 2.0

# -- NFR-3: upload ack < 3s --------------------------------------------------
NFR3_BUDGET_SECONDS = 3.0
ONE_MB = 1024 * 1024


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #

@dataclass
class Sample:
    """One timed HTTP call (or a recorded failure to even attempt one)."""

    endpoint: str
    elapsed: float
    ok: bool
    detail: str = ''
    #: False for a synthetic placeholder (e.g. "never got a report to
    #: download") that must count as an error but must never be fabricated
    #: into the latency statistics.
    measured: bool = True


def _percentile(values, pct):
    """Nearest-rank percentile of *values* (0-100 scale). ``0.0`` if empty."""
    if not values:
        return 0.0
    data = sorted(values)
    rank = min(len(data), max(1, round(pct / 100.0 * len(data))))
    return data[rank - 1]


def _timed(method, url, **kwargs):
    """Perform one real HTTP call. Returns ``(elapsed_seconds, response, error)``."""
    timeout = kwargs.pop('timeout', 120)
    started = time.perf_counter()
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        return time.perf_counter() - started, None, str(exc)
    return time.perf_counter() - started, response, None


def _small_image_bytes(seed=0):
    """A tiny, genuinely decodable JPEG. Content does not matter here."""
    import io

    buffer = io.BytesIO()
    Image.new('RGB', (96, 72), (90 + seed % 100, 110, 130)).save(buffer, format='JPEG')
    return buffer.getvalue()


def _padded_png_bytes(total_bytes):
    """A real, decodable PNG padded with trailing bytes to ``total_bytes``."""
    import io

    buffer = io.BytesIO()
    Image.new('RGB', (64, 48), (70, 130, 180)).save(buffer, format='PNG')
    base = buffer.getvalue()
    if len(base) >= total_bytes:
        return base
    return base + (b'\x00' * (total_bytes - len(base)))


def _login(base_url, email, password, max_attempts=2):
    """
    ``POST /api/login``, waiting out ``accounts.views.LoginRateThrottle``
    (10/min per IP) once if it is hit, then retrying.

    Every request this whole script makes comes from the same loopback
    address, so two NFR phases run back-to-back inevitably share one
    throttle bucket in a way a real fleet of clients never would (NFR-1
    alone uses all ten of its own slots). Waiting for ``Retry-After`` here
    keeps that a benchmarking artefact rather than a fabricated failure: the
    wait happens before authentication succeeds, so it is never counted
    towards a login-latency sample, and it never touches the NFR-2 history
    timings, which only start once a token is already in hand.
    """
    for attempt in range(max_attempts):
        elapsed, response, error = _timed(
            'POST', f'{base_url}/api/login',
            json={'email': email, 'password': password},
        )
        throttled = response is not None and response.status_code == 429
        if throttled and attempt + 1 < max_attempts:
            wait = float(response.headers.get('Retry-After', 61)) + 0.5
            print(f'  (login throttled -- the shared per-IP limit is 10/min; '
                  f'waiting {wait:.0f}s for it to clear)')
            time.sleep(wait)
            continue
        return elapsed, response, error
    return elapsed, response, error  # pragma: no cover -- unreachable


def _server_reachable(base_url):
    """True if ``GET /api/health/`` answers at all."""
    try:
        response = requests.get(f'{base_url}/api/health/', timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


# --------------------------------------------------------------------------- #
# NFR-1 — 10 simultaneous users
# --------------------------------------------------------------------------- #

def _user_session(base_url, index, run_token):
    """
    One simulated user's full, realistic round trip.

    Never raises: a failure at any step is recorded as an ``ok=False``
    :class:`Sample` and the session stops there (there is nothing meaningful
    left to measure once, say, login itself failed).
    """
    samples = []
    email = f'nfr1-{run_token}-{index}@{BENCHMARK_EMAIL_DOMAIN}'

    def record(endpoint, elapsed, response, error, expect, measured=True):
        ok = error is None and response is not None and response.status_code in expect
        if ok:
            detail = ''
        elif error:
            detail = error
        else:
            detail = f'HTTP {response.status_code}: {response.text[:200]}'
        samples.append(Sample(endpoint, elapsed, ok, detail, measured))
        return ok

    # -- register ------------------------------------------------------- #
    elapsed, response, error = _timed(
        'POST', f'{base_url}/api/register',
        json={'email': email, 'password': BENCHMARK_PASSWORD,
              'full_name': f'NFR-1 Benchmark User {index}'},
    )
    if not record('register', elapsed, response, error, expect=(201,)):
        return samples

    # -- login ------------------------------------------------------------ #
    elapsed, response, error = _login(base_url, email, BENCHMARK_PASSWORD)
    if not record('login', elapsed, response, error, expect=(200,)):
        return samples
    headers = {'Authorization': f'Bearer {response.json()["access"]}'}

    # -- upload an image ---------------------------------------------------- #
    files = {'file': (f'nfr1-{index}.jpg', _small_image_bytes(index), 'image/jpeg')}
    elapsed, response, error = _timed(
        'POST', f'{base_url}/api/upload', headers=headers, files=files,
    )
    if not record('upload', elapsed, response, error, expect=(200, 201)):
        return samples
    media_id = response.json()['media_id']

    # -- start an analysis --------------------------------------------------- #
    elapsed, response, error = _timed(
        'POST', f'{base_url}/api/analyze', headers=headers,
        json={'media_id': media_id, 'settings': {'auto_generate_pdf': True}},
    )
    if not record('analyze', elapsed, response, error, expect=(202,)):
        return samples
    job_id = response.json()['job_id']

    # -- poll status to completion, then to a rendered report --------------- #
    deadline = time.monotonic() + NFR1_POLL_TIMEOUT_SECONDS
    status_value, report_id = None, None
    while time.monotonic() < deadline:
        elapsed, response, error = _timed(
            'GET', f'{base_url}/api/status/{job_id}', headers=headers,
        )
        if not record('status', elapsed, response, error, expect=(200,)):
            break
        body = response.json()
        status_value = body['status']
        report_id = body.get('report_id')
        if status_value == 'failed':
            break
        if status_value == 'done' and report_id:
            break
        time.sleep(NFR1_POLL_INTERVAL_SECONDS)

    # -- list history --------------------------------------------------- #
    elapsed, response, error = _timed(
        'GET', f'{base_url}/api/history', headers=headers, params={'page_size': 10},
    )
    record('history', elapsed, response, error, expect=(200,))

    # -- fetch the dashboard -------------------------------------------------- #
    elapsed, response, error = _timed(
        'GET', f'{base_url}/api/dashboard/stats', headers=headers,
    )
    record('dashboard', elapsed, response, error, expect=(200,))

    # -- download the PDF -------------------------------------------------- #
    if report_id:
        elapsed, response, error = _timed(
            'GET', f'{base_url}/api/download-report/{report_id}', headers=headers,
        )
        record('report_download', elapsed, response, error, expect=(200,))
    else:
        samples.append(Sample(
            'report_download', 0.0, False,
            f'no report ready after {NFR1_POLL_TIMEOUT_SECONDS:.0f}s '
            f'(last status={status_value!r})',
            measured=False,
        ))

    return samples


def run_nfr1(base_url, user_count=NFR1_USER_COUNT):
    """Run the NFR-1 concurrency benchmark. Returns ``True`` if it passed."""
    print()
    print('=' * 78)
    print(f'NFR-1 -- {user_count} concurrent users, one full session each')
    print('=' * 78)

    run_token = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    all_samples = []
    with ThreadPoolExecutor(max_workers=user_count) as pool:
        futures = [pool.submit(_user_session, base_url, i, run_token)
                   for i in range(user_count)]
        for future in as_completed(futures):
            all_samples.extend(future.result())
    wall_clock = time.perf_counter() - started

    by_endpoint = {}
    for sample in all_samples:
        by_endpoint.setdefault(sample.endpoint, []).append(sample)

    header = f'{"endpoint":<18}{"n":>5}{"p50":>10}{"p95":>10}{"max":>10}{"errors":>9}'
    print(header)
    print('-' * len(header))

    read_p95_ok = True
    for endpoint in sorted(by_endpoint):
        rows = by_endpoint[endpoint]
        latencies = [s.elapsed for s in rows if s.measured]
        n_errors = sum(1 for s in rows if not s.ok)
        if latencies:
            p50, p95, worst = (_percentile(latencies, 50),
                               _percentile(latencies, 95), max(latencies))
            print(f'{endpoint:<18}{len(rows):>5}{p50:>9.3f}s{p95:>9.3f}s'
                  f'{worst:>9.3f}s{n_errors:>9}')
        else:
            print(f'{endpoint:<18}{len(rows):>5}{"n/a":>10}{"n/a":>10}'
                  f'{"n/a":>10}{n_errors:>9}')
        if endpoint in NFR1_READ_ENDPOINTS and latencies and p95 > NFR1_READ_P95_BUDGET_SECONDS:
            read_p95_ok = False

    errors = [s for s in all_samples if not s.ok]
    print('-' * len(header))
    print(f'total wall clock: {wall_clock:.2f}s for {user_count} concurrent sessions')
    if errors:
        print(f'{len(errors)} error(s):')
        for sample in errors[:20]:
            print(f'  [{sample.endpoint}] {sample.detail}')
        if len(errors) > 20:
            print(f'  ... and {len(errors) - 20} more')

    passed = read_p95_ok and not errors
    print(f'pass criterion: zero errors AND p95 <= {NFR1_READ_P95_BUDGET_SECONDS:.1f}s '
          f'for {", ".join(NFR1_READ_ENDPOINTS)}')
    print(f'NFR-1: {"PASS" if passed else "FAIL"}')
    return passed


# --------------------------------------------------------------------------- #
# NFR-2 — history query < 2s
# --------------------------------------------------------------------------- #

def _ensure_nfr2_seed_user():
    """Get-or-create the fixed NFR-2 seed account. Idempotent across runs."""
    User = get_user_model()
    user = User.objects.filter(email=NFR2_SEED_EMAIL).first()
    if user is not None:
        return user, False
    user = User.objects.create_user(
        email=NFR2_SEED_EMAIL, password=BENCHMARK_PASSWORD,
        full_name='NFR-2 Seed Account',
    )
    return user, True


def _ensure_media_pool(user, size=NFR2_MEDIA_POOL_SIZE):
    """Reuse an existing pool of uploads if one is already there."""
    import io

    existing = list(UploadedMedia.objects.filter(user=user).order_by('media_id')[:size])
    if len(existing) >= size:
        return existing

    pool = list(existing)
    for index in range(len(existing), size):
        buffer = io.BytesIO()
        Image.new('RGB', (48, 36), (index % 255, 90, 140)).save(buffer, format='JPEG')
        content = buffer.getvalue()
        media = UploadedMedia(
            user=user, filename=f'nfr2-seed-{index:04d}.jpg', format='jpg',
            media_type='image', size_bytes=len(content),
        )
        media.file.save(f'nfr2-seed-{index:04d}.jpg', ContentFile(content), save=False)
        media.save()
        pool.append(media)
    return pool


def _top_up_nfr2_rows(user, media_pool, target=NFR2_SEED_COUNT):
    """
    Bulk-ORM-create :class:`AnalysisResult` rows (with vehicles attached)
    for ``user`` until there are at least ``target`` of them. Idempotent:
    a re-run on an already-seeded database creates nothing.
    """
    existing = AnalysisResult.objects.filter(user=user).count()
    missing = target - existing
    if missing <= 0:
        return existing

    severities = ('low', 'moderate', 'high', '')
    vehicle_types = tuple(choice for choice, _label in VEHICLE_TYPE_CHOICES)
    now = timezone.now()

    rows = []
    for _ in range(missing):
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

    with transaction.atomic():
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

    return existing + missing


def run_nfr2(base_url):
    """Seed (idempotently) and time ``GET /api/history`` for real. ``True`` if it passed."""
    print()
    print('=' * 78)
    print(f'NFR-2 -- history query wall clock, >= {NFR2_SEED_COUNT} rows')
    print('=' * 78)

    user, created = _ensure_nfr2_seed_user()
    media_pool = _ensure_media_pool(user)
    total = _top_up_nfr2_rows(user, media_pool)
    print(f'seed account: {NFR2_SEED_EMAIL} ({"created" if created else "reused"}), '
          f'{total} analysis row(s) on record')

    elapsed, response, error = _login(base_url, NFR2_SEED_EMAIL, BENCHMARK_PASSWORD)
    if error or response is None or response.status_code != 200:
        print(f'NFR-2: FAIL -- could not log in as the seed account '
              f'({error or response.status_code})')
        return False
    headers = {'Authorization': f'Bearer {response.json()["access"]}'}

    url = f'{base_url}/api/history'
    thirty_days_ago = (timezone.now() - timedelta(days=30)).date().isoformat()
    today = timezone.now().date().isoformat()
    cases = [
        ('unfiltered', {'page_size': 50}),
        ('status=done', {'status': 'done', 'page_size': 50}),
        ('severity=high', {'severity': 'high', 'page_size': 50}),
        ('vehicle_type=truck', {'vehicle_type': 'truck', 'page_size': 50}),
        ('date range (30d)', {'date_from': thirty_days_ago, 'date_to': today,
                              'page_size': 50}),
        ('search', {'search': 'nfr2-seed-000', 'page_size': 50}),
    ]

    header = f'{"case":<20}{"elapsed":>12}{"budget":>10}{"result":>9}'
    print(header)
    print('-' * len(header))
    passed = True
    last_pages = None
    for label, params in cases:
        elapsed, response, error = _timed('GET', url, headers=headers, params=params)
        ok = error is None and response is not None and response.status_code == 200
        within = ok and elapsed <= NFR2_BUDGET_SECONDS
        passed = passed and within
        if ok and label == 'unfiltered':
            last_pages = response.json()['pages']
        note = '' if ok else f'  ({error or response.status_code})'
        print(f'{label:<20}{elapsed:>11.3f}s{NFR2_BUDGET_SECONDS:>9.1f}s'
              f'{"PASS" if within else "FAIL":>9}{note}')

    if last_pages:
        elapsed, response, error = _timed(
            'GET', url, headers=headers, params={'page_size': 50, 'page': last_pages},
        )
        ok = error is None and response is not None and response.status_code == 200
        within = ok and elapsed <= NFR2_BUDGET_SECONDS
        passed = passed and within
        label = f'last page ({last_pages})'
        print(f'{label:<20}{elapsed:>11.3f}s{NFR2_BUDGET_SECONDS:>9.1f}s'
              f'{"PASS" if within else "FAIL":>9}')
    else:
        passed = False
        print(f'{"last page":<20}{"n/a":>12}{NFR2_BUDGET_SECONDS:>9.1f}s{"FAIL":>9}'
              '  (could not determine page count)')

    print('-' * len(header))
    print(f'NFR-2: {"PASS" if passed else "FAIL"}')
    return passed


# --------------------------------------------------------------------------- #
# NFR-3 — upload acknowledgement < 3s
# --------------------------------------------------------------------------- #

def run_nfr3(base_url):
    """Register one throwaway user and time three upload sizes for real."""
    print()
    print('=' * 78)
    print('NFR-3 -- upload acknowledgement wall clock')
    print('=' * 78)

    email = f'nfr3-{uuid.uuid4().hex[:10]}@{BENCHMARK_EMAIL_DOMAIN}'
    elapsed, response, error = _timed(
        'POST', f'{base_url}/api/register',
        json={'email': email, 'password': BENCHMARK_PASSWORD,
              'full_name': 'NFR-3 Benchmark User'},
    )
    if error or response is None or response.status_code != 201:
        print(f'NFR-3: FAIL -- could not register a benchmark user '
              f'({error or response.status_code})')
        return False
    headers = {'Authorization': f'Bearer {response.json()["access"]}'}

    cases = [
        ('small image', _small_image_bytes(), 'jpg', 'image/jpeg'),
        ('~5 MB file', _padded_png_bytes(5 * ONE_MB), 'png', 'image/png'),
        ('~50 MB file', _padded_png_bytes(50 * ONE_MB), 'png', 'image/png'),
    ]

    header = f'{"case":<14}{"size":>12}{"elapsed":>12}{"budget":>10}{"result":>9}'
    print(header)
    print('-' * len(header))
    passed = True
    for label, data, extension, content_type in cases:
        files = {'file': (f'{label.replace(" ", "-")}.{extension}', data, content_type)}
        elapsed, response, error = _timed(
            'POST', f'{base_url}/api/upload', headers=headers, files=files,
            timeout=180,
        )
        ok = error is None and response is not None and response.status_code in (200, 201)
        within = ok and elapsed <= NFR3_BUDGET_SECONDS
        passed = passed and within
        note = '' if ok else f'  ({error or response.status_code})'
        print(f'{label:<14}{len(data) / ONE_MB:>10.2f}MB{elapsed:>11.3f}s'
              f'{NFR3_BUDGET_SECONDS:>9.1f}s{"PASS" if within else "FAIL":>9}{note}')

    print('-' * len(header))
    print(f'NFR-3: {"PASS" if passed else "FAIL"}')
    return passed


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #

def cleanup_benchmark_data():
    """
    Remove every account (and its cascaded rows and files) this script has
    ever created, identified solely by the reserved
    ``@nfr-benchmark.invalid`` domain. Never touches a real account.
    """
    User = get_user_model()
    users = list(User.objects.filter(email__iendswith=f'@{BENCHMARK_EMAIL_DOMAIN}'))
    if not users:
        print('Nothing to clean up -- no benchmark accounts found.')
        return

    analysis_ids = list(
        AnalysisResult.objects.filter(user__in=users)
        .values_list('analysis_id', flat=True)
    )
    report_ids = list(
        AnalysisResult.objects.filter(user__in=users, report__isnull=False)
        .values_list('report__report_id', flat=True)
    )

    for analysis_id in analysis_ids:
        delete_analysis_artifacts(analysis_id)
    for report_id in report_ids:
        report_path(report_id, create=False).unlink(missing_ok=True)

    for media in UploadedMedia.objects.filter(user__in=users):
        try:
            if media.file and media.file.storage.exists(media.file.name):
                media.file.delete(save=False)
        except Exception:                                 # noqa: BLE001
            logger.warning('Could not remove file for benchmark media %s',
                           media.media_id)

    deleted = len(users)
    User.objects.filter(pk__in=[user.pk for user in users]).delete()
    print(f'Removed {deleted} benchmark account(s), {len(analysis_ids)} '
          f'analysis row(s) and {len(report_ids)} report(s), plus their files.')


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv=None):
    """Parse arguments, run the three NFR benchmarks, print PASS/FAIL."""
    parser = argparse.ArgumentParser(
        description="AutoSmokeGuard NFR benchmark. Talks to an already-"
                    "running server; starts nothing itself.",
    )
    parser.add_argument('--host', default=DEFAULT_HOST,
                        help=f'Server host (default: {DEFAULT_HOST}).')
    parser.add_argument('--api-port', type=int, default=DEFAULT_PORT,
                        help=f'Server port (default: {DEFAULT_PORT}).')
    parser.add_argument('--users', type=int, default=NFR1_USER_COUNT,
                        help='Concurrent users for NFR-1 '
                             f'(default: {NFR1_USER_COUNT}, the SRS minimum).')
    parser.add_argument('--cleanup', action='store_true',
                        help='Remove every account this script has ever '
                             'created and exit, without running any benchmark.')
    args = parser.parse_args(argv)

    if args.cleanup:
        cleanup_benchmark_data()
        return 0

    base_url = f'http://{args.host}:{args.api_port}'
    print(f'AutoSmokeGuard NFR benchmark against {base_url}')
    if not _server_reachable(base_url):
        print(f'Could not reach {base_url}/api/health/. Start the server '
              f'first (e.g. `python manage.py runserver {args.api_port}`) '
              'and try again.')
        return 2

    results = {
        f'NFR-1 ({args.users} concurrent users)': run_nfr1(base_url, args.users),
        'NFR-2 (history query < 2s)': run_nfr2(base_url),
        'NFR-3 (upload ack < 3s)': run_nfr3(base_url),
    }

    print()
    print('=' * 78)
    print('Summary')
    print('=' * 78)
    for name, ok in results.items():
        print(f'  {name:<40} {"PASS" if ok else "FAIL"}')

    all_passed = all(results.values())
    print()
    print('OVERALL: ' + ('PASS' if all_passed else 'FAIL'))
    print()
    print('Run `python tools/benchmark.py --cleanup` to remove the accounts, '
          'uploads, analyses and reports this run created.')
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
