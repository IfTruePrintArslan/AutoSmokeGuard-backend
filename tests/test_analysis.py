"""
Tests for the analysis orchestration layer: the worker, the job endpoints,
history and the dashboard.

Two decisions make this suite fast and deterministic.

**No real machine learning.**  ``mlcore`` imports torch, ultralytics and
OpenCV and loads two sets of weights; doing that in a unit test would add
tens of seconds and make the suite depend on files that are not in git.  The
``fake_mlcore`` fixture installs a stand-in module into ``sys.modules``
*before* the worker's lazy ``from mlcore import ...`` ever runs, so the
pipeline boundary is exercised for real — same call signature, same result
dict, same relative-path contract, same progress callbacks — without a single
tensor.  The one test that does use the real pipeline is marked ``slow`` and
skips itself when ``sample_media/`` is empty.

**The worker runs inline.**  ``ASG['WORKER_ENABLED'] = False`` (equivalently
``ASG_WORKER_ENABLED=False`` in the environment) makes ``enqueue`` execute the
job on the calling thread, so a POST to ``/api/analyze`` has already finished
by the time it answers and there is nothing to sleep on.  ``test_tc14_*`` is
the deliberate exception: it turns the real thread pool back on to prove
concurrent submissions do not lose or corrupt jobs, and therefore needs
``transaction=True`` so worker threads (which have their own connections) can
see committed rows.
"""
import sys
import time
import types
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from django.core.files.base import ContentFile
from django.urls import reverse
from django.utils import timezone

from analysis import services, worker
from analysis.models import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    AnalysisResult,
    DetectedVehicle,
    SmokeRegion,
)
from common.storage import analysis_artifact_dir
from uploads.models import UploadedMedia

# ---------------------------------------------------------------------------
# The fake ML core
# ---------------------------------------------------------------------------

#: What the stand-in pipeline reports. Mirrors the real result dict exactly,
#: including the rule that every path is relative to ``output_dir``.
FAKE_VEHICLES = [
    {
        'vehicle_type': 'truck',
        'bounding_box': {'x': 412, 'y': 233, 'w': 180, 'h': 140},
        'confidence': 0.93,
        'frame_number': 140,
        'timestamp_seconds': 7.0,
        'crop_path': 'crops/v00000_f000140.jpg',
        'smoke': {
            'mask_path': 'masks/m00000_f000140.png',
            'intensity': 0.74,
            'severity': 'high',
            'confidence': 0.88,
            'area_ratio': 0.21,
            'opacity': 0.66,
        },
    },
    {
        'vehicle_type': 'car',
        'bounding_box': {'x': 10, 'y': 20, 'w': 30, 'h': 40},
        'confidence': 0.71,
        'frame_number': 12,
        'timestamp_seconds': 0.6,
        'crop_path': 'crops/v00001_f000012.jpg',
        'smoke': None,
    },
]


def fake_result():
    """A deterministic copy of the pipeline's result dict."""
    import copy

    return {
        'total_vehicles': 2,
        'total_smoke': 1,
        'frames_processed': 41,
        'avg_confidence': 0.82,
        'overall_severity': 'high',
        'severity_counts': {'low': 0, 'moderate': 0, 'high': 1},
        'mean_intensity': 0.74,
        'preview_path': 'preview.jpg',
        'annotated_frames': ['frames/frame_000140.jpg',
                             'frames/frame_000012.jpg'],
        'vehicles': copy.deepcopy(FAKE_VEHICLES),
        'segmenter_mode': 'unet',
        'device': 'cpu',
        'elapsed_seconds': 0.01,
        'media_meta': {'kind': 'video', 'width': 1280, 'height': 720},
    }


class FakeMLConfig:
    """
    Stand-in for :class:`mlcore.MLConfig`.

    Keeps whatever it was handed so a test can assert the *translation* from
    ``settings_snapshot`` to ML keys — the step where a typo would silently
    run every analysis at the library defaults.
    """

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, mapping):
        """Mirror the real ``from_dict``; unknown keys are simply kept."""
        return cls(**(mapping or {}))


class FakePipeline:
    """
    Records every call and writes the artefacts a real run would leave behind.

    The files matter: the delete test asserts the artefact tree is actually
    removed, which is meaningless if nothing was ever written.
    """

    def __init__(self, result_factory=fake_result, raises=None):
        self.result_factory = result_factory
        self.raises = raises
        self.calls = []
        self.progress = []

    def __call__(self, media_path, output_dir, config=None, progress_cb=None):
        self.calls.append({'media_path': media_path,
                           'output_dir': output_dir,
                           'config': config})
        if progress_cb is not None:
            for percent, stage in ((5, 'loading'), (40, 'segmenting'),
                                   (90, 'writing artifacts'), (100, 'done')):
                progress_cb(percent, stage)
                self.progress.append((percent, stage))

        if self.raises is not None:
            raise self.raises

        result = self.result_factory()
        root = Path(output_dir)
        for relative in filter(None, [result.get('preview_path'),
                                      *(result.get('annotated_frames') or [])]):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'fake-jpeg')
        for vehicle in result.get('vehicles') or []:
            for relative in filter(None, [vehicle.get('crop_path'),
                                          (vehicle.get('smoke') or {})
                                          .get('mask_path')]):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'fake-artifact')
        return result


@pytest.fixture
def fake_mlcore(monkeypatch):
    """
    Install a stand-in ``mlcore`` module for the duration of one test.

    ``monkeypatch.setitem(sys.modules, ...)`` rather than patching an
    attribute: the worker imports the package lazily *inside* the function
    that needs it (so Django can start without torch), which means there is no
    module attribute to patch until the job is already running.  Replacing the
    entry in ``sys.modules`` intercepts the import itself, and pytest puts the
    real module back afterwards.
    """
    pipeline = FakePipeline()
    module = types.ModuleType('mlcore')
    module.MLConfig = FakeMLConfig
    module.analyze_media = pipeline
    module.warmup = lambda config=None: {'device': 'cpu'}
    module.get_device = lambda prefer=None: 'cpu'
    module.MLPipelineError = type('MLPipelineError', (RuntimeError,), {})
    module.pipeline = pipeline          # test handle, not part of the real API
    monkeypatch.setitem(sys.modules, 'mlcore', module)
    return module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def inline_worker(settings):
    """
    Run every job on the calling thread.

    Replaces the whole ``ASG`` dict rather than mutating it in place so
    pytest-django restores the original cleanly; mutating a nested dict leaks
    between tests.
    """
    settings.ASG = {**settings.ASG, 'WORKER_ENABLED': False}
    return settings


@pytest.fixture
def media_factory(db, sample_files):
    """Create a real ``UploadedMedia`` row with real bytes under MEDIA_ROOT."""
    def _make(user, filename='clip.mp4', media_type='video', extension='mp4'):
        content = sample_files[extension]
        media = UploadedMedia(
            user=user,
            filename=filename,
            format=extension,
            media_type=media_type,
            size_bytes=len(content),
            width=1280,
            height=720,
            duration_seconds=12.5 if media_type == 'video' else None,
        )
        media.file.save(filename, ContentFile(content), save=False)
        media.save()
        return media

    return _make


@pytest.fixture
def media(auth_client, media_factory):
    """One upload belonging to ``auth_client.user``."""
    return media_factory(auth_client.user)


def make_analysis(user, media, **overrides):
    """Seed an ``AnalysisResult`` straight into the database."""
    fields = {
        'media': media,
        'user': user,
        'status': STATUS_DONE,
        'progress': 100,
        'total_vehicles': 3,
        'total_smoke': 1,
        'frames_processed': 20,
        'avg_confidence': 0.8,
        'overall_severity': 'moderate',
        'severity_counts': {'low': 0, 'moderate': 1, 'high': 0},
        'settings_snapshot': {'confidence_threshold': 0.35},
        'start_time': timezone.now() - timedelta(seconds=8),
        'end_time': timezone.now(),
    }
    fields.update(overrides)
    created_at = fields.pop('created_at', None)
    analysis = AnalysisResult.objects.create(**fields)
    if created_at is not None:
        # ``created_at`` is auto_now_add, so it can only be back-dated with an
        # UPDATE that bypasses the field's pre_save.
        AnalysisResult.objects.filter(pk=analysis.pk).update(
            created_at=created_at)
        analysis.refresh_from_db()
    return analysis


def analyze(client, media, **settings_overrides):
    """
    POST /api/analyze and return the response.

    ``auto_generate_pdf`` defaults to ``False`` here even though the system
    default is ``True``: the reports app renders a real PDF, which takes
    several seconds per run and would dominate this suite's runtime while
    testing somebody else's code. The one test that cares passes ``True``
    explicitly.
    """
    payload = {'auto_generate_pdf': False}
    payload.update(settings_overrides)
    body = {'media_id': str(media.media_id), 'settings': payload}
    return client.post(reverse('analysis-analyze'), body, format='json')


# ---------------------------------------------------------------------------
# TC-08 — starting a run
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_tc08_analyze_accepts_and_queues(auth_client, media, monkeypatch,
                                         fake_mlcore):
    """A valid media_id returns 202 with a job id and leaves a queued row."""
    # Stop the inline worker short so the row is observable in `queued`; the
    # full happy path is covered by the pipeline tests below.
    monkeypatch.setattr(worker, 'run_analysis', lambda analysis_id: None)

    response = analyze(auth_client, media)

    assert response.status_code == 202
    body = response.json()
    assert set(body) == {'job_id', 'analysis_id', 'status', 'message'}
    assert body['job_id'] == body['analysis_id']
    assert body['status'] == STATUS_QUEUED

    analysis = AnalysisResult.objects.get(pk=body['analysis_id'])
    assert analysis.status == STATUS_QUEUED
    assert analysis.user_id == auth_client.user.pk
    assert analysis.media_id == media.media_id
    # The configuration was frozen before any work started.
    assert analysis.settings_snapshot['confidence_threshold'] == pytest.approx(
        0.35)


@pytest.mark.django_db
def test_analyze_runs_the_pipeline_and_persists_everything(
        auth_client, media, fake_mlcore):
    """The inline worker takes a queued row all the way to a stored result."""
    response = analyze(auth_client, media)
    assert response.status_code == 202

    analysis = AnalysisResult.objects.get(pk=response.json()['analysis_id'])
    assert analysis.status == STATUS_DONE
    assert analysis.progress == 100
    assert analysis.start_time is not None and analysis.end_time is not None
    assert analysis.total_vehicles == 2
    assert analysis.total_smoke == 1
    assert analysis.frames_processed == 41
    assert analysis.overall_severity == 'high'
    assert analysis.severity_counts == {'low': 0, 'moderate': 0, 'high': 1}

    vehicles = list(analysis.vehicles.all())
    assert len(vehicles) == 2
    assert {v.vehicle_type for v in vehicles} == {'truck', 'car'}
    assert SmokeRegion.objects.filter(vehicle__analysis=analysis).count() == 1

    # Every ML-relative path was rebased onto MEDIA_ROOT.
    assert analysis.preview_path == f'analyses/{analysis.analysis_id}/preview.jpg'
    truck = analysis.vehicles.get(vehicle_type='truck')
    assert truck.crop_path.startswith(f'analyses/{analysis.analysis_id}/crops/')
    smoke = truck.smoke_regions.get()
    assert smoke.mask_path.startswith(f'analyses/{analysis.analysis_id}/masks/')


@pytest.mark.django_db
def test_analyze_translates_snapshot_keys_for_mlcore(auth_client, media,
                                                     fake_mlcore):
    """
    The snapshot -> MLConfig mapping is explicit, not a pass-through.

    ``MLConfig.from_dict`` drops keys it does not recognise, so handing it
    ``confidence_threshold`` verbatim would silently run at the library
    default. This locks the rename in.
    """
    analyze(auth_client, media, confidence_threshold=0.5, sensitivity=80)

    config = fake_mlcore.pipeline.calls[0]['config']
    assert config.kwargs['conf_threshold'] == pytest.approx(0.5)
    assert config.kwargs['mask_threshold'] == pytest.approx(0.2)
    assert 'confidence_threshold' not in config.kwargs
    assert 'smoke_mask_threshold' not in config.kwargs


@pytest.mark.django_db
def test_analyze_rejects_foreign_media(auth_client, user_factory,
                                       media_factory, fake_mlcore):
    """Someone else's media_id is indistinguishable from a missing one."""
    stranger_media = media_factory(user_factory())

    response = analyze(auth_client, stranger_media)

    assert response.status_code == 404
    assert response.json()['code'] == 'media_not_found'
    assert not AnalysisResult.objects.exists()


@pytest.mark.django_db
def test_analyze_unknown_media(auth_client, fake_mlcore):
    """An id that belongs to nobody gets the same 404."""
    response = auth_client.post(
        reverse('analysis-analyze'), {'media_id': str(uuid.uuid4())},
        format='json',
    )
    assert response.status_code == 404
    assert response.json()['code'] == 'media_not_found'


@pytest.mark.django_db
def test_analyze_requires_media_id(auth_client):
    """A malformed body fails with the standard validation envelope."""
    response = auth_client.post(reverse('analysis-analyze'), {}, format='json')

    assert response.status_code == 400
    body = response.json()
    assert body['code'] == 'validation_error'
    assert 'media_id' in body['errors']


@pytest.mark.django_db
def test_analyze_requires_authentication(api, media):
    """No token, no analysis."""
    response = api.post(reverse('analysis-analyze'),
                        {'media_id': str(media.media_id)}, format='json')
    assert response.status_code == 401


@pytest.mark.django_db
def test_reanalyzing_media_in_flight_returns_409_with_the_existing_id(
        auth_client, media, fake_mlcore):
    """A second submission for the same upload points at the first run."""
    existing = make_analysis(auth_client.user, media, status=STATUS_RUNNING,
                             progress=37, end_time=None)

    response = analyze(auth_client, media)

    assert response.status_code == 409
    body = response.json()
    assert body['code'] == 'analysis_in_progress'
    assert body['analysis_id'] == str(existing.analysis_id)
    assert body['job_id'] == str(existing.analysis_id)
    assert body['status'] == STATUS_RUNNING
    # Nothing new was created.
    assert AnalysisResult.objects.count() == 1


@pytest.mark.django_db
def test_client_settings_are_clamped_not_rejected(auth_client, media,
                                                  fake_mlcore):
    """Nonsense from the client is corrected into range and recorded."""
    response = analyze(auth_client, media,
                       confidence_threshold=99, frame_sample_rate=0,
                       sensitivity=500)

    assert response.status_code == 202
    analysis = AnalysisResult.objects.get(pk=response.json()['analysis_id'])
    snapshot = analysis.settings_snapshot

    assert snapshot['confidence_threshold'] == pytest.approx(
        services.CONFIDENCE_MAX)
    assert snapshot['frame_sample_rate'] == services.FRAME_SAMPLE_RATE_MIN
    assert snapshot['sensitivity'] == services.SENSITIVITY_MAX
    assert (services.MASK_THRESHOLD_MIN
            <= snapshot['smoke_mask_threshold']
            <= services.MASK_THRESHOLD_MAX)


@pytest.mark.django_db
def test_unknown_model_name_cannot_escape_the_assets_directory(
        auth_client, media, fake_mlcore):
    """A path-shaped ``model`` override is ignored, not resolved."""
    response = analyze(auth_client, media, model='../../etc/passwd')

    assert response.status_code == 202
    snapshot = AnalysisResult.objects.get(
        pk=response.json()['analysis_id']).settings_snapshot
    assert 'yolo_weights' not in snapshot
    assert 'model' not in snapshot


# ---------------------------------------------------------------------------
# TC-09 — polling
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_tc09_status_reports_progress_then_completion(auth_client, media,
                                                      fake_mlcore, monkeypatch):
    """A running job reports progress; a finished one reports done / 100."""
    real_run_analysis = worker.run_analysis
    monkeypatch.setattr(worker, 'run_analysis', lambda analysis_id: None)
    job_id = analyze(auth_client, media).json()['job_id']

    AnalysisResult.objects.filter(pk=job_id).update(
        status=STATUS_RUNNING, progress=62, stage='segmenting frames',
        start_time=timezone.now(),
    )

    running = auth_client.get(reverse('analysis-status', args=[job_id]))
    assert running.status_code == 200
    body = running.json()
    assert body['job_id'] == job_id
    assert body['analysis_id'] == job_id
    assert body['status'] == STATUS_RUNNING
    assert body['progress'] == 62
    assert body['stage'] == 'segmenting frames'
    assert body['started_at'] is not None
    assert body['ended_at'] is None
    assert body['report_id'] is None
    assert set(body) == {
        'job_id', 'analysis_id', 'status', 'progress', 'stage', 'started_at',
        'ended_at', 'error_message', 'report_id', 'total_vehicles',
        'total_smoke', 'overall_severity',
    }

    # Now let the run finish for real (the real function, not the stub above).
    AnalysisResult.objects.filter(pk=job_id).update(status=STATUS_QUEUED)
    real_run_analysis(job_id)

    finished = auth_client.get(
        reverse('analysis-status', args=[job_id])).json()
    assert finished['status'] == STATUS_DONE
    assert finished['progress'] == 100
    assert finished['ended_at'] is not None
    assert finished['total_vehicles'] == 2
    assert finished['total_smoke'] == 1
    assert finished['overall_severity'] == 'high'
    assert finished['error_message'] == ''


@pytest.mark.django_db
def test_status_of_a_foreign_job_is_404(auth_client, user_factory,
                                        media_factory):
    """Polling somebody else's job leaks nothing."""
    stranger = user_factory()
    stranger_analysis = make_analysis(stranger, media_factory(stranger))

    response = auth_client.get(
        reverse('analysis-status', args=[stranger_analysis.analysis_id]))

    assert response.status_code == 404
    assert response.json()['code'] == 'analysis_not_found'


@pytest.mark.django_db
def test_status_of_an_unknown_job_is_404(auth_client):
    """An id nobody owns behaves the same way."""
    response = auth_client.get(
        reverse('analysis-status', args=[uuid.uuid4()]))
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# TC-14 — concurrency
# ---------------------------------------------------------------------------

@pytest.mark.django_db(transaction=True)
def test_tc14_concurrent_submissions_all_reach_a_terminal_state(
        settings, auth_client, media_factory, fake_mlcore):
    """
    Five jobs, two threads, nothing lost and nothing corrupted.

    Needs ``transaction=True``: the pool threads open their own database
    connections and can only see committed rows, which the default
    test-in-a-transaction isolation would hide from them.
    """
    settings.ASG = {**settings.ASG, 'WORKER_ENABLED': True,
                    'WORKER_THREADS': 2}
    worker.job_queue.reset()

    try:
        uploads = [media_factory(auth_client.user, filename=f'clip-{i}.mp4')
                   for i in range(5)]
        job_ids = []
        for upload in uploads:
            response = analyze(auth_client, upload)
            assert response.status_code == 202
            job_ids.append(response.json()['job_id'])

        assert len(set(job_ids)) == 5
        assert worker.job_queue.wait(timeout=60), 'worker pool did not drain'

        rows = AnalysisResult.objects.filter(pk__in=job_ids)
        assert rows.count() == 5
        for row in rows:
            assert row.status == STATUS_DONE, row.error_message
            assert row.progress == 100
            assert row.total_vehicles == 2
            assert row.vehicles.count() == 2
            assert row.end_time is not None
        assert DetectedVehicle.objects.filter(
            analysis__in=rows).count() == 10
    finally:
        # Always tear the pool down: a leaked executor would carry the
        # two-thread configuration (and its connections) into later tests.
        worker.job_queue.reset()


@pytest.mark.django_db(transaction=True)
def test_tc14_concurrent_submissions_all_produce_their_reports(
        settings, auth_client, media_factory, fake_mlcore):
    """
    The same five jobs, but with auto-PDF on — the shipped default.

    This is the regression test for the write-contention defect.  Everywhere
    else in this module ``analyze()`` turns ``auto_generate_pdf`` off to keep
    the suite fast, which means nothing else here exercises the one thing the
    worker does *after* an analysis commits: a second, synchronous write from
    the same pool thread, racing every other thread's writes.  With SQLite in
    its stock rollback-journal mode that reliably lost PDFs — and lost them
    *silently*, because ``_maybe_generate_report`` swallows every exception by
    design so a bad render cannot fail a good analysis.  So the symptom was
    never a red test, just a missing report.

    Five analyses in, five reports out.  Nothing less passes.
    """
    from reports.models import GeneratedReport   # noqa: PLC0415 - another app

    settings.ASG = {**settings.ASG, 'WORKER_ENABLED': True,
                    'WORKER_THREADS': 2}
    worker.job_queue.reset()

    try:
        job_ids = []
        for index in range(5):
            upload = media_factory(auth_client.user,
                                   filename=f'reported-{index}.mp4')
            response = analyze(auth_client, upload, auto_generate_pdf=True)
            assert response.status_code == 202
            job_ids.append(response.json()['job_id'])

        assert worker.job_queue.wait(timeout=60), 'worker pool did not drain'

        rows = AnalysisResult.objects.filter(pk__in=job_ids)
        for row in rows:
            assert row.status == STATUS_DONE, row.error_message

        reports = GeneratedReport.objects.filter(analysis__in=rows)
        assert reports.count() == 5, (
            f'{reports.count()}/5 reports survived five concurrent runs — '
            'the report write is losing races for the database lock again; '
            "check DATABASES['default']['OPTIONS'] in config/settings.py"
        )
        for report in reports:
            assert report.page_count >= 1
            assert report.file_size_bytes > 0
    finally:
        worker.job_queue.reset()


@pytest.mark.django_db(transaction=True)
def test_inference_never_runs_on_two_threads_at_once(
        settings, auth_client, media_factory, fake_mlcore):
    """
    The forward pass is serialised process-wide, and it must stay that way.

    ``mlcore`` shares one YOLO model and one U-Net across the whole process.
    Driving them from two threads does not just race — on Apple silicon it
    trips a Metal assertion that ``abort()``s the interpreter, killing every
    in-flight request. This reproduces the overlap opportunity (four jobs,
    four threads, a pipeline slow enough to collide) and asserts it never
    happens.
    """
    import threading

    state = {'live': 0, 'peak': 0}
    guard = threading.Lock()
    inner = FakePipeline()

    def counting_pipeline(*args, **kwargs):
        with guard:
            state['live'] += 1
            state['peak'] = max(state['peak'], state['live'])
        try:
            time.sleep(0.05)
            return inner(*args, **kwargs)
        finally:
            with guard:
                state['live'] -= 1

    fake_mlcore.analyze_media = counting_pipeline
    settings.ASG = {**settings.ASG, 'WORKER_ENABLED': True,
                    'WORKER_THREADS': 4}
    worker.job_queue.reset()

    try:
        for index in range(4):
            upload = media_factory(auth_client.user,
                                   filename=f'parallel-{index}.mp4')
            assert analyze(auth_client, upload).status_code == 202
        assert worker.job_queue.wait(timeout=60)

        assert state['peak'] == 1, (
            f'{state["peak"]} concurrent inference calls — the process-wide '
            'inference lock is gone'
        )
        assert AnalysisResult.objects.filter(status=STATUS_DONE).count() == 4
    finally:
        worker.job_queue.reset()


@pytest.mark.django_db
def test_double_enqueue_runs_the_job_once(auth_client, media, fake_mlcore):
    """
    The compare-and-set claim makes ``run_analysis`` idempotent.

    A retry, a duplicated submission or a queue replay must not produce two
    sets of detections for one analysis.
    """
    analysis = make_analysis(auth_client.user, media, status=STATUS_QUEUED,
                             progress=0, total_vehicles=0, total_smoke=0,
                             start_time=None, end_time=None)

    worker.run_analysis(analysis.analysis_id)
    worker.run_analysis(analysis.analysis_id)      # already 'done' — declined

    assert len(fake_mlcore.pipeline.calls) == 1
    assert DetectedVehicle.objects.filter(analysis=analysis).count() == 2


# ---------------------------------------------------------------------------
# TC-15 — failure handling
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_tc15_pipeline_failure_lands_in_failed_with_a_safe_message(
        auth_client, media, fake_mlcore, tmp_path, caplog):
    """A crashing pipeline fails the row without leaking internals."""
    boom = RuntimeError(
        f'CUDA kernel exploded at {tmp_path}/secret/module.py line 42'
    )
    fake_mlcore.analyze_media = FakePipeline(raises=boom)

    response = analyze(auth_client, media)
    assert response.status_code == 202

    analysis = AnalysisResult.objects.get(pk=response.json()['analysis_id'])
    assert analysis.status == STATUS_FAILED
    assert analysis.end_time is not None
    assert analysis.error_message == worker.DEFAULT_ERROR_MESSAGE

    message = analysis.error_message
    assert 'Traceback' not in message
    assert 'CUDA' not in message
    assert str(tmp_path) not in message
    assert '.py' not in message

    # The endpoint still answers with a clean, contract-shaped envelope.
    status_body = auth_client.get(
        reverse('analysis-status', args=[analysis.analysis_id])).json()
    assert status_body['status'] == STATUS_FAILED
    assert status_body['error_message'] == worker.DEFAULT_ERROR_MESSAGE
    assert status_body['progress'] >= 0

    detail = auth_client.get(
        reverse('analysis-detail', args=[analysis.analysis_id]))
    assert detail.status_code == 200
    assert detail.json()['error_message'] == worker.DEFAULT_ERROR_MESSAGE


@pytest.mark.django_db
def test_known_failures_get_an_actionable_message(auth_client, media,
                                                  fake_mlcore):
    """A missing source file explains itself instead of saying 'unexpected'."""
    fake_mlcore.analyze_media = FakePipeline(
        raises=FileNotFoundError('/private/var/media/uploads/x.mp4'))

    response = analyze(auth_client, media)
    analysis = AnalysisResult.objects.get(pk=response.json()['analysis_id'])

    assert analysis.status == STATUS_FAILED
    assert 'no longer available' in analysis.error_message
    assert '/private' not in analysis.error_message


@pytest.mark.django_db
def test_report_failure_never_fails_the_analysis(auth_client, media,
                                                 fake_mlcore, monkeypatch):
    """auto_generate_pdf is best-effort; the measurement is the product."""
    def explode(analysis):
        raise RuntimeError('reportlab fell over')

    fake_reports = types.ModuleType('reports.services')
    fake_reports.generate_report_for_analysis = explode
    monkeypatch.setitem(sys.modules, 'reports.services', fake_reports)

    response = analyze(auth_client, media, auto_generate_pdf=True)

    analysis = AnalysisResult.objects.get(pk=response.json()['analysis_id'])
    assert analysis.status == STATUS_DONE
    assert analysis.error_message == ''


# ---------------------------------------------------------------------------
# Severity re-banding
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_severity_is_rebanded_against_the_configured_thresholds(
        auth_client, media, fake_mlcore, settings_row, caplog):
    """
    Per-region severity (and therefore the distribution) follows the admin's
    live bands, not the thresholds the pipeline happened to run with.

    The fake pipeline always claims ``high`` at intensity 0.74. With the low
    band widened past that, the same measurement's *region* and *counts* must
    be stored as ``low`` — but ``overall_severity`` is the single source of
    truth taken straight from the pipeline's own verdict (see
    ``analysis.worker._persist``), so it still reads ``high`` here even
    though it now disagrees with the live-rebanded distribution.  That is the
    documented, logged edge case: an admin changed the bands after the
    pipeline had already classified this run, and it must be logged rather
    than silently resolved.
    """
    settings_row.severity_low_max = 0.9
    settings_row.severity_moderate_max = 0.95
    settings_row.save()

    with caplog.at_level('WARNING', logger='asg.worker'):
        response = analyze(auth_client, media)
    analysis = AnalysisResult.objects.get(pk=response.json()['analysis_id'])

    region = SmokeRegion.objects.get(vehicle__analysis=analysis)
    assert region.intensity == pytest.approx(0.74)
    assert region.severity == 'low'
    assert analysis.severity_counts == {'low': 1, 'moderate': 0, 'high': 0}
    assert analysis.overall_severity == 'high'
    assert 'disagrees with the live-rebanded' in caplog.text


@pytest.mark.django_db
def test_no_smoke_means_no_overall_severity(auth_client, media, fake_mlcore):
    """A clean run reads as 'nothing found', not as 'low'."""
    def clean():
        result = fake_result()
        result['vehicles'] = [dict(result['vehicles'][1])]
        result['total_smoke'] = 0
        result['overall_severity'] = 'none'
        return result

    fake_mlcore.analyze_media = FakePipeline(result_factory=clean)

    response = analyze(auth_client, media)
    analysis = AnalysisResult.objects.get(pk=response.json()['analysis_id'])

    assert analysis.overall_severity == ''
    assert analysis.total_smoke == 0

    body = auth_client.get(
        reverse('analysis-detail', args=[analysis.analysis_id])).json()
    assert body['overall_severity'] is None


# ---------------------------------------------------------------------------
# Progress throttling
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_progress_writes_are_throttled_but_never_drop_the_last_one(
        auth_client, media):
    """
    A 300-frame video must not mean 300 UPDATEs.

    The reporter is exercised directly: a burst of intermediate percentages
    collapses to one write, while the terminal 100 always lands.
    """
    analysis = make_analysis(auth_client.user, media, status=STATUS_RUNNING,
                             progress=0, end_time=None)
    reporter = worker._ProgressReporter(analysis.analysis_id,
                                        min_interval=60.0)

    for percent in range(1, 80):
        reporter(percent, 'segmenting')
    assert reporter.writes == 1

    reporter(100, 'done')
    assert reporter.writes == 2

    analysis.refresh_from_db()
    assert analysis.progress == 100
    assert analysis.stage == 'done'


@pytest.mark.django_db
def test_progress_cannot_resurrect_a_failed_run(auth_client, media):
    """A late callback must not overwrite a terminal state."""
    analysis = make_analysis(auth_client.user, media, status=STATUS_FAILED,
                             progress=0, stage='failed')
    reporter = worker._ProgressReporter(analysis.analysis_id, min_interval=0.0)

    reporter(55, 'segmenting')

    analysis.refresh_from_db()
    assert analysis.progress == 0
    assert analysis.stage == 'failed'


# ---------------------------------------------------------------------------
# Restart recovery
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_restart_recovery_fails_orphaned_jobs(auth_client, media_factory):
    """Rows with no thread behind them are resolved, not left spinning."""
    user = auth_client.user
    running = make_analysis(user, media_factory(user), status=STATUS_RUNNING,
                            end_time=None)
    queued = make_analysis(user, media_factory(user), status=STATUS_QUEUED,
                           end_time=None)
    finished = make_analysis(user, media_factory(user), status=STATUS_DONE)

    assert worker.recover_interrupted_jobs() == 2

    running.refresh_from_db()
    queued.refresh_from_db()
    finished.refresh_from_db()
    assert running.status == STATUS_FAILED
    assert running.error_message == worker.INTERRUPTED_MESSAGE
    assert running.end_time is not None
    assert queued.status == STATUS_FAILED
    assert finished.status == STATUS_DONE


def test_bootstrap_is_suppressed_under_pytest():
    """The boot thread must never fire inside the test suite."""
    assert worker.should_bootstrap() is False
    assert worker.start_bootstrap() is None


# ---------------------------------------------------------------------------
# Detail endpoint
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_detail_matches_the_contract_shape(auth_client, media, settings_row,
                                           fake_mlcore):
    """
    Every documented key is present and every path is a URL.

    ``auto_generate_pdf`` is switched off at the *source* — the SystemSetting
    row — as well as in the request ``analyze()`` sends.  ``report: null``
    below is a statement about the serializer shape (the key exists and is
    nullable), not a race on whether the reports app has finished yet: with
    the inline worker this analysis is already ``done`` when the GET runs, so
    leaving auto-PDF on would make the assertion true or false depending on
    how fast reportlab was.  Pinning the setting makes "no report yet" the
    only possible outcome instead of the likely one.
    """
    settings_row.auto_generate_pdf = False
    settings_row.save(update_fields=['auto_generate_pdf'])

    job_id = analyze(auth_client, media).json()['analysis_id']

    response = auth_client.get(reverse('analysis-detail', args=[job_id]))
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        'analysis_id', 'media', 'status', 'progress', 'created_at',
        'start_time', 'end_time', 'duration_seconds', 'total_vehicles',
        'total_smoke', 'avg_confidence', 'overall_severity',
        'severity_counts', 'preview_url', 'report', 'settings_snapshot',
        'frames_processed', 'error_message', 'annotated_frames', 'vehicles',
        'segmenter_mode', 'device',
    }
    assert set(body['media']) == {'media_id', 'filename', 'media_type', 'url'}
    assert body['media']['url'].startswith('/media/uploads/')
    assert body['preview_url'] == f'/media/analyses/{job_id}/preview.jpg'
    assert body['annotated_frames']
    assert all(url.startswith(f'/media/analyses/{job_id}/frames/')
               for url in body['annotated_frames'])
    assert body['report'] is None
    assert body['segmenter_mode'] == 'unet'
    assert body['device'] == 'cpu'
    assert body['duration_seconds'] is not None
    assert set(body['severity_counts']) == {'low', 'moderate', 'high'}

    # The runtime stash is an implementation detail and must not leak out.
    assert services.RUNTIME_KEY not in body['settings_snapshot']
    assert 'confidence_threshold' in body['settings_snapshot']

    assert len(body['vehicles']) == 2
    truck = next(v for v in body['vehicles'] if v['vehicle_type'] == 'truck')
    assert set(truck) == {'vehicle_id', 'vehicle_type', 'bounding_box',
                          'confidence', 'frame_number', 'timestamp_seconds',
                          'crop_path', 'smoke'}
    assert truck['bounding_box'] == {'x': 412, 'y': 233, 'w': 180, 'h': 140}
    assert truck['crop_path'].startswith(f'/media/analyses/{job_id}/crops/')
    assert set(truck['smoke']) == {'smoke_id', 'mask_path', 'intensity',
                                   'severity', 'confidence', 'area_ratio',
                                   'opacity'}
    assert truck['smoke']['mask_path'].startswith(
        f'/media/analyses/{job_id}/masks/')
    assert truck['smoke']['severity'] == 'high'

    car = next(v for v in body['vehicles'] if v['vehicle_type'] == 'car')
    assert car['smoke'] is None


@pytest.mark.django_db
def test_detail_of_a_foreign_analysis_is_404(auth_client, user_factory,
                                             media_factory):
    """Ownership is enforced by the queryset, so a stranger sees a 404."""
    stranger = user_factory()
    analysis = make_analysis(stranger, media_factory(stranger))

    response = auth_client.get(
        reverse('analysis-detail', args=[analysis.analysis_id]))
    assert response.status_code == 404
    assert response.json()['code'] == 'not_found'


@pytest.mark.django_db
def test_admin_may_read_any_single_analysis(admin_client, user_factory,
                                            media_factory):
    """Administrators can support a user without impersonating them."""
    owner = user_factory()
    analysis = make_analysis(owner, media_factory(owner))

    response = admin_client.get(
        reverse('analysis-detail', args=[analysis.analysis_id]))
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_deleting_a_running_analysis_is_refused(auth_client, media):
    """Delete is not cancel; a live job keeps its row."""
    analysis = make_analysis(auth_client.user, media, status=STATUS_RUNNING,
                             end_time=None)

    response = auth_client.delete(
        reverse('analysis-detail', args=[analysis.analysis_id]))

    assert response.status_code == 409
    assert response.json()['code'] == 'analysis_in_progress'
    assert AnalysisResult.objects.filter(pk=analysis.analysis_id).exists()


@pytest.mark.django_db
def test_deleting_a_finished_analysis_removes_its_artifacts(
        auth_client, media, fake_mlcore):
    """204, the row is gone, and so is the entire artefact tree."""
    job_id = analyze(auth_client, media).json()['analysis_id']
    artifact_dir = analysis_artifact_dir(job_id, create=False)
    assert artifact_dir.exists()
    assert (artifact_dir / 'preview.jpg').exists()

    response = auth_client.delete(reverse('analysis-detail', args=[job_id]))

    assert response.status_code == 204
    assert not AnalysisResult.objects.filter(pk=job_id).exists()
    assert not DetectedVehicle.objects.exists()
    assert not SmokeRegion.objects.exists()
    assert not artifact_dir.exists()


@pytest.mark.django_db
def test_deleting_a_foreign_analysis_is_404(auth_client, user_factory,
                                            media_factory):
    """You cannot delete what you cannot see."""
    stranger = user_factory()
    analysis = make_analysis(stranger, media_factory(stranger))

    response = auth_client.delete(
        reverse('analysis-detail', args=[analysis.analysis_id]))

    assert response.status_code == 404
    assert AnalysisResult.objects.filter(pk=analysis.analysis_id).exists()


# ---------------------------------------------------------------------------
# Report delegation
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_report_for_an_unfinished_analysis_is_409(auth_client, media):
    """A PDF of a half-run analysis would be a lie."""
    analysis = make_analysis(auth_client.user, media, status=STATUS_RUNNING,
                             end_time=None)

    response = auth_client.post(
        reverse('analysis-report', args=[analysis.analysis_id]))

    assert response.status_code == 409
    assert response.json()['code'] == 'report_not_ready'


@pytest.mark.django_db
def test_report_delegates_to_the_reports_service(auth_client, media,
                                                 monkeypatch):
    """The 201 body is the frozen ReportObj, built from the returned row."""
    analysis = make_analysis(auth_client.user, media, status=STATUS_DONE)

    class StubReport:
        report_id = uuid.uuid4()
        analysis_id = analysis.analysis_id
        generated_at = timezone.now()
        page_count = 4
        file_size_bytes = 284113

    monkeypatch.setattr(services, 'generate_report',
                        lambda _analysis: StubReport())

    response = auth_client.post(
        reverse('analysis-report', args=[analysis.analysis_id]))

    assert response.status_code == 201
    body = response.json()
    assert body['report_id'] == str(StubReport.report_id)
    assert body['analysis_id'] == str(analysis.analysis_id)
    assert body['page_count'] == 4
    assert body['download_url'] == (
        f'/api/download-report/{StubReport.report_id}')


@pytest.mark.django_db
def test_report_returns_503_when_the_reports_app_is_missing(auth_client, media,
                                                            monkeypatch):
    """A not-yet-landed sibling degrades cleanly instead of 500-ing."""
    analysis = make_analysis(auth_client.user, media, status=STATUS_DONE)

    def unavailable(_analysis):
        raise services.ReportsUnavailable('no module named reports.services')

    monkeypatch.setattr(services, 'generate_report', unavailable)

    response = auth_client.post(
        reverse('analysis-report', args=[analysis.analysis_id]))

    assert response.status_code == 503
    assert response.json()['code'] == 'reports_unavailable'


# ---------------------------------------------------------------------------
# Listing and history
# ---------------------------------------------------------------------------

@pytest.fixture
def history_rows(auth_client, user_factory, media_factory):
    """
    A small, deliberately varied corpus for the filter tests.

    Each row differs from the others in exactly the dimensions the filters
    slice on, and a stranger's row is included so every assertion also proves
    the owner scoping holds.
    """
    user = auth_client.user
    today = timezone.now()

    truck_run = make_analysis(
        user, media_factory(user, filename='highway-truck.mp4'),
        overall_severity='high', total_vehicles=9, avg_confidence=0.95,
        created_at=today - timedelta(days=1),
    )
    DetectedVehicle.objects.create(
        analysis=truck_run, vehicle_type='truck',
        bounding_box={'x': 0, 'y': 0, 'w': 5, 'h': 5}, confidence=0.9,
        frame_number=1,
    )

    bus_run = make_analysis(
        user, media_factory(user, filename='depot-bus.mp4'),
        overall_severity='low', total_vehicles=2, avg_confidence=0.55,
        created_at=today - timedelta(days=10),
    )
    DetectedVehicle.objects.create(
        analysis=bus_run, vehicle_type='bus',
        bounding_box={'x': 0, 'y': 0, 'w': 5, 'h': 5}, confidence=0.6,
        frame_number=2,
    )
    # A second detection of the same type must not duplicate the row.
    DetectedVehicle.objects.create(
        analysis=bus_run, vehicle_type='bus',
        bounding_box={'x': 1, 'y': 1, 'w': 5, 'h': 5}, confidence=0.7,
        frame_number=3,
    )

    failed_run = make_analysis(
        user, media_factory(user, filename='corrupt-clip.avi',
                            extension='avi'),
        status=STATUS_FAILED, overall_severity='', total_vehicles=0,
        avg_confidence=0.0, error_message='could not decode',
        created_at=today - timedelta(days=40),
    )

    stranger = user_factory()
    stranger_run = make_analysis(
        stranger, media_factory(stranger, filename='highway-truck.mp4'),
        overall_severity='high',
    )

    return {'truck': truck_run, 'bus': bus_run, 'failed': failed_run,
            'stranger': stranger_run}


def ids_in(response):
    """Analysis ids from a paginated list response, in order."""
    return [row['analysis_id'] for row in response.json()['results']]


@pytest.mark.django_db
def test_history_lists_only_your_own_rows_newest_first(auth_client,
                                                       history_rows):
    """The default ordering is most-recent-first, scoped to the caller."""
    response = auth_client.get(reverse('analysis-history'))

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {'count', 'page', 'pages', 'page_size', 'next',
                         'previous', 'results'}
    assert body['count'] == 3
    assert ids_in(response) == [
        str(history_rows['truck'].analysis_id),
        str(history_rows['bus'].analysis_id),
        str(history_rows['failed'].analysis_id),
    ]
    assert str(history_rows['stranger'].analysis_id) not in ids_in(response)


@pytest.mark.django_db
def test_history_row_matches_the_contract_shape(auth_client, history_rows):
    """A row carries its nested media object and a null report."""
    row = auth_client.get(reverse('analysis-history')).json()['results'][0]

    assert set(row) == {
        'analysis_id', 'media', 'status', 'progress', 'created_at',
        'start_time', 'end_time', 'duration_seconds', 'total_vehicles',
        'total_smoke', 'avg_confidence', 'overall_severity',
        'severity_counts', 'preview_url', 'report',
    }
    assert set(row['media']) == {'media_id', 'filename', 'media_type', 'url'}
    assert row['report'] is None


@pytest.mark.django_db
def test_history_filters_by_severity(auth_client, history_rows):
    """severity slices on the run's overall verdict."""
    high = auth_client.get(reverse('analysis-history'), {'severity': 'high'})
    assert ids_in(high) == [str(history_rows['truck'].analysis_id)]

    low = auth_client.get(reverse('analysis-history'), {'severity': 'low'})
    assert ids_in(low) == [str(history_rows['bus'].analysis_id)]

    moderate = auth_client.get(reverse('analysis-history'),
                               {'severity': 'moderate'})
    assert moderate.json()['count'] == 0


@pytest.mark.django_db
def test_history_filters_by_vehicle_type_without_duplicating_rows(
        auth_client, history_rows):
    """
    vehicle_type keeps runs containing that type — exactly once.

    The bus run has two bus detections; a naive join would return it twice.
    """
    buses = auth_client.get(reverse('analysis-history'),
                            {'vehicle_type': 'bus'})
    assert buses.json()['count'] == 1
    assert ids_in(buses) == [str(history_rows['bus'].analysis_id)]

    trucks = auth_client.get(reverse('analysis-history'),
                             {'vehicle_type': 'truck'})
    assert ids_in(trucks) == [str(history_rows['truck'].analysis_id)]

    cars = auth_client.get(reverse('analysis-history'),
                           {'vehicle_type': 'car'})
    assert cars.json()['count'] == 0


@pytest.mark.django_db
def test_history_filters_by_status(auth_client, history_rows):
    """status slices on the job state."""
    failed = auth_client.get(reverse('analysis-history'), {'status': 'failed'})
    assert ids_in(failed) == [str(history_rows['failed'].analysis_id)]

    done = auth_client.get(reverse('analysis-history'), {'status': 'done'})
    assert done.json()['count'] == 2


@pytest.mark.django_db
def test_history_filters_by_inclusive_date_range(auth_client, history_rows):
    """date_from / date_to bracket created_at on whole days, inclusively."""
    today = timezone.localdate()

    recent = auth_client.get(
        reverse('analysis-history'),
        {'date_from': (today - timedelta(days=5)).isoformat()},
    )
    assert ids_in(recent) == [str(history_rows['truck'].analysis_id)]

    old = auth_client.get(
        reverse('analysis-history'),
        {'date_to': (today - timedelta(days=30)).isoformat()},
    )
    assert ids_in(old) == [str(history_rows['failed'].analysis_id)]

    middle = auth_client.get(
        reverse('analysis-history'),
        {'date_from': (today - timedelta(days=20)).isoformat(),
         'date_to': (today - timedelta(days=2)).isoformat()},
    )
    assert ids_in(middle) == [str(history_rows['bus'].analysis_id)]

    # The bounds really are inclusive: the bus run is exactly ten days old.
    edge = auth_client.get(
        reverse('analysis-history'),
        {'date_from': (today - timedelta(days=10)).isoformat(),
         'date_to': (today - timedelta(days=10)).isoformat()},
    )
    assert ids_in(edge) == [str(history_rows['bus'].analysis_id)]


@pytest.mark.django_db
def test_history_searches_the_filename(auth_client, history_rows):
    """search is a case-insensitive substring of the source filename."""
    hits = auth_client.get(reverse('analysis-history'), {'search': 'TRUCK'})
    assert ids_in(hits) == [str(history_rows['truck'].analysis_id)]

    misses = auth_client.get(reverse('analysis-history'),
                             {'search': 'nothing-like-this'})
    assert misses.json()['count'] == 0


@pytest.mark.django_db
def test_history_ordering_is_whitelisted(auth_client, history_rows):
    """Known sort keys work; an unknown one is a 400, not a silent default."""
    by_vehicles = auth_client.get(reverse('analysis-history'),
                                  {'ordering': '-total_vehicles'})
    assert ids_in(by_vehicles)[0] == str(history_rows['truck'].analysis_id)

    by_confidence = auth_client.get(reverse('analysis-history'),
                                    {'ordering': 'avg_confidence'})
    assert ids_in(by_confidence)[0] == str(history_rows['failed'].analysis_id)

    oldest_first = auth_client.get(reverse('analysis-history'),
                                   {'ordering': 'created_at'})
    assert ids_in(oldest_first)[0] == str(history_rows['failed'].analysis_id)

    rejected = auth_client.get(reverse('analysis-history'),
                               {'ordering': 'error_message'})
    assert rejected.status_code == 400


@pytest.mark.django_db
def test_history_paginates(auth_client, history_rows):
    """The list envelope reports page numbers, not just links."""
    first = auth_client.get(reverse('analysis-history'), {'page_size': 2})
    body = first.json()

    assert body['count'] == 3
    assert body['pages'] == 2
    assert body['page'] == 1
    assert body['page_size'] == 2
    assert len(body['results']) == 2
    assert body['next'] is not None
    assert body['previous'] is None

    second = auth_client.get(reverse('analysis-history'),
                             {'page_size': 2, 'page': 2}).json()
    assert second['page'] == 2
    assert len(second['results']) == 1
    assert second['next'] is None


@pytest.mark.django_db
def test_history_has_no_n_plus_one(auth_client, history_rows,
                                   django_assert_num_queries):
    """
    The query count must not grow with the number of rows.

    Four: authenticate the bearer token, count for the paginator, fetch the
    page (media and report joined in), and SQLite's savepoint bookkeeping.
    If this assertion ever fails after adding a field to the row serializer,
    the fix is another ``select_related`` — not a bigger number.
    """
    with django_assert_num_queries(3):
        response = auth_client.get(reverse('analysis-history'),
                                   {'page_size': 50})
    assert response.json()['count'] == 3


@pytest.mark.django_db
def test_analysis_list_endpoint_mirrors_history(auth_client, history_rows):
    """/api/analysis is the same rows with the same status filter."""
    response = auth_client.get(reverse('analysis-list'), {'status': 'done'})

    assert response.status_code == 200
    assert response.json()['count'] == 2


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_KEYS = {'totals', 'sparklines', 'detections_over_time',
                  'severity_distribution', 'recent_analyses',
                  'processing_queue'}


@pytest.mark.django_db
def test_dashboard_is_all_zeros_for_a_new_account(auth_client):
    """A brand-new account gets honest zeros, never fabricated demo data."""
    response = auth_client.get(reverse('dashboard-stats'))

    assert response.status_code == 200
    body = response.json()
    assert set(body) == DASHBOARD_KEYS

    assert body['totals'] == {
        'analyses': 0, 'high_severity': 0, 'reports': 0, 'avg_confidence': 0.0,
        'analyses_delta_pct': 0.0, 'high_delta_pct': 0.0,
        'reports_delta_pct': 0.0, 'confidence_delta_pct': 0.0,
    }
    assert body['recent_analyses'] == []
    assert body['processing_queue'] == []

    for series in body['sparklines'].values():
        assert len(series) == 14
        assert set(series) == {0} or set(series) == {0.0}

    assert len(body['detections_over_time']) == 30
    assert all(point['detections'] == 0 and point['high'] == 0
               for point in body['detections_over_time'])

    # The donut is still described, just empty.
    assert [slice_['key'] for slice_ in body['severity_distribution']] == [
        'low', 'moderate', 'high']
    assert all(slice_['value'] == 0 and slice_['pct'] == 0
               for slice_ in body['severity_distribution'])
    assert [slice_['color'] for slice_ in body['severity_distribution']] == [
        '#4ade80', '#fbbf24', '#f87171']


@pytest.mark.django_db
def test_dashboard_totals_series_and_percentages(auth_client, media_factory):
    """Seeded rows produce correct totals, a dense series and pcts of 100."""
    user = auth_client.user
    now = timezone.now()

    for _ in range(3):
        make_analysis(user, media_factory(user), overall_severity='high',
                      avg_confidence=0.9, created_at=now - timedelta(days=1))
    for _ in range(2):
        make_analysis(user, media_factory(user), overall_severity='moderate',
                      avg_confidence=0.6, created_at=now - timedelta(days=3))
    make_analysis(user, media_factory(user), overall_severity='low',
                  avg_confidence=0.3, created_at=now - timedelta(days=5))
    # Outside the 30-day window — must not be counted.
    make_analysis(user, media_factory(user), overall_severity='high',
                  created_at=now - timedelta(days=45))

    body = auth_client.get(reverse('dashboard-stats')).json()

    assert body['totals']['analyses'] == 6
    assert body['totals']['high_severity'] == 3
    assert body['totals']['reports'] == 0
    assert body['totals']['avg_confidence'] == pytest.approx(
        (0.9 * 3 + 0.6 * 2 + 0.3) / 6, abs=1e-4)

    series = body['detections_over_time']
    assert len(series) == 30
    assert [point['date'] for point in series] == sorted(
        point['date'] for point in series)
    by_date = {point['date']: point for point in series}
    yesterday = (timezone.localdate() - timedelta(days=1)).isoformat()
    assert by_date[yesterday]['detections'] == 3
    assert by_date[yesterday]['high'] == 3
    # Zero-filled, not skipped.
    quiet = (timezone.localdate() - timedelta(days=2)).isoformat()
    assert by_date[quiet]['detections'] == 0
    assert by_date[quiet]['label']

    pcts = {slice_['key']: slice_['pct']
            for slice_ in body['severity_distribution']}
    values = {slice_['key']: slice_['value']
              for slice_ in body['severity_distribution']}
    assert values == {'low': 1, 'moderate': 2, 'high': 3}
    assert sum(pcts.values()) == 100

    assert len(body['recent_analyses']) == 5
    assert set(body['recent_analyses'][0]) >= {'analysis_id', 'media',
                                               'status', 'severity_counts'}


@pytest.mark.django_db
def test_dashboard_percentages_sum_to_100_on_an_awkward_split(auth_client,
                                                              media_factory):
    """One of each: 33/33/33 rounds to 99 without largest-remainder."""
    user = auth_client.user
    for severity in ('low', 'moderate', 'high'):
        make_analysis(user, media_factory(user), overall_severity=severity)

    body = auth_client.get(reverse('dashboard-stats')).json()
    assert sum(slice_['pct']
               for slice_ in body['severity_distribution']) == 100


@pytest.mark.django_db
def test_dashboard_deltas_compare_with_the_preceding_window(auth_client,
                                                            media_factory):
    """The delta is versus the equal-length window immediately before."""
    user = auth_client.user
    now = timezone.now()

    for _ in range(4):
        make_analysis(user, media_factory(user),
                      created_at=now - timedelta(days=2))
    for _ in range(2):
        make_analysis(user, media_factory(user),
                      created_at=now - timedelta(days=12))

    body = auth_client.get(reverse('dashboard-stats'),
                           {'days': 7}).json()

    assert body['totals']['analyses'] == 4
    assert body['totals']['analyses_delta_pct'] == pytest.approx(100.0)
    assert len(body['detections_over_time']) == 7
    # Sparklines stay 14 points wide regardless of the window.
    assert len(body['sparklines']['analyses']) == 14


@pytest.mark.django_db
def test_dashboard_days_parameter_is_clamped(auth_client):
    """Garbage and out-of-range windows fall back instead of failing."""
    default = auth_client.get(reverse('dashboard-stats'), {'days': 'abc'})
    assert len(default.json()['detections_over_time']) == 30

    huge = auth_client.get(reverse('dashboard-stats'), {'days': 100000})
    assert len(huge.json()['detections_over_time']) == services.DASHBOARD_DAYS_MAX

    tiny = auth_client.get(reverse('dashboard-stats'), {'days': -5})
    assert len(tiny.json()['detections_over_time']) == services.DASHBOARD_DAYS_MIN


@pytest.mark.django_db
def test_dashboard_processing_queue_lists_live_jobs(auth_client,
                                                    media_factory):
    """Anything not yet terminal shows up in the queue strip."""
    user = auth_client.user
    running = make_analysis(user, media_factory(user, filename='live.mp4'),
                            status=STATUS_RUNNING, progress=42,
                            stage='segmenting frames', end_time=None)
    make_analysis(user, media_factory(user), status=STATUS_DONE)

    body = auth_client.get(reverse('dashboard-stats')).json()

    assert len(body['processing_queue']) == 1
    item = body['processing_queue'][0]
    assert item == {
        'analysis_id': str(running.analysis_id),
        'filename': 'live.mp4',
        'status': STATUS_RUNNING,
        'progress': 42,
        'stage': 'segmenting frames',
    }


@pytest.mark.django_db
def test_dashboard_ignores_other_users(auth_client, user_factory,
                                       media_factory):
    """"My dashboard" means mine, for administrators too."""
    stranger = user_factory()
    for _ in range(4):
        make_analysis(stranger, media_factory(stranger),
                      overall_severity='high')

    body = auth_client.get(reverse('dashboard-stats')).json()
    assert body['totals']['analyses'] == 0
    assert body['recent_analyses'] == []


@pytest.mark.django_db
def test_dashboard_requires_authentication(api):
    """No token, no numbers."""
    assert api.get(reverse('dashboard-stats')).status_code == 401


# ---------------------------------------------------------------------------
# The real pipeline
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.django_db
def test_real_pipeline_end_to_end(auth_client, settings):
    """
    Run an actual sample file through the actual models.

    Skipped unless ``backend/sample_media/`` contains something and both sets
    of weights are on disk, so a clean checkout without the ML assets still
    goes green. Run it with ``pytest -m slow``.
    """
    sample_dir = Path(settings.ASG['SAMPLE_MEDIA_DIR'])
    candidates = sorted(
        path for path in (sample_dir.glob('*') if sample_dir.is_dir() else [])
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.mp4', '.mov',
                                   '.avi'}
    )
    if not candidates:
        pytest.skip('no files in sample_media/')
    if not Path(settings.ASG['YOLO_WEIGHTS']).is_file():
        pytest.skip('YOLO weights are not present')

    source = candidates[0]
    extension = source.suffix.lstrip('.').lower()
    media = UploadedMedia(
        user=auth_client.user,
        filename=source.name,
        format=extension,
        media_type='image' if extension in {'jpg', 'jpeg', 'png'} else 'video',
        size_bytes=source.stat().st_size,
    )
    media.file.save(source.name, ContentFile(source.read_bytes()), save=False)
    media.save()

    response = analyze(auth_client, media)
    assert response.status_code == 202

    analysis = AnalysisResult.objects.get(pk=response.json()['analysis_id'])
    assert analysis.status == STATUS_DONE, analysis.error_message
    assert analysis.progress == 100
    assert analysis.frames_processed >= 1
    runtime = analysis.settings_snapshot[services.RUNTIME_KEY]
    assert runtime['device']
    assert runtime['segmenter_mode'] in {'unet', 'classical'}
