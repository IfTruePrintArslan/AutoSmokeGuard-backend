"""
Persistence confirmation: turning per-ROI responses into a media verdict.

Until the ``MIN_SMOKE_TRACK_*`` block existed, the media-level verdict was
``any(smoke_regions)`` -- one exhaust ROI in one sampled frame condemned the
whole clip.  With a per-ROI false-alarm rate of 12.8% on clean vehicles and
359 vehicle ROIs in a single clean-traffic clip, that rule reported **12 of
12** clean-traffic clips as smoking.  The replacement associates detections
into per-vehicle tracks and requires the evidence to persist along one.

Every test below is written to fail against the old rule.  The pipeline-level
ones assert verdicts that ``any(...)`` cannot produce; the unit-level ones
exercise the module that ``any(...)`` did not have.  ``test_revert_proof_*``
names mark the ones whose whole point is the A/B.

**No real machine learning.**  ``_FrameAnalyzer`` is replaced with a stub that
returns hand-written detections, and the frame iterator with a list of small
synthetic images, so these run in milliseconds and assert the *decision*, not
the segmenter.  The four sample-media verdicts are the pipeline's own
end-to-end check (``python -m mlcore.selftest``).
"""
import numpy as np
import pytest

from mlcore import pipeline as P
from mlcore.tracking import (
    TRACK_IOU_THRESHOLD,
    Track,
    VehicleTracker,
    iou,
    persistent_tracks,
    track_is_persistent,
)

# The shipped operating point, read live so a future retune moves the tests
# with the product instead of silently pinning an old number.
HITS = P.MIN_SMOKE_TRACK_HITS
RUN = P.MIN_SMOKE_TRACK_RUN
FRACTION = P.MIN_SMOKE_TRACK_FRACTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detection(x, y=100, w=120, h=90, smoking=False, intensity=0.42,
              severity='moderate', vehicle_type='car'):
    """One detector output, in the shape ``_FrameAnalyzer.process`` returns."""
    record = {
        'vehicle_type': vehicle_type,
        'bbox': {'x': x, 'y': y, 'w': w, 'h': h},
        'confidence': 0.9,
        'smoke': None,
    }
    if smoking:
        record['smoke'] = {
            'mask': np.ones((h, w), dtype=np.float32),
            'roi': {'x': x, 'y': y + h, 'w': w, 'h': 30},
            'intensity': intensity,
            'severity': severity,
            'confidence': 0.8,
            'area_ratio': 0.3,
            'opacity': 0.6,
            'features': {},
        }
    return record


class StubAnalyzer:
    """Stands in for :class:`mlcore.pipeline._FrameAnalyzer`.

    Hands back one scripted frame of detections per call, so a test can write
    the *temporal pattern* of the per-ROI decisions directly and assert what
    the aggregation makes of it.
    """

    def __init__(self, config=None):
        self.script = list(StubAnalyzer.script)
        self.calls = 0

        class _Segmenter:
            @staticmethod
            def resolved_mode():
                return 'unet'

        self.segmenter = _Segmenter()

    def process(self, frame_bgr):
        frame = self.script[self.calls] if self.calls < len(self.script) else []
        self.calls += 1
        return [dict(d) for d in frame], frame_bgr


def run_pipeline(monkeypatch, tmp_path, script, kind='video'):
    """Drive the real :func:`analyze_media` over a scripted detection stream."""
    StubAnalyzer.script = script
    monkeypatch.setattr(P, '_FrameAnalyzer', StubAnalyzer)

    frames = [
        (index, index * 0.2, np.full((240, 320, 3), 40, dtype=np.uint8))
        for index in range(len(script))
    ]
    if kind == 'video':
        monkeypatch.setattr(P, 'probe_video', lambda _path: {
            'width': 320, 'height': 240, 'fps': 15.0,
            'frame_count': len(script) * 5, 'duration': len(script) / 3.0,
        })
        monkeypatch.setattr(P, 'iter_frames',
                            lambda *a, **kw: iter(frames))
        source = tmp_path / 'clip.mp4'
    else:
        monkeypatch.setattr(P, 'load_image', lambda _path: frames[0][2])
        source = tmp_path / 'still.jpg'
    source.write_bytes(b'not really media; every decoder is stubbed out')

    return P.analyze_media(source, tmp_path / 'out')


def smoking_for(frames, hits, start=0, step=1, x=50):
    """A script of *frames* frames in which one vehicle fires *hits* times."""
    script = []
    for index in range(frames):
        firing = start <= index and (index - start) % step == 0 and \
            (index - start) // step < hits
        script.append([detection(x + index * 4, smoking=firing)])
    return script


# ---------------------------------------------------------------------------
# The verdict (pipeline level) -- these are the A/B tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('pattern, label', [
    ([0], 'a single isolated response'),
    ([0, 7], 'two responses seven frames apart'),
    ([0, 2, 4, 6, 8, 10], 'six responses, never two in a row'),
])
def test_revert_proof_scattered_responses_are_not_an_emission(
        monkeypatch, tmp_path, pattern, label):
    """
    Scattered per-ROI responses on one vehicle report **no** smoke.

    This is the defect, stated as a test.  The old ``any(smoke_regions)`` rule
    returns ``total_smoke == len(pattern)`` and a non-``none`` severity for
    every one of these; the persistence rule returns zero, because a plume
    that appears for one frame at a time is flicker, not exhaust.

    The third case matters most: six responses clear any pure *count* bar, so
    only the consecutive-run term rejects it.
    """
    script = [[detection(50 + index * 4, smoking=index in pattern)]
              for index in range(12)]
    result = run_pipeline(monkeypatch, tmp_path, script)

    assert result['total_smoke'] == 0, label
    assert result['overall_severity'] == 'none'
    assert result['severity_counts'] == {'low': 0, 'moderate': 0, 'high': 0}
    assert all(vehicle['smoke'] is None for vehicle in result['vehicles'])
    # The vehicles themselves are still reported -- suppressing the verdict
    # must not suppress the detections it was drawn from.
    assert result['total_vehicles'] == 12
    assert result['smoke_persistence']['discarded_regions'] == len(pattern)


def test_revert_proof_one_false_roi_among_many_clean_vehicles(
        monkeypatch, tmp_path):
    """
    The measured failure, in miniature: many vehicles, one spurious ROI.

    Twelve frames of eight clean vehicles each is 96 ROIs.  One of them fires.
    Under 1-of-N that clip is "SMOKING"; under the persistence rule it is not,
    and this is precisely the arithmetic that made 12 of 12 clean-traffic
    clips false-positive.
    """
    script = []
    for index in range(12):
        frame = [detection(20 + lane * 140 + index * 3, y=60 + lane * 10)
                 for lane in range(8)]
        if index == 5:
            frame[3] = detection(20 + 3 * 140 + index * 3, y=90, smoking=True)
        script.append(frame)

    result = run_pipeline(monkeypatch, tmp_path, script)

    assert result['total_vehicles'] == 96
    assert result['total_smoke'] == 0
    assert result['overall_severity'] == 'none'


def test_sustained_plume_on_one_vehicle_is_reported(monkeypatch, tmp_path):
    """A vehicle that fires on consecutive frames is still reported.

    The companion to the tests above: a rule that never fires is not a fix.
    """
    result = run_pipeline(monkeypatch, tmp_path, smoking_for(12, hits=8))

    assert result['total_smoke'] == 8
    assert result['overall_severity'] == 'moderate'
    assert result['smoke_persistence']['confirmed_vehicles'] == 1
    assert result['smoke_persistence']['discarded_regions'] == 0
    confirmed = result['smoke_persistence']['confirmed'][0]
    assert confirmed['hits'] == 8
    assert confirmed['longest_run'] == 8
    assert confirmed['observations'] == 12


def test_a_confirmed_clip_keeps_every_region_it_measured(
        monkeypatch, tmp_path):
    """
    Confirmation is a media-level gate, not a per-region filter.

    Once one vehicle has sustained a plume, the clip's remaining responses are
    kept as measurements rather than being second-guessed one by one.  This is
    load-bearing: ``sample_truck_smoking.mp4``'s 29 regions are spread over
    six tracked vehicles of which only one is persistent, and the selftest
    pins that total at 29.
    """
    script = smoking_for(12, hits=8)
    script[9].append(detection(300, y=30, smoking=True, severity='high',
                               intensity=0.8, vehicle_type='truck'))
    for index in (2, 3, 4, 5, 6, 7, 8, 10, 11):
        script[index].append(detection(300 + index * 60, y=30))

    result = run_pipeline(monkeypatch, tmp_path, script)

    assert result['total_smoke'] == 9
    assert result['severity_counts'] == {'low': 0, 'moderate': 8, 'high': 1}
    # Worst-case-present survives: the one-frame 'high' still sets the verdict.
    assert result['overall_severity'] == 'high'


def test_borderline_vehicle_just_under_the_bar(monkeypatch, tmp_path):
    """One hit short of confirmation is not confirmed; one more hit is.

    Pins the decision to the constants rather than to a lucky script.  Unlike
    the ``test_revert_proof_*`` tests this one is written relative to whatever
    the constants say, so it degenerates rather than failing if they are set
    back to 1-of-N -- ``test_shipped_operating_point_is_stricter_than_one_of_n``
    is what catches that.
    """
    under = run_pipeline(
        monkeypatch, tmp_path, smoking_for(HITS + 4, hits=HITS - 1))
    assert under['total_smoke'] == 0
    assert under['overall_severity'] == 'none'

    over = run_pipeline(
        monkeypatch, tmp_path, smoking_for(HITS + 4, hits=HITS))
    assert over['total_smoke'] == HITS
    assert over['overall_severity'] != 'none'


def test_revert_proof_intermittent_vehicle_fails_the_fraction_term(
        monkeypatch, tmp_path):
    """
    Long-visible vehicle, a short burst of smoke: rejected on fraction.

    This is the tyre-smoke shape -- the burnout clip in the evaluation corpus
    fires on 7 of the 35 frames its car is visible for (20%), below
    ``MIN_SMOKE_TRACK_FRACTION``.  Both the hit count and the consecutive run
    are satisfied here, so only the fraction term can reject it.
    """
    frames = int(round((HITS + 1) / FRACTION)) + 4
    script = smoking_for(frames, hits=HITS + 1)
    track = VehicleTracker()
    for index, frame in enumerate(script):
        track.update(index, frame)
    only = track.tracks[0]
    assert only.hits >= HITS and only.longest_run >= RUN, 'script is wrong'
    assert only.hit_fraction < FRACTION, 'script is wrong'

    result = run_pipeline(monkeypatch, tmp_path, script)
    assert result['total_smoke'] == 0
    assert result['overall_severity'] == 'none'


def test_two_separate_vehicles_do_not_pool_their_evidence(
        monkeypatch, tmp_path):
    """
    Evidence accumulates per vehicle, not per clip.

    Two vehicles at opposite ends of the frame fire alternately, so the clip
    has plenty of responses in total and neither vehicle has enough of its
    own.  A clip-level counting rule ("at least N regions anywhere") passes
    this; the per-vehicle rule does not, and that distinction is the whole
    reason the tracker exists.
    """
    script = []
    for index in range(12):
        script.append([
            detection(20, y=40, smoking=index % 2 == 0),
            detection(220, y=150, smoking=index % 2 == 1),
        ])
    result = run_pipeline(monkeypatch, tmp_path, script)

    assert result['smoke_persistence']['tracked_vehicles'] == 2
    assert result['smoke_persistence']['confirmed_vehicles'] == 0
    assert result['total_smoke'] == 0


# ---------------------------------------------------------------------------
# Degrading to a single frame
# ---------------------------------------------------------------------------

def test_still_image_keeps_the_per_roi_verdict(monkeypatch, tmp_path):
    """
    A still image has one frame, so there is no persistence evidence to test.

    The rule must degrade to the per-ROI decision rather than to "never".
    ``sample_bus_smoking.jpg`` is one vehicle in one frame and has to stay
    ``moderate``; if this ever inverts, the selftest goes from 4/4 to 3/4.
    """
    result = run_pipeline(
        monkeypatch, tmp_path, [[detection(60, smoking=True)]], kind='image')

    assert result['total_smoke'] == 1
    assert result['overall_severity'] == 'moderate'
    assert result['smoke_persistence']['applied'] is False
    assert result['media_meta']['kind'] == 'image'


def test_single_frame_video_also_degrades(monkeypatch, tmp_path):
    """A video that yielded one decodable frame is in the same position."""
    result = run_pipeline(monkeypatch, tmp_path, [[detection(60, smoking=True)]])

    assert result['frames_processed'] == 1
    assert result['total_smoke'] == 1
    assert result['smoke_persistence']['applied'] is False


def test_clean_media_is_untouched_by_the_rule(monkeypatch, tmp_path):
    """No responses in, no responses out, and nothing claimed to be discarded."""
    result = run_pipeline(
        monkeypatch, tmp_path, [[detection(50 + i * 4)] for i in range(10)])

    assert result['total_smoke'] == 0
    assert result['overall_severity'] == 'none'
    assert result['smoke_persistence']['discarded_regions'] == 0
    assert result['smoke_persistence']['confirmed_vehicles'] == 0


# ---------------------------------------------------------------------------
# Artifacts stay consistent with the verdict
# ---------------------------------------------------------------------------

def test_revert_proof_discarded_run_leaves_no_orphan_masks(
        monkeypatch, tmp_path):
    """
    A clip reported clean must not leave smoke masks on disk or in the record.

    Masks are written during the scan, before the verdict can be known, so
    discarding has to clean up after itself; otherwise a "no smoke" analysis
    ships mask PNGs and a preview with a plume burnt into it.
    """
    script = [[detection(50 + index * 4, smoking=index in (0, 4, 9))]
              for index in range(12)]
    result = run_pipeline(monkeypatch, tmp_path, script)

    assert result['total_smoke'] == 0
    assert not list((tmp_path / 'out' / 'masks').glob('*.png'))
    # Vehicle crops survive: a crop is a picture of a vehicle either way.
    assert list((tmp_path / 'out' / 'crops').glob('*.jpg'))
    assert result['preview_path']
    assert (tmp_path / 'out' / result['preview_path']).is_file()


def test_confirmed_run_still_writes_its_masks(monkeypatch, tmp_path):
    """The cleanup is conditional -- a confirmed run keeps its evidence."""
    result = run_pipeline(monkeypatch, tmp_path, smoking_for(12, hits=8))

    assert result['total_smoke'] == 8
    assert len(list((tmp_path / 'out' / 'masks').glob('*.png'))) == 8
    assert all(vehicle['smoke']['mask_path']
               for vehicle in result['vehicles'] if vehicle['smoke'])


def test_progress_contract_is_unchanged(monkeypatch, tmp_path):
    """Deferring annotation must not move the documented progress milestones."""
    StubAnalyzer.script = smoking_for(6, hits=5)
    monkeypatch.setattr(P, '_FrameAnalyzer', StubAnalyzer)
    monkeypatch.setattr(P, 'load_image',
                        lambda _p: np.full((240, 320, 3), 40, dtype=np.uint8))
    source = tmp_path / 'still.jpg'
    source.write_bytes(b'stub')

    seen = []
    P.analyze_media(source, tmp_path / 'out',
                    progress_cb=lambda pct, stage: seen.append(pct))

    for milestone in (5, 15, 40, 75, 90, 100):
        assert milestone in seen
    assert seen[-1] == 100


# ---------------------------------------------------------------------------
# The tracker and the rule, in isolation
# ---------------------------------------------------------------------------

def test_iou_of_identical_and_disjoint_boxes():
    box = {'x': 10, 'y': 10, 'w': 100, 'h': 100}
    assert iou(box, box) == pytest.approx(1.0)
    assert iou(box, {'x': 500, 'y': 500, 'w': 10, 'h': 10}) == 0.0
    # Malformed input is answered, not raised on.
    assert iou(box, {'x': 0, 'y': 0}) == 0.0
    assert iou(box, {'x': 0, 'y': 0, 'w': 0, 'h': 5}) == 0.0


def test_tracker_follows_one_vehicle_across_frames():
    tracker = VehicleTracker()
    for index in range(6):
        tracker.update(index * 5, [detection(40 + index * 8, smoking=True)])

    assert len(tracker.tracks) == 1
    only = tracker.tracks[0]
    assert only.observations == 6
    assert only.hits == 6
    assert only.longest_run == 6
    assert only.hit_fraction == pytest.approx(1.0)
    assert only.first_frame == 0 and only.last_frame == 25


def test_tracker_splits_vehicles_that_do_not_overlap():
    tracker = VehicleTracker()
    tracker.update(0, [detection(10), detection(400)])
    tracker.update(1, [detection(14), detection(404)])

    assert len(tracker.tracks) == 2
    assert all(track.observations == 2 for track in tracker.tracks)


def test_tracker_bridges_one_missed_frame_but_not_two():
    bridged = VehicleTracker()
    bridged.update(0, [detection(40, smoking=True)])
    bridged.update(1, [detection(900)])                  # vehicle not detected
    bridged.update(2, [detection(48, smoking=True)])
    assert len([t for t in bridged.tracks if t.hits]) == 1

    dropped = VehicleTracker()
    dropped.update(0, [detection(40, smoking=True)])
    dropped.update(1, [detection(900)])
    dropped.update(2, [detection(900)])
    dropped.update(3, [detection(48, smoking=True)])
    assert len([t for t in dropped.tracks if t.hits]) == 2


def test_a_gap_in_observation_breaks_the_run_but_not_the_track():
    """Frames the vehicle was not seen in are not evidence that it stopped."""
    tracker = VehicleTracker()
    tracker.update(0, [detection(40, smoking=True)])
    tracker.update(1, [detection(900)])
    tracker.update(2, [detection(48, smoking=True)])

    smoking = next(t for t in tracker.tracks if t.hits)
    assert smoking.observations == 2
    assert smoking.hits == 2
    # Observed in steps 0 and 2, so the two hits are not consecutive.
    assert smoking.longest_run == 1


def test_empty_frames_do_not_advance_the_tracker():
    tracker = VehicleTracker()
    tracker.update(0, [detection(40, smoking=True)])
    assert tracker.update(1, []) == []
    tracker.update(2, [detection(44, smoking=True)])

    only = tracker.tracks[0]
    assert only.observations == 2
    assert only.longest_run == 2, 'a frame with no vehicles is not a break'


def test_tracker_tolerates_a_detection_with_no_box():
    tracker = VehicleTracker()
    tracker.update(0, [{'vehicle_type': 'car', 'confidence': 0.5}])
    assert tracker.tracks[0].observations == 1, (
        'an unreadable box still counts in the denominator; dropping it would '
        'flatter the hit fraction'
    )


def test_each_term_of_the_rule_rejects_on_its_own():
    passes = Track(track_id=0, observations=HITS + 1, hits=HITS,
                   longest_run=RUN)
    assert track_is_persistent(passes, HITS, RUN, FRACTION)

    too_few = Track(track_id=1, observations=HITS + 1, hits=HITS - 1,
                    longest_run=RUN)
    assert not track_is_persistent(too_few, HITS, RUN, FRACTION)

    too_broken = Track(track_id=2, observations=HITS + 1, hits=HITS,
                       longest_run=RUN - 1)
    assert not track_is_persistent(too_broken, HITS, RUN, FRACTION)

    too_sparse = Track(track_id=3, observations=HITS * 10, hits=HITS,
                       longest_run=RUN)
    assert too_sparse.hit_fraction < FRACTION
    assert not track_is_persistent(too_sparse, HITS, RUN, FRACTION)

    assert [t.track_id for t in persistent_tracks(
        [passes, too_few, too_broken, too_sparse], HITS, RUN, FRACTION)] == [0]


def test_shipped_operating_point_is_stricter_than_one_of_n():
    """
    Guards the constants themselves.

    ``MIN_SMOKE_TRACK_HITS = 1`` with a zero run and zero fraction *is* the
    old rule; if a future retune slides the parameters back to that, every
    behavioural test above would still pass on the way past, so assert the
    bound here.
    """
    assert P.MIN_SMOKE_TRACK_HITS >= 2
    assert P.MIN_SMOKE_TRACK_RUN >= 2
    assert P.MIN_SMOKE_TRACK_FRACTION > 0.0
    assert 0.0 < TRACK_IOU_THRESHOLD < 1.0
