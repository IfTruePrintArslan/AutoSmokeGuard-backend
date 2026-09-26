"""Associating per-frame vehicle detections into tracks, and deciding which of
those tracks is *persistently* emitting.

Why this module exists
----------------------

The segmenter decides, independently, for every vehicle in every sampled
frame, whether that one exhaust ROI looks like smoke.  Until this module was
written the media-level verdict was ``any(...)`` over those decisions: one ROI
anywhere in the clip condemned the whole clip.

That is arithmetically hopeless.  A single clean-traffic clip in the evaluation
corpus yields 359 vehicle ROIs, and the shipped checkpoint's per-ROI false
alarm rate on clean vehicles is 12.8% at matched scale (35.3% on distant
traffic).  Even a 1% per-ROI error gives ``1 - 0.99**359 = 97%`` clip-level
false alarms.  Measured: **12 of 12** clean-traffic clips were reported
smoking, one of them "high" on clear daylight highway footage.  The
aggregation, not the model, was the binding constraint.

What replaces it
----------------

Exhaust is emitted *continuously*.  A vehicle that is genuinely smoking fires
on many of the frames in which it is visible, and fires on consecutive frames.
A false positive -- a shadow under a bumper, a dark tarmac patch, a wet
reflection -- fires on scattered single frames as the vehicle moves through
it.  So: associate detections into per-vehicle tracks, and require the
evidence to *persist along a track* before it counts.

Deliberately not a real tracker
-------------------------------

Greedy IoU association between adjacent sampled frames, one missed frame of
tolerance, no motion model, no appearance model, no Kalman filter, no
re-identification.  It runs on frames that are already 5 apart by default
(``MLConfig.frame_sample_rate``), where a motion model would be predicting
across a third of a second of unobserved movement, and its only job is to
answer "is this the same vehicle as the one in the previous sampled frame".
Fragmenting one vehicle into two tracks is a *conservative* error here: it
shortens runs and can only make the verdict stricter, never looser.

The measured cost of getting association wrong is small: sweeping the IoU
threshold from 0.15 to 0.5 and the gap tolerance from 0 to 2 moves clip-level
recall by one clip (16/22 -> 15/22) and clip-level false positives not at all
(18/43 throughout).  A better tracker is not where the remaining error is.

What this module does **not** do
--------------------------------

It does not re-decide whether an ROI contains smoke.  Every per-ROI detection
floor (``MIN_SMOKE_AREA_RATIO``, ``MIN_SMOKE_BLOB_RATIO``,
``MIN_SMOKE_BLOB_PIXELS``, ``MIN_MEASURABLE_ROI_SIDE``, ``MIN_SMOKE_DENSITY``)
lives in :mod:`mlcore.pipeline` and is untouched.  This module only combines
their outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "TRACK_IOU_THRESHOLD",
    "TRACK_MAX_MISSED_FRAMES",
    "Track",
    "VehicleTracker",
    "iou",
    "track_is_persistent",
]

#: Minimum box overlap for two detections in adjacent sampled frames to be
#: called the same vehicle.  0.3 is loose on purpose: at
#: ``frame_sample_rate=5`` a vehicle moves a long way between samples, and the
#: failure mode of a *tight* threshold (one vehicle split into several short
#: tracks) is the one that silently costs recall.
TRACK_IOU_THRESHOLD = 0.3

#: How many intervening frames a track may go unmatched and still be picked up
#: again.  One: a vehicle briefly occluded, or missed by the detector for a
#: single frame, is still the same vehicle.  Beyond that it starts a new track.
TRACK_MAX_MISSED_FRAMES = 1


def iou(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Intersection-over-union of two ``{'x','y','w','h'}`` pixel boxes.

    Returns ``0.0`` for disjoint, degenerate or unreadable boxes rather than
    raising -- a malformed detection must not abort a run.
    """
    try:
        ax, ay = float(a["x"]), float(a["y"])
        aw, ah = float(a["w"]), float(a["h"])
        bx, by = float(b["x"]), float(b["y"])
        bw, bh = float(b["w"]), float(b["h"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


@dataclass
class Track:
    """One vehicle followed across sampled frames, and its smoke history.

    Only counters are retained.  Nothing here holds a frame, a crop or a mask,
    so a 300-frame clip with a hundred vehicles costs a few kilobytes.
    """

    track_id: int
    vehicle_type: str = "vehicle"
    #: Number of sampled frames this vehicle was observed in.
    observations: int = 0
    #: Of those, how many had an ROI that cleared every detection floor.
    hits: int = 0
    #: Longest unbroken sequence of *consecutively observed* frames that fired.
    longest_run: int = 0
    #: Strongest per-ROI intensity seen on this vehicle, whether it fired or not.
    peak_intensity: float = 0.0
    first_frame: int | None = None
    last_frame: int | None = None

    # -- association bookkeeping (not part of the verdict) ------------------ #
    _last_box: dict[str, Any] = field(default_factory=dict, repr=False)
    _last_step: int = -1
    _current_run: int = 0

    @property
    def hit_fraction(self) -> float:
        """Share of this vehicle's observations that fired, in ``[0, 1]``."""
        return (self.hits / self.observations) if self.observations else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Plain-data summary, safe to log or serialise."""
        return {
            "track_id": self.track_id,
            "vehicle_type": self.vehicle_type,
            "observations": self.observations,
            "hits": self.hits,
            "longest_run": self.longest_run,
            "hit_fraction": round(self.hit_fraction, 6),
            "peak_intensity": round(float(self.peak_intensity), 6),
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
        }


class VehicleTracker:
    """Greedy IoU association of detections into :class:`Track` objects.

    Feed it one frame at a time in capture order via :meth:`update`; read the
    accumulated evidence from :attr:`tracks` once the scan is over.

    Frames that contain **no** detections do not advance the tracker's internal
    step counter.  "Consecutive" therefore means "consecutive frames in which
    *some* vehicle was seen", which is the unit the persistence rule was
    measured in and the only one that is meaningful: a stretch of empty road
    between two sightings of the same lorry is not evidence that it stopped
    smoking.
    """

    __slots__ = ("iou_threshold", "max_missed", "_tracks", "_step", "_next_id")

    def __init__(
        self,
        iou_threshold: float = TRACK_IOU_THRESHOLD,
        max_missed: int = TRACK_MAX_MISSED_FRAMES,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self._tracks: list[Track] = []
        self._step = -1
        self._next_id = 0

    @property
    def tracks(self) -> list[Track]:
        """Every track seen so far, in creation order."""
        return list(self._tracks)

    def update(
        self,
        frame_index: int,
        detections: Sequence[Mapping[str, Any]],
    ) -> list[int]:
        """Associate one frame's detections with the running tracks.

        Args:
            frame_index: Source frame number, recorded on the track for
                reporting.  Ordering comes from call order, not from this
                value, so a caller that samples irregularly is still correct.
            detections: Mappings with a ``bbox``/``bounding_box`` box and an
                optional truthy ``smoke`` mapping.  A detection whose box
                cannot be read is still counted as an observation -- dropping
                it would flatter the hit fraction.

        Returns:
            One track id per detection, in the same order.
        """
        if not detections:
            return []
        self._step += 1
        step = self._step

        boxes = [self._box_of(d) for d in detections]

        # Only tracks seen recently enough are eligible; everything older has
        # left the scene (or become a different vehicle in the same place).
        live = [t for t in self._tracks if step - t._last_step <= self.max_missed + 1]

        candidates: list[tuple[float, int, int]] = []
        for di, box in enumerate(boxes):
            if box is None:
                continue
            for ti, track in enumerate(live):
                overlap = iou(box, track._last_box)
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, di, ti))
        # Best overlap first; each detection and each track may be used once.
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        assigned: dict[int, Track] = {}
        taken_tracks: set[int] = set()
        for _overlap, di, ti in candidates:
            if di in assigned or ti in taken_tracks:
                continue
            assigned[di] = live[ti]
            taken_tracks.add(ti)

        ids: list[int] = []
        for di, detection in enumerate(detections):
            track = assigned.get(di)
            if track is None:
                track = Track(track_id=self._next_id)
                self._next_id += 1
                self._tracks.append(track)
            self._observe(track, detection, boxes[di], step, int(frame_index))
            ids.append(track.track_id)
        return ids

    def _observe(
        self,
        track: Track,
        detection: Mapping[str, Any],
        box: dict[str, Any] | None,
        step: int,
        frame_index: int,
    ) -> None:
        """Fold one detection into *track*'s counters."""
        contiguous = track._last_step == step - 1
        smoke = detection.get("smoke")
        fired = isinstance(smoke, Mapping)

        track.observations += 1
        if track.first_frame is None:
            track.first_frame = frame_index
        track.last_frame = frame_index
        vehicle_type = detection.get("vehicle_type")
        if vehicle_type:
            track.vehicle_type = str(vehicle_type)

        if fired:
            track.hits += 1
            track._current_run = track._current_run + 1 if contiguous else 1
            track.longest_run = max(track.longest_run, track._current_run)
            try:
                intensity = float(smoke.get("intensity") or 0.0)
            except (TypeError, ValueError):
                intensity = 0.0
            track.peak_intensity = max(track.peak_intensity, intensity)
        else:
            track._current_run = 0

        if box is not None:
            track._last_box = box
        track._last_step = step

    @staticmethod
    def _box_of(detection: Mapping[str, Any]) -> dict[str, Any] | None:
        box = detection.get("bbox") or detection.get("bounding_box")
        return box if isinstance(box, Mapping) else None


def track_is_persistent(
    track: Track,
    min_hits: int,
    min_run: int,
    min_fraction: float,
) -> bool:
    """Is *track*'s smoke evidence persistent enough to be believed?

    All three conditions must hold.  They are not redundant -- each removes a
    different failure mode, and each was measured to earn its place on the
    evaluation corpus (see :mod:`mlcore.pipeline`'s persistence block):

    * **min_hits** -- total firing observations.  Rejects the vehicle that
      fired once or twice as it crossed a shadow.
    * **min_run** -- consecutive firing observations.  Rejects the vehicle
      that fired often but never twice in a row, which is flicker, not a plume.
    * **min_fraction** -- share of the vehicle's own observations that fired.
      Scale-free, so it does not weaken when ``frame_sample_rate`` changes,
      and it is the condition that rejects tyre smoke: a car doing donuts
      fires on 7 of the 35 frames it is visible for (20%), a genuinely
      smoking exhaust on most of them.
    """
    return (
        track.hits >= int(min_hits)
        and track.longest_run >= int(min_run)
        and track.hit_fraction >= float(min_fraction)
    )


def persistent_tracks(
    tracks: Iterable[Track],
    min_hits: int,
    min_run: int,
    min_fraction: float,
) -> list[Track]:
    """Every track in *tracks* that satisfies :func:`track_is_persistent`."""
    return [t for t in tracks if track_is_persistent(t, min_hits, min_run, min_fraction)]
