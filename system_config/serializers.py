"""
Serializers for the system-configuration API (UC-09).

The model (see ``system_config/models.py``) stores the two format allow-lists
as comma-separated strings so a plain ``CharField`` is enough and there is
nothing to migrate when the pipeline learns a new container.  The API,
however, promises JSON arrays (``API_CONTRACT.md`` -> "System settings").
:class:`FormatListField` is the seam between the two: it also carries the
format-level validation (token shape + decoder whitelist), because that
validation is inseparable from "what does a format string even mean here".

The serializer additionally:

* Cross-validates the severity band (``severity_low_max <
  severity_moderate_max``), which no single field can check on its own.
* Computes the read-only ``runtime`` block — device, segmenter mode, weight
  presence, worker thread count, and (cached) model metrics — none of which
  is stored, all of which the Upload/Settings screens need to explain the
  numbers above them.
* Logs every accepted change as a field-by-field before/after diff to
  ``asg.sysconfig``, because this is one of the few surfaces in the system
  that can silently change behaviour for every user at once.
"""
import json
import logging
import re
from pathlib import Path

from django.conf import settings
from drf_spectacular.utils import extend_schema_field
from rest_framework import serializers

from common.validators import (
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
)

from .models import SystemSetting

logger = logging.getLogger('asg.sysconfig')

# ---------------------------------------------------------------------------
# Format list <-> CSV translation
# ---------------------------------------------------------------------------

#: Lower-case letters/digits only, 2-5 characters — rejects dots, slashes,
#: whitespace and anything else that is not a bare extension token.
FORMAT_TOKEN_RE = re.compile(r'^[a-z0-9]{2,5}$')

#: What this deployment can actually decode — re-exported from the uploader's
#: own policy table rather than restated here.
#:
#: These two names used to be hand-written literals and they had drifted from
#: ``common.validators._EXTENSION_POLICY``: the settings API accepted
#: ``bmp``/``webp``/``mkv``/``webm`` (the Settings chip editor even
#: placeholders "Add format (e.g. webp)") and every subsequent upload of one
#: was rejected — by a message that listed the rejected format among the
#: accepted ones.  Deriving them makes "an admin can only enable something
#: the uploader will accept" true by construction (review finding F8).
#:
#: To add a container: teach ``common.validators`` its signature, prove
#: Pillow/OpenCV genuinely decode it, add the ``_EXTENSION_POLICY`` entry —
#: and it appears here on its own.
SUPPORTED_IMAGE_FORMATS = SUPPORTED_IMAGE_EXTENSIONS
SUPPORTED_VIDEO_FORMATS = SUPPORTED_VIDEO_EXTENSIONS


def _normalise_tokens(raw_items):
    """
    Lower-case, dot-strip, whitespace-strip and de-duplicate format tokens,
    preserving first-seen order.

    Pure and side-effect free so both directions of :class:`FormatListField`
    (and the tests) can share it.
    """
    tokens = []
    for item in raw_items:
        token = str(item).strip().lstrip('.').lower()
        if token and token not in tokens:
            tokens.append(token)
    return tokens


@extend_schema_field(serializers.ListField(child=serializers.CharField()))
class FormatListField(serializers.Field):
    """
    A model ``CharField`` of comma-separated extensions, exposed as a JSON
    array.

    ``to_representation`` turns the stored CSV string into a list.
    ``to_internal_value`` accepts either a JSON list or a CSV string (for
    convenience/symmetry), normalises it, checks every token against
    ``whitelist``, and returns a normalised CSV string ready to store —
    callers never see the model's storage format directly.
    """

    default_error_messages = {
        'not_a_list': 'Expected a list of format strings (or a comma-separated string).',
        'not_a_string': 'Each format must be a string.',
        'empty': 'At least one format is required.',
        'bad_token': (
            '"{value}" is not a valid format token — use lower-case letters/'
            'digits only, 2-5 characters, no dots, slashes or whitespace.'
        ),
        'unsupported': '"{value}" is not supported. Supported formats: {supported}.',
    }

    def __init__(self, whitelist, **kwargs):
        self.whitelist = tuple(whitelist)
        super().__init__(**kwargs)

    def to_representation(self, value):
        if isinstance(value, str):
            raw_items = value.split(',')
        elif isinstance(value, (list, tuple)):
            raw_items = list(value)
        else:
            raw_items = []
        return _normalise_tokens(raw_items)

    def to_internal_value(self, data):
        if isinstance(data, str):
            raw_items = data.split(',')
        elif isinstance(data, (list, tuple)):
            raw_items = list(data)
        else:
            self.fail('not_a_list')

        for item in raw_items:
            if not isinstance(item, str):
                self.fail('not_a_string')

        tokens = _normalise_tokens(raw_items)
        if not tokens:
            self.fail('empty')

        for token in tokens:
            if not FORMAT_TOKEN_RE.match(token):
                self.fail('bad_token', value=token)
            if token not in self.whitelist:
                self.fail('unsupported', value=token,
                          supported=', '.join(self.whitelist))

        return ','.join(tokens)


# ---------------------------------------------------------------------------
# Runtime block — computed, never stored, must never raise
# ---------------------------------------------------------------------------

#: ``{(metrics_path, mtime): parsed_dict_or_None}``. Keyed on mtime (per the
#: spec) *and* path, so two tests writing different temp ``metrics.json``
#: files in the same wall-clock tick can never shadow one another.
_METRICS_CACHE = {}


def _model_metrics():
    """
    The evaluation metrics for the current segmenter checkpoint, reduced to
    ``{dice, iou, pixel_accuracy}``.

    Returns ``None`` when ``ml_assets/metrics.json`` is absent or cannot be
    parsed into that shape — a fresh checkout before ``evaluate.py`` has ever
    run is a normal state, not an error. The parse is cached by the file's
    mtime so a client polling ``GET /api/settings`` does not force a disk read
    (and a JSON parse) on every single request.
    """
    metrics_path = Path(settings.ASG['ML_ASSETS_DIR']) / 'metrics.json'

    try:
        mtime = metrics_path.stat().st_mtime
    except OSError:
        return None

    cache_key = (str(metrics_path), mtime)
    if cache_key in _METRICS_CACHE:
        return _METRICS_CACHE[cache_key]

    try:
        payload = json.loads(metrics_path.read_text())
        val = payload['val']
        metrics = {
            'dice': val['dice'],
            'iou': val['iou'],
            'pixel_accuracy': val['pixel_accuracy'],
        }
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning('Could not parse %s into model metrics.', metrics_path)
        metrics = None

    _METRICS_CACHE[cache_key] = metrics
    return metrics


def _runtime_snapshot():
    """
    Best-effort description of what the ML pipeline is actually running on.

    ``mlcore`` is imported lazily, inside a ``try/except``: a sibling agent is
    still writing that package, and a half-installed/missing ``mlcore`` must
    never take down the settings endpoint the Upload screen polls for its
    limits. On any failure ``device`` and ``segmenter_mode`` fall back to
    ``"unknown"`` rather than propagating the exception.
    """
    asg = settings.ASG
    yolo_present = Path(asg['YOLO_WEIGHTS']).exists()
    segmenter_present = Path(asg['SEGMENTER_WEIGHTS']).exists()

    device = 'unknown'
    segmenter_mode = 'unknown'
    try:
        from mlcore import device_string, get_device
        device = device_string(get_device())
        segmenter_mode = 'unet' if segmenter_present else 'classical'
    except Exception:  # noqa: BLE001 - deliberately broad, see docstring
        logger.warning(
            'mlcore unavailable; reporting device/segmenter_mode as unknown.',
            exc_info=True,
        )

    return {
        'device': device,
        'segmenter_mode': segmenter_mode,
        'yolo_weights_present': yolo_present,
        'segmenter_weights_present': segmenter_present,
        'worker_threads': asg['WORKER_THREADS'],
        'model_metrics': _model_metrics(),
    }


class SystemRuntimeSerializer(serializers.Serializer):
    """
    Read-only shape of the computed ``runtime`` block (see
    :func:`_runtime_snapshot`). Never bound to a model — it exists purely so
    ``@extend_schema_field`` can describe the nested object properly instead
    of drf-spectacular defaulting an untyped dict to a bare string.
    """
    device = serializers.CharField()
    segmenter_mode = serializers.CharField()
    yolo_weights_present = serializers.BooleanField()
    segmenter_weights_present = serializers.BooleanField()
    worker_threads = serializers.IntegerField()
    model_metrics = serializers.DictField(allow_null=True)


# ---------------------------------------------------------------------------
# The configuration object itself
# ---------------------------------------------------------------------------

class SystemSettingSerializer(serializers.ModelSerializer):
    """
    Read/write view of the :class:`~system_config.models.SystemSetting`
    singleton, shaped exactly like ``SettingsObj`` in ``API_CONTRACT.md``.

    Every writable field re-declares its own range so a 400 always carries a
    per-field message under ``errors.<field>`` (the exception handler in
    ``common.exceptions`` turns serializer errors into that shape
    automatically); :meth:`validate` adds the one check that spans two
    fields.
    """

    allowed_image_formats = FormatListField(whitelist=SUPPORTED_IMAGE_FORMATS)
    allowed_video_formats = FormatListField(whitelist=SUPPORTED_VIDEO_FORMATS)

    max_upload_mb = serializers.IntegerField(min_value=1, max_value=4096)
    max_video_seconds = serializers.IntegerField(min_value=1, max_value=3600)
    frame_sample_rate = serializers.IntegerField(min_value=1, max_value=60)

    confidence_threshold = serializers.FloatField(min_value=0.0, max_value=1.0)
    smoke_mask_threshold = serializers.FloatField(min_value=0.0, max_value=1.0)
    severity_low_max = serializers.FloatField(min_value=0.0, max_value=1.0)
    severity_moderate_max = serializers.FloatField(min_value=0.0, max_value=1.0)

    auto_generate_pdf = serializers.BooleanField()

    updated_by = serializers.SerializerMethodField()
    runtime = serializers.SerializerMethodField()

    class Meta:
        model = SystemSetting
        fields = [
            'allowed_image_formats', 'allowed_video_formats',
            'max_upload_mb', 'confidence_threshold', 'smoke_mask_threshold',
            'severity_low_max', 'severity_moderate_max', 'frame_sample_rate',
            'auto_generate_pdf', 'max_video_seconds',
            'updated_at', 'updated_by', 'runtime',
        ]
        read_only_fields = ['updated_at']

    # -- computed / read-only fields -----------------------------------

    @extend_schema_field(serializers.EmailField(allow_null=True))
    def get_updated_by(self, instance):
        """The acting admin's e-mail, or ``None`` if never edited."""
        user = instance.updated_by
        return user.email if user else None

    @extend_schema_field(SystemRuntimeSerializer)
    def get_runtime(self, instance):
        """Delegate to :func:`_runtime_snapshot`; ``instance`` is unused."""
        return _runtime_snapshot()

    # -- validation -------------------------------------------------------

    def validate(self, attrs):
        """
        ``severity_low_max`` must stay strictly below
        ``severity_moderate_max``.

        A ``PATCH`` may touch only one of the two, so the comparison falls
        back to the current instance value for whichever side was not sent —
        a lone ``{"severity_low_max": 0.9}`` against a moderate cap of 0.66
        must fail just as loudly as sending both at once.
        """
        instance = self.instance
        low = attrs.get(
            'severity_low_max',
            getattr(instance, 'severity_low_max', None) if instance else None,
        )
        moderate = attrs.get(
            'severity_moderate_max',
            getattr(instance, 'severity_moderate_max', None) if instance else None,
        )
        if low is not None and moderate is not None and low >= moderate:
            raise serializers.ValidationError({
                'severity_low_max': (
                    'severity_low_max must be strictly less than '
                    'severity_moderate_max.'
                ),
            })
        return attrs

    # -- persistence --------------------------------------------------------

    def update(self, instance, validated_data):
        """
        Apply a partial update, stamp the audit columns, and log a
        field-by-field before/after diff.

        Only keys actually present in ``validated_data`` are touched, so an
        unspecified field is guaranteed to keep its previous value (the
        contract's "PATCH is a partial update").
        """
        request = self.context.get('request')
        actor = getattr(request, 'user', None)

        changes = {}
        for field, new_value in validated_data.items():
            old_value = getattr(instance, field)
            if old_value != new_value:
                changes[field] = (old_value, new_value)
                setattr(instance, field, new_value)

        instance.updated_by = actor
        instance.save()

        actor_label = getattr(actor, 'email', None) or 'unknown'
        if changes:
            diff = '; '.join(
                f'{field}: {old!r} -> {new!r}'
                for field, (old, new) in changes.items()
            )
            logger.info('SystemSetting updated by %s -- %s', actor_label, diff)
        else:
            logger.info(
                'SystemSetting PATCH by %s applied no field changes.',
                actor_label,
            )

        return instance
