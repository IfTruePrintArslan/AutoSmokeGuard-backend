"""
Business logic for accepting and probing an uploaded file.

Views stay thin (parse the request, call :func:`store_upload`, translate the
result to a response); everything that touches the filesystem, the upload
policy or the decoders lives here so it can be unit-tested without spinning up
the DRF request/response cycle.

The upload path has three gates, cheapest first:

1. :func:`common.validators.validate_upload` — extension, size and magic-byte
   sniff.  No bytes are decoded yet.
2. De-duplication — a SHA-256 match against this user's existing uploads skips
   writing a second copy of a file they already sent us.
3. :func:`probe_image` / :func:`probe_video` — actually decode the saved file
   to prove it is not just "the right magic bytes" but genuinely openable, and
   to fill in ``width`` / ``height`` / ``duration_seconds``.  A video that
   decodes fine but runs longer than the configured ceiling is rejected here
   too, since duration can only be known once the file has been probed.

Any failure at any gate raises :class:`UploadError`, which carries the exact
``code`` the API contract promises (``invalid_format`` | ``file_too_large`` |
``corrupt_file`` | ``empty_file`` | ``video_too_long`` | ``no_file``) plus a
human-readable message.  The view catches it once and turns it into the
standard error envelope.
"""
import logging
import math

from django.db import transaction

from common.validators import (
    ERR_CORRUPT_FILE,
    ERR_EMPTY_FILE,
    ERR_FILE_TOO_LARGE,
    ERR_FORMAT_MISMATCH,
    ERR_INVALID_EXTENSION,
    ERR_NO_FILE,
    MAX_IMAGE_PIXELS,
    check_image_dimensions,
    configure_pillow_limits,
    human_bytes,
    sha256_checksum,
    validate_upload,
)

from .models import MEDIA_TYPE_IMAGE, MEDIA_TYPE_VIDEO, UploadedMedia

logger = logging.getLogger('asg.uploads')

#: Frame rate assumed when a container reports an unusable one — see
#: :func:`probe_video`.  30 fps is the most common real-world capture rate and
#: is what ``mlcore`` assumes when it samples frames, so it is the honest
#: conversion between "how many frames" and "how many seconds".
FALLBACK_VIDEO_FPS = 30.0

#: Shown to the user when the filesystem under ``MEDIA_ROOT`` will not accept
#: the file (read-only mount, full disk, wrong ownership).  Deliberately says
#: nothing about paths, errno values or the storage backend: the operator gets
#: the detail from ``logger.exception``, the caller gets a sentence they can
#: act on.  Mirrors the discipline ``analysis.worker._SAFE_MESSAGES`` already
#: applies to the same class of fault during a run.
STORAGE_UNAVAILABLE_MESSAGE = (
    'The server could not store the uploaded file. Please try again in a '
    'moment, or contact an administrator if the problem continues.'
)

# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class UploadError(Exception):
    """
    A validation or integrity failure that maps directly onto an HTTP
    response.

    ``code`` is one of the machine-readable strings the API contract defines
    for ``POST /api/upload``; ``message`` is safe to show a user as-is.
    """

    def __init__(self, code, message, status_code=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


# Map common.validators' generic codes onto the contract's upload-specific
# vocabulary.  ``format_mismatch`` (contents don't match the claimed
# extension) is not a code the contract lists separately — it is, in effect,
# "the file is not what it claims to be", which is what ``corrupt_file``
# means to the caller.
_VALIDATOR_CODE_MAP = {
    ERR_NO_FILE: 'no_file',
    ERR_INVALID_EXTENSION: 'invalid_format',
    ERR_EMPTY_FILE: 'empty_file',
    ERR_FILE_TOO_LARGE: 'file_too_large',
    ERR_CORRUPT_FILE: 'corrupt_file',
    ERR_FORMAT_MISMATCH: 'corrupt_file',
}


def _map_validation_error(err_code, err_message, meta, settings_row):
    """Translate a ``validate_upload`` failure into an :class:`UploadError`."""
    code = _VALIDATOR_CODE_MAP.get(err_code, 'corrupt_file')

    if code == 'invalid_format':
        accepted = ', '.join(settings_row.all_formats) or 'none'
        extension = meta.get('extension') or ''
        message = (
            f"Unsupported file format '.{extension}'. Accepted: {accepted}."
        )
    elif code == 'file_too_large':
        size_bytes = meta.get('size_bytes')
        message = (
            f'File is {human_bytes(size_bytes)}; the limit is '
            f'{settings_row.max_upload_mb} MB.'
        )
    else:
        message = err_message

    return UploadError(code, message)


# ---------------------------------------------------------------------------
# Probing — proving the bytes are genuinely decodable, not just well-signed
# ---------------------------------------------------------------------------


def probe_image(path):
    """
    Decode the image at ``path`` and return ``(width, height)``.

    Three gates, cheapest first:

    1. **Header + geometry.** ``Image.open`` parses only the header, so the
       declared dimensions are known before a single pixel is decoded. They
       are checked against the decompression-bomb policy in
       :mod:`common.validators` and an oversized canvas is rejected here —
       nothing downstream ever allocates it.
    2. **Integrity.** Pillow's ``verify()`` catches truncated/corrupt data.
    3. **Geometry, again, from a fresh handle** — ``verify()`` leaves the
       ``Image`` unusable for anything else, including reading ``.size``.

    Any decode failure is reported as ``corrupt_file`` (the magic bytes
    matched but the payload behind them did not); an over-large canvas is
    ``file_too_large``.

    Security (finding ASG-02): ``PIL.Image.DecompressionBombError`` subclasses
    plain ``Exception``, so it is listed explicitly in every ``except`` below.
    Leaving it out is what previously turned a 2.4 MB crafted PNG into a 500
    with the rejected file stranded in ``MEDIA_ROOT``.
    """
    from PIL import Image, UnidentifiedImageError

    configure_pillow_limits()
    decode_errors = (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
    )

    # -- 1. Header only: reject an oversized canvas before decoding it -------
    try:
        with Image.open(path) as header:
            width, height = header.size
    except Image.DecompressionBombError as exc:
        # Pillow refused the geometry outright (declared pixels > 2x limit).
        raise UploadError(
            'file_too_large',
            'The image declares more pixels than this service will decode. '
            f'The limit is {MAX_IMAGE_PIXELS // 1_000_000} megapixels.',
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise UploadError(
            'corrupt_file',
            'The image could not be decoded — it may be corrupt or truncated.',
        ) from exc

    ok, err_code, err_message = check_image_dimensions(width, height)
    if not ok:
        raise UploadError(
            'file_too_large' if err_code == ERR_FILE_TOO_LARGE else 'corrupt_file',
            err_message,
        )

    # -- 2. Integrity ---------------------------------------------------------
    try:
        with Image.open(path) as candidate:
            candidate.verify()
    except decode_errors as exc:
        raise UploadError(
            'corrupt_file',
            'The image could not be decoded — it may be corrupt or truncated.',
        ) from exc

    # -- 3. Geometry from a fresh handle -------------------------------------
    try:
        with Image.open(path) as reopened:
            width, height = reopened.size
    except decode_errors as exc:
        raise UploadError(
            'corrupt_file',
            'The image could not be decoded — it may be corrupt or truncated.',
        ) from exc

    return width, height


def _finite(value):
    """
    Coerce an OpenCV property to a finite float, or ``0.0``.

    ``cv2.VideoCapture.get`` returns a C double straight out of the demuxer.
    A container with a damaged header can make that ``nan`` or ``inf``, and
    both slip past a naive ``<= 0`` guard: ``nan <= 0`` is ``False``, and every
    later comparison against ``nan`` is ``False`` too, so a check written as
    ``duration > max_seconds`` silently passes.  Folding them to ``0.0`` puts
    them back on the rejection path with the other unusable values.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def probe_video(path, max_seconds):
    """
    Decode the video at ``path`` with OpenCV and return
    ``(width, height, duration_seconds)``.

    ``cv2`` is imported lazily so the app still imports — and images still
    upload — on a deployment that somehow lacks OpenCV; in that case metadata
    is reported as unknown (``None, None, None``) rather than failing the
    request with a 500.  A capture that will not open, or that reports zero
    (or a negative, or a non-finite) frame count, is ``corrupt_file``.  A clip
    that decodes fine but runs longer than ``max_seconds`` is
    ``video_too_long`` — this is the only gate that can raise that code, since
    duration is not known until the file is actually probed.

    Unusable frame rates (robustness finding §C)
    --------------------------------------------
    A container whose frame-rate atom is missing, zero or corrupt makes
    OpenCV report ``fps <= 0``.  This function used to set ``duration = None``
    in that case, and the length check below is guarded on
    ``duration is not None`` — so the ``max_video_seconds`` cap was skipped
    **entirely** and a clip of unbounded real length went straight to the ML
    pipeline.

    The fix bounds the work instead of the wall clock.  ``frame_count`` is
    still trustworthy when ``fps`` is not (ffmpeg counts packets; the frame
    rate comes from a header field), and what the cap actually protects is
    how many frames the pipeline has to decode and run inference on.  So an
    unusable frame rate is compared against the *frame budget* the cap
    implies at :data:`FALLBACK_VIDEO_FPS`.

    Rejecting the file outright as ``corrupt_file`` was the other option and
    was not taken: OpenCV has already proved it can open the container and
    count its frames, so a missing frame-rate atom alone is not evidence of
    corruption, and refusing it would turn an oddly-muxed but perfectly
    decodable clip into a false rejection.  A clip with an odd-but-*valid*
    rate — a 0.5 fps time-lapse, a 119.88 fps capture — never reaches this
    branch at all; it takes the exact arithmetic above.

    ``duration_seconds`` is still returned as ``None`` in that case: the frame
    budget is good enough to enforce a ceiling, but it is an assumption, and
    persisting an assumption as if it were a measurement would put a wrong
    number on the detail screen and in the PDF.
    """
    try:
        import cv2
    except ImportError:
        logger.warning(
            'OpenCV is not available; skipping video probe for %s '
            '(metadata will be unknown).', path,
        )
        return None, None, None

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise UploadError(
                'corrupt_file',
                'The video could not be opened — it may be corrupt or use '
                'an unsupported codec.',
            )

        frame_count = _finite(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = _finite(capture.get(cv2.CAP_PROP_FPS))
        raw_width = _finite(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = _finite(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_count <= 0:
            raise UploadError(
                'corrupt_file',
                'The video contains no readable frames.',
            )

        width = int(raw_width) or None
        height = int(raw_height) or None
        duration = (frame_count / fps) if fps > 0 else None
    finally:
        capture.release()

    if max_seconds is not None:
        if duration is not None:
            if duration > max_seconds:
                raise UploadError(
                    'video_too_long',
                    f'The video is {duration:.1f}s long; the limit is '
                    f'{max_seconds}s.',
                )
        else:
            # No usable frame rate: enforce the cap as a frame budget instead
            # of skipping it.  See the docstring for why this is not a
            # corrupt_file rejection.
            frame_budget = max_seconds * FALLBACK_VIDEO_FPS
            logger.warning(
                'Video %s reports no usable frame rate (fps=%s); checking '
                '%d frames against a %d-frame budget.',
                path, fps, int(frame_count), int(frame_budget),
            )
            if frame_count > frame_budget:
                raise UploadError(
                    'video_too_long',
                    f'The video does not report a usable frame rate, so its '
                    f'length is checked by frame count: '
                    f'{int(frame_count)} frames is more than the '
                    f'{int(frame_budget)} a {max_seconds}s clip holds at '
                    f'{FALLBACK_VIDEO_FPS:.0f} fps.',
                )

    return width, height, duration


def _probe_and_apply(media, settings_row):
    """Probe ``media``'s saved file and persist the resulting metadata."""
    if media.media_type == MEDIA_TYPE_IMAGE:
        width, height = probe_image(media.file.path)
        media.width = width
        media.height = height
        media.save(update_fields=['width', 'height'])
        return

    width, height, duration = probe_video(media.file.path, settings_row.max_video_seconds)
    media.width = width
    media.height = height
    media.duration_seconds = duration
    media.save(update_fields=['width', 'height', 'duration_seconds'])


def _delete_media_hard(media):
    """Remove a just-created row and its file after a failed probe."""
    try:
        if media.file:
            media.file.delete(save=False)
    except Exception:  # pragma: no cover - best-effort cleanup
        logger.warning('Could not remove file for rejected upload %s', media.media_id)
    media.delete()


def _file_exists_on_disk(field_file):
    """True when ``field_file`` points at a real, still-present file."""
    if not field_file:
        return False
    try:
        return field_file.storage.exists(field_file.name)
    except Exception:  # pragma: no cover - storage backend misbehaving
        return False


class _NameRecordingStorage:
    """
    Transparent storage wrapper that remembers the path it was asked to write.

    Needed because :func:`common.storage.user_media_path` mints a **fresh
    UUID on every call**: after a failed write there is otherwise no way to
    ask the field where it had been writing.  ``FieldFile.save`` only assigns
    ``self.name`` and sets ``_committed`` *after* ``storage.save()`` returns,
    so on failure ``media.file.name`` is still the client-supplied filename —
    a path this request never wrote, and one that must never be handed to
    ``storage.delete()``.

    That matters for one specific fault: a write that dies part-way through
    (``ENOSPC``).  Django's ``FileSystemStorage._save`` opens the destination
    with ``O_CREAT`` and then streams into it, and cleans nothing up if a
    ``write()`` raises — so the file exists, truncated, referenced by no row.
    Recording the name on the way in is what lets
    :func:`_discard_after_storage_failure` remove it.

    Installed on the **instance's** ``FieldFile`` only (``FieldFile.storage``
    is a per-instance attribute assigned in ``FieldFile.__init__``), never on
    the field or on the shared storage object, so concurrent uploads cannot
    see each other's wrapper.
    """

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.attempted_name = None

    def save(self, name, content, max_length=None):
        self.attempted_name = name
        return self._wrapped.save(name, content, max_length=max_length)

    def __getattr__(self, item):
        # object.__getattribute__ rather than self._wrapped: if _wrapped is
        # somehow missing this raises AttributeError instead of recursing.
        return getattr(object.__getattribute__(self, '_wrapped'), item)


def _discard_after_storage_failure(media, attempted_name=None):
    """
    Leave neither a row nor a half-written file behind after an ``OSError``.

    Ordering matters here.  ``Model.save()`` writes the file from
    ``FileField.pre_save``, i.e. *before* the ``INSERT``, so an ``OSError``
    from storage means the row was never created — but ``media.pk`` is
    populated regardless (``media_id`` is a UUID primary key with a
    ``default``), so "was it saved?" has to be asked of the database, not of
    the instance.

    Two candidate paths are removed, and only these two:

    * the **committed** field name, if storage got far enough to return one;
    * ``attempted_name``, the exact path
      :class:`_NameRecordingStorage` saw on the way in.

    An *uncommitted* ``media.file.name`` is deliberately never used: it is
    still whatever the client called their file, so deleting it would aim at
    ``MEDIA_ROOT/<their filename>`` — a path this request never wrote.
    """
    field_file = getattr(media, 'file', None)
    storage = getattr(field_file, 'storage', None)

    candidates = []
    if getattr(field_file, '_committed', False) and getattr(field_file, 'name', None):
        candidates.append(field_file.name)
    if attempted_name and attempted_name not in candidates:
        candidates.append(attempted_name)

    for name in candidates:
        try:
            if storage is not None and storage.exists(name):
                storage.delete(name)
                logger.warning(
                    'Removed the partial upload left by a storage failure '
                    'for media %s', media.media_id,
                )
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.warning(
                'Could not remove the partial upload left by a storage '
                'failure for media %s', media.media_id,
            )

    try:
        if type(media).objects.filter(pk=media.pk).exists():
            media.delete()
    except Exception:  # pragma: no cover - best-effort cleanup
        logger.warning(
            'Could not remove the orphan row left by a storage failure for '
            'media %s', media.media_id,
        )


def _save_media_file(media, user, meta):
    """
    Persist ``media`` (which writes its file), or fail with a clean 503.

    ``store_upload`` is the *synchronous* half of the pipeline, and it was the
    only half with no filesystem-fault handling: a read-only ``MEDIA_ROOT``, a
    full disk or a bad volume mount turned ``POST /api/upload`` into an
    unhandled 500 with a traceback in the log (robustness finding §E).  The
    background worker already treats exactly this class of fault as a clean,
    user-safe failure (``analysis.worker._SAFE_MESSAGES``); this brings the
    upload path to the same standard.

    ``OSError`` rather than ``PermissionError`` on purpose — it is the common
    base of every way the filesystem can refuse the write that matters here:
    ``PermissionError`` (EACCES, the read-only mount / wrong ownership case),
    ``OSError(ENOSPC)`` (full disk), ``OSError(EROFS)`` (read-only filesystem)
    and ``OSError(EDQUOT)`` (quota).  ``os.makedirs`` for the per-user
    directory raises the same family and is inside ``FileSystemStorage._save``,
    so it is covered by the same ``try``.

    503 rather than 500: the request was valid and will succeed once the
    operator fixes the volume, which is what "Service Unavailable" means and
    what tells a client it is worth retrying.

    The ``atomic()`` block is load-bearing, not decoration.  ``Model.save()``
    runs its field ``pre_save`` hooks (which is where the file is written)
    inside ``transaction.mark_for_rollback_on_error``, so an exception
    escaping it poisons *the enclosing transaction* — ``needs_rollback`` goes
    up and every later query on that connection raises
    ``TransactionManagementError``.  Wrapping the save in its own atomic
    block turns that into a savepoint rollback instead, so the cleanup below
    (and the view that has to render a response afterwards) still has a
    usable connection.  Without it the 503 would only work outside an
    enclosing transaction, i.e. not under ``ATOMIC_REQUESTS`` and not under
    the test suite.
    """
    field_file = media.file
    real_storage = field_file.storage
    recorder = _NameRecordingStorage(real_storage)
    field_file.storage = recorder
    try:
        with transaction.atomic():
            media.save()
    except OSError as exc:
        # exception(), not warning(): unlike a rejected upload this is an
        # operational fault an administrator has to see in full.
        logger.exception(
            'upload storage failure user=%s filename=%s size=%s',
            getattr(user, 'pk', None), meta.get('filename'),
            meta.get('size_bytes'),
        )
        _discard_after_storage_failure(media, recorder.attempted_name)
        raise UploadError(
            'storage_unavailable',
            STORAGE_UNAVAILABLE_MESSAGE,
            status_code=503,
        ) from exc
    finally:
        # A successful save replaces the FieldFile wholesale (FileField's
        # descriptor rebuilds it from the stored name), so the wrapper
        # normally disappears on its own — but restore it explicitly so
        # nothing downstream can ever be handed the proxy.
        if getattr(media.file, 'storage', None) is recorder:
            media.file.storage = real_storage


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------


def store_upload(user, django_file, settings_row):
    """
    Validate, de-duplicate, save and probe one uploaded file.

    Returns ``(media, created)`` — ``created`` is ``False`` when an identical
    file (same owner, same SHA-256, and the previous copy is still on disk)
    already existed, in which case the *existing* row is returned untouched
    and nothing new is written to disk.

    Raises :class:`UploadError` for every rejection reason the contract
    defines — plus ``storage_unavailable`` (503) when the filesystem under
    ``MEDIA_ROOT`` refuses the write; the caller (the view) is expected to
    catch it once and turn ``exc.status_code`` into the response.
    """
    ok, err_code, err_message, meta = validate_upload(
        django_file, settings_row.all_formats, settings_row.max_upload_bytes,
    )
    if not ok:
        logger.warning(
            'upload rejected user=%s filename=%s size=%s format=%s code=%s',
            getattr(user, 'pk', None), meta.get('filename'),
            meta.get('size_bytes'), meta.get('format'), err_code,
        )
        raise _map_validation_error(err_code, err_message, meta, settings_row)

    checksum = sha256_checksum(django_file)

    existing = (
        UploadedMedia.objects
        .filter(user=user, checksum=checksum)
        .order_by('-upload_timestamp')
        .first()
    )
    if existing is not None and _file_exists_on_disk(existing.file):
        logger.info(
            'upload deduplicated user=%s filename=%s size=%s format=%s '
            'media_id=%s',
            user.pk, meta['filename'], meta['size_bytes'], meta['format'],
            existing.media_id,
        )
        return existing, False

    media = UploadedMedia(
        user=user,
        filename=meta['filename'] or getattr(django_file, 'name', ''),
        format=meta['format'],
        media_type=meta['media_type'],
        size_bytes=meta['size_bytes'],
        checksum=checksum,
        file=django_file,
    )
    _save_media_file(media, user, meta)

    try:
        _probe_and_apply(media, settings_row)
    except UploadError as exc:
        logger.warning(
            'upload rejected after probe user=%s filename=%s code=%s',
            user.pk, media.filename, exc.code,
        )
        _delete_media_hard(media)
        raise

    logger.info(
        'upload stored user=%s filename=%s size=%s format=%s media_id=%s',
        user.pk, media.filename, media.size_bytes, media.format,
        media.media_id,
    )
    return media, True
