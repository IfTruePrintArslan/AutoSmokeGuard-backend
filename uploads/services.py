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


def probe_video(path, max_seconds):
    """
    Decode the video at ``path`` with OpenCV and return
    ``(width, height, duration_seconds)``.

    ``cv2`` is imported lazily so the app still imports — and images still
    upload — on a deployment that somehow lacks OpenCV; in that case metadata
    is reported as unknown (``None, None, None``) rather than failing the
    request with a 500.  A capture that will not open, or that reports zero
    frames, is ``corrupt_file``.  A clip that decodes fine but runs longer
    than ``max_seconds`` is ``video_too_long`` — this is the only gate that
    can raise that code, since duration is not known until the file is
    actually probed.
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

        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        raw_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
        raw_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0

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

    if duration is not None and max_seconds is not None and duration > max_seconds:
        raise UploadError(
            'video_too_long',
            f'The video is {duration:.1f}s long; the limit is '
            f'{max_seconds}s.',
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
    defines; the caller (the view) is expected to catch it once.
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
    media.save()

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
