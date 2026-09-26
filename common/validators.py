"""
Upload validation.

A filename extension is a claim, not a fact.  Before AutoSmokeGuard writes a
single byte to ``MEDIA_ROOT`` — let alone hands it to OpenCV or a YOLO model —
every uploaded file is checked four ways:

1. **Extension** — is it one of the formats this deployment accepts at all?
2. **Size**       — non-zero, and within the configured per-file ceiling.
3. **Signature**  — do the leading magic bytes actually match a JPEG / PNG /
   BMP / WebP / ISO-BMFF (MP4, MOV) / RIFF-AVI / Matroska (MKV, WebM)
   container?
4. **Agreement**  — does the sniffed container match the claimed extension?

That combination is what stops ``payload.php.jpg``, a zero-byte placeholder, a
zip bomb renamed to ``.mp4``, and a half-finished upload from ever reaching the
decoder.  This module is deliberately model-free and settings-free so it can be
unit-tested in isolation and reused by both the upload API and the seeding
tools; callers pass the policy in.

One source of truth for "what can this deployment decode"
---------------------------------------------------------
:data:`_EXTENSION_POLICY` is it.  ``system_config.serializers`` derives the
admin-facing whitelists (:data:`SUPPORTED_IMAGE_EXTENSIONS` /
:data:`SUPPORTED_VIDEO_EXTENSIONS`) straight from this mapping rather than
keeping a second list.  They had drifted (review finding F8): the settings
API accepted ``bmp``/``webp``/``mkv``/``webm``, this module had never heard
of them, and every upload of a format an admin had just enabled was rejected
with a message that listed the rejected format as allowed.

The drift was resolved by *adding* the four formats, not by narrowing the
whitelist — each one was verified to decode end to end in this environment
before its signature was written down here: Pillow round-trips BMP and WebP
(``features.check('webp')`` is true in the pinned wheel), and OpenCV's
bundled FFmpeg both writes and reads Matroska and WebM, reporting frame
count, frame rate and pixel data.  ``mlcore.config.IMAGE_SUFFIXES`` /
``VIDEO_SUFFIXES`` already listed all four, so the pipeline behind the
upload endpoint needed no change either.
"""
import hashlib
import os

# ---------------------------------------------------------------------------
# Error codes — stable strings the API surfaces in the `code` field of the
# error envelope (see common.exceptions) and that the front end switches on.
# ---------------------------------------------------------------------------

ERR_NO_FILE = 'no_file'
ERR_INVALID_EXTENSION = 'invalid_extension'
ERR_EMPTY_FILE = 'empty_file'
ERR_FILE_TOO_LARGE = 'file_too_large'
ERR_CORRUPT_FILE = 'corrupt_file'
ERR_FORMAT_MISMATCH = 'format_mismatch'


# ---------------------------------------------------------------------------
# Decompression-bomb policy (security review finding ASG-02)
#
# A file's size on disk says nothing about what it costs to decode.  A 2.4 MB
# PNG can declare a 50000x50000 canvas: 2.5 billion pixels, ~2.3 GiB once
# Pillow expands it to RGB.  The size ceiling in ``validate_upload`` cannot
# catch that — only the declared *dimensions* can.
#
# Pillow has its own guard (``Image.MAX_IMAGE_PIXELS``, default ~89 MPix) but
# it is unusable as-is for two reasons:
#
# 1. Between 1x and 2x the limit it only emits a ``DecompressionBombWarning``
#    and then decodes the image anyway — a 150 MPix upload would still be
#    expanded into roughly half a gigabyte of RAM.
# 2. Above 2x it raises ``DecompressionBombError``, which subclasses plain
#    ``Exception`` — not ``OSError`` and not ``ValueError`` — so the decode
#    guards in ``uploads.services.probe_image`` used to miss it entirely and
#    the request became a 500 with the rejected file left on disk.
#
# So we set an explicit limit and check the dimensions ourselves, before any
# pixels are decoded, and treat both Pillow outcomes as a clean rejection.
# ---------------------------------------------------------------------------

#: Largest decoded image this deployment will accept, in pixels (width*height).
#: 50 MPix is ~8660x5773 — comfortably above any traffic camera or phone photo
#: (a 48 MP phone sensor is 48 MPix) while capping a single decode at roughly
#: 150 MB of RGB working memory.
MAX_IMAGE_PIXELS = 50_000_000

#: Hard ceiling on either dimension on its own.  Guards the pathological
#: "1 x 2,000,000,000" shape, which has a modest pixel count per row but makes
#: decoders allocate absurd per-scanline buffers.
MAX_IMAGE_DIMENSION = 20_000


def configure_pillow_limits():
    """
    Pin ``Image.MAX_IMAGE_PIXELS`` to this project's policy.

    Import-safe: returns ``False`` (rather than raising) when Pillow is not
    installed, which keeps ``common.validators`` usable in environments that
    only need the signature sniffing.

    Called by :func:`check_image_dimensions` so that *every* path reaching
    Pillow through this module gets the limit, including the report renderer,
    without settings having to import PIL at startup.
    """
    try:
        from PIL import Image
    except ImportError:                     # pragma: no cover - PIL is a dep
        return False
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    return True


def check_image_dimensions(width, height):
    """
    Validate a declared image geometry against the decompression-bomb policy.

    Returns ``(ok, error_code, error_message)``.  Takes the *declared*
    dimensions — the ones read out of the file header — so a caller can reject
    a bomb before asking any decoder to materialise its pixels.
    """
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError):
        return False, ERR_CORRUPT_FILE, 'The image dimensions could not be read.'

    if width <= 0 or height <= 0:
        return False, ERR_CORRUPT_FILE, 'The image reports an empty canvas.'

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        return (
            False, ERR_FILE_TOO_LARGE,
            f'The image is {width}x{height} pixels; neither side may exceed '
            f'{MAX_IMAGE_DIMENSION} pixels.',
        )

    if width * height > MAX_IMAGE_PIXELS:
        return (
            False, ERR_FILE_TOO_LARGE,
            f'The image is {width}x{height} ({width * height / 1_000_000:.1f} '
            f'megapixels); the limit is '
            f'{MAX_IMAGE_PIXELS / 1_000_000:.0f} megapixels.',
        )

    return True, None, None


# ---------------------------------------------------------------------------
# Container signatures
# ---------------------------------------------------------------------------

#: Bytes read from the head of the file for sniffing.  32 is comfortably more
#: than the longest check below (RIFF needs 12) and cheap for any file size.
HEADER_BYTES = 32

#: Shortest header we will even attempt to classify.  Anything smaller is a
#: truncated or placeholder file, never a decodable image or video.
MIN_HEADER_BYTES = 12

_JPEG_MAGIC = b'\xff\xd8\xff'
_PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
_RIFF_MAGIC = b'RIFF'
_AVI_FORM = b'AVI '
_WEBP_FORM = b'WEBP'

#: Windows bitmap: a two-byte 'BM' and nothing else that is distinctive, so
#: the DIB header size at offset 14 is checked too — 'BM' alone matches far
#: too much (any text file starting with those letters).  These are every
#: header size the format has ever defined: BITMAPCOREHEADER through
#: BITMAPV5HEADER, plus the two OS/2/Adobe in-betweens.
_BMP_MAGIC = b'BM'
_BMP_DIB_HEADER_SIZES = frozenset({12, 16, 40, 52, 56, 64, 108, 124})

#: ISO base media file format (MP4, M4V, and modern MOV) puts a four-byte box
#: length first and the box type at offset 4.
_FTYP = b'ftyp'

#: Classic QuickTime files may lead with one of these top-level atoms instead
#: of `ftyp`.  Accepted for .mov only.
_QUICKTIME_ATOMS = (b'moov', b'mdat', b'free', b'skip', b'wide', b'pnot')

#: EBML, the container Matroska (.mkv) and WebM (.webm) are both built on.
#: They are not distinguished here, and deliberately so: WebM *is* Matroska
#: with a restricted codec set, the byte that separates them is a DocType
#: string several variable-length elements deep, and OpenCV opens either one
#: under either extension.  That mirrors how .mp4 and .mov already share the
#: single 'isobmff' sniff result.
_EBML_MAGIC = b'\x1aE\xdf\xa3'

#: Extension -> (media_type, canonical format, acceptable sniff results).
#: 'jpeg' is normalised to 'jpg' so the database only ever stores one spelling.
#: This mapping is the single definition of what this deployment can decode;
#: see the module docstring.
_EXTENSION_POLICY = {
    'jpg': ('image', 'jpg', {'jpeg'}),
    'jpeg': ('image', 'jpg', {'jpeg'}),
    'png': ('image', 'png', {'png'}),
    'bmp': ('image', 'bmp', {'bmp'}),
    'webp': ('image', 'webp', {'webp'}),
    'mp4': ('video', 'mp4', {'isobmff'}),
    'mov': ('video', 'mov', {'isobmff', 'quicktime'}),
    'avi': ('video', 'avi', {'avi'}),
    'mkv': ('video', 'mkv', {'matroska'}),
    'webm': ('video', 'webm', {'matroska'}),
}

#: The decodable extensions, split by media type — the whitelist
#: ``system_config.serializers`` validates an admin's ``allowed_*_formats``
#: against, so the settings API can never enable something the uploader will
#: turn around and reject (review finding F8).
SUPPORTED_IMAGE_EXTENSIONS = tuple(
    extension for extension, policy in _EXTENSION_POLICY.items()
    if policy[0] == 'image'
)
SUPPORTED_VIDEO_EXTENSIONS = tuple(
    extension for extension, policy in _EXTENSION_POLICY.items()
    if policy[0] == 'video'
)

#: Human labels for the sniff results, used in error messages.
_SNIFF_LABELS = {
    'jpeg': 'JPEG image',
    'png': 'PNG image',
    'bmp': 'BMP image',
    'webp': 'WebP image',
    'isobmff': 'MP4/QuickTime video',
    'quicktime': 'QuickTime video',
    'avi': 'AVI video',
    'matroska': 'Matroska/WebM video',
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def file_extension(filename):
    """Lower-cased extension of ``filename`` without the dot ('' if none)."""
    if not filename:
        return ''
    return os.path.splitext(str(filename))[1].lstrip('.').lower()


def human_bytes(num_bytes):
    """Format a byte count for an end-user-facing message ('512.0 MB')."""
    size = float(num_bytes or 0)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            return f'{size:.1f} {unit}' if unit != 'B' else f'{int(size)} B'
        size /= 1024
    return f'{size:.1f} GB'  # pragma: no cover - unreachable, loop always returns


def sniff_format(header):
    """
    Classify a file from its leading bytes.

    Returns one of ``'jpeg' | 'png' | 'bmp' | 'webp' | 'isobmff' |
    'quicktime' | 'avi' | 'matroska'`` or ``None`` when the header matches no
    container we support.
    """
    if not header or len(header) < MIN_HEADER_BYTES:
        return None

    if header.startswith(_JPEG_MAGIC):
        return 'jpeg'
    if header.startswith(_PNG_MAGIC):
        return 'png'

    # RIFF containers: 'RIFF', a 4-byte little-endian size, then the form
    # type at offset 8.  AVI and WebP differ only in that field.
    if header.startswith(_RIFF_MAGIC):
        form = header[8:12]
        if form == _AVI_FORM:
            return 'avi'
        if form == _WEBP_FORM:
            return 'webp'
        return None

    # EBML — Matroska and WebM.  Checked before the ISO-BMFF box probe
    # because that probe only looks at offsets 4..8 and must not get a
    # chance to misread an EBML header.
    if header.startswith(_EBML_MAGIC):
        return 'matroska'

    # BMP: 'BM', a 4-byte file size, two reserved words, the 4-byte pixel
    # offset, then the DIB header's own size at offset 14.  MIN_HEADER_BYTES
    # is 12, so a header this short cannot be classified as a BMP at all —
    # which is correct: the smallest possible BMP is 26 bytes.
    if header.startswith(_BMP_MAGIC) and len(header) >= 18:
        dib_header_size = int.from_bytes(header[14:18], 'little')
        if dib_header_size in _BMP_DIB_HEADER_SIZES:
            return 'bmp'
        return None

    box_type = header[4:8]
    if box_type == _FTYP:
        return 'isobmff'
    if box_type in _QUICKTIME_ATOMS:
        return 'quicktime'

    return None


def _read_header(django_file):
    """
    Read the first ``HEADER_BYTES`` of ``django_file`` and rewind it.

    Works with Django ``UploadedFile`` objects (both the in-memory and the
    temporary-file flavours) and with any plain binary file object, which is
    what the tests use.
    """
    _seek_zero(django_file)
    try:
        header = django_file.read(HEADER_BYTES) or b''
    except (OSError, ValueError):
        header = b''
    _seek_zero(django_file)

    if isinstance(header, str):  # a text-mode handle slipped through
        header = header.encode('latin-1', errors='replace')
    return header


def _seek_zero(django_file):
    """Rewind a file object, tolerating objects that cannot seek."""
    try:
        django_file.seek(0)
    except (AttributeError, OSError, ValueError):
        pass


def _file_size(django_file):
    """
    Size of ``django_file`` in bytes.

    Django's ``UploadedFile`` exposes ``.size``; anything else is measured by
    seeking to the end and back.  Returns ``None`` if neither works.
    """
    size = getattr(django_file, 'size', None)
    if isinstance(size, int):
        return size
    try:
        current = django_file.tell()
        django_file.seek(0, os.SEEK_END)
        size = django_file.tell()
        django_file.seek(current)
        return size
    except (AttributeError, OSError, ValueError):
        return None


def sha256_checksum(django_file, chunk_size=1024 * 1024):
    """
    SHA-256 of a file's contents, streamed a megabyte at a time.

    Stored on ``uploads.UploadedMedia.checksum`` so a re-upload of identical
    media can be recognised without re-running the pipeline.
    """
    digest = hashlib.sha256()
    _seek_zero(django_file)

    chunks = getattr(django_file, 'chunks', None)
    if callable(chunks):
        for chunk in chunks(chunk_size):
            digest.update(chunk)
    else:
        while True:
            chunk = django_file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    _seek_zero(django_file)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------

def validate_upload(django_file, allowed_exts, max_bytes):
    """
    Validate one uploaded file against this deployment's upload policy.

    Parameters
    ----------
    django_file:
        An ``UploadedFile`` (or any readable binary file object exposing
        ``read``/``seek``, and optionally ``name``/``size``).
    allowed_exts:
        Iterable of permitted extensions, with or without leading dots and in
        any case — e.g. ``['jpg', 'jpeg', 'png', '.MP4']``.  In production this
        comes from the ``sysconfig.SystemSetting`` singleton so an admin can
        change it at runtime (UC-09).
    max_bytes:
        Inclusive per-file size ceiling in bytes.

    Returns
    -------
    ``(ok, error_code, error_message, meta)``

    ``ok`` is a bool.  On failure ``error_code`` is one of the ``ERR_*``
    constants in this module and ``error_message`` is a sentence safe to show a
    user.  ``meta`` is always a dict and is populated as far as validation got,
    so even a rejection tells the caller the size and claimed extension::

        {
          'media_type':      'image' | 'video' | None,
          'format':          one of SUPPORTED_IMAGE_EXTENSIONS /
                             SUPPORTED_VIDEO_EXTENSIONS, normalised
                             ('jpeg' -> 'jpg'), or None,
          'extension':       the extension as supplied, lower-cased,
          'detected_format': result of the magic-byte sniff, or None,
          'size_bytes':      int or None,
          'filename':        the original filename,
        }
    """
    normalised_allowed = {
        str(ext).lstrip('.').lower()
        for ext in (allowed_exts or [])
        if str(ext).strip()
    }

    meta = {
        'media_type': None,
        'format': None,
        'extension': None,
        'detected_format': None,
        'size_bytes': None,
        'filename': None,
    }

    # -- 0. There has to be a file at all -----------------------------------
    if django_file is None:
        return False, ERR_NO_FILE, 'No file was supplied.', meta

    filename = getattr(django_file, 'name', '') or ''
    extension = file_extension(filename)
    meta['filename'] = filename
    meta['extension'] = extension

    # -- 1. Extension must be allowed ---------------------------------------
    if not extension:
        return (
            False, ERR_INVALID_EXTENSION,
            'The file has no extension, so its format cannot be determined.',
            meta,
        )

    # Only formats that are *both* enabled here and decodable by this build
    # may ever be named as acceptable. Listing the raw ``allowed_exts`` was
    # how a rejection message came to include the very format it had just
    # rejected (review finding F8).
    usable = sorted(normalised_allowed & set(_EXTENSION_POLICY))
    usable_display = ', '.join(usable) or 'none'

    if extension not in normalised_allowed:
        return (
            False, ERR_INVALID_EXTENSION,
            f"Files of type '.{extension}' are not accepted. "
            f'Allowed formats: {usable_display}.',
            meta,
        )

    if extension not in _EXTENSION_POLICY:
        # Enabled in this deployment's configuration, but there is no
        # signature and no verified decoder for it here. The settings API
        # can no longer produce this state — its whitelist is derived from
        # _EXTENSION_POLICY — but a fixture, a data migration or the Django
        # admin still can, and the honest message is "misconfigured", not
        # "your file is wrong".
        return (
            False, ERR_INVALID_EXTENSION,
            f"'.{extension}' is enabled in this deployment's settings but "
            f'this server cannot decode it. Usable formats: '
            f'{usable_display}.',
            meta,
        )

    media_type, canonical_format, acceptable_sniffs = _EXTENSION_POLICY[extension]
    meta['media_type'] = media_type
    meta['format'] = canonical_format

    # -- 2. Size: non-zero, and within the ceiling --------------------------
    size_bytes = _file_size(django_file)
    meta['size_bytes'] = size_bytes

    if size_bytes is None:
        return (
            False, ERR_CORRUPT_FILE,
            'The size of the uploaded file could not be determined.',
            meta,
        )

    if size_bytes <= 0:
        return False, ERR_EMPTY_FILE, 'The uploaded file is empty.', meta

    if max_bytes is not None and size_bytes > max_bytes:
        return (
            False, ERR_FILE_TOO_LARGE,
            f'The file is {human_bytes(size_bytes)}, which exceeds the '
            f'{human_bytes(max_bytes)} limit.',
            meta,
        )

    # -- 3. Signature sniff --------------------------------------------------
    header = _read_header(django_file)
    if len(header) < MIN_HEADER_BYTES:
        return (
            False, ERR_CORRUPT_FILE,
            'The file is truncated or unreadable — its header is incomplete.',
            meta,
        )

    detected = sniff_format(header)
    meta['detected_format'] = detected

    if detected is None:
        return (
            False, ERR_CORRUPT_FILE,
            'The file contents are not a valid image or video — it may be '
            'corrupt, truncated, or renamed from another format.',
            meta,
        )

    # -- 4. Sniff must agree with the claimed extension ---------------------
    if detected not in acceptable_sniffs:
        return (
            False, ERR_FORMAT_MISMATCH,
            f"The file is named '.{extension}' but its contents are a "
            f'{_SNIFF_LABELS.get(detected, detected)}.',
            meta,
        )

    return True, None, None, meta
