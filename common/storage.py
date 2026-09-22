"""
Where things live on disk.

Every byte AutoSmokeGuard writes for a user lands under ``MEDIA_ROOT`` in one
of three trees::

    media/
      uploads/<user_id>/<uuid>.<ext>        original image / video
      analyses/<analysis_id>/               pipeline artefacts
        preview.jpg                           annotated overview frame
        crops/<vehicle_id>.jpg                per-vehicle crop
        masks/<smoke_id>.png                  per-region smoke mask
      reports/<report_id>.pdf               generated emission report

Two rules drive the layout:

* **Never trust the uploaded filename.** The original name is kept in the
  database (``UploadedMedia.filename``) for display only; on disk the file is
  renamed to a fresh UUID.  That neutralises path traversal (``../../etc``),
  Windows reserved names, unicode look-alikes and collisions in one move.
* **Namespace by owner / job id.** One directory per user and per analysis
  keeps listings small and makes "delete this account" or "purge this
  analysis" a single ``rmtree``.

Database columns such as ``AnalysisResult.preview_path`` and
``GeneratedReport.report_path`` always store the *relative* form, so media can
be moved to another volume or an object store without rewriting rows.
"""
import logging
import os
import posixpath
import shutil
import uuid
from pathlib import Path

from django.conf import settings

from .validators import file_extension

logger = logging.getLogger('asg.storage')

# Top-level trees inside MEDIA_ROOT.
UPLOADS_DIR = 'uploads'
ANALYSES_DIR = 'analyses'
REPORTS_DIR = 'reports'

# Conventional sub-folders of an analysis artefact directory.
CROPS_SUBDIR = 'crops'
MASKS_SUBDIR = 'masks'


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------

def user_media_path(instance, filename):
    """
    ``upload_to`` callable for ``uploads.UploadedMedia.file``.

    Returns a storage-relative, forward-slashed path of the form
    ``uploads/<user_id>/<uuid4>.<ext>``.  Called by Django during the model's
    first save, at which point ``instance.user_id`` is already populated; the
    ``anonymous`` fallback exists only so a half-built instance in a test or a
    shell session cannot raise from inside a field's ``pre_save``.
    """
    extension = file_extension(filename) or 'bin'
    owner = getattr(instance, 'user_id', None) or 'anonymous'
    return posixpath.join(UPLOADS_DIR, str(owner), f'{uuid.uuid4()}.{extension}')


# ---------------------------------------------------------------------------
# Analysis artefacts
# ---------------------------------------------------------------------------

def analysis_artifact_dir(analysis_id, subdir=None, create=True):
    """
    Absolute directory for one analysis run's artefacts.

    ``analysis_artifact_dir(a.analysis_id)`` -> ``<MEDIA_ROOT>/analyses/<id>``
    ``analysis_artifact_dir(a.analysis_id, 'crops')`` -> the crops sub-folder.

    The directory is created by default because every caller is about to write
    into it; pass ``create=False`` when you only need the path (for example to
    delete the tree).
    """
    path = _media_root() / ANALYSES_DIR / str(analysis_id)
    if subdir:
        path = path / str(subdir)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def analysis_preview_path(analysis_id, filename='preview.jpg', create=True):
    """Absolute path for an analysis's annotated preview image."""
    return analysis_artifact_dir(analysis_id, create=create) / filename


def vehicle_crop_path(analysis_id, vehicle_id, extension='jpg', create=True):
    """Absolute path for one detected vehicle's cropped image."""
    directory = analysis_artifact_dir(analysis_id, CROPS_SUBDIR, create=create)
    return directory / f'{vehicle_id}.{extension.lstrip(".")}'


def smoke_mask_path(analysis_id, smoke_id, extension='png', create=True):
    """Absolute path for one smoke region's binary mask image."""
    directory = analysis_artifact_dir(analysis_id, MASKS_SUBDIR, create=create)
    return directory / f'{smoke_id}.{extension.lstrip(".")}'


def delete_analysis_artifacts(analysis_id):
    """
    Remove an analysis's entire artefact tree; silent when already gone.

    Note what this does **not** cover: the analysis's generated PDF lives in
    a different tree (``reports/<report_id>.pdf``), so a caller deleting an
    analysis needs :func:`delete_paths` over
    ``analysis.services.artifact_paths_for_analysis`` rather than this
    function on its own.  Kept because the benchmarking tool wants exactly
    "throw away the frames, keep the row".
    """
    shutil.rmtree(analysis_artifact_dir(analysis_id, create=False),
                  ignore_errors=True)


def delete_paths(paths):
    """
    Reclaim a batch of files and directory trees under ``MEDIA_ROOT``.

    Accepts the mixed list that deleting an analysis produces — artefact
    *directories* and report *files* — and dispatches per entry, so callers
    do not have to remember which is which.  Absent entries are not an
    error: the whole point of calling this is that the database rows which
    pointed at them are already gone, and a half-reclaimed tree from an
    earlier crash must not turn a 204 into a 500.

    Anything resolving outside ``MEDIA_ROOT`` is refused and logged rather
    than deleted.  Every path handed in today is derived from a database
    column (``GeneratedReport.report_path``) or from an id, and
    :func:`from_media_relative` already confines the former — but this
    function's whole job is ``rmtree``/``unlink``, so it does not take that
    on trust from its caller.

    Returns the number of entries that existed and were removed.
    """
    root = os.path.normpath(os.path.abspath(str(_media_root())))
    removed = 0

    for entry in paths or ():
        if entry is None:
            continue
        path = Path(entry)
        resolved = os.path.normpath(os.path.abspath(str(path)))
        if resolved != root and not resolved.startswith(root + os.sep):
            logger.error(
                'Refusing to delete %s: it is outside MEDIA_ROOT.', path,
            )
            continue

        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
                removed += 0 if path.exists() else 1
            elif path.is_symlink() or path.exists():
                path.unlink(missing_ok=True)
                removed += 1
        except OSError:
            # Best effort by design: leaking bytes is recoverable, failing
            # the delete request is not.
            logger.warning('Could not reclaim %s', path, exc_info=True)

    return removed


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def report_path(report_id, create=True):
    """
    Absolute path of a generated PDF report: ``<MEDIA_ROOT>/reports/<id>.pdf``.

    ``create`` refers to the parent directory, not the file itself — reportlab
    needs somewhere to write but will create the PDF on its own.
    """
    directory = _media_root() / REPORTS_DIR
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory / f'{report_id}.pdf'


# ---------------------------------------------------------------------------
# Absolute <-> relative conversion
# ---------------------------------------------------------------------------

def to_media_relative(path):
    """
    Convert an absolute path under ``MEDIA_ROOT`` to the relative, POSIX-style
    form stored in ``*_path`` database columns.

    Paths outside ``MEDIA_ROOT`` are returned unchanged (already relative, or
    somebody handed us something from elsewhere and knows what they are doing).
    """
    candidate = Path(path)
    try:
        return candidate.relative_to(_media_root()).as_posix()
    except ValueError:
        return candidate.as_posix()


def from_media_relative(relative_path):
    """
    Inverse of :func:`to_media_relative` — relative column value -> Path.

    The result is **confined to MEDIA_ROOT** (security review finding
    ASG-07).  ``MEDIA_ROOT / value`` on its own is not safe: pathlib's ``/``
    discards the left operand entirely when the right is absolute, so a
    stored ``/etc/passwd`` resolves to ``/etc/passwd``, and a stored
    ``../../etc/passwd`` walks straight out of the tree.

    Every caller passes a value read back from a database column
    (``GeneratedReport.report_path``, ``AnalysisResult.preview_path``, the
    stashed frame list), and today those columns are only ever written by
    trusted pipeline code — so this is defence in depth rather than a live
    hole.  It is worth having because the consumers hand the result straight
    to ``open()``: anything that ever lets a tainted value reach one of those
    columns (a future import path, a data migration, a mis-scoped admin form)
    would otherwise become an arbitrary-file-read.

    Raises:
        SuspiciousFileOperation: the value resolves outside ``MEDIA_ROOT``.
    """
    from django.core.exceptions import SuspiciousFileOperation

    root = _media_root()
    candidate = root / str(relative_path)

    # normpath, not resolve(): the target frequently does not exist yet (the
    # report renderer asks for its output path before writing it), and
    # resolve() would additionally follow symlinks out of the tree.
    root_normalised = os.path.normpath(os.path.abspath(str(root)))
    candidate_normalised = os.path.normpath(os.path.abspath(str(candidate)))

    if (candidate_normalised != root_normalised
            and not candidate_normalised.startswith(root_normalised + os.sep)):
        raise SuspiciousFileOperation(
            f'Refusing to resolve {relative_path!r}: it escapes MEDIA_ROOT.'
        )

    return Path(candidate_normalised)


def media_url(relative_path):
    """
    Public URL for a relative media path, or ``None`` when there is no path.

    Keeps ``MEDIA_URL`` joining in one place so serializers never hand-roll it.
    """
    if not relative_path:
        return None
    return posixpath.join(
        settings.MEDIA_URL.rstrip('/') + '/',
        str(relative_path).lstrip('/'),
    )


def _media_root():
    """``MEDIA_ROOT`` as a ``Path``, read lazily so tests can override it."""
    return Path(settings.MEDIA_ROOT)
