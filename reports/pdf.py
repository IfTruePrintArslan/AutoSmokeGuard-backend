"""
Rendering for the "Smoke Analysis Report" PDF (UC-07).

Why ``reportlab.platypus`` rather than an HTML-to-PDF engine
--------------------------------------------------------------
``reportlab`` is already a hard dependency, needs no external binary (no
headless Chromium, no wkhtmltopdf), and its flowable model gives exact control
over the two things this report cannot get wrong: the 200-row cap on the
detections table (a long video must not produce a 400-page PDF) and correct
page numbering (``Page N of M``) before the total page count is known.

This module is intentionally free of Django view/request concerns and of any
database writes — :func:`build_report` takes already-loaded ORM objects
(``analysis.models.AnalysisResult`` plus its related ``DetectedVehicle`` /
``SmokeRegion`` rows) and a destination path, and returns the page count of
the file it just wrote.  All bookkeeping (creating/updating the
``GeneratedReport`` row) lives in :mod:`reports.services`, which is the only
caller.

Visual language mirrors the product UI (ink ``#0a0a0a``, rule ``#d4d4d4``,
muted text ``#6f6f6f``, and the shared low/moderate/high severity palette) so
the PDF reads as part of the same product rather than a generic export.  Two
built-in fonts only (Helvetica / Helvetica-Bold, plus the oblique variant for
the disclaimer): the product's real webfont ships as ``.woff2``, which
``reportlab`` cannot embed, so the report leans on layout, colour and spacing
to carry the brand instead of a matching typeface.
"""
import logging
import re
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from django.conf import settings as django_settings
from django.utils import timezone

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from common.storage import analysis_artifact_dir, from_media_relative

logger = logging.getLogger('asg.reports')

# ---------------------------------------------------------------------------
# Visual language — mirrors the product UI palette exactly.
# ---------------------------------------------------------------------------

INK = colors.HexColor('#0a0a0a')
RULE = colors.HexColor('#d4d4d4')
MUTED = colors.HexColor('#6f6f6f')
PANEL_BG = colors.HexColor('#fafafa')
ZEBRA_BG = colors.HexColor('#f5f5f5')
CHIP_NEUTRAL = colors.HexColor('#e5e5e5')

SEVERITY_COLORS = {
    'low': colors.HexColor('#4ade80'),
    'moderate': colors.HexColor('#fbbf24'),
    'high': colors.HexColor('#f87171'),
}
SEVERITY_ORDER = {'low': 1, 'moderate': 2, 'high': 3}

PAGE_SIZE = A4
MARGIN = 18 * mm
CONTENT_WIDTH = PAGE_SIZE[0] - 2 * MARGIN  # usable flowable width

#: "Up to 6 annotated frames" — the evidence section.
MAX_ANNOTATED_FRAMES = 6
#: A long video must never produce a 400-page PDF.
MAX_DETECTION_ROWS = 200
#: Long edge, in pixels, that every embedded frame is downscaled to.
MAX_IMAGE_EDGE_PX = 900

_FRAME_NUMBER_RE = re.compile(r'(\d+)')

#: Reserved key the analysis worker stashes pipeline-run facts (device,
#: segmenter mode, the annotated-frame list) under, inside the frozen
#: ``settings_snapshot`` JSON — mirrors ``analysis.services.RUNTIME_KEY``.
#: Duplicated as a literal (rather than imported) so this module never gains
#: a hard, import-time dependency on the analysis app's *services* layer —
#: only its already-final ``models`` module is depended on elsewhere, per
#: the same lazy-coupling principle used for ``analysis.serializers``.
_RUNTIME_SNAPSHOT_KEY = '_runtime'


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def _esc(text):
    """XML-escape arbitrary text before it goes inside a ``Paragraph``."""
    return _xml_escape(str(text if text is not None else ''))


def _truncate(text, limit=52):
    """Truncate long filenames with an ellipsis rather than overflow a cell."""
    text = text or ''
    if len(text) <= limit:
        return _esc(text)
    return _esc(text[:max(0, limit - 1)] + '…')


def _fmt_dt(value):
    """``2026-09-21 14:03:11`` in the local timezone, or an em dash."""
    if not value:
        return '—'
    local = timezone.localtime(value) if timezone.is_aware(value) else value
    return local.strftime('%Y-%m-%d %H:%M:%S')


def _fmt_pct(value):
    """``0.913`` -> ``91%``."""
    try:
        return f'{float(value) * 100:.0f}%'
    except (TypeError, ValueError):
        return '—'


def _fmt_seconds(value):
    """``12.4`` -> ``12.4s``, or an em dash for a missing value."""
    try:
        return f'{float(value):.1f}s'
    except (TypeError, ValueError):
        return '—'


def _dominant_severity(severities):
    """The worst severity present in *severities*, or ``None`` if empty."""
    best = None
    for severity in severities:
        if best is None or SEVERITY_ORDER.get(severity, 0) > SEVERITY_ORDER.get(best, 0):
            best = severity
    return best


# ---------------------------------------------------------------------------
# Paragraph styles
# ---------------------------------------------------------------------------

def _styles():
    """Fresh ``ParagraphStyle`` set for one render pass (styles are cheap)."""
    return {
        'wordmark': ParagraphStyle(
            'wordmark', fontName='Helvetica-Bold', fontSize=9.5,
            textColor=MUTED, leading=11,
        ),
        'title': ParagraphStyle(
            'title', fontName='Helvetica-Bold', fontSize=20,
            textColor=INK, leading=23, spaceBefore=1,
        ),
        'h2': ParagraphStyle(
            'h2', fontName='Helvetica-Bold', fontSize=12.5,
            textColor=INK, leading=15, spaceBefore=1,
        ),
        'meta': ParagraphStyle(
            'meta', fontName='Helvetica', fontSize=9,
            textColor=MUTED, leading=12.5,
        ),
        'kv_label': ParagraphStyle(
            'kv_label', fontName='Helvetica', fontSize=8.5,
            textColor=MUTED, leading=11,
        ),
        'kv_value': ParagraphStyle(
            'kv_value', fontName='Helvetica-Bold', fontSize=9.5,
            textColor=INK, leading=12,
        ),
        'body': ParagraphStyle(
            'body', fontName='Helvetica', fontSize=9.5,
            textColor=INK, leading=13,
        ),
        'caption': ParagraphStyle(
            'caption', fontName='Helvetica', fontSize=8,
            textColor=MUTED, leading=10.5,
        ),
        'muted': ParagraphStyle(
            'muted', fontName='Helvetica-Oblique', fontSize=9,
            textColor=MUTED, leading=12.5,
        ),
        'muted_center': ParagraphStyle(
            'muted_center', fontName='Helvetica-Oblique', fontSize=9,
            textColor=MUTED, leading=12.5, alignment=TA_CENTER,
        ),
        'panel': ParagraphStyle(
            'panel', fontName='Helvetica', fontSize=10.5,
            textColor=INK, leading=14, alignment=TA_CENTER,
        ),
        'cell': ParagraphStyle(
            'cell', fontName='Helvetica', fontSize=8.5,
            textColor=INK, leading=11, alignment=TA_CENTER,
        ),
        'disclaimer': ParagraphStyle(
            'disclaimer', fontName='Helvetica', fontSize=8,
            textColor=MUTED, leading=11.5,
        ),
    }


# ---------------------------------------------------------------------------
# Small drawn primitives (chips, bars) — real vector shapes, not images.
# ---------------------------------------------------------------------------

def _chip_drawing(severity, width=28 * mm, height=6 * mm, font_size=8):
    """A rounded severity "chip", in the product's palette."""
    fill = SEVERITY_COLORS.get(severity, CHIP_NEUTRAL)
    label = severity.upper() if severity else 'N/A'
    drawing = Drawing(width, height)
    drawing.add(Rect(
        0, 0, width, height, rx=height / 2, ry=height / 2,
        fillColor=fill, strokeColor=None,
    ))
    drawing.add(String(
        width / 2, height / 2 - font_size * 0.35, label,
        fontName='Helvetica-Bold', fontSize=font_size,
        fillColor=INK, textAnchor='middle',
    ))
    return drawing


def _severity_bar_drawing(counts, width=CONTENT_WIDTH, height=8 * mm):
    """A horizontal stacked bar: low/moderate/high, proportional to count."""
    total = sum(counts.values())
    drawing = Drawing(width, height)
    if total <= 0:
        drawing.add(Rect(
            0, 0, width, height, fillColor=PANEL_BG,
            strokeColor=RULE, strokeWidth=0.5,
        ))
        return drawing
    x = 0.0
    for key in ('low', 'moderate', 'high'):
        share = width * (counts.get(key, 0) / total)
        if share > 0:
            drawing.add(Rect(x, 0, share, height, fillColor=SEVERITY_COLORS[key], strokeColor=None))
            x += share
    return drawing


# ---------------------------------------------------------------------------
# Image loading — downscale before embedding (NFR: keep the PDF small & fast).
# ---------------------------------------------------------------------------

def _load_scaled_image(absolute_path, max_width_pt, max_edge_px=MAX_IMAGE_EDGE_PX):
    """
    Open *absolute_path* with PIL, downscale so its long edge is at most
    *max_edge_px* pixels, and return a ``reportlab`` ``Image`` flowable sized
    to *max_width_pt* wide with the aspect ratio preserved.

    Raises whatever PIL/``OSError`` raises on a missing or corrupt file —
    every caller decides individually how to degrade (a caption-only note
    beats aborting the whole report).
    """
    from PIL import Image as PILImage

    with PILImage.open(absolute_path) as source:
        source.load()  # force-read now, so a truncated file fails here.
        image = source.convert('RGB')

    width_px, height_px = image.size
    longest = max(width_px, height_px)
    if longest > max_edge_px:
        scale = max_edge_px / float(longest)
        image = image.resize(
            (max(1, round(width_px * scale)), max(1, round(height_px * scale))),
            PILImage.LANCZOS,
        )

    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=82)
    buffer.seek(0)

    out_width, out_height = image.size
    height_pt = max_width_pt * (out_height / float(out_width))
    return RLImage(buffer, width=max_width_pt, height=height_pt)


# ---------------------------------------------------------------------------
# Section builders — each returns a list of flowables.
# ---------------------------------------------------------------------------

def _cover_thumbnail(analysis, max_width=42 * mm):
    """A small preview thumbnail for the cover, or ``None`` if unavailable."""
    if not analysis.preview_path:
        return None
    try:
        absolute = from_media_relative(analysis.preview_path)
        return _load_scaled_image(absolute, max_width_pt=max_width)
    except Exception:
        logger.warning(
            'Preview image unavailable for analysis %s; omitting from cover.',
            analysis.analysis_id, exc_info=True,
        )
        return None


def _cover_section(analysis, report_id, generated_at):
    styles = _styles()
    user = analysis.user
    analyst_name = _esc(user.full_name or user.email)
    analyst_email = _esc(user.email)

    left = [
        Paragraph('AUTOSMOKEGUARD', styles['wordmark']),
        Paragraph('Smoke Analysis Report', styles['title']),
        Spacer(1, 2.5 * mm),
        Paragraph(f'Report ID: {_esc(str(report_id)[:8])}', styles['meta']),
        Paragraph(f'Generated: {_esc(_fmt_dt(generated_at))}', styles['meta']),
        Paragraph(f'Analyst: {analyst_name} &lt;{analyst_email}&gt;', styles['meta']),
    ]

    right = [_chip_drawing(analysis.overall_severity or None, width=32 * mm, height=8 * mm, font_size=9.5)]
    thumbnail = _cover_thumbnail(analysis)
    if thumbnail is not None:
        right.append(Spacer(1, 3 * mm))
        right.append(thumbnail)

    table = Table([[left, right]], colWidths=[CONTENT_WIDTH - 44 * mm, 44 * mm])
    table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    return [table, Spacer(1, 3 * mm), HRFlowable(width='100%', thickness=0.75, color=RULE)]


def _summary_section(analysis, media):
    styles = _styles()
    resolution = f'{media.width}x{media.height}' if media.width and media.height else '—'
    duration = _fmt_seconds(media.duration_seconds) if media.duration_seconds else '—'
    elapsed = analysis.duration_seconds

    pairs = [
        ('Source file', _truncate(media.filename)),
        ('Media type', _esc((media.media_type or '').title() or '—')),
        ('Resolution', resolution),
        ('Duration', duration),
        ('Uploaded', _esc(_fmt_dt(media.upload_timestamp))),
        ('Analysis started', _esc(_fmt_dt(analysis.start_time))),
        ('Analysis ended', _esc(_fmt_dt(analysis.end_time))),
        ('Elapsed', _esc(_fmt_seconds(elapsed) if elapsed is not None else '—')),
        ('Frames processed', str(analysis.frames_processed)),
        ('Total vehicles', str(analysis.total_vehicles)),
        ('Total smoke regions', str(analysis.total_smoke)),
        ('Avg. confidence', _fmt_pct(analysis.avg_confidence)),
    ]

    rows = []
    for i in range(0, len(pairs), 2):
        chunk = pairs[i:i + 2]
        row = []
        for label, value in chunk:
            row.append(Paragraph(label, styles['kv_label']))
            row.append(Paragraph(str(value), styles['kv_value']))
        if len(chunk) == 1:
            row.extend(['', ''])
        rows.append(row)

    col = CONTENT_WIDTH / 4.0
    table = Table(rows, colWidths=[col, col, col, col], hAlign='LEFT')
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.4, RULE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))

    severity_row = Table(
        [[Paragraph('Overall severity', styles['kv_label']),
          _chip_drawing(analysis.overall_severity or None, width=col - 6, height=6 * mm)]],
        colWidths=[col, col],
    )
    severity_row.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
    ]))

    return [
        Paragraph('Summary', styles['h2']), Spacer(1, 2 * mm),
        table, Spacer(1, 2 * mm), severity_row,
    ]


def _severity_section(analysis):
    styles = _styles()
    raw_counts = analysis.severity_counts or {}
    counts = {key: int(raw_counts.get(key, 0) or 0) for key in ('low', 'moderate', 'high')}

    bar = _severity_bar_drawing(counts)

    swatch_size = 3.5 * mm
    legend_cells = []
    for key in ('low', 'moderate', 'high'):
        swatch = Drawing(swatch_size, swatch_size)
        swatch.add(Rect(0, 0, swatch_size, swatch_size, fillColor=SEVERITY_COLORS[key], strokeColor=None))
        legend_cells.append(swatch)
        legend_cells.append(Paragraph(key.title(), styles['kv_label']))
        legend_cells.append(Paragraph(str(counts[key]), styles['kv_value']))

    legend_col = CONTENT_WIDTH / 9.0
    legend = Table([legend_cells], colWidths=[legend_col] * 9, hAlign='LEFT')
    legend.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
    ]))

    return [
        Paragraph('Severity Distribution', styles['h2']), Spacer(1, 2 * mm),
        bar, Spacer(1, 3 * mm), legend,
    ]


def _annotated_frame_paths(analysis, limit):
    """
    Absolute paths of the frames to embed, preferring the worker's own
    ``settings_snapshot['_runtime']['annotated_frames']`` list (the exact
    frames the pipeline judged "interesting", stored MEDIA_ROOT-relative) and
    falling back to a directory listing of ``analyses/<id>/frames/`` when
    that stash is absent — the normal case for a report built straight from
    ORM fixtures (the test suite) rather than a real pipeline run.
    """
    runtime = (analysis.settings_snapshot or {}).get(_RUNTIME_SNAPSHOT_KEY)
    if isinstance(runtime, dict):
        stashed = [f for f in (runtime.get('annotated_frames') or []) if f]
        if stashed:
            return [from_media_relative(relative) for relative in stashed[:limit]]

    frames_dir = analysis_artifact_dir(analysis.analysis_id, 'frames', create=False)
    if not frames_dir.is_dir():
        return []
    return sorted(
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )[:limit]


def _gather_annotated_frames(analysis, limit=MAX_ANNOTATED_FRAMES):
    """
    Up to *limit* annotated frame images for this analysis, each paired with
    the best metadata we can recover: the frame number (parsed from the
    filename), a timestamp and the worst severity present, both matched via
    any ``DetectedVehicle`` rows that share that frame number.

    Returns ``[]`` when no frames can be found by any means — must not raise.
    """
    files = _annotated_frame_paths(analysis, limit)
    if not files:
        return []

    vehicles_by_frame = {}
    for vehicle in analysis.vehicles.prefetch_related('smoke_regions').all():
        if vehicle.frame_number is not None:
            vehicles_by_frame.setdefault(vehicle.frame_number, []).append(vehicle)

    items = []
    for path in files:
        match = _FRAME_NUMBER_RE.search(path.stem)
        frame_number = int(match.group(1)) if match else None
        vehicles_here = vehicles_by_frame.get(frame_number, [])
        timestamp = next(
            (v.timestamp_seconds for v in vehicles_here if v.timestamp_seconds is not None),
            None,
        )
        severities = [
            region.severity
            for vehicle in vehicles_here
            for region in vehicle.smoke_regions.all()
        ]
        items.append({
            'path': path,
            'frame_number': frame_number,
            'timestamp': timestamp,
            'severity': _dominant_severity(severities),
        })
    return items


def _evidence_section(analysis):
    styles = _styles()
    heading = [Paragraph('Annotated Evidence', styles['h2']), Spacer(1, 2 * mm)]

    try:
        frames = _gather_annotated_frames(analysis)
    except Exception:
        logger.warning(
            'Could not list annotated frames for analysis %s.',
            analysis.analysis_id, exc_info=True,
        )
        frames = []

    if not frames:
        return heading + [Paragraph('No annotated frames are available for this analysis.', styles['muted'])]

    cell_width = (CONTENT_WIDTH - 4 * mm) / 2.0
    grid_rows = []
    current_row = []
    for item in frames:
        try:
            image = _load_scaled_image(item['path'], max_width_pt=cell_width)
        except Exception:
            logger.warning('Skipping unreadable annotated frame %s.', item['path'], exc_info=True)
            cell = [Paragraph(f"Frame unavailable: {_esc(item['path'].name)}", styles['muted_center'])]
        else:
            bits = []
            if item['frame_number'] is not None:
                bits.append(f"Frame {item['frame_number']}")
            if item['timestamp'] is not None:
                bits.append(_fmt_seconds(item['timestamp']))
            bits.append((item['severity'] or 'no smoke').title())
            cell = [image, Spacer(1, 1.5 * mm), Paragraph(_esc(' · '.join(bits)), styles['caption'])]
        current_row.append(cell)
        if len(current_row) == 2:
            grid_rows.append(current_row)
            current_row = []
    if current_row:
        current_row.append('')
        grid_rows.append(current_row)

    table = Table(grid_rows, colWidths=[cell_width + 4 * mm, cell_width + 4 * mm], hAlign='LEFT')
    table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
    ]))
    return heading + [table]


def _detections_section(analysis):
    styles = _styles()
    heading = [Paragraph('Detections', styles['h2']), Spacer(1, 2 * mm)]

    total = analysis.vehicles.count()
    if total == 0:
        panel = Table(
            [[Paragraph('No vehicles were detected in this media.', styles['panel'])]],
            colWidths=[CONTENT_WIDTH],
        )
        panel.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 0.6, RULE),
            ('BACKGROUND', (0, 0), (-1, -1), PANEL_BG),
            ('TOPPADDING', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ]))
        return heading + [panel]

    vehicles = list(
        analysis.vehicles.prefetch_related('smoke_regions').all()[:MAX_DETECTION_ROWS]
    )

    header = ['#', 'Vehicle', 'Frame', 'Time', 'Conf.', 'Severity', 'Intensity', 'Area']
    rows = [header]
    for index, vehicle in enumerate(vehicles, start=1):
        smoke_regions = list(vehicle.smoke_regions.all())
        primary = smoke_regions[0] if smoke_regions else None
        rows.append([
            str(index),
            _esc((vehicle.vehicle_type or '').title()),
            str(vehicle.frame_number) if vehicle.frame_number is not None else '—',
            _fmt_seconds(vehicle.timestamp_seconds) if vehicle.timestamp_seconds is not None else '—',
            _fmt_pct(vehicle.confidence),
            _chip_drawing(primary.severity, width=20 * mm, height=5 * mm, font_size=7) if primary
                else Paragraph('none', styles['cell']),
            f'{primary.intensity:.2f}' if primary else '—',
            _fmt_pct(primary.area_ratio) if primary else '—',
        ])

    col_widths = [
        0.05, 0.16, 0.10, 0.11, 0.10, 0.20, 0.14, 0.14,
    ]
    col_widths = [CONTENT_WIDTH * w for w in col_widths]

    table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign='LEFT')
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), INK),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.3, RULE),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]
    for row_index in range(1, len(rows)):
        if row_index % 2 == 0:
            style.append(('BACKGROUND', (0, row_index), (-1, row_index), ZEBRA_BG))
    table.setStyle(TableStyle(style))

    blocks = heading + [table]
    if total > MAX_DETECTION_ROWS:
        blocks.append(Spacer(1, 1.5 * mm))
        blocks.append(Paragraph(
            f'…and {total - MAX_DETECTION_ROWS} more detection(s), not shown.',
            styles['muted'],
        ))
    if analysis.total_smoke == 0:
        blocks.append(Spacer(1, 1.5 * mm))
        blocks.append(Paragraph(
            'No smoke was detected on any vehicle in this analysis.', styles['muted'],
        ))
    return blocks


def _runtime_info(analysis):
    """
    ``(device, segmenter_mode)`` for the methodology section.

    Preferred source is the worker's own stash at ``settings_snapshot
    ['_runtime']`` — the exact device and segmenter mode *that analysis run*
    actually used, which is what makes the section reproducible.  When that is
    absent (an ORM-only fixture, or an older row predating the stash), this
    falls back to a best-effort guess of the *current* runtime instead; both
    lookups are wrapped defensively so a broken/absent ``mlcore``/``torch``
    install degrades the methodology section rather than crashing the report.
    """
    runtime = (analysis.settings_snapshot or {}).get(_RUNTIME_SNAPSHOT_KEY)
    if isinstance(runtime, dict):
        device = runtime.get('device') or ''
        segmenter_mode = runtime.get('segmenter_mode') or ''
        if device and segmenter_mode:
            return device, segmenter_mode

    device = 'cpu'
    try:
        from mlcore.config import device_string, get_device
        device = device_string(get_device())
    except Exception:
        logger.warning('Could not resolve the compute device for the report.', exc_info=True)

    segmenter_mode = 'classical'
    try:
        weights = Path(django_settings.ASG['SEGMENTER_WEIGHTS'])
        if weights.is_file() and weights.stat().st_size > 0:
            segmenter_mode = 'unet'
    except Exception:
        logger.warning('Could not resolve the segmenter mode for the report.', exc_info=True)

    return device, segmenter_mode


def _methodology_section(analysis):
    styles = _styles()
    device, segmenter_mode = _runtime_info(analysis)
    snapshot = analysis.settings_snapshot or {}

    labelled = [
        ('confidence_threshold', 'Detector confidence threshold'),
        ('smoke_mask_threshold', 'Smoke mask threshold'),
        ('severity_low_max', 'Low / moderate boundary'),
        ('severity_moderate_max', 'Moderate / high boundary'),
        ('frame_sample_rate', 'Frame sample rate'),
        ('max_video_seconds', 'Max video length (s)'),
    ]
    rows = [['Setting', 'Value']]
    for key, label in labelled:
        if key in snapshot:
            rows.append([label, _esc(snapshot[key])])
    rows.append(['Vehicle detector', 'YOLO (Ultralytics)'])
    rows.append(['Smoke segmenter', 'SmokeUNet'])
    rows.append(['Segmenter mode', _esc(segmenter_mode)])
    rows.append(['Compute device', _esc(device)])

    table = Table(rows, colWidths=[CONTENT_WIDTH * 0.42, CONTENT_WIDTH * 0.58],
                  repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), INK),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.3, RULE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    return [Paragraph('Methodology &amp; Thresholds', styles['h2']), Spacer(1, 2 * mm), table]


def _disclaimer_section():
    styles = _styles()
    text = (
        'This report was generated automatically by AutoSmokeGuard’s vehicle '
        'detection and smoke segmentation pipeline, without human review. It is '
        'intended as supporting evidence for further investigation, not as a '
        'legal certification of an emissions violation. Severity bands (low / '
        'moderate / high) are derived from the confidence and intensity '
        'thresholds configured in the system at the time of analysis — see '
        '“Methodology &amp; Thresholds” above for the exact values used.'
    )
    return [
        Spacer(1, 4 * mm),
        HRFlowable(width='100%', thickness=0.5, color=RULE),
        Spacer(1, 2 * mm),
        Paragraph(text, styles['disclaimer']),
    ]


# ---------------------------------------------------------------------------
# Header / footer — the "standard canvas-counting trick": buffer every page,
# then redraw with the true page count once it is known, on ``save()``.
# ---------------------------------------------------------------------------

class _NumberedCanvas(pdfcanvas.Canvas):
    """A canvas that can print "Page N of M" with a correct M."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_chrome(total_pages)
            super().showPage()
        super().save()

    def _draw_chrome(self, total_pages):
        width, height = PAGE_SIZE

        # Header rule, just inside the top margin gutter.
        header_y = height - MARGIN + 6
        self.setStrokeColor(RULE)
        self.setLineWidth(0.75)
        self.line(MARGIN, header_y, width - MARGIN, header_y)
        self.setFont('Helvetica', 7.5)
        self.setFillColor(MUTED)
        self.drawString(MARGIN, header_y + 3, 'AutoSmokeGuard — Smoke Analysis Report')

        # Footer rule plus "Page N of M", just inside the bottom margin gutter.
        footer_y = MARGIN - 10
        self.setStrokeColor(RULE)
        self.setLineWidth(0.5)
        self.line(MARGIN, footer_y + 9, width - MARGIN, footer_y + 9)
        self.setFont('Helvetica', 7.5)
        self.setFillColor(MUTED)
        self.drawString(MARGIN, footer_y, 'Automatically generated — supporting evidence, not a certification.')
        self.drawRightString(width - MARGIN, footer_y, f'Page {self._pageNumber} of {total_pages}')


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def count_pdf_pages(path):
    """
    Count ``/Type /Page`` objects in a raw PDF, minus the one ``/Type
    /Pages`` container object (whose dictionary name is a superstring match).

    Used as the authoritative ``page_count`` because it needs no PDF-parsing
    dependency (none is installed) beyond the bytes ``reportlab`` itself just
    wrote.
    """
    raw = Path(path).read_bytes()
    return max(raw.count(b'/Type /Page') - raw.count(b'/Type /Pages'), 0)


def build_report(analysis, report_id, output_path):
    """
    Render the full "Smoke Analysis Report" PDF for *analysis* to
    *output_path* and return its page count.

    ``report_id`` is accepted separately from *analysis* (rather than read
    off a ``GeneratedReport`` row) because :mod:`reports.services` calls this
    before the row exists for a first-time generation — the id it hands in is
    the one the row will be saved under.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    media = analysis.media
    generated_at = timezone.now()

    story = []
    story.extend(_cover_section(analysis, report_id, generated_at))
    story.append(Spacer(1, 4 * mm))
    story.extend(_summary_section(analysis, media))
    story.append(Spacer(1, 4 * mm))
    story.extend(_severity_section(analysis))
    story.append(Spacer(1, 4 * mm))
    story.extend(_evidence_section(analysis))
    story.append(Spacer(1, 4 * mm))
    story.extend(_detections_section(analysis))
    story.append(Spacer(1, 4 * mm))
    story.extend(_methodology_section(analysis))
    story.extend(_disclaimer_section())

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=PAGE_SIZE,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title='Smoke Analysis Report — AutoSmokeGuard',
        author='AutoSmokeGuard',
    )
    doc.build(story, canvasmaker=_NumberedCanvas)

    return count_pdf_pages(output_path)
