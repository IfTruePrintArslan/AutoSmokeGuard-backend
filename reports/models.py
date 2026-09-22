"""
Generated PDF emission reports.

A report is the deliverable a user actually takes away: the annotated preview,
the per-vehicle severity table and the summary statistics, rendered by
reportlab into a single downloadable file.

The relationship to ``AnalysisResult`` is one-to-one.  Re-generating a report
for the same analysis overwrites the existing row and file rather than piling
up copies, which keeps "download my report" unambiguous and means deleting an
analysis disposes of exactly one PDF.

Views and serializers are owned by the reports API agent; this module is the
schema only.
"""
import uuid

from django.db import models


class GeneratedReport(models.Model):
    """The PDF produced for one completed analysis."""

    report_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name='report ID',
    )
    analysis = models.OneToOneField(
        'analysis.AnalysisResult',
        related_name='report',
        on_delete=models.CASCADE,
    )

    report_path = models.CharField(
        max_length=500,
        help_text='PDF location, relative to MEDIA_ROOT.',
    )
    generated_at = models.DateTimeField(auto_now_add=True)
    page_count = models.IntegerField(default=0)
    file_size_bytes = models.BigIntegerField(default=0)

    class Meta:
        db_table = 'reports_generated_report'
        ordering = ('-generated_at',)
        indexes = [
            models.Index(fields=['generated_at'], name='report_generated_idx'),
        ]
        verbose_name = 'generated report'
        verbose_name_plural = 'generated reports'

    def __str__(self):
        return f'Report {self.report_id} for analysis {self.analysis_id}'

    @property
    def size_kb(self):
        """File size in kilobytes, rounded for display."""
        return round((self.file_size_bytes or 0) / 1024, 1)

    @property
    def download_filename(self):
        """Suggested ``Content-Disposition`` filename for the download."""
        return f'autosmokeguard-report-{self.analysis_id}.pdf'
