"""Django admin registration for generated reports."""
from django.contrib import admin

from .models import GeneratedReport


@admin.register(GeneratedReport)
class GeneratedReportAdmin(admin.ModelAdmin):
    """Audit view of every PDF the system has produced."""

    list_display = ('report_id', 'analysis', 'owner_email', 'page_count',
                    'size_kb', 'generated_at')
    list_filter = ('generated_at', 'page_count')
    search_fields = ('report_id', 'analysis__analysis_id',
                     'analysis__user__email', 'report_path')
    ordering = ('-generated_at',)
    date_hierarchy = 'generated_at'
    list_select_related = ('analysis', 'analysis__user')
    autocomplete_fields = ('analysis',)
    readonly_fields = ('report_id', 'report_path', 'generated_at',
                       'page_count', 'file_size_bytes')

    @admin.display(description='Owner', ordering='analysis__user__email')
    def owner_email(self, obj):
        return obj.analysis.user.email

    @admin.display(description='Size (KB)', ordering='file_size_bytes')
    def size_kb(self, obj):
        return obj.size_kb
