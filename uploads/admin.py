"""Django admin registration for uploaded media."""
from django.contrib import admin

from .models import UploadedMedia


@admin.register(UploadedMedia)
class UploadedMediaAdmin(admin.ModelAdmin):
    """Browse what users have uploaded, without exposing an edit path."""

    list_display = ('filename', 'user', 'media_type', 'format', 'size_mb',
                    'resolution', 'duration_seconds', 'upload_timestamp')
    list_filter = ('media_type', 'format', 'upload_timestamp')
    search_fields = ('filename', 'media_id', 'checksum', 'user__email')
    ordering = ('-upload_timestamp',)
    date_hierarchy = 'upload_timestamp'
    list_select_related = ('user',)
    autocomplete_fields = ('user',)
    readonly_fields = ('media_id', 'upload_timestamp', 'file_path',
                       'size_bytes', 'checksum', 'width', 'height',
                       'duration_seconds')
    fieldsets = (
        (None, {'fields': ('media_id', 'user', 'filename', 'file')}),
        ('Detected properties', {
            'fields': ('media_type', 'format', 'size_bytes', 'width', 'height',
                       'duration_seconds', 'checksum'),
        }),
        ('Storage', {'fields': ('file_path', 'upload_timestamp')}),
    )

    @admin.display(description='Size (MB)', ordering='size_bytes')
    def size_mb(self, obj):
        return obj.size_mb
