"""
Django admin registration for the system-configuration singleton.

The admin is locked down to match the model: you may edit the one row, but you
may not add a second or delete the only one.  ``updated_by`` is stamped from
the logged-in administrator automatically so the audit trail cannot be faked
by leaving the field blank.
"""
from django.contrib import admin

from .models import SystemSetting


@admin.register(SystemSetting)
class SystemSettingAdmin(admin.ModelAdmin):
    """Single-row editor for the runtime configuration."""

    list_display = ('__str__', 'max_upload_mb', 'confidence_threshold',
                    'smoke_mask_threshold', 'frame_sample_rate',
                    'auto_generate_pdf', 'updated_at', 'updated_by')
    list_filter = ('auto_generate_pdf',)
    search_fields = ('allowed_image_formats', 'allowed_video_formats')
    readonly_fields = ('id', 'updated_at', 'updated_by')
    fieldsets = (
        ('Upload policy', {
            'fields': ('allowed_image_formats', 'allowed_video_formats',
                       'max_upload_mb', 'max_video_seconds'),
        }),
        ('Model thresholds', {
            'fields': ('confidence_threshold', 'smoke_mask_threshold'),
        }),
        ('Severity banding', {
            'description': 'Intensity ≤ low_max is "low", ≤ moderate_max is '
                           '"moderate", anything higher is "high".',
            'fields': ('severity_low_max', 'severity_moderate_max'),
        }),
        ('Pipeline behaviour', {
            'fields': ('frame_sample_rate', 'auto_generate_pdf'),
        }),
        ('Audit', {'fields': ('id', 'updated_at', 'updated_by')}),
    )

    def has_add_permission(self, request):
        """Allow "add" only while the singleton does not exist yet."""
        return not SystemSetting.objects.exists()

    def has_delete_permission(self, request, obj=None):
        """The configuration row is never deletable."""
        return False

    def save_model(self, request, obj, form, change):
        """Stamp the acting administrator onto the audit column."""
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)
