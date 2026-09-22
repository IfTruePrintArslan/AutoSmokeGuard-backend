"""
Django admin registration for analysis results.

Results are written by the ML worker, never by hand, so everything derived is
read-only; the admin exists for support and debugging ("why did this run
fail?"), not for data entry.
"""
from django.contrib import admin

from .models import AnalysisResult, DetectedVehicle, SmokeRegion


class SmokeRegionInline(admin.TabularInline):
    """Smoke regions shown inline on their parent vehicle."""

    model = SmokeRegion
    extra = 0
    can_delete = False
    fields = ('smoke_id', 'severity', 'intensity', 'confidence', 'area_ratio',
              'opacity', 'mask_path')
    readonly_fields = fields
    show_change_link = True

    def has_add_permission(self, request, obj=None):
        return False


class DetectedVehicleInline(admin.TabularInline):
    """Detections shown inline on their parent analysis."""

    model = DetectedVehicle
    extra = 0
    can_delete = False
    fields = ('vehicle_id', 'vehicle_type', 'confidence', 'frame_number',
              'timestamp_seconds', 'bounding_box')
    readonly_fields = fields
    show_change_link = True

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    """Job board plus results summary for every pipeline run."""

    list_display = ('analysis_id', 'user', 'status', 'progress', 'stage',
                    'total_vehicles', 'total_smoke', 'overall_severity',
                    'created_at')
    list_filter = ('status', 'overall_severity', 'created_at')
    search_fields = ('analysis_id', 'user__email', 'media__filename',
                     'error_message')
    ordering = ('-created_at',)
    date_hierarchy = 'created_at'
    list_select_related = ('user', 'media')
    autocomplete_fields = ('user', 'media')
    inlines = (DetectedVehicleInline,)
    readonly_fields = ('analysis_id', 'created_at', 'start_time', 'end_time',
                       'duration_display', 'total_vehicles', 'total_smoke',
                       'frames_processed', 'avg_confidence', 'severity_counts',
                       'settings_snapshot', 'preview_path')
    fieldsets = (
        (None, {'fields': ('analysis_id', 'media', 'user')}),
        ('Job state', {
            'fields': ('status', 'progress', 'stage', 'start_time',
                       'end_time', 'duration_display', 'error_message'),
        }),
        ('Results', {
            'fields': ('total_vehicles', 'total_smoke', 'frames_processed',
                       'avg_confidence', 'overall_severity',
                       'severity_counts', 'preview_path'),
        }),
        ('Provenance', {'fields': ('settings_snapshot', 'created_at')}),
    )

    @admin.display(description='Duration (s)')
    def duration_display(self, obj):
        duration = obj.duration_seconds
        return '—' if duration is None else f'{duration:.2f}'


@admin.register(DetectedVehicle)
class DetectedVehicleAdmin(admin.ModelAdmin):
    """Flat view of every detection, for cross-run queries."""

    list_display = ('vehicle_id', 'analysis', 'vehicle_type', 'confidence',
                    'frame_number', 'timestamp_seconds')
    list_filter = ('vehicle_type',)
    search_fields = ('vehicle_id', 'analysis__analysis_id')
    ordering = ('-confidence',)
    list_select_related = ('analysis',)
    autocomplete_fields = ('analysis',)
    inlines = (SmokeRegionInline,)
    readonly_fields = ('vehicle_id', 'bounding_box', 'crop_path')


@admin.register(SmokeRegion)
class SmokeRegionAdmin(admin.ModelAdmin):
    """Flat view of every smoke region, for threshold tuning."""

    list_display = ('smoke_id', 'vehicle', 'severity', 'intensity',
                    'confidence', 'area_ratio', 'opacity')
    list_filter = ('severity',)
    search_fields = ('smoke_id', 'vehicle__vehicle_id',
                     'vehicle__analysis__analysis_id')
    ordering = ('-intensity',)
    list_select_related = ('vehicle',)
    autocomplete_fields = ('vehicle',)
    readonly_fields = ('smoke_id', 'mask_path')
