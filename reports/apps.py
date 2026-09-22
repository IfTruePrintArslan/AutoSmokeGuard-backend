from django.apps import AppConfig


class ReportsConfig(AppConfig):
    """App config for generated PDF emission reports."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'reports'
    verbose_name = 'Reports'
