from django.apps import AppConfig


class CommonConfig(AppConfig):
    """App config for the shared utilities package."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'common'
    verbose_name = 'Common'
