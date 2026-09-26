from django.apps import AppConfig


class UploadsConfig(AppConfig):
    """App config for user-uploaded traffic media."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'uploads'
    verbose_name = 'Uploads'
