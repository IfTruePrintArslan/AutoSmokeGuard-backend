from django.apps import AppConfig


class HealthConfig(AppConfig):
    """App config for the liveness/readiness probe."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'health'
    verbose_name = 'Health'
