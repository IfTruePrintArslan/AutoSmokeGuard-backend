from django.apps import AppConfig


class AccountsConfig(AppConfig):
    """App config for authentication and user profiles."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'accounts'
    verbose_name = 'Accounts'
