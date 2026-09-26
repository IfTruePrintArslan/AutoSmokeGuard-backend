from django.apps import AppConfig


class SystemConfigConfig(AppConfig):
    """
    App config for the runtime system-configuration singleton.

    ``label`` is pinned to ``'sysconfig'`` even though the package is called
    ``system_config`` — see the module docstring in ``system_config/__init__``
    for why the directory cannot use the stdlib-colliding name.
    """

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'system_config'
    label = 'sysconfig'
    verbose_name = 'System configuration'
