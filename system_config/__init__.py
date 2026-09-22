"""
Runtime system configuration (UC-09).

.. note:: **Why the directory is called ``system_config`` and the Django app
   label is ``sysconfig``.**

   Django apps live at the project root, which Python puts first on
   ``sys.path``.  A package literally named ``sysconfig`` there shadows the
   *standard library* ``sysconfig`` module, and ``zoneinfo`` imports that
   during start-up — so naming the folder ``sysconfig`` makes Django refuse to
   boot at all with ``AttributeError: module 'sysconfig' has no attribute
   'get_config_var'``.

   The folder is therefore ``system_config``, while
   :class:`system_config.apps.SystemConfigConfig` pins ``label = 'sysconfig'``.
   Everything SDS-facing keeps the documented name: the app label, the
   ``sysconfig_system_setting`` table, model references written as
   ``'sysconfig.SystemSetting'``, and management commands such as
   ``manage.py makemigrations sysconfig``.  Only the ``import`` path differs.
"""
