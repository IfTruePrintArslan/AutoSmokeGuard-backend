#!/usr/bin/env python
"""
Seed a database with the accounts and configuration the project assumes exist.

Run it from the backend directory::

    python tools/seed.py

It is **idempotent**: running it on an already-seeded database changes nothing
and reports ``unchanged`` for every record.  That matters because it is meant
to be safe to wire into container start-up and into CI, not just run once by
hand after ``migrate``.

What it creates
---------------
* ``admin@autosmokeguard.local`` / ``Admin@12345`` — superuser, role ``admin``.
  Used for the Django admin and for exercising the UC-09 configuration APIs.
* ``demo@autosmokeguard.local`` / ``Demo@12345`` — ordinary user
  ("Demo Analyst"), used for demos and manual front-end testing.
* The ``sysconfig.SystemSetting`` singleton, with the shipped defaults.

.. warning::
   These are fixed development credentials.  Never run this against a
   production database.

``seed()`` is importable, so a future ``manage.py seed`` management command is
a three-line wrapper around it.
"""
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Django bootstrap — this script runs as __main__, not through manage.py, so it
# has to put the project root on sys.path and configure Django itself.
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django  # noqa: E402  (must follow the sys.path / env setup above)

django.setup()

from django.contrib.auth import get_user_model  # noqa: E402
from django.db import transaction  # noqa: E402

from system_config.models import SystemSetting  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

ADMIN_EMAIL = 'admin@autosmokeguard.local'
ADMIN_PASSWORD = 'Admin@12345'
ADMIN_NAME = 'System Administrator'

DEMO_EMAIL = 'demo@autosmokeguard.local'
DEMO_PASSWORD = 'Demo@12345'
DEMO_NAME = 'Demo Analyst'


def _ensure_superuser(User):
    """Create the admin account if missing. Returns ``(user, created)``."""
    existing = User.objects.filter(email=ADMIN_EMAIL).first()
    if existing is not None:
        return existing, False

    user = User.objects.create_superuser(
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
        full_name=ADMIN_NAME,
    )
    return user, True


def _ensure_demo_user(User):
    """Create the demo analyst account if missing. ``(user, created)``."""
    existing = User.objects.filter(email=DEMO_EMAIL).first()
    if existing is not None:
        return existing, False

    user = User.objects.create_user(
        email=DEMO_EMAIL,
        password=DEMO_PASSWORD,
        full_name=DEMO_NAME,
    )
    return user, True


def _ensure_system_setting():
    """Create the configuration singleton if missing. ``(row, created)``."""
    created = not SystemSetting.objects.filter(pk=1).exists()
    return SystemSetting.get_solo(), created


def seed(verbose=True):
    """
    Create the baseline records, skipping anything that already exists.

    Returns a summary dict::

        {'admin': bool, 'demo': bool, 'settings': bool}

    where each value says whether that record was *created* by this run.
    The whole thing runs in one transaction so a partially-seeded database is
    never left behind.
    """
    User = get_user_model()

    with transaction.atomic():
        admin_user, admin_created = _ensure_superuser(User)
        demo_user, demo_created = _ensure_demo_user(User)
        settings_row, settings_created = _ensure_system_setting()

    if verbose:
        _report(admin_user, admin_created, demo_user, demo_created,
                settings_row, settings_created)

    return {
        'admin': admin_created,
        'demo': demo_created,
        'settings': settings_created,
    }


def _report(admin_user, admin_created, demo_user, demo_created,
            settings_row, settings_created):
    """Print a short, human-readable summary of what the run did."""
    def mark(created):
        return 'created' if created else 'unchanged'

    print('AutoSmokeGuard — seeding baseline data')
    print('-' * 56)
    print(f'  admin user      {mark(admin_created):<10} {admin_user.email} '
          f'(role={admin_user.role})')
    print(f'  demo user       {mark(demo_created):<10} {demo_user.email} '
          f'(role={demo_user.role})')
    print(f'  system settings {mark(settings_created):<10} pk={settings_row.pk}, '
          f'max_upload_mb={settings_row.max_upload_mb}, '
          f'confidence_threshold={settings_row.confidence_threshold}')
    print('-' * 56)

    if admin_created or demo_created:
        print('Development credentials (change before any real deployment):')
        if admin_created:
            print(f'  {ADMIN_EMAIL} / {ADMIN_PASSWORD}')
        if demo_created:
            print(f'  {DEMO_EMAIL} / {DEMO_PASSWORD}')
    else:
        print('Nothing to do — the database was already seeded.')


if __name__ == '__main__':
    seed()
