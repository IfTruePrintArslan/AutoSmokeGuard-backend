"""
Django settings for the AutoSmokeGuard backend.

Layout of this module
---------------------
Settings are grouped into labelled blocks separated by ``# ---`` rules so that
the file stays readable as the project grows.  Anything environment-specific is
read from the process environment (optionally seeded from a ``.env`` file that
sits next to ``manage.py``) and always has a development-safe fallback, so a
fresh clone runs with zero configuration.

Reference: https://docs.djangoproject.com/en/6.0/ref/settings/
"""

import os
import tempfile
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlparse

from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load ``backend/.env`` before anything else reads os.environ.  ``load_dotenv``
# is a no-op when the file is absent, which is the normal case in CI.
load_dotenv(BASE_DIR / '.env')


def _env_bool(name, default='False'):
    """Read a boolean-ish env var. Only the exact string 'True' means True."""
    return os.environ.get(name, default) == 'True'


def _env_int(name, default):
    """Read an int env var, falling back to ``default`` when unset/garbage."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


# ---------------------------------------------------------------------------
# Security — read from environment; fall back to dev-only defaults.
# In production, set SECRET_KEY, DEBUG=False, and ALLOWED_HOSTS via env vars
# or a .env file loaded before Django starts (handled above with python-dotenv).
# ---------------------------------------------------------------------------

DEBUG = _env_bool('DEBUG', 'True')

#: The zero-configuration development key.  It is committed to this repository,
#: so it is public: anyone who can read this file can mint a valid token with
#: it.  It is only ever acceptable with DEBUG=True — see the guard below.
INSECURE_DEV_SECRET_KEY = (
    'django-insecure-dev-fallback-replace-in-production-do-not-use-as-is'
)

_SECRET_KEY_FROM_ENV = os.environ.get('SECRET_KEY', '').strip()

# Fail closed, not open (security review finding ASG-01).
#
# SECRET_KEY is also SIMPLE_JWT['SIGNING_KEY'] (see the JWT block below), so a
# deployment that boots on the public fallback above is not merely "using a
# weak key": every access token in the system can be forged by anyone who has
# read this file, which is a complete authentication bypass.  Django's own
# ``check --deploy`` reports that only as a *warning* (security.W009), and only
# when somebody remembers to run it, so it cannot be the control we rely on.
#
# With DEBUG=False an explicit SECRET_KEY is therefore mandatory, and its
# absence stops the process at import time instead of serving traffic on a
# publicly known key.  docker-compose.yml already enforces the same rule from
# the outside (``${SECRET_KEY:?...}``); this covers every other way the app
# can be started.
if _SECRET_KEY_FROM_ENV:
    SECRET_KEY = _SECRET_KEY_FROM_ENV
elif DEBUG:
    SECRET_KEY = INSECURE_DEV_SECRET_KEY
else:
    from django.core.exceptions import ImproperlyConfigured

    raise ImproperlyConfigured(
        'SECRET_KEY must be set in the environment when DEBUG=False. It signs '
        'every JWT this service issues, so booting on the built-in development '
        'fallback would let anyone who has read config/settings.py forge a '
        'token for any account. Generate one with:\n'
        '    python -c "import secrets; print(secrets.token_urlsafe(64))"\n'
        'and supply it via the SECRET_KEY environment variable or backend/.env.'
    )

ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get('ALLOWED_HOSTS', '127.0.0.1,localhost').split(',')
    if host.strip()
]

# The whole project authenticates against our own UUID-keyed user table.
AUTH_USER_MODEL = 'accounts.User'


# ---------------------------------------------------------------------------
# Application definition
# Local apps are listed last so their checks/signals load after third-party
# machinery is in place.  ``common`` comes first among them because the other
# apps import its abstract models, validators and storage helpers.
# ---------------------------------------------------------------------------

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Third-party
    'rest_framework',
    'rest_framework_simplejwt',
    'rest_framework_simplejwt.token_blacklist',
    'corsheaders',
    'django_filters',
    'drf_spectacular',
    # Local
    'common',
    'accounts',
    'uploads',
    'analysis',
    'reports',
    'system_config',   # Django app label is 'sysconfig' — see its __init__.py
    'health',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    # Gates the /media/ tree (blocks the reports/ subtree and any traversal
    # attempt) before anything can serve a file from it. Sits this high on
    # purpose: it must run ahead of WhiteNoise and ahead of the MEDIA_URL
    # route config.urls adds under DEBUG. See common.middleware for the
    # scope of what it does and does not protect (finding ASG-04).
    'common.middleware.MediaGuardMiddleware',
    # Rewrites an HTML 404 under /api/ into the {detail, code, errors}
    # envelope (robustness finding §B: a malformed <uuid:...> path parameter
    # 404s at the URL resolver, so DRF's exception handler never runs — and
    # under DEBUG Django's technical 404 page lists every registered URL
    # pattern to an unauthenticated caller). Placed here, near the top, so
    # its *response* phase runs last: CommonMiddleware has already had its
    # APPEND_SLASH say and CorsMiddleware has already attached its headers.
    # See common.middleware for the full rationale and for the responses it
    # deliberately leaves alone.
    'common.middleware.ApiErrorEnvelopeMiddleware',
    # WhiteNoise serves collected static files (admin CSS, Swagger UI assets)
    # straight from the WSGI app; it must sit directly after SecurityMiddleware.
    'whitenoise.middleware.WhiteNoiseMiddleware',
    # CorsMiddleware must come as high as possible — before any middleware that
    # generates responses such as CommonMiddleware.
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # Last in the chain so the timing it records covers every other middleware.
    'common.middleware.RequestLogMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'


# ---------------------------------------------------------------------------
# Database
# Default is SQLite so a fresh clone runs immediately.  Setting DATABASE_URL to
# a postgres:// (or postgresql://) URL switches to PostgreSQL.  The URL is
# parsed by hand with urllib so we do not take a dependency on dj-database-url.
#
# SQLite write contention (NFR: 10 concurrent users)
# --------------------------------------------------
# This project runs its analysis jobs on a thread pool *inside* the web
# process, so a background worker and an HTTP request routinely write to the
# same tables at the same moment.  SQLite serialises writers across the whole
# database, and in its stock rollback-journal mode a writer also blocks every
# reader — under five concurrent jobs that surfaces as ``database is locked``
# and, on the shared-cache connection Django's in-memory test database uses,
# as ``database table is locked``.  Measured on this machine (10 jobs x 4
# worker threads x 12 rounds, auto-PDF on): 45 lock collisions without the
# options below, 0 with them.
#
# The three settings that matter, all supported by the Django 6.0 SQLite
# backend (``django.db.backends.sqlite3.base.DatabaseWrapper``):
#
# * ``init_command`` runs once per new connection.
#   - ``journal_mode=WAL`` lets readers proceed while a writer holds the lock.
#     WAL is a persistent property of the *database file*, so the first
#     connection to set it converts the file and every later connection —
#     including worker threads — inherits it.
#   - ``synchronous=NORMAL`` is the documented companion to WAL: durable
#     across process crashes, and only at risk of losing the last commits in a
#     full OS/power failure, which is an acceptable trade for a queue whose
#     jobs are re-runnable.
#   - ``busy_timeout=20000`` makes a blocked writer wait up to 20 s for the
#     lock instead of failing immediately.
# * ``timeout: 20`` is the same 20 s expressed through the DB-API driver, so
#   the value holds even on a connection opened before ``init_command`` runs.
# * ``transaction_mode: 'IMMEDIATE'`` issues ``BEGIN IMMEDIATE``, taking the
#   write lock up front.  Without it a transaction starts out read-only and
#   has to *upgrade* when it first writes; an upgrade that loses the race
#   cannot be resolved by waiting (both parties already hold a read lock), so
#   SQLite gives up instantly instead of honouring the busy timeout.  Verified
#   under pytest: no misbehaviour, so the key is kept.
#
# ``analysis.services.retry_on_lock`` stays in place as defence in depth — it
# is what covers the residual case and any deployment that overrides these
# options away.  These settings remove the collisions; the retry survives the
# ones we have not thought of.
# ---------------------------------------------------------------------------

#: Connection options for the SQLite branch. See the note above before
#: changing any of them — each one is load-bearing under concurrency.
_SQLITE_OPTIONS = {
    # DB-API level busy timeout, in seconds.
    'timeout': 20,
    # Executed on every new connection, including worker-thread connections.
    'init_command': (
        'PRAGMA journal_mode=WAL; '
        'PRAGMA synchronous=NORMAL; '
        'PRAGMA busy_timeout=20000;'
    ),
    # BEGIN IMMEDIATE — take the write lock up front, never upgrade into it.
    'transaction_mode': 'IMMEDIATE',
}

# The test database must be a real file, not Django's default
# ``file:memorydb_default?mode=memory&cache=shared``.  An in-memory database
# cannot be put into WAL mode (``PRAGMA journal_mode`` reports ``memory``) and
# its shared cache raises SQLITE_LOCKED on table-level contention, which no
# busy timeout can wait out.  Tests that exercise the real worker pool would
# therefore keep hitting a failure mode production does not have — and, worse,
# would *not* exercise the one it does.  Pointing TEST at a file makes the
# suite run against the same locking model as the deployed database.
#
# The name carries the PID so two test runs on one machine (a developer and a
# CI job, or two agents) cannot clobber each other's database.  Set
# ASG_TEST_DB_NAME to pin a stable path when you want ``--reuse-db``.
_TEST_DB_NAME = os.environ.get('ASG_TEST_DB_NAME', '').strip() or str(
    Path(tempfile.gettempdir()) / f'asg_test_db_{os.getpid()}.sqlite3'
)

_DATABASE_URL = os.environ.get('DATABASE_URL', '').strip()

if _DATABASE_URL.startswith('postgres'):
    _url = urlparse(_DATABASE_URL)
    # Anything after '?' (e.g. ?sslmode=require) is handed to psycopg verbatim.
    _options = dict(parse_qsl(_url.query))
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': unquote(_url.path.lstrip('/')),
            'USER': unquote(_url.username or ''),
            'PASSWORD': unquote(_url.password or ''),
            'HOST': _url.hostname or 'localhost',
            'PORT': str(_url.port or 5432),
            'CONN_MAX_AGE': _env_int('DB_CONN_MAX_AGE', 60),
            'OPTIONS': _options,
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
            # Copied, so nothing downstream can mutate the shared dict.
            'OPTIONS': dict(_SQLITE_OPTIONS),
            'TEST': {'NAME': _TEST_DB_NAME},
        }
    }


# ---------------------------------------------------------------------------
# Password storage and validation
# PBKDF2 is listed first explicitly: it is the hasher every password in this
# project is written with, and pinning it keeps hashes reproducible across
# machines that may or may not have argon2/bcrypt available.
# ---------------------------------------------------------------------------

PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.PBKDF2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher',
    'django.contrib.auth.hashers.ScryptPasswordHasher',
]

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {'min_length': 8},
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# ---------------------------------------------------------------------------
# Internationalization
# https://docs.djangoproject.com/en/6.0/topics/i18n/
# ---------------------------------------------------------------------------

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# ---------------------------------------------------------------------------
# Static and media files
# STATIC_ROOT is where `collectstatic` writes; WhiteNoise serves from there.
# MEDIA_ROOT holds user uploads plus every artefact the ML pipeline produces
# (annotated previews, vehicle crops, smoke masks, generated PDF reports).
# ---------------------------------------------------------------------------

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Created eagerly so WhiteNoise does not warn on a checkout that has not run
# `collectstatic` yet, and so the first upload does not race on mkdir.  Both
# directories are gitignored.
STATIC_ROOT.mkdir(parents=True, exist_ok=True)
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)

# Default primary key field type
# https://docs.djangoproject.com/en/6.0/ref/settings/#default-auto-field
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# ---------------------------------------------------------------------------
# Upload limits
# Keep the in-memory threshold small (5 MB) so multi-hundred-megabyte videos
# are streamed to a temporary file on disk instead of being buffered in RAM.
# The real per-file ceiling lives in ASG['MAX_UPLOAD_MB'] / the SystemSetting
# row and is enforced by common.validators.validate_upload.
# ---------------------------------------------------------------------------

DATA_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024
DATA_UPLOAD_MAX_NUMBER_FIELDS = 2000


# ---------------------------------------------------------------------------
# Django REST Framework
# ---------------------------------------------------------------------------

REST_FRAMEWORK = {
    # SimpleJWT's JWTAuthentication plus one extra check: an access token
    # issued before the owner's last password change is refused with
    # code="token_not_valid" (security review finding ASG-08 — blacklisting
    # refresh tokens on a reset left the already-minted access token alive for
    # up to its full 30-minute lifetime).  Set as the *default* rather than
    # per-view so no endpoint can be added later that quietly skips it.
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'accounts.authentication.PasswordChangeAwareJWTAuthentication',
    ),
    # Locked down by default — public endpoints (health, schema, docs, auth)
    # opt out explicitly with permission_classes = [AllowAny].
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_PAGINATION_CLASS': 'common.pagination.StandardPagination',
    'PAGE_SIZE': 10,
    'DEFAULT_FILTER_BACKENDS': (
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.OrderingFilter',
        'rest_framework.filters.SearchFilter',
    ),
    'EXCEPTION_HANDLER': 'common.exceptions.api_exception_handler',
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
    'DEFAULT_THROTTLE_CLASSES': (
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle',
    ),
    'DEFAULT_THROTTLE_RATES': {
        'anon': '60/min',
        'user': '600/min',
    },
}


# ---------------------------------------------------------------------------
# JSON Web Tokens (djangorestframework-simplejwt)
# Short-lived access tokens with rotating refresh tokens; rotated refreshes are
# blacklisted so a stolen refresh token cannot be replayed after the legitimate
# client has used it.  Claims carry our UUID primary key, not an integer id.
# ---------------------------------------------------------------------------

from datetime import timedelta  # noqa: E402  (kept next to the block it serves)

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=30),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'UPDATE_LAST_LOGIN': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_FIELD': 'user_id',
    'USER_ID_CLAIM': 'user_id',
    'TOKEN_TYPE_CLAIM': 'token_type',
    'JTI_CLAIM': 'jti',
}


# ---------------------------------------------------------------------------
# OpenAPI schema (drf-spectacular)
# Served at /api/schema/ with Swagger UI at /api/docs/.
# ---------------------------------------------------------------------------

SPECTACULAR_SETTINGS = {
    'TITLE': 'AutoSmokeGuard API',
    'VERSION': '1.0.0',
    'DESCRIPTION': (
        'REST API for AutoSmokeGuard — upload traffic images or video, run the '
        'YOLO vehicle-detection plus U-Net smoke-segmentation pipeline, review '
        'per-vehicle emission severity, and download a PDF emission report.'
    ),
    # The schema endpoint itself should not appear as an operation in the docs.
    'SERVE_INCLUDE_SCHEMA': False,
    'SCHEMA_PATH_PREFIX': '/api',
    'COMPONENT_SPLIT_REQUEST': True,
    # Keeps the deprecated '/health/' alias out of the docs so it does not
    # collide with '/api/health/' on operationId.
    'PREPROCESSING_HOOKS': ['common.schema.exclude_legacy_paths'],
}


# ---------------------------------------------------------------------------
# CORS — the React/Vite front end runs on a different origin in development.
# Override with CORS_ORIGINS (comma separated) in any deployed environment.
# ---------------------------------------------------------------------------

_DEFAULT_CORS_ORIGINS = ','.join([
    'http://localhost:5173',    # vite dev server
    'http://127.0.0.1:5173',
    'http://localhost:4173',    # vite preview
    'http://127.0.0.1:4173',
])

CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get('CORS_ORIGINS', _DEFAULT_CORS_ORIGINS).split(',')
    if origin.strip()
]

CORS_ALLOW_CREDENTIALS = True


# ---------------------------------------------------------------------------
# Hardening — only switched on outside DEBUG so local http:// development and
# the test suite are unaffected.
# ---------------------------------------------------------------------------

SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# NOTE: Django 6.0 removed SECURE_BROWSER_XSS_FILTER (the X-XSS-Protection
# header is deprecated by every current browser).  It is declared here for
# documentation/parity with the SDS security checklist; Django ignores it.
SECURE_BROWSER_XSS_FILTER = True

if not DEBUG:
    SESSION_COOKIE_HTTPONLY = True
    CSRF_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = _env_bool('SESSION_COOKIE_SECURE', 'True')
    CSRF_COOKIE_SECURE = _env_bool('CSRF_COOKIE_SECURE', 'True')
    SECURE_SSL_REDIRECT = _env_bool('SECURE_SSL_REDIRECT', 'True')
    SECURE_HSTS_SECONDS = _env_int('SECURE_HSTS_SECONDS', 31536000)
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    STORAGES = {
        'default': {
            'BACKEND': 'django.core.files.storage.FileSystemStorage',
        },
        'staticfiles': {
            'BACKEND': 'whitenoise.storage.CompressedManifestStaticFilesStorage',
        },
    }


# ---------------------------------------------------------------------------
# AutoSmokeGuard application settings
# One namespaced dict rather than a dozen loose module-level names, so app code
# can do `settings.ASG['YOLO_WEIGHTS']` and tests can override the whole block.
# Values that an administrator can change at runtime (UC-09) live in the
# sysconfig.SystemSetting singleton; the entries here are the boot defaults.
# ---------------------------------------------------------------------------

ASG = {
    'ML_ASSETS_DIR': BASE_DIR / 'ml_assets',
    'SAMPLE_MEDIA_DIR': BASE_DIR / 'sample_media',
    'YOLO_WEIGHTS': BASE_DIR / 'ml_assets' / 'yolo11n.pt',
    'SEGMENTER_WEIGHTS': BASE_DIR / 'ml_assets' / 'smoke_unet.pt',
    'WORKER_THREADS': int(os.environ.get('ASG_WORKER_THREADS', '2')),
    'WORKER_ENABLED': os.environ.get('ASG_WORKER_ENABLED', 'True') == 'True',
    'MAX_UPLOAD_MB': int(os.environ.get('ASG_MAX_UPLOAD_MB', '512')),
    'ALLOWED_IMAGE_FORMATS': ['jpg', 'jpeg', 'png'],
    'ALLOWED_VIDEO_FORMATS': ['mp4', 'avi', 'mov'],
}


# ---------------------------------------------------------------------------
# Logging
# Everything the project emits goes through the 'asg' logger (and its children,
# e.g. 'asg.request', 'asg.worker').  Two sinks: the console for development
# and a size-rotating file under backend/logs/ for anything long-running.
# ---------------------------------------------------------------------------

LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{asctime}] {levelname:<8} {name}: {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname:<8} {name}: {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOGS_DIR / 'backend.log'),
            'maxBytes': 5 * 1024 * 1024,   # 5 MB per file
            'backupCount': 5,
            'encoding': 'utf-8',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'WARNING',
    },
    'loggers': {
        'asg': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'django.request': {
            'handlers': ['console', 'file'],
            'level': 'WARNING',
            'propagate': False,
        },
    },
}
