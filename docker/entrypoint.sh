#!/bin/bash
# AutoSmokeGuard backend container entrypoint.
#
# Order of operations: wait for the database (only meaningful for Postgres —
# SQLite is a local file and is always "ready"), apply migrations, collect
# static files, optionally seed baseline data, then hand off to gunicorn.
set -euo pipefail

cd /app

# ---------------------------------------------------------------------------
# Wait for the database.
#
# Only relevant when DATABASE_URL points at PostgreSQL (config/settings.py
# switches engines based on that same prefix check). SQLite is a file on the
# container's own filesystem, so there is nothing to wait for.
#
# This is a bounded retry loop, not a fixed `sleep 10`: a fixed sleep either
# wastes time when the DB is already up, or isn't long enough when it isn't
# (e.g. Postgres running its own first-boot initdb behind `pg_isready`'s own
# healthcheck). We poll a plain TCP connect to host:port — cheap, needs no
# extra client tool, and works before Django/psycopg are even asked to
# authenticate — and give up loudly after a fixed number of attempts instead
# of hanging forever.
# ---------------------------------------------------------------------------
if [[ "${DATABASE_URL:-}" == postgres* ]]; then
    echo "entrypoint: DATABASE_URL is PostgreSQL — waiting for the database to accept connections..."
    python - <<'PY'
import os
import socket
import sys
import time
from urllib.parse import urlparse

url = urlparse(os.environ["DATABASE_URL"])
host = url.hostname or "localhost"
port = url.port or 5432

MAX_ATTEMPTS = 30
DELAY_SECONDS = 2

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"entrypoint: database reachable at {host}:{port} "
                  f"(attempt {attempt}/{MAX_ATTEMPTS})")
            sys.exit(0)
    except OSError as exc:
        print(f"entrypoint: database not ready yet ({exc}); "
              f"attempt {attempt}/{MAX_ATTEMPTS}")
        time.sleep(DELAY_SECONDS)

print(f"entrypoint: database at {host}:{port} did not become reachable "
      f"after {MAX_ATTEMPTS * DELAY_SECONDS}s", file=sys.stderr)
sys.exit(1)
PY
fi

echo "entrypoint: applying database migrations..."
python manage.py migrate --noinput

echo "entrypoint: collecting static files..."
python manage.py collectstatic --noinput

# Optional, idempotent baseline data (admin + demo users, default system
# settings) — see tools/seed.py. Off by default; opt in per environment.
# Never run this against a real production database with real customer data.
if [[ "${ASG_SEED:-0}" == "1" ]]; then
    echo "entrypoint: ASG_SEED=1 — seeding baseline data..."
    python tools/seed.py
fi

# ---------------------------------------------------------------------------
# Hand off to gunicorn.
#
# --threads and a 300s --timeout matter a lot here specifically because
# AutoSmokeGuard has no Celery/Redis worker: analysis jobs run on a small
# in-process thread pool inside the SAME Python process that serves HTTP
# (see ASG_WORKER_THREADS / analysis/worker.py). A request that kicks off or
# polls a long-running video analysis can legitimately keep a request cycle
# open for a couple of minutes, so:
#   * gunicorn's default 30s --timeout would kill in-flight requests, and
#   * the default sync worker (1 request per worker at a time) would let one
#     slow request block every other request that process is meant to serve.
# --threads N lets a single gunicorn *process* hold N concurrent requests
# instead of one; --workers M then multiplies that by M independent OS
# processes (each with its own GIL and its own copy of ASG_WORKER_THREADS
# analysis threads), which is how this deployment scales analysis capacity
# horizontally on a single box short of introducing a real task broker.
exec gunicorn config.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers "${GUNICORN_WORKERS:-2}" \
    --threads "${GUNICORN_THREADS:-4}" \
    --timeout 300
