# AutoSmokeGuard backend — multi-stage image (builder venv -> slim runtime).
#
# Base image: python:3.12-slim, not 3.14.
# requirements.txt says the app is developed against Python 3.14, but the
# application code itself is not 3.14-specific — it only uses stable-for-
# years language/stdlib features. 3.12 is chosen deliberately for the
# container because prebuilt wheel availability for the heavy ML stack
# (torch, torchvision, ultralytics, opencv-python, numpy) is dramatically
# better on 3.12 than on a release as new as 3.14: fewer missing wheels means
# fewer packages that fall back to a slow (and sometimes outright broken,
# for something like torch) source build inside the image.

# =============================================================================
# Stage 1: builder — installs everything into an isolated venv.
# =============================================================================
FROM python:3.12-slim AS builder

# build-essential covers the rare pure-Python-on-3.12 package that has no
# prebuilt wheel and needs to compile a C extension from source. Removed
# entirely in the runtime stage below — it is never needed after `pip
# install` has produced .so files inside the venv.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# A dedicated venv (rather than installing into the system interpreter) is
# what lets the runtime stage copy *only* /opt/venv and get a byte-for-byte
# identical dependency set, without dragging pip's cache, build-essential,
# or any stray system site-packages along with it.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ---------------------------------------------------------------------------
# Torch / torchvision get their own layer, installed from PyTorch's CPU wheel
# index instead of PyPI.
#
# This is the single biggest image-size lever available here: the default
# PyPI torch wheel bundles the full CUDA runtime (libcublas, libcudnn,
# libnccl, ...) — on the order of 2GB — even though this service only ever
# does CPU inference (a t3.medium EC2 instance, the SDS's target, has no
# GPU). Pointing pip at the CPU-only wheel index avoids pulling any of that
# in the first place. Doing this as its own RUN/layer also means it's cached
# independently of the rest of requirements.txt, so touching an unrelated
# dependency doesn't force a ~200MB re-download.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.14.0 torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Dependency manifests copied (and installed) before application code so
# this layer is only invalidated when a dependency actually changes, not on
# every source edit. torch/torchvision are already satisfied by the layer
# above, so this second install is a no-op for them and only resolves the
# rest of requirements.txt (Django, DRF, ultralytics, opencv, reportlab, ...)
# plus requirements-prod.txt (gunicorn).
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-prod.txt

# =============================================================================
# Stage 2: runtime — slim image, venv only, non-root.
# =============================================================================
FROM python:3.12-slim AS runtime

# libgl1 / libglib2.0-0: opencv-python is built against a GUI-capable OpenCV
# and dlopens libGL.so.1 / libgthread at `import cv2` time even when running
# perfectly headless on a server — without these two apt packages the import
# fails outright. curl is here only so HEALTHCHECK below has something to
# call; it is not an application dependency.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    DJANGO_SETTINGS_MODULE=config.settings

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Application code. .dockerignore is what actually keeps this COPY safe and
# small: it excludes .venv/ (1.2GB), ml_assets/datasets/ (263MB training
# data), media/, logs/, db.sqlite3 and other dev-only/runtime-generated
# state, so none of that can leak into the image even though this is a
# broad `COPY . .`. ml_assets/*.pt (the YOLO + U-Net weights, ~13MB total)
# and ml_assets/metrics.json are NOT excluded — the analysis pipeline cannot
# run without them and this image has no volume-mount/init-container story
# for model weights today.
COPY . .

RUN chmod +x docker/entrypoint.sh \
    && mkdir -p media staticfiles logs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# /api/health/ is unauthenticated and unthrottled by design (see
# health/views.py) specifically so probes like this one can call it with no
# credentials. start-period is generous because migrate + collectstatic run
# before gunicorn even binds the port; the (much slower) YOLO/U-Net model
# load happens afterwards on a background thread and is NOT a precondition
# for a healthy response, so it doesn't need to be covered here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fs http://localhost:8000/api/health/ || exit 1

ENTRYPOINT ["docker/entrypoint.sh"]
