#!/usr/bin/env python3
"""AutoSmokeGuard one-command developer launcher (core implementation).

This module is the real launcher.  ``FYP/start.py`` is a three-line shim that
puts ``backend/tools`` on ``sys.path`` and calls :func:`main` — that keeps the
logic version-controlled inside the backend repository while the repo root
keeps a convenient double-clickable entry point.

It is deliberately **standard library only**.  It has to run on a machine where
nothing but a system Python exists, *before* any virtual environment has been
created, so it cannot import anything from ``requirements.txt``.

Responsibilities, in order:

  1. Preflight        OS / arch / python / node / npm / disk checks
  2. Backend venv     create ``backend/.venv``, install ``requirements.txt``
  3. Frontend deps    ``npm ci`` / ``npm install`` in ``frontend/``
  4. ML assets        fetch YOLO weights, train the smoke U-Net on first run
  5. Database         ``manage.py migrate`` + ``tools/seed.py``
  6. Sample media     ``tools/make_samples.py`` when ``sample_media/`` is empty
  7. Start servers    Django + Vite as supervised children with prefixed output
  8. Readiness        poll the API health endpoint and the Vite dev server
  9. Browser          open the UI
 10. Supervise        stream logs, restart nothing, die cleanly together

Run ``python start.py --help`` for the flag list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import urllib.error
import urllib.request
import venv
import webbrowser
from collections import deque
from pathlib import Path, PureWindowsPath

# ---------------------------------------------------------------------------
# Locations.  dev_runner.py lives at <repo>/backend/tools/dev_runner.py, so the
# backend root is two parents up and the repo root is three.  Everything else
# is derived from those two anchors — no string concatenation, no cwd
# assumptions, so the launcher works no matter where it is invoked from.
# ---------------------------------------------------------------------------
TOOLS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TOOLS_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = REPO_ROOT / 'frontend'

VENV_DIR = BACKEND_DIR / '.venv'
REQUIREMENTS = BACKEND_DIR / 'requirements.txt'
MANAGE_PY = BACKEND_DIR / 'manage.py'
ML_ASSETS_DIR = BACKEND_DIR / 'ml_assets'
SAMPLE_MEDIA_DIR = BACKEND_DIR / 'sample_media'
LOG_DIR = BACKEND_DIR / 'logs'
DB_FILE = BACKEND_DIR / 'db.sqlite3'
SEED_SCRIPT = TOOLS_DIR / 'seed.py'
SAMPLES_SCRIPT = TOOLS_DIR / 'make_samples.py'
CREDENTIALS_FILE = TOOLS_DIR / 'seed_credentials.json'

BACKEND_STAMP = VENV_DIR / '.asg_deps_stamp'
FRONTEND_STAMP = FRONTEND_DIR / 'node_modules' / '.asg_deps_stamp'
PACKAGE_JSON = FRONTEND_DIR / 'package.json'
PACKAGE_LOCK = FRONTEND_DIR / 'package-lock.json'
NODE_MODULES = FRONTEND_DIR / 'node_modules'

API_LOG = LOG_DIR / 'launcher-api.log'
WEB_LOG = LOG_DIR / 'launcher-web.log'

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# Django 6.0 (pinned in requirements.txt) declares ``Requires-Python
# >=3.12``. CI proved this is a hard install-time floor, not a suggestion:
# on Python 3.11 `pip install -r requirements.txt` cannot even resolve it
# ("Could not find a version that satisfies the requirement Django==6.0.6"
# / "No matching distribution found for Django==6.0.6") — there is no
# friendlier failure mode below 3.12. Keep this in sync with the `test`
# job's python-version matrix in .github/workflows/ci.yml and with
# check_python_floor() below, which turns a violation of this floor into an
# actionable preflight failure instead of that confusing pip error.
MIN_PYTHON = (3, 12)
MIN_NODE_MAJOR = 18
MIN_FREE_DISK_MB = 2048

DEFAULT_API_PORT = 8000
DEFAULT_WEB_PORT = 5173
PORT_SCAN_ATTEMPTS = 50
HEALTH_PATH = '/api/health/'
READY_TIMEOUT = 60.0
READY_INTERVAL = 0.5

YOLO_WEIGHTS = ML_ASSETS_DIR / 'yolo11n.pt'
YOLO_URL = (
    'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
)
UNET_WEIGHTS = ML_ASSETS_DIR / 'smoke_unet.pt'
SYNTH_MODULE = 'mlcore.training.synth_dataset'
TRAIN_MODULE = 'mlcore.training.train_unet'

# Fallback demo credentials.  tools/seed.py may drop a
# tools/seed_credentials.json ({"demo": {...}, "admin": {...}}) to override
# these without either file having to know about the other.
DEFAULT_CREDENTIALS = {
    'demo': {'email': 'demo@autosmokeguard.local', 'password': 'Demo@12345'},
    'admin': {'email': 'admin@autosmokeguard.local', 'password': 'Admin@12345'},
}

# Deliberately ordered, not alphabetical/numeric: 3.13 first because it is
# the newer of the two versions exercised by CI (see ci.yml's `test` job
# matrix) and matches the primary supported target; 3.12 next because it is
# MIN_PYTHON, the floor Django 6.0 requires; 3.14 after that as a
# forward-compatible fallback (newest CPython, not yet in the CI matrix);
# the unversioned `python3`/`python` names last because they could resolve
# to literally anything the platform ships under that name. 3.11 is
# deliberately ABSENT — Django 6.0 requires >=3.12, so offering an
# interpreter below that floor here would just relocate today's confusing
# pip resolver error a few minutes later instead of avoiding it. Do not add
# it back; see MIN_PYTHON and check_python_floor() above/below.
PYTHON_CANDIDATES = (
    'python3.13', 'python3.12', 'python3.14', 'python3', 'python',
)
# The Windows ``py`` launcher understands version selectors that are not on
# PATH as standalone executables, so it gets its own probe list. Same
# ordering rationale as PYTHON_CANDIDATES; 3.11 is absent for the same
# reason.
PY_LAUNCHER_ARGS = ('-3.13', '-3.12', '-3.14', '-3')

VERSION_PROBE = 'import sys;print(sys.version_info[:2])'
USER_AGENT = 'AutoSmokeGuard-launcher/1.0'
ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]')

PYTHON_DOWNLOAD_URL = 'https://www.python.org/downloads/'
NODE_DOWNLOAD_URL = 'https://nodejs.org/en/download'


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class LauncherError(Exception):
    """A fatal, *explained* failure.

    Every raise site fills in what was being attempted, the command that
    failed, the tail of its output and the most likely fix, so the console
    never shows a bare traceback for an expected problem.
    """

    def __init__(self, message, *, detail='', hint='', cmd=None, output=''):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.hint = hint
        self.cmd = cmd
        self.output = output

    def render(self):
        lines = ['', style('  FAILED: ' + self.message, 'bold', 'red')]
        if self.detail:
            for chunk in str(self.detail).splitlines():
                lines.append('    ' + chunk)
        if self.cmd:
            lines.append('')
            lines.append(style('    command: ', 'grey') + format_cmd(self.cmd))
        tail = tail_lines(self.output, 20)
        if tail:
            lines.append(style('    last output:', 'grey'))
            for chunk in tail:
                lines.append(style('      | ', 'grey') + chunk)
        if self.hint:
            lines.append('')
            lines.append(style('    likely fix:', 'bold', 'yellow'))
            for chunk in textwrap.wrap(self.hint, 92) or ['']:
                lines.append('      ' + chunk)
        lines.append('')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Console styling.  ANSI only, no dependencies.  Disabled when NO_COLOR is set,
# when stdout is not a tty, or when --no-color is passed.  On Windows the
# os.system('') call is the cheapest documented way to flip the console into
# virtual-terminal mode on Windows 10+; on older hosts colour simply degrades
# to plain text because the escape codes are then stripped by our own guard.
# ---------------------------------------------------------------------------
class Style:
    CODES = {
        'reset': '\033[0m', 'bold': '\033[1m', 'dim': '\033[2m',
        'red': '\033[31m', 'green': '\033[32m', 'yellow': '\033[33m',
        'blue': '\033[34m', 'magenta': '\033[35m', 'cyan': '\033[36m',
        'grey': '\033[90m',
    }

    def __init__(self, enabled=False):
        self.enabled = enabled

    def __call__(self, text, *names):
        if not self.enabled or not names:
            return text
        prefix = ''.join(self.CODES[n] for n in names if n in self.CODES)
        return prefix + str(text) + self.CODES['reset'] if prefix else str(text)


style = Style(False)

PRINT_LOCK = threading.RLock()
_STDOUT_BROKEN = False
_ASCII_ONLY = False
GLYPH_OK, GLYPH_BAD, GLYPH_SKIP, GLYPH_ARROW = '✓', '✗', '·', '==>'


def configure_console(force_plain=False):
    """Turn on colour + unicode glyphs when the terminal can handle them."""
    global _ASCII_ONLY, GLYPH_OK, GLYPH_BAD, GLYPH_SKIP

    if platform.system() == 'Windows':
        # No-op shell call: it initialises the console host and enables VT
        # processing for the rest of the process on Windows 10+.
        try:
            os.system('')
        except OSError:
            pass

    try:
        is_tty = sys.stdout.isatty()
    except (AttributeError, ValueError):
        is_tty = False

    enabled = (
        not force_plain
        and is_tty
        and not os.environ.get('NO_COLOR')
        and os.environ.get('TERM') != 'dumb'
    )
    style.enabled = bool(enabled)

    # Legacy Windows consoles default to cp1252 and raise UnicodeEncodeError on
    # the tick/cross glyphs.  Try to switch stdout to UTF-8 first; if that is
    # not possible, fall back to ASCII markers rather than crashing.
    encoding = getattr(sys.stdout, 'encoding', None) or 'ascii'
    try:
        '✓✗'.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, OSError, ValueError):
            _ASCII_ONLY = True
            GLYPH_OK, GLYPH_BAD, GLYPH_SKIP = 'OK', 'XX', '--'


def silence_stdout():
    """Point stdout at the void after a broken pipe.

    Without this, the interpreter's own flush at shutdown re-raises and prints
    "Exception ignored", which is exactly the noise we are trying to avoid.
    """
    global _STDOUT_BROKEN
    _STDOUT_BROKEN = True
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):
        pass


def emit(text=''):
    """Thread-safe println — the log reader threads share stdout with us.

    A closed pipe (``start.py --help | head``) is normal, not an error: we go
    quiet instead of spraying a BrokenPipeError traceback.
    """
    if _STDOUT_BROKEN:
        return
    with PRINT_LOCK:
        try:
            sys.stdout.write(str(text) + '\n')
            sys.stdout.flush()
        except (BrokenPipeError, ValueError):
            silence_stdout()


def info(text):
    emit('    ' + str(text))


def note(text):
    emit(style('    ' + str(text), 'grey'))


def warn(text):
    emit(style('    ! ' + str(text), 'yellow'))


def format_cmd(cmd):
    """Render an argv list the way a human would have typed it."""
    if isinstance(cmd, str):
        return cmd
    out = []
    for part in cmd:
        part = str(part)
        out.append('"' + part + '"' if ' ' in part and '"' not in part else part)
    return ' '.join(out)


def tail_lines(text, count):
    if not text:
        return []
    lines = [ln.rstrip() for ln in str(text).splitlines() if ln.strip()]
    return lines[-count:]


def human_bytes(num):
    value = float(num)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if value < 1024 or unit == 'TB':
            return ('%.0f %s' % (value, unit)) if unit == 'B' else ('%.1f %s' % (value, unit))
        value /= 1024
    return '%.1f TB' % value


def rel(path):
    """Path relative to the repo root when possible — shorter console lines."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(path)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def sha256_file(path):
    """Hex sha256 of a file, or '' when it does not exist."""
    path = Path(path)
    if not path.is_file():
        return ''
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_stamp(path):
    """Read a dependency stamp. Tolerates both the JSON and bare-hex formats."""
    try:
        raw = Path(path).read_text(encoding='utf-8').strip()
    except (OSError, UnicodeDecodeError):
        return ''
    if not raw:
        return ''
    if raw.startswith('{'):
        try:
            return str(json.loads(raw).get('sha256', ''))
        except (ValueError, AttributeError):
            return ''
    return raw


def write_stamp(path, digest, **extra):
    payload = {'sha256': digest, 'written': time.strftime('%Y-%m-%dT%H:%M:%S')}
    payload.update(extra)
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    except OSError as exc:  # a missing stamp only costs a re-install
        warn('could not write stamp %s (%s)' % (rel(path), exc))


def port_is_free(port, host='127.0.0.1'):
    """True when *port* can be bound right now.

    SO_REUSEADDR is deliberately NOT set on Windows: there it means
    "steal the port from whoever holds it", which would report a busy port as
    free.  On POSIX it only affects TIME_WAIT sockets, which is what we want.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if os.name != 'nt':
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def free_port(preferred, host='127.0.0.1', attempts=PORT_SCAN_ATTEMPTS):
    """Return *preferred* when free, else the next free port above it."""
    preferred = int(preferred)
    for offset in range(attempts):
        candidate = preferred + offset
        if candidate > 65535:
            break
        if port_is_free(candidate, host):
            return candidate
    raise LauncherError(
        'no free TCP port found near %d' % preferred,
        detail='Scanned %d ports starting at %d on %s.' % (attempts, preferred, host),
        hint='Something is occupying a large block of ports. Pass an explicit '
             'free port with --api-port / --web-port, or reboot.',
    )


def probe_python(cmd, timeout=25):
    """Run the interpreter and return its (major, minor), or None."""
    argv = list(cmd) + ['-c', VERSION_PROBE]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, errors='replace', timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    match = re.search(r'(\d+)\s*,\s*(\d+)', proc.stdout or '')
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def check_python_floor(version, description, *, is_existing_venv=False):
    """Raise :class:`LauncherError` when *version* is below :data:`MIN_PYTHON`.

    This is the explicit, preflight-time gate that stops a too-old
    interpreter *before* it ever reaches ``pip install -r requirements.txt``.
    Django 6.0 (pinned in requirements.txt) declares ``Requires-Python
    >=3.12``; CI proved this is a hard floor, not a suggestion — on Python
    3.11 pip cannot even resolve the dependency ("Could not find a version
    that satisfies the requirement Django==6.0.6"), and that resolver error
    gives no hint that the interpreter itself is the actual problem. Calling
    this up front turns that opaque, minutes-later failure into one clear,
    actionable message naming the interpreter that was found, the floor it
    missed, and why.

    Called from two places:
      * :func:`find_python`, for a *pre-existing* ``backend/.venv`` — an old
        venv is not something this launcher rebuilds automatically (nothing
        else deletes it for the user), so the fix here is to say so and tell
        the user to delete it, rather than silently limping on with it.
      * :meth:`Launcher.phase_preflight`, as a final defensive check on
        whatever :func:`find_python` ultimately selected — including a bare
        ``python3``/``python``, which could resolve to anything on a given
        machine.

    *version* may be ``None`` (interpreter could not be probed at all); that
    also fails the floor check.
    """
    if version is not None and version >= MIN_PYTHON:
        return
    found = ('%d.%d' % version) if version else 'unknown'
    want = '%d.%d' % MIN_PYTHON
    if is_existing_venv:
        raise LauncherError(
            '%s was built with Python %s, older than the required %s+' % (
                rel(VENV_DIR), found, want),
            detail='Django 6.0 (pinned in backend/requirements.txt) requires '
                   'Python >= 3.12. A virtualenv built with an older '
                   'interpreter cannot install it, and would fail later with '
                   'a confusing pip resolver error instead of this message.',
            hint='Delete %s and re-run this launcher so it rebuilds the '
                 'virtualenv with a supported interpreter (%s+).' % (
                     rel(VENV_DIR), want),
        )
    raise LauncherError(
        '%s is Python %s, older than the required %s+' % (description, found, want),
        detail='Django 6.0 (pinned in backend/requirements.txt) declares '
               'Requires-Python >= 3.12. Python %s cannot install this '
               'project at all — pip would fail with "Could not find a '
               'version that satisfies the requirement Django==6.0.6".' % found,
        hint='Install Python %s+ from %s and make sure it is the interpreter '
             'PATH resolves to, then re-run.' % (want, PYTHON_DOWNLOAD_URL),
    )


def find_python(venv_python=None, is_windows=None, verbose=False):
    """Locate an interpreter good enough to build the backend venv.

    Order: an existing project venv, then the explicit version-tagged names
    (newest-but-proven first), then the bare names, then the Windows ``py``
    launcher, then whatever is running this script.  Returns
    ``(argv_list, (major, minor), description)``.

    Every candidate is checked against :data:`MIN_PYTHON` (Django 6.0's
    floor) via :func:`check_python_floor` before it can be selected, so this
    function never returns an interpreter below that floor — a pre-existing
    ``backend/.venv`` built with a too-old interpreter fails loudly here
    instead of being silently reused later.
    """
    if is_windows is None:
        is_windows = platform.system() == 'Windows'

    tried = []

    if venv_python is not None and Path(venv_python).exists():
        version = probe_python([str(venv_python)])
        if version:
            check_python_floor(version, 'existing project virtualenv',
                                is_existing_venv=True)
            return [str(venv_python)], version, 'existing project virtualenv'
        tried.append('%s (existing venv, unusable — could not run it)'
                     % rel(venv_python))

    for name in PYTHON_CANDIDATES:
        found = shutil.which(name)
        if not found:
            continue
        version = probe_python([found])
        if verbose:
            note('probe %-12s -> %s' % (name, ('%d.%d' % version) if version else 'no answer'))
        if version and version >= MIN_PYTHON:
            return [found], version, name
        tried.append('%s (%s)' % (name, ('%d.%d' % version) if version else 'did not run'))

    if is_windows:
        launcher = shutil.which('py')
        if launcher:
            for selector in PY_LAUNCHER_ARGS:
                version = probe_python([launcher, selector])
                if verbose:
                    note('probe py %-6s -> %s' % (
                        selector, ('%d.%d' % version) if version else 'no answer'))
                if version and version >= MIN_PYTHON:
                    return [launcher, selector], version, 'py ' + selector
                tried.append('py %s (%s)' % (
                    selector, ('%d.%d' % version) if version else 'not installed'))

    running = probe_python([sys.executable])
    if running and running >= MIN_PYTHON:
        return [sys.executable], running, 'the interpreter running this script'
    tried.append('%s (%s)' % (sys.executable, ('%d.%d' % running) if running else 'unknown'))

    want = '%d.%d' % MIN_PYTHON
    if is_windows:
        fix = ('Install Python from %s (tick "Add python.exe to PATH" in the '
               'installer) or run: winget install Python.Python.3.12' % PYTHON_DOWNLOAD_URL)
    elif platform.system() == 'Darwin':
        fix = ('Install Python from %s or run: brew install python@3.12'
               % PYTHON_DOWNLOAD_URL)
    else:
        fix = ('Install Python from %s or run: sudo apt install python3.12 '
               'python3.12-venv' % PYTHON_DOWNLOAD_URL)
    raise LauncherError(
        'no Python >= %s found on this machine' % want,
        detail='Django 6.0 (pinned in backend/requirements.txt) requires '
               'Python >= 3.12; this project cannot install on anything '
               'older.\nTried:\n  - ' + '\n  - '.join(tried),
        hint=fix,
    )


def which_npm(is_windows=None):
    """Resolve the npm executable. On Windows npm is a .cmd shim."""
    if is_windows is None:
        is_windows = platform.system() == 'Windows'
    if is_windows:
        return shutil.which('npm.cmd') or shutil.which('npm')
    return shutil.which('npm')


def wait_for_http(url, timeout=READY_TIMEOUT, interval=READY_INTERVAL,
                  abort=None, on_tick=None):
    """Poll *url* until it answers. Returns ``(ok, description)``.

    Any HTTP response — including 404 or 500 — proves the server is listening,
    which is what readiness means here; the status code is reported so a
    non-2xx answer is still visible to the operator.
    """
    deadline = time.monotonic() + timeout
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    last = 'no connection'
    attempts = 0
    while time.monotonic() < deadline:
        if abort is not None and abort():
            return False, 'aborted: the server process exited'
        attempts += 1
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                response.read(2048)
                return True, 'HTTP %d' % response.status
        except urllib.error.HTTPError as exc:
            return True, 'HTTP %d' % exc.code
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as exc:
            last = getattr(exc, 'reason', exc)
            last = str(last) or exc.__class__.__name__
        if on_tick is not None:
            on_tick(attempts, max(0.0, deadline - time.monotonic()))
        time.sleep(interval)
    return False, 'timed out after %.0fs (last error: %s)' % (timeout, last)


def download_file(url, dest, timeout=90):
    """Stream a URL to disk with a single-line progress indicator."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        show = style.enabled and sys.stdout.isatty()
    except (AttributeError, ValueError):
        show = False
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total = int(response.headers.get('Content-Length') or 0)
            got = 0
            last_render = 0.0
            with temp.open('wb') as handle:
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    handle.write(chunk)
                    got += len(chunk)
                    now = time.monotonic()
                    if show and now - last_render >= 0.15:
                        last_render = now
                        _render_progress(got, total, now - started)
            if show:
                _render_progress(got, total, time.monotonic() - started, final=True)
        temp.replace(dest)
        return got
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        temp.unlink(missing_ok=True)
        raise LauncherError(
            'download failed: %s' % dest.name,
            detail='%s\nfrom %s' % (exc, url),
            hint='Check your internet connection / proxy, or download the file '
                 'manually and save it as %s' % rel(dest),
        ) from exc


def _render_progress(got, total, elapsed, final=False):
    rate = got / elapsed if elapsed > 0 else 0
    if total:
        frac = min(1.0, got / total)
        width = 28
        filled = int(width * frac)
        bar = '#' * filled + '-' * (width - filled)
        line = '    [%s] %5.1f%%  %s / %s  %s/s' % (
            bar, frac * 100, human_bytes(got), human_bytes(total), human_bytes(rate))
    else:
        line = '    %s downloaded  %s/s' % (human_bytes(got), human_bytes(rate))
    if _STDOUT_BROKEN:
        return
    with PRINT_LOCK:
        try:
            sys.stdout.write('\r' + line[:110].ljust(112))
            if final:
                sys.stdout.write('\n')
            sys.stdout.flush()
        except (BrokenPipeError, ValueError):
            silence_stdout()


# ---------------------------------------------------------------------------
# OS abstraction.  Every platform-dependent decision lives here so the phase
# code below reads the same on all three operating systems, and so that
# --simulate-os can ask "what WOULD you do on Windows?" without being there.
# ---------------------------------------------------------------------------
class OSProfile:
    """Platform-specific paths, binaries, spawn flags and kill strategy."""

    ALIASES = {
        'windows': 'windows', 'win32': 'windows', 'win': 'windows', 'nt': 'windows',
        'darwin': 'darwin', 'macos': 'darwin', 'mac': 'darwin', 'osx': 'darwin',
        'linux': 'linux', 'linux2': 'linux',
    }

    def __init__(self, system=None, simulated=False):
        raw = (system or platform.system() or 'linux').strip().lower()
        self.name = self.ALIASES.get(raw, raw)
        self.simulated = bool(simulated)

    @property
    def is_windows(self):
        return self.name == 'windows'

    @property
    def is_macos(self):
        return self.name == 'darwin'

    @property
    def label(self):
        return {'windows': 'Windows', 'darwin': 'macOS', 'linux': 'Linux'}.get(
            self.name, self.name.title())

    # -- paths --------------------------------------------------------------
    @property
    def display_root(self):
        """Root used when printing paths (synthetic while simulating Windows)."""
        if self.simulated and self.is_windows:
            return PureWindowsPath(r'C:\Users\you\FYP')
        return REPO_ROOT

    def display(self, path):
        """Render *path* the way the target OS would show it."""
        path = Path(path)
        if not self.is_windows:
            return str(path)
        try:
            relative = path.resolve().relative_to(REPO_ROOT)
        except (ValueError, OSError):
            return str(PureWindowsPath(path))
        return str(PureWindowsPath(self.display_root) / PureWindowsPath(str(relative)))

    def venv_python(self, venv_dir=VENV_DIR):
        """backend\\.venv\\Scripts\\python.exe vs backend/.venv/bin/python."""
        venv_dir = Path(venv_dir)
        if self.is_windows:
            return venv_dir / 'Scripts' / 'python.exe'
        return venv_dir / 'bin' / 'python'

    def venv_pip(self, venv_dir=VENV_DIR):
        """The pip *executable*. Note that we normally prefer `python -m pip`,
        which survives a venv being relocated; this path exists because the
        spec calls for it and because it is what users look for."""
        venv_dir = Path(venv_dir)
        if self.is_windows:
            return venv_dir / 'Scripts' / 'pip.exe'
        return venv_dir / 'bin' / 'pip'

    # -- binaries -----------------------------------------------------------
    def npm(self):
        """Resolved npm path, or None. Windows npm is a .cmd batch shim."""
        return which_npm(self.is_windows)

    def npm_display(self):
        """What `npm` resolves to. While simulating a foreign OS we report the
        binary *name* that would be looked up, not this machine's path."""
        default = 'npm.cmd' if self.is_windows else 'npm'
        if self.simulated and self.name != OSProfile(None).name:
            return default
        return self.npm() or default

    # -- process control ----------------------------------------------------
    def popen_kwargs(self):
        """Spawn children in their own group so we can kill the whole tree.

        Django's autoreloader forks a worker, and Vite spawns esbuild helpers;
        signalling only the direct child would leave orphans behind.
        """
        if self.is_windows:
            return {'creationflags': getattr(
                subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x00000200)}
        return {'start_new_session': True}

    def popen_kwargs_display(self):
        if self.is_windows:
            return 'creationflags=subprocess.CREATE_NEW_PROCESS_GROUP (0x200)'
        return 'start_new_session=True  (child becomes its own process group leader)'

    def terminate_display(self):
        if self.is_windows:
            return ('proc.send_signal(CTRL_BREAK_EVENT) -> wait 10s -> '
                    'taskkill /F /T /PID <pid>')
        return ('os.killpg(pgid, SIGTERM) -> wait 10s -> os.killpg(pgid, SIGKILL)')

    def signal_child(self, proc, hard=False):
        """Ask a child (and its whole tree) to stop. Never raises."""
        if proc.poll() is not None:
            return
        try:
            if self.is_windows:
                if hard:
                    subprocess.run(
                        ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                        capture_output=True, text=True, errors='replace', timeout=20,
                    )
                else:
                    proc.send_signal(getattr(signal, 'CTRL_BREAK_EVENT', signal.SIGTERM))
            else:
                sig = signal.SIGKILL if hard else signal.SIGTERM
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill() if hard else proc.terminate()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            warn('could not signal pid %s: %s' % (proc.pid, exc))


# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------
def run_cmd(cmd, *, cwd=None, env=None, what=None, hint='', stream=False,
            timeout=None, check=True, echo=False):
    """Run a command, always capturing merged stdout+stderr.

    Returns ``(returncode, output)``.  When *check* is true a non-zero exit
    raises :class:`LauncherError` carrying the command, the exit code and the
    tail of the output — there are no silent failures anywhere in this file.
    """
    cmd = [str(part) for part in cmd]
    label = what or Path(cmd[0]).name
    if echo or stream:
        note('$ ' + format_cmd(cmd) + (' (cwd: %s)' % rel(cwd) if cwd else ''))

    try:
        proc = popen_with_fallback(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors='replace',
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise LauncherError(
            '%s could not be started: executable not found' % label,
            detail=str(exc), cmd=cmd,
            hint='"%s" is not on PATH. Install it, or open a new terminal so a '
                 'freshly installed tool is picked up.' % cmd[0],
        ) from exc
    except (OSError, ValueError) as exc:
        raise LauncherError(
            '%s could not be started' % label, detail=str(exc), cmd=cmd,
            hint='Check the file exists and is executable.',
        ) from exc

    killer = None
    timed_out = {'hit': False}
    if timeout:
        def _kill_on_timeout():
            timed_out['hit'] = True
            try:
                proc.kill()
            except OSError:
                pass
        killer = threading.Timer(timeout, _kill_on_timeout)
        killer.daemon = True
        killer.start()

    collected = deque(maxlen=600)
    try:
        if proc.stdout is not None:
            for raw in iter(proc.stdout.readline, ''):
                line = ANSI_RE.sub('', raw.rstrip('\r\n'))
                collected.append(line)
                if stream and line.strip():
                    emit(style('      | ', 'grey') + line)
            proc.stdout.close()
        returncode = proc.wait()
    finally:
        if killer is not None:
            killer.cancel()

    output = '\n'.join(collected)
    if timed_out['hit']:
        raise LauncherError(
            '%s timed out after %.0fs' % (label, timeout),
            cmd=cmd, output=output,
            hint='The command hung. Re-run with --verbose to watch it live, or '
                 'check for a network/proxy problem.',
        )
    if check and returncode != 0:
        raise LauncherError(
            '%s failed (exit code %d)' % (label, returncode),
            cmd=cmd, output=output, hint=hint,
        )
    return returncode, output


def stream_process(proc, label, colour, log_path, ring):
    """Pump a child's merged output to the console AND to a log file.

    Console lines are prefixed and colour-coded (``[api]`` / ``[web]``); the
    log file gets the same lines with ANSI escapes stripped.  The last lines
    are also kept in *ring* so a crash can be explained without opening a file.
    """
    def _pump():
        handle = None
        try:
            handle = open(log_path, 'a', encoding='utf-8', errors='replace')
        except OSError as exc:
            warn('cannot write %s: %s' % (rel(log_path), exc))
        prefix = style('[' + label + ']', colour, 'bold')
        try:
            stream = proc.stdout
            if stream is None:
                return
            for raw in iter(stream.readline, ''):
                line = raw.rstrip('\r\n')
                plain = ANSI_RE.sub('', line)
                ring.append(plain)
                if handle is not None:
                    handle.write(plain + '\n')
                    handle.flush()
                emit(prefix + ' ' + line)
        except (OSError, ValueError):
            pass  # the pipe closed while we were reading: the child is gone
        finally:
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except (OSError, ValueError):
                pass

    thread = threading.Thread(target=_pump, name='stream-' + label, daemon=True)
    thread.start()
    return thread


class ManagedProcess:
    """A supervised dev server: spawn, tee its output, stop it without orphans."""

    def __init__(self, label, colour, cmd, cwd, env, log_path, profile):
        self.label = label
        self.colour = colour
        self.cmd = [str(part) for part in cmd]
        self.cwd = Path(cwd)
        self.env = env
        self.log_path = Path(log_path)
        self.profile = profile
        self.proc = None
        self.ring = deque(maxlen=200)
        self.thread = None
        self.stopped_by_us = False

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.log_path.open('w', encoding='utf-8') as handle:
                handle.write('# %s | %s\n# cwd: %s\n# started: %s\n\n' % (
                    self.label, format_cmd(self.cmd), self.cwd,
                    time.strftime('%Y-%m-%d %H:%M:%S')))
        except OSError as exc:
            warn('cannot create %s: %s' % (rel(self.log_path), exc))

        try:
            self.proc = popen_with_fallback(
                self.cmd,
                cwd=str(self.cwd),
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors='replace',
                bufsize=1,
                **self.profile.popen_kwargs()
            )
        except FileNotFoundError as exc:
            raise LauncherError(
                'cannot start the %s server: %s not found' % (self.label, self.cmd[0]),
                detail=str(exc), cmd=self.cmd,
                hint='The executable is missing from PATH. Re-run the launcher '
                     'so dependencies are installed, or install it manually.',
            ) from exc
        except (OSError, ValueError) as exc:
            raise LauncherError(
                'cannot start the %s server' % self.label,
                detail=str(exc), cmd=self.cmd,
                hint='Check %s exists and is executable.' % self.cmd[0],
            ) from exc

        self.thread = stream_process(
            self.proc, self.label, self.colour, self.log_path, self.ring)
        return self.proc

    @property
    def pid(self):
        return self.proc.pid if self.proc else None

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def returncode(self):
        return self.proc.poll() if self.proc else None

    def tail(self, count=30):
        return list(self.ring)[-count:]

    def stop(self, grace=10.0):
        """Polite stop, then a forced one. Safe to call twice."""
        if self.proc is None or self.proc.poll() is not None:
            return
        self.stopped_by_us = True
        self.profile.signal_child(self.proc, hard=False)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                break
            time.sleep(0.1)
        if self.proc.poll() is None:
            warn('%s did not stop in %.0fs — forcing it' % (self.label, grace))
            self.profile.signal_child(self.proc, hard=True)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                warn('%s (pid %s) is still alive; kill it manually' % (
                    self.label, self.proc.pid))
        if self.thread is not None:
            self.thread.join(timeout=3)


# ---------------------------------------------------------------------------
# Numbered phase output
# ---------------------------------------------------------------------------
class Step:
    """``==> [3/10] Title`` ... ``✓ done in 1.2s``, with skip support."""

    def __init__(self, index, total, title):
        self.index = index
        self.total = total
        self.title = title
        self.started = 0.0
        self.skipped = None
        self.summary = None
        self.failed = False
        self.soft_failed = False

    def __enter__(self):
        self.started = time.monotonic()
        emit('')
        emit('%s %s %s' % (
            style(GLYPH_ARROW, 'blue', 'bold'),
            style('[%d/%d]' % (self.index, self.total), 'bold'),
            style(self.title, 'bold')))
        return self

    def skip(self, reason):
        self.skipped = reason

    def done(self, summary):
        self.summary = summary

    def problem(self, summary):
        """Mark the step as failed without raising — the caller records why."""
        self.soft_failed = True
        self.summary = summary

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.failed = True
            emit('  %s %s %s' % (
                style(GLYPH_BAD, 'red', 'bold'),
                style(self.title + ' failed', 'red'),
                style('(%.1fs)' % self.elapsed, 'grey')))
            return False
        if self.soft_failed:
            self.failed = True
            emit('  %s %s %s' % (
                style(GLYPH_BAD, 'red', 'bold'),
                style(self.summary or (self.title + ' failed'), 'red'),
                style('(%.1fs)' % self.elapsed, 'grey')))
        elif self.skipped:
            emit('  %s %s %s' % (
                style(GLYPH_SKIP, 'grey'),
                style('skipped — ' + self.skipped, 'grey'),
                style('(%.1fs)' % self.elapsed, 'grey')))
        else:
            emit('  %s %s %s' % (
                style(GLYPH_OK, 'green', 'bold'),
                self.summary or 'done',
                style('(%.1fs)' % self.elapsed, 'grey')))
        return False


def popen_with_fallback(cmd, **kwargs):
    """``subprocess.Popen`` that copes with Windows batch shims.

    ``npm`` on Windows is ``npm.cmd``.  Direct execution normally works, but on
    some hosts CreateProcess refuses a batch file with WinError 193 — in that
    case we re-issue the command through ``%COMSPEC% /d /s /c "<line>"``, which
    is the documented way to run a .cmd and keeps the child inside our process
    group so Ctrl-Break / taskkill still reach it.
    """
    try:
        return subprocess.Popen(cmd, **kwargs)
    except OSError as exc:
        is_batch = str(cmd[0]).lower().endswith(('.cmd', '.bat'))
        if os.name == 'nt' and is_batch and getattr(exc, 'winerror', None) in (193, 216):
            comspec = os.environ.get('COMSPEC') or 'cmd.exe'
            line = '%s /d /s /c "%s"' % (comspec, format_cmd(cmd))
            return subprocess.Popen(line, **kwargs)
        raise


# ---------------------------------------------------------------------------
# The launcher
# ---------------------------------------------------------------------------
class Launcher:
    """Runs the numbered phases and supervises the two dev servers."""

    def __init__(self, args):
        self.args = args
        self.profile = OSProfile(
            getattr(args, 'simulate_os', None),
            simulated=bool(getattr(args, 'simulate_os', None)),
        )
        self.verbose = bool(args.verbose)
        self.venv_python = self.profile.venv_python(VENV_DIR)
        self.venv_pip = self.profile.venv_pip(VENV_DIR)
        self.host_python = None          # argv able to build the venv
        self.host_python_version = None
        self.npm = None
        self.node_version = None
        self.npm_version = None
        self.api_port = int(args.api_port or DEFAULT_API_PORT)
        self.web_port = int(args.web_port or DEFAULT_WEB_PORT)
        self.processes = []
        self.warnings = []
        self.failures = []
        self.stop_requested = threading.Event()
        self.shutdown_done = threading.Event()
        self.started_at = time.monotonic()

    # -- small utilities ----------------------------------------------------
    @property
    def wants_backend(self):
        return not self.args.frontend_only

    @property
    def wants_frontend(self):
        return not self.args.backend_only

    def warn(self, message):
        self.warnings.append(message)
        warn(message)

    def base_env(self, extra=None):
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env.setdefault('PYTHONIOENCODING', 'utf-8')
        if extra:
            env.update({str(k): str(v) for k, v in extra.items()})
        return env

    def run(self, cmd, **kwargs):
        kwargs.setdefault('stream', self.verbose)
        kwargs.setdefault('echo', self.verbose)
        return run_cmd(cmd, **kwargs)

    def python_cmd(self, *args):
        """Argv for "the backend interpreter" — venv first, host as fallback."""
        if self.venv_python.exists():
            return [str(self.venv_python), *[str(a) for a in args]]
        if self.host_python:
            return [*self.host_python, *[str(a) for a in args]]
        return [sys.executable, *[str(a) for a in args]]

    def credentials(self):
        """Demo/admin logins, overridable by tools/seed_credentials.json."""
        creds = {key: dict(value) for key, value in DEFAULT_CREDENTIALS.items()}
        if CREDENTIALS_FILE.is_file():
            try:
                data = json.loads(CREDENTIALS_FILE.read_text(encoding='utf-8'))
                for key in ('demo', 'admin'):
                    entry = data.get(key)
                    if isinstance(entry, dict) and entry.get('email'):
                        creds[key] = {
                            'email': str(entry.get('email')),
                            'password': str(entry.get('password', '')),
                        }
            except (OSError, ValueError, AttributeError) as exc:
                self.warn('ignoring unreadable %s (%s)' % (rel(CREDENTIALS_FILE), exc))
        return creds

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    def build_plan(self):
        plan = [('Preflight — toolchain and environment', self.phase_preflight)]
        if self.wants_backend:
            plan.append(('Backend virtual environment', self.phase_backend_venv))
        if self.wants_frontend:
            plan.append(('Frontend dependencies', self.phase_frontend_deps))
        if self.wants_backend:
            plan.append(('ML assets — detector weights + smoke model',
                         self.phase_ml_assets))
            plan.append(('Database — migrate and seed', self.phase_database))
            plan.append(('Sample media', self.phase_sample_media))
        if self.args.check:
            plan.append(('Test suites', self.phase_tests))
            return plan
        plan.append(('Start servers', self.phase_start_servers))
        plan.append(('Readiness', self.phase_readiness))
        plan.append(('Open browser', self.phase_open_browser))
        plan.append(('Supervise', self.phase_supervise))
        return plan

    def header(self):
        emit('')
        emit(style('  AutoSmokeGuard  ', 'bold', 'cyan')
             + style('one-command developer launcher', 'cyan'))
        emit(style('  ' + '-' * 70, 'grey'))
        mode = []
        if self.args.check:
            mode.append('check-only')
        if self.args.backend_only:
            mode.append('backend-only')
        if self.args.frontend_only:
            mode.append('frontend-only')
        if self.args.skip_ml:
            mode.append('skip-ml')
        if self.args.reinstall:
            mode.append('reinstall')
        if self.args.reset_db:
            mode.append('reset-db')
        note('  repo: %s' % REPO_ROOT)
        if mode:
            note('  mode: ' + ', '.join(mode))

    def execute(self):
        """Run the plan. Returns a process exit code."""
        self.header()
        plan = self.build_plan()
        total = len(plan)
        for index, (title, handler) in enumerate(plan, start=1):
            with Step(index, total, title) as step:
                handler(step)
        return 1 if self.failures else 0

    # ------------------------------------------------------------------
    # Phase 1 — preflight
    # ------------------------------------------------------------------
    def phase_preflight(self, step):
        if not BACKEND_DIR.is_dir() or not MANAGE_PY.is_file():
            raise LauncherError(
                'this does not look like the AutoSmokeGuard repository',
                detail='Expected to find %s' % MANAGE_PY,
                hint='Run start.py from the repository root — the one that '
                     'contains the backend/ and frontend/ folders.',
            )

        info('%-12s %s %s (%s)' % (
            'platform:', self.profile.label, platform.release(), platform.machine()))
        info('%-12s %s' % ('launcher:', 'python %s at %s' % (
            platform.python_version(), sys.executable)))

        # --- interpreter for the backend venv ---
        self.host_python, self.host_python_version, source = find_python(
            self.venv_python, self.profile.is_windows, self.verbose)
        # Defense in depth: find_python() already filters every candidate
        # against MIN_PYTHON internally, so this should never actually fire
        # — but the whole point of a *preflight* floor check is that it is
        # not allowed to depend on nobody ever changing that internal
        # filter. This is the line that turns a stray regression into a
        # clear failure here rather than an opaque pip error minutes later.
        check_python_floor(self.host_python_version, source)
        info('%-12s %s  (%d.%d, from %s)' % (
            'python:', format_cmd(self.host_python),
            self.host_python_version[0], self.host_python_version[1], source))

        # --- node + npm ---
        node = shutil.which('node')
        self.npm = self.profile.npm()
        if node:
            _, out = self.run([node, '--version'], what='node --version')
            self.node_version = (out.strip().splitlines() or [''])[-1].strip()
            match = re.search(r'(\d+)', self.node_version)
            major = int(match.group(1)) if match else 0
            info('%-12s %s' % ('node:', self.node_version or 'unknown'))
            if major and major < MIN_NODE_MAJOR and self.wants_frontend:
                raise LauncherError(
                    'Node.js %s is too old (need >= %d)' % (
                        self.node_version, MIN_NODE_MAJOR),
                    hint='Install the current LTS from %s, or use nvm: '
                         '`nvm install --lts`. Vite 8 requires Node 18+.'
                         % NODE_DOWNLOAD_URL,
                )
        elif self.wants_frontend:
            raise LauncherError(
                'Node.js is not installed (or not on PATH)',
                hint='Install Node.js LTS (>= %d) from %s, then open a new '
                     'terminal and re-run this launcher.'
                     % (MIN_NODE_MAJOR, NODE_DOWNLOAD_URL),
            )
        else:
            self.warn('node not found — fine for --backend-only')

        if self.npm:
            _, out = self.run([self.npm, '--version'], what='npm --version')
            self.npm_version = (out.strip().splitlines() or [''])[-1].strip()
            info('%-12s %s  (%s)' % ('npm:', self.npm_version or 'unknown', self.npm))
        elif self.wants_frontend:
            raise LauncherError(
                'npm is not installed (or not on PATH)',
                detail='Looked for %s' % ('npm.cmd / npm' if self.profile.is_windows
                                          else 'npm'),
                hint='npm ships with Node.js — install Node LTS from %s. On '
                     'Windows make sure the installer added nodejs to PATH, '
                     'then open a NEW terminal.' % NODE_DOWNLOAD_URL,
            )
        else:
            self.warn('npm not found — fine for --backend-only')

        # --- disk ---
        try:
            usage = shutil.disk_usage(str(REPO_ROOT))
            free_mb = usage.free / (1024 * 1024)
            info('%-12s %s free of %s' % (
                'disk:', human_bytes(usage.free), human_bytes(usage.total)))
            if free_mb < MIN_FREE_DISK_MB:
                self.warn('only %s free — the venv + node_modules + torch '
                          'wheels need roughly 2 GB' % human_bytes(usage.free))
        except OSError as exc:
            self.warn('could not read disk usage: %s' % exc)

        if self.wants_frontend and not PACKAGE_JSON.is_file():
            raise LauncherError(
                'frontend/package.json is missing',
                detail='Expected %s' % PACKAGE_JSON,
                hint='Clone the frontend repository into %s, or run with '
                     '--backend-only.' % FRONTEND_DIR,
            )

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        step.done('environment looks good')

    # ------------------------------------------------------------------
    # Phase 2 — backend virtualenv + python dependencies
    # ------------------------------------------------------------------
    def phase_backend_venv(self, step):
        if not REQUIREMENTS.is_file():
            raise LauncherError(
                'backend/requirements.txt is missing',
                detail='Expected %s' % REQUIREMENTS,
                hint='Restore the file from git: git -C backend checkout -- '
                     'requirements.txt',
            )

        created = False
        if not self.venv_python.exists():
            if VENV_DIR.exists():
                self.warn('%s exists but has no interpreter — rebuilding it'
                          % rel(VENV_DIR))
            info('creating virtualenv at %s' % rel(VENV_DIR))
            self._create_venv(clear=VENV_DIR.exists())
            created = True
            if not self.venv_python.exists():
                raise LauncherError(
                    'the virtualenv was created but %s is missing'
                    % self.profile.display(self.venv_python),
                    hint='Delete %s and re-run. On Debian/Ubuntu you may need '
                         '`sudo apt install python3-venv`.' % rel(VENV_DIR),
                )
        else:
            info('virtualenv: %s' % self.profile.display(self.venv_python))

        digest = sha256_file(REQUIREMENTS)
        stamped = read_stamp(BACKEND_STAMP)
        if not created and not self.args.reinstall and stamped and stamped == digest:
            info('requirements.txt sha256 %s matches the install stamp' % digest[:12])
            step.skip('python dependencies already up to date')
            return

        reason = ('new virtualenv' if created else
                  '--reinstall' if self.args.reinstall else
                  'requirements.txt changed' if stamped else 'no install stamp yet')
        info('installing python dependencies (%s)' % reason)

        self.run(
            self.python_cmd('-m', 'pip', 'install', '--upgrade', 'pip'),
            cwd=BACKEND_DIR, env=self.base_env(), what='pip self-upgrade',
            timeout=900,
            hint='pip could not reach PyPI. Check your network/proxy '
                 '(HTTPS_PROXY) and try again.',
        )
        self.run(
            self.python_cmd('-m', 'pip', 'install', '-r', str(REQUIREMENTS)),
            cwd=BACKEND_DIR, env=self.base_env(),
            what='pip install -r requirements.txt',
            stream=True, timeout=3600,
            hint='A dependency failed to build or download. Read the pip output '
                 'above: the usual causes are no network access, a Python '
                 'version with no matching wheel (try 3.12), or a missing C '
                 'toolchain. Re-run with --reinstall after fixing it.',
        )
        write_stamp(BACKEND_STAMP, digest,
                    requirements=str(REQUIREMENTS.name),
                    python='%d.%d' % self.host_python_version
                    if self.host_python_version else '')
        step.done('python dependencies installed')

    def _create_venv(self, clear=False):
        """Build the venv with the selected interpreter."""
        if self.host_python and len(self.host_python) == 1 and \
                Path(self.host_python[0]).resolve() == Path(sys.executable).resolve():
            # Same interpreter as the one running us: use the stdlib API
            # directly, which avoids a subprocess and works on frozen hosts.
            try:
                builder = venv.EnvBuilder(with_pip=True, clear=clear, upgrade_deps=False)
                builder.create(str(VENV_DIR))
                return
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                self.warn('venv module failed (%s) — retrying via subprocess' % exc)
        cmd = list(self.host_python or [sys.executable]) + ['-m', 'venv']
        if clear:
            cmd.append('--clear')
        cmd.append(str(VENV_DIR))
        self.run(
            cmd, cwd=BACKEND_DIR, what='python -m venv', timeout=600,
            hint='Creating the virtualenv failed. On Debian/Ubuntu install '
                 '`python3-venv`; on Windows re-run the Python installer and '
                 'enable "pip". You can also delete backend/.venv and retry.',
        )

    # ------------------------------------------------------------------
    # Phase 3 — frontend dependencies
    # ------------------------------------------------------------------
    def phase_frontend_deps(self, step):
        if not self.npm:
            raise LauncherError('npm is unavailable', hint='See the preflight step.')

        source = PACKAGE_LOCK if PACKAGE_LOCK.is_file() else PACKAGE_JSON
        digest = sha256_file(source)
        stamped = read_stamp(FRONTEND_STAMP)
        installed = NODE_MODULES.is_dir() and any(NODE_MODULES.iterdir())

        if installed and not self.args.reinstall and stamped and stamped == digest:
            info('%s sha256 %s matches the install stamp' % (source.name, digest[:12]))
            step.skip('node_modules already up to date')
            return

        reason = ('--reinstall' if self.args.reinstall else
                  'node_modules missing' if not installed else
                  '%s changed' % source.name if stamped else 'no install stamp yet')
        use_ci = PACKAGE_LOCK.is_file()
        info('installing frontend dependencies with `npm %s` (%s)'
             % ('ci' if use_ci else 'install', reason))

        env = self.base_env({'npm_config_fund': 'false', 'npm_config_audit': 'false'})
        if use_ci:
            try:
                self.run([self.npm, 'ci'], cwd=FRONTEND_DIR, env=env,
                         what='npm ci', stream=True, timeout=3600)
            except LauncherError as exc:
                # The classic cause is a package.json edited without refreshing
                # package-lock.json; npm install repairs that.
                self.warn('npm ci failed (%s) — falling back to npm install'
                          % exc.message)
                self.run([self.npm, 'install'], cwd=FRONTEND_DIR, env=env,
                         what='npm install', stream=True, timeout=3600,
                         hint='npm could not install the dependency tree. Try '
                              'deleting frontend/node_modules and re-running, '
                              'or `npm cache clean --force`.')
        else:
            self.run([self.npm, 'install'], cwd=FRONTEND_DIR, env=env,
                     what='npm install', stream=True, timeout=3600,
                     hint='npm could not install the dependency tree. Check '
                          'your network and that frontend/package.json is valid.')

        # `npm install` may have *created* package-lock.json; stamp whatever
        # the authoritative source is now, so the next run is a clean skip.
        final = PACKAGE_LOCK if PACKAGE_LOCK.is_file() else PACKAGE_JSON
        write_stamp(FRONTEND_STAMP, sha256_file(final), source=final.name)
        step.done('frontend dependencies installed')

    # ------------------------------------------------------------------
    # Phase 4 — ML assets
    # ------------------------------------------------------------------
    def phase_ml_assets(self, step):
        if self.args.skip_ml:
            step.skip('--skip-ml')
            return

        ML_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

        if YOLO_WEIGHTS.is_file() and YOLO_WEIGHTS.stat().st_size > 1_000_000:
            info('detector: %s already present (%s)' % (
                YOLO_WEIGHTS.name, human_bytes(YOLO_WEIGHTS.stat().st_size)))
        else:
            info('detector: downloading %s' % YOLO_WEIGHTS.name)
            note('  from %s' % YOLO_URL)
            size = download_file(YOLO_URL, YOLO_WEIGHTS)
            if size < 1_000_000:
                YOLO_WEIGHTS.unlink(missing_ok=True)
                raise LauncherError(
                    'the downloaded YOLO checkpoint is implausibly small (%s)'
                    % human_bytes(size),
                    hint='A proxy or captive portal probably returned an HTML '
                         'page. Download %s manually and save it to %s.'
                         % (YOLO_URL, rel(YOLO_WEIGHTS)),
                )
            info('detector: saved %s (%s)' % (rel(YOLO_WEIGHTS), human_bytes(size)))

        if UNET_WEIGHTS.is_file() and UNET_WEIGHTS.stat().st_size > 0:
            info('segmenter: %s already present (%s)' % (
                UNET_WEIGHTS.name, human_bytes(UNET_WEIGHTS.stat().st_size)))
            step.done('ML assets ready')
            return

        training_pkg = BACKEND_DIR / 'mlcore' / 'training'
        if not (training_pkg / 'train_unet.py').is_file():
            raise LauncherError(
                'the smoke model is missing and the training code is not here yet',
                detail='Expected %s' % (training_pkg / 'train_unet.py'),
                hint='Re-run with --skip-ml to start the servers without the '
                     'smoke segmenter, or pull the branch that contains '
                     'backend/mlcore/training/.',
            )

        emit('')
        emit(style('    FIRST RUN: training the smoke segmentation model.', 'bold', 'yellow'))
        emit(style('    No pretrained smoke network exists publicly, so the project', 'yellow'))
        emit(style('    generates a synthetic dataset and trains a compact U-Net.', 'yellow'))
        emit(style('    This takes a few minutes and happens only once.', 'yellow'))
        emit('')

        env = self.base_env({'PYTHONPATH': str(BACKEND_DIR)})
        self.run(
            self.python_cmd('-m', SYNTH_MODULE), cwd=BACKEND_DIR, env=env,
            what='synthetic dataset generation', stream=True, timeout=7200,
            hint='Dataset generation failed. Check the traceback above; you can '
                 'skip the ML phase entirely with --skip-ml.',
        )
        self.run(
            self.python_cmd('-m', TRAIN_MODULE), cwd=BACKEND_DIR, env=env,
            what='smoke U-Net training', stream=True, timeout=14400,
            hint='Training failed. Check the traceback above (usually a missing '
                 'torch install — re-run with --reinstall). --skip-ml lets you '
                 'start the app without it.',
        )
        if not UNET_WEIGHTS.is_file():
            raise LauncherError(
                'training finished but %s was not written' % UNET_WEIGHTS.name,
                detail='Expected %s' % UNET_WEIGHTS,
                hint='Check the output path in backend/mlcore/training/train_unet.py.',
            )
        info('segmenter: trained and saved %s' % rel(UNET_WEIGHTS))
        step.done('ML assets ready')

    # ------------------------------------------------------------------
    # Phase 5 — database
    # ------------------------------------------------------------------
    def phase_database(self, step):
        if self.args.reset_db:
            self._reset_database()

        env = self.base_env()
        self.run(
            self.python_cmd('manage.py', 'migrate', '--noinput'),
            cwd=BACKEND_DIR, env=env, what='manage.py migrate',
            stream=self.verbose, timeout=900,
            hint='Migrations failed. Read the traceback above: a missing app '
                 'dependency means the venv is stale (--reinstall), and a '
                 'schema conflict is usually fixed with --reset-db.',
        )
        info('migrations applied (%s)' % rel(DB_FILE))

        if SEED_SCRIPT.is_file():
            self.run(
                self.python_cmd(str(SEED_SCRIPT)), cwd=BACKEND_DIR, env=env,
                what='tools/seed.py', stream=self.verbose, timeout=900,
                hint='Seeding failed. Re-run with --verbose to see the script '
                     'output; --reset-db gives it a clean database.',
            )
            info('demo data seeded via %s' % rel(SEED_SCRIPT))
        else:
            self.warn('%s not found — skipping the demo data seed'
                      % rel(SEED_SCRIPT))
        step.done('database ready')

    def _reset_database(self):
        if not self.args.yes:
            if not sys.stdin or not sys.stdin.isatty():
                raise LauncherError(
                    '--reset-db needs confirmation but stdin is not a terminal',
                    hint='Re-run as: python start.py --reset-db --yes',
                )
            emit('')
            emit(style('    --reset-db will DELETE %s and every uploaded record '
                       'in it.' % rel(DB_FILE), 'bold', 'yellow'))
            try:
                answer = input('    Type "yes" to continue: ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ''
            if answer != 'yes':
                raise LauncherError(
                    'database reset cancelled',
                    hint='Re-run without --reset-db to keep the existing data.',
                )
        if DB_FILE.exists():
            try:
                DB_FILE.unlink()
                info('deleted %s' % rel(DB_FILE))
            except OSError as exc:
                raise LauncherError(
                    'could not delete %s' % rel(DB_FILE), detail=str(exc),
                    hint='Stop any running Django server (it holds the file '
                         'open on Windows) and try again.',
                ) from exc
        else:
            info('%s did not exist — nothing to delete' % rel(DB_FILE))

    # ------------------------------------------------------------------
    # Phase 6 — sample media
    # ------------------------------------------------------------------
    def phase_sample_media(self, step):
        existing = []
        if SAMPLE_MEDIA_DIR.is_dir():
            existing = [p for p in SAMPLE_MEDIA_DIR.rglob('*') if p.is_file()]
        if existing:
            info('%d sample file(s) already in %s' % (len(existing), rel(SAMPLE_MEDIA_DIR)))
            step.skip('sample media already generated')
            return
        if not SAMPLES_SCRIPT.is_file():
            self.warn('%s not found — no demo images/videos will be available'
                      % rel(SAMPLES_SCRIPT))
            step.skip('generator script not present yet')
            return
        info('generating demo images and videos')
        self.run(
            self.python_cmd(str(SAMPLES_SCRIPT)), cwd=BACKEND_DIR,
            env=self.base_env(), what='tools/make_samples.py',
            stream=True, timeout=1800,
            hint='Sample generation failed — check the traceback above. The '
                 'app still runs without it; you just have to supply your own '
                 'test media.',
        )
        made = [p for p in SAMPLE_MEDIA_DIR.rglob('*')] if SAMPLE_MEDIA_DIR.is_dir() else []
        step.done('sample media ready (%d file(s))' % len([p for p in made if p.is_file()]))

    # ------------------------------------------------------------------
    # --check — phases 1-6 plus the test suites
    # ------------------------------------------------------------------
    def phase_tests(self, step):
        ran = []
        if self.wants_backend:
            if not self.venv_python.exists():
                self.failures.append('backend tests: no virtualenv interpreter')
            else:
                code, _ = run_cmd(self.python_cmd('-c', 'import pytest'),
                                  cwd=BACKEND_DIR, check=False)
                if code != 0:
                    self.warn('pytest is not installed in the venv — skipping '
                              'the backend suite (add it to requirements.txt)')
                else:
                    info('running backend tests: pytest')
                    try:
                        self.run(self.python_cmd('-m', 'pytest', '-q'),
                                 cwd=BACKEND_DIR, env=self.base_env(),
                                 what='pytest', stream=True, timeout=3600)
                        ran.append('pytest')
                    except LauncherError as exc:
                        emit(exc.render())
                        self.failures.append('backend tests failed')

        if self.wants_frontend:
            scripts = {}
            try:
                scripts = json.loads(
                    PACKAGE_JSON.read_text(encoding='utf-8')).get('scripts', {})
            except (OSError, ValueError) as exc:
                self.warn('cannot read frontend/package.json scripts: %s' % exc)
            if 'test' not in scripts:
                self.warn('frontend package.json has no "test" script — '
                          'skipping the UI suite')
            elif not self.npm:
                self.failures.append('frontend tests: npm unavailable')
            else:
                info('running frontend tests: npm run test')
                try:
                    # CI=1 makes vitest run once and exit instead of watching.
                    self.run([self.npm, 'run', 'test', '--silent'],
                             cwd=FRONTEND_DIR,
                             env=self.base_env({'CI': '1'}),
                             what='npm run test', stream=True, timeout=3600)
                    ran.append('npm run test')
                except LauncherError as exc:
                    emit(exc.render())
                    self.failures.append('frontend tests failed')

        if self.failures:
            step.problem('%d check(s) failed' % len(self.failures))
        elif ran:
            step.done('passed: ' + ', '.join(ran))
        else:
            step.skip('no test suites available yet')

    # ------------------------------------------------------------------
    # Ports
    # ------------------------------------------------------------------
    def _resolve_ports(self):
        """Pick ports that are actually free and report any change."""
        if self.wants_backend:
            chosen = free_port(self.api_port)
            if chosen != self.api_port:
                self.warn('port %d is busy — the API will use %d instead'
                          % (self.api_port, chosen))
            self.api_port = chosen
        if self.wants_frontend:
            start = self.web_port if self.web_port != self.api_port else self.web_port + 1
            chosen = free_port(start)
            if chosen != self.web_port:
                self.warn('port %d is busy — the UI will use %d instead'
                          % (self.web_port, chosen))
            self.web_port = chosen

    @property
    def api_url(self):
        return 'http://127.0.0.1:%d' % self.api_port

    @property
    def web_url(self):
        return 'http://127.0.0.1:%d' % self.web_port

    # ------------------------------------------------------------------
    # Phase 7 — start the dev servers
    # ------------------------------------------------------------------
    def api_command(self):
        return self.python_cmd('manage.py', 'runserver',
                               '127.0.0.1:%d' % self.api_port)

    def web_command(self):
        # `npm run dev -- <args>` forwards the flags to vite itself.
        # --strictPort makes Vite fail loudly instead of silently hopping to
        # another port, which would break the proxy contract we just printed.
        return [self.npm or 'npm', 'run', 'dev', '--',
                '--port', str(self.web_port), '--strictPort',
                '--host', '127.0.0.1']

    def web_env(self):
        extra = {
            'VITE_API_PROXY': self.api_url,
            'VITE_API_URL': self.api_url,
            'BROWSER': 'none',
        }
        if style.enabled:
            extra['FORCE_COLOR'] = '1'
        return self.base_env(extra)

    def phase_start_servers(self, step):
        self._resolve_ports()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._install_signal_handlers()

        if self.wants_backend:
            proc = ManagedProcess(
                'api', 'cyan', self.api_command(), BACKEND_DIR,
                self.base_env({'PYTHONPATH': str(BACKEND_DIR)}),
                API_LOG, self.profile)
            info('api : %s' % format_cmd(proc.cmd))
            note('      log -> %s' % rel(API_LOG))
            proc.start()
            self.processes.append(proc)

        if self.wants_frontend:
            proc = ManagedProcess(
                'web', 'magenta', self.web_command(), FRONTEND_DIR,
                self.web_env(), WEB_LOG, self.profile)
            info('web : %s' % format_cmd(proc.cmd))
            note('      VITE_API_PROXY=%s' % self.api_url)
            note('      log -> %s' % rel(WEB_LOG))
            proc.start()
            self.processes.append(proc)

        time.sleep(0.4)  # let the first log lines land under the step header
        for proc in self.processes:
            if not proc.is_running():
                raise LauncherError(
                    'the %s server exited immediately (code %s)'
                    % (proc.label, proc.returncode()),
                    cmd=proc.cmd, output='\n'.join(proc.tail(30)),
                    hint='Read the output above. Common causes: a syntax error '
                         'in the app code, a missing dependency (--reinstall), '
                         'or a port grabbed by another process.',
                )
        step.done('%d server(s) started (pids: %s)' % (
            len(self.processes), ', '.join(str(p.pid) for p in self.processes)))

    # ------------------------------------------------------------------
    # Phase 8 — readiness
    # ------------------------------------------------------------------
    def phase_readiness(self, step):
        targets = []
        by_label = {p.label: p for p in self.processes}
        if self.wants_backend:
            targets.append(('api', self.api_url + HEALTH_PATH))
        if self.wants_frontend:
            targets.append(('web', self.web_url + '/'))

        for label, url in targets:
            proc = by_label.get(label)
            info('waiting for %s at %s' % (label, url))
            ok, detail = wait_for_http(
                url, timeout=READY_TIMEOUT, interval=READY_INTERVAL,
                abort=(lambda p=proc: p is not None and not p.is_running()))
            if not ok:
                tail = '\n'.join(proc.tail(30)) if proc else ''
                raise LauncherError(
                    'the %s server never became ready' % label,
                    detail='%s -> %s' % (url, detail),
                    cmd=proc.cmd if proc else None, output=tail,
                    hint='Look at the %s output above / in %s. If it is still '
                         'compiling, re-run; if it crashed, fix the error it '
                         'printed.' % (label, rel(API_LOG if label == 'api' else WEB_LOG)),
                )
            info('%s ready (%s)' % (label, detail))

        step.done('all servers answering')
        self.print_ready_banner()

    def print_ready_banner(self):
        creds = self.credentials()
        width = 74
        line = style('  ' + '=' * width, 'green')
        emit('')
        emit(line)
        emit('  ' + style('AutoSmokeGuard is running', 'bold', 'green'))
        emit(line)
        if self.wants_frontend:
            emit('  %-16s %s' % ('Web UI', style(self.web_url, 'bold', 'magenta')))
        if self.wants_backend:
            emit('  %-16s %s' % ('API', style(self.api_url + '/api/', 'bold', 'cyan')))
            emit('  %-16s %s' % ('API docs', self.api_url + '/api/docs/'))
            emit('  %-16s %s' % ('Django admin', self.api_url + '/admin/'))
            emit('  %-16s %s' % ('Health', self.api_url + HEALTH_PATH))
        emit('')
        emit('  ' + style('Sign in with', 'bold'))
        emit('  %-16s %s / %s' % ('demo user',
                                  creds['demo']['email'], creds['demo']['password']))
        emit('  %-16s %s / %s' % ('admin user',
                                  creds['admin']['email'], creds['admin']['password']))
        emit('')
        emit('  %-16s %s' % ('Sample media', self._sample_media_summary()))
        emit('  %-16s %s , %s' % ('Logs', rel(API_LOG), rel(WEB_LOG)))
        if self.warnings:
            emit('')
            emit('  ' + style('Warnings raised during startup:', 'yellow'))
            for item in self.warnings:
                emit(style('    - ' + item, 'yellow'))
        emit(line)
        emit('  ' + style('Press Ctrl-C to stop both servers.', 'grey'))
        emit('')

    def _sample_media_summary(self):
        if not SAMPLE_MEDIA_DIR.is_dir():
            return rel(SAMPLE_MEDIA_DIR) + '  (not generated)'
        count = sum(1 for item in SAMPLE_MEDIA_DIR.rglob('*') if item.is_file())
        if not count:
            return rel(SAMPLE_MEDIA_DIR) + '  (empty)'
        return '%s  (%d file%s)' % (rel(SAMPLE_MEDIA_DIR), count,
                                    '' if count == 1 else 's')

    # ------------------------------------------------------------------
    # Phase 9 — browser
    # ------------------------------------------------------------------
    def phase_open_browser(self, step):
        if self.args.no_browser:
            step.skip('--no-browser')
            return
        if not self.wants_frontend:
            step.skip('no web UI in this mode')
            return
        target = self.web_url
        try:
            opened = webbrowser.open(target)
        except (webbrowser.Error, OSError) as exc:
            self.warn('could not open a browser: %s' % exc)
            opened = False
        if opened:
            step.done('opened %s' % target)
        else:
            self.warn('no browser could be opened automatically')
            step.done('open %s yourself' % target)

    # ------------------------------------------------------------------
    # Phase 10 — supervise
    # ------------------------------------------------------------------
    def _install_signal_handlers(self):
        def _handler(signum, _frame):
            if not self.stop_requested.is_set():
                emit('')
                emit(style('  received %s — shutting down...'
                           % signal.Signals(signum).name, 'bold', 'yellow'))
            self.stop_requested.set()

        for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError, RuntimeError):
                pass  # not the main thread, or unsupported on this platform

    def phase_supervise(self, step):
        note('streaming server output — Ctrl-C to stop')
        crashed = None
        while not self.stop_requested.is_set():
            for proc in self.processes:
                if not proc.is_running() and not proc.stopped_by_us:
                    crashed = proc
                    break
            if crashed is not None:
                break
            time.sleep(0.4)

        if crashed is not None:
            emit('')
            emit(style('  the %s server exited unexpectedly (code %s)'
                       % (crashed.label, crashed.returncode()), 'bold', 'red'))
            emit(style('  last %d lines of %s:'
                       % (min(30, len(crashed.ring)), rel(crashed.log_path)), 'grey'))
            for line in crashed.tail(30):
                emit(style('    | ', 'grey') + line)
            emit('')
            self.failures.append('%s server crashed' % crashed.label)

        self.shutdown()
        if crashed is not None:
            step.problem('stopped after the %s server died' % crashed.label)
        else:
            step.done('both servers stopped cleanly')

    def shutdown(self):
        """Stop every child. Idempotent, safe from any exit path."""
        if self.shutdown_done.is_set():
            return
        self.shutdown_done.set()
        alive = [p for p in self.processes if p.is_running()]
        if not alive:
            return
        emit('')
        for proc in alive:
            note('stopping %s (pid %s)...' % (proc.label, proc.pid))
            proc.stop()
        for proc in alive:
            note('%s stopped (code %s)' % (proc.label, proc.returncode()))

    # ------------------------------------------------------------------
    # --simulate-os: describe what this launcher WOULD do on another OS
    # ------------------------------------------------------------------
    def _sim(self, parts):
        rendered = []
        for part in parts:
            rendered.append(self.profile.display(part) if isinstance(part, Path)
                            else str(part))
        return format_cmd(rendered)

    def simulate(self):
        prof = self.profile
        vpy = prof.venv_python(VENV_DIR)
        vpip = prof.venv_pip(VENV_DIR)
        def head(text):
            emit('')
            emit(style('  ' + text, 'bold', 'blue'))

        emit('')
        emit(style('  AutoSmokeGuard launcher — dry run for %s' % prof.label,
                   'bold', 'cyan'))
        note('  host OS is %s; no command below is executed' % platform.system())
        if prof.is_windows:
            note('  paths are shown with a synthetic root (%s) so the Windows'
                 % prof.display_root)
            note('  separators and the Scripts/ layout are visible')

        head('interpreter resolution')
        info('%-22s %s' % ('venv python:', prof.display(vpy)))
        info('%-22s %s' % ('venv pip:', prof.display(vpip)))
        info('%-22s %s' % ('search order:', ', '.join(PYTHON_CANDIDATES)))
        if prof.is_windows:
            info('%-22s %s' % ('py launcher:',
                               ', '.join('py ' + a for a in PY_LAUNCHER_ARGS)))
        else:
            info('%-22s %s' % ('py launcher:', 'not used off Windows'))
        info('%-22s %s' % ('version probe:', 'python -c "%s"' % VERSION_PROBE))
        info('%-22s >= %d.%d' % ('minimum accepted:', MIN_PYTHON[0], MIN_PYTHON[1]))

        head('binaries')
        info('%-22s %s' % ('npm lookup:',
                           "shutil.which('npm.cmd') or shutil.which('npm')"
                           if prof.is_windows else "shutil.which('npm')"))
        info('%-22s %s' % ('npm resolves to:', prof.npm_display()))
        info('%-22s %s' % ('node lookup:', "shutil.which('node')"))

        head('process control')
        info('%-22s %s' % ('spawn:', prof.popen_kwargs_display()))
        info('%-22s %s' % ('stop:', prof.terminate_display()))
        info('%-22s %s' % ('batch shim:',
                           'npm.cmd run via CreateProcess, %COMSPEC% /d /s /c '
                           'fallback on WinError 193'
                           if prof.is_windows else 'not applicable'))
        info('%-22s %s' % ('colour:',
                           "os.system('') to enable VT, ANSI after that"
                           if prof.is_windows else 'ANSI directly'))

        head('stamps, logs and assets')
        info('%-22s %s' % ('backend stamp:', prof.display(BACKEND_STAMP)))
        info('%-22s %s' % ('frontend stamp:', prof.display(FRONTEND_STAMP)))
        info('%-22s %s' % ('api log:', prof.display(API_LOG)))
        info('%-22s %s' % ('web log:', prof.display(WEB_LOG)))
        info('%-22s %s' % ('yolo weights:', prof.display(YOLO_WEIGHTS)))
        info('%-22s %s' % ('smoke model:', prof.display(UNET_WEIGHTS)))

        head('ports (probed on this host)')
        api_free = port_is_free(self.api_port)
        web_free = port_is_free(self.web_port)
        api_port = self.api_port if api_free else free_port(self.api_port)
        web_port = self.web_port if web_free else free_port(self.web_port)
        info(('%-22s %d %s' % ('api:', api_port,
              '' if api_free else '(%d was busy)' % self.api_port)).rstrip())
        info(('%-22s %d %s' % ('web:', web_port,
              '' if web_free else '(%d was busy)' % self.web_port)).rstrip())

        head('commands per phase')
        host = '<selected-python>'
        rows = [
            ('2 venv', self._sim([host, '-m', 'venv', VENV_DIR])),
            ('2 pip', self._sim([vpy, '-m', 'pip', 'install', '--upgrade', 'pip'])),
            ('2 deps', self._sim([vpy, '-m', 'pip', 'install', '-r', REQUIREMENTS])),
            ('3 npm', self._sim([prof.npm_display(), 'ci'])
                + '   (cwd: %s)' % prof.display(FRONTEND_DIR)),
            ('4 yolo', 'GET ' + YOLO_URL),
            ('4 synth', self._sim([vpy, '-m', SYNTH_MODULE])),
            ('4 train', self._sim([vpy, '-m', TRAIN_MODULE])),
            ('5 migrate', self._sim([vpy, MANAGE_PY.name, 'migrate', '--noinput'])),
            ('5 seed', self._sim([vpy, SEED_SCRIPT])),
            ('6 samples', self._sim([vpy, SAMPLES_SCRIPT])),
            ('7 api', self._sim([vpy, MANAGE_PY.name, 'runserver',
                                 '127.0.0.1:%d' % api_port])),
            ('7 web', self._sim([prof.npm_display(), 'run', 'dev', '--',
                                 '--port', str(web_port), '--strictPort',
                                 '--host', '127.0.0.1'])),
            ('7 web env', 'VITE_API_PROXY=http://127.0.0.1:%d' % api_port),
            ('8 ready', 'GET http://127.0.0.1:%d%s  and  GET http://127.0.0.1:%d/'
                        % (api_port, HEALTH_PATH, web_port)),
            ('9 browser', "webbrowser.open('http://127.0.0.1:%d')" % web_port),
        ]
        for label, text in rows:
            info('%-12s %s' % (label, text))
        emit('')
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
EPILOG = """
examples
  python start.py                      full setup, then run both servers
  python start.py --no-browser         same, without opening a browser
  python start.py --check --skip-ml    CI-style: set up + run the tests, exit
  python start.py --backend-only -v    just the Django API, verbose
  python start.py --reset-db --yes     wipe the SQLite DB, re-migrate, re-seed
  python start.py --api-port 8100      pin the API port (busy ports auto-scan)

double-click entry points
  macOS / Linux   START.command
  Windows         START.bat   (or START.ps1 from PowerShell)
"""


def build_parser():
    parser = argparse.ArgumentParser(
        prog='start.py',
        description='AutoSmokeGuard one-command launcher: sets up the backend '
                    'venv, the frontend packages, the ML assets and the '
                    'database, then runs the Django API and the Vite UI '
                    'together with merged, prefixed logs.',
        epilog=textwrap.dedent(EPILOG),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--no-browser', action='store_true',
                        help='do not open the web UI in a browser')
    parser.add_argument('--reinstall', action='store_true',
                        help='force pip install and npm install even when the '
                             'dependency stamps are current')
    parser.add_argument('--backend-only', action='store_true',
                        help='set up and run only the Django API')
    parser.add_argument('--frontend-only', action='store_true',
                        help='set up and run only the Vite dev server')
    parser.add_argument('--skip-ml', action='store_true',
                        help='skip the ML asset phase (no weight download, no '
                             'first-run smoke-model training)')
    parser.add_argument('--api-port', type=int, metavar='N',
                        default=DEFAULT_API_PORT,
                        help='preferred Django port (default %d; the next free '
                             'port is used when it is taken)' % DEFAULT_API_PORT)
    parser.add_argument('--web-port', type=int, metavar='N',
                        default=DEFAULT_WEB_PORT,
                        help='preferred Vite port (default %d; the next free '
                             'port is used when it is taken)' % DEFAULT_WEB_PORT)
    parser.add_argument('--check', action='store_true',
                        help='run setup plus the test suites and exit without '
                             'starting servers (non-zero exit on any failure)')
    parser.add_argument('--reset-db', action='store_true',
                        help='delete backend/db.sqlite3, then migrate and seed '
                             'again (asks for confirmation unless --yes)')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='answer yes to confirmation prompts')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='echo every command and stream its output')
    parser.add_argument('--no-color', action='store_true',
                        help='plain text output (also honours NO_COLOR)')
    # Hidden: prints what the launcher WOULD do on another OS, so the Windows
    # code paths can be reviewed from a Mac.
    parser.add_argument('--simulate-os', choices=('windows', 'darwin', 'linux'),
                        metavar='OS', help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    try:
        return _main(argv)
    except BrokenPipeError:
        silence_stdout()
        return 141          # 128 + SIGPIPE, what a shell would report


def _main(argv=None):
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.backend_only and args.frontend_only:
        parser.error('--backend-only and --frontend-only are mutually exclusive')
    for name in ('api_port', 'web_port'):
        value = getattr(args, name)
        if value is not None and not (1 <= int(value) <= 65535):
            parser.error('--%s must be between 1 and 65535' % name.replace('_', '-'))

    configure_console(args.no_color)
    launcher = Launcher(args)

    if args.simulate_os:
        return launcher.simulate()

    code = 1
    try:
        code = launcher.execute()
    except LauncherError as exc:
        emit(exc.render())
        code = 1
    except KeyboardInterrupt:
        emit('')
        emit(style('  interrupted — shutting down', 'yellow'))
        code = 130
    except Exception:                       # unexpected: show it, stay tidy
        emit('')
        emit(style('  INTERNAL ERROR in the launcher itself', 'bold', 'red'))
        for line in traceback.format_exc().splitlines():
            emit(style('    ' + line, 'grey'))
        emit(style('  Please report this with the traceback above. You can '
                   'still run the servers manually:', 'yellow'))
        emit(style('    cd backend && .venv/bin/python manage.py runserver', 'yellow'))
        emit(style('    cd frontend && npm run dev', 'yellow'))
        code = 1
    finally:
        try:
            launcher.shutdown()
        except Exception as exc:                      # never mask the real error
            emit(style('  shutdown problem: %s' % exc, 'red'))

    elapsed = time.monotonic() - launcher.started_at
    emit('')
    if code == 0:
        emit('  %s %s %s' % (style(GLYPH_OK, 'green', 'bold'),
                             style('finished', 'bold'),
                             style('in %.1fs' % elapsed, 'grey')))
    else:
        emit('  %s %s %s' % (style(GLYPH_BAD, 'red', 'bold'),
                             style('exited with code %d' % code, 'bold', 'red'),
                             style('after %.1fs' % elapsed, 'grey')))
        for item in launcher.failures:
            emit(style('    - ' + item, 'red'))
    if launcher.warnings and code != 0:
        for item in launcher.warnings:
            emit(style('    ! ' + item, 'yellow'))
    emit('')
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
