# AutoSmokeGuard — developer scripts & the one-command launcher

Everything in this project starts from **one command**. You do not need to
create a virtualenv, install packages, download model weights, run migrations,
or open two terminals — the launcher does all of it and then runs the Django
API and the Vite UI together.

---

## TL;DR

| Your machine | Do this |
|---|---|
| macOS / Linux | double-click **`START.command`**, or `python3 start.py` |
| Windows (Explorer) | double-click **`START.bat`** |
| Windows (PowerShell) | `.\START.ps1`  (see *Execution policy* below) |
| Any, from a terminal | `python start.py` |

First run takes a few minutes (virtualenv + npm install + model download and,
once, training the smoke model). Every run after that starts in seconds
because each expensive step is stamped and skipped when nothing changed.

Prerequisites: **Python >= 3.10** and **Node.js >= 18** on `PATH`. Nothing else.

---

## What the launcher actually does

Ten numbered phases, each printed with a tick/cross and how long it took:

| # | Phase | What happens | Skipped when |
|---|-------|--------------|--------------|
| 1 | Preflight | OS, arch, Python, Node, npm, free disk. Hard-fails with install instructions if Node < 18 or npm is missing | never |
| 2 | Backend venv | creates `backend/.venv`, upgrades pip, `pip install -r requirements.txt` | the sha256 of `requirements.txt` matches `.venv/.asg_deps_stamp` |
| 3 | Frontend deps | `npm ci` (or `npm install` with no lockfile) in `frontend/` | the sha256 of `package-lock.json` matches `node_modules/.asg_deps_stamp` |
| 4 | ML assets | downloads `yolo11n.pt`; on the very first run generates the synthetic dataset and trains `smoke_unet.pt` | both files already exist, or `--skip-ml` |
| 5 | Database | `manage.py migrate --noinput`, then `tools/seed.py` | never (both are idempotent) |
| 6 | Sample media | `tools/make_samples.py` | `backend/sample_media/` already has files |
| 7 | Start servers | Django + Vite as supervised children; output is merged, prefixed `[api]`/`[web]`, colour-coded, and tee'd to `backend/logs/` | `--check` |
| 8 | Readiness | polls `/api/health/` and the Vite root (60 s, every 0.5 s), then prints the ready banner with URLs and logins | `--check` |
| 9 | Browser | opens the UI | `--no-browser` |
| 10 | Supervise | streams logs until Ctrl-C; if one server dies it prints that server's last 30 log lines and stops the other | `--check` |

With `--check`, phases 7-10 are replaced by the **test suites** (`pytest` in
`backend/`, `npm run test` in `frontend/`) and the process exits — non-zero if
anything failed, so CI can call it directly.

---

## Flags

```
--no-browser      do not open a browser
--reinstall       force pip install and npm install, ignoring the stamps
--backend-only    set up and run only the Django API
--frontend-only   set up and run only the Vite dev server
--skip-ml         skip phase 4 entirely (no download, no training)
--api-port N      preferred Django port (default 8000)
--web-port N      preferred Vite port (default 5173)
--check           phases 1-6 + the test suites, then exit (CI mode)
--reset-db        delete backend/db.sqlite3, migrate and seed again
-y, --yes         answer yes to confirmation prompts (needed by --reset-db in CI)
-v, --verbose     echo every command and stream its output live
--no-color        plain output (NO_COLOR is honoured automatically)
```

Examples:

```bash
python start.py                          # the normal thing
python start.py --no-browser --skip-ml   # fast loop, no model work
python start.py --check --skip-ml        # what CI runs
python start.py --reset-db --yes         # start from a clean database
python start.py --backend-only -v        # debug the API alone
python start.py --api-port 8100 --web-port 5200
```

### Ports

8000 and 5173 are only *preferences*. If either is busy the launcher scans
upward for the next free port, tells you which one it picked, and keeps
everything consistent: Django binds the new API port and Vite is started with
`--port <webport>` **and** `VITE_API_PROXY=http://127.0.0.1:<apiport>` in its
environment, so the dev proxy still reaches the API.

### Logins printed in the ready banner

| Role | Email | Password |
|---|---|---|
| Demo user | `demo@autosmokeguard.local` | `Demo@12345` |
| Admin | `admin@autosmokeguard.local` | `Admin@12345` |

These mirror `backend/tools/seed.py`. If the seed script ever changes them, it
can drop a `backend/tools/seed_credentials.json` like

```json
{"demo": {"email": "...", "password": "..."},
 "admin": {"email": "...", "password": "..."}}
```

and the banner will use that instead — neither file needs to import the other.

---

## Where things live

```
FYP/
  start.py                      thin shim: puts backend/tools on sys.path, calls dev_runner.main()
  START.command                 macOS / Linux double-click wrapper (chmod +x)
  START.bat                     Windows Explorer double-click wrapper
  START.ps1                     Windows PowerShell wrapper
  backend/
    tools/dev_runner.py         the launcher itself (stdlib only, ~2000 lines)
    .venv/.asg_deps_stamp       sha256 of requirements.txt at last successful install
    logs/launcher-api.log       everything the Django child printed this run
    logs/launcher-web.log       everything the Vite child printed this run
  frontend/
    node_modules/.asg_deps_stamp  sha256 of package-lock.json at last successful install
```

The real code lives in the backend repository so it is version-controlled with
the rest of the server; `start.py` at the repo root is only a convenience entry
point (and something the double-click wrappers can call).

Both log files are truncated at the start of every run, and both are written
with ANSI colour codes stripped, so they are safe to paste into a bug report.

---

## Cross-platform behaviour

The launcher detects the OS with `platform.system()` and adapts:

| Concern | Windows | macOS / Linux |
|---|---|---|
| venv interpreter | `backend\.venv\Scripts\python.exe` | `backend/.venv/bin/python` |
| venv pip | `Scripts\pip.exe` | `bin/pip` |
| npm | `shutil.which('npm.cmd') or shutil.which('npm')` | `shutil.which('npm')` |
| spawning | `creationflags=CREATE_NEW_PROCESS_GROUP` | `start_new_session=True` |
| stopping | `CTRL_BREAK_EVENT`, then `taskkill /F /T /PID` | `SIGTERM` to the process group, then `SIGKILL` |
| interpreter search | version-tagged names, then the `py` launcher (`py -3.12`, `py -3`) | version-tagged names |
| colour | VT enabled via `os.system('')`, ASCII glyph fallback on legacy code pages | ANSI |

Children are always put in their own process group. That is what makes Ctrl-C
reliable: the launcher receives the interrupt, then signals each group, so
Django's autoreloader worker and Vite's esbuild helpers go down too instead of
surviving as orphans holding the ports.

To inspect the decisions for another OS without being on it:

```bash
python start.py --simulate-os windows     # also: darwin, linux
```

It prints every path, binary, spawn flag, kill strategy and per-phase command
it *would* use, and runs nothing.

---

## Troubleshooting

**"no Python >= 3.10 found"** — install from <https://www.python.org/downloads/>
(on Windows tick *Add python.exe to PATH*), open a **new** terminal, retry.

**"Node.js is not installed"** — install the LTS from
<https://nodejs.org/en/download>, open a new terminal, retry.

**PowerShell: "running scripts is disabled on this system"** — either

```powershell
powershell -ExecutionPolicy Bypass -File .\START.ps1
```

or, for the current window only (no admin rights, reverts when you close it):

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

**Dependencies look wrong / a package is missing** — the stamp thinks the
install is current. Force it:

```bash
python start.py --reinstall
```

**Migrations conflict, or the DB is full of junk** — `python start.py --reset-db`
(add `--yes` to skip the prompt). This deletes `backend/db.sqlite3` only;
uploaded files under `media/` are left alone.

**A server died on startup** — the launcher already printed that child's last
30 lines; the full output is in `backend/logs/launcher-api.log` or
`backend/logs/launcher-web.log`.

**Training takes too long / you are offline** — `--skip-ml` starts everything
except the smoke segmentation model.

**Something is still holding port 8000/5173** — you do not need to do anything:
the launcher scans for the next free port and prints what it chose. To pin a
port anyway, use `--api-port` / `--web-port`.
