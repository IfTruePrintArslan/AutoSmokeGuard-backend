"""
Process-lifecycle tests for the developer launcher (``tools/dev_runner.py``).

These cover the part of the launcher that CI has never been able to execute.
The ``launcher-smoke-test`` job in ``.github/workflows/ci.yml`` runs the real
launcher on a real Windows runner, but every invocation passes ``--check``,
and ``build_plan()`` stops right after the test-suite phase when that flag is
set — so ``phase_start_servers`` never runs and neither does anything it
reaches:

  * :meth:`OSProfile.popen_kwargs`  — ``CREATE_NEW_PROCESS_GROUP`` (Windows)
    vs ``start_new_session=True`` (POSIX)
  * :meth:`OSProfile.signal_child`  — ``CTRL_BREAK_EVENT`` then
    ``taskkill /F /T`` (Windows) vs ``killpg(SIGTERM)`` then
    ``killpg(SIGKILL)`` (POSIX)
  * :func:`popen_with_fallback`     — the WinError 193 ``%COMSPEC% /d /s /c``
    batch-shim fallback
  * the live port probe reached from ``Launcher._resolve_ports``

That is the code standing between a user pressing Ctrl-C and two orphaned
servers still holding ports 8000 and 5173, so it is tested here against real
child processes rather than reasoned about.

How these tests are written
---------------------------
Everything here calls the shipped code — ``ManagedProcess``, ``OSProfile``,
``popen_with_fallback``, ``port_is_free``, ``free_port``,
``Launcher._resolve_ports``.  Nothing is reimplemented, so a change in
``dev_runner.py`` is what these tests see.

Tests 1-4 and 6 run for real on **both** POSIX and Windows and assert on the
*outcome* (the process is gone, the group is separate, no grandchild
survives); the only ``sys.platform`` branches are where the mechanism
genuinely differs and the mechanism itself is worth pinning.  Test 5 is
Windows-only logic, which is exactly why it is driven entirely through
monkeypatched ``subprocess.Popen`` / ``os.name``: it makes an otherwise
unreachable branch executable on every developer's machine instead of being
skipped into meaninglessness.

Nothing here may leak a child.  Every spawn goes through the ``managed``
fixture, whose teardown force-kills anything still alive with a mechanism
independent of the code under test, so even a failing assertion cannot leave
a ``sleep 300`` behind.
"""
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time

import pytest

from tools import dev_runner
from tools.dev_runner import (
    ManagedProcess,
    OSProfile,
    format_cmd,
    free_port,
    popen_with_fallback,
    port_is_free,
)

IS_WINDOWS = os.name == 'nt'

#: ``subprocess.CREATE_NEW_PROCESS_GROUP``.  Spelled out because the constant
#: does not exist on POSIX, and the whole point of the simulated-profile tests
#: below is to check the Windows value from a Mac.
CREATE_NEW_PROCESS_GROUP = 0x00000200

#: Budget for a cooperative child to die after the *soft* stop signal.  A
#: bare ``time.sleep`` child needs milliseconds; 3s is slack for a loaded CI
#: runner, and staying under it is what proves Ctrl-C alone does the job.
GRACE = 3.0

#: Grace used for the child that ignores the soft signal.  The production
#: default is 10s; ``ManagedProcess.stop`` takes it as a parameter, so the
#: escalation can be exercised honestly without a 10s sleep in the suite.
SHORT_GRACE = 0.75

#: How long to wait for a child to announce itself, and for a killed process
#: to disappear from the OS table.  Only ever reached on failure.
READY_TIMEOUT = 10.0
REAP_TIMEOUT = 15.0

#: Present in every child's source so a stray survivor is greppable.
MARKER = 'ASG-LAUNCHER-TEST-CHILD'


# ---------------------------------------------------------------------------
# Child programs
# ---------------------------------------------------------------------------

#: A plain long-lived child that handles signals the default way.
SLEEPER = textwrap.dedent('''
    # {marker}
    import sys, time
    print('ready', flush=True)
    time.sleep(300)
''').format(marker=MARKER)

#: A child that refuses the polite request.  SIGTERM (POSIX) and SIGBREAK /
#: SIGINT (Windows, which is what ``CTRL_BREAK_EVENT`` is delivered as) are
#: all set to SIG_IGN, so only the hard path — SIGKILL or ``taskkill /F /T``,
#: neither of which can be blocked — can reap it.
STUBBORN = textwrap.dedent('''
    # {marker}
    import signal, sys, time
    for name in ('SIGTERM', 'SIGINT', 'SIGBREAK'):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            signal.signal(number, signal.SIG_IGN)
        except (OSError, ValueError, RuntimeError):
            pass
    print('ready', flush=True)
    while True:
        time.sleep(0.05)
''').format(marker=MARKER)

#: A child that spawns a grandchild of its own — the shape of the real thing
#: (Django's autoreloader forks a worker, Vite spawns esbuild).  The
#: grandchild gets DEVNULL rather than inheriting our pipe, so the parent's
#: output stream still reaches EOF the moment the parent dies.
GRANDPARENT = textwrap.dedent('''
    # {marker}
    import subprocess, sys, time
    grandchild = subprocess.Popen(
        [sys.executable, '-c', 'import time  # {marker}\\ntime.sleep(300)'],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print('grandchild', grandchild.pid, flush=True)
    time.sleep(300)
''').format(marker=MARKER)


# ---------------------------------------------------------------------------
# Helpers.  Deliberately NOT built on dev_runner: a cleanup path that shares
# code with the thing under test cannot be trusted to clean up after it.
# ---------------------------------------------------------------------------

def pid_is_alive(pid):
    """True while *pid* still exists, asked of the OS directly.

    ``os.kill(pid, 0)`` is the POSIX idiom, but on Windows ``os.kill`` is
    implemented with ``TerminateProcess`` for every signal that is not a
    console control event — calling it with 0 there would *kill* the process
    being asked about and make the assertion pass for the wrong reason.
    Windows therefore goes through ``tasklist`` with a PID filter; CSV output
    keeps the parse unambiguous when an image name contains a space.
    """
    if IS_WINDOWS:
        try:
            result = subprocess.run(
                ['tasklist', '/FI', 'PID eq %d' % pid, '/NH', '/FO', 'CSV'],
                capture_output=True, text=True, errors='replace', timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return ('"%d"' % pid) in (result.stdout or '')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:      # it exists, it just is not ours any more
        return True
    return True


def wait_until_dead(pid, timeout=REAP_TIMEOUT):
    """Poll until *pid* is gone; returns False if it outlives *timeout*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_is_alive(pid):
            return True
        time.sleep(0.05)
    return not pid_is_alive(pid)


def hard_kill(pid):
    """Unconditional cleanup for a bare pid. Never raises."""
    try:
        if IS_WINDOWS:
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                           capture_output=True, timeout=30)
        else:
            os.kill(pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass


def child_has_its_own_group(child):
    """POSIX: True when signalling *child*'s group cannot reach us.

    Always True on Windows, where the equivalent isolation is a creation flag
    rather than something the OS will answer a question about; the Windows leg
    of :meth:`TestProcessGroupIsolation` checks that flag directly instead.
    """
    if IS_WINDOWS:
        return True
    try:
        return os.getpgid(child.pid) != os.getpgid(0)
    except (ProcessLookupError, OSError):
        return True          # already gone: nothing to signal, nothing to hit


def assert_safe_to_signal(child):
    """Refuse to run a group-wide kill that would also hit this test runner.

    Not paranoia: with ``popen_kwargs()`` returning ``{}`` the launcher's own
    ``stop()`` sends SIGTERM to its own process group, which kills the
    launcher — and, here, pytest.  Without this guard a regression in the
    spawn flags would show up as the whole suite vanishing mid-run instead of
    as one test failing with a readable message.
    """
    assert child_has_its_own_group(child), (
        'the child shares this process group, so the launcher\'s killpg would '
        'terminate the test runner: refusing to signal. The spawn flags from '
        'OSProfile.popen_kwargs() are not reaching Popen.'
    )


def force_reap(managed_proc):
    """Guaranteed teardown for a :class:`ManagedProcess`. Never raises."""
    child = managed_proc.proc
    if child is not None and child.poll() is None:
        try:
            if IS_WINDOWS:
                subprocess.run(
                    ['taskkill', '/F', '/T', '/PID', str(child.pid)],
                    capture_output=True, timeout=30)
            elif child_has_its_own_group(child):
                # The group, not the pid: the group is what could be left
                # behind.  Guarded, for the reason in assert_safe_to_signal.
                try:
                    os.killpg(os.getpgid(child.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    child.kill()
            else:
                child.kill()
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            child.wait(timeout=REAP_TIMEOUT)
        except (subprocess.SubprocessError, OSError):
            pass
    # Join the output pump too, so no thread is still writing to pytest's
    # captured stdout after the test has finished with it.
    if managed_proc.thread is not None:
        managed_proc.thread.join(timeout=5)


def wait_for_line(managed_proc, needle, timeout=READY_TIMEOUT):
    """Return the first streamed line containing *needle*.

    Reads ``ManagedProcess.ring``, which is filled by the launcher's own
    ``stream_process`` pump — so this doubles as a check that a supervised
    child's output actually reaches the ring buffer used to explain crashes.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for line in list(managed_proc.ring):
            if needle in line:
                return line
        time.sleep(0.02)
    raise AssertionError(
        '%r never printed %r within %.1fs (exit code: %r, output: %r)'
        % (managed_proc.label, needle, timeout,
           managed_proc.returncode(), list(managed_proc.ring))
    )


def recording_profile(profile=None):
    """A real :class:`OSProfile` that logs every ``signal_child`` escalation.

    The instance attribute shadows the bound method but delegates to it, so
    the production kill logic still runs unmodified; the returned list records
    the ``hard`` flag of each call, which is how the tests below prove the
    escalation happened rather than assuming it did.
    """
    profile = profile or OSProfile(None)
    calls = []
    genuine = profile.signal_child

    def recorder(proc, hard=False):
        calls.append(hard)
        return genuine(proc, hard=hard)

    profile.signal_child = recorder
    return profile, calls


def occupied_port(host='127.0.0.1'):
    """A bound, listening socket and the port it holds.

    An ephemeral port is used rather than a guessed one so the port really is
    occupied.  Ports near the top of the range are re-rolled: ``free_port``
    scans upwards from its preferred port and gives up past 65535, which would
    make an otherwise deterministic test fail a few times in a thousand.
    """
    for _ in range(20):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        if port <= 65000:
            return sock, port
        sock.close()
    raise AssertionError('could not obtain an ephemeral port below 65000')


def free_port_number(host='127.0.0.1'):
    """A port that was just proven bindable and then released."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def free_adjacent_ports(host='127.0.0.1'):
    """A port whose immediate neighbour is free too.

    Re-rolled rather than skipped: a skipped test verifies nothing, and the
    ephemeral range is large enough that a free pair is found on the first or
    second try in practice.
    """
    for _ in range(50):
        port = free_port_number(host)
        if port <= 65000 and port_is_free(port) and port_is_free(port + 1):
            return port
    raise AssertionError('could not find two adjacent free ports')


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def managed(tmp_path):
    """Factory for :class:`ManagedProcess` children with guaranteed cleanup."""
    created = []

    def _make(source, label='child', profile=None):
        proc = ManagedProcess(
            label, 'cyan',
            [sys.executable, '-c', source],
            tmp_path,
            os.environ.copy(),
            tmp_path / ('%s.log' % label),
            profile if profile is not None else OSProfile(None),
        )
        created.append(proc)
        return proc

    try:
        yield _make
    finally:
        for proc in created:
            force_reap(proc)


# ===========================================================================
# 1. Spawn and terminate a real child through the real code path
# ===========================================================================

class TestSpawnAndTerminate:
    """The Ctrl-C contract: a polite stop must actually stop the server."""

    def test_child_is_spawned_and_stopped_by_the_soft_signal(self, managed):
        """Start a real child via ManagedProcess, stop it via ManagedProcess.

        This is the whole lifecycle through the shipped code: spawn with
        ``popen_with_fallback`` + ``profile.popen_kwargs()``, stream its
        output, then ``stop()`` -> ``signal_child(hard=False)``.  On POSIX
        that is ``killpg(SIGTERM)``; on Windows it is ``CTRL_BREAK_EVENT``.
        Finishing inside *grace* is the assertion that matters: it means the
        polite signal alone was enough and the forced ``SIGKILL`` /
        ``taskkill /F /T`` fallback never had to run.
        """
        profile, escalations = recording_profile()
        proc = managed(SLEEPER, label='api', profile=profile)

        child = proc.start()
        wait_for_line(proc, 'ready')
        assert proc.is_running()
        assert proc.returncode() is None
        assert child.pid > 0
        assert_safe_to_signal(child)

        started = time.monotonic()
        proc.stop(grace=GRACE)
        elapsed = time.monotonic() - started

        assert child.poll() is not None, (
            'the child survived ManagedProcess.stop(): on this platform the '
            'launcher cannot stop a dev server at all'
        )
        assert proc.returncode() is not None
        assert proc.stopped_by_us is True
        assert escalations == [False], (
            'the soft stop should have been enough for a child that does not '
            'trap signals, but the launcher had to escalate: %r' % escalations
        )
        assert elapsed < GRACE, (
            'the child needed %.2fs to die but the soft signal budget is '
            '%.2fs — Ctrl-C does not reach the child on this platform'
            % (elapsed, GRACE)
        )

        if not IS_WINDOWS:
            # Mechanism check, POSIX only: negative returncode == killed by
            # that signal, which proves the SIGTERM came from killpg and not
            # from some internal proc.terminate() fallback.
            assert child.returncode == -signal.SIGTERM

    def test_stop_is_idempotent_and_safe_on_a_dead_child(self, managed):
        """``stop()`` is called from both the signal handler and shutdown()."""
        proc = managed(SLEEPER, label='api')
        child = proc.start()
        wait_for_line(proc, 'ready')
        assert_safe_to_signal(child)

        proc.stop(grace=GRACE)
        first = proc.returncode()
        assert first is not None

        proc.stop(grace=GRACE)                 # must be a no-op, not a crash
        assert proc.returncode() == first
        assert proc.is_running() is False


# ===========================================================================
# 2. The child must be in its own process group
# ===========================================================================

class TestProcessGroupIsolation:
    """Signalling the group must not take the test runner down with it."""

    def test_spawn_kwargs_reach_popen_and_isolate_the_child(self, managed,
                                                            monkeypatch):
        """The kwargs ManagedProcess actually passes to Popen isolate the child.

        Asserting on ``popen_kwargs()`` alone would not prove ``start()``
        applies it, so ``subprocess.Popen`` is wrapped (and still called for
        real) to capture the kwargs that were genuinely used.

        On POSIX the guarantee is then confirmed against the OS itself: the
        child is a process-group leader of a group that is not ours, which is
        precisely what makes ``killpg`` safe.  On Windows the equivalent
        guarantee is CREATE_NEW_PROCESS_GROUP — without it,
        ``GenerateConsoleCtrlEvent`` would hit this test runner as well.
        """
        seen = []
        genuine_popen = subprocess.Popen

        def spy(*args, **kwargs):
            seen.append(dict(kwargs))
            return genuine_popen(*args, **kwargs)

        monkeypatch.setattr(dev_runner.subprocess, 'Popen', spy)

        proc = managed(SLEEPER, label='web')
        child = proc.start()
        wait_for_line(proc, 'ready')

        assert seen, 'ManagedProcess.start() did not reach subprocess.Popen'
        used = seen[0]

        if IS_WINDOWS:
            flags = used.get('creationflags', 0)
            assert flags & CREATE_NEW_PROCESS_GROUP, (
                'the child was spawned without CREATE_NEW_PROCESS_GROUP '
                '(creationflags=%#x): CTRL_BREAK_EVENT would be delivered to '
                'the launcher itself instead of the server' % flags
            )
            assert 'start_new_session' not in used
        else:
            assert used.get('start_new_session') is True
            assert 'creationflags' not in used
            assert os.getpgid(child.pid) != os.getpgid(0), (
                'the child shares our process group: killpg would kill the '
                'launcher (and this test runner) too'
            )
            assert os.getpgid(child.pid) == child.pid, (
                'the child is not the leader of its own group, so killpg on '
                'its pid would not reach the tree'
            )

        proc.stop(grace=GRACE)
        assert proc.returncode() is not None

    def test_windows_profile_asks_for_a_new_process_group(self):
        """Runs everywhere: the Windows spawn flags, checked from any host."""
        kwargs = OSProfile('windows', simulated=True).popen_kwargs()
        assert kwargs == {'creationflags': CREATE_NEW_PROCESS_GROUP}
        assert 'start_new_session' not in kwargs

    @pytest.mark.parametrize('system', ['darwin', 'linux'])
    def test_posix_profile_asks_for_a_new_session(self, system):
        """Runs everywhere: the POSIX spawn flags, checked from any host."""
        assert OSProfile(system, simulated=True).popen_kwargs() == {
            'start_new_session': True,
        }


# ===========================================================================
# 3. A child that ignores the soft signal must still be killed
# ===========================================================================

class TestHardKillEscalation:
    """SIGKILL / ``taskkill /F /T`` is the promise that nothing can refuse."""

    def test_stubborn_child_is_killed_after_the_grace_period(self, managed):
        """A child ignoring SIGTERM / CTRL_BREAK is still reaped.

        ``grace`` is passed explicitly (production default is 10s) so the
        escalation is genuinely exercised without the suite sleeping for ten
        seconds.  ``escalations == [False, True]`` is the direct evidence that
        the soft attempt was made first and the forced path then ran.
        """
        profile, escalations = recording_profile()
        proc = managed(STUBBORN, label='api', profile=profile)

        child = proc.start()
        wait_for_line(proc, 'ready')
        assert proc.is_running()
        assert_safe_to_signal(child)

        started = time.monotonic()
        proc.stop(grace=SHORT_GRACE)
        elapsed = time.monotonic() - started

        assert child.poll() is not None, (
            'a child that ignores the polite signal survived stop(): the '
            'forced kill path does not work on this platform'
        )
        assert escalations == [False, True], (
            'expected a soft signal followed by a forced one, got %r'
            % escalations
        )
        assert elapsed < SHORT_GRACE + 12, (
            'the forced kill took %.1fs' % elapsed
        )

        if not IS_WINDOWS:
            # SIG_IGN cannot block SIGKILL, so this pins down which signal
            # actually did it: the escalation, not the ignored SIGTERM.
            assert child.returncode == -signal.SIGKILL


# ===========================================================================
# 4. No orphans: the grandchild must die with the parent
# ===========================================================================

class TestNoOrphanedGrandchildren:
    """The entire reason for process groups and ``taskkill /T``."""

    def test_grandchild_does_not_survive_the_parent(self, managed):
        """Kill the supervised child; its own child must go too.

        Django's autoreloader and Vite's esbuild helpers are grandchildren of
        the launcher.  If only the direct child were signalled, those would
        survive and keep holding ports 8000 / 5173 — the exact failure this
        test exists to catch.
        """
        proc = managed(GRANDPARENT, label='web')
        child = proc.start()
        line = wait_for_line(proc, 'grandchild ')
        grandchild_pid = int(line.split()[-1])
        assert_safe_to_signal(child)

        try:
            assert pid_is_alive(grandchild_pid), (
                'the grandchild was never running, so this test would pass '
                'for the wrong reason'
            )

            proc.stop(grace=GRACE)
            assert proc.returncode() is not None

            assert wait_until_dead(grandchild_pid), (
                'grandchild pid %d outlived its parent: stopping the launcher '
                'leaves an orphan holding the port'
                % grandchild_pid
            )
        finally:
            hard_kill(grandchild_pid)


# ===========================================================================
# 5. popen_with_fallback: the Windows batch-shim path
#
# Windows-only production logic, exercised on every platform by faking the
# two things that make it Windows-only: ``os.name`` and the WinError that
# CreateProcess raises for a .cmd shim.  Nothing real is spawned.
# ===========================================================================

#: The npm shim as it looks on a default Windows install: a .cmd batch file
#: under a directory whose name contains a space.
NPM_CMD = r'C:\Program Files\nodejs\npm.cmd'
NPM_ARGV = [NPM_CMD, 'run', 'dev', '--', '--port', '5173', '--strictPort']

#: What the launcher must hand to CreateProcess on the retry.  The inner
#: quotes around the exe are what ``/s`` consumes; cmd.exe then sees the
#: outermost pair as the command line and the inner pair as the quoted path.
EXPECTED_LINE = (
    'cmd.exe /d /s /c ""C:\\Program Files\\nodejs\\npm.cmd" run dev -- '
    '--port 5173 --strictPort"'
)


class TestComspecFallback:
    """WinError 193: CreateProcess refused the .cmd, retry through cmd.exe."""

    @staticmethod
    def _popen_raising(winerror, sentinel, calls):
        """A ``Popen`` double: fails once with *winerror*, then succeeds."""
        def fake_popen(command, **kwargs):
            calls.append((command, dict(kwargs)))
            if len(calls) == 1:
                error = OSError(8, '%1 is not a valid Win32 application')
                error.winerror = winerror
                raise error
            return sentinel
        return fake_popen

    def _as_windows(self, monkeypatch, comspec='cmd.exe'):
        monkeypatch.setattr(dev_runner.os, 'name', 'nt')
        if comspec is None:
            monkeypatch.delenv('COMSPEC', raising=False)
        else:
            monkeypatch.setenv('COMSPEC', comspec)

    def test_fallback_command_line_is_exactly_right(self, monkeypatch):
        """The retried command line, quoting included, is locked in here.

        ``npm`` on Windows is ``npm.cmd``, and the default install path
        contains a space.  ``cmd.exe /d /s /c "<line>"`` with the executable
        itself quoted is the documented way to run a batch file: ``/s`` makes
        cmd.exe strip the outer quotes and treat the rest verbatim, so the
        inner ``"C:\\Program Files\\nodejs\\npm.cmd"`` survives as one token.
        Get this wrong and Windows users get "C:\\Program is not recognised".
        """
        calls = []
        sentinel = object()
        self._as_windows(monkeypatch)
        monkeypatch.setattr(dev_runner.subprocess, 'Popen',
                            self._popen_raising(193, sentinel, calls))

        kwargs = {
            'cwd': r'C:\Users\you\FYP\frontend',
            'stdout': subprocess.PIPE,
            'stderr': subprocess.STDOUT,
            'text': True,
            'creationflags': CREATE_NEW_PROCESS_GROUP,
        }
        result = popen_with_fallback(NPM_ARGV, **kwargs)

        assert result is sentinel
        assert len(calls) == 2, 'expected one failed attempt and one retry'

        # First attempt: the argv list, untouched.
        assert calls[0][0] == NPM_ARGV

        # Retry: a single string, because that is what makes cmd.exe parse it.
        retried, retried_kwargs = calls[1]
        assert isinstance(retried, str)
        assert retried == EXPECTED_LINE
        assert retried.startswith('cmd.exe /d /s /c "')
        assert retried.endswith('"')
        assert '"C:\\Program Files\\nodejs\\npm.cmd"' in retried

        # The retry must keep the spawn flags, or the fallback child escapes
        # the process group and Ctrl-Break / taskkill can no longer reach it.
        assert retried_kwargs == kwargs
        assert retried_kwargs['creationflags'] & CREATE_NEW_PROCESS_GROUP

    def test_bare_npm_run_dev_produces_the_documented_line(self, monkeypatch):
        """The minimal, hand-verified case, pinned on its own.

        ``cmd.exe /d /s /c ""C:\\Program Files\\nodejs\\npm.cmd" run dev"``
        — doubled quotes at the start and a single one at the end are correct
        and deliberate, and look enough like a typo that someone will
        eventually "fix" them.  This is that someone's failing test.
        """
        calls = []
        sentinel = object()
        self._as_windows(monkeypatch)
        monkeypatch.setattr(dev_runner.subprocess, 'Popen',
                            self._popen_raising(193, sentinel, calls))

        assert popen_with_fallback([NPM_CMD, 'run', 'dev']) is sentinel
        assert calls[1][0] == (
            'cmd.exe /d /s /c ""C:\\Program Files\\nodejs\\npm.cmd" run dev"'
        )

    def test_format_cmd_quotes_a_path_containing_spaces(self):
        """The half of the line that ``format_cmd`` is responsible for."""
        assert format_cmd(NPM_ARGV) == (
            '"C:\\Program Files\\nodejs\\npm.cmd" run dev -- '
            '--port 5173 --strictPort'
        )
        # Already-quoted parts are left alone rather than double-quoted.
        assert format_cmd(['"a b"']) == '"a b"'
        # And a plain argv is untouched.
        assert format_cmd(['npm', 'run', 'dev']) == 'npm run dev'

    def test_comspec_is_honoured_when_set(self, monkeypatch):
        """A non-default ``%COMSPEC%`` is used verbatim."""
        calls = []
        sentinel = object()
        self._as_windows(monkeypatch, comspec=r'C:\Windows\System32\cmd.exe')
        monkeypatch.setattr(dev_runner.subprocess, 'Popen',
                            self._popen_raising(193, sentinel, calls))

        assert popen_with_fallback(NPM_ARGV) is sentinel
        assert calls[1][0].startswith(r'C:\Windows\System32\cmd.exe /d /s /c "')

    def test_comspec_defaults_to_cmd_exe_when_unset(self, monkeypatch):
        """A stripped environment must not break the fallback."""
        calls = []
        sentinel = object()
        self._as_windows(monkeypatch, comspec=None)
        monkeypatch.setattr(dev_runner.subprocess, 'Popen',
                            self._popen_raising(193, sentinel, calls))

        assert popen_with_fallback(NPM_ARGV) is sentinel
        assert calls[1][0] == EXPECTED_LINE

    @pytest.mark.parametrize('winerror, retried', [
        (193, True),    # %1 is not a valid Win32 application  (the .cmd case)
        (216, True),    # image type mismatch — same remedy
        (2, False),     # file not found: retrying through cmd.exe hides it
        (5, False),     # access denied: likewise
        (None, False),  # a plain POSIX OSError carries no winerror at all
    ])
    def test_only_the_batch_shim_errors_are_retried(self, monkeypatch,
                                                    winerror, retried):
        """Every other OSError must propagate, not be papered over."""
        calls = []
        sentinel = object()
        self._as_windows(monkeypatch)

        def fake_popen(command, **kwargs):
            calls.append(command)
            if len(calls) == 1:
                error = OSError(8, 'boom')
                if winerror is not None:
                    error.winerror = winerror
                raise error
            return sentinel

        monkeypatch.setattr(dev_runner.subprocess, 'Popen', fake_popen)

        if retried:
            assert popen_with_fallback(NPM_ARGV) is sentinel
            assert len(calls) == 2
        else:
            with pytest.raises(OSError):
                popen_with_fallback(NPM_ARGV)
            assert len(calls) == 1

    @pytest.mark.parametrize('argv0', [
        r'C:\Program Files\nodejs\npm.exe',   # not a batch file
        'python',
    ])
    def test_non_batch_executables_are_not_retried(self, monkeypatch, argv0):
        """The fallback is for .cmd/.bat shims only."""
        calls = []
        self._as_windows(monkeypatch)

        def fake_popen(command, **kwargs):
            calls.append(command)
            error = OSError(8, 'boom')
            error.winerror = 193
            raise error

        monkeypatch.setattr(dev_runner.subprocess, 'Popen', fake_popen)
        with pytest.raises(OSError):
            popen_with_fallback([argv0, 'run', 'dev'])
        assert len(calls) == 1

    def test_fallback_never_fires_off_windows(self, monkeypatch):
        """On POSIX a .cmd argv is simply a missing file; do not mangle it."""
        calls = []
        monkeypatch.setattr(dev_runner.os, 'name', 'posix')

        def fake_popen(command, **kwargs):
            calls.append(command)
            error = OSError(8, 'boom')
            error.winerror = 193       # nonsensical here, and must be ignored
            raise error

        monkeypatch.setattr(dev_runner.subprocess, 'Popen', fake_popen)
        with pytest.raises(OSError):
            popen_with_fallback(NPM_ARGV)
        assert len(calls) == 1

    def test_happy_path_spawns_once_and_returns_the_process(self, managed):
        """No fallback, no wrapper: a working command is spawned directly."""
        proc = managed(SLEEPER, label='api')
        child = proc.start()          # start() goes through popen_with_fallback
        wait_for_line(proc, 'ready')
        assert child.poll() is None
        assert_safe_to_signal(child)
        proc.stop(grace=GRACE)
        assert child.poll() is not None


# ===========================================================================
# 6. The live port probe (only reached from phase_start_servers)
# ===========================================================================

class TestPortProbe:
    """``port_is_free`` / ``free_port`` decide which ports the servers get."""

    def test_a_bound_port_is_reported_busy(self):
        """Bound and listening for real — no assumption about any port."""
        sock, port = occupied_port()
        try:
            assert port_is_free(port) is False, (
                'port %d is bound and listening but the probe called it free; '
                'the launcher would hand it to a server that cannot bind it'
                % port
            )
        finally:
            sock.close()

    def test_a_released_port_is_reported_free(self):
        """The same probe must not report everything busy."""
        port = free_port_number()
        assert port_is_free(port) is True

    def test_free_port_returns_the_preferred_port_when_it_is_free(self):
        port = free_port_number()
        assert free_port(port) == port

    def test_free_port_skips_an_occupied_port(self):
        sock, port = occupied_port()
        try:
            chosen = free_port(port)
            assert chosen != port
            assert chosen > port
            assert port_is_free(chosen)
        finally:
            sock.close()

    def test_resolve_ports_moves_the_api_off_a_busy_port(self):
        """``Launcher._resolve_ports`` — live probe, real parsed args.

        This is the code that runs at the top of ``phase_start_servers`` and
        has therefore never executed in CI.  The warning matters as much as
        the port: a user whose API silently moved to 8001 needs to be told.
        """
        sock, busy = occupied_port()
        try:
            args = dev_runner.build_parser().parse_args(
                ['--backend-only', '--api-port', str(busy), '--no-browser'])
            launcher = dev_runner.Launcher(args)
            assert launcher.api_port == busy

            launcher._resolve_ports()

            assert launcher.api_port != busy
            assert launcher.api_port > busy
            assert port_is_free(launcher.api_port)
            assert any('busy' in message for message in launcher.warnings), (
                'the API port moved without telling the user: %r'
                % launcher.warnings
            )
        finally:
            sock.close()

    def test_resolve_ports_never_gives_both_servers_the_same_port(self):
        """Asking for one port twice must still yield two distinct ports."""
        port = free_adjacent_ports()

        args = dev_runner.build_parser().parse_args(
            ['--api-port', str(port), '--web-port', str(port), '--no-browser'])
        launcher = dev_runner.Launcher(args)

        launcher._resolve_ports()

        assert launcher.api_port == port
        assert launcher.web_port != launcher.api_port
        assert launcher.web_port > launcher.api_port
        assert port_is_free(launcher.web_port)
