"""
Python-version-floor tests for the developer launcher (``tools/dev_runner.py``).

The fact these tests pin (measured in CI, not re-derived here): Django 6.0
(pinned in ``backend/requirements.txt``) declares ``Requires-Python
>=3.12``. On Python 3.11, ``pip install -r requirements.txt`` cannot even
resolve the dependency:

    ERROR: Ignored the following versions that require a different python
    version: 6.0 Requires-Python >=3.12
    ERROR: Could not find a version that satisfies the requirement
    Django==6.0.6

That is an opaque, minutes-later failure with no hint that the interpreter
itself is the problem. Before this fix, ``find_python()`` happily offered
``python3.11`` as a candidate and would silently keep reusing a pre-existing
``backend/.venv`` built with Python 3.11 (``phase_backend_venv`` only checks
whether ``backend/.venv``'s python *exists*, not what version it is) — a
user on 3.11 only discovered the problem inside that pip resolver error.

These tests exercise the real, shipped code (``check_python_floor``,
``find_python``), not a re-implementation, so a regression that reopens the
3.11 gap is caught here instead of in someone's confusing pip failure.
"""
import pytest

from tools import dev_runner
from tools.dev_runner import LauncherError, MIN_PYTHON, check_python_floor


class TestMinPythonFloor:
    """Pins the constant itself: the floor is 3.12, not merely ``>= 3.10``."""

    def test_min_python_is_the_django_6_floor(self):
        assert MIN_PYTHON == (3, 12)


class TestCheckPythonFloor:
    """Direct tests of the standalone preflight gate.

    ``check_python_floor`` is what turns "the selected interpreter is too
    old" into an explicit, actionable failure *before* preflight hands off
    to ``pip install`` — this is "the preflight check" the fix describes.
    """

    def test_a_3_11_interpreter_is_rejected(self):
        with pytest.raises(LauncherError) as exc_info:
            check_python_floor((3, 11), 'python3.11')
        message = str(exc_info.value)
        assert '3.11' in message
        assert '3.12' in message

    def test_the_rejection_names_django_6_as_the_reason(self):
        with pytest.raises(LauncherError) as exc_info:
            check_python_floor((3, 11), 'python3.11')
        assert 'Django' in exc_info.value.detail
        assert '6.0' in exc_info.value.detail

    def test_an_unprobeable_interpreter_is_rejected(self):
        # version=None means the interpreter could not even be run — treated
        # as failing the floor, not silently accepted.
        with pytest.raises(LauncherError):
            check_python_floor(None, 'mystery-python')

    def test_older_than_3_11_is_also_rejected(self):
        with pytest.raises(LauncherError):
            check_python_floor((3, 9), 'python3.9')

    def test_a_3_12_interpreter_is_accepted(self):
        check_python_floor((3, 12), 'python3.12')  # must not raise

    def test_a_newer_interpreter_is_accepted(self):
        check_python_floor((3, 14), 'python3.14')  # must not raise

    def test_existing_venv_rejection_tells_the_user_to_delete_it(self):
        with pytest.raises(LauncherError) as exc_info:
            check_python_floor((3, 11), 'existing project virtualenv',
                                is_existing_venv=True)
        assert 'delete' in exc_info.value.hint.lower()
        assert '.venv' in exc_info.value.message


class TestFindPythonExistingVenvFloor:
    """``find_python()``'s handling of a pre-existing ``backend/.venv``.

    This is the actual user-facing bug: ``phase_backend_venv`` only rebuilds
    the venv when its python *does not exist on disk*, so a stale 3.11 venv
    was previously reused silently. ``find_python()`` must fail loudly for
    it instead of letting a caller reuse it.
    """

    def test_a_3_11_venv_is_rejected_before_pip_ever_runs(self, monkeypatch, tmp_path):
        venv_python = tmp_path / 'python'
        venv_python.write_text('#!/bin/sh\n')
        venv_python.chmod(0o755)
        monkeypatch.setattr(
            dev_runner, 'probe_python', lambda cmd, timeout=25: (3, 11))

        with pytest.raises(LauncherError) as exc_info:
            dev_runner.find_python(venv_python=venv_python, is_windows=False)

        message = str(exc_info.value)
        assert '3.11' in message
        assert 'delete' in exc_info.value.hint.lower()

    def test_a_3_12_venv_is_accepted_without_searching_further(self, monkeypatch, tmp_path):
        venv_python = tmp_path / 'python'
        venv_python.write_text('#!/bin/sh\n')
        venv_python.chmod(0o755)
        monkeypatch.setattr(
            dev_runner, 'probe_python', lambda cmd, timeout=25: (3, 12))

        argv, version, source = dev_runner.find_python(
            venv_python=venv_python, is_windows=False)

        assert version == (3, 12)
        assert source == 'existing project virtualenv'
        assert argv == [str(venv_python)]

    def test_a_missing_venv_path_is_not_treated_as_a_floor_violation(self, tmp_path):
        # No venv exists yet: find_python() must fall through to searching
        # the rest of its candidates rather than raising about a venv that
        # was never there. Real candidates are used here since none of this
        # machine's real interpreters can be below MIN_PYTHON without the
        # whole backend test suite itself failing to even start.
        argv, version, source = dev_runner.find_python(
            venv_python=tmp_path / 'does-not-exist', is_windows=False)
        assert version >= MIN_PYTHON
