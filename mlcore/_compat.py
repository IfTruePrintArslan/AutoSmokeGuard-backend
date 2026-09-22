"""Import-time compatibility guards for :mod:`mlcore`.

This module exists for exactly one reason: to protect the ML stack from
*standard-library shadowing*.

When ``mlcore`` is imported from a Django project the project root sits on
``sys.path``.  Any top-level package in that root whose name collides with a
standard-library module wins the import race, and third-party libraries that
expect the real stdlib module blow up at import time.

The concrete collision this guards against is a local package named
``sysconfig``: ``torch._dynamo.config`` calls ``sysconfig.get_config_var()``
while it is being imported and raises ``AttributeError`` if ``sysconfig``
resolves to anything other than the real module.  That failure cascades
through ``torchvision`` and ``ultralytics``.

Everything in here is a no-op when no collision is present.  It is *not* a
substitute for renaming the offending package -- it only keeps ``mlcore``
usable while that happens.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from types import ModuleType
from typing import Iterable

logger = logging.getLogger("asg.ml")

#: Stdlib modules that the ML stack imports transitively and that must never be
#: shadowed.  Deliberately narrow: repairing an arbitrary name could hijack a
#: legitimate application package.
PROTECTED_STDLIB_MODULES: tuple[str, ...] = ("sysconfig",)

_STDLIB_DIR = os.path.dirname(os.path.abspath(os.__file__))


def _is_stdlib_path(path: str | None) -> bool:
    """Return ``True`` when *path* lives inside the interpreter's stdlib dir."""
    if not path:
        return False
    return os.path.abspath(path).startswith(_STDLIB_DIR)


def _load_real_stdlib_module(name: str) -> ModuleType | None:
    """Import *name* straight from the stdlib directory, bypassing ``sys.path``."""
    pkg_init = os.path.join(_STDLIB_DIR, name, "__init__.py")
    plain = os.path.join(_STDLIB_DIR, name + ".py")

    if os.path.isfile(pkg_init):
        spec = importlib.util.spec_from_file_location(
            name, pkg_init, submodule_search_locations=[os.path.join(_STDLIB_DIR, name)]
        )
    elif os.path.isfile(plain):
        spec = importlib.util.spec_from_file_location(name, plain)
    else:  # frozen / builtin / unusual layout -- nothing we can do
        return None

    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover - defensive
        sys.modules.pop(name, None)
        return None
    return module


def _shadow_origin(name: str) -> str | None:
    """Return the non-stdlib file that currently owns *name*, if any."""
    existing = sys.modules.get(name)
    if existing is not None:
        origin = getattr(existing, "__file__", None)
        return None if _is_stdlib_path(origin) else (origin or "<unknown>")

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return None
    if spec is None:
        return None
    if spec.origin in (None, "built-in", "frozen"):
        return None
    return None if _is_stdlib_path(spec.origin) else spec.origin


def ensure_stdlib(names: Iterable[str] = PROTECTED_STDLIB_MODULES) -> list[str]:
    """Force *names* to resolve to their real standard-library implementations.

    Args:
        names: Module names to check.  Defaults to :data:`PROTECTED_STDLIB_MODULES`.

    Returns:
        The list of module names that were actually repaired (empty in the
        healthy case).
    """
    repaired: list[str] = []
    for name in names:
        shadow = _shadow_origin(name)
        if shadow is None:
            continue
        if _load_real_stdlib_module(name) is None:
            logger.error(
                "Stdlib module %r is shadowed by %s and could not be repaired; "
                "rename that package.",
                name,
                shadow,
            )
            continue
        repaired.append(name)
        logger.warning(
            "Stdlib module %r was shadowed by %s. mlcore has re-bound the real "
            "standard-library module for this process -- please rename the "
            "offending package, this shim is a stopgap.",
            name,
            shadow,
        )
    return repaired
