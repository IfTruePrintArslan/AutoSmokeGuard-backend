"""
Object- and view-level permission classes shared by the feature apps.

AutoSmokeGuard has exactly two roles (``user`` and ``admin``, see
``accounts.User.role``).  Regular users may only ever see their own uploads,
analyses and reports; admins may see everything and additionally manage the
system configuration singleton (UC-09).
"""
from rest_framework.permissions import SAFE_METHODS, BasePermission


def _is_admin(user):
    """True when ``user`` is an authenticated administrator."""
    return bool(
        user
        and user.is_authenticated
        and (getattr(user, 'role', None) == 'admin' or user.is_staff)
    )


def _owner_of(obj):
    """
    Best-effort "who owns this record?".

    Most records carry a direct ``user`` FK.  A few (e.g. a report hanging off
    an analysis) reach it one hop away, so fall back to ``obj.owner`` and then
    to the object itself being a user.
    """
    if hasattr(obj, 'user'):
        return obj.user
    if hasattr(obj, 'owner'):
        return obj.owner
    if hasattr(obj, 'user_id') and hasattr(obj, 'set_password'):
        # The object *is* a User row.
        return obj
    return None


class IsAdminRole(BasePermission):
    """Allow only administrators (``role == 'admin'`` or ``is_staff``)."""

    message = 'Administrator privileges are required for this action.'

    def has_permission(self, request, view):
        return _is_admin(request.user)

    def has_object_permission(self, request, view, obj):
        return _is_admin(request.user)


class IsOwner(BasePermission):
    """Allow only the user the object belongs to."""

    message = 'You do not have access to this resource.'

    def has_object_permission(self, request, view, obj):
        user = request.user
        if not (user and user.is_authenticated):
            return False
        return _owner_of(obj) == user


class IsOwnerOrAdmin(BasePermission):
    """Allow the owning user, or any administrator."""

    message = 'You do not have access to this resource.'

    def has_object_permission(self, request, view, obj):
        user = request.user
        if not (user and user.is_authenticated):
            return False
        return _is_admin(user) or _owner_of(obj) == user


class IsAdminOrReadOnly(BasePermission):
    """
    Read for any authenticated user, write for administrators only.

    Used by the system-configuration endpoints, where every user needs to know
    the current upload limits but only an admin may change them.
    """

    message = 'Administrator privileges are required to modify this resource.'

    def has_permission(self, request, view):
        user = request.user
        if request.method in SAFE_METHODS:
            return bool(user and user.is_authenticated)
        return _is_admin(user)
