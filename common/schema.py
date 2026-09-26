"""
drf-spectacular customisation.

Registered from ``SPECTACULAR_SETTINGS['PREPROCESSING_HOOKS']``.
"""

#: Routes that exist for backwards compatibility only.  They still serve
#: traffic, but documenting them would give the same operation two entries and
#: invite new clients to adopt the deprecated spelling.
LEGACY_PATHS = frozenset({'/health/'})


def exclude_legacy_paths(endpoints, **kwargs):
    """
    Drop deprecated aliases from the generated OpenAPI schema.

    ``endpoints`` is a list of ``(path, path_regex, method, callback)``
    tuples; returning a filtered list removes those operations entirely, which
    also silences drf-spectacular's operationId-collision warning for the
    ``/health/`` alias of ``/api/health/``.
    """
    return [
        endpoint for endpoint in endpoints if endpoint[0] not in LEGACY_PATHS
    ]
