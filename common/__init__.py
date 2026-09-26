"""
Shared building blocks for every AutoSmokeGuard app.

``common`` deliberately owns no database tables of its own.  It is the place
for the cross-cutting pieces the feature apps all need: abstract model mixins,
the paginated response envelope, the error envelope, object-level permissions,
request logging, upload validation and media path helpers.
"""
