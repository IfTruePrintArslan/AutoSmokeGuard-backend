"""
Abstract model mixins shared by the AutoSmokeGuard feature apps.

Neither class here creates a table — both are ``abstract`` — so this app
intentionally ships an empty migration history.
"""
from django.db import models


class UUIDModel(models.Model):
    """
    Marker base for models whose primary key is a UUID.

    It deliberately declares **no** primary-key field.  The SDS gives every
    table its own descriptive PK name (``user_id``, ``media_id``,
    ``analysis_id``, ``vehicle_id``, ``smoke_id``, ``report_id``, ``token_id``,
    ...), so each concrete model declares its own::

        analysis_id = models.UUIDField(
            primary_key=True, default=uuid.uuid4, editable=False,
        )

    Subclassing this mixin documents that intent, gives the codebase a single
    ``isinstance`` check for "UUID-keyed record", and provides the shared
    helpers below.
    """

    class Meta:
        abstract = True

    @property
    def pk_str(self):
        """The record's primary key as a plain string (handy for paths/URLs)."""
        return str(self.pk)


class TimeStampedModel(models.Model):
    """Adds self-maintaining ``created_at`` / ``updated_at`` columns."""

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True
