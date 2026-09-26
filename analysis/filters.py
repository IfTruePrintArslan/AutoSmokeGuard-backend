"""
Query filters for the analysis list and the history screen (UC-08).

The history table is the second-most-hit authenticated endpoint in the
product and the NFR gives it a two-second budget over a few thousand rows, so
every filter here is written with the query plan in mind:

* ``severity``, ``status`` and the date range hit indexed columns on
  ``analysis_result`` directly — no joins at all.
* ``search`` joins ``uploads_media`` on an indexed foreign key, one row to one
  row, so the join cannot multiply the result set.
* ``vehicle_type`` is the awkward one.  ``.filter(vehicles__vehicle_type=...)``
  joins a *to-many* relation, which returns one row per matching detection —
  a truck seen in forty frames would appear forty times — and the usual fix,
  ``.distinct()``, forces the database to sort or hash the entire result set
  before paginating.  A ``pk__in`` subquery over the covered
  ``(analysis, frame_number)`` index gives the same answer, keeps every row
  unique by construction, and leaves the outer query able to stop at
  ``LIMIT 10``.  No ``.distinct()`` is needed anywhere in this module.

``ordering`` is a strict whitelist: an unrecognised value is a 400 rather
than a silently ignored parameter, so a typo in the front end surfaces
immediately instead of quietly serving the wrong order.
"""
import django_filters

from .models import STATUS_CHOICES, AnalysisResult, DetectedVehicle

#: Sort keys the client may ask for, ascending and descending.
ORDERING_FIELDS = (
    ('created_at', 'created_at'),
    ('total_vehicles', 'total_vehicles'),
    ('avg_confidence', 'avg_confidence'),
)

#: Exactly the strings the contract promises to accept for ``?ordering=``.
ALLOWED_ORDERING = tuple(
    prefix + name for name, _label in ORDERING_FIELDS for prefix in ('', '-')
)

SEVERITY_FILTER_CHOICES = (
    ('low', 'Low'),
    ('moderate', 'Moderate'),
    ('high', 'High'),
)

VEHICLE_TYPE_FILTER_CHOICES = (
    ('car', 'Car'),
    ('motorcycle', 'Motorcycle'),
    ('bus', 'Bus'),
    ('truck', 'Truck'),
)


class AnalysisFilterSet(django_filters.FilterSet):
    """
    Filters shared by ``GET /api/analysis`` and ``GET /api/history``.

    Every parameter is optional and independent; they compose with ``AND``.
    """

    status = django_filters.ChoiceFilter(
        choices=STATUS_CHOICES,
        help_text='pending | queued | running | done | failed',
    )
    severity = django_filters.ChoiceFilter(
        field_name='overall_severity',
        choices=SEVERITY_FILTER_CHOICES,
        help_text="Overall verdict of the run: low | moderate | high.",
    )
    vehicle_type = django_filters.ChoiceFilter(
        choices=VEHICLE_TYPE_FILTER_CHOICES,
        method='filter_vehicle_type',
        help_text='Keep analyses containing at least one vehicle of this type.',
    )
    date_from = django_filters.DateFilter(
        field_name='created_at',
        lookup_expr='date__gte',
        help_text='Inclusive lower bound on the run date (YYYY-MM-DD).',
    )
    date_to = django_filters.DateFilter(
        field_name='created_at',
        lookup_expr='date__lte',
        help_text='Inclusive upper bound on the run date (YYYY-MM-DD).',
    )
    search = django_filters.CharFilter(
        field_name='media__filename',
        lookup_expr='icontains',
        help_text='Case-insensitive substring match on the source filename.',
    )
    ordering = django_filters.OrderingFilter(
        fields=ORDERING_FIELDS,
        help_text='One of: ' + ', '.join(ALLOWED_ORDERING),
    )

    class Meta:
        model = AnalysisResult
        fields = ('status', 'severity', 'vehicle_type', 'date_from',
                  'date_to', 'search', 'ordering')

    def filter_vehicle_type(self, queryset, name, value):
        """
        Keep analyses that detected at least one vehicle of ``value``.

        Implemented as a subquery rather than a join + ``.distinct()``; see
        the module docstring for why.
        """
        if not value:
            return queryset
        matching = (
            DetectedVehicle.objects
            .filter(vehicle_type=value)
            .values('analysis_id')
        )
        return queryset.filter(pk__in=matching)
