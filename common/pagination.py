"""
The single pagination style used by every list endpoint in the API.

DRF's stock ``PageNumberPagination`` answers with ``{count, next, previous,
results}``, which forces the front end to parse the ``next`` URL just to learn
which page it is on.  ``StandardPagination`` adds the three numbers a paging UI
actually needs — ``page``, ``pages`` and ``page_size`` — so the React history
and admin tables can render "Page 3 of 12" without any URL archaeology.
"""
from collections import OrderedDict

from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response


class StandardPagination(PageNumberPagination):
    """
    Page-number pagination with a client-tunable, server-capped page size.

    ``?page=2&page_size=25`` — ``page_size`` is clamped to ``max_page_size`` so
    a client cannot ask for the whole analysis history in one request.
    """

    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100

    def get_paginated_response(self, data):
        """Wrap ``data`` in the project-wide list envelope."""
        return Response(OrderedDict([
            ('count', self.page.paginator.count),
            ('page', self.page.number),
            ('pages', self.page.paginator.num_pages),
            ('page_size', self.get_page_size(self.request)),
            ('next', self.get_next_link()),
            ('previous', self.get_previous_link()),
            ('results', data),
        ]))

    def get_paginated_response_schema(self, schema):
        """Describe the envelope above to drf-spectacular / OpenAPI."""
        return {
            'type': 'object',
            'required': ['count', 'page', 'pages', 'page_size', 'results'],
            'properties': {
                'count': {'type': 'integer', 'example': 123},
                'page': {'type': 'integer', 'example': 1},
                'pages': {'type': 'integer', 'example': 13},
                'page_size': {'type': 'integer', 'example': 10},
                'next': {
                    'type': 'string', 'nullable': True, 'format': 'uri',
                    'example': 'http://api.example.org/api/analysis/?page=2',
                },
                'previous': {
                    'type': 'string', 'nullable': True, 'format': 'uri',
                    'example': None,
                },
                'results': schema,
            },
        }
