from rest_framework.decorators import api_view
from rest_framework.response import Response


@api_view(['GET'])
def health_check(request):
    """
    Liveness probe — confirms the service is up and reachable.
    Returns HTTP 200 with a minimal JSON body.
    """
    return Response({"status": "ok", "service": "autosmokeguard-backend"})
