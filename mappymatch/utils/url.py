from functools import reduce
from typing import List
from urllib.parse import urljoin


def _parse_uri(uri: str) -> str:
    """Internal use."""
    return uri if uri.endswith("/") else f"{uri}/"


def multiurljoin(urls: List[str]) -> str:
    """
    Join multiple URL components into a single URL.

    This function combines a list of URL path components, ensuring proper forward
    slash separators between components. Each component is normalized to end with
    a slash before joining.

    Args:
        urls: A list of URL components to join. Can include protocol, domain, and path segments.
            Example: ['http://localhost:5000', 'api', 'v1', 'match']

    Returns:
        A complete URL with all components joined using forward slashes.
        Example: 'http://localhost:5000/api/v1/match/'

    Examples:
        >>> multiurljoin(['http://localhost:5000', 'api', 'v1', 'match'])
        'http://localhost:5000/api/v1/match/'
        >>>
        >>> # Build OSRM API endpoint
        >>> base = 'http://router.project-osrm.org'
        >>> endpoint = multiurljoin([base, 'match', 'v1', 'driving'])
        >>> print(endpoint)  # 'http://router.project-osrm.org/match/v1/driving/'
    """
    parsed_urls = [_parse_uri(uri) for uri in urls]
    return reduce(urljoin, parsed_urls)
