from __future__ import annotations

import logging

import requests

from mappymatch.constructs.match import Match
from mappymatch.constructs.road import Road, RoadId
from mappymatch.constructs.trace import Trace
from mappymatch.matchers.matcher_interface import MatcherInterface, MatchResult
from mappymatch.utils.crs import LATLON_CRS
from mappymatch.utils.url import multiurljoin

log = logging.getLogger(__name__)

DEFAULT_OSRM_ADDRESS = "http://router.project-osrm.org"


def parse_osrm_json(j: dict, trace: Trace) -> list[Match]:
    """
    Parse the JSON response from the OSRM match service into Match objects.

    Extracts matching information from the OSRM response and creates Match objects
    linking GPS coordinates to road segments (represented by OSM node IDs).

    Args:
        j: The JSON response dictionary from OSRM's match endpoint
        trace: The original GPS trace that was matched

    Returns:
        A list of Match objects, one for each coordinate in the trace

    Raises:
        ValueError: If the response is missing required fields (matchings, legs, annotations, or nodes)

    Note:
        Currently the geometry and distance information from OSRM is not fully utilized.
        This is a TODO for future improvement.
    """
    matchings = j.get("matchings")
    if not matchings:
        raise ValueError("could not find any link matchings in response")

    legs = matchings[0].get("legs")
    if not legs:
        raise ValueError("could not find any link legs in response")

    def _parse_leg(d: dict, i: int) -> Match:
        annotation = d.get("annotation")
        if not annotation:
            raise ValueError("leg has no annotation information")
        nodes = annotation.get("nodes")
        if not nodes:
            raise ValueError("leg has no osm node information")
        origin_junction_id = f"{nodes[0]}"
        destination_junction_id = f"{nodes[0]}"

        # TODO: we need to get geometry, distance info from OSRM if available
        road_id = RoadId(origin_junction_id, destination_junction_id, 0)
        road = Road(
            road_id=road_id,
            geom=None,
        )
        match = Match(road=road, coordinate=trace.coords[i], distance=float("infinity"))
        return match

    return [_parse_leg(d, i) for i, d in enumerate(legs)]


class OsrmMatcher(MatcherInterface):
    """
    Map matcher that uses an OSRM server for matching GPS traces to OpenStreetMap data.

    OSRM (Open Source Routing Machine) is a routing engine for OpenStreetMap data.
    This matcher sends GPS coordinates to an OSRM match endpoint and receives back
    matched road segments based on OSM node IDs.

    The matcher communicates with OSRM's match API, which snaps GPS points to the
    nearest roads in the OSM network.

    Args:
        osrm_address: The base URL of the OSRM server. Default is the public OSRM demo server (not recommended for production use).
        osrm_profile: The routing profile to use ('driving', 'walking', 'cycling', etc.). Default is 'driving'.
        osrm_version: The OSRM API version. Default is 'v1'.

    Attributes:
        osrm_api_base: The constructed OSRM API endpoint URL

    Examples:
        >>> from mappymatch.matchers.osrm import OsrmMatcher
        >>>
        >>> # Use default public OSRM server (for testing only)
        >>> matcher = OsrmMatcher()
        >>> result = matcher.match_trace(trace)
        >>>
        >>> # Use your own OSRM instance
        >>> matcher = OsrmMatcher(
        ...     osrm_address='http://localhost:5000',
        ...     osrm_profile='cycling'
        ... )

    Note:
        - Traces must be in WGS84 (EPSG:4326) coordinate system
        - Traces are automatically downsampled to 100 points if longer
        - The public demo server is rate-limited; use your own instance for production
    """

    def __init__(
        self,
        osrm_address=DEFAULT_OSRM_ADDRESS,
        osrm_profile="driving",
        osrm_version="v1",
    ):
        self.osrm_api_base = multiurljoin(
            [osrm_address, "match", osrm_version, osrm_profile]
        )

    def match_trace(self, trace: Trace) -> MatchResult:
        """
        Match a GPS trace to roads using the OSRM map matching service.

        This method sends the trace to an OSRM server for map matching against OpenStreetMap data.
        The trace must be in EPSG:4326 (lat/lon). If the trace has more than 100 points,
        it's automatically downsampled to meet OSRM's typical API limits.

        Args:
            trace: The GPS trace to match. Must be in EPSG:4326 (lat/lon).

        Returns:
            A MatchResult containing matches for each GPS point. Note that the current
            implementation has limited geometry/distance information from OSRM.

        Raises:
            TypeError: If the trace is not in EPSG:4326
            requests.HTTPError: If the OSRM server returns an error response

        Examples:
            >>> matcher = OsrmMatcher()
            >>> trace = Trace.from_csv('gps_data.csv', xy=False)  # Keep in lat/lon
            >>> result = matcher.match_trace(trace)
            >>>
            >>> # Long traces are automatically downsampled
            >>> long_trace = Trace.from_gpx('long_journey.gpx')  # 500 points
            >>> result = matcher.match_trace(long_trace)  # Downsampled to 100 points
        """
        if not trace.crs == LATLON_CRS:
            raise TypeError(
                f"this matcher requires traces to be in the CRS of EPSG:{LATLON_CRS.to_epsg()} "
                f"but found EPSG:{trace.crs.to_epsg()}"
            )

        if len(trace.coords) > 100:
            trace = trace.downsample(100)

        coordinate_str = ""
        for coord in trace.coords:
            coordinate_str += f"{coord.x},{coord.y};"

        # remove the trailing semicolon
        coordinate_str = coordinate_str[:-1]

        osrm_request = self.osrm_api_base + coordinate_str + "?annotations=true"

        r = requests.get(osrm_request)

        if not r.status_code == requests.codes.ok:
            r.raise_for_status()

        result = parse_osrm_json(r.json(), trace)

        return MatchResult(result)
