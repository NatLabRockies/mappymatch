from __future__ import annotations

import json
import logging
from typing import List, Tuple

import numpy as np
import polyline
import requests
from shapely.geometry import LineString

from mappymatch.constructs.match import Match
from mappymatch.constructs.road import Road, RoadId
from mappymatch.constructs.trace import Trace
from mappymatch.matchers.matcher_interface import MatcherInterface, MatchResult
from mappymatch.utils.crs import LATLON_CRS

log = logging.getLogger(__name__)

DEMO_VALHALLA_ADDRESS = "https://valhalla1.openstreetmap.de/trace_attributes"
REQUIRED_ATTRIBUTES = set(
    [
        "edge.way_id",
        "matched.distance_from_trace_point",
        "shape",
        "edge.begin_shape_index",
        "edge.end_shape_index",
        "matched.edge_index",
    ]
)
DEFAULT_ATTRIBUTES = set(
    [
        "edge.length",
        "edge.speed",
    ]
)


def build_path_from_result(
    edges: List[dict], shape: List[Tuple[float, float]]
) -> List[Road]:
    """
    Build a list of Road objects from Valhalla map matching response.

    This parses the 'edges' array from a Valhalla trace_attributes response and converts
    each edge into a Road object with geometry and metadata.

    Args:
        edges: List of edge dictionaries from the Valhalla response, containing
            way_id, begin_shape_index, end_shape_index, speed, length, etc.
        shape: List of (lon, lat) coordinate tuples representing the matched path,
            decoded from Valhalla's polyline

    Returns:
        A list of Road objects representing the matched path through the network
    """
    path = []
    for edge in edges:
        way_id = edge["way_id"]
        road_id = RoadId(start=None, end=None, key=way_id)
        start_point_i = edge["begin_shape_index"]
        end_point_i = edge["end_shape_index"]
        start_point = shape[start_point_i]
        end_point = shape[end_point_i]
        geom = LineString([start_point, end_point])

        speed = edge["speed"]
        length = edge["length"]

        metadata = {
            "speed_mph": speed,
            "length_miles": length,
        }

        road = Road(road_id=road_id, geom=geom, metadata=metadata)

        path.append(road)

    return path


def build_match_result(
    trace: Trace, matched_points: List[dict], path: List[Road]
) -> MatchResult:
    """
    Build a MatchResult from Valhalla map matching response data.

    This combines the Valhalla matched_points array with the parsed path to create
    Match objects linking each GPS coordinate to its matched road segment.

    Args:
        trace: The original GPS trace that was submitted for matching
        matched_points: List of matched point dictionaries from Valhalla response, containing edge_index and distance_from_trace_point for each GPS point
        path: List of Road objects representing the matched route (from build_path_from_result)

    Returns:
        A MatchResult containing matches for each coordinate and the full path
    """
    matches = []
    for i, coord in enumerate(trace.coords):
        mp = matched_points[i]
        ei = mp.get("edge_index")
        dist = mp.get("distance_from_trace_point")
        if ei is None:
            road = None
        else:
            try:
                road = path[ei]
            except IndexError:
                road = None

        if dist is None:
            dist = np.inf

        match = Match(road, coord, dist)

        matches.append(match)

    return MatchResult(matches=matches, path=path)


class ValhallaMatcher(MatcherInterface):
    """
    Map matcher that uses a Valhalla server for matching GPS traces to road networks.

    Valhalla is an open-source routing engine that provides map matching capabilities.
    This matcher sends GPS coordinates to a Valhalla server and receives back matched
    road segments and routing results.

    The matcher communicates with Valhalla's trace_attributes API endpoint, which returns
    detailed information about matched edges including geometry, speed, and length.

    Args:
        valhalla_url: The base URL of the Valhalla trace_attributes endpoint.
            Default is a public demo server (not for production use).
        cost_model: The routing cost model to use ('auto', 'bicycle', 'pedestrian', etc.).
            Default is 'auto'.
        shape_match: The shape matching algorithm ('map_snap', 'edge_walk', 'walk_or_snap').
            Default is 'map_snap'.
        attributes: Additional edge attributes to request from Valhalla beyond the required ones.
            Default includes 'edge.length' and 'edge.speed'.

    Attributes:
        url_base: The Valhalla API endpoint URL
        cost_model: The routing cost model being used
        shape_match: The shape matching algorithm being used
        attributes: List of all requested edge attributes (required + additional)

    Examples:
        >>> from mappymatch.matchers.valhalla import ValhallaMatcher
        >>>
        >>> # Use default demo server (for testing only)
        >>> matcher = ValhallaMatcher()
        >>> result = matcher.match_trace(trace)
        >>>
        >>> # Use your own Valhalla instance
        >>> matcher = ValhallaMatcher(
        ...     valhalla_url='http://localhost:8002/trace_attributes',
        ...     cost_model='bicycle'
        ... )
        >>>
        >>> # Request additional attributes
        >>> matcher = ValhallaMatcher(
        ...     attributes=['edge.length', 'edge.speed', 'edge.names', 'edge.surface']
        ... )

    Note:
        The default demo server is rate-limited and should only be used for testing.
        For production use, deploy your own Valhalla instance.
    """

    def __init__(
        self,
        valhalla_url=DEMO_VALHALLA_ADDRESS,
        cost_model="auto",
        shape_match="map_snap",
        attributes=DEFAULT_ATTRIBUTES,
    ):
        self.url_base = valhalla_url
        self.cost_model = cost_model
        self.shape_match = shape_match

        all_attributes = list(REQUIRED_ATTRIBUTES.union(set(attributes)))
        self.attributes = all_attributes

    def match_trace(self, trace: Trace) -> MatchResult:
        """
        Match a GPS trace to roads using the Valhalla map matching service.

        This method sends the trace to a Valhalla server, which performs map matching
        and returns the matched path and statistics. The trace is automatically converted
        to EPSG:4326 (lat/lon) if needed, as required by Valhalla.

        Args:
            trace: The GPS trace to match. Will be converted to EPSG:4326 if in a different CRS.

        Returns:
            A MatchResult containing:
            - matches: List of Match objects linking each GPS point to a road
            - path: List of Road objects representing the matched route

        Raises:
            requests.HTTPError: If the Valhalla server returns an error response

        Examples:
            >>> matcher = ValhallaMatcher()
            >>> trace = Trace.from_csv('gps_data.csv')
            >>> result = matcher.match_trace(trace)
            >>>
            >>> # Access results
            >>> print(f"Matched {len(result.matches)} points")
            >>> print(f"Path has {len(result.path)} road segments")
            >>>
            >>> # Check road metadata from Valhalla
            >>> for road in result.path[:5]:
            ...     print(f"Speed: {road.metadata['speed_mph']} mph")
            ...     print(f"Length: {road.metadata['length_miles']} miles")
        """
        if not trace.crs == LATLON_CRS:
            trace = trace.to_crs(LATLON_CRS)

        points = [{"lat": c.y, "lon": c.x} for c in trace.coords]

        json_payload = json.dumps(
            {
                "shape": points,
                "costing": self.cost_model,
                "shape_match": self.shape_match,
                "filters": {
                    "attributes": self.attributes,
                    "action": "include",
                },
                "units": "miles",
            }
        )

        valhalla_request = f"{self.url_base}?json={json_payload}"

        r = requests.get(valhalla_request)

        if not r.status_code == requests.codes.ok:
            r.raise_for_status()

        j = r.json()

        edges = j["edges"]
        shape = polyline.decode(j["shape"], precision=6, geojson=True)
        matched_points = j["matched_points"]

        path = build_path_from_result(edges, shape)
        result = build_match_result(trace, matched_points, path)

        return result
