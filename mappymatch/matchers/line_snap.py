import logging
from typing import List

from mappymatch.constructs.match import Match
from mappymatch.constructs.trace import Trace
from mappymatch.maps.map_interface import MapInterface
from mappymatch.matchers.matcher_interface import MatcherInterface, MatchResult

log = logging.getLogger(__name__)


class LineSnapMatcher(MatcherInterface):
    """
    A simple, fast map matcher that snaps each GPS point to the nearest road segment.

    This is the most basic matching approach - it independently matches each GPS coordinate
    to its nearest road without considering the overall trajectory or road network topology.
    While fast and simple to implement, it may produce unrealistic results where the matched
    path jumps between disconnected roads.

    Use this matcher when:
    - You need very fast matching performance
    - GPS data is very accurate and roads are well-separated
    - You only need point-to-road snapping, not path reconstruction

    For more sophisticated matching that considers path continuity and network topology,
    use LCSSMatcher, ValhallaMatcher, or OsrmMatcher instead.

    Args:
        road_map: The road network to match against (must implement MapInterface)

    Attributes:
        map: The road network being used for matching

    Examples:
        >>> from mappymatch.matchers.line_snap import LineSnapMatcher
        >>> from mappymatch.maps.nx import NxMap
        >>>
        >>> # Load a road network
        >>> road_map = NxMap.from_file('network.pickle')
        >>>
        >>> # Create the matcher
        >>> matcher = LineSnapMatcher(road_map)
        >>> result = matcher.match_trace(trace)
        >>>
        >>> # Each point is matched independently
        >>> for match in result.matches:
        ...     print(f"Distance to road: {match.distance}m")
    """

    def __init__(self, road_map: MapInterface):
        self.map = road_map

    def match_trace(self, trace: Trace) -> MatchResult:
        """
        Match a GPS trace by snapping each point to its nearest road.

        This performs simple nearest-neighbor matching without considering trajectory
        continuity or network topology. Each GPS coordinate is independently matched
        to the spatially nearest road segment.

        Args:
            trace: The GPS trace to match. Should be in the same CRS as the road map.

        Returns:
            A MatchResult containing:
            - matches: List of Match objects, one per GPS point
            - path: None (this matcher doesn't compute paths)

        Examples:
            >>> from mappymatch.matchers.line_snap import LineSnapMatcher
            >>> from mappymatch.maps.nx import NxMap
            >>>
            >>> road_map = NxMap.from_file('network.gpickle')
            >>> matcher = LineSnapMatcher(road_map)
            >>>
            >>> trace = Trace.from_csv('gps_data.csv')
            >>> result = matcher.match_trace(trace)
            >>>
            >>> # Check matching quality
            >>> distances = [m.distance for m in result.matches]
            >>> avg_dist = sum(distances) / len(distances)
            >>> print(f"Average distance to road: {avg_dist:.2f}m")
            >>>
            >>> # Find poorly matched points
            >>> poor_matches = [m for m in result.matches if m.distance > 50]
            >>> print(f"Points > 50m from road: {len(poor_matches)}")
        """
        matches = []

        for coord in trace.coords:
            nearest_road = self.map.nearest_road(coord)
            nearest_point = nearest_road.geom.interpolate(
                nearest_road.geom.project(coord.geom)
            )
            dist = nearest_road.geom.distance(nearest_point)
            match = Match(nearest_road, coord, dist)
            matches.append(match)

        return MatchResult(matches)

    def match_trace_batch(self, trace_batch: List[Trace]) -> List[MatchResult]:
        return [self.match_trace(t) for t in trace_batch]
