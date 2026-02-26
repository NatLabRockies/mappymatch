from abc import ABCMeta, abstractmethod

from mappymatch.constructs.trace import Trace
from mappymatch.matchers.match_result import MatchResult


class MatcherInterface(metaclass=ABCMeta):
    """
    Abstract base class defining the interface for all map-matching algorithms.

    All map matchers in mappymatch implement this interface, providing a consistent API
    for matching GPS trajectories to road networks. Subclasses must implement the
    match_trace method to perform the actual matching algorithm.

    Examples:
        >>> from mappymatch.matchers.lcss import LCSSMatcher
        >>> from mappymatch.matchers.valhalla import ValhallaMatcher
        >>>
        >>> # All matchers follow the same interface
        >>> matcher = LCSSMatcher(road_map)
        >>> result = matcher.match_trace(trace)
    """

    @abstractmethod
    def match_trace(self, trace: Trace) -> MatchResult:
        """
        Match a GPS trace to the underlying road network.

        This is the primary method of any matcher. It takes a sequence of GPS coordinates
        and returns matching results that link each coordinate to road segments.

        Args:
            trace: A Trace object containing the GPS coordinates to match

        Returns:
            A MatchResult containing:
            - matches: A list of Match objects linking each GPS point to a road
            - path: An optional list of Road objects representing the matched route

        Examples:
            >>> matcher = SomeMatcher(road_map)
            >>> trace = Trace.from_csv('gps_data.csv')
            >>> result = matcher.match_trace(trace)
            >>>
            >>> # Access the matches
            >>> for match in result.matches:
            ...     if match.road is not None:
            ...         print(f"Matched to road {match.road.road_id}")
        """
