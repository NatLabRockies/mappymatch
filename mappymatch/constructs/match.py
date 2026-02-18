from typing import NamedTuple, Optional

from mappymatch.constructs.coordinate import Coordinate
from mappymatch.constructs.road import Road


class Match(NamedTuple):
    """
    Represents a map-matching result linking a GPS coordinate to a road segment.

    A Match is the fundamental output of map-matching algorithms, connecting a GPS coordinate
    to its best-matching road segment. When no suitable road is found within the matching
    threshold, the road field is None and the distance is infinite.

    Attributes:
        road: The road segment that was matched to the coordinate. None if no suitable road was found within the matching parameters.
        coordinate: The original GPS coordinate that was matched
        distance: The perpendicular distance from the coordinate to the matched road, in the units of the coordinate's CRS (typically meters). Set to infinity if no road was matched.

    Examples:
        >>> from mappymatch.constructs.coordinate import Coordinate
        >>> from mappymatch.constructs.road import Road, RoadId
        >>> from mappymatch.constructs.match import Match
        >>> from shapely.geometry import LineString
        >>>
        >>> # Create a successful match
        >>> coord = Coordinate.from_lat_lon(40.7128, -74.0060)
        >>> road = Road(RoadId('1', '2', 0), LineString([(0, 0), (1, 1)]))
        >>> match = Match(road=road, coordinate=coord, distance=5.2)
        >>>
        >>> # Check if matching was successful
        >>> if match.road is not None:
        ...     print(f"Matched to road with distance: {match.distance}m")
        >>>
        >>> # Create a failed match (no road found)
        >>> no_match = Match(road=None, coordinate=coord, distance=float('inf'))
    """

    road: Optional[Road]
    coordinate: Coordinate
    distance: float

    def set_coordinate(self, c: Coordinate):
        """
        Create a new match with a different coordinate.

        This is useful when you need to update the coordinate while preserving the matched
        road and distance information. Since Match is immutable (NamedTuple), this returns
        a new Match instance.

        Args:
            c: The new coordinate to associate with this match

        Returns:
            A new Match instance with the updated coordinate, preserving the road and distance
        """
        return self._replace(coordinate=c)

    def to_flat_dict(self) -> dict:
        """
        Convert this match to a flat dictionary suitable for DataFrame creation.

        This method creates a dictionary representation of the match, unpacking road metadata
        if a road was matched. If no road was found, only the coordinate_id is included.

        Returns:
            A flat dictionary containing:
            - coordinate_id: The ID of the matched coordinate
            - distance_to_road: The distance to the matched road (only if road is not None)
            - All fields from road.to_flat_dict() (only if road is not None)

        Examples:
            >>> # Successful match
            >>> match = Match(road=some_road, coordinate=coord, distance=5.2)
            >>> data = match.to_flat_dict()
            >>> print(data['distance_to_road'])  # 5.2
            >>>
            >>> # Failed match
            >>> no_match = Match(road=None, coordinate=coord, distance=float('inf'))
            >>> data = no_match.to_flat_dict()
            >>> print(data)  # {'coordinate_id': ..., 'road_id': None}
        """
        out = {"coordinate_id": self.coordinate.coordinate_id}

        if self.road is None:
            out["road_id"] = None
            return out
        else:
            out["distance_to_road"] = self.distance
            road_dict = self.road.to_flat_dict()
            out.update(road_dict)
            return out
