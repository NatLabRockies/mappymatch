from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Callable, List, Optional, Union

from mappymatch.constructs.coordinate import Coordinate
from mappymatch.constructs.road import Road, RoadId

DEFAULT_DISTANCE_WEIGHT = "kilometers"
DEFAULT_TIME_WEIGHT = "minutes"


class MapInterface(metaclass=ABCMeta):
    """
    Abstract base class defining the interface for road network representations.

    All map implementations in mappymatch implement this interface, providing a consistent
    API for road network operations including spatial queries, routing, and road lookups.
    This allows different backend implementations (NetworkX, igraph, etc.) to be used
    interchangeably.

    Subclasses must implement methods for:
    - Finding nearest roads to coordinates
    - Computing shortest paths between points
    - Looking up roads by ID
    - Providing lists of all roads

    """

    @property
    @abstractmethod
    def distance_weight(self) -> str:
        """
        Get the name of the edge attribute used for distance-based routing.

        This property identifies which edge attribute should be used when computing
        shortest paths based on physical distance (as opposed to time or other metrics).

        Returns:
            The name of the distance weight attribute (e.g., 'kilometers', 'miles', 'length')

        """
        return DEFAULT_DISTANCE_WEIGHT

    @property
    @abstractmethod
    def time_weight(self) -> str:
        """
        Get the name of the edge attribute used for time-based routing.

        This property identifies which edge attribute should be used when computing
        fastest paths based on travel time (as opposed to distance or other metrics).

        Returns:
            The name of the time weight attribute (e.g., 'minutes', 'seconds', 'travel_time')

        """
        return DEFAULT_TIME_WEIGHT

    @property
    @abstractmethod
    def roads(self) -> List[Road]:
        """
        Get a list of all the roads in the map

        Returns:
            A list of all the roads in the map
        """

    @abstractmethod
    def road_by_id(self, road_id: RoadId) -> Optional[Road]:
        """
        Get a road by its id

        Args:
            road_id: The id of the road to get

        Returns:
            The road with the given id or None if it does not exist
        """

    @abstractmethod
    def nearest_road(
        self,
        coord: Coordinate,
    ) -> Road:
        """
        Find the road segment nearest to a given coordinate.

        This method performs a spatial search to identify the closest road segment
        to the specified coordinate. It typically uses a spatial index (like an R-tree)
        for efficient querying.

        Args:
            coord: The coordinate to search from

        Returns:
            The Road object that is spatially nearest to the coordinate
        """

    @abstractmethod
    def shortest_path(
        self,
        origin: Coordinate,
        destination: Coordinate,
        weight: Optional[Union[str, Callable]] = None,
    ) -> List[Road]:
        """
        Compute the shortest path through the road network between two coordinates.

        This method finds the optimal route from origin to destination using the
        road network graph. The "shortest" path can be based on distance, time,
        or any custom weight function.

        Args:
            origin: The starting coordinate. The path begins from the nearest road to this point.
            destination: The ending coordinate. The path ends at the nearest road to this point.
            weight: The edge attribute or function to minimize. Can be a string attribute name (e.g., 'kilometers', 'minutes'), a callable that takes edge data and returns a weight, or None to use the default distance weight.

        Returns:
            A list of Road objects representing the path from origin to destination,
            ordered sequentially. Returns empty list if no path exists (disconnected components).
        """
