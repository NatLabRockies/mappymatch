from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Union

from shapely.geometry import LineString


class RoadId(NamedTuple):
    start: Optional[Union[int, str]]
    end: Optional[Union[int, str]]
    key: Optional[Union[int, str]]

    def to_string(self) -> str:
        return f"{self.start},{self.end},{self.key}"

    def to_json(self) -> Dict[str, Any]:
        return self._asdict()

    @classmethod
    def from_string(cls, s: str) -> RoadId:
        start, end, key = s.split(",")
        return cls(start, end, key)

    @classmethod
    def from_json(cls, json: Dict[str, Any]) -> RoadId:
        return cls(**json)


class Road(NamedTuple):
    """
    Represents a road segment in the road network that can be matched to GPS trajectories.

    A Road is an immutable object representing a directional edge in a road network graph.
    Roads have a unique identifier (composed of start/end junctions and a key), a geometry
    (typically a LineString), and optional metadata for storing additional attributes like
    speed limits, road names, etc.

    Attributes:
        road_id: A RoadId tuple uniquely identifying this road segment (start, end, key)
        geom: The Shapely LineString geometry representing the road's path
        metadata: An optional dictionary for storing additional road attributes such as speed limits, road names, surface type, etc.
    """

    road_id: RoadId

    geom: LineString
    metadata: Optional[dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the road to a dictionary representation.

        This creates a dictionary with all road attributes, extracting the origin and
        destination junction IDs from the road_id for convenience.

        Returns:
            A dictionary containing the road's attributes with separate keys for
            origin_junction_id, destination_junction_id, and road_key
        """
        d = self._asdict()
        d["origin_junction_id"] = self.road_id.start
        d["destination_junction_id"] = self.road_id.end
        d["road_key"] = self.road_id.key

        return d

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert the road to a flat dictionary with metadata unpacked.

        This method creates a single-level dictionary by unpacking the metadata dictionary
        and merging it with the road's other attributes. This is useful for creating
        DataFrames or exporting to formats that don't support nested structures.

        Returns:
            A flat dictionary with all road attributes and metadata fields at the top level.
            The 'metadata' key itself is removed.

        """
        if self.metadata is None:
            return self.to_dict()
        else:
            d = {**self.to_dict(), **self.metadata}
            del d["metadata"]
            return d
