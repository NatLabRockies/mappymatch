from __future__ import annotations

import math
from typing import Any, NamedTuple

from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError
from shapely.geometry import Point

from mappymatch.utils.crs import LATLON_CRS


class Coordinate(NamedTuple):
    """
    Represents a single geographic coordinate point with a coordinate reference system (CRS).

    A Coordinate is an immutable object that combines a spatial point geometry with its
    coordinate reference system, allowing for accurate coordinate transformations between
    different projection systems.

    Attributes:
        coordinate_id: The unique identifier for this coordinate (can be any hashable type)
        geom: The Shapely Point geometry representing the spatial location
        crs: The pyproj CRS (Coordinate Reference System) defining the coordinate space
        x: The x-coordinate value (longitude in lat/lon systems, easting in projected systems)
        y: The y-coordinate value (latitude in lat/lon systems, northing in projected systems)

    Examples:
        >>> from mappymatch.constructs.coordinate import Coordinate
        >>> # Create a coordinate from latitude and longitude
        >>> coord = Coordinate.from_lat_lon(40.7128, -74.0060)
        >>> print(coord.x, coord.y)
        -74.0060 40.7128

        >>> # Transform to a different CRS (Web Mercator)
        >>> web_mercator = coord.to_crs('EPSG:3857')
        >>> print(web_mercator.crs.to_epsg())
        3857
    """

    coordinate_id: Any
    geom: Point
    crs: CRS

    def __repr__(self):
        crs_a = self.crs.to_authority() if self.crs else "Null"
        return f"Coordinate(coordinate_id={self.coordinate_id}, x={self.x}, y={self.y}, crs={crs_a})"

    @classmethod
    def from_lat_lon(cls, lat: float, lon: float) -> Coordinate:
        """
        Create a coordinate from latitude and longitude values in WGS84 (EPSG:4326).

        This is a convenience method for creating coordinates from standard GPS coordinates.
        The resulting coordinate will use the LATLON_CRS (EPSG:4326) coordinate system.

        Args:
            lat: The latitude in decimal degrees (range: -90 to 90)
            lon: The longitude in decimal degrees (range: -180 to 180)

        Returns:
            A new Coordinate instance in EPSG:4326 CRS with no coordinate_id

        Examples:
            >>> # New York City coordinates
            >>> nyc = Coordinate.from_lat_lon(40.7128, -74.0060)
            >>> print(f"Lat: {nyc.y}, Lon: {nyc.x}")
            Lat: 40.7128, Lon: -74.0060
        """
        return cls(coordinate_id=None, geom=Point(lon, lat), crs=LATLON_CRS)

    @property
    def x(self) -> float:
        return self.geom.x

    @property
    def y(self) -> float:
        return self.geom.y

    def to_crs(self, new_crs: Any) -> Coordinate:
        """
        Transform this coordinate to a different coordinate reference system (CRS).

        This method reprojects the coordinate geometry from its current CRS to the target CRS
        using pyproj transformations. If the target CRS is the same as the current CRS,
        the original coordinate is returned unchanged.

        Args:
            new_crs: The target CRS. Can be a pyproj.CRS object, an EPSG code as a string
                (e.g., 'EPSG:4326'), an integer EPSG code, or any CRS format that pyproj.CRS() accepts

        Returns:
            A new Coordinate instance with transformed geometry in the target CRS.
            The coordinate_id is preserved from the original coordinate.

        Raises:
            ValueError: If the new_crs cannot be parsed into a valid CRS, or if the
                transformation results in infinite coordinate values (indicating an invalid transformation)

        Examples:
            >>> # Transform from lat/lon to Web Mercator
            >>> coord = Coordinate.from_lat_lon(40.7128, -74.0060)
            >>> mercator_coord = coord.to_crs('EPSG:3857')
            >>>
            >>> # Transform using EPSG integer code
            >>> utm_coord = coord.to_crs(32618)  # UTM Zone 18N
        """
        # convert the incoming crs to an pyproj.crs.CRS object; this could fail
        try:
            new_crs = CRS(new_crs)
        except ProjError as e:
            raise ValueError(
                f"Could not parse incoming `new_crs` parameter: {new_crs}"
            ) from e

        if new_crs == self.crs:
            return self

        transformer = Transformer.from_crs(self.crs, new_crs)
        new_x, new_y = transformer.transform(self.geom.y, self.geom.x)

        if math.isinf(new_x) or math.isinf(new_y):
            raise ValueError(
                f"Unable to convert {self.crs} ({self.geom.x}, {self.geom.y}) -> {new_crs} ({new_x}, {new_y})"
            )

        return Coordinate(
            coordinate_id=self.coordinate_id,
            geom=Point(new_x, new_y),
            crs=new_crs,
        )
