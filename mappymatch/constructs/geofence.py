from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from geopandas import read_file
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import transform

from mappymatch.constructs.trace import Trace
from mappymatch.utils.crs import LATLON_CRS


class Geofence:
    """
    A geographic boundary polygon with an associated coordinate reference system (CRS).

    A Geofence defines a spatial region used to constrain map data or filter geographic queries.
    It's commonly used to download only the relevant portion of a road network for map matching
    by creating a buffer around a GPS trajectory.

    Args:
        crs: The coordinate reference system of the geofence geometry
        geometry: A Shapely Polygon defining the geographic boundary

    Attributes:
        crs: The CRS of the geofence
        geometry: The Polygon geometry representing the bounded area

    Examples:
        >>> from mappymatch.constructs.geofence import Geofence
        >>>
        >>> # Create a geofence from a trace to download relevant map data
        >>> trace = Trace.from_gpx('route.gpx')
        >>> geofence = Geofence.from_trace(trace, padding=1000)  # 1km buffer
        >>>
        >>> # Load a geofence from a GeoJSON file
        >>> geofence = Geofence.from_geojson('city_boundary.geojson')
        >>>
        >>> # Export to GeoJSON
        >>> geojson_str = geofence.to_geojson()
    """

    def __init__(self, crs: CRS, geometry: Polygon):
        self.crs = crs
        self.geometry = geometry

    @classmethod
    def from_geojson(cls, file: Union[Path, str]) -> Geofence:
        """
        Create a geofence from a GeoJSON file containing a polygon.

        The GeoJSON file must contain exactly one polygon feature and must include
        CRS information. This is typically used to define study areas or regions
        for downloading map data.

        Args:
            file: Path to the GeoJSON file (as string or Path object)

        Returns:
            A new Geofence instance

        Raises:
            TypeError: If the file contains multiple polygons or lacks CRS information

        Examples:
            >>> # Load a boundary polygon
            >>> geofence = Geofence.from_geojson('study_area.geojson')
            >>> print(geofence.crs)  # EPSG:4326
        """
        filepath = Path(file)
        frame = read_file(filepath)

        if len(frame) > 1:
            raise TypeError(
                "found multiple polygons in the input; please only provide one"
            )
        elif frame.crs is None:
            raise TypeError(
                "no crs information found in the file; please make sure file has a crs"
            )

        polygon = frame.iloc[0].geometry

        return Geofence(crs=frame.crs, geometry=polygon)

    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        padding: float = 1e3,
        crs: CRS = LATLON_CRS,
        buffer_res: int = 2,
    ) -> Geofence:
        """
        Create a geofence by buffering around a GPS trace.

        This method creates a polygonal boundary around a trace by converting the trace
        to a LineString, creating a buffer zone around the line, and transforming to the
        specified CRS.

        This is particularly useful for downloading map data that covers a GPS trajectory,
        ensuring you get all relevant roads while minimizing unnecessary data.

        Args:
            trace: The GPS trace to create a boundary around
            padding: The buffer distance in meters around the trace. Default is 1000m (1km).
            crs: The target coordinate reference system for the geofence. Default is WGS84 (EPSG:4326).
            buffer_res: The resolution of the buffer polygon (number of segments per quadrant). Lower values create simpler polygons. Default is 2.

        Returns:
            A new Geofence encompassing the trace with the specified padding

        Examples:
            >>> # Create a 500m buffer around a trace for map download
            >>> trace = Trace.from_csv('morning_commute.csv')
            >>> geofence = Geofence.from_trace(trace, padding=500)
            >>>
            >>> # Create a larger 2km buffer with higher resolution
            >>> geofence = Geofence.from_trace(trace, padding=2000, buffer_res=8)
        """

        trace_line_string = LineString([c.geom for c in trace.coords])

        # Add buffer to LineString.
        polygon = trace_line_string.buffer(padding, buffer_res)

        if trace.crs != crs:
            project = Transformer.from_crs(trace.crs, crs, always_xy=True).transform
            polygon = transform(project, polygon)
            return Geofence(crs=crs, geometry=polygon)

        return Geofence(crs=trace.crs, geometry=polygon)

    def to_geojson(self) -> str:
        """
        Convert the geofence to a GeoJSON string.

        The geofence is automatically transformed to WGS84 (EPSG:4326) if it's in a different CRS,
        since GeoJSON uses lat/lon coordinates by convention.

        Returns:
            A GeoJSON string representation of the geofence polygon

        Examples:
            >>> geofence = Geofence.from_trace(trace, padding=1000)
            >>> geojson_str = geofence.to_geojson()
            >>> # Write to file
            >>> with open('boundary.geojson', 'w') as f:
            ...     f.write(geojson_str)
        """
        if self.crs != LATLON_CRS:
            transformer = Transformer.from_crs(self.crs, LATLON_CRS)
            geometry: Polygon = transformer.transform(self.geometry)  # type: ignore
        else:
            geometry = self.geometry

        return json.dumps(mapping(geometry))
