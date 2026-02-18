from typing import Tuple

from pyproj import Transformer

from mappymatch.constructs.coordinate import Coordinate
from mappymatch.utils.crs import LATLON_CRS, XY_CRS


def xy_to_latlon(x: float, y: float) -> Tuple[float, float]:
    """
    Transform Web Mercator (EPSG:3857) coordinates to WGS84 latitude/longitude.

    This function converts from the projected coordinate system commonly used for
    web mapping (meters, sometimes called "xy" coordinates) to standard geographic
    coordinates (degrees).

    Args:
        x: The x-coordinate (easting) in Web Mercator projection (meters)
        y: The y-coordinate (northing) in Web Mercator projection (meters)

    Returns:
        A tuple of (latitude, longitude) in decimal degrees (WGS84/EPSG:4326)

    Examples:
        >>> # New York City in Web Mercator
        >>> x, y = -8238310.4, 4970241.3
        >>> lat, lon = xy_to_latlon(x, y)
        >>> print(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
        Lat: 40.7128, Lon: -74.0060
    """
    transformer = Transformer.from_crs(XY_CRS, LATLON_CRS)
    lat, lon = transformer.transform(x, y)

    return lat, lon


def latlon_to_xy(lat: float, lon: float) -> Tuple[float, float]:
    """
    Transform WGS84 latitude/longitude to Web Mercator (EPSG:3857) coordinates.

    This function converts from standard geographic coordinates (degrees) to the
    projected coordinate system commonly used for web mapping and distance calculations
    (meters, sometimes called "xy" coordinates).

    Args:
        lat: The latitude in decimal degrees (range: -90 to 90)
        lon: The longitude in decimal degrees (range: -180 to 180)

    Returns:
        A tuple of (x, y) in Web Mercator projection meters (EPSG:3857)

    Examples:
        >>> # New York City coordinates
        >>> lat, lon = 40.7128, -74.0060
        >>> x, y = latlon_to_xy(lat, lon)
        >>> print(f"X: {x:.1f}m, Y: {y:.1f}m")
        X: -8238310.4m, Y: 4970241.3m
    """
    transformer = Transformer.from_crs(LATLON_CRS, XY_CRS)
    x, y = transformer.transform(lat, lon)

    return x, y


def coord_to_coord_dist(a: Coordinate, b: Coordinate) -> float:
    """
    Calculate the Euclidean distance between two coordinates.

    The distance is computed using the geometries' coordinate reference system.
    For accurate distance measurements, coordinates should be in a projected CRS
    (like EPSG:3857) rather than lat/lon (EPSG:4326).

    Args:
        a: The first coordinate
        b: The second coordinate. Must be in the same CRS as coordinate a.

    Returns:
        The Euclidean distance in the units of the coordinates' CRS (typically meters
        if using a projected CRS like EPSG:3857)

    Note:
        For coordinates in lat/lon (EPSG:4326), this computes angular distance in degrees,
        not actual ground distance. Convert to a projected CRS first for accurate distances.

    Examples:
        >>> from mappymatch.constructs.coordinate import Coordinate
        >>> # Create coordinates in Web Mercator (meters)
        >>> coord1 = Coordinate.from_lat_lon(40.7128, -74.0060).to_crs('EPSG:3857')
        >>> coord2 = Coordinate.from_lat_lon(40.7589, -73.9851).to_crs('EPSG:3857')
        >>> distance = coord_to_coord_dist(coord1, coord2)
        >>> print(f"Distance: {distance:.1f} meters")
    """
    dist = a.geom.distance(b.geom)

    return dist
