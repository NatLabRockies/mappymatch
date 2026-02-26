"""Coordinate Reference System (CRS) constants used throughout mappymatch.

This module defines the standard CRS objects used for geographic transformations:
- LATLON_CRS: WGS84 geographic coordinates (EPSG:4326)
- XY_CRS: Web Mercator projected coordinates (EPSG:3857)
"""

from pyproj import CRS

# WGS84 latitude/longitude coordinate system (EPSG:4326)
# Standard GPS coordinates in decimal degrees
# Range: latitude [-90, 90], longitude [-180, 180]
LATLON_CRS = CRS(4326)

# Web Mercator projected coordinate system (EPSG:3857)
# Used for web mapping and accurate distance calculations in meters
# Coordinates are in meters (easting, northing)
XY_CRS = CRS(3857)
