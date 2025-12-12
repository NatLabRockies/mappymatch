from typing import Optional

import folium

from mappymatch.constructs.geofence import Geofence
from mappymatch.utils.crs import LATLON_CRS


def plot_geofence(geofence: Geofence, m: Optional[folium.Map] = None):
    """
    Plot geofence.

    Args:
        geofence: The geofence to plot
        m: the folium map to plot on

    Returns:
        The updated folium map with the geofence.
    """
    if not geofence.crs == LATLON_CRS:
        raise NotImplementedError("can currently only plot a geofence with lat lon crs")

    if not m:
        c = geofence.geometry.centroid.coords[0]
        m = folium.Map(location=[c[1], c[0]], zoom_start=11)

    folium.GeoJson(geofence.geometry).add_to(m)

    return m
