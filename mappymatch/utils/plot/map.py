from typing import Optional

import folium
import geopandas as gpd
import pandas as pd

from mappymatch.constructs.road import RoadId
from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.utils.crs import LATLON_CRS


def plot_map(tmap: NxMap, m: Optional[folium.Map] = None, highlight: bool = False):
    """
    Plot the roads on an NxMap.

    Args:
        tmap: The Nxmap to plot.
        m: the folium map to add to
        highlight: Whether to enable hover highlighting and popups (default: False)

    Returns:
        The folium map with the roads plotted.
    """

    # TODO make this generic to all map types, not just NxMap
    roads = list(tmap.g.edges(data=True, keys=True))
    road_data = []
    for u, v, key, data in roads:
        road_id = RoadId(start=u, end=v, key=key)
        data_copy = data.copy()
        data_copy["road_id"] = road_id
        road_data.append(data_copy)

    road_df = pd.DataFrame(road_data)
    gdf = gpd.GeoDataFrame(road_df, geometry=road_df[tmap._geom_key], crs=tmap.crs)
    if gdf.crs != LATLON_CRS:
        gdf = gdf.to_crs(LATLON_CRS)

    if not m:
        c = gdf.iloc[int(len(gdf) / 2)].geometry.centroid.coords[0]
        m = folium.Map(location=[c[1], c[0]], zoom_start=11)

    # Convert road_id to string for GeoJSON compatibility
    gdf["road_id_str"] = gdf["road_id"].astype(str)

    # Create GeoJson layer with optional popup and highlighting
    if highlight:
        popup = folium.GeoJsonPopup(fields=["road_id_str"])
        tooltip = folium.GeoJsonTooltip(fields=["road_id_str"])
        folium.GeoJson(
            gdf.to_json(),
            style_function=lambda x: {
                "color": "red",
                "weight": 3,
                "opacity": 0.7,
            },
            highlight_function=lambda x: {
                "color": "yellow",
                "weight": 6,
                "opacity": 1.0,
            },
            popup=popup,
            tooltip=tooltip,
            popup_keep_highlighted=True,
        ).add_to(m)
    else:
        folium.GeoJson(
            gdf.to_json(),
            style_function=lambda x: {
                "color": "red",
                "weight": 3,
                "opacity": 0.7,
            },
        ).add_to(m)

    return m
