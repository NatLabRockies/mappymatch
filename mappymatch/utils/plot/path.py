from typing import List, Optional

import folium
import geopandas as gpd
import pandas as pd
from pyproj import CRS

from mappymatch.constructs.road import Road
from mappymatch.utils.crs import LATLON_CRS


def plot_path(
    path: List[Road],
    crs: CRS,
    m: Optional[folium.Map] = None,
    line_color="red",
    line_weight=10,
    line_opacity=0.8,
):
    """
    Plot a list of roads.

    Args:
        path: The path to plot.
        crs: The crs of the path.
        m: The folium map to add to.
        line_color: The color of the line.
        line_weight: The weight of the line.
        line_opacity: The opacity of the line.
    """
    road_df = pd.DataFrame([{"geom": r.geom} for r in path])
    road_gdf = gpd.GeoDataFrame(road_df, geometry=road_df.geom, crs=crs)
    road_gdf = road_gdf.to_crs(LATLON_CRS)

    if m is None:
        mid_i = int(len(road_gdf) / 2)
        mid_coord = road_gdf.iloc[mid_i].geometry.coords[0]

        m = folium.Map(location=[mid_coord[1], mid_coord[0]], zoom_start=11)

    for i, road in enumerate(road_gdf.itertuples()):
        folium.PolyLine(
            [(lat, lon) for lon, lat in road.geometry.coords],
            color=line_color,
            tooltip=i,
            weight=line_weight,
            opacity=line_opacity,
        ).add_to(m)

    return m
