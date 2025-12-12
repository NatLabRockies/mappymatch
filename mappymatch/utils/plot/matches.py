from typing import List, Union

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point

from mappymatch.constructs.match import Match
from mappymatch.matchers.matcher_interface import MatchResult
from mappymatch.utils.crs import LATLON_CRS, XY_CRS


def plot_matches(matches: Union[MatchResult, List[Match]], crs=XY_CRS):
    """
    Plots a trace and the relevant matches on a folium map.

    Args:
    matches: A list of matches or a MatchResult.
    crs: what crs to plot in. Defaults to XY_CRS.

    Returns:
        A folium map with trace and matches plotted.
    """
    if isinstance(matches, MatchResult):
        matches = matches.matches

    def _match_to_road(m):
        """Private function."""
        d = {"road_id": m.road.road_id, "geom": m.road.geom}
        return d

    def _match_to_coord(m):
        """Private function."""
        d = {
            "road_id": m.road.road_id,
            "geom": Point(m.coordinate.x, m.coordinate.y),
            "distance": m.distance,
        }

        return d

    road_df = pd.DataFrame([_match_to_road(m) for m in matches if m.road])
    road_df = road_df.loc[road_df.road_id.shift() != road_df.road_id]
    road_gdf = gpd.GeoDataFrame(road_df, geometry=road_df.geom, crs=crs).drop(
        columns=["geom"]
    )
    road_gdf = road_gdf.to_crs(LATLON_CRS)

    coord_df = pd.DataFrame([_match_to_coord(m) for m in matches if m.road])

    coord_gdf = gpd.GeoDataFrame(coord_df, geometry=coord_df.geom, crs=crs).drop(
        columns=["geom"]
    )
    coord_gdf = coord_gdf.to_crs(LATLON_CRS)

    mid_i = int(len(coord_gdf) / 2)
    mid_coord = coord_gdf.iloc[mid_i].geometry

    fmap = folium.Map(location=[mid_coord.y, mid_coord.x], zoom_start=11)

    for coord in coord_gdf.itertuples():
        folium.Circle(
            location=(coord.geometry.y, coord.geometry.x),
            radius=5,
            tooltip=f"road_id: {coord.road_id}\ndistance: {coord.distance}",
        ).add_to(fmap)

    for road in road_gdf.itertuples():
        folium.PolyLine(
            [(lat, lon) for lon, lat in road.geometry.coords],
            color="red",
            tooltip=road.road_id,
        ).add_to(fmap)

    return fmap


def plot_match_distances(matches: MatchResult):
    """
    Plot the points deviance from known roads with matplotlib.

    Args:
        matches (MatchResult): The coordinates of guessed points in the area in the form of a MatchResult object.
    """

    y = [
        m.distance for m in matches.matches
    ]  # y contains distances to the expected line for all of the matches which will be plotted on the y-axis.
    x = [
        i for i in range(0, len(y))
    ]  # x contains placeholder values for every y value (distance measurement) along the x-axis.

    plt.figure(figsize=(15, 7))
    plt.autoscale(enable=True)
    plt.scatter(x, y)
    plt.title("Distance To Nearest Road")
    plt.ylabel("Meters")
    plt.xlabel("Point Along The Path")
    plt.show()
