from typing import Optional

import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from mappymatch.matchers.lcss.constructs import TrajectorySegment
from mappymatch.utils.crs import LATLON_CRS


def plot_trajectory_segment(
    segment: TrajectorySegment,
    m: Optional[folium.Map] = None,
    trace_point_color: str = "black",
    path_line_color: str = "red",
    path_line_weight: int = 10,
    path_line_opacity: float = 0.8,
    show_matches: bool = True,
    match_point_color: str = "blue",
    show_cutting_points: bool = True,
    cutting_point_color: str = "orange",
):
    """
    Plot a TrajectorySegment showing the trace, path, matches, and cutting points.

    Args:
        segment: The TrajectorySegment to plot.
        m: The folium map to plot on. If None, a new map will be created.
        trace_point_color: The color for trace points.
        trace_line_color: The color for the trace line.
        path_line_color: The color for the path line.
        path_line_weight: The weight of the path line.
        path_line_opacity: The opacity of the path line.
        show_matches: Whether to show matched points.
        match_point_color: The color for matched points.
        show_cutting_points: Whether to show cutting points.
        cutting_point_color: The color for cutting points.

    Returns:
        A folium map with the trajectory segment plotted.
    """
    trace = segment.trace
    path = segment.path
    matches = segment.matches
    cutting_points = segment.cutting_points

    original_crs = trace.crs

    if trace.crs != LATLON_CRS:
        trace = trace.to_crs(LATLON_CRS)

    # Create map if not provided
    if m is None:
        mid_coord = trace.coords[int(len(trace) / 2)]
        m = folium.Map(location=[mid_coord.y, mid_coord.x], zoom_start=13)

    # Plot trace points
    for i, c in enumerate(trace.coords):
        folium.Circle(
            location=(c.y, c.x),
            radius=5,
            color=trace_point_color,
            tooltip=f"Trace Point {i}",
            fill=True,
            fill_opacity=0.8,
            fill_color=trace_point_color,
        ).add_to(m)

    # Plot path (roads) if available
    if path:
        road_df = pd.DataFrame([{"road_id": r.road_id, "geom": r.geom} for r in path])
        road_gdf = gpd.GeoDataFrame(
            road_df, geometry=road_df.geom, crs=original_crs
        ).drop(columns=["geom"])
        road_gdf = road_gdf.to_crs(LATLON_CRS)

        for road in road_gdf.itertuples():
            folium.PolyLine(
                [(lat, lon) for lon, lat in road.geometry.coords],
                color=path_line_color,
                tooltip=f"Road ID: {road.road_id}",
                weight=path_line_weight,
                opacity=path_line_opacity,
            ).add_to(m)

    # Plot matches if requested
    if show_matches and matches:
        for i, match in enumerate(matches):
            if match.road:
                coord = match.coordinate
                if original_crs != LATLON_CRS:
                    # Convert coordinate to lat/lon
                    coord_gdf = gpd.GeoDataFrame(
                        [{"geom": Point(coord.x, coord.y)}],
                        geometry="geom",
                        crs=original_crs,
                    )
                    coord_gdf = coord_gdf.to_crs(LATLON_CRS)
                    coord_point = coord_gdf.iloc[0].geometry
                    y, x = coord_point.y, coord_point.x
                else:
                    y, x = coord.y, coord.x

                folium.CircleMarker(
                    location=(y, x),
                    radius=7,
                    color=match_point_color,
                    tooltip=f"Match {i}<br>Road ID: {match.road.road_id}<br>Distance: {match.distance:.2f}m",
                    fill=True,
                    fill_opacity=0.6,
                    fill_color=match_point_color,
                ).add_to(m)

    # Plot cutting points if requested
    if show_cutting_points and cutting_points:
        for cp in cutting_points:
            coord = trace.coords[cp.trace_index]
            folium.CircleMarker(
                location=(coord.y, coord.x),
                radius=10,
                color=cutting_point_color,
                tooltip=f"Cutting Point at index {cp.trace_index}",
                fill=True,
                fill_opacity=0.9,
                fill_color=cutting_point_color,
            ).add_to(m)

    # Add segment score to map if available
    if segment.score > 0:
        folium.Marker(
            location=(trace.coords[0].y, trace.coords[0].x),
            popup=f"Segment Score: {segment.score:.4f}",
            icon=folium.Icon(color="lightgray", icon="info-sign"),
        ).add_to(m)

    return m
