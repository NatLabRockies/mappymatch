from typing import Optional

import folium

from mappymatch.constructs.trace import Trace
from mappymatch.utils.crs import LATLON_CRS


def plot_trace(
    trace: Trace,
    m: Optional[folium.Map] = None,
    point_color: str = "black",
    line_color: Optional[str] = "green",
):
    """
    Plot a trace.

    Args:
        trace: The trace.
        m: the folium map to plot on
        point_color: The color the points will be plotted in.
        line_color: The color for lines. If None, no lines will be plotted.

    Returns:
        An updated folium map with a plot of trace.
    """

    if not trace.crs == LATLON_CRS:
        trace = trace.to_crs(LATLON_CRS)

    if not m:
        mid_coord = trace.coords[int(len(trace) / 2)]
        m = folium.Map(location=[mid_coord.y, mid_coord.x], zoom_start=11)

    for i, c in enumerate(trace.coords):
        folium.Circle(
            location=(c.y, c.x),
            radius=5,
            color=point_color,
            tooltip=str(i),
            fill=True,
            fill_opacity=0.8,
            fill_color=point_color,
        ).add_to(m)

    if line_color is not None:
        folium.PolyLine([(p.y, p.x) for p in trace.coords], color=line_color).add_to(m)

    return m
