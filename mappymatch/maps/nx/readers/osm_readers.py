from __future__ import annotations

import logging as log
from enum import Enum
from typing import Optional

import networkx as nx
from shapely.geometry import LineString

from mappymatch.constructs.geofence import Geofence
from mappymatch.utils.crs import XY_CRS
from mappymatch.utils.exceptions import MapException
from mappymatch.utils.keys import DEFAULT_METADATA_KEY

log.basicConfig(level=log.INFO)


METERS_TO_KM = 1 / 1000
DEFAULT_MPH = 30


class NetworkType(Enum):
    """
    Enumerator for Network Types supported by osmnx.
    """

    ALL_PRIVATE = "all_private"
    ALL = "all"
    BIKE = "bike"
    DRIVE = "drive"
    DRIVE_SERVICE = "drive_service"
    WALK = "walk"


def nx_graph_from_osmnx(
    geofence: Geofence,
    network_type: NetworkType,
    xy: bool = True,
    custom_filter: Optional[str] = None,
    additional_metadata_keys: Optional[set] = None,
) -> nx.MultiDiGraph:
    """
    Build a networkx graph from OSM data

    Args:
        geofence: the geofence to clip the graph to
        network_type: the network type to use for the graph
        xy: whether to use xy coordinates or lat/lon
        custom_filter: a custom filter to pass to osmnx
        additional_metadata_keys: additional keys to preserve in metadata

    Returns:
        a networkx graph of the OSM network
    """
    try:
        import osmnx as ox
    except ImportError:
        raise MapException("osmnx is not installed but is required for this map type")
    ox.settings.log_console = False

    raw_graph = ox.graph_from_polygon(
        geofence.geometry,
        network_type=network_type.value,
        custom_filter=custom_filter,
        truncate_by_edge=True,
    )
    return parse_osmnx_graph(
        raw_graph,
        network_type,
        xy=xy,
        additional_metadata_keys=additional_metadata_keys,
    )


def parse_osmnx_graph(
    graph: nx.MultiDiGraph,
    network_type: NetworkType,
    xy: bool = True,
    additional_metadata_keys: Optional[set] = None,
) -> nx.MultiDiGraph:
    """
    Parse the raw osmnx graph into a graph that we can use with our NxMap

    Args:
        geofence: the geofence to clip the graph to
        xy: whether to use xy coordinates or lat/lon
        network_type: the network type to use for the graph
        additional_metadata_keys: additional keys to preserve in metadata

    Returns:
        a cleaned networkx graph of the OSM network
    """
    try:
        import osmnx as ox
    except ImportError:
        raise MapException("osmnx is not installed but is required for this map type")
    ox.settings.log_console = False
    g = graph

    if xy:
        g = ox.project_graph(g, to_crs=XY_CRS)

    g = ox.add_edge_speeds(g)
    g = ox.add_edge_travel_times(g)

    length_meters = nx.get_edge_attributes(g, "length")
    kilometers = {k: v * METERS_TO_KM for k, v in length_meters.items()}
    nx.set_edge_attributes(g, kilometers, "kilometers")

    # this makes sure there are no graph 'dead-ends'
    sg_components = nx.strongly_connected_components(g)

    if not sg_components:
        raise MapException(
            "road network has no strongly connected components and is not routable; "
            "check polygon boundaries."
        )

    g = nx.MultiDiGraph(g.subgraph(max(sg_components, key=len)))

    for u, v, d in g.edges(data=True):
        if "geometry" not in d:
            # we'll build a pseudo-geometry using the x, y data from the nodes
            unode = g.nodes[u]
            vnode = g.nodes[v]
            line = LineString([(unode["x"], unode["y"]), (vnode["x"], vnode["y"])])
            d["geometry"] = line

    g = compress(g, additional_metadata_keys=additional_metadata_keys)

    # TODO: these should all be sourced from the same location
    g.graph["distance_weight"] = "kilometers"
    g.graph["time_weight"] = "travel_time"
    g.graph["geometry_key"] = "geometry"
    g.graph["network_type"] = network_type.value

    return g


def compress(
    g: nx.MultiDiGraph, additional_metadata_keys: Optional[set] = None
) -> nx.MultiDiGraph:
    """
    Remove unnecessary data from the networkx graph while preserving essential attributes

    Args:
        g: the networkx graph to compress
        additional_metadata_keys: additional keys to preserve in metadata

    Returns:
        the compressed networkx graph
    """
    # Define attributes to keep for edges
    edge_keep_keys = {
        "geometry",
        "kilometers",
        "travel_time",
        DEFAULT_METADATA_KEY,
    }

    # Define attributes to move to metadata
    default_metadata_keys = {"osmid", "name"}
    if additional_metadata_keys:
        default_metadata_keys.update(additional_metadata_keys)

    # Define attributes to keep for nodes (only what we need)
    node_keep_keys: set[str] = set()

    # Process edges
    for _, _, d in g.edges(data=True):
        # Initialize metadata dict if needed
        if DEFAULT_METADATA_KEY not in d:
            d[DEFAULT_METADATA_KEY] = {}

        # Move specified keys to metadata
        for key in default_metadata_keys:
            if key in d:
                d[DEFAULT_METADATA_KEY][key] = d[key]

        # Delete all keys not in keep list
        keys_to_remove = [k for k in list(d.keys()) if k not in edge_keep_keys]
        for key in keys_to_remove:
            del d[key]

    # Process nodes
    for _, d in g.nodes(data=True):
        keys_to_remove = [k for k in list(d.keys()) if k not in node_keep_keys]
        for key in keys_to_remove:
            del d[key]

    return g
