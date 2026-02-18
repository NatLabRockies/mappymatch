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
    Enumeration of road network types supported by OSMnx for downloading from OpenStreetMap.

    These network types determine which roads are included when downloading OSM data.
    Each type corresponds to a predefined filter in OSMnx for different use cases and
    transportation modes.

    Values:
        ALL_PRIVATE: All road types including private roads
        ALL: All public road types
        BIKE: Roads suitable for cycling
        DRIVE: Roads suitable for driving (excludes footpaths, bike paths, etc.)
        DRIVE_SERVICE: Driving roads including service roads (parking aisles, driveways, etc.)
        WALK: Roads suitable for walking (includes footpaths, pedestrian areas, etc.)

    Examples:
        >>> from mappymatch.maps.nx.readers.osm_readers import NetworkType
        >>> # Download a driving network
        >>> road_map = NxMap.from_geofence(geofence, network_type=NetworkType.DRIVE)
        >>> # Download a walking network
        >>> walk_map = NxMap.from_geofence(geofence, network_type=NetworkType.WALK)
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
    filter_to_largest_component: bool = True,
) -> nx.MultiDiGraph:
    """
    Download and process a road network from OpenStreetMap using OSMnx.

    This function:
    1. Downloads OSM data within the geofence using OSMnx
    2. Optionally projects to Web Mercator for accurate distances
    3. Adds speed and travel time estimates
    4. Converts distances to kilometers
    5. Optionally filters to the largest connected component
    6. Compresses the graph to remove unnecessary data

    Args:
        geofence: The geographic boundary for downloading. Must be in EPSG:4326 (lat/lon).
        network_type: The type of network to download (DRIVE, WALK, BIKE, etc.)
        xy: If True, project to Web Mercator (EPSG:3857) for accurate metric distances. If False, keep in lat/lon. Default is True.
        custom_filter: A custom OSMnx filter string for advanced queries (e.g., '["highway"~"motorway|trunk|primary"]'). If specified, overrides network_type.
        additional_metadata_keys: Set of OSM tag keys to preserve in road metadata (e.g., {'maxspeed', 'lanes', 'surface'}).
        filter_to_largest_component: If True (default), keep only the largest strongly connected component to ensure all parts of the network are routable to each other. If False, keep disconnected components (may cause routing failures).

    Returns:
        A processed NetworkX MultiDiGraph ready for use with NxMap

    Raises:
        MapException: If OSMnx is not installed

    Examples:
        >>> from mappymatch.maps.nx.readers.osm_readers import nx_graph_from_osmnx, NetworkType
        >>>
        >>> # Download a driving network
        >>> graph = nx_graph_from_osmnx(
        ...     geofence=my_geofence,
        ...     network_type=NetworkType.DRIVE,
        ...     xy=True
        ... )
        >>>
        >>> # Download with custom filter and metadata
        >>> graph = nx_graph_from_osmnx(
        ...     geofence=my_geofence,
        ...     network_type=NetworkType.DRIVE,
        ...     custom_filter='["highway"~"motorway|primary|secondary"]',
        ...     additional_metadata_keys={'maxspeed', 'lanes', 'name'}
        ... )
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
    )
    return parse_osmnx_graph(
        raw_graph,
        network_type,
        xy=xy,
        additional_metadata_keys=additional_metadata_keys,
        filter_to_largest_component=filter_to_largest_component,
    )


def parse_osmnx_graph(
    graph: nx.MultiDiGraph,
    network_type: NetworkType,
    xy: bool = True,
    additional_metadata_keys: Optional[set] = None,
    filter_to_largest_component: bool = True,
) -> nx.MultiDiGraph:
    """
    Process and clean a raw OSMnx graph for use with mappymatch.

    This function takes a graph downloaded from OSMnx and processes it by:
    - Projecting to a metric coordinate system (optional)
    - Computing edge speeds and travel times
    - Adding distance in kilometers
    - Filtering to the largest connected component (optional)
    - Creating geometries for edges that lack them
    - Compressing by removing unnecessary data

    Args:
        graph: A raw NetworkX MultiDiGraph from OSMnx
        network_type: The type of network (used for metadata)
        xy: If True, project to Web Mercator (EPSG:3857). If False, keep in lat/lon.
        additional_metadata_keys: Set of OSM tag keys to preserve in road metadata
        filter_to_largest_component: If True, keep only the largest strongly connected component. If False, keep all components.

    Returns:
        A cleaned and processed NetworkX MultiDiGraph ready for NxMap

    Raises:
        MapException: If OSMnx is not installed or if the network has no connected components
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
    if filter_to_largest_component:
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
    Remove unnecessary data from a NetworkX graph while preserving essential attributes.

    This function reduces memory usage and file size by:
    - Moving specified OSM tags to a metadata dictionary
    - Removing edge attributes that aren't needed for routing or matching
    - Removing all node attributes (only the node IDs are needed)

    Essential attributes for edges are preserved:
    - geometry: The LineString geometry of the road
    - kilometers: Distance in kilometers for routing
    - travel_time: Estimated travel time for routing
    - metadata: Dictionary of additional OSM tags

    Args:
        g: The NetworkX MultiDiGraph to compress
        additional_metadata_keys: Set of OSM tag keys to preserve in the metadata dictionary. Default preserved keys are 'osmid' and 'name'. Any additional keys specified here will also be moved to metadata.

    Returns:
        The compressed NetworkX MultiDiGraph (modified in-place and returned)

    Examples:
        >>> # Compress with default metadata
        >>> compressed = compress(graph)
        >>>
        >>> # Preserve additional OSM tags
        >>> compressed = compress(
        ...     graph,
        ...     additional_metadata_keys={'maxspeed', 'lanes', 'surface', 'highway'}
        ... )
        >>>
        >>> # Access metadata
        >>> edge_data = graph.edges[('123', '456', 0)]
        >>> print(edge_data['metadata']['name'])  # Street name
        >>> print(edge_data['metadata']['maxspeed'])  # Speed limit if available
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
