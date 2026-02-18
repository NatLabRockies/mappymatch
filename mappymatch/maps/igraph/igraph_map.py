from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

import igraph as ig
import networkx as nx
from shapely.geometry import Point
from shapely.strtree import STRtree

from mappymatch.constructs.coordinate import Coordinate
from mappymatch.constructs.geofence import Geofence
from mappymatch.constructs.road import Road, RoadId
from mappymatch.maps.map_interface import (
    DEFAULT_DISTANCE_WEIGHT,
    DEFAULT_TIME_WEIGHT,
    MapInterface,
)
from mappymatch.maps.nx.readers.osm_readers import (
    NetworkType,
    nx_graph_from_osmnx,
)
from mappymatch.utils.crs import CRS, LATLON_CRS

DEFAULT_GEOMETRY_KEY = "geometry"
DEFAULT_METADATA_KEY = "metadata"
DEFAULT_CRS_KEY = "crs"
DEFAULT_NODE_ID_NAME = "node_id"
DEFAULT_EDGE_ID_NAME = "edge_id"


class IGraphMap(MapInterface):
    """
    A road network map implementation using igraph for high-performance graph operations.

    IGraphMap wraps an igraph.Graph to represent a road network, providing potentially
    faster performance than NetworkX for very large networks. It uses an R-tree spatial
    index for efficient nearest-neighbor searches and supports both distance and time-based routing.

    The underlying igraph must have:
    - A pyproj CRS stored in graph['crs']
    - Road geometries (LineStrings) stored as edge attributes
    - Node IDs stored in vertex attributes
    - Edge IDs (keys) for multi-edges between the same nodes
    - Optional distance and time weights for routing

    This implementation is particularly useful when working with very large road networks
    where performance is critical.

    Attributes:
        g: The igraph.Graph representing the road network
        road_mapping: A dictionary mapping RoadId tuples to igraph edge indices
        crs: The coordinate reference system of the map

    Note:
        igraph must be installed to use this class. Install with: pip install igraph
    """

    def __init__(self, graph: ig.Graph):
        self.g = graph

        if "crs_key" not in graph.attributes():
            crs_key = DEFAULT_CRS_KEY
        else:
            crs_key = graph["crs_key"]

        if crs_key not in graph.attributes():
            raise ValueError(
                "Input graph must have pyproj crs;"
                "You can set it like: `graph['crs'] = pyproj.CRS('EPSG:4326')`"
            )

        crs = graph[crs_key]

        if not isinstance(crs, CRS):
            raise TypeError(
                "Input graph must have pyproj crs;"
                "You can set it like: `graph.graph['crs'] = pyproj.CRS('EPSG:4326')`"
            )

        self.crs = crs

        if "distance_weight" not in graph.attributes():
            dist_weight = DEFAULT_DISTANCE_WEIGHT
        else:
            dist_weight = graph["distance_weight"]

        if "time_weight" not in graph.attributes():
            time_weight = DEFAULT_TIME_WEIGHT
        else:
            time_weight = graph["time_weight"]

        if "geometry_key" not in graph.attributes():
            geom_key = DEFAULT_GEOMETRY_KEY
        else:
            geom_key = graph["geometry_key"]

        if "metadata_key" not in graph.attributes():
            metadata_key = DEFAULT_METADATA_KEY
        else:
            metadata_key = graph["metadata_key"]

        if "node_id_name" not in graph.attributes():
            node_id_name = DEFAULT_NODE_ID_NAME
        else:
            node_id_name = graph["node_id_name"]

        if "edge_id_name" not in graph.attributes():
            edge_id_name = DEFAULT_EDGE_ID_NAME
        else:
            edge_id_name = graph["edge_id_name"]

        self._dist_weight = dist_weight
        self._time_weight = time_weight
        self._geom_key = geom_key
        self._metadata_key = metadata_key
        self._crs_key = crs_key
        self._node_id_name = node_id_name
        self._edge_id_name = edge_id_name

        # store the names of any additional added attributes
        self._additional_attribute_names: Set[str] = set()

        self._build_rtree()

        # build mapping from mappymatch road id to igraph edge id
        self.road_mapping = {}
        for e in self.g.es:
            source_node_id = e.source_vertex[self._node_id_name]
            target_node_id = e.target_vertex[self._node_id_name]
            road_key = e[self._edge_id_name]

            road_id = RoadId(source_node_id, target_node_id, road_key)
            self.road_mapping[road_id] = e.index

    def _build_road(
        self,
        edge_index: int,
    ) -> Road:
        """
        Build a road from a road id, pulling the edge data from the graph

        Be sure to check if the road id (_has_road_id) is in the graph before calling this method
        """
        edge = self.g.es[edge_index]
        source_node_id = edge.source_vertex[self._node_id_name]
        target_node_id = edge.target_vertex[self._node_id_name]
        road_key = edge[self._edge_id_name]

        edge_data = edge.attributes()
        metadata = edge_data.get(self._metadata_key)

        if metadata is None:
            metadata = {}
        else:
            metadata = metadata.copy()

        metadata[self._dist_weight] = edge_data.get(self._dist_weight)
        metadata[self._time_weight] = edge_data.get(self._time_weight)

        for attr in self._additional_attribute_names:
            metadata[attr] = edge_data.get(attr)

        road = Road(
            RoadId(source_node_id, target_node_id, road_key),
            edge_data[self._geom_key],
            metadata=metadata,
        )

        return road

    def _build_rtree(self):
        geometries = []
        edge_indices = []

        for e in self.g.es:
            geom = e.attributes()[self._geom_key]
            geometries.append(geom)
            edge_indices.append(e.index)

        if len(geometries) == 0:
            raise ValueError("No geometries found in graph; cannot build spatial index")

        self.strtree = STRtree(geometries)
        self.edge_indices = edge_indices

    def __str__(self):
        output_lines = [
            "Mappymatch IGraphMap object",
            f"graph: {self.g}",
        ]
        return "\n".join(output_lines)

    def __repr__(self):
        return self.__str__()

    @property
    def distance_weight(self) -> str:
        return self._dist_weight

    @property
    def time_weight(self) -> str:
        return self._time_weight

    def road_by_id(self, road_id: RoadId) -> Optional[Road]:
        """
        Get a road by its id

        Args:
            road_id: The id of the road to get

        Returns:
            The road with the given id, or None if it does not exist
        """
        if road_id not in self.road_mapping:
            return None

        return self._build_road(self.road_mapping[road_id])

    def set_road_attributes(self, attributes: Dict[RoadId, Dict[str, Any]]):
        """
        Set the attributes of the roads in the map

        Args:
            attributes: A dictionary mapping road ids to dictionaries of attributes

        Returns:
            None
        """
        geom_updated = False
        for road_id, attrs in attributes.items():
            edge_id = self.road_mapping.get(road_id)
            if edge_id is None:
                raise ValueError(f"Road id {road_id} not found in graph")
            for attr, val in attrs.items():
                self._additional_attribute_names.add(attr)
                if attr == self._geom_key:
                    geom_updated = True
                self.g.es[edge_id][attr] = val

        if geom_updated:
            self._build_rtree()

    @property
    def roads(self) -> List[Road]:
        roads = [self._build_road(e.index) for e in self.g.es]
        return roads

    @classmethod
    def from_nx_graph(cls, nx_graph: nx.MultiDiGraph) -> IGraphMap:
        """
        Build an IGraphMap from a networkx graph
        """
        igraph = ig.Graph.from_networkx(nx_graph)

        igraph.vs[DEFAULT_NODE_ID_NAME] = igraph.vs["_nx_name"]
        del igraph.vs["_nx_name"]
        igraph.es[DEFAULT_EDGE_ID_NAME] = igraph.es["_nx_multiedge_key"]
        del igraph.es["_nx_multiedge_key"]

        return IGraphMap(igraph)

    @classmethod
    def from_file(cls, file: Union[str, Path]) -> IGraphMap:
        """
        Build a IGraphMap instance from a file

        Args:
            file: The graph pickle file to load the graph from

        Returns:
            A IGraphMap instance
        """
        file = Path(file)
        if not file.suffix == ".pickle":
            raise ValueError("file must be a pickle file")
        g = ig.Graph.Read_Pickle(str(file))
        return IGraphMap(g)

    @classmethod
    def from_geofence(
        cls,
        geofence: Geofence,
        xy: bool = True,
        network_type: NetworkType = NetworkType.DRIVE,
    ) -> IGraphMap:
        """
        Read an OSM network graph into a IGraphMap

        Args:
            geofence: the geofence to clip the graph to
            xy: whether to use xy coordinates or lat/lon
            network_type: the network type to use for the graph

        Returns:
            a IGraphMap
        """
        if geofence.crs != LATLON_CRS:
            raise TypeError(
                f"the geofence must in the epsg:4326 crs but got {geofence.crs.to_authority()}"
            )

        nx_graph = nx_graph_from_osmnx(
            geofence=geofence, network_type=network_type, xy=xy
        )

        return IGraphMap.from_nx_graph(nx_graph)

    def to_file(self, outfile: Union[str, Path]):
        """
        Save the graph to a pickle file

        Args:
            outfile: The file to save the graph to
        """
        outfile = Path(outfile)
        if outfile.suffix != ".pickle":
            raise ValueError("outfile must have a .pickle suffix")

        self.g.write_pickle(str(outfile))

    def _nearest_edge_index(self, coord: Coordinate) -> int:
        """
        An internal method to find the nearest edge to a coordinate

        Args:
            coord: The coordinate to find the nearest road to

        Returns:
            The nearest edge index to the coordinate
        """
        if coord.crs != self.crs:
            raise ValueError(
                f"crs of origin {coord.crs} must match crs of map {self.crs}"
            )

        nearest_idx = self.strtree.nearest(coord.geom)
        if nearest_idx is None:
            raise ValueError(f"No roads found for {coord}")

        nearest_edge_index = self.edge_indices[nearest_idx]

        return nearest_edge_index

    def nearest_road(self, coord: Coordinate) -> Road:
        """
        A helper function to get the nearest road.

        Args:
            coord: The coordinate to find the nearest road to

        Returns:
            The nearest road to the coordinate
        """
        nearest_edge_index = self._nearest_edge_index(coord)
        return self._build_road(nearest_edge_index)

    def shortest_path(
        self,
        origin: Coordinate,
        destination: Coordinate,
        weight: Optional[Union[str, Callable]] = None,
    ) -> List[Road]:
        """
        Computes the shortest path between an origin and a destination

        Args:
            origin: The origin coordinate
            destination: The destination coordinate
            weight: The weight to use for the path, either a string or a function

        Returns:
            A list of roads that form the shortest path
        """
        if weight is None:
            weight = self._time_weight

        if callable(weight):
            raise NotImplementedError("IGraphMap does not support custom weights")

        if weight not in self.g.es.attributes():
            raise ValueError(f"weight {weight} is not a valid attribute of the graph")

        if origin.crs != self.crs:
            raise ValueError(
                f"crs of origin {origin.crs} must match crs of map {self.crs}"
            )
        elif destination.crs != self.crs:
            raise ValueError(
                f"crs of destination {destination.crs} must match crs of map {self.crs}"
            )

        origin_edge_index = self._nearest_edge_index(origin)
        dest_edge_index = self._nearest_edge_index(destination)

        # find which road is closest to the origin and destination
        origin_road = self._build_road(origin_edge_index)
        dest_road = self._build_road(dest_edge_index)

        ostart = Point(origin_road.geom.coords[0])
        oend = Point(origin_road.geom.coords[-1])

        dstart = Point(dest_road.geom.coords[0])
        dend = Point(dest_road.geom.coords[-1])

        u_dist = ostart.distance(origin.geom)
        v_dist = oend.distance(origin.geom)

        if u_dist <= v_dist:
            origin_vertex_id = self.g.es[origin_edge_index].source
        else:
            origin_vertex_id = self.g.es[origin_edge_index].target

        u_dist = dstart.distance(destination.geom)
        v_dist = dend.distance(destination.geom)

        if u_dist <= v_dist:
            dest_vertex_id = self.g.es[dest_edge_index].source
        else:
            dest_vertex_id = self.g.es[dest_edge_index].target

        edge_path = self.g.get_shortest_paths(
            origin_vertex_id,
            dest_vertex_id,
            weights=self.g.es[weight],
            output="epath",
        )

        roads = [self._build_road(i) for i in edge_path[0]]

        return roads
