from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, List, Optional, Set, Union

import networkx as nx
import shapely.wkt as wkt
from shapely.geometry import Point
from shapely.strtree import STRtree

from mappymatch.constructs.coordinate import Coordinate
from mappymatch.constructs.geofence import Geofence
from mappymatch.constructs.road import Road, RoadId
from mappymatch.maps.igraph.igraph_map import DEFAULT_METADATA_KEY
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
from mappymatch.utils.keys import DEFAULT_CRS_KEY, DEFAULT_GEOMETRY_KEY


class NxMap(MapInterface):
    """
    A road map that uses a networkx graph to represent its roads.

    Attributes:
        g: The networkx graph that represents the road map
        crs: The coordinate reference system of the map
    """

    def __init__(self, graph: nx.MultiDiGraph):
        self.g = graph

        crs_key = graph.graph.get("crs_key", DEFAULT_CRS_KEY)

        if crs_key not in graph.graph:
            raise ValueError(
                "Input graph must have pyproj crs;"
                "You can set it like: `graph.graph['crs'] = pyproj.CRS('EPSG:4326')`"
            )

        crs = graph.graph[crs_key]

        if not isinstance(crs, CRS):
            raise TypeError(
                "Input graph must have pyproj crs;"
                "You can set it like: `graph.graph['crs'] = pyproj.CRS('EPSG:4326')`"
            )

        self.crs = crs

        dist_weight = graph.graph.get("distance_weight", DEFAULT_DISTANCE_WEIGHT)
        time_weight = graph.graph.get("time_weight", DEFAULT_TIME_WEIGHT)
        geom_key = graph.graph.get("geometry_key", DEFAULT_GEOMETRY_KEY)
        metadata_key = graph.graph.get("metadata_key", DEFAULT_METADATA_KEY)

        self._dist_weight = dist_weight
        self._time_weight = time_weight
        self._geom_key = geom_key
        self._metadata_key = metadata_key
        self._crs_key = crs_key

        self._addtional_attribute_names: Set[str] = set()

        self._build_rtree()

    def _has_road_id(self, road_id: RoadId) -> bool:
        return self.g.has_edge(*road_id)

    def _build_road(
        self,
        road_id: RoadId,
    ) -> Road:
        """
        Build a road from a road id, pulling the edge data from the graph

        Be sure to check if the road id (_has_road_id) is in the graph before calling this method
        """
        edge_data = self.g.get_edge_data(*road_id)

        metadata = edge_data.get(self._metadata_key)

        if metadata is None:
            metadata = {}
        else:
            metadata = metadata.copy()

        metadata[self._dist_weight] = edge_data.get(self._dist_weight)
        metadata[self._time_weight] = edge_data.get(self._time_weight)

        for attr_name in self._addtional_attribute_names:
            metadata[attr_name] = edge_data.get(attr_name)

        road = Road(
            road_id,
            edge_data[self._geom_key],
            metadata=metadata,
        )

        return road

    def _build_rtree(self):
        geoms = []
        road_ids = []

        for u, v, k, d in self.g.edges(data=True, keys=True):
            road_id = RoadId(u, v, k)
            geom = d[self._geom_key]
            geoms.append(geom)
            road_ids.append(road_id)

        if len(geoms) == 0:
            raise ValueError("No geometries found in graph; cannot build spatial index")

        self.rtree = STRtree(geoms)
        self._road_id_mapping = road_ids

    def __str__(self):
        output_lines = [
            "Mappymatch NxMap object:\n",
            f" - roads: {len(self.g.edges)} Road objects",
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
        if self._has_road_id(road_id):
            return self._build_road(road_id)
        else:
            return None

    def set_road_attributes(self, attributes: Dict[RoadId, Dict[str, Any]]):
        """
        Set the attributes of the roads in the map

        Args:
            attributes: A dictionary mapping road ids to dictionaries of attributes

        Returns:
            None
        """
        for attrs in attributes.values():
            for attr_name in attrs.keys():
                self._addtional_attribute_names.add(attr_name)

        nx.set_edge_attributes(self.g, attributes)
        self._build_rtree()

    @property
    def roads(self) -> List[Road]:
        roads = [
            self._build_road(RoadId(u, v, k)) for u, v, k in self.g.edges(keys=True)
        ]
        return roads

    @classmethod
    def from_file(cls, file: Union[str, Path]) -> NxMap:
        """
        Build a NxMap instance from a file

        Args:
            file: The graph pickle file to load the graph from

        Returns:
            A NxMap instance
        """
        p = Path(file)
        if p.suffix == ".pickle":
            with open(p, "rb") as f:
                return pickle.load(f)
        elif p.suffix == ".json":
            with p.open("r") as f:
                return NxMap.from_dict(json.load(f))
        else:
            raise TypeError("NxMap only supports reading from json and pickle files")

    @classmethod
    def from_geofence(
        cls,
        geofence: Geofence,
        xy: bool = True,
        network_type: NetworkType = NetworkType.DRIVE,
        custom_filter: Optional[str] = None,
        additional_metadata_keys: Optional[set | list] = None,
        filter_to_largest_component: bool = True,
    ) -> NxMap:
        """
        Read an OSM network graph into a NxMap

        Args:
            geofence: the geofence to clip the graph to
            xy: whether to use xy coordinates or lat/lon
            network_type: the network type to use for the graph
            custom_filter: a custom filter to pass to osmnx like '["highway"~"motorway|primary"]'
            additional_metadata_keys: additional keys to preserve in road metadata like '["maxspeed", "highway"]'
            filter_to_largest_component: if True, keep only the largest strongly connected component; if False, keep all components (may result in routing failures between disconnected components)

        Returns:
            a NxMap
        """
        if geofence.crs != LATLON_CRS:
            raise TypeError(
                f"the geofence must in the epsg:4326 crs but got {geofence.crs.to_authority()}"
            )

        if additional_metadata_keys is not None:
            additional_metadata_keys = set(additional_metadata_keys)

        nx_graph = nx_graph_from_osmnx(
            geofence=geofence,
            network_type=network_type,
            xy=xy,
            custom_filter=custom_filter,
            additional_metadata_keys=additional_metadata_keys,
            filter_to_largest_component=filter_to_largest_component,
        )

        return NxMap(nx_graph)

    def to_file(self, outfile: Union[str, Path]):
        """
        Save the graph to a pickle file

        Args:
            outfile: The file to save the graph to
        """
        outfile = Path(outfile)

        if outfile.suffix == ".pickle":
            with open(outfile, "wb") as f:
                pickle.dump(self, f)
        elif outfile.suffix == ".json":
            graph_dict = self.to_dict()
            with open(outfile, "w") as f:
                json.dump(graph_dict, f)
        else:
            raise TypeError("NxMap only supports writing to json and pickle files")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> NxMap:
        """
        Build a NxMap instance from a dictionary
        """
        geom_key = d["graph"].get("geometry_key", DEFAULT_GEOMETRY_KEY)

        for link in d["links"]:
            geom_wkt = link[geom_key]
            link[geom_key] = wkt.loads(geom_wkt)

        crs_key = d["graph"].get("crs_key", DEFAULT_CRS_KEY)
        crs = CRS.from_wkt(d["graph"][crs_key])
        d["graph"][crs_key] = crs

        g = nx.readwrite.json_graph.node_link_graph(d)

        return NxMap(g)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the map to a dictionary
        """
        graph_dict = nx.readwrite.json_graph.node_link_data(self.g)

        # convert geometries to well know text
        for link in graph_dict["links"]:
            geom = link[self._geom_key]
            link[self._geom_key] = geom.wkt

        # convert crs to well known text
        crs_key = graph_dict["graph"].get("crs_key", DEFAULT_CRS_KEY)
        graph_dict["graph"][crs_key] = self.crs.to_wkt()

        return graph_dict

    def nearest_road(self, coord: Coordinate) -> Road:
        """
        A helper function to get the nearest road.

        Args:
            coord: The coordinate to find the nearest road to

        Returns:
            The nearest road to the coordinate
        """
        if coord.crs != self.crs:
            raise ValueError(
                f"crs of origin {coord.crs} must match crs of map {self.crs}"
            )

        nearest_idx = self.rtree.nearest(coord.geom)
        if nearest_idx is None:
            raise ValueError(f"No roads found for {coord}")
        nearest_id = self._road_id_mapping[nearest_idx]

        road = self._build_road(nearest_id)

        return road

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
        if origin.crs != self.crs:
            raise ValueError(
                f"crs of origin {origin.crs} must match crs of map {self.crs}"
            )
        elif destination.crs != self.crs:
            raise ValueError(
                f"crs of destination {destination.crs} must match crs of map {self.crs}"
            )

        if weight is None:
            weight = self._time_weight

        origin_road = self.nearest_road(origin)
        dest_road = self.nearest_road(destination)

        ostart = Point(origin_road.geom.coords[0])
        oend = Point(origin_road.geom.coords[-1])

        dstart = Point(dest_road.geom.coords[0])
        dend = Point(dest_road.geom.coords[-1])

        u_dist = ostart.distance(origin.geom)
        v_dist = oend.distance(origin.geom)

        if u_dist <= v_dist:
            origin_id = origin_road.road_id.start
        else:
            origin_id = origin_road.road_id.end

        u_dist = dstart.distance(destination.geom)
        v_dist = dend.distance(destination.geom)

        if u_dist <= v_dist:
            dest_id = dest_road.road_id.start
        else:
            dest_id = dest_road.road_id.end

        try:
            nx_route = nx.shortest_path(
                self.g,
                origin_id,
                dest_id,
                weight=weight,
            )
        except nx.NetworkXNoPath:
            # No path exists between origin and destination
            # This can happen when the graph has multiple disconnected components
            return []

        path = []
        for i in range(1, len(nx_route)):
            road_start_node = nx_route[i - 1]
            road_end_node = nx_route[i]

            edge_data = self.g.get_edge_data(road_start_node, road_end_node)
            road_key = list(edge_data.keys())[0]

            road_id = RoadId(road_start_node, road_end_node, road_key)

            road = self._build_road(road_id)

            path.append(road)

        return path
