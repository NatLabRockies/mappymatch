from unittest import TestCase
from unittest.mock import Mock

import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString, Point

from mappymatch.constructs.coordinate import Coordinate
from mappymatch.constructs.road import Road, RoadId
from mappymatch.constructs.trace import Trace
from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.maps.nx.readers.osm_readers import (
    NetworkType,
    parse_osmnx_graph,
)
from mappymatch.matchers.lcss.constructs import TrajectorySegment
from mappymatch.matchers.lcss.ops import join_segment
from mappymatch.utils.crs import XY_CRS
from tests import get_test_dir


class TestDisconnectedComponents(TestCase):
    """Test handling of disconnected graph components"""

    def setUp(self):
        """Create a simple disconnected graph for testing"""
        # Create a graph with two disconnected components
        self.g = nx.MultiDiGraph()

        # Component 1: Simple path from 0 -> 1 -> 2
        self.g.add_edge(
            0,
            1,
            0,
            geometry=LineString([(0, 0), (1, 0)]),
            kilometers=1.0,
            travel_time=60.0,
            metadata={},
        )
        self.g.add_edge(
            1,
            2,
            0,
            geometry=LineString([(1, 0), (2, 0)]),
            kilometers=1.0,
            travel_time=60.0,
            metadata={},
        )

        # Component 2: Separate path from 10 -> 11 -> 12
        self.g.add_edge(
            10,
            11,
            0,
            geometry=LineString([(10, 10), (11, 10)]),
            kilometers=1.0,
            travel_time=60.0,
            metadata={},
        )
        self.g.add_edge(
            11,
            12,
            0,
            geometry=LineString([(11, 10), (12, 10)]),
            kilometers=1.0,
            travel_time=60.0,
            metadata={},
        )

        # Add required graph attributes
        self.g.graph["crs"] = XY_CRS
        self.g.graph["distance_weight"] = "kilometers"
        self.g.graph["time_weight"] = "travel_time"
        self.g.graph["geometry_key"] = "geometry"

    def test_parse_osmnx_graph_keeps_all_components(self):
        """Test that parse_osmnx_graph can keep all components when filter_to_largest_component=False"""
        # Load test graph
        gfile = get_test_dir() / "test_assets" / "osmnx_drive_graph.graphml"
        osmnx_graph = ox.load_graphml(gfile)

        # Parse without filtering
        cleaned_graph = parse_osmnx_graph(
            osmnx_graph, NetworkType.DRIVE, filter_to_largest_component=False
        )

        # Graph should have edges and basic structure
        self.assertGreater(len(cleaned_graph.edges), 0)
        self.assertEqual(cleaned_graph.graph["network_type"], NetworkType.DRIVE.value)

    def test_parse_osmnx_graph_filters_to_largest(self):
        """Test that parse_osmnx_graph filters to largest component by default"""
        # Load test graph
        gfile = get_test_dir() / "test_assets" / "osmnx_drive_graph.graphml"
        osmnx_graph = ox.load_graphml(gfile)

        # Parse with filtering (default behavior)
        cleaned_graph = parse_osmnx_graph(
            osmnx_graph, NetworkType.DRIVE, filter_to_largest_component=True
        )

        # Graph should be strongly connected
        self.assertTrue(nx.is_strongly_connected(cleaned_graph))

    def test_shortest_path_returns_empty_for_disconnected_nodes(self):
        """Test that shortest_path returns empty list when no path exists"""
        # Create NxMap from disconnected graph
        nx_map = NxMap(self.g)

        # Create coordinates in different components
        origin = Coordinate(None, Point(0.5, 0), XY_CRS)
        destination = Coordinate(None, Point(10.5, 10), XY_CRS)

        # Should return empty list instead of raising exception
        path = nx_map.shortest_path(origin, destination)

        self.assertEqual(path, [])

    def test_shortest_path_works_within_component(self):
        """Test that shortest_path works normally within a connected component"""
        # Create NxMap from disconnected graph
        nx_map = NxMap(self.g)

        # Create coordinates in the same component
        origin = Coordinate(None, Point(0.5, 0), XY_CRS)
        destination = Coordinate(None, Point(1.5, 0), XY_CRS)

        # Should find a path
        path = nx_map.shortest_path(origin, destination)

        self.assertGreater(len(path), 0)
        self.assertIsInstance(path[0], Road)

    def test_lcss_merge_handles_empty_path(self):
        """Test that LCSS merge handles empty path between disconnected segments"""
        # Create mock road map
        mock_map = Mock(spec=NxMap)
        mock_map.crs = XY_CRS

        # Mock shortest_path to return empty list (disconnected components)
        mock_map.shortest_path.return_value = []

        # Create mock trajectory segments with paths
        coords1 = [
            Coordinate(None, Point(0, 0), XY_CRS),
            Coordinate(None, Point(1, 0), XY_CRS),
        ]
        coords2 = [
            Coordinate(None, Point(10, 10), XY_CRS),
            Coordinate(None, Point(11, 10), XY_CRS),
        ]

        gdf1 = gpd.GeoDataFrame(
            {"geometry": [c.geom for c in coords1]}, crs=XY_CRS, index=[0, 1]
        )
        gdf2 = gpd.GeoDataFrame(
            {"geometry": [c.geom for c in coords2]}, crs=XY_CRS, index=[2, 3]
        )

        trace1 = Trace(gdf1)
        trace2 = Trace(gdf2)

        road1 = Road(
            RoadId(0, 1, 0),
            LineString([(0, 0), (1, 0)]),
            metadata={},
        )
        road2 = Road(
            RoadId(10, 11, 0),
            LineString([(10, 10), (11, 10)]),
            metadata={},
        )

        segment_a = TrajectorySegment(trace=trace1, path=[road1])
        segment_b = TrajectorySegment(trace=trace2, path=[road2])

        # Merge segments using imported join_segment function
        result = join_segment(mock_map, segment_a, segment_b)

        # Should concatenate paths without intermediate routing
        self.assertEqual(len(result.path), 2)
        self.assertEqual(result.path[0].road_id, road1.road_id)
        self.assertEqual(result.path[1].road_id, road2.road_id)

    def test_lcss_merge_handles_connected_path(self):
        """Test that LCSS merge works normally when routing succeeds"""
        # Create mock road map
        mock_map = Mock(spec=NxMap)
        mock_map.crs = XY_CRS

        # Mock shortest_path to return a connecting road
        connecting_road = Road(
            RoadId(1, 2, 0),
            LineString([(1, 0), (2, 0)]),
            metadata={},
        )
        mock_map.shortest_path.return_value = [connecting_road]

        # Create mock trajectory segments
        coords1 = [
            Coordinate(None, Point(0, 0), XY_CRS),
            Coordinate(None, Point(1, 0), XY_CRS),
        ]
        coords2 = [
            Coordinate(None, Point(2, 0), XY_CRS),
            Coordinate(None, Point(3, 0), XY_CRS),
        ]

        gdf1 = gpd.GeoDataFrame(
            {"geometry": [c.geom for c in coords1]}, crs=XY_CRS, index=[0, 1]
        )
        gdf2 = gpd.GeoDataFrame(
            {"geometry": [c.geom for c in coords2]}, crs=XY_CRS, index=[2, 3]
        )

        trace1 = Trace(gdf1)
        trace2 = Trace(gdf2)

        road1 = Road(
            RoadId(0, 1, 0),
            LineString([(0, 0), (1, 0)]),
            metadata={},
        )
        road3 = Road(
            RoadId(2, 3, 0),
            LineString([(2, 0), (3, 0)]),
            metadata={},
        )

        segment_a = TrajectorySegment(trace=trace1, path=[road1])
        segment_b = TrajectorySegment(trace=trace2, path=[road3])

        # Merge segments using imported join_segment function
        result = join_segment(mock_map, segment_a, segment_b)

        # Should include connecting road
        self.assertEqual(len(result.path), 3)
        self.assertEqual(result.path[0].road_id, road1.road_id)
        self.assertEqual(result.path[1].road_id, connecting_road.road_id)
        self.assertEqual(result.path[2].road_id, road3.road_id)

    def test_networkx_no_path_exception_handling(self):
        """Test that NetworkXNoPath exception is caught and handled"""
        # Create NxMap from disconnected graph
        nx_map = NxMap(self.g)

        # Verify graph is disconnected
        self.assertFalse(nx.is_strongly_connected(self.g))

        # Try to find path between disconnected components
        origin = Coordinate(None, Point(0.1, 0), XY_CRS)
        destination = Coordinate(None, Point(10.1, 10), XY_CRS)

        # Should not raise exception, should return empty list
        try:
            path = nx_map.shortest_path(origin, destination)
            self.assertEqual(path, [])
        except nx.NetworkXNoPath:
            self.fail("NetworkXNoPath exception should be caught and handled")
