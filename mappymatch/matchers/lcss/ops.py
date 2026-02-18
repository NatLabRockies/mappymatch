import logging
from copy import deepcopy
from typing import Any, List, NamedTuple

from shapely.geometry import Point

from mappymatch.constructs.coordinate import Coordinate
from mappymatch.constructs.match import Match
from mappymatch.constructs.road import Road
from mappymatch.constructs.trace import Trace
from mappymatch.maps.map_interface import MapInterface
from mappymatch.matchers.lcss.constructs import (
    TrajectoryScheme,
    TrajectorySegment,
)
from mappymatch.matchers.lcss.utils import merge

log = logging.getLogger(__name__)


def join_segment(
    road_map: MapInterface, a: TrajectorySegment, b: TrajectorySegment
) -> TrajectorySegment:
    """
    Join two trajectory segments together, routing between them if there's a gap.

    This function concatenates two trajectory segments end-to-end. If the paths don't
    connect directly (i.e., the end junction of segment A doesn't match the start junction
    of segment B), it attempts to find a shortest path to bridge the gap.

    Used during the LCSS algorithm after splitting segments to recombine them into a
    complete trajectory.

    Args:
        road_map: The road network used for finding connecting paths
        a: The first trajectory segment (will be the start of the combined segment)
        b: The second trajectory segment (will be the end of the combined segment)

    Returns:
        A new TrajectorySegment with combined traces and paths. If a connecting path
        is found, it's inserted between the two path segments. If no path exists
        (disconnected network components), the paths are simply concatenated.
    """
    new_traces = a.trace + b.trace
    new_path = a.path + b.path

    # test to see if there is a gap between the paths and if so,
    # try to connect it
    if len(a.path) > 0 and len(b.path) > 0:
        end_road = a.path[-1]
        start_road = b.path[0]
        if end_road.road_id.end != start_road.road_id.start:
            o = Coordinate(
                coordinate_id=None,
                geom=Point(end_road.geom.coords[-1]),
                crs=new_traces.crs,
            )
            d = Coordinate(
                coordinate_id=None,
                geom=Point(start_road.geom.coords[0]),
                crs=new_traces.crs,
            )
            path = road_map.shortest_path(o, d)
            # If no path exists (disconnected components), just concatenate the paths
            if path:
                new_path = a.path + path + b.path
            else:
                new_path = a.path + b.path

    return TrajectorySegment(new_traces, new_path)


def new_path(
    road_map: MapInterface,
    trace: Trace,
) -> List[Road]:
    """
    Compute a candidate path through the road network for a GPS trace.

    This computes the shortest path from the first coordinate to the last coordinate
    in the trace, using the road network's shortest path algorithm. This path serves
    as the initial candidate path for LCSS matching.

    Args:
        road_map: The road network to route on
        trace: The GPS trace to compute a path for

    Returns:
        A list of Road objects representing the shortest path from the trace's first
        to last coordinate. Returns an empty list if the trace has fewer than 1 coordinate.
    """
    if len(trace.coords) < 1:
        return []

    origin = trace.coords[0]
    destination = trace.coords[-1]

    new_path = road_map.shortest_path(origin, destination)

    return new_path


def split_trajectory_segment(
    road_map: MapInterface,
    trajectory_segment: TrajectorySegment,
) -> List[TrajectorySegment]:
    """
    Split a trajectory segment at its cutting points into multiple sub-segments.

    This is a core operation in the LCSS algorithm. It divides a trajectory at identified
    cutting points, computes new candidate paths for each sub-segment, and merges any
    resulting segments that are too short (fewer than 2 trace points or 1 path edge).

    The splitting process helps refine matches by allowing different parts of the trajectory
    to follow different paths through the network.

    Args:
        road_map: The road network used to compute paths for the new segments
        trajectory_segment: The segment to split, must have cutting_points populated

    Returns:
        A list of new TrajectorySegments created by splitting at cutting points.
        Returns the original segment unchanged if:
        - The trace has fewer than 2 points
        - No cutting points are defined
        - Splitting would not improve the match

        Short segments (< 2 trace points or < 1 path edge) are automatically merged
        with adjacent segments.
    """
    trace = trajectory_segment.trace
    cutting_points = trajectory_segment.cutting_points

    def _short_segment(ts: TrajectorySegment):
        if len(ts.trace) < 2 or len(ts.path) < 1:
            return True
        return False

    if len(trace.coords) < 2:
        # segment is too short to split
        return [trajectory_segment]
    elif len(cutting_points) < 1:
        # no points to cut
        return [trajectory_segment]

    new_paths = []
    new_traces = []

    # using type: ignore below because, trace_index can only be a signedinteger or integer
    # mypy wants it to only be an int, but this should never affect code functionality
    # start
    scp = cutting_points[0]
    new_trace = trace[: scp.trace_index]  # type: ignore
    new_paths.append(new_path(road_map, new_trace))
    new_traces.append(new_trace)

    # mids
    for i in range(len(cutting_points) - 1):
        cp = cutting_points[i]
        ncp = cutting_points[i + 1]
        new_trace = trace[cp.trace_index : ncp.trace_index]  # type: ignore
        new_paths.append(new_path(road_map, new_trace))
        new_traces.append(new_trace)

    # end
    ecp = cutting_points[-1]
    new_trace = trace[ecp.trace_index :]  # type: ignore
    new_paths.append(new_path(road_map, new_trace))
    new_traces.append(new_trace)

    if not any(new_paths):
        # can't split
        return [trajectory_segment]
    elif not any(new_traces):
        # can't split
        return [trajectory_segment]
    else:
        segments = [TrajectorySegment(t, p) for t, p in zip(new_traces, new_paths)]

    merged_segments = merge(segments, _short_segment)

    return merged_segments


def same_trajectory_scheme(
    scheme1: TrajectoryScheme, scheme2: TrajectoryScheme
) -> bool:
    """
    Compares two trajectory schemes for equality

    Args:
        scheme1: the first trajectory scheme
        scheme2: the second trajectory scheme

    Returns:
        True if the two schemes are equal, False otherwise
    """
    same_paths = all(map(lambda a, b: a.path == b.path, scheme1, scheme2))
    same_traces = all(
        map(lambda a, b: a.trace.coords == b.trace.coords, scheme1, scheme2)
    )

    return same_paths and same_traces


class StationaryIndex(NamedTuple):
    """
    An index of a stationary point in a trajectory

    Attributes:
        trace_index: the index of the trace
        coord_index: the index of the coordinate
    """

    i_index: List[int]  # i based index on the trace
    c_index: List[Any]  # coordinate ids


def find_stationary_points(trace: Trace) -> List[StationaryIndex]:
    """
    Identify groups of consecutive GPS points that represent stationary positions.

    Stationary points occur when a GPS device records multiple positions while not moving,
    or moving very slowly. These can be caused by waiting at traffic lights, parking, or
    GPS noise. Identifying them allows the LCSS matcher to handle them specially.

    Points are considered stationary if they are within 0.001 meters (1mm) of the previous
    point - essentially the same location accounting for floating-point precision.

    Args:
        trace: The GPS trace to analyze for stationary points

    Returns:
        A list of StationaryIndex objects, each representing a group of consecutive
        stationary points. Each StationaryIndex contains:
        - i_index: List of integer indices in the trace
        - c_index: List of coordinate IDs
    """
    f = trace._frame
    coords = trace.coords
    dist = f.distance(f.shift())
    index_collections = []
    index = set()
    for i in range(1, len(dist)):
        d = dist.iloc[i]  # distance to previous point
        if d < 0.001:
            index.add(i - 1)
            index.add(i)
        else:
            # there is distance between this point and the previous
            if index:
                l_index = sorted(list(index))
                cids = [coords[li].coordinate_id for li in l_index]
                si = StationaryIndex(l_index, cids)
                index_collections.append(si)
                index = set()

    # catch any group of points at the end
    if index:
        l_index = sorted(list(index))
        cids = [coords[li].coordinate_id for li in l_index]
        si = StationaryIndex(l_index, cids)
        index_collections.append(si)

    return index_collections


def drop_stationary_points(
    trace: Trace, stationary_index: List[StationaryIndex]
) -> Trace:
    """
    Remove stationary points from a trace while keeping the first point of each group.

    This is used to simplify traces before matching by collapsing groups of stationary
    points into single representatives. The LCSS matching is performed on the simplified
    trace, then stationary points are restored in the final results.

    Args:
        trace: The GPS trace to clean
        stationary_index: List of StationaryIndex objects identifying stationary point groups (from find_stationary_points)

    Returns:
        A new Trace with duplicate stationary points removed. For each stationary group,
        only the first point is retained.
    """
    for si in stationary_index:
        trace = trace.drop(si.c_index[1:])

    return trace


def add_matches_for_stationary_points(
    matches: List[Match],
    stationary_index: List[StationaryIndex],
) -> List[Match]:
    """
    Restore stationary points to the matched results.

    After matching a simplified trace (with stationary points removed), this function
    adds back Match objects for all the removed stationary points. Each stationary point
    gets the same road match as the first point in its group, but retains its original
    coordinate ID.

    This ensures the final MatchResult has one match for every point in the original trace.

    Args:
        matches: The matches from matching the simplified trace (without stationary points)
        stationary_index: List of StationaryIndex objects identifying which points were removed (from find_stationary_points)

    Returns:
        A new list of Match objects with stationary points restored, maintaining the
        original trace order and coordinate IDs
    """
    matches = deepcopy(matches)

    for si in stationary_index:
        mi = si.i_index[0]
        m = matches[mi]
        new_matches = [
            m.set_coordinate(
                Coordinate(ci, geom=m.coordinate.geom, crs=m.coordinate.crs)
            )
            for ci in si.c_index[1:]
        ]
        matches[si.i_index[1] : si.i_index[1]] = new_matches

    return matches
