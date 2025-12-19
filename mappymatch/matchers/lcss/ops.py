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
    Join two trajectory segments together, attempting to route between them if needed.

    Args:
        road_map: The road map to use for routing
        a: The first trajectory segment
        b: The second trajectory segment

    Returns:
        A new trajectory segment combining both segments
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
    Computes a shortest path and returns the path

    Args:
        road_map: the road map to match to
        trace: the trace to match

    Returns:
        the path that most closely matches the trace
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
    Splits a trajectory segment based on the provided cutting points.

    Merge back any segments that are too short

    Args:
        road_map: the road map to match to
        trajectory_segment: the trajectory segment to split
        distance_epsilon: the distance epsilon

    Returns:
        a list of split segments or the original segment if it can't be split
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
    Find the positional index of all stationary points in a trace

    Args:
        trace: the trace to find the stationary points in

    Returns:
        a list of stationary indices
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
    Drops stationary points from the trace, keeping the first point

    Args:
        trace: the trace to drop the stationary points from
        stationary_index: the stationary indices to drop

    Returns:
        the trace with the stationary points dropped
    """
    for si in stationary_index:
        trace = trace.drop(si.c_index[1:])

    return trace


def add_matches_for_stationary_points(
    matches: List[Match],
    stationary_index: List[StationaryIndex],
) -> List[Match]:
    """
    Takes a set of matches and adds duplicate match entries for stationary

    Args:
        matches: the matches to add the stationary points to
        stationary_index: the stationary indices to add

    Returns:
        the matches with the stationary points added
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
