from __future__ import annotations

import logging
import random
from typing import List, NamedTuple, Union

import numpy as np
from numpy import ndarray, signedinteger

from mappymatch.constructs.match import Match
from mappymatch.constructs.road import Road
from mappymatch.constructs.trace import Trace
from mappymatch.matchers.lcss.utils import compress
from mappymatch.utils.geo import coord_to_coord_dist

log = logging.getLogger(__name__)


class CuttingPoint(NamedTuple):
    """
    Represents a location where the LCSS algorithm splits a trajectory into sub-segments.

    Cutting points are identified during the iterative refinement process as locations where
    the GPS trajectory deviates significantly from the candidate path. Splitting at these
    points allows the algorithm to find better local matches.

    Attributes:
        trace_index: The integer index in the trace where the cut should be made.
            This indexes into the trace's coordinate list.
    """

    trace_index: Union[signedinteger, int]


class TrajectorySegment(NamedTuple):
    """
    Represents a pairing of a GPS trace segment with a candidate path through the road network.

    TrajectorySegments are the core data structure used by the LCSS matcher. Each segment
    contains a portion of the GPS trace, a candidate path through the road network, matching
    results, a similarity score, and potential cutting points for further refinement.

    During the iterative LCSS algorithm, segments are scored, split at cutting points, and
    merged to find the best overall match between the GPS trajectory and the road network.

    Attributes:
        trace: The GPS trace segment being matched
        path: The candidate path through the road network (list of Road objects)
        matches: List of Match objects linking each GPS point to a road. Empty until score_and_match is called.
        score: The LCSS similarity score between trace and path (0-1). 0 until scored.
        cutting_points: List of CuttingPoint objects indicating where to split this segment for further refinement. Empty until compute_cutting_points is called.
    """

    trace: Trace
    path: List[Road]

    matches: List[Match] = []

    score: float = 0

    cutting_points: List[CuttingPoint] = []

    def __add__(self, other):
        new_traces = self.trace + other.trace
        new_paths = self.path + other.path
        return TrajectorySegment(new_traces, new_paths)

    def set_score(self, score: float) -> TrajectorySegment:
        """
        Sets the score of the trajectory segment

        Args:
            score: The score of the trajectory segment

        Returns:
            The updated trajectory segment
        """
        return self._replace(score=score)

    def set_cutting_points(self, cutting_points) -> TrajectorySegment:
        """
        Sets the cutting points of the trajectory segment

        Args:
            cutting_points: The cutting points of the trajectory segment

        Returns:
            The updated trajectory segment
        """
        return self._replace(cutting_points=cutting_points)

    def set_matches(self, matches) -> TrajectorySegment:
        """
        Sets the matches of the trajectory segment

        Args:
            matches: The matches of the trajectory segment

        Returns:
            The updated trajectory segment
        """
        return self._replace(matches=matches)

    def score_and_match(
        self,
        distance_epsilon: float,
        max_distance: float,
    ) -> TrajectorySegment:
        """
        Compute the LCSS similarity score and match GPS points to road segments.

        This method implements the core LCSS (Longest Common Subsequence) algorithm for
        trajectory matching. It computes a similarity score between the GPS trace and
        the candidate path based on point-to-path distances, and simultaneously matches
        each GPS coordinate to its nearest road in the path.

        The LCSS similarity score ranges from 0 (no similarity) to 1 (perfect match),
        with higher scores indicating better alignment between the trajectory and path.

        Args:
            distance_epsilon: The distance threshold (in meters) for similarity. GPS points
                within this distance of the path contribute positively to the score. Points
                farther away contribute zero.
            max_distance: The maximum distance (in meters) for matching. GPS points beyond
                this distance from the nearest road are left unmatched (road=None, distance=inf).

        Returns:
            A new TrajectorySegment with populated matches list and computed score

        Raises:
            Exception: If the trace has 0 points (edge case that cannot be matched)
        """
        trace = self.trace
        path = self.path

        m = len(trace.coords)
        n = len(path)

        matched_roads = []

        if m < 1:
            # todo: find a better way to handle this edge case
            raise Exception("traces of 0 points can't be matched")
        elif n == 0:
            # a path was not found for this segment; might not be matchable;
            # we set a score of zero and return a set of no-matches
            matches = [
                Match(road=None, distance=np.inf, coordinate=c)
                for c in self.trace.coords
            ]
            return self.set_score(0).set_matches(matches)

        C = [[0 for i in range(n + 1)] for j in range(m + 1)]

        f = trace._frame
        distances = np.array([f.distance(r.geom).values for r in path])

        for i in range(1, m + 1):
            nearest_road = None
            min_dist = np.inf
            coord = trace.coords[i - 1]
            for j in range(1, n + 1):
                road = path[j - 1]

                # dt = road_to_coord_dist(road, coord)
                dt = distances[j - 1][i - 1]

                if dt < min_dist:
                    min_dist = dt
                    nearest_road = road

                if dt < distance_epsilon:
                    point_similarity = 1 - (dt / distance_epsilon)
                else:
                    point_similarity = 0

                C[i][j] = max(
                    (C[i - 1][j - 1] + point_similarity),
                    C[i][j - 1],
                    C[i - 1][j],
                )

            if min_dist > max_distance:
                nearest_road = None
                min_dist = np.inf

            match = Match(
                road=nearest_road,
                distance=min_dist,
                coordinate=coord,
            )
            matched_roads.append(match)

        sim_score = C[m][n] / float(min(m, n))

        return self.set_score(sim_score).set_matches(matched_roads)

    def compute_cutting_points(
        self,
        distance_epsilon: float,
        cutting_thresh: float,
        random_cuts: int,
    ) -> TrajectorySegment:
        """
        Identify locations where the trajectory should be split for refinement.

        Cutting points are locations where the GPS trace deviates from the candidate path,
        suggesting that splitting the trajectory at these points and computing separate
        paths for each segment may yield better matches.

        The method identifies cutting points by:
        1. Finding the point with maximum distance from the matched path (highest deviation)
        2. Finding points near the distance_epsilon threshold (borderline matches)
        3. Optionally adding random points for exploration
        4. Handling edge cases (no path, circular routes, etc.)

        Adjacent cutting points are compressed to avoid splitting too finely, and cutting
        points at the start/end of the trace are removed (can't split there).

        Args:
            distance_epsilon: The distance threshold (in meters) used for matching.
                Points near this threshold are candidates for cutting.
            cutting_thresh: The distance tolerance (in meters) around distance_epsilon.
                Points within cutting_thresh of distance_epsilon are marked as cutting points.
            random_cuts: Number of random cutting points to add for exploration.
                Typically 0 for deterministic results.

        Returns:
            A new TrajectorySegment with populated cutting_points list
        """
        cutting_points = []

        no_match = all([not m.road for m in self.matches])

        if not self.path or no_match:
            # no path computed or no matches found, possible edge cases:
            # 1. trace starts and ends in the same location: pick points far from the start and end
            start = self.trace.coords[0]
            end = self.trace.coords[-1]

            start_end_dist = start.geom.distance(end.geom)

            if start_end_dist < distance_epsilon:
                p1 = np.argmax(
                    [coord_to_coord_dist(start, c) for c in self.trace.coords]
                )
                p2 = np.argmax([coord_to_coord_dist(end, c) for c in self.trace.coords])
                assert not isinstance(p1, ndarray)
                assert not isinstance(p2, ndarray)
                # To do - np.argmax returns array of indices where the highest value is found.
                # if there is only one highest value an int is returned. CuttingPoint takes an int.
                # if an array is returned by argmax, this throws an error
                cp1 = CuttingPoint(p1)
                cp2 = CuttingPoint(p2)

                cutting_points.extend([cp1, cp2])
            else:
                # pick the middle point on the trace:
                mid = int(len(self.trace) / 2)
                cp = CuttingPoint(mid)
                cutting_points.append(cp)
        else:
            # find furthest point
            pre_i = np.argmax([m.distance for m in self.matches if m.road])
            cutting_points.append(CuttingPoint(pre_i))

            # collect points that are close to the distance threshold
            for i, m in enumerate(self.matches):
                if m.road:
                    if abs(m.distance - distance_epsilon) < cutting_thresh:
                        cutting_points.append(CuttingPoint(i))

        # add random points
        for _ in range(random_cuts):
            cpi = random.randint(0, len(self.trace) - 1)
            cutting_points.append(CuttingPoint(cpi))

        # merge cutting points that are adjacent to one another
        compressed_cuts = list(compress(cutting_points))

        # it doesn't make sense to cut the trace at the start or end so discard any
        # points that apear in the [0, 1, -1, -2] position with respect to a trace
        n = len(self.trace)
        final_cuts = list(
            filter(
                lambda cp: cp.trace_index not in [0, 1, n - 2, n - 1],
                compressed_cuts,
            )
        )

        return self.set_cutting_points(final_cuts)


TrajectoryScheme = List[TrajectorySegment]
