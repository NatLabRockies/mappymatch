from __future__ import annotations

from mappymatch.constructs.trace import Trace


def split_large_trace(trace: Trace, ideal_size: int) -> list[Trace]:
    """
    Split a large GPS trace into smaller sub-traces of approximately equal size.

    This is useful for processing very long GPS trajectories that might be too large
    for certain matching algorithms or API limits. The trace is divided into segments
    of the specified size, with intelligent handling of the final segment to avoid
    creating very small trailing segments.

    Args:
        trace: The GPS trace to split
        ideal_size: The target number of coordinates for each sub-trace. Must be greater than 0.

    Returns:
        A list of Trace objects. If the trace is already smaller than ideal_size,
        returns a list containing just the original trace. Otherwise, returns multiple
        traces of approximately ideal_size coordinates each.

    Raises:
        ValueError: If ideal_size is 0 or negative

    Note:
        If the final segment would be 10 points or fewer, it's merged with the previous
        segment to avoid creating very small traces.

    Examples:
        >>> # Split a 500-point trace into ~100-point chunks
        >>> long_trace = Trace.from_csv('long_journey.csv')  # 500 points
        >>> sub_traces = split_large_trace(long_trace, ideal_size=100)
        >>> print(f"Split into {len(sub_traces)} traces")  # 5 traces
        >>> for i, t in enumerate(sub_traces):
        ...     print(f"Trace {i}: {len(t)} points")
        >>>
        >>> # Small traces are returned unchanged
        >>> small_trace = Trace.from_csv('short_trip.csv')  # 50 points
        >>> result = split_large_trace(small_trace, ideal_size=100)
        >>> assert len(result) == 1  # Just the original trace
    """
    if ideal_size == 0:
        raise ValueError("ideal_size must be greater than 0")

    if len(trace) <= ideal_size:
        return [trace]
    else:
        ts = [trace[i : i + ideal_size] for i in range(0, len(trace), ideal_size)]

        # check to make sure the last trace isn't too small
        if len(ts[-1]) <= 10:
            last_trace = ts.pop()
            ts[-1] = ts[-1] + last_trace

        return ts


def remove_bad_start_from_trace(trace: Trace, distance_threshold: float) -> Trace:
    """
    Remove leading points from a trace if there's a large gap at the beginning.

    This function detects and removes GPS points at the start of a trace that are
    separated by unusually large distances, which often indicates GPS initialization
    errors or teleportation artifacts. It scans from the beginning until it finds a
    pair of consecutive points within the distance threshold.

    Args:
        trace: The GPS trace to clean
        distance_threshold: The maximum acceptable distance (in the trace's CRS units, typically meters) between consecutive points. Points separated by more than this distance at the start are removed.

    Returns:
        A new Trace with leading outlier points removed. If no large gaps are found,
        returns a trace equivalent to the original.

    Examples:
        >>> # Remove points with gaps > 500 meters at the start
        >>> trace = Trace.from_gpx('gps_track.gpx')
        >>> cleaned = remove_bad_start_from_trace(trace, distance_threshold=500)
        >>> print(f"Removed {len(trace) - len(cleaned)} leading points")
        >>>
        >>> # Common use case: remove GPS initialization errors
        >>> # GPS might record (0, 0) or last known position before getting fix
        >>> trace = Trace.from_csv('raw_gps.csv')
        >>> trace = remove_bad_start_from_trace(trace, distance_threshold=1000)
    """

    def _trim_frame(frame):
        for index in range(len(frame)):
            rows = frame.iloc[index : index + 2]

            if len(rows) < 2:
                return frame

            current_point = rows.geometry.iloc[0]
            next_point = rows.geometry.iloc[1]

            if current_point != next_point:
                dist = current_point.distance(next_point)
                if dist > distance_threshold:
                    return frame.iloc[index + 1 :]
                else:
                    return frame

    return Trace.from_geo_dataframe(_trim_frame(trace._frame))
