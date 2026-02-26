"""
# LCSS Example

An example of using the LCSSMatcher to match a gps trace to the Open Street Maps road network
"""


def main():
    from mappymatch import package_root

    """
    First, we load the trace from a file. 
    The mappymatch package has a few sample traces included that we can use for demonstration.
    
    Before we build the trace, though, let's take a look at the file to see how mappymatch expects the input data:
    """

    import pandas as pd

    df = pd.read_csv(package_root() / "resources/traces/sample_trace_3.csv")
    df.head()

    """
    Notice that we expect the input data to be in the EPSG:4326 coordinate reference system. 
    If your input data is not in this format, you'll need to convert it prior to building a Trace object.
    
    In order to idenfiy which coordinate is which in a trace, mappymatch uses the dataframe index as the coordinate index and so in this case, we just have a simple range based index for each coordinate.
    We could set a different index on the dataframe and mappymatch would use that to identify the coordinates.
    
    Now, let's load the trace from the same file:
    """

    from mappymatch.constructs.trace import Trace

    trace = Trace.from_csv(
        package_root() / "resources/traces/sample_trace_3.csv",
        lat_column="latitude",
        lon_column="longitude",
        xy=True,
    )

    """
    Notice here that we pass three optional arguments to the `from_csv` function. 
    By default, mappymatch expects the latitude and longitude columns to be named "latitude" and "longitude" but you can pass your own values if needed.
    Also by default, mappymatch converts the trace into the web mercator coordinate reference system (EPSG:3857) by setting `xy=True`.
    The LCSS matcher computes the cartesian distance between geometries and so a projected coordiante reference system is ideal.
    In a future version of mappymatch we hope to support any projected coordiante system but right now we only support EPSG:3857.
    
    Okay, let's plot the trace to see what it looks like (mappymatch uses folium under the hood for plotting):
    """

    from mappymatch.utils.plot import plot_trace

    plot_trace(trace, point_color="black", line_color="yellow")

    """
    Next, we need to get a road map to match our Trace to.
    One way to do this is to build a small geofence around the trace and then download a map that just fits around our trace:
    """

    from mappymatch.constructs.geofence import Geofence

    geofence = Geofence.from_trace(trace, padding=2e3)

    """
    Notice that we pass an optional argument to the constructor.
    The padding defines how large around the trace we should build our geofence and is in the same units as the trace.
    In our case, the trace has been projected to the web mercator CRS and so our units would be in approximate meters, 1e3 meters or 1 kilomter 
    
    Now, let's plot both the trace and the geofence:
    """

    from mappymatch.utils.plot import plot_geofence

    plot_trace(trace, point_color="black", m=plot_geofence(geofence))

    """
    At this point, we're ready to download a road network.
    Mappymatch has a couple of ways to represent a road network: The `NxMap` and the `IGraphMap` which use `networkx` and `igraph`, respectively, under the hood to represent the road graph structure.
    You might experiment with both to see if one is more performant or memory efficient in your use case.
    
    In this example we'll use the `NxMap`:
    """

    from mappymatch.maps.nx.nx_map import NxMap, NetworkType

    nx_map = NxMap.from_geofence(
        geofence,
        network_type=NetworkType.DRIVE,
    )

    """
    The `from_geofence` constructor uses the osmnx package under the hood to download a road network.
    
    Notice we pass the optional argument `network_type` which defaults to `NetworkType.DRIVE` but can be used to get a different network like `NetworkType.BIKE` or `NetworkType.WALK`
    
    Now, we can plot the map to make sure we have the network that we want to match to:
    """

    from mappymatch.utils.plot import plot_map

    plot_map(nx_map)

    """
    Now, we're ready to perform the actual map matching. 
    
    In this example we'll use the `LCSSMatcher` which implements the algorithm described in this paper:
    
    [Zhu, Lei, Jacob R. Holden, and Jeffrey D. Gonder.
    "Trajectory Segmentation Map-Matching Approach for Large-Scale, High-Resolution GPS Data."
    Transportation Research Record: Journal of the Transportation Research Board 2645 (2017): 67-75.](https://doi.org/10.3141%2F2645-08)
    
    We won't go into detail here for how to tune the paramters but checkout the referenced paper for more details if you're interested. 
    The default parameters have been set based on internal testing on high resolution driving GPS traces. 
    """

    from mappymatch.matchers.lcss.lcss import LCSSMatcher

    matcher = LCSSMatcher(nx_map)

    match_result = matcher.match_trace(trace)

    """
    Now that we have the results, let's plot them:
    """

    from mappymatch.utils.plot import plot_matches

    plot_matches(match_result.matches)

    match_result.path_to_geodataframe().plot()

    """
    The `plot_matches` function plots the roads that each point has been matched to and labels them with the road id.
    
    In some cases, if the trace is much sparser (for example if it was collected a lower resolution), you might want see the estimated path, rather than the explict matched roads.
    
    For example, let's reduce the trace frequency to every 30th point and re-match it:
    """

    reduced_trace = trace[0::30]

    plot_trace(reduced_trace, point_color="black", line_color="yellow")

    reduced_matches = matcher.match_trace(reduced_trace)

    plot_matches(reduced_matches.matches)

    """
    The match result also has a `path` attribute with the estiamted path through the network:
    """

    from mappymatch.utils.plot import plot_path

    plot_trace(
        reduced_trace,
        point_color="blue",
        m=plot_path(reduced_matches.path, crs=trace.crs),
    )

    """
    Lastly, we might want to convert the results into a format more suitible for saving to file or merging with some other dataset. 
    To do this, we can convert the result into a dataframe:
    """

    result_df = reduced_matches.matches_to_dataframe()
    result_df.head()

    """
    Here, for each coordinate, we have the distance to the matched road, and then attributes of the road itself like the geometry, the OSM node id and the road distance and travel time.
    
    We can also get a dataframe for the path:
    """

    path_df = reduced_matches.path_to_dataframe()
    path_df.head()

    """
    Another thing we can do is to only get a certain set of road types to match to. For example, let's say I only want to consider highways and primary roads for matching, I can do so by passing a custom filter when building the road network: 
    """

    nx_map = NxMap.from_geofence(
        geofence,
        network_type=NetworkType.DRIVE,
        custom_filter='["highway"~"motorway|primary"]',
    )

    plot_map(nx_map)

    """
    Above you can see that now we have a much reduced graph to match to, let's see what happens
    """

    matcher = LCSSMatcher(nx_map)

    match_result = matcher.match_trace(trace)

    plot_matches(match_result.matches)

    """
    Plot the path
    """

    plot_path(match_result.path, crs=trace.crs)

    """
    Plot the geodataframe version of the path
    """

    path_gdf = match_result.path_to_geodataframe()

    path_gdf.plot()


if __name__ == "__main__":
    main()
