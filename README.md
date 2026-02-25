# mappymatch

Mappymatch is a pure-python package developed and open sourced by the National Laboratory of the Rockies. It contains a collection of "Matchers" that enable matching a GPS trace (series of GPS coordinates) to a map.

![Map Matching Animation](docs/images/map-matching.gif?raw=true)

The current matchers are:

- `LCSSMatcher`: A matcher that implements the LCSS algorithm described in this [paper](https://doi.org/10.3141%2F2645-08). Works best with high resolution GPS traces.
- `OsrmMatcher`: A light matcher that pings an OSRM server to request map matching results. See the [official documentation](http://project-osrm.org/) for more info.
- `ValhallaMatcher`: A matcher to ping a [Valhalla](https://www.interline.io/valhalla/) server for map matching results.

Currently supported map formats are:

- OpenStreetMap

## Installation

```console
pip install mappymatch
```

If you have trouble with that, check out [the docs](https://nrel.github.io/mappymatch/install.html) for more detailed install instructions.

## Example Usage

The current primary workflow is to use [osmnx](https://github.com/gboeing/osmnx) to download a road network and match it using the `LCSSMatcher`.

The `LCSSMatcher` implements the map matching algorithm described in this paper:

[Zhu, Lei, Jacob R. Holden, and Jeffrey D. Gonder.
"Trajectory Segmentation Map-Matching Approach for Large-Scale, High-Resolution GPS Data."
Transportation Research Record: Journal of the Transportation Research Board 2645 (2017): 67-75.](https://doi.org/10.3141%2F2645-08)

usage:

```python
from mappymatch import package_root
from mappymatch.constructs.geofence import Geofence
from mappymatch.constructs.trace import Trace
from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.matchers.lcss.lcss import LCSSMatcher

trace = Trace.from_csv(package_root() / "resources/traces/sample_trace_1.csv")

# generate a geofence polygon that surrounds the trace; units are in meters;
# this is used to query OSM for a small map that we can match to
geofence = Geofence.from_trace(trace, padding=1e3)

# uses osmnx to pull a networkx map from the OSM database
nx_map = NxMap.from_geofence(geofence)

matcher = LCSSMatcher(nx_map)

matches = matcher.match_trace(trace)

# convert the matches to a dataframe
df = matches.matches_to_dataframe()
```

## Example Notebooks

Check out the [LCSS Example](https://nrel.github.io/mappymatch/lcss-example.html) for a more detailed example of working with the LCSSMatcher.
