"""Standard attribute key names used for NetworkX graph data structures.

These constants define the dictionary keys used to store road network data in NetworkX graphs.
Using consistent keys ensures compatibility between different map readers and the NxMap class.
"""

# Key for storing road geometry (LineString) in edge data dictionaries
DEFAULT_GEOMETRY_KEY = "geometry"

# Key for storing additional metadata (OSM tags, custom attributes) in edge data dictionaries
DEFAULT_METADATA_KEY = "metadata"

# Key for storing the CRS (Coordinate Reference System) in graph.graph dictionary
DEFAULT_CRS_KEY = "crs"
