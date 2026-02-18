from __future__ import annotations

import re
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, points_from_xy, read_file, read_parquet
from pyproj import CRS

from mappymatch.constructs.coordinate import Coordinate
from mappymatch.utils.crs import LATLON_CRS, XY_CRS


class Trace:
    """
    A collection of coordinates representing a GPS trajectory or path to be map-matched.

    A Trace wraps a GeoDataFrame of point geometries and provides methods for creating,
    manipulating, and transforming GPS trajectories. Traces are the primary input for
    map matching algorithms.

    The underlying GeoDataFrame must have unique indices - duplicate indices will raise
    an IndexError during initialization.

    Attributes:
        coords: A list of Coordinate objects representing each point in the trajectory
        crs: The coordinate reference system (CRS) of the trace
        index: The pandas Index from the underlying GeoDataFrame

    Examples:
        >>> import pandas as pd
        >>> from mappymatch.constructs.trace import Trace
        >>>
        >>> # Create from a DataFrame with lat/lon columns
        >>> df = pd.DataFrame({
        ...     'latitude': [40.7128, 40.7589, 40.7614],
        ...     'longitude': [-74.0060, -73.9851, -73.9776]
        ... })
        >>> trace = Trace.from_dataframe(df)
        >>>
        >>> # Create from a GPX file
        >>> trace = Trace.from_gpx('path/to/track.gpx')
        >>>
        >>> # Access coordinates
        >>> print(len(trace))  # Number of points
        >>> first_coord = trace.coords[0]
    """

    _frame: GeoDataFrame

    def __init__(self, frame: GeoDataFrame):
        if frame.index.has_duplicates:
            duplicates = frame.index[frame.index.duplicated()].values
            raise IndexError(
                f"Trace cannot have duplicates in the index but found {duplicates}"
            )
        self._frame = frame

    def __getitem__(self, i) -> Trace:
        if isinstance(i, int):
            i = [i]
        new_frame = self._frame.iloc[i]
        return Trace(new_frame)

    def __add__(self, other: Trace) -> Trace:
        if self.crs != other.crs:
            raise TypeError("cannot add two traces together with different crs")
        new_frame = pd.concat([self._frame, other._frame])
        return Trace(new_frame)

    def __len__(self):
        """Number of coordinate pairs."""
        return len(self._frame)

    def __str__(self):
        output_lines = [
            "Mappymatch Trace object",
            f"coords: {self.coords if hasattr(self, 'coords') else None}",
            f"frame: {self._frame}",
        ]
        return "\n".join(output_lines)

    def __repr__(self):
        return self.__str__()

    @property
    def index(self) -> pd.Index:
        """Get index to underlying GeoDataFrame."""
        return self._frame.index

    @cached_property
    def coords(self) -> List[Coordinate]:
        """
        Get all coordinates in the trace as Coordinate objects.

        This property constructs Coordinate objects from the underlying GeoDataFrame,
        preserving the index values as coordinate IDs. The result is cached for performance.

        Returns:
            A list of Coordinate objects, one for each point in the trace, ordered by the trace index
        """
        coords_list = [
            Coordinate(i, g, self.crs)
            for i, g in zip(self._frame.index, self._frame.geometry)
        ]
        return coords_list

    @property
    def crs(self) -> CRS:
        """Get Coordinate Reference System(CRS) to underlying GeoDataFrame."""
        return self._frame.crs

    @classmethod
    def from_geo_dataframe(
        cls,
        frame: GeoDataFrame,
        xy: bool = True,
    ) -> Trace:
        """
        Create a trace from a GeoPandas GeoDataFrame.

        The GeoDataFrame must contain a geometry column with Point geometries representing
        the GPS trajectory. Additional columns are discarded - only the geometry and index
        are retained.

        Args:
            frame: A GeoDataFrame with Point geometries representing the trajectory. Must have a valid CRS and unique index values.
            xy: If True, reproject the trace to Web Mercator (EPSG:3857) for distance calculations. If False, keep the original CRS. Default is True.

        Returns:
            A new Trace instance

        Examples:
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>>
            >>> # Create a GeoDataFrame with point geometries
            >>> gdf = gpd.GeoDataFrame(
            ...     geometry=[Point(-74.0060, 40.7128), Point(-73.9851, 40.7589)],
            ...     crs='EPSG:4326'
            ... )
            >>> trace = Trace.from_geo_dataframe(gdf)
        """
        # get rid of any extra info besides geometry and index
        frame = GeoDataFrame(geometry=frame.geometry, index=frame.index)
        if xy:
            frame = frame.to_crs(XY_CRS)
        return Trace(frame)

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        xy: bool = True,
        lat_column: str = "latitude",
        lon_column: str = "longitude",
    ) -> Trace:
        """
        Create a trace from a pandas DataFrame with latitude/longitude columns.

        This is one of the most common ways to create a Trace from GPS data. The DataFrame
        must contain columns with latitude and longitude values in WGS84 (EPSG:4326) format.

        Args:
            dataframe: A pandas DataFrame containing GPS coordinates in EPSG:4326 format
            xy: If True, reproject to Web Mercator (EPSG:3857) for accurate distance calculations. If False, maintain lat/lon coordinates. Default is True.
            lat_column: The name of the column containing latitude values. Default is "latitude".
            lon_column: The name of the column containing longitude values. Default is "longitude".

        Returns:
            A new Trace instance

        Examples:
            >>> import pandas as pd
            >>>
            >>> # Create from a DataFrame with default column names
            >>> df = pd.DataFrame({
            ...     'latitude': [40.7128, 40.7589, 40.7614],
            ...     'longitude': [-74.0060, -73.9851, -73.9776]
            ... })
            >>> trace = Trace.from_dataframe(df)
            >>>
            >>> # Use custom column names
            >>> df_custom = pd.DataFrame({
            ...     'lat': [40.7128, 40.7589],
            ...     'lon': [-74.0060, -73.9851]
            ... })
            >>> trace = Trace.from_dataframe(df_custom, lat_column='lat', lon_column='lon')
        """
        frame = GeoDataFrame(
            geometry=points_from_xy(dataframe[lon_column], dataframe[lat_column]),
            index=dataframe.index,
            crs=LATLON_CRS,
        )

        return Trace.from_geo_dataframe(frame, xy)

    @classmethod
    def from_gpx(
        cls,
        file: Union[str, Path],
        xy: bool = True,
    ) -> Trace:
        """
        Create a trace from a GPX (GPS Exchange Format) file.

        Parses GPX track data and extracts latitude/longitude coordinates from trackpoints.
        This method expects a simple GPX structure with a sequence of lat/lon coordinate pairs.

        Args:
            file: Path to the GPX file (as string or Path object)
            xy: If True, reproject to Web Mercator (EPSG:3857) for accurate distance calculations. If False, maintain lat/lon coordinates. Default is True.

        Returns:
            A new Trace instance with coordinates extracted from the GPX file

        Raises:
            FileNotFoundError: If the specified file does not exist
            TypeError: If the file does not have a .gpx extension

        Examples:
            >>> # Load a GPX track from a file
            >>> trace = Trace.from_gpx('morning_run.gpx')
            >>>
            >>> # Keep in lat/lon instead of projecting
            >>> trace_latlon = Trace.from_gpx('bike_ride.gpx', xy=False)
        """
        filepath = Path(file)
        if not filepath.is_file():
            raise FileNotFoundError(file)
        elif not filepath.suffix == ".gpx":
            raise TypeError(
                f"file of type {filepath.suffix} does not appear to be a gpx file"
            )
        data = open(filepath).read()

        lat_column, lon_column = "lat", "lon"
        lat = np.array(re.findall(r'lat="([^"]+)', data), dtype=float)
        lon = np.array(re.findall(r'lon="([^"]+)', data), dtype=float)
        df = pd.DataFrame(zip(lat, lon), columns=[lat_column, lon_column])
        return Trace.from_dataframe(df, xy, lat_column, lon_column)

    @classmethod
    def from_csv(
        cls,
        file: Union[str, Path],
        xy: bool = True,
        lat_column: str = "latitude",
        lon_column: str = "longitude",
    ) -> Trace:
        """
        Create a trace from a CSV file containing latitude/longitude coordinates.

        The CSV file must contain columns with latitude and longitude values in WGS84
        (EPSG:4326) format. The DataFrame index will be used as coordinate IDs.

        Args:
            file: Path to the CSV file (as string or Path object)
            xy: If True, reproject to Web Mercator (EPSG:3857) for accurate distance calculations. If False, maintain lat/lon coordinates. Default is True.
            lat_column: The name of the column containing latitude values. Default is "latitude".
            lon_column: The name of the column containing longitude values. Default is "longitude".

        Returns:
            A new Trace instance with coordinates from the CSV file

        Raises:
            FileNotFoundError: If the specified file does not exist
            TypeError: If the file does not have a .csv extension
            ValueError: If the specified lat/lon columns are not found in the CSV

        Examples:
            >>> # Load from CSV with default column names
            >>> trace = Trace.from_csv('gps_data.csv')
            >>>
            >>> # Load with custom column names
            >>> trace = Trace.from_csv('track.csv', lat_column='lat', lon_column='lng')
        """
        filepath = Path(file)
        if not filepath.is_file():
            raise FileNotFoundError(file)
        elif not filepath.suffix == ".csv":
            raise TypeError(
                f"file of type {filepath.suffix} does not appear to be a csv file"
            )

        columns = pd.read_csv(filepath, nrows=0).columns.to_list()
        if lat_column in columns and lon_column in columns:
            df = pd.read_csv(filepath)
            return Trace.from_dataframe(df, xy, lat_column, lon_column)
        else:
            raise ValueError(
                "Could not find any geometry information in the file; "
                "Make sure there are latitude and longitude columns "
                "[and provide the lat/lon column names to this function]"
            )

    @classmethod
    def from_parquet(cls, file: Union[str, Path], xy: bool = True):
        """
        Create a trace from a GeoParquet file.

        GeoParquet is a columnar storage format for geospatial data. The file must contain
        a geometry column with Point geometries and a valid CRS.

        Args:
            file: Path to the GeoParquet file (as string or Path object)
            xy: If True, reproject to Web Mercator (EPSG:3857) for accurate distance calculations. If False, maintain the original CRS. Default is True.

        Returns:
            A new Trace instance with coordinates from the GeoParquet file

        Examples:
            >>> # Load from a GeoParquet file
            >>> trace = Trace.from_parquet('trajectory.parquet')
        """
        filepath = Path(file)
        frame = read_parquet(filepath)

        return Trace.from_geo_dataframe(frame, xy)

    @classmethod
    def from_geojson(
        cls,
        file: Union[str, Path],
        index_property: Optional[str] = None,
        xy: bool = True,
    ):
        """
        Create a trace from a GeoJSON file containing Point features.

        The GeoJSON file should contain Point geometries representing the GPS trajectory.
        If index_property is specified, that property will be used as the DataFrame index;
        otherwise, all non-geometry properties will be combined to create the index.

        Args:
            file: Path to the GeoJSON file (as string or Path object)
            index_property: The name of a GeoJSON property to use as the DataFrame index. If None, all properties excluding geometry will be used as index columns. Default is None.
            xy: If True, reproject to Web Mercator (EPSG:3857) for accurate distance calculations. If False, maintain the original CRS. Default is True.

        Returns:
            A new Trace instance with coordinates from the GeoJSON file

        Examples:
            >>> # Load from GeoJSON, using all properties as index
            >>> trace = Trace.from_geojson('path.geojson')
            >>>
            >>> # Use a specific property as index
            >>> trace = Trace.from_geojson('points.geojson', index_property='timestamp')
        """
        filepath = Path(file)
        frame = read_file(filepath)
        if index_property and index_property in frame.columns:
            frame = frame.set_index(index_property)
        else:
            gname = frame.geometry.name
            index_cols = [c for c in frame.columns if c != gname]
            frame = frame.set_index(index_cols)

        return Trace.from_geo_dataframe(frame, xy)

    def downsample(self, npoints: int) -> Trace:
        """
        Downsample the trace to a specified number of evenly-spaced points.

        This method uses linear interpolation across the trace indices to select a subset
        of points that are approximately evenly distributed along the trajectory.

        Args:
            npoints: The target number of points in the downsampled trace

        Returns:
            A new Trace with approximately npoints evenly-distributed points

        Examples:
            >>> # Reduce a 1000-point trace to 100 points
            >>> long_trace = Trace.from_csv('detailed_track.csv')
            >>> print(len(long_trace))  # 1000
            >>> short_trace = long_trace.downsample(100)
            >>> print(len(short_trace))  # 100
        """
        s = list(np.linspace(0, len(self._frame) - 1, npoints).astype(int))

        new_frame = self._frame.iloc[s]

        return Trace(new_frame)

    def drop(self, index=List) -> Trace:
        """
        Remove points from the trace by their index values.

        This method creates a new trace with specified points removed. The index parameter
        should contain the DataFrame index values (not positional integers) of the points
        to remove.

        Args:
            index: A list of index values identifying the points to remove. These should be
                values from the trace's DataFrame index, not integer positions.

        Returns:
            A new Trace with the specified points removed

        Examples:
            >>> # Remove points with specific index values
            >>> trace = Trace.from_dataframe(df)  # df has index [0, 1, 2, 3, 4]
            >>> cleaned_trace = trace.drop([1, 3])  # Removes points at index 1 and 3
            >>> print(len(cleaned_trace))  # 3 (originally 5, removed 2)
        """
        new_frame = self._frame.drop(index)

        return Trace(new_frame)

    def to_crs(self, new_crs: CRS) -> Trace:
        """
        Transform the trace to a different coordinate reference system (CRS).

        This method reprojects all coordinates in the trace to the specified CRS.

        Args:
            new_crs: The target CRS. Can be a pyproj.CRS object, EPSG code string
                (e.g., 'EPSG:4326'), or any format accepted by pyproj.CRS()

        Returns:
            A new Trace with all coordinates transformed to the target CRS

        Examples:
            >>> # Convert from Web Mercator to WGS84 lat/lon
            >>> trace_xy = Trace.from_csv('data.csv', xy=True)  # In EPSG:3857
            >>> trace_latlon = trace_xy.to_crs('EPSG:4326')
            >>>
            >>> # Convert to a UTM zone
            >>> from pyproj import CRS
            >>> utm_crs = CRS('EPSG:32618')  # UTM Zone 18N
            >>> trace_utm = trace_latlon.to_crs(utm_crs)
        """
        new_frame = self._frame.to_crs(new_crs)
        return Trace(new_frame)

    def to_geojson(self, file: Union[str, Path]):
        """
        Write the trace to a GeoJSON file.

        This exports the trace as a GeoJSON FeatureCollection where each point is a Feature
        with Point geometry. The CRS information and any index data are preserved.

        Args:
            file: Path where the GeoJSON file should be written (as string or Path object)

        Examples:
            >>> trace = Trace.from_csv('input.csv')
            >>> trace.to_geojson('output.geojson')
        """
        self._frame.to_file(file, driver="GeoJSON")
