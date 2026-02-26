from dataclasses import dataclass
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from mappymatch.constructs.match import Match
from mappymatch.constructs.road import Road


@dataclass
class MatchResult:
    matches: List[Match]
    path: Optional[List[Road]] = None

    @property
    def crs(self):
        first_crs = self.matches[0].coordinate.crs
        if not all([first_crs.equals(m.coordinate.crs) for m in self.matches]):
            raise ValueError(
                "Found that there were different CRS within the matches. "
                "These must all be equal to use this function"
            )
        return first_crs

    def matches_to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Convert the matching results to a GeoDataFrame with point geometries.

        Each row represents one GPS coordinate and its matched road (or NA if no match).
        The CRS of the GeoDataFrame matches the CRS of the input trace.

        Returns:
            A GeoDataFrame with columns including:
            - geometry: The original coordinate point geometries
            - coordinate_id: The ID of each coordinate
            - road_id: The ID of the matched road (or NA)
            - distance_to_road: The distance to the matched road (or NA)
            - Additional columns from road metadata if available

        Examples:
            >>> result = matcher.match_trace(trace)
            >>> gdf = result.matches_to_geodataframe()
            >>> # Save to file
            >>> gdf.to_file('matches.geojson', driver='GeoJSON')
            >>> # Filter to successful matches
            >>> matched = gdf[gdf['road_id'].notna()]
        """
        df = self.matches_to_dataframe()
        gdf = gpd.GeoDataFrame(df, geometry="geom")

        if len(self.matches) == 0:
            return gdf

        gdf = gdf.set_crs(self.crs)

        return gdf

    def matches_to_dataframe(self) -> pd.DataFrame:
        """
        Convert the matching results to a pandas DataFrame.

        Similar to matches_to_geodataframe but without spatial functionality. Each row
        represents one GPS coordinate and its matched road.

        Returns:
            A pandas DataFrame with columns including:
            - coordinate_id: The ID of each coordinate
            - road_id: The ID of the matched road (or NaN)
            - distance_to_road: The distance to the matched road (or NaN)
            - Additional columns from road metadata if available

        Examples:
            >>> result = matcher.match_trace(trace)
            >>> df = result.matches_to_dataframe()
            >>> # Calculate matching statistics
            >>> match_rate = df['road_id'].notna().sum() / len(df)
            >>> avg_distance = df['distance_to_road'].mean()
        """
        df = pd.DataFrame([m.to_flat_dict() for m in self.matches])
        df = df.fillna(np.nan)

        return df

    def path_to_dataframe(self) -> pd.DataFrame:
        """
        Convert the matched path to a pandas DataFrame.

        The path represents the estimated route through the road network. If no path
        was computed (path is None), returns an empty DataFrame.

        Returns:
            A pandas DataFrame where each row represents one road segment in the path.
            Contains road IDs, geometries, and metadata. Returns empty DataFrame if no path exists.

        Examples:
            >>> result = matcher.match_trace(trace)
            >>> if result.path is not None:
            ...     path_df = result.path_to_dataframe()
            ...     print(f"Route has {len(path_df)} road segments")
            ...     total_length = path_df['length_miles'].sum()
        """
        if self.path is None:
            return pd.DataFrame()

        df = pd.DataFrame([r.to_flat_dict() for r in self.path])
        df = df.fillna(np.nan)

        return df

    def path_to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Convert the matched path to a GeoDataFrame with LineString geometries.

        The path represents the estimated route through the road network. If no path
        was computed (path is None), returns an empty GeoDataFrame.

        Returns:
            A GeoDataFrame where each row represents one road segment with its LineString geometry.
            Contains road IDs, geometries, and metadata. Returns empty GeoDataFrame if no path exists.
            The CRS matches the input trace CRS.

        Examples:
            >>> result = matcher.match_trace(trace)
            >>> if result.path is not None:
            ...     path_gdf = result.path_to_geodataframe()
            ...     # Save the matched route
            ...     path_gdf.to_file('matched_route.geojson', driver='GeoJSON')
            ...     # Visualize with matplotlib
            ...     path_gdf.plot()
        """
        if self.path is None:
            return gpd.GeoDataFrame()

        df = self.path_to_dataframe()
        gdf = gpd.GeoDataFrame(df, geometry="geom")

        gdf = gdf.set_crs(self.crs)

        return gdf
