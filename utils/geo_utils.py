import geopandas as gpd
from shapely.geometry import Point, box
import pandas as pd

def detections_to_geodataframe(detections, crs="EPSG:4326"):
    """
    Convert model detections to GeoDataFrame with geographic coordinates
    
    Args:
        detections: List of detection dicts with lat, lon, bbox, etc.
        crs: Coordinate reference system (default: WGS84)
    
    Returns:
        GeoDataFrame with Point geometries at ship locations
    """
    
    if len(detections) == 0:
        return gpd.GeoDataFrame()
    
    # Create points from lat/lon
    geometries = [Point(det['lon'], det['lat']) for det in detections]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(detections, geometry=geometries, crs=crs)
    
    return gdf

def export_to_shapefile(gdf, output_path):
    """Export GeoDataFrame to Shapefile (for ArcGIS)"""
    # Shapefile has column name limits (10 chars)
    # Rename columns if needed
    gdf_export = gdf.copy()
    
    # Keep only essential columns for shapefile
    columns_to_keep = ['class', 'confidence', 'lat', 'lon', 'geometry']
    gdf_export = gdf_export[[col for col in columns_to_keep if col in gdf_export.columns]]
    
    # Rename for shapefile compatibility
    gdf_export = gdf_export.rename(columns={
        'confidence': 'conf',
        'class': 'class_name'
    })
    
    gdf_export.to_file(output_path, driver='ESRI Shapefile')
    print(f"  ✓ Saved Shapefile: {output_path}")

def export_to_geojson(gdf, output_path):
    """Export GeoDataFrame to GeoJSON"""
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"  ✓ Saved GeoJSON: {output_path}")