import folium
from folium.plugins import MarkerCluster

def create_interactive_map(gdf, output_path='map.html'):
    """
    Create interactive Folium map with ship detections on satellite imagery
    
    Args:
        gdf: GeoDataFrame with ship detections
        output_path: Path to save HTML map
    """
    
    if len(gdf) == 0:
        print("  ⚠️  No detections to map")
        return
    
    # Calculate map center
    center_lat = gdf['lat'].mean()
    center_lon = gdf['lon'].mean()
    
    # Create map with Esri World Imagery (SATELLITE VIEW!)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery'
    )
    
    # Add layer control for switching between views
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Street Map',
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite',
        control=True,
        overlay=False
    ).add_to(m)
    
    # Add marker cluster for better performance
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add ship detections
    for idx, row in gdf.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8,
            popup=f"<b>Ship Detected</b><br>"
                  f"Confidence: {row['confidence']:.2f}<br>"
                  f"Lat: {row['lat']:.4f}<br>"
                  f"Lon: {row['lon']:.4f}",
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.7
        ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    m.save(output_path)
    print(f"  ✓ Saved interactive map: {output_path}")