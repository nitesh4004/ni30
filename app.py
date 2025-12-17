import streamlit as st
import pystac_client
import planetary_computer
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import fiona
import stackstac
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
import os

# Enable KML support
fiona.drvsupport.supported_drivers['KML'] = 'rw'

st.set_page_config(page_title="Geospatial Index Analyzer", layout="wide")

def get_catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

def calculate_index(data, index_name):
    """Calculate Remote Sensing Indices using Sentinel-2 Bands."""
    # S2 Bands: B04=Red, B08=NIR, B03=Green, B11=SWIR
    if index_name == "NDVI (Vegetation)":
        return (data.sel(band="B08") - data.sel(band="B04")) / (data.sel(band="B08") + data.sel(band="B04"))
    elif index_name == "NDWI (Water)":
        return (data.sel(band="B03") - data.sel(band="B08")) / (data.sel(band="B03") + data.sel(band="B08"))
    elif index_name == "NDBI (Built-up)":
        return (data.sel(band="B11") - data.sel(band="B08")) / (data.sel(band="B11") + data.sel(band="B08"))
    return None

# --- UI Sidebar ---
st.title("🛰️ Satellite Index Analyzer (Nitesh Kumar)")
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload AOI (KML)", type=['kml'])

# Index Selection
analysis_mode = st.sidebar.selectbox(
    "Select Analysis / View",
    ["True Color (Visual)", "NDVI (Vegetation)", "NDWI (Water)", "NDBI (Built-up)"]
)

max_cloud = st.sidebar.slider("Max Cloud Cover (%)", 0, 100, 10)
date_start = st.sidebar.date_input("Start Date", datetime(2024, 1, 1))
date_end = st.sidebar.date_input("End Date", datetime(2024, 12, 31))

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        gdf = gpd.read_file(tmp_path, driver='KML')
        bbox = list(gdf.total_bounds)
        center = [gdf.centroid.y.iloc[0], gdf.centroid.x.iloc[0]]

        if st.sidebar.button("Run Analysis"):
            catalog = get_catalog()
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{date_start.strftime('%Y-%m-%d')}/{date_end.strftime('%Y-%m-%d')}",
                query={"eo:cloud_cover": {"lt": max_cloud}},
                limit=1
            )
            
            items = list(search.get_items())

            if items:
                item = items[0]
                st.success(f"Processing Item: {item.id}")

                # Create Map with Google Hybrid Basemap
                m = folium.Map(location=center, zoom_start=14, tiles=None)
                
                # Adding Google Hybrid Basemap
                folium.TileLayer(
                    tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                    attr='Google',
                    name='Google Hybrid',
                    overlay=False,
                    control=True
                ).add_to(m)

                # Process calculations if an Index is selected
                if analysis_mode != "True Color (Visual)":
                    # Load specific bands for calculation
                    bands = ["B03", "B04", "B08", "B11"]
                    stack = stackstac.stack(item, assets=bands, bounds_latlon=bbox)
                    merged = stack.compute()
                    
                    # Calculate index
                    result = calculate_index(merged.squeeze(), analysis_mode)
                    
                    # Plot and show as image (Simplified for Webapp)
                    fig, ax = plt.subplots()
                    result.plot(ax=ax, cmap="RdYlGn" if "NDVI" in analysis_mode else "Blues")
                    st.pyplot(fig)
                else:
                    # Show True Color using the Data API tiles
                    tile_url = f"https://planetarycomputer.microsoft.com/api/data/v1/item/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}@1x?collection=sentinel-2-l2a&item={item.id}&assets=visual&format=png"
                    folium.TileLayer(tiles=tile_url, attr="Planetary Computer", name="S2 Visual").add_to(m)

                folium.GeoJson(gdf, name="AOI Boundary").add_to(m)
                st_folium(m, width=900, height=500)
            else:
                st.warning("No imagery found for these parameters.")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
else:
    st.info("Please upload a KML file to begin calculation.")
