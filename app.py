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
import numpy as np
from datetime import datetime
import tempfile
import os

# Enable KML support
fiona.drvsupport.supported_drivers['KML'] = 'rw'

st.set_page_config(page_title="Geospatial Analysis Hub", layout="wide")

def get_catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

def calculate_index(data, index_name):
    data = data.astype(float)
    if index_name == "NDVI (Vegetation)":
        return (data.sel(band="B08") - data.sel(band="B04")) / (data.sel(band="B08") + data.sel(band="B04"))
    elif index_name == "NDWI (Water)":
        return (data.sel(band="B03") - data.sel(band="B08")) / (data.sel(band="B03") + data.sel(band="B08"))
    elif index_name == "NDBI (Built-up)":
        return (data.sel(band="B11") - data.sel(band="B08")) / (data.sel(band="B11") + data.sel(band="B08"))
    return None

# --- UI Sidebar ---
st.title("🛰️ Satellite Index & Flood Analyzer (Nitesh Kumar)")
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("1. Upload AOI (KML)", type=['kml'])

dataset_choice = st.sidebar.selectbox(
    "2. Select Dataset",
    ["Sentinel-2 (Optical)", "Sentinel-1 (Radar/Flood)"]
)

if dataset_choice == "Sentinel-2 (Optical)":
    analysis_mode = st.sidebar.selectbox(
        "Select Index",
        ["True Color", "NDVI (Vegetation)", "NDWI (Water)", "NDBI (Built-up)"]
    )
else:
    analysis_mode = st.sidebar.selectbox("Select Mode", ["SAR Visualization", "Flood Mask (Threshold)"])
    flood_threshold = st.sidebar.slider("Flood Threshold (dB)", -30, -10, -18)

max_cloud = st.sidebar.slider("Max Cloud Cover (%) (S2 only)", 0, 100, 10)
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
            
            collection = "sentinel-2-l2a" if "Sentinel-2" in dataset_choice else "sentinel-1-grd"
            
            search_params = {
                "collections": [collection],
                "bbox": bbox,
                "datetime": f"{date_start.strftime('%Y-%m-%d')}/{date_end.strftime('%Y-%m-%d')}",
                "limit": 1
            }
            
            if collection == "sentinel-2-l2a":
                search_params["query"] = {"eo:cloud_cover": {"lt": max_cloud}}

            search = catalog.search(**search_params)
            items = list(search.get_items())

            if items:
                item = items[0]
                st.success(f"Processing Item: {item.id}")

                # Base Map
                m = folium.Map(location=center, zoom_start=14, tiles=None)
                folium.TileLayer(
                    tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                    attr='Google', name='Google Hybrid', overlay=False
                ).add_to(m)

                # --- Sentinel-2 Logic ---
                if collection == "sentinel-2-l2a":
                    if analysis_mode == "True Color":
                        tile_url = f"https://planetarycomputer.microsoft.com/api/data/v1/item/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}@1x?collection={collection}&item={item.id}&assets=visual&format=png"
                        folium.TileLayer(tiles=tile_url, attr="PC", name="S2 Visual").add_to(m)
                    else:
                        stack = stackstac.stack(item, assets=["B03", "B04", "B08", "B11"], bounds_latlon=bbox, epsg=4326, resolution=0.0001)
                        data = stack.compute().squeeze()
                        result = calculate_index(data, analysis_mode)
                        fig, ax = plt.subplots()
                        result.plot(ax=ax, cmap="RdYlGn" if "NDVI" in analysis_mode else "Blues")
                        st.pyplot(fig)

                # --- Sentinel-1 Flood Logic ---
                else:
                    # Sentinel-1 uses VV and VH bands. VV is better for water.
                    stack = stackstac.stack(item, assets=["vv"], bounds_latlon=bbox, epsg=4326, resolution=0.0001)
                    data = stack.compute().squeeze()
                    
                    if analysis_mode == "Flood Mask (Threshold)":
                        # Flood mask: Pixels below threshold are marked as 1 (Water)
                        flood_mask = xr.where(data < flood_threshold, 1, 0)
                        fig, ax = plt.subplots()
                        flood_mask.plot(ax=ax, cmap="Blues", add_colorbar=False)
                        ax.set_title("Identified Flooded Areas (Blue)")
                        st.pyplot(fig)
                    else:
                        fig, ax = plt.subplots()
                        data.plot(ax=ax, cmap="Greys_r")
                        ax.set_title("SAR Backscatter (VV)")
                        st.pyplot(fig)

                folium.GeoJson(gdf, name="AOI Boundary").add_to(m)
                st_folium(m, width=900, height=500)
            else:
                st.warning("No data found.")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
else:
    st.info("Upload a KML to begin.")
