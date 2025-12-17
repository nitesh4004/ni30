import streamlit as st
import pystac_client
import planetary_computer
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import fiona
from datetime import datetime
import tempfile
import os

# Enable KML support in fiona
fiona.drvsupport.supported_drivers['KML'] = 'rw'

# Page Configuration
st.set_page_config(page_title="Satellite AOI Explorer", layout="wide")

def get_catalog():
    """Opens the Planetary Computer STAC catalog."""
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

def get_tiles_url(item, asset_key="visual"):
    """Generates a tile URL for a STAC item."""
    collection = item.collection_id
    item_id = item.id
    return f"https://planetarycomputer.microsoft.com/api/data/v1/item/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}@1x?collection={collection}&item={item_id}&assets={asset_key}&format=png"

# --- UI Setup ---
st.title("🛰️ Satellite AOI Explorer (Nitesh Kumar)")
st.sidebar.header("Upload & Parameters")

# 1. KML Upload
uploaded_file = st.sidebar.file_uploader("Upload AOI (KML file)", type=['kml'])

# 2. Dataset Selection
dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    ["Sentinel-2 L2A", "Sentinel-1 GRD", "NAIP (USA Only)"]
)

collection_map = {
    "Sentinel-2 L2A": "sentinel-2-l2a",
    "Sentinel-1 GRD": "sentinel-1-grd",
    "NAIP (USA Only)": "naip"
}
collection_id = collection_map[dataset_choice]

# 3. Filters
max_cloud = st.sidebar.slider("Max Cloud Cover (%)", 0, 100, 20)
date_start = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
date_end = st.sidebar.date_input("End Date", datetime(2023, 12, 31))

# --- Processing ---
if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Read KML
        gdf = gpd.read_file(tmp_path, driver='KML')
        # Get Bounding Box for STAC search
        bbox = list(gdf.total_bounds) # [minx, miny, maxx, maxy]
        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2

        if st.sidebar.button("Search AOI Data"):
            catalog = get_catalog()
            
            # Formulate Search
            datetime_range = f"{date_start.strftime('%Y-%m-%d')}/{date_end.strftime('%Y-%m-%d')}"
            
            search_params = {
                "collections": [collection_id],
                "bbox": bbox,
                "datetime": datetime_range,
                "limit": 10
            }
            
            # Add cloud filter for Sentinel-2
            if collection_id == "sentinel-2-l2a":
                query = {"eo:cloud_cover": {"lt": max_cloud}}
                search = catalog.search(query=query, **search_params)
            else:
                search = catalog.search(**search_params)

            items = list(search.get_items())

            if items:
                # Pick the first item (usually most recent or least cloudy based on search)
                item = items[0]
                st.success(f"Found {len(items)} items. Displaying ID: {item.id}")

                # Select Asset based on collection
                asset_key = "visual"
                if collection_id == "naip": asset_key = "image"
                elif collection_id == "sentinel-1-grd": asset_key = "rendered_preview"

                tile_url = get_tiles_url(item, asset_key)

                # --- Map Display ---
                m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
                
                # Add AOI Polygon
                folium.GeoJson(gdf, name="Your AOI").add_to(m)
                
                # Add Satellite Tiles
                if tile_url:
                    folium.TileLayer(
                        tiles=tile_url,
                        attr="Planetary Computer",
                        name="Satellite Imagery",
                        overlay=True
                    ).add_to(m)

                folium.LayerControl().add_to(m)
                st_folium(m, width=1000, height=600)
                
            else:
                st.error("No imagery found for this AOI and date range.")
    except Exception as e:
        st.error(f"Error processing KML: {e}")
    finally:
        os.remove(tmp_path)
else:
    st.info("Please upload a KML file to begin.")
