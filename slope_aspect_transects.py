#
# This script takes slope and aspect of No-Snow DEM and plots slope and aspect along transect lines
#

# Import necessary libraries
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from osgeo import gdal
import os
import matplotlib.pyplot as plt
import seaborn as sns
gdal.UseExceptions()

# Change working directory
os.chdir('/Users/christinakrause/EAGLE/third_semester/Zugspitze/Final_Exam')

# Read data
slope_path = "October/20241025_dem_slope.tif"
aspect_path = "October/20241025_dem_aspect.tif"
transect = "earlyApril/transect_points_05m.gpkg"

gdf = gpd.read_file(transect)
gdf = gdf.to_crs('EPSG:32632')

# Extract coordinates
coords = [(x,y) for x,y in zip(gdf.geometry.x, gdf.geometry.y)]


# Extract slope (bilinear)
with rasterio.open(slope_path) as src:
    with WarpedVRT(src, resampling=Resampling.bilinear) as vrt:
        slope_vals = list(vrt.sample(coords))
        slope_vals = [val[0] if val and not np.isnan(val[0]) else None for val in slope_vals]

# Extract aspect (bilinear)
with rasterio.open(aspect_path) as src:
    with WarpedVRT(src, resampling=Resampling.bilinear) as vrt:
        aspect_vals = list(vrt.sample(coords))
        aspect_vals = [val[0] if val and not np.isnan(val[0]) else None for val in aspect_vals]

# In GeoDataFrame schreiben
gdf['slope_deg'] = slope_vals
gdf['aspect_deg'] = aspect_vals

# Aspect klassifizieren
def aspect_to_class(aspect):
    if aspect is None:
        return 'Flat'
    elif (aspect >= 337.5 or aspect < 22.5):
        return 'N'
    elif aspect < 67.5:
        return 'NE'
    elif aspect < 112.5:
        return 'E'
    elif aspect < 157.5:
        return 'SE'
    elif aspect < 202.5:
        return 'S'
    elif aspect < 247.5:
        return 'SW'
    elif aspect < 292.5:
        return 'W'
    else:
        return 'NW'

gdf['aspect_class'] = gdf['aspect_deg'].apply(aspect_to_class)

def classify_slope(slope_val):
    if slope_val is None:
        return 'Unknown'
    elif slope_val < 5:
        return 'Very flat'
    elif slope_val < 15:
        return 'Flat'
    elif slope_val < 30:
        return 'Medium'
    elif slope_val < 50:
        return 'Steep'
    elif slope_val < 100:
        return 'Very steep'
    else:
        return 'Extreme'

gdf['slope_class'] = gdf['slope_deg'].apply(classify_slope)

# Optional speichern
gdf.to_file("earlyApril/transect_points_with_slope_aspect.gpkg", layer="transect_slope_aspect", driver="GPKG")# Plot vorbereiten


unique_transects = gdf['transect_id'].unique()
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False, sharey=True)
axes = axes.flatten()

transect_name_map = {
    1: "Transect A",
    2: "Transect B",
    3: "Transect C",
    4: "Transect D"
}

# Mapping-Spalte hinzufügen
gdf['transect_label'] = gdf['transect_id'].map(transect_name_map)
# Aspect Class umbenennen
gdf = gdf.rename(columns={'aspect_class': 'Aspect class'})


sns.set(style="whitegrid", font_scale=1.1)

# Falls keine Distanz-Spalte existiert, berechne relative Distanz entlang jedes Transects
if 'dist' not in gdf.columns:
    gdf['dist'] = gdf.groupby('transect_id')['geometry'].apply(
        lambda g: g.distance(g.iloc[0])
    ).reset_index(drop=True)

unique_transects = gdf['transect_id'].unique()
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False, sharey=True)
axes = axes.flatten()

aspect_classes = sorted(gdf['Aspect class'].dropna().unique())
palette = sns.color_palette("Set2", len(aspect_classes))
palette_dict = dict(zip(aspect_classes, palette))

for i, transect in enumerate(unique_transects):
    ax = axes[i]
    sub = gdf[gdf['transect_id'] == transect].sort_values(by='dist')

    sns.scatterplot(
        data=sub,
        x='dist',
        y='slope_deg',
        hue='Aspect class',
        palette=palette_dict,
        ax=ax,
        s=40,
        edgecolor='k',
        linewidth=0.3,
    )

    ax.set_title(transect_name_map[transect])
    ax.set_xlabel("Distance along transect (m)")
    ax.set_ylabel("Slope (°)")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Aspect class", loc='upper right')

# Legende nach außen
plt.tight_layout()
plt.savefig("Analysis/20250402_slope_transects.png")

