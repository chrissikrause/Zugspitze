#
# This script compares in-situ Snow Depth measurements (GCPs and Transects) with UAV-derived Snow Depth (DEM)
#

# Import necessary libraries
import rasterio
from shapely.geometry import LineString
import numpy as np
import geopandas as gpd
import pandas as pd
import string
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from osgeo import gdal
import os
import matplotlib.pyplot as plt
gdal.UseExceptions()

# Change working directory
os.chdir('/Users/christinakrause/EAGLE/third_semester/Zugspitze/Final_Exam')

'''
Transect Analysis Function

This function creates lines from a shapefile containing transect start end end point, interpolates to given number of points along transect line
and extracts the corresponding Snow Depth values from a raster using bilinear interpolation

Parameter crs_epsg and raster input have to be the same metric CRS!
'''
def process_transects(
    input_gpkg,
    output_dir,
    measurements_xlsx,
    dem_path,
    num_interp_points,
    crs_epsg
):
    os.makedirs(output_dir, exist_ok=True)

    # Load transect start and end points
    transects = gpd.read_file(input_gpkg)
    transects = transects.to_crs(crs_epsg)

    # Create transect lines from start and end points
    lines = []
    for tid, group in transects.groupby("transect_id"):
        if len(group) != 2:
            print(f"Skipping {tid}, not exactly 2 points")
            continue
        points = list(group.geometry)
        line = LineString(points)
        lines.append({"transect_id": tid, "geometry": line})

    lines_gdf = gpd.GeoDataFrame(lines, crs=transects.crs)
    lines_gdf["length_m"] = lines_gdf.geometry.length
    lines_gdf.to_file(os.path.join(output_dir, "lines.gpkg"))

    # Interpolate lines to 23 points
    def interpolate_points(line, num_points=num_interp_points):
        return [line.interpolate(d, normalized=True) for d in np.linspace(0, 1, num_points)]

    interp_points = []
    for idx, row in lines_gdf.iterrows():
        points = interpolate_points(row.geometry)
        for i, pt in enumerate(points):
            interp_points.append({
                "transect_id": row["transect_id"],
                "point_id": i,
                "geometry": pt
            })

    points_gdf = gpd.GeoDataFrame(interp_points, crs=transects.crs)
    points_gdf.to_file(os.path.join(output_dir, "transect_points_05m.gpkg"))

    # Convert transect ids to letters (1→A, 2→B, 3→C, 4→D)
    id_to_letter = {i: letter for i, letter in enumerate(string.ascii_uppercase, start=1)}
    points_gdf['Transect'] = points_gdf['transect_id'].map(id_to_letter)
    points_gdf = points_gdf.drop(columns='transect_id')
    points_gdf = points_gdf.rename(columns={'point_id': 'Point'})

    # Join measurement data along transects with interpolated points
    measurements = pd.read_excel(measurements_xlsx)
    measurements["Measurement_1"] = measurements["Measurement_1"].replace(">280", 280)
    measurements["Measurement_1"] = pd.to_numeric(measurements["Measurement_1"], errors='coerce')

    merged_gdf = points_gdf.merge(measurements, on=["Transect", "Point"], how="left")

    # Get corresponding DEM values via bilinear interpolation
    with rasterio.open(dem_path) as src:
        with WarpedVRT(src, resampling=Resampling.bilinear) as vrt:
            coords = [(geom.x, geom.y) for geom in merged_gdf.geometry]
            sampled_vals = list(vrt.sample(coords))

    threshold = -1e-3  # minimal tolerance
    merged_gdf['Raster_Snow_Depth'] = [
        int(round(val[0] * 100)) if val and not np.isnan(val[0]) and val[0] > threshold else None
        for val in sampled_vals
    ]

    # Save transect measurements and DEM values
    merged_gdf.to_file(os.path.join(output_dir, "transect_dem_measurements.gpkg"))
    print("Processing complete.")

process_transects(input_gpkg='earlyApril/20250402_Zugspitze_Transectends_snowdepth.gpkg',
    output_dir='earlyApril',
    measurements_xlsx='earlyApril/transect_points_measurements.xlsx',
    dem_path='Snow_Depth/20250402_dem_difference_20241025_m.tif',
    num_interp_points=23,
    crs_epsg="EPSG:32632")


'''
Transect Analysis Plotting Function

This function plots the in-situ data along the transects versus the extracted snow septh values from the raster
'''
def plot_transect_snow_depth_comparison(merged_gdf_path, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    merged_gdf = gpd.read_file(merged_gdf_path)
    transects_to_plot = merged_gdf['Transect'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    for i, transect in enumerate(transects_to_plot):
        if i >= len(axes):
            print(f"Skipping Transect {transect} (more than 4)")
            continue

        ax = axes[i]
        df = merged_gdf[merged_gdf['Transect'] == transect].copy()

        # Ensure sorting by point index
        def point_sort_key(x):
            if isinstance(x, str) and x.startswith('start_'):
                return -1
            elif isinstance(x, str) and x.startswith('end_'):
                return 9999
            else:
                return int(x)

        df['sort_key'] = df['Point'].apply(point_sort_key)
        df = df.sort_values('sort_key')

        ax.plot(df['Point'], df['Measurement_1'], marker='o', label='Transect Point Snow Depth')
        ax.plot(df['Point'], df['Raster_Snow_Depth'], marker='x', linestyle='--', label='Raster Snow Depth')

        ax.set_title(f'Transect {transect}')
        ax.set_xlabel('Point')
        ax.set_ylabel('Snow Depth (cm)')
        ax.legend()
        ax.grid(True)

    # Hide unused subplots if fewer than 4 transects
    for j in range(len(transects_to_plot), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")

plot_transect_snow_depth_comparison(merged_gdf_path='earlyApril/transect_dem_measurements.gpkg', output_path='Analysis/insitu_vs_dem_transect_earlyApril.png')


'''
GCP Analysis Function

This function reads in snow depth data from GCPs and extracts corresponding snow depth values from a raster
using bilinear interpolation

The results are saved as xls

Snow Depth in GCP data has to be stored in column named 'snowDepth'
Parameter crs_epsg and raster input have to be the same metric CRS!
'''
#
# Read in gcp data and reproject to EPSG 32632
def compare_gcp_and_raster_snowdepth(
    gcp_path,
    raster_path,
    output_path,
    crs_epsg
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # GCP-Daten einlesen und reprojizieren
    gcp = gpd.read_file(gcp_path)
    print(f"Original CRS: {gcp.crs}")
    gcp_df = gcp.to_crs(crs_epsg)
    print(f"Reprojected CRS: {gcp_df.crs}")

    # Werte aus DEM (virtuelles Raster mit bilinearer Interpolation)
    with rasterio.open(raster_path) as src:
        with WarpedVRT(src, resampling=Resampling.bilinear) as vrt:
            coords = [(geom.x, geom.y) for geom in gcp_df.geometry]
            sampled_vals = list(vrt.sample(coords))

    # Zu Zentimeter konvertieren und als neue Spalte speichern
    gcp_df['Raster_Snow_Depth'] = [
        int(round(val[0]*100)) if val and val[0] is not None else None
        for val in sampled_vals
    ]

    # Vergleichs-DataFrame bereinigen
    comparison_df = gcp_df.dropna(subset=['snowDepth', 'Raster_Snow_Depth'])

    # Als GeoPackage speichern
    comparison_df.to_file(output_path, driver="GPKG")
    print(f"Comparison saved to: {output_path}")

    return comparison_df

compare_gcp_and_raster_snowdepth(
    gcp_path='earlyApril/20250402_Zugspitze_GCP_snowdepth.gpkg',
    raster_path='earlyApril/20250402_dem_difference_20241025_m.tif',
    output_path="earlyApril/gcp_vs_raster_comparison.gpkg",
    crs_epsg="EPSG:32632")