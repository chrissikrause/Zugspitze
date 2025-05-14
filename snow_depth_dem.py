#
# This script is an analysis of LIDAR UAV drone data to map snow depth at Zugspitze, Germany
#

# Import necessary libraries
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.mask import mask
from osgeo import gdal
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker
from datetime import datetime
gdal.UseExceptions()

# Change working directory
os.chdir('/Users/christinakrause/EAGLE/third_semester/Zugspitze/Final_Exam')

## Function to open, reproject and resample target DEMs (early April '25, late April '25)
## and calculate DEM difference to reference DEM (Ocotber '24)
def reproject_raster(input_path, output_path, target_crs):
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': src.nodata if src.nodata is not None else -9999
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )


def clip_raster_with_aoi(input_path, output_path, aoi_path):
    with rasterio.open(input_path) as src:
        aoi = gpd.read_file(aoi_path).to_crs(src.crs)
        out_image, out_transform = mask(src, aoi.geometry, crop=True)
        nodata = src.nodata if src.nodata is not None else -9999
        out_image = np.where(out_image == nodata, np.nan, out_image)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": nodata
        })

        print("Clipped raster min:", np.nanmin(out_image), "max:", np.nanmax(out_image))

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)


def calculate_snow_depth(dem_snow_path, dem_nosnow_path, output_path):
    with rasterio.open(dem_snow_path) as snow_src, rasterio.open(dem_nosnow_path) as nosnow_src:
        snow_data = snow_src.read(1)
        nosnow_data = nosnow_src.read(1)

        nodata = snow_src.nodata if snow_src.nodata is not None else -9999
        snow_data = np.where(snow_data == nodata, np.nan, snow_data)
        nosnow_data = np.where(nosnow_data == nodata, np.nan, nosnow_data)

        snow_depth = snow_data - nosnow_data

        print("Snow depth min:", np.nanmin(snow_depth), "max:", np.nanmax(snow_depth))

        profile = snow_src.profile
        profile.update({'nodata': nodata})

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(np.where(np.isnan(snow_depth), nodata, snow_depth), 1)

# Process al 3 functions in one
def process_snow_depth_pipeline(
    snow_dem_path, nosnow_dem_path, aoi_path, target_crs,
    reproj_snow_path, reproj_nosnow_path,
    clipped_snow_path, clipped_nosnow_path,
    snow_depth_path
):
    reproject_raster(snow_dem_path, reproj_snow_path, target_crs)
    reproject_raster(nosnow_dem_path, reproj_nosnow_path, target_crs)

    clip_raster_with_aoi(reproj_snow_path, clipped_snow_path, aoi_path)
    clip_raster_with_aoi(reproj_nosnow_path, clipped_nosnow_path, aoi_path)

    calculate_snow_depth(clipped_snow_path, clipped_nosnow_path, snow_depth_path)

# Apply function
process_snow_depth_pipeline('20250423_Zugspitze/20250423_Zugspitze_100m_DJIM300L1_PPK_DSM.tif',
                            'October/20241025_Zugspitze_100m_DJIM300L1_PPK_DSM.tif',
                            'aoi/aoi_chrissi.shp',
                            'EPSG:32632',
                            '20250423_Zugspitze/20250423_DEM_32632.tif',
                            'October/October_DEM_32632.tif',
                            '20250423_Zugspitze/aoi_DEM_32632.tif',
                            'October/aoi_DEM_32632.tif',
                            'Snow_Depth/20250423_SnowDepth_aoi_32632_m.tif')


# Process February snow depth data (already calculated from DEM difference)
def process_snow_depth_pipeline_february(
    snow_dem_path, aoi_path, target_crs,
    reproj_snow_path,
    clipped_snow_path
):
    reproject_raster(snow_dem_path, reproj_snow_path, target_crs)
    clip_raster_with_aoi(reproj_snow_path, clipped_snow_path, aoi_path)

process_snow_depth_pipeline_february('February/20250225_Zugspitze_SnowDepth_PPK.tif',
                                     'aoi/aoi_chrissi.shp',
                                     'EPSG:32632',
                                     'February/February_SnowDepth_32632.tif',
                                     'Snow_Depth/20250225_SnowDepth_aoi_32632.tif'
                                     )

'''
Plotting section
'''
## Reproject Snow Depth maps for plotting
input_path = "Snow_Depth"
output_path = "Snow_Depth_Reprojected_4326"
os.makedirs(output_path, exist_ok=True)
target_crs = 'EPSG:4326'

for file in os.listdir(input_path):
    if file.endswith(".tif"):
        reproject_raster(os.path.join(input_path, file), os.path.join(output_path, file), "EPSG:4326")


# Plot Snow Depth Maps
input_path = "Snow_Depth_Reprojected_4326"
tif_files = sorted([f for f in os.listdir(input_path) if f.endswith(".tif")])[:3]

data_list = []
extent_list = []
titles = []

for tif_file in tif_files:
    with rasterio.open(os.path.join(input_path, tif_file)) as src:
        data = src.read(1).astype(np.float32)

        # Maskierung und Umrechnung (in cm)
        data = np.where((data <= 0) | (data > 2000), np.nan, data * 100)

        data_list.append(data)
        extent_list.append([
            src.bounds.left, src.bounds.right,
            src.bounds.bottom, src.bounds.top
        ])

        #Subplot titel extrahieren (Datum)
        basename = os.path.splitext(tif_file)[0]
        date_str = basename.split("_")[0]
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        formatted_date = date_obj.strftime("%d.%m.%Y")  # oder "%Y/%m/%d"
        titles.append(formatted_date)

# Gemeinsame Skala
vmin = np.nanmin([np.nanmin(d) for d in data_list])
vmax = np.nanmax([np.nanmax(d) for d in data_list])

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i in range(3):
    ax = axes[i]
    img = ax.imshow(
        data_list[i],
        cmap='Blues',
        extent=extent_list[i],
        origin='upper',
        norm=Normalize(vmin=vmin, vmax=vmax)
    )
    ax.set_title(titles[i], fontsize=12, fontweight='bold', loc='center')

    # Longitude (x-Achse oben, mit 2 Dezimalstellen)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Longitude')
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Latitude (nur im linken Subplot, 2 Dezimalstellen)
    if i == 0:
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_position('left')
        ax.set_ylabel('Latitude')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    else:
        ax.set_yticks([])  # Kein Y-Achse bei mittleren/rechten Subplots

# Colorbar unten, horizontal
fig.subplots_adjust(bottom=0.05)
cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
fig.colorbar(img, cax=cbar_ax, orientation='horizontal', label='Snow Depth (cm)')

plt.savefig('Analysis/SnowDepth_Comparison1.png', dpi=300)



## Histograms of snow depth
input_path = "Snow_Depth"
tif_files = sorted([f for f in os.listdir(input_path) if f.endswith(".tif")])

# Setup
n_files = len(tif_files)
fig, axes = plt.subplots(1, n_files, figsize=(6 * n_files, 5), tight_layout=True)

if n_files == 1:
    axes = [axes]  # sicherstellen, dass axes iterierbar bleibt

all_valid_values = []

# Daten und Statistiken sammeln
stats_list = []
for tif_file in tif_files:
    with rasterio.open(os.path.join(input_path, tif_file)) as src:
        data = src.read(1).astype(np.float32)
        data_cm = data * 100
        valid = data_cm[(data_cm > 0) & (data_cm <= 2000)]
        all_valid_values.append(valid)

        mean_val = np.nanmean(valid)
        median_val = np.nanmedian(valid)
        std_val = np.nanstd(valid)
        max_val = np.nanmax(valid)

        date_str = os.path.splitext(tif_file)[0].split("_")[0]
        try:
            date = datetime.strptime(date_str, "%Y%m%d").strftime("%d.%m.%Y")
        except:
            date = date_str

        stats_list.append((valid, date, mean_val, median_val, std_val, max_val))

# Gemeinsame x-Achsen-Grenzen (optional auch z. B. auf (0, 1200) begrenzen)
x_min = 0
x_max = 2000

# Histogramme plotten
for i, (valid, date, mean_val, median_val, std_val, max_val) in enumerate(stats_list):
    ax = axes[i]
    ax.hist(valid, bins=50, range=(x_min, x_max), color='skyblue', edgecolor='black')
    ax.set_title(f"{date}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Snow Depth (cm)")
    if i == 0:
        ax.set_ylabel("Pixel Count")
    else:
        ax.set_yticks([])  # Nur beim ersten Plot Y-Achse

    ax.set_xlim(x_min, x_max)

    # Statistik-Textbox
    stats_text = (
        f"Mean: {mean_val:.1f} cm\n"
        f"Median: {median_val:.1f} cm\n"
        f"Std Dev: {std_val:.1f} cm\n"
        f"Max: {max_val:.1f} cm\n"

    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.savefig("Analysis/SnowDepth_Histograms_Comparison.png", dpi=300)


'''
Snow Depth maps around transects
'''
aoi_transect = gpd.read_file("aoi/20250402_aoi_transects.gpkg")
print(aoi_transect.crs)
transect_lines = gpd.read_file("earlyApril/lines.gpkg")
print(transect_lines.crs)
# Reproject transect lines to target crs
if transect_lines.crs != aoi_transect.crs:
    transect_lines = transect_lines.to_crs(aoi_transect.crs)
print(transect_lines.crs)

# Open Tif and clip
with rasterio.open("Snow_Depth_Reprojected_4326/20250402_dem_difference_20241025_m.tif") as src:
    # Check if crs match
    if aoi_transect.crs != src.crs:
        aoi_transect = aoi_transect.to_crs(src.crs)

    # Clip no data vlaues
    clipped_image, clipped_transform = mask(
        dataset=src,
        shapes=aoi_transect.geometry,
        crop=True,
        nodata=src.nodata,  # wichtig
        filled=True
    )
    # Update metadata
    clipped_meta = src.meta.copy()
    clipped_meta.update({
        "height": clipped_image.shape[1],
        "width": clipped_image.shape[2],
        "transform": clipped_transform,
        "nodata": src.nodata,
        "count": clipped_image.shape[0],
        "dtype": clipped_image.dtype
    })

    # Save tif
    with rasterio.open("earlyApril/snow_depth_transect_aoi.tif", "w", **clipped_meta) as dest:
        dest.write(clipped_image)


# Plot
# Mask no data and convert snow depth from m to cm
clipped_data = clipped_image[0].astype(np.float32)
masked_data = np.where(
    (clipped_data == src.nodata) | (clipped_data < 0) | (clipped_data > 20),
    np.nan,
    clipped_data * 100
)


fig, ax = plt.subplots(figsize=(10, 8))
# Calculate extent from clipped image
extent = [
    clipped_transform[2],  # xmin
    clipped_transform[2] + clipped_transform[0] * masked_data.shape[1],  # xmax
    clipped_transform[5] + clipped_transform[4] * masked_data.shape[0],  # ymin
    clipped_transform[5]  # ymax
]
# Calculate range/norm
vmin = np.nanpercentile(masked_data, 2)
vmax = np.nanpercentile(masked_data, 98)

# Plot
img = ax.imshow(
    masked_data,
    cmap='Blues',
    extent=extent,
    origin='upper',
    norm=Normalize(vmin=vmin, vmax=vmax)
)

# Plot AOI and tra sect lines
aoi_transect.boundary.plot(ax=ax, edgecolor='grey', linewidth=1, label='AOI Transects')
transect_lines.plot(ax=ax, color='red', linewidth=1, label='Transect Lines')

# Set ax labels
ax.set_title("Snow Depth around Transects 02. April 2025", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

# Colorbar
cbar = plt.colorbar(img, ax=ax, orientation='vertical', label='Snow Depth (cm)')

# Add legend and save plot
ax.legend()
plt.tight_layout()
plt.savefig("Analysis/02042025_snowDepthDEM_transect_plot.png", dpi=300)
plt.close()





