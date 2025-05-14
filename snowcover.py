# # #
# This script calculates NDSI based on Green and NIR band of multispectral UAV imagery,
# masks out non-snow covered areas based on NDSI threshold (>0)
# calculates snow cover extent area within aoi
# and compares snow cover extent with DEM derived snow cover (DEM changes > 0)
# # #

import os
import geopandas as gpd
import pandas as pd
import glob
import rasterio
import numpy as np
import csv
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

# Change working directory
os.chdir('/Users/christinakrause/EAGLE/third_semester/Zugspitze/Final_Exam')

'''
This function reprojects multispectral data to metric crs and clips imagery with given AOI
'''
aoi = gpd.read_file("aoi/aoi_chrissi.shp")
aoi_32632 = aoi.to_crs('EPSG:32632')
print(aoi_32632.crs)
aoi.to_file("aoi/aoi_32632.shp")
def reproject_and_clip_tifs(input_folder, output_folder, aoi_path, target_epsg=32632):
    os.makedirs(output_folder, exist_ok=True)
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))

    # Read AOI geometry and ensure it's in the target CRS
    aoi = gpd.read_file(aoi_path)
    aoi = aoi.to_crs(epsg=target_epsg)
    aoi_geom = [feature["geometry"] for feature in aoi.__geo_interface__["features"]]

    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            print(f"Processing: {tif_file}")
            if src.crs is None:
                print(f"No CRS in {tif_file}, skipping.")
                continue

            transform, width, height = calculate_default_transform(
                src.crs, f"EPSG:{target_epsg}", src.width, src.height, *src.bounds)

            kwargs = src.meta.copy()
            kwargs.update({
                'crs': f"EPSG:{target_epsg}",
                'transform': transform,
                'width': width,
                'height': height
            })

            # Temporary reprojected image (in memory)
            with rasterio.MemoryFile() as memfile:
                with memfile.open(**kwargs) as temp_dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(temp_dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=f"EPSG:{target_epsg}",
                            resampling=Resampling.nearest
                        )

                    # Clip with AOI
                    clipped_data, clipped_transform = mask(temp_dst, aoi_geom, crop=True)

                    # Update metadata for clipped file
                    clipped_meta = temp_dst.meta.copy()
                    clipped_meta.update({
                        "height": clipped_data.shape[1],
                        "width": clipped_data.shape[2],
                        "transform": clipped_transform
                    })

                    output_path = os.path.join(output_folder, os.path.basename(tif_file))
                    with rasterio.open(output_path, "w", **clipped_meta) as dst:
                        dst.write(clipped_data)

                    print(f"Reprojected + clipped saved: {output_path}")

reproject_and_clip_tifs(
    input_folder="Multispectral/Input",
    output_folder="Multispectral/Reprojected_Clipped",
    aoi_path="aoi/aoi_32632.shp"
)

'''
These functions process NDSI calculation of clipped and reprojected multispectral imagery
'''
def calculate_ndsi(green_band, nir_band):
    # Convert to float and scale
    green = green_band.astype(np.float32)
    nir = nir_band.astype(np.float32)

    # Avoid division by zero and suppress warnings
    np.seterr(divide='ignore', invalid='ignore')
    ndsi = (green - nir) / (green + nir)
    return ndsi

def process_ndsi_from_tif(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))

    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            print("Band-Datentypen:", src.dtypes)
            print("Definierter NoData-Wert:", src.nodata)
            print(f"Processing: {tif_file}")
            green = src.read(2)
            nir = src.read(5)
            print("Green band stats:", green.min(), green.max())
            print("NIR band stats:", nir.min(), nir.max())
            ndsi = calculate_ndsi(green, nir)

            # Set metadata for output file
            meta = src.meta.copy()
            meta.update(dtype=rasterio.float32, count=1)

            output_filename = os.path.join(output_folder,
                                           os.path.splitext(os.path.basename(tif_file))[0] + "_NDSI.tif")

            with rasterio.open(output_filename, 'w', **meta) as dst:
                dst.write(ndsi, 1)

            print(f"NDSI saved to: {output_filename}")

# Apply function
process_ndsi_from_tif("Multispectral/Reprojected_Clipped", "Multispectral/NDSI")

## Reproject NDSI maps for plotting
input_path = "Multispectral/NDSI"
output_path = "Multispectral/NDSI_reprojected_4326"
os.makedirs(output_path, exist_ok=True)
dst_crs = 'EPSG:4326'

aoi = gpd.read_file("aoi/aoi_chrissi.shp")
if aoi.crs != dst_crs:
    aoi = aoi.to_crs(dst_crs)

shapes = aoi.geometry.values

for file in os.listdir(input_path):
    if file.endswith(".tif"):
        input_file = os.path.join(input_path, file)
        temp_file = os.path.join(output_path, f"_tmp_{file}")
        final_file = os.path.join(output_path, file)

        # Reprojection
        with rasterio.open(input_file) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(temp_file, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear
                    )

        # Clipping mit AOI
        with rasterio.open(temp_file) as src:
            out_image, out_transform = mask(src, shapes, crop=True)
            out_meta = src.meta.copy()

        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(final_file, "w", **out_meta) as dest:
            dest.write(out_image)

        os.remove(temp_file)

# Plot NDSI Maps
aoi = gpd.read_file("aoi/aoi_chrissi.shp")
aoi = aoi.to_crs("EPSG:4326")
input_path = "Multispectral/NDSI_reprojected_4326"
tif_files = sorted([f for f in os.listdir(input_path) if f.endswith(".tif")])[:3]

data_list = []
extent_list = []
titles = []

for tif_file in tif_files:
    with rasterio.open(os.path.join(input_path, tif_file)) as src:
        out_image, out_transform = mask(src, aoi.geometry, crop=True)
        data = out_image[0].astype(np.float32)

        # Mask data values outside [-1, 1]
        data = np.where((data < -1) | (data > 1), np.nan, data)

        # Mask NaN (transparent)
        masked_data = np.ma.masked_invalid(data)

        data_list.append(masked_data)
        extent_list.append([
            out_transform[2],
            out_transform[2] + masked_data.shape[1] * out_transform[0],
            out_transform[5] + masked_data.shape[0] * out_transform[4],
            out_transform[5]
        ])

        # Extract subplot title (date)
        basename = os.path.splitext(tif_file)[0]
        date_str = basename.split("_")[0]
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        formatted_date = date_obj.strftime("%d.%m.%Y")
        titles.append(formatted_date)

# Joined scala
vmin = np.nanmin([np.nanmin(d) for d in data_list])
vmax = np.nanmax([np.nanmax(d) for d in data_list])
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i in range(3):
    ax = axes[i]
    cmap = plt.get_cmap('RdBu')
    cmap.set_bad(color='white', alpha=0)
    img = ax.imshow(
        data_list[i],
        cmap=cmap,
        extent=extent_list[i],
        origin='upper',
        norm=norm
    )
    ax.set_title(titles[i], fontsize=12, fontweight='bold', loc='center')

    # Longitude (position at top of x-Axis, 2 Decimals)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Longitude')
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_position('left')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    # Latitude label (only in left subplots, 2 Decimals)
    if i == 0:
        ax.set_ylabel('Latitude')

# Create horizontal colorbar at bottom
fig.subplots_adjust(bottom=0.05)
cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
fig.colorbar(img, cax=cbar_ax, orientation='horizontal', label='NDSI')

plt.savefig('Analysis/NDSI_Comparison.png', dpi=300)



# Plot NDSI histograms
input_path = "Multispectral/NDSI"
tif_files = sorted([f for f in os.listdir(input_path) if f.endswith(".tif")])

n_files = len(tif_files)
fig, axes = plt.subplots(1, n_files, figsize=(6 * n_files, 5), tight_layout=True)

if n_files == 1:
    axes = [axes]

all_valid_values = []

# Get data and statistics
stats_list = []
for tif_file in tif_files:
    with rasterio.open(os.path.join(input_path, tif_file)) as src:
        data = src.read(1).astype(np.float32)
        valid = data[(data >= -1) & (data <= 1)]
        all_valid_values.append(valid)

        mean_val = np.nanmean(valid)
        median_val = np.nanmedian(valid)
        std_val = np.nanstd(valid)

        date_str = os.path.splitext(tif_file)[0].split("_")[0]
        try:
            date = datetime.strptime(date_str, "%Y%m%d").strftime("%d.%m.%Y")
        except:
            date = date_str

        stats_list.append((valid, date, mean_val, median_val, std_val))

# Joined axes
x_min = -1
x_max = 1

# Plot histograms
for i, (valid, date, mean_val, median_val, std_val) in enumerate(stats_list):
    ax = axes[i]
    ax.hist(valid, bins=50, range=(x_min, x_max), color='skyblue', edgecolor='black')
    ax.set_title(f"{date}", fontsize=11, fontweight='bold')
    ax.set_xlabel("NDSI")
    if i == 0:
        ax.set_ylabel("Pixel Count")
    else:
        ax.set_yticks([])

    ax.set_xlim(x_min, x_max)

    # Textbox with statistics
    stats_text = (
        f"Mean: {mean_val:.1f} \n"
        f"Median: {median_val:.1f} \n"
        f"Std Dev: {std_val:.1f}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.savefig("Analysis/NDSI_Histograms_Comparison.png", dpi=300)



'''
This function masks non-snow covered area based on NDSI threshold (0) and calculates snow cover extent area
'''
def mask_and_calculate_snow_area(ndsi_folder, output_folder, aoi_path, threshold=0, csv_output="snow_areas.csv"):
    os.makedirs(output_folder, exist_ok=True)
    ndsi_files = glob.glob(os.path.join(ndsi_folder, "*_NDSI.tif"))
    aoi = gpd.read_file(aoi_path)

    results = []

    for ndsi_file in ndsi_files:
        with rasterio.open(ndsi_file) as src:
            # Reproject AOI to raster crs
            aoi_proj = aoi.to_crs(src.crs)

            # Geometries to create masks
            geoms = [geom.__geo_interface__ for geom in aoi_proj.geometry]

            try:
                clipped_data, clipped_transform = mask(src, geoms, crop=True)
            except ValueError:
                print(f"Warning: No overlap of AOI with {ndsi_file}")
                continue

            ndsi_data = clipped_data[0]
            meta = src.meta.copy()
            meta.update({
                "height": ndsi_data.shape[0],
                "width": ndsi_data.shape[1],
                "transform": clipped_transform,
                "dtype": rasterio.uint8,
                "count": 1,
                "nodata": 0
            })

            # Binary snow mask: 1 = Snow, 0 = no Snow
            snow_mask = (ndsi_data >= threshold).astype(np.uint8)

            # Area caclulation
            snow_pixel_count = np.count_nonzero(snow_mask == 1)
            pixel_size = src.res[0] * src.res[1]
            snow_area_m2 = snow_pixel_count * pixel_size

            # Save mask results as new tifs
            base_name = os.path.splitext(os.path.basename(ndsi_file))[0]
            masked_filename = os.path.join(output_folder, base_name + "_snow_masked.tif")
            with rasterio.open(masked_filename, 'w', **meta) as dst:
                dst.write(snow_mask, 1)

            print(f"Snow mask saved to: {masked_filename}")
            print(f"Pixels: {snow_pixel_count}, Area: {snow_area_m2:.2f} m²")

            results.append({
                'filename': os.path.basename(ndsi_file),
                'snow_pixels': snow_pixel_count,
                'snow_area_m2': snow_area_m2
            })

    # Write csv with results
    csv_path = os.path.join(output_folder, csv_output)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'snow_pixels', 'snow_area_m2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Summary saved to: {csv_path}")


ndsi_input_folder = "Multispectral/NDSI"
masked_output_folder = "Multispectral/Snow_Cover"
aoi_path = "aoi/20250423_aoi_transects.gpkg"

mask_and_calculate_snow_area(ndsi_input_folder, masked_output_folder, aoi_path)


#
# Calculate Snow Cover extent from DEM substraction > 0 for smaller AOI
#
input_path = 'Snow_Depth'
output_path = 'Analysis'
output_file = os.path.join(output_path, 'DEM_SnowCover.xlsx')
aoi_path = "aoi/20250423_aoi_transects.gpkg"
aoi = gpd.read_file(aoi_path)

# List to store results
results = []

# Process all tifs in directory
for filename in os.listdir(input_path):
    if filename.endswith('.tif'):
        filepath = os.path.join(input_path, filename)
        with rasterio.open(filepath) as src:
            aoi_proj = aoi.to_crs(src.crs)
            geoms = [geom.__geo_interface__ for geom in aoi_proj.geometry]

            try:
                clipped_data, clipped_transform = mask(src, geoms, crop=True)
            except ValueError:
                print(f"No overlap with AOI in {filename}")
                continue

            data = clipped_data[0]
            pixel_size = src.res[0] * src.res[1]
            snow_mask = data > 0.1
            flaeche_m2 = np.sum(snow_mask) * pixel_size

            results.append({
                'Filename': filename,
                'Flaeche_m2': round(flaeche_m2, 3)
            })
# Save as gdf and xls
df = pd.DataFrame(results)
df.to_excel(output_file, index=False)

print(f"Results saved in: {output_file}")

