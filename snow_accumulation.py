# Import necessary libraries
import os
import numpy as np
import rasterio
import csv

# Change working directory
os.chdir('/Users/christinakrause/EAGLE/third_semester/Zugspitze/Final_Exam')

input_folder = "Snow_Depth"
output_csv = "Analysis/snow_volume_summary_DEM.csv"

def calculate_snow_volume_and_area(input_folder, output_csv):
    results = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            filepath = os.path.join(input_folder, filename)
            with rasterio.open(filepath) as src:
                data = src.read(1)
                pixel_area = abs(src.res[0] * src.res[1])  # m² per Pixel

                # Mask valid pixels with positive values
                mask = data > 0.1
                positive_pixels = data[mask]

                area = mask.sum() * pixel_area
                volume = np.sum(positive_pixels * pixel_area)

                results.append([filename, area, volume])
                print(f"{filename}: Area = {area:.2f} m², Volume = {volume:.2f} m³")

    # Write output csv
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Area_m²", "Volume_m³"])
        writer.writerows(results)


calculate_snow_volume_and_area(input_folder, output_csv)
