from astropy.io import fits
import numpy as np
import os
import sep
import re
from pathlib import Path

def calculate_snr(fit_file, star_params):
    """Calculate SNR for a single star (implementation remains unchanged)"""
    hdu_list = fits.open(fit_file)
    image_data = hdu_list[0].data.astype(np.float32)
    background = sep.Background(image_data)
    
    x, y, a, b, theta = star_params
    yy, xx = np.indices(image_data.shape)
    
    x_rot = (xx - x) * np.cos(theta) + (yy - y) * np.sin(theta)  # Assuming theta in radians
    y_rot = -(xx - x) * np.sin(theta) + (yy - y) * np.cos(theta)
    
    star_mask = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1
    star_region = image_data[star_mask]
    
    signal = np.sum(star_region - background.back()[star_mask])
    noise_std = np.std(image_data[~star_mask] - background.back()[~star_mask])
    N = np.sum(star_mask)
    
    return signal / (np.sqrt(N) * noise_std)

def process_all_stars(txt_folder, fit_folder, save_filtered_folder):
    """Process files with SNR filtering and save results"""
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]
    Path(save_filtered_folder).mkdir(parents=True, exist_ok=True)

    for txt_file in txt_files:
        # Match corresponding FITS file 
        # match = re.search(r"image\d+", txt_file)
        fit_file = txt_file[:-4] 
        print(fit_file)
        # if not match:
        #     continue
            
        fit_path = Path(fit_folder) / (fit_file+ ".fit")
        if not fit_path.exists():
            print(f"Missing FITS file: {fit_path}")
            continue

        # Read star data (preserve all columns)
        txt_path = Path(txt_folder) / txt_file
        try:
            # stars = np.loadtxt(txt_path, comments='#', delimiter='\t', skiprows=1)
            stars = np.loadtxt(txt_path, comments='#', skiprows=1, delimiter='\t', usecols=(0, 1, 2, 3, 4))
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")
            continue
            
        if stars.size == 0:
            print(f"No stars in {txt_file}")
            continue
        if stars.ndim == 1:
            stars = stars.reshape(1, -1)

        # Calculate SNR for each star
        snr_values = []
        for star in stars:
            if star.shape[0] < 5:  # Validate parameter format
                print(f"Invalid star parameters in {txt_file}")
                continue
                
            try:
                snr = calculate_snr(str(fit_path), star[:5])
                snr_values.append(snr)
            except Exception as e:
                print(f"Error calculating SNR for {txt_file}: {e}")
                snr_values.append(0)  # Mark invalid entries

        # Filter stars based on SNR
        snr_array = np.array(snr_values)
        valid_mask = (snr_array >= 3) & (snr_array <= 20)
        filtered_stars = stars[valid_mask]

        # Save filtered results with original formatting
        if filtered_stars.size > 0:
            header = "x\ty\ta\tb\ttheta\t[other columns]"  # Modify based on your data
            save_path = Path(save_filtered_folder) / txt_file
            np.savetxt(save_path, filtered_stars, 
                      fmt="%.5f",  # Maintain original precision
                      delimiter="\t",
                      header=header,
                      comments='# ')
            print(f"Saved {len(filtered_stars)} stars to {save_path}")
        else:
            print(f"No valid stars in {txt_file}")

if __name__ == "__main__":
    # Configuration - modify these paths
    txt_dir = "/home/flh/datasets/LAMOST_new/dataset_ori/test/gt_norm"
    fits_dir = "/home/flh/datasets/LAMOST_new/images_1k"
    output_dir = "/home/flh/datasets/LAMOST_new/dataset_ori/test/snr_txt_filter"
    
    process_all_stars(txt_dir, fits_dir, output_dir)