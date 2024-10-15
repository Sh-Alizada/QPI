import numpy as np
import scipy.io
import time
import os
from tracking_utils import load_frame_and_time
from canny_simple_label_cb import canny_simple_label_cb
from scipy.signal import medfilt2d
import utils_v2 as utils


exp_folder_path = r"C:\Users\shukr\Desktop\NP_files_for_Python"

raw_image_folder='DPCImages'
file_pattern= r'pos(?P<location>\d+)_frame(?P<frame>\d+)'

# Number of frames to be selected randomly
select_random = None  # None if all frames are to be used, or an integer to randomly select a number of frames

# Location numbers to be used
location_range = None  # None for all available locations, or an integer for a specific location, or a list to select a range of locations (e.g., [1, 100])

# Location frames to be used
frame_range = None  # None for all available frames, or an integer for a specific frame, or a list to select a range of frames (e.g., [1, 100])

# Get the files
all_files_df = utils.find_files(os.path.join(exp_folder_path, raw_image_folder), pattern=file_pattern, location=location_range, frame=frame_range)

# Check if any files were found before proceeding
if all_files_df.empty:
    raise ValueError("No files found. Please check the file pattern, location range, or frame range.")

# Select random frames if specified, otherwise use all frames
if select_random is not None:
    if not isinstance(select_random, int):
        raise TypeError("select_random must be an integer or None")
    elif select_random > len(all_files_df):
        raise ValueError(f"select_random ({select_random}) is larger than the number of available files ({len(all_files_df)}).")
    else:
        # Select random indices
        selected_indices = np.random.choice(len(all_files_df), select_random, replace=False)
        selected_indices = sorted(selected_indices)
else:
    selected_indices = range(len(all_files_df))  # Use all files

# Proceed with selected files
selected_files = all_files_df.iloc[selected_indices]


conv_fac_to_rads = (2 * np.pi) / 65536

common_back = np.zeros((1200, 1920))

c = np.zeros((1200, 1920))

total_start_time = time.time()

for pp in numPos:

    print(f'Computing pos {pp}')
    frame_start_time = time.time()

    for ff in numFrames:
        fname = f'{froot}{fstart}pos{(pp):03d}_frame{(ff):03d}.mat'
        try:
            #Load current raw phase image and its time
            raw_phase_image, _ = load_frame_and_time(fname)
            
            #Convert raw phase image to nanometers from saved type (commonly int16) and apply background correction
            raw_phase_image = (raw_phase_image.astype(np.float64) * (2 * np.pi)) / 65536 * wavelength # nanometer
            
            #Apply median filter to reduce noise
            phase_image = medfilt2d(raw_phase_image, kernel_size=3)
            
            #Get the background label
            bckg_label = ~canny_simple_label_cb(phase_image).astype(bool)
            
            # D1, M = load_segment(fname, wavelength)
            common_back[bckg_label] = (common_back[bckg_label] * c[bckg_label] + phase_image[bckg_label]) / (c[bckg_label] + 1)
            c[bckg_label] += 1
            
        except Exception as e:
            print(f'Error loading {fname}: {e}')

    print(f'Finished frame {ff} in {time.time() - frame_start_time:.2f} seconds')

total_end_time = time.time()
print(f'Total computation time: {total_end_time - total_start_time:.2f} seconds')


# Save the results to a single .mat file
scipy.io.savemat(os.path.join(froot, 'common_back.mat'), {'common_back': common_back})
