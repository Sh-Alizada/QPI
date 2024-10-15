import numpy as np
import scipy.io
import time
import os
from canny_simple_label import canny_simple_label
from scipy.signal import medfilt2d
import utils_v2 as utils


exp_folder_path = r"S:\Shukran\GH_NanoParticle_090624"

raw_image_folder='DPCImages'
folder = os.path.join(exp_folder_path, raw_image_folder)
file_pattern= r'pos(?P<location>\d+)_frame(?P<frame>\d+)'

# Number of frames to be selected randomly
select_random = 3000  # None if all frames are to be used, or an integer to randomly select a number of frames

# Location numbers to be used
location_range = [10, 150]  # None for all available locations, or an integer for a specific location, or a list to select a range of locations (e.g., [1, 100])

# Location frames to be used
frame_range = None  # None for all available frames, or an integer for a specific frame, or a list to select a range of frames (e.g., [1, 100])

# Conversion factor radians
conv_fac_to_rads = (4 * np.pi) / 32767

gauss_sigma=1
canny_low_thr=5
canny_high_thr=25
edge_dilate_kernel=(7,7)
remove_size=700
mask_dilate_kernel=(6,6)


# Common back image size
common_back = np.zeros((1200, 1920))

# Counter image size
c = np.zeros((1200, 1920))

# Get the files
all_files_df = utils.find_files(folder, pattern=file_pattern, location=location_range, frame=frame_range)

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
selected_files = all_files_df.iloc[selected_indices]['filename']


total_start_time = time.time()


for file in selected_files:
    try:
        #Construct file path
        frame_path = os.path.join(folder,file)
        
        #Load raw phase image
        raw_phase_image = utils.load_mat_file(frame_path, ['Phase']).get('Phase')
        
        #Convert raw phase image to nanometers from saved type 
        raw_phase_image = raw_phase_image.astype(np.float64) * conv_fac_to_rads
        
        #Apply median filter to reduce noise
        phase_image =  medfilt2d(raw_phase_image, kernel_size=3) # This takes 60% of total processing time
        
        #Get the background label
        bckg_label = ~canny_simple_label(phase_image,  gauss_sigma=gauss_sigma, canny_low_thr=canny_low_thr, 
                                         canny_high_thr=canny_high_thr, edge_dilate_kernel=edge_dilate_kernel, 
                                         remove_size=remove_size, mask_dilate_kernel=mask_dilate_kernel, clean_border=True).astype(bool)
        
        common_back[bckg_label] = (common_back[bckg_label] * c[bckg_label] + phase_image[bckg_label]) / (c[bckg_label] + 1)
        
        c[bckg_label] += 1
        
    except Exception as e:
        print(f'Error loading {file}: {e}')

total_end_time = time.time()
print(f'Total computation time: {total_end_time - total_start_time:.2f} seconds')


# Save the results to a single .mat file
scipy.io.savemat(os.path.join(exp_folder_path, 'common_back.mat'), {'common_back': common_back})
