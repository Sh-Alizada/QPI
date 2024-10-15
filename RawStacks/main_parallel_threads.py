import numpy as np
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from process_location_v3 import process_location
from input_excel_handler_v2 import read_exp_file
import utils_v2 as utils

####################################################

# This is for raw phase stacks. Use withKwargs Folder

####################################################

def process_location_wrapper(loc, loc_file, exp_folder_path, common_back, image_shape, conv_fac_to_rads,
                             bckg_params, watershed_params, imageprops_params, tracking_params,
                             reference_datetime, frame_range, maps_dict, treatment_conds_df,
                             metadata, raw_image_folder, image_folder, data_folder, 
                             image_mat_key, time_mat_key, file_pattern):
    """Wrapper function to pass multiple arguments to process_location."""
    process_location(loc, loc_file, exp_folder_path, common_back, 
                     image_shape=image_shape, conv_fac_to_rads=conv_fac_to_rads, 
                     bckg_params=bckg_params, watershed_params=watershed_params, 
                     imageprops_params=imageprops_params, tracking_params=tracking_params,
                     reference_datetime=reference_datetime, frame_range=frame_range,
                     maps_dict=maps_dict, treatment_conds_df=treatment_conds_df, 
                     ome_metadata=metadata, image_mat_key=image_mat_key, 
                     time_mat_key=time_mat_key, file_pattern=file_pattern, 
                     raw_image_folder=raw_image_folder, image_folder=image_folder, 
                     data_folder=data_folder)

def main():
    exp_folder_path = r"S:\Shukran\GH_NanoParticle_092424"

    excel_file_name = 'plateMap_NP_09.24.2024.xlsx'
    excel_file_path = os.path.join(exp_folder_path, excel_file_name)
    maps_dict, treatment_conds_df, exp_info_df = read_exp_file(excel_file_path, verbose=True)

    file_pattern = r'phase_loc_(?P<location>\d+)'

    raw_image_folder = 'DPCImages'
    image_folder = 'image_folder'
    data_folder = 'data_folder'

    conv_fac_to_rads = (4 * np.pi) / 32767

    common_back_file = 'common_back.mat'
    cb_mat_key = 'common_back'
    common_back = utils.load_mat_file(os.path.join(exp_folder_path, common_back_file), [cb_mat_key]).get(cb_mat_key)

    image_mat_key = 'Phase'
    time_mat_key = 'timestamp'
    # abs_mat_key = 'Absorption'

    location_range = [78, 93]  # None to process all of the available locations
    frame_range = None  # None to process all of the available frames for each location

    bckg_params = {
        "poly_order": 8,
        "poly_reduction": 8,
        "gauss_sigma": 1,
        "canny_low_thr": 5,
        "canny_high_thr": 30,
        "edge_dilate_kernel": (4, 4),
        "remove_size": 0,
        "mask_dilate_kernel": (5, 5)
    }

    watershed_params = {
       "gauss_sigma_1": 2,
       "gauss_sigma_2": 2,
       "canny_low_thr": 10, #5
       "canny_high_thr": 30,
       "edge_dilate_kernel": (5, 5),
       "remove_size": 250,
       "mask_dilate_kernel": (4, 4),
       "maxima_thr": 30,
       "maxima_dilate_kernel": (4, 4)
   }

    imageprops_params = {
       'wavelength': 624,
       'pixel_size': 5e-4,
       'min_MI_thr': 15, #15
       'max_MI_thr': 1000,
       'min_area_thr': 0,
       'max_area_thr': 100000
   }

    tracking_params = {
        'mass_factor': 1,
        'search_radius': 80,
        'tracking_memory': 1,
        'adaptive_step': 0.95,
        'adaptive_stop': 5,
    }

    # OME TIFF image metadata
    metadata = {'axes': 'TYX',
                'experiment_name': exp_info_df.get('experiment_name', 'N/A'),
                'experiment date': exp_info_df.get('date', 'N/A'),
                'experiment folder': exp_folder_path,
                'pixelsize (um)': imageprops_params.get('pixel_size', 'N/A'),
                'wavelength (um)': imageprops_params.get('wavelength', 'N/A'),
                'conversion_factor_to_rads': str(conv_fac_to_rads)
    }

    # Get all raw phase files
    all_files_df = utils.find_files(os.path.join(exp_folder_path, raw_image_folder), pattern=file_pattern, location=location_range)

    # Get image size
    image_shape = (1200, 1920)
    
    # Get reference time
    reference_datetime = (pd.to_datetime(exp_info_df['date'], format='%m/%d/%y') +
                          pd.to_timedelta(exp_info_df['dose_time'][0].strftime('%H:%M:%S')))[0]
   
    unique_locs = all_files_df['location'].unique()

    # Parallel processing using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = []
        for loc in unique_locs:
            loc_file = all_files_df['filename'][all_files_df['location'] == loc].item()

            # Submit each location to be processed in parallel
            futures.append(executor.submit(process_location_wrapper, loc, loc_file, exp_folder_path, common_back, 
                                           image_shape, conv_fac_to_rads, bckg_params, watershed_params, 
                                           imageprops_params, tracking_params, reference_datetime, frame_range, 
                                           maps_dict, treatment_conds_df, metadata, raw_image_folder, 
                                           image_folder, data_folder, image_mat_key, time_mat_key, file_pattern))

        # Optionally, collect and handle results
        for future in as_completed(futures):
            loc = None  # Track which location is causing the issue
            try:
                result = future.result()
                print(f"Location processed successfully: {result}")
            except Exception as e:
                loc = futures[futures.index(future)]  # Find the corresponding location
                print(f"Error processing location {loc}: {e}")
                traceback.print_exc()  # Print the full traceback

import time
if __name__ == "__main__":
    t0=time.perf_counter()
    main()
    t1=time.perf_counter()
    print(f'Total processing time for 16 cores: {t1-t0}')
