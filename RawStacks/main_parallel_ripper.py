import numpy as np
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from process_location_v3 import process_location
from input_excel_handler_v2 import read_exp_file
import utils_v2 as utils
import gc
import psutil

####################################################
# Optimized version with batched processing and core pinning for Windows
####################################################

def set_process_affinity(core_id):
    """Sets the process affinity to a specific core using psutil, compatible with Windows."""
    pid = os.getpid()
    p = psutil.Process(pid)
    p.cpu_affinity([core_id])  # Set affinity to the specified core

def process_location_wrapper(locations, loc_files, exp_folder_path, common_back, image_shape, conv_fac_to_rads,
                             bckg_params, watershed_params, imageprops_params, tracking_params,
                             reference_datetime, frame_range, maps_dict, treatment_conds_df,
                             metadata, raw_image_folder, image_folder, data_folder, 
                             image_mat_key, time_mat_key, file_pattern):
    """Wrapper function to process multiple locations and pass multiple arguments to process_location."""
    for loc, loc_file in zip(locations, loc_files):
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
    # Path setup and reading experiment files
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
    location_range = [64, 73]  # Location range to process
    frame_range = None  # Frame range to process

    # Parameters
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
       "canny_low_thr": 10,
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
       'min_MI_thr': 15,
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

    # Find files
    all_files_df = utils.find_files(os.path.join(exp_folder_path, raw_image_folder), pattern=file_pattern, location=location_range)

    # Image size and reference time
    image_shape = (1200, 1920)
    reference_datetime = (pd.to_datetime(exp_info_df['date'], format='%m/%d/%y') +
                          pd.to_timedelta(exp_info_df['dose_time'][0].strftime('%H:%M:%S')))[0]

    unique_locs = all_files_df['location'].unique()

    # Batch locations for processing
    num_cores = 10  # Set number of cores to use
    locations_per_batch = 2  # Set number of locations to process per batch (adjust as needed)

    # Prepare locations for processing in batches
    batched_locs = [unique_locs[i:i + locations_per_batch] for i in range(0, len(unique_locs), locations_per_batch)]
    batched_files = [[all_files_df['filename'][all_files_df['location'] == loc].item() for loc in batch] for batch in batched_locs]

    # Disable garbage collection during processing to reduce overhead
    gc.disable()

    # Parallel processing with core pinning
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for i, (locations, loc_files) in enumerate(zip(batched_locs, batched_files)):
            # Pin each batch to a different core using psutil
            set_process_affinity(i % psutil.cpu_count())  # Pin to a core
            futures.append(executor.submit(process_location_wrapper, locations, loc_files, exp_folder_path, common_back,
                                           image_shape, conv_fac_to_rads, bckg_params, watershed_params,
                                           imageprops_params, tracking_params, reference_datetime, frame_range,
                                           maps_dict, treatment_conds_df, metadata, raw_image_folder, image_folder,
                                           data_folder, image_mat_key, time_mat_key, file_pattern))

        # Collect results
        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"Batch processed successfully: {result}")
            except Exception as e:
                print(f"Error processing batch: {e}")

    # Enable garbage collection after processing
    gc.enable()


# Time the entire execution
import time
if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    print(f'Total processing time: {t1 - t0} seconds')
