import numpy as np
import os
import psutil
import pandas as pd
from process_location_v3 import process_location
from input_excel_handler_v2 import read_exp_file
import utils_v2 as utils
from concurrent.futures import ProcessPoolExecutor

def set_cpu_affinity(core_id):
    """Set CPU affinity for the current process."""
    p = psutil.Process()  # Get the current process
    p.cpu_affinity([core_id])  # Set the process to only use the specified core

def process_location_wrapper(args):
    """Wrapper function for processing locations with CPU affinity."""
    loc, exp_folder_path, common_back, image_shape, conv_fac_to_rads, bckg_params, watershed_params, \
    imageprops_params, tracking_params, reference_datetime, frame_range, maps_dict, treatment_conds_df, \
    metadata, image_mat_key, time_mat_key, file_pattern, raw_image_folder, image_folder, data_folder, core_id = args

    set_cpu_affinity(core_id)  # Set process to run on the assigned core
    
    return process_location(
        loc, exp_folder_path, common_back, image_shape=image_shape, conv_fac_to_rads=conv_fac_to_rads,
        bckg_params=bckg_params, watershed_params=watershed_params, imageprops_params=imageprops_params,
        tracking_params=tracking_params, reference_datetime=reference_datetime, frame_range=frame_range,
        maps_dict=maps_dict, treatment_conds_df=treatment_conds_df, ome_metadata=metadata,
        image_mat_key=image_mat_key, time_mat_key=time_mat_key, file_pattern=file_pattern,
        raw_image_folder=raw_image_folder, image_folder=image_folder, data_folder=data_folder
    )

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

    location_range = [10, 19]
    frame_range = None

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

    metadata = {
        'axes': 'TYX',
        'experiment_name': exp_info_df.get('experiment_name', 'N/A'),
        'experiment date': exp_info_df.get('date', 'N/A'),
        'experiment folder': exp_folder_path,
        'pixelsize (um)': imageprops_params.get('pixel_size', 'N/A'),
        'wavelength (um)': imageprops_params.get('wavelength', 'N/A'),
        'conversion_factor_to_rads': str(conv_fac_to_rads)
    }

    all_files_df = utils.find_files(os.path.join(exp_folder_path, raw_image_folder), pattern=file_pattern, location=location_range)

    image_shape = (1200, 1920)

    reference_datetime = (pd.to_datetime(exp_info_df['date'], format='%m/%d/%y') +
                          pd.to_timedelta(exp_info_df['dose_time'][0].strftime('%H:%M:%S')))[0]

    unique_locs = all_files_df['location'].unique()

    # Prepare arguments for parallel processing and assign each process a unique core
    process_args = [
        (loc, exp_folder_path, common_back, image_shape, conv_fac_to_rads, bckg_params, watershed_params,
         imageprops_params, tracking_params, reference_datetime, frame_range, maps_dict, treatment_conds_df,
         metadata, image_mat_key, time_mat_key, file_pattern, raw_image_folder, image_folder, data_folder, i % os.cpu_count())
        for i, loc in enumerate(unique_locs)
    ]

    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(process_location_wrapper, process_args)

import time
if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    print(f'Total processing time: {t1 - t0}')
