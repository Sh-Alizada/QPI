import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed

from process_location_v3 import process_location
from input_excel_handler_v2 import read_exp_file
from sgr_calculator_v5 import sgr_calculator
import utils_v2 as utils
import sgr_utils_v2

def main():
    exp_folder_path = r"W:\Tarek\GH_RadiationExp2_091624"

    excel_file_name = 'plateMap_96_BR_DrugTreatement.xlsx'
    excel_file_path = os.path.join(exp_folder_path, excel_file_name)
    maps_dict, treatment_conds_df, exp_info_df = read_exp_file(excel_file_path, verbose=True)

    file_pattern = r'pos(?P<location>\d+)_frame(?P<frame>\d+)'

    raw_image_folder = 'DPCImages'
    image_folder = 'image_folder'
    data_folder = 'data_folder'

    conv_fac_to_rads = (2 * np.pi) / 32768

    common_back_file = 'Btotal.mat'
    cb_mat_key = 'Btotal'
    common_back = utils.load_mat_file(os.path.join(exp_folder_path, common_back_file), [cb_mat_key]).get(cb_mat_key)/624

    image_mat_key = 'Phase'
    time_mat_key = 'timestamp'
    # abs_mat_key = 'Absorption'

    location_range = None  # None to process all of the available locations
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


    # Set the number of jobs (parallel workers)
    num_jobs = 24
    
    # Get all raw phase files
    all_files_df = utils.find_files(os.path.join(exp_folder_path, raw_image_folder), pattern=file_pattern, location=location_range, frame=frame_range)

    # Load the first frame and location to get image shape and initial time
    loaded_dict = utils.load_mat_file(os.path.join(exp_folder_path, raw_image_folder, all_files_df['filename'][0]), [image_mat_key, time_mat_key])
    image = loaded_dict[image_mat_key]
    initial_time = loaded_dict[time_mat_key]

    # Get image size
    image_shape = image.shape
    # Get reference time
    dose_time = exp_info_df.get('dose_time', None)

    if dose_time is not None:
        # Combine date and time to get reference time
        reference_datetime = (pd.to_datetime(exp_info_df['date'], format='%m/%d/%y') +
                              pd.to_timedelta(exp_info_df['dose_time'][0].strftime('%H:%M:%S')))[0]
    else:
        reference_datetime = utils.matlab_serial_to_datetime(initial_time.item())

    # Function to process each location
    def process_single_location(loc):
        
        print(f'Process location {loc}')
        frame_list_df = all_files_df[all_files_df['location'] == loc].reset_index(drop=True)
        
        # Process the location and extract properties
        loc_properties_df, image_stored = process_location(loc, exp_folder_path, common_back, 
                                             image_shape=image_shape, conv_fac_to_rads=conv_fac_to_rads, 
                                             bckg_params=bckg_params, watershed_params=watershed_params, 
                                             imageprops_params=imageprops_params, reference_datetime=reference_datetime, 
                                             frame_list_df=frame_list_df, frame_range=frame_range,
                                             maps_dict=maps_dict, treatment_conds_df=treatment_conds_df, 
                                             ome_metadata=metadata, image_mat_key=image_mat_key, 
                                             time_mat_key=time_mat_key, file_pattern=file_pattern, 
                                             raw_image_folder=raw_image_folder, image_folder=image_folder, 
                                             data_folder=data_folder)
        
        # Perform cell tracking
        utils.track_cells(loc, loc_properties_df, exp_folder_path, **tracking_params)

    # Process all locations in parallel using joblib
    unique_locs = all_files_df['location'].unique()


    Parallel(n_jobs=num_jobs)(delayed(process_single_location)(loc) for loc in unique_locs)

    # Combine all the location data (if needed)
    all_cell_data = utils.concat_track_files(os.path.join(exp_folder_path, data_folder))

    # Calculate SGRs
    cell_sgr_df, well_sgr_df, conc_sgr_df = sgr_calculator(all_cell_data, exp_folder_path, data_folder=data_folder)

    # EC50 params
    ec50_params_df = sgr_utils_v2.EC50_params(well_sgr_df, p_cutoff=0.01)

    # Save EC50 figures
    sgr_utils_v2.plot_EC50_curves(conc_sgr_df, ec50_params_df, save_fig=True, exp_folder_path=exp_folder_path, data_folder=data_folder)

    # Save Violins
    sgr_utils_v2.plot_violins(cell_sgr_df, treatment_conds_df=treatment_conds_df, 
                              save_fig=True, exp_folder_path=exp_folder_path, data_folder=data_folder)

    # Save SGR map (saved in experiment excel files as additional sheet)
    sgr_utils_v2.make_sgr_map(well_sgr_df, maps_dict['well_num'], excel_file_path)

if __name__ == "__main__":
    main()
