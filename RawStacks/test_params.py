import numpy as np
import os
from input_excel_handler_v2 import read_exp_file
import utils_v2 as utils
import image_utils
from bckg_correction import bckg_correction
from process_frame import process_frame 


#------------------------------------------------------------------------------------------------------

exp_folder_path = r"V:\Tarek\GH_RadiationExp_091324"

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

# Get all raw phase files
all_files_df = utils.find_files(os.path.join(exp_folder_path, raw_image_folder), pattern=file_pattern)

#-----------------------------------------------------------------------------------------------------


test_loc = 140

test_frame = 70

file = all_files_df[(all_files_df['location']==test_loc) & (all_files_df['frame']==test_frame)]['filename'].item()

file_path = os.path.join(exp_folder_path, raw_image_folder, file)

#-----------------------------------------------------------------------------------------------------

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

raw_phase_image = utils.load_mat_file(file_path, [image_mat_key]).get(image_mat_key)

raw_phase_rads = raw_phase_image.astype(np.float64) * conv_fac_to_rads

phase_image, used_label = bckg_correction(raw_phase_rads, common_back, **bckg_params)

image_utils.overlay_outlines(phase_image, used_label)

#-----------------------------------------------------------------------------------------------------

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
    'pixel_size': 5.316e-4,
    'min_MI_thr': 15, #15
    'max_MI_thr': 1000,
    'min_area_thr': 0,
    'max_area_thr': 100000
}


int16_image, uint16_label, frame_props_df, frame_time=process_frame(file_path, common_back, conv_fac_to_rads=conv_fac_to_rads,
                  bckg_params=bckg_params, watershed_params=watershed_params, imageprops_params=imageprops_params)


phase_rads = int16_image.astype('float64')*conv_fac_to_rads

image_utils.annotate_props(phase_rads, uint16_label, **imageprops_params, prop='mass')






















