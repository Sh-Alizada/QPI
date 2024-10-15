import os
import numpy as np
import pandas as pd
import utils_v2 as utils
from input_excel_handler_v2 import read_exp_file

class Inputs:
    def __init__(self):
        
        # Main Logicals
        self.tracking = True # True to perform cell tracking after processing
        self.QDF = False # True to get cell QDF signals
        
        
        # Define all the configuration parameters here
        self.exp_folder_path = r"S:\Shukran\GH_NanoParticle_092424"
        self.excel_file_name = 'plateMap_NP_09.24.2024.xlsx'
        self.file_pattern = r'QPM10x_092424__ph_loc_(?P<location>\d+)'
        self.raw_image_folder = 'DPCImages'
        self.image_folder = 'image_folder'
        self.data_folder = 'data_folder'
        self.conv_fac_to_rads = (4 * np.pi) / 32767 # FActor used to conver int16 to rads
        self.common_back_file = 'common_back.mat'
        self.cb_mat_key = 'common_back'
        self.frame_range = None # None for all available frames or [min, max]

        # Parameters for various image processing steps
        self.bckg_params = {
            "poly_order": 8, # Order of polynomial function fit to the background
            "poly_reduction": 10, # Factor by which the pixels are downsampled when fitting polynomial function 
            "gauss_sigma": 2, # Gaussian blurring of binary label (increase if there are a lot of noisy edges)
            "canny_low_thr": 5, # 
            "canny_high_thr": 25,
            "edge_dilate_kernel": (5, 5),
            "remove_size": 0,
            "mask_dilate_kernel": (5, 5)
        }

        self.watershed_params = {
           "gauss_sigma_1": 2, # Gaussian blurring of binary label (increase if there are a lot of noisy edges)
           "canny_low_thr": 10, # 
           "canny_high_thr": 30,
           "edge_dilate_kernel": (5, 5),
           "remove_size": 250,
           "mask_dilate_kernel": (4, 4),
           "gauss_sigma_2": 2, # Gaussian blurring of watershed label (increase if there are a lot of oversegmentation)
           "scale_factor": 0.5, # Factor by which an image is downsampled when h_maxima transform is performed (1 for no downsampling)
           "maxima_thr": 30, # Peak threshold above which the pixels are selected in h_maxima transform 
           "maxima_dilate_kernel": (4, 4) # 
        }

        self.imageprops_params = {
           'wavelength': 624,
           'pixel_size': 5e-4,
           'min_MI_thr': 15,
           'max_MI_thr': 1000,
           'min_area_thr': 0,
           'max_area_thr': 100000
        }

        self.tracking_params = {
            'mass_factor': 1,
            'search_radius': 80,
            'tracking_memory': 1,
            'adaptive_step': 0.95,
            'adaptive_stop': 5,
        }

    def load_excel_file(self):
        """Loads the Excel file and updates relevant attributes."""
        excel_file_path = os.path.join(self.exp_folder_path, self.excel_file_name)
        maps_dict, treatment_conds_df, exp_info_df = read_exp_file(excel_file_path, verbose=True)
        self.maps_dict = maps_dict
        self.treatment_conds_df = treatment_conds_df
        self.exp_info_df = exp_info_df

    def load_common_back(self):
        """Loads the common background file."""
        common_back_path = os.path.join(self.exp_folder_path, self.common_back_file)
        self.common_back = utils.load_mat_file(common_back_path, [self.cb_mat_key]).get(self.cb_mat_key)

    def prepare_metadata(self):
        """Prepares the metadata for processing."""
        self.metadata = {
            'axes': 'TYX',
            'experiment_name': self.exp_info_df.get('experiment_name', 'N/A'),
            'experiment date': self.exp_info_df.get('date', 'N/A'),
            'experiment folder': self.exp_folder_path,
            'pixelsize (um)': self.imageprops_params.get('pixel_size', 'N/A'),
            'wavelength (um)': self.imageprops_params.get('wavelength', 'N/A'),
            'conversion_factor_to_rads': str(self.conv_fac_to_rads)
        }
    
    def get_reference_datetime(self):
        """Gets the reference time for the experiment."""
        return (pd.to_datetime(self.exp_info_df['date'], format='%m/%d/%y') +
                pd.to_timedelta(self.exp_info_df['dose_time'][0].strftime('%H:%M:%S')))[0]

    def test_background_params(self):
        """Testing background parameters"""
        import matplotlib.pyplot as plt
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        set_matplotlib_formats('svg')
        from bckg_correction import bckg_correction
        
        # Which location to test
        test_raw_stack = 'QPM10x_092424__ph_loc_58.ome.tiff'
        # Which frame to test
        test_frame = 10
        
        file_path = os.path.join(self.exp_folder_path, self.raw_image_folder, test_raw_stack)
        image_stack, _ = utils.get_stack_and_t(file_path)
        
        raw_img = image_stack[test_frame-1]
        self.load_common_back() # Get common back
        bckg_corrected, binary_label = bckg_correction(raw_img, self.common_back, **self.bckg_params)
        
        plt.imshow(raw_img, interpolation='none')
        plt.title('Raw image')
        plt.show()
        
        plt.imshow(binary_label, interpolation='none')
        plt.title('binary_label')
        plt.show()
        
        plt.imshow(bckg_corrected, interpolation='none')
        plt.title('Bckg corrected image')
        plt.show()
        
        # TODO: test watershed params
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        