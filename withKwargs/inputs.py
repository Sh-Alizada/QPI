class ExpParams:
    def __init__(self, imageprops_params=None,
                 bckg_params=None, watershed_params=None, tracking_params=None):
        
        self.exp_folder=r'C:\Users\shukr\Desktop\NP_files_for_Python'
        
        self.common_back_file=r'C:\Users\shukr\Desktop\NP_files_for_Python\DPCImages\Btotal.mat'
        
        self.file_pattern = r'pos(?P<location>\d+)_frame(?P<frame>\d+)'
        
        self.imageprops_params = imageprops_params or {
            'wavelength': 624,
            'pixel_size': 5e-4,
            'min_MI_thr': 0,
            'max_MI_thr': 1000,
            'min_area_thr': 1,
            'max_area_thr': 100000,
        }
        self.bckg_params = bckg_params or {
            "poly_order": 8, 
            "poly_reduction": 10, 
            "gauss_sigma": 1.4, 
            "canny_low_thr": 1, 
            "canny_high_thr": 30, 
            "edge_dilate_kernel": (6, 6), 
            "remove_size": 400, 
            "mask_dilate_kernel": (4, 4)
        }
        self.watershed_params = watershed_params or {
            "gauss_sigma_1": 1.4, 
            "gauss_sigma_2": 1.4, 
            "canny_low_thr": 1, 
            "canny_high_thr": 30, 
            "edge_dilate_kernel": (6, 6), 
            "remove_size": 400, 
            "mask_dilate_kernel": (4, 4),
            "maxima_thr": 50, 
            "maxima_dilate_kernel": (4, 4)
        }
        
        self.tracking_params=tracking_params or {
            'tracking_memory':1,
            'search_radius': 30,
            'adaptive_search_factor': 0.8,
            'adaptive_stop': 10
            }

    def update_params(self, param_group, updates, inline=True):
        """
        Update multiple parameter values in a nested dictionary.
        
        :param param_group: The parameter group to be updated 
                            ('imageprops_params', 'background_params', 'watershed_params').
        :param updates: A dictionary containing key-value pairs to update in the selected parameter group.
        :param inline: If True, update the original parameter group. 
                       If False, return a new instance of the class with updated values.
        :return: The same ExpParams object with updated parameters.
        """
        if param_group in ['imageprops_params', 'background_params', 'watershed_params']:
            params_dict = getattr(self, param_group)
            
            if inline:
                # Update the original dictionary
                for key, new_value in updates.items():
                    if key in params_dict:
                        params_dict[key] = new_value
                    else:
                        print(f"Key '{key}' not found in '{param_group}'.")
            else:
                # Create a copy of the class object with updated values
                new_params = ExpParams(
                    imageprops_params=self.imageprops_params.copy(),
                    background_params=self.background_params.copy(),
                    watershed_params=self.watershed_params.copy()
                )
                updated_dict = getattr(new_params, param_group)
                for key, new_value in updates.items():
                    if key in updated_dict:
                        updated_dict[key] = new_value
                    else:
                        print(f"Key '{key}' not found in '{param_group}'.")
                return new_params
        
        return self
