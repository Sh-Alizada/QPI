# import utils_v2 as utils
import numpy as np
from bckg_correction import bckg_correction
from canny_watershed_label import canny_watershed_label
from imageprops_v3 import imageprops
import my_decorators

@my_decorators.timer
def process_frame(raw_phase_image, common_back, conv_fac_to_rads=(4 * np.pi) / 32767,
                  bckg_params=None, watershed_params=None, imageprops_params=None):
    """
    Process a single frame by loading the image, applying background correction, 
    labeling the cells, and extracting cell properties.

    Parameters:
    - file_path (str): Path to the file containing the phase image.
    - common_back (numpy.ndarray): The common background image used for correction.
    - conversion_factor (float, optional): Conversion factor to convert raw phase image 
      to radians. Default is (2 * np.pi) / 65536.
    - bckg_params (dict, optional): Dictionary of parameters for background correction.
    - watershed_params (dict, optional): Dictionary of parameters for the watershed 
      cell labeling function.
    - imageprops_params (dict, optional): Dictionary of parameters for the image properties 
      extraction function.

    Returns:
    - int16_image (numpy.ndarray): Phase image converted back to int16 format.
    - uint16_label (numpy.ndarray): Labeled image with uint16 data type.
    - frame_props_df (DataFrame): DataFrame containing the extracted properties of the cells.
    - frame_datetime (float): The acquisition time of the frame.
    """
    
    if bckg_params is None:
        bckg_params = {}
    if watershed_params is None:
        watershed_params = {}
    if imageprops_params is None:
        imageprops_params = {}
        
    if raw_phase_image.std() == 0:  # Skip if blank image
        print('Blank Image')
        return 
    
    # Convert raw phase image to radians
    raw_phase_rads = raw_phase_image.astype(np.float64) * conv_fac_to_rads
    
    # Apply background correction
    phase_image, _ = bckg_correction(raw_phase_rads, common_back, **bckg_params)
    
    # Label the cells
    label_image = canny_watershed_label(phase_image, **watershed_params)
    
    # Extract cell properties for current frame
    frame_props_df = imageprops(phase_image, label_image, **imageprops_params)
    
    # Change data type to save space when saved
    int16_image = (phase_image / conv_fac_to_rads).astype(np.int16)
    
    uint16_label = label_image.astype(np.uint16)
    
    return int16_image, uint16_label, frame_props_df
    