import multiprocessing
import numpy as np
import pandas as pd
import os
import utils_v2 as utils
from process_frame import process_frame
from input_excel_handler_v2 import get_loc_conditions
import my_decorators


# Move this function outside process_location
def process_single_frame(args):
    """
    Helper function to process a single frame for parallel execution.
    Needs to be outside of process_location to be picklable for multiprocessing.
    """
    frame_loop, frames, image_stack, common_back, bckg_params, watershed_params, imageprops_params, conv_fac_to_rads, serial_list, reference_datetime = args

    frame_num = frames[frame_loop]
    print(f'Processing frame {frame_num}')
    
    processed_image, processed_label, frame_properties_df = process_frame(image_stack[frame_num-1], 
                                                                          common_back, 
                                                                          bckg_params=bckg_params, 
                                                                          watershed_params=watershed_params, 
                                                                          imageprops_params=imageprops_params,
                                                                          conv_fac_to_rads=conv_fac_to_rads)

    time_value = round((utils.matlab_serial_to_datetime(serial_list[frame_num-1])-reference_datetime).total_seconds() / 3600, 2)

    frame_properties_df.insert(loc=0, column='frame', value=frame_num) # Add frame number
    frame_properties_df.insert(loc=1, column='time', value=time_value) # Add frame time

    return frame_loop, processed_image, processed_label, frame_properties_df, time_value


@my_decorators.timer
def process_location(location: int, loc_file: str, exp_folder_path: str, common_back, 
                     image_shape=(1200,1920), conv_fac_to_rads=(4 * np.pi) / 32767, 
                     bckg_params=None, watershed_params=None, imageprops_params=None, tracking_params=None,
                     reference_datetime=None, frame_range=None, tracking=True,
                     maps_dict=None, treatment_conds_df=None, ome_metadata=None, 
                     image_mat_key='Phase', time_mat_key='timestamp',
                     file_pattern= r'pos(?P<location>\d+)_frame(?P<frame>\d+)',
                     raw_image_folder='DPCImages', image_folder='image_folder', data_folder='data_folder'):

    """
    Process a set of frames for a specific imaging location, including background correction and 
    cell labeling. Save the results as OME-TIFF files and pixel sums in a CSV file.

    Parameters:
    (Same as before)

    Returns:
    - location_properties_df (pandas.DataFrame): DataFrame containing tracking data for the specified location.
    """
    
    # Raw phase image folder path
    raw_image_folder_path = os.path.join(exp_folder_path, raw_image_folder)

    # Get file path
    loc_file_path = os.path.join(raw_image_folder_path, loc_file)
    
    # Get raw phase image stack and serial time list
    image_stack, serial_list = utils.get_stack_and_t(loc_file_path)
    
    # Check if frame range is assigned
    if frame_range is None:
        frame_range = [1, len(image_stack)]
    
    # Number of frames in current imaging location
    frames = list(range(frame_range[0], frame_range[1]+1))
    num_of_frames = len(frames)
    
    # Initialize image arrays
    image_stored = np.zeros([num_of_frames, image_stack.shape[1], image_stack.shape[2]], dtype=np.int16) # to store processed phase images
    label_stored = np.zeros([num_of_frames, image_stack.shape[1], image_stack.shape[2]], dtype=np.uint16) # to store cell label arrays
    time_stored = np.zeros([num_of_frames, 1])
    
    # Initialize dataframe to collect single cell properties for current imaging location
    location_properties_df = pd.DataFrame()
    
    # Get location conditions
    if maps_dict is not None:
        location_conds_df = get_loc_conditions(location, maps_dict, treatment_conds_df=treatment_conds_df)
        # Skip if treatment is 'Empty'
        if location_conds_df['treatment'].str.upper().item() == 'EMPTY':
            return
    
    # Prepare arguments to pass to the multiprocessing pool
    args = [
        (frame_loop, frames, image_stack, common_back, bckg_params, watershed_params, imageprops_params, conv_fac_to_rads, serial_list, reference_datetime)
        for frame_loop in range(num_of_frames)
    ]

    # Parallelize frame processing using multiprocessing
    with multiprocessing.Pool(processes=10) as pool:
        results = pool.map(process_single_frame, args)

    # Handle the results from parallel processing
    for result in results:
        frame_loop, processed_image, processed_label, frame_properties_df, time_value = result
        image_stored[frame_loop] = processed_image
        label_stored[frame_loop] = processed_label
        time_stored[frame_loop] = time_value

        # If no cells found, skip the frame
        if len(frame_properties_df) == 0:
            continue

        # Collect tracking data for each frame
        location_properties_df = pd.concat([location_properties_df, frame_properties_df], ignore_index=True)
    
    if maps_dict is not None:
        broadcasted_conds = location_conds_df.loc[location_conds_df.index.repeat(location_properties_df.shape[0])].reset_index(drop=True)
        location_properties_df = pd.concat([broadcasted_conds, location_properties_df], axis=1)
    
    # Save images
    image_folder_path = os.path.join(exp_folder_path, image_folder) # Directory path
    os.makedirs(image_folder_path, exist_ok=True) # Make the directory if not exists
    image_file_path = os.path.join(image_folder_path, f'phase_loc_{location}.ome.tiff') # Name the phase image file
    utils.save_ome_tiff(image_file_path, image_stored, metadata=ome_metadata) # Save the phase images
    label_file_path = os.path.join(image_folder_path, f'label_loc_{location}.ome.tiff') # Name the label image file
    utils.save_ome_tiff(label_file_path, label_stored) # Save the label images
    
    # Get pixel sums for overall mass calculations
    pixel_sums = np.sum(image_stored.astype('float64'), axis=(1, 2)).reshape(-1,1) # Sum pixels for each frame
    
    # Combine frame data for current location
    location_info_df = pd.DataFrame(frames, columns=['frame'])
    location_info_df['time(h)'] = time_stored
    location_info_df['pixel_sums'] = pixel_sums

    if maps_dict is not None:
        location_info_df = location_info_df.join(location_conds_df, how='left')
    
    # Save pixel sums
    data_folder_path = os.path.join(exp_folder_path, data_folder) # Processed data folder
    os.makedirs(data_folder_path, exist_ok=True) # Make the directory if not exists
    info_file_path = os.path.join(data_folder_path, f"frame_data_loc_{location}.csv") # Assign pixel sums file name
    location_info_df.to_csv(info_file_path, index=False) # Save location info
    
    if tracking:
        utils.track_cells(location, location_properties_df, exp_folder_path, **tracking_params)
