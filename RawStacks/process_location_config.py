import numpy  as np
import pandas as pd
import utils_v2 as utils
from process_frame import process_frame
from input_excel_handler_v2 import get_loc_conditions
import os
import my_decorators
from inputs import Inputs

@my_decorators.timer
def process_location(location: int):
    
    """
    Process a set of frames for a specific imaging location, including background correction and 
    cell labeling. Save the results as OME-TIFF files and pixel sums in a CSV file.

    Parameters:
    - location (int): The imaging location to process.
    - exp_folder_path (str): Path to the experiment folder.
    - common_back (numpy.ndarray): The common background image used for correction.
    - exp_name (str): Name of the experiment.
    - exp_date (str): Date of the experiment.
    - bckg_params (dict, optional): Parameters for background correction.
    - watershed_params (dict, optional): Parameters for watershed cell labeling.
    - imageprops_params (dict, optional): Parameters for extracting image properties.
    - image_size (tuple, optional): Size of the images (height, width). Default is (1200, 1920).
    - reference_serial_time (float, optional): Serial time of the reference frame. If None, 
      the time is inferred from the first frame.
    - frame_range (tuple, optional): Range of frames to process (min_frame, max_frame). If None, 
      all frames are processed.
    - file_pattern (str, optional): Regex pattern for file naming. Default is 'pos(?P<location>\d+)_frame(?P<frame>\d+)'.
    - raw_image_folder (str, optional): Folder containing raw phase images. Default is 'DPCImages'.
    - image_folder (str, optional): Folder to save processed images. Default is 'image_folder'.
    - data_folder (str, optional): Folder to save data files. Default is 'data_folder'.

    Returns:
    - location_properties_df (pandas.DataFrame): DataFrame containing tracking data for the specified location.
    """
    
    # Create the inputs instance and load necessary data
    inputs = Inputs()
    inputs.load_excel_file()
    inputs.load_common_back()
    inputs.prepare_metadata()
    reference_datetime = inputs.get_reference_datetime()
    
    # Get inputs params
    exp_folder_path = inputs.exp_folder_path
    raw_image_folder = inputs.raw_image_folder
    file_pattern = inputs.file_pattern
    frame_range = inputs.frame_range
    maps_dict = inputs.maps_dict
    treatment_conds_df = inputs.treatment_conds_df
    common_back = inputs.common_back
    bckg_params = inputs.bckg_params
    watershed_params = inputs.watershed_params
    imageprops_params = inputs.imageprops_params
    conv_fac_to_rads = inputs.conv_fac_to_rads
    image_folder = inputs.image_folder
    ome_metadata = inputs.metadata
    data_folder = inputs.data_folder
    tracking = inputs.tracking
    tracking_params = inputs.tracking_params
    
    
    
    
    #Raw phase image folder path
    raw_image_folder_path = os.path.join(exp_folder_path,raw_image_folder)

    loc_file = utils.find_files(raw_image_folder_path, pattern=file_pattern, location=location)
    
    # Get file path
    loc_file_path = os.path.join(raw_image_folder_path, loc_file['filename'].item())
    
    # Get raw phase image stack and serial time list
    image_stack, serial_list = utils.get_stack_and_t(loc_file_path)
    
    #Check if frame range is assigned
    if frame_range is None:
        frame_range = [1, len(image_stack)]
    
    #Number of frames in current imaging location
    frames = list(range(frame_range[0], frame_range[1]+1))
    num_of_frames = len(frames)
    
    #Initialize image arrays
    image_stored = np.zeros([num_of_frames, image_stack.shape[1], image_stack.shape[2]], dtype=np.int16) # to store processed phase images
    label_stored = np.zeros([num_of_frames, image_stack.shape[1], image_stack.shape[2]], dtype=np.uint16) # to store cell label arrays
    frame_shift_stored = np.zeros([num_of_frames, 2]) # to store frame shift in pixels
    time_stored = np.zeros([num_of_frames, 1])
    
    #Initialize dataframe to collect singe cell properties for current imageing location
    location_properties_df=pd.DataFrame()
    
    # Get location conditions
    if maps_dict is not None:
        location_conds_df = get_loc_conditions(location, maps_dict, treatment_conds_df=treatment_conds_df)
        # Skip if treatment is 'Empty'
        if location_conds_df['treatment'].str.upper().item()=='EMPTY':
            return
    
    #Loop over the frames in current imaging location
    for frame_loop in range(num_of_frames):
        
        #Get file name and number
        frame_num = frames[frame_loop]
        
        print(f'processing frame {frame_num}')
        
        image_stored[frame_loop], label_stored[frame_loop], frame_properties_df = process_frame(image_stack[frame_num-1], 
                                                                                               common_back, 
                                                                                               bckg_params=bckg_params, 
                                                                                               watershed_params=watershed_params, 
                                                                                               imageprops_params=imageprops_params,
                                                                                               conv_fac_to_rads=conv_fac_to_rads)
        
        time_stored[frame_loop] = round((utils.matlab_serial_to_datetime(serial_list[frame_num-1])-reference_datetime).total_seconds() / 3600, 2)
        
        frame_properties_df.insert(loc=0, column='frame', value=frame_num) # add frame number
        frame_properties_df.insert(loc=1, column='time', value=time_stored[frame_loop].item()) # add frame time
        
        
        #Find average shift between current frame and first frame
        if frame_loop == 0:
            phase_image_old = image_stored[frame_loop]
        else:
            shift = utils.find_shift(phase_image_old, image_stored[frame_loop])  
            frame_shift_stored[frame_loop] = frame_shift_stored[frame_loop-1] + shift
            phase_image_old = image_stored[frame_loop]
        
        # If there's no cells found move to the next frame    
        if len(frame_properties_df) == 0:
            continue
        
        #Adjust centroids for frame shift
        frame_properties_df['x'] = frame_properties_df['x'] - frame_shift_stored[frame_loop, 0] # adjust x coordinate
        frame_properties_df['y'] = frame_properties_df['y'] - frame_shift_stored[frame_loop, 1] # adjust y coordinate
        
        #Collect tracking data for each frame
        location_properties_df = pd.concat([location_properties_df, frame_properties_df], ignore_index=True) 
            
    if maps_dict is not None:
        broadcasted_conds = location_conds_df.loc[location_conds_df.index.repeat(location_properties_df.shape[0])].reset_index(drop=True)
        location_properties_df = pd.concat([broadcasted_conds, location_properties_df], axis=1)
        
    #Save images
    image_folder_path = os.path.join(exp_folder_path, image_folder) # directory path
    os.makedirs(image_folder_path, exist_ok=True) # make the directory if not exists
    image_file_path = os.path.join(image_folder_path, f'phase_loc_{location}.ome.tiff') # name the phase image file
    utils.save_ome_tiff(image_file_path, image_stored, metadata=ome_metadata) # save the phase images
    label_file_path = os.path.join(image_folder_path, f'label_loc_{location}.ome.tiff') # name the label image file
    utils.save_ome_tiff(label_file_path, label_stored) # save the label images
    
    #Get pixels sums for overall mass calculations
    pixel_sums = np.sum(image_stored.astype('float64'), axis=(1, 2)).reshape(-1,1) # sum pixels for each frame
    
    #Combine frame data for current location
    location_info_df=pd.DataFrame(frames, columns=['frame'])
    location_info_df['time(h)'] = time_stored
    location_info_df['pixel_sums'] = pixel_sums
    location_info_df['frame_shift_x'] = frame_shift_stored[:,0]
    location_info_df['frame_shift_y'] = frame_shift_stored[:,1]
    if maps_dict is not None:
        location_info_df = location_info_df.join(location_conds_df, how='left')
    
    #Save pixel sums
    data_folder_path = os.path.join(exp_folder_path,data_folder) # processed data folder
    os.makedirs(data_folder_path, exist_ok=True) # make the directory if not exists
    info_file_path = os.path.join(data_folder_path, f"frame_data_loc_{location}.csv") #assign pixel sums file name
    location_info_df.to_csv(info_file_path, index=False) # save location info
    
    if tracking:
        utils.track_cells(location, location_properties_df, exp_folder_path, **tracking_params)
    
    
    
    
    
    
    
    