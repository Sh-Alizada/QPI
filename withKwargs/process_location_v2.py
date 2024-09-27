import numpy  as np
import pandas as pd
import utils
from process_frame import process_frame
from input_excel_handler_v2 import get_loc_conditions
import os
import my_decorators

@my_decorators.timer
def process_location(location: int, exp_folder_path: str, common_back, 
                     image_shape=(1200,1920), conv_fac_to_rads=(4 * np.pi) / 32767, 
                     bckg_params=None, watershed_params=None, imageprops_params=None,
                     reference_datetime=None, frame_list_df=None, frame_range=None, 
                     maps_dict=None, add_conds_df=None, ome_metadata=None, 
                     image_mat_key='Phase', time_mat_key='timestamp',
                     file_pattern= r'pos(?P<location>\d+)_frame(?P<frame>\d+)',
                     raw_image_folder='DPCImages', image_folder='image_folder', data_folder='data_folder'):
    
    """
    Process a set of frames for a specific imaging location, including background correction, 
    cell labeling, and tracking. Save the results as OME-TIFF files and pixel sums in a CSV file.

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
    
    # Print some information about processing
    print(f'processing location {location}')
    
    #Raw phase image folder path
    raw_image_folder_path = os.path.join(exp_folder_path,raw_image_folder)
    
    #Get the list of raw phase images for current imaging location
    if frame_list_df is None:
        frame_list_df = utils.find_files(raw_image_folder_path, pattern=file_pattern, location=location, frame=frame_range)
       
    if reference_datetime is None:
        reference_serial_time = list(utils.load_mat_file(os.path.join(raw_image_folder_path, frame_list_df['filename'][0]), [time_mat_key]).values())[0]
        reference_datetime = utils.matlab_serial_to_datetime(reference_serial_time.item())
        
    #Number of frames in current imaging location
    num_of_frames=len(frame_list_df)
    
    #Initialize arrays
    image_stored = np.zeros([num_of_frames, image_shape[0], image_shape[1]], dtype=np.int16) # to store processed phase images
    label_stored = np.zeros([num_of_frames, image_shape[0], image_shape[1]], dtype=np.uint16) # to store cell label arrays
    frame_shift_stored = np.zeros([num_of_frames, 2]) # to store frame shift in pixels
    time_stored = np.zeros(num_of_frames) # to store frame times
    
    #Initialize dataframe to collect singe cell properties for current imageing location
    location_properties_df=pd.DataFrame()
    
    #Loop over the frames in current imaging location
    for frame_loop in range(num_of_frames):
        
        #Get file name and number
        frame_name = frame_list_df['filename'][frame_loop]
        frame_num = frame_list_df['frame'][frame_loop]
        
        # print(f'processing frame {frame_num}')
        
        #Construct file path
        frame_path = os.path.join(raw_image_folder_path,frame_name)
        
        int16_image, uint16_label, frame_properties_df, frame_datetime = process_frame(frame_path, 
                                                                                       common_back, 
                                                                                       bckg_params=bckg_params, 
                                                                                       watershed_params=watershed_params, 
                                                                                       imageprops_params=imageprops_params,
                                                                                       conv_fac_to_rads=conv_fac_to_rads)
         
        #Add frame number and time
        frame_time = (frame_datetime-reference_datetime).total_seconds() / 3600  #hours
        frame_properties_df.insert(loc=0, column='frame', value=frame_num) # add frame number
        frame_properties_df.insert(loc=1, column='time', value=frame_time) # add frame time
        
        #Find average shift between current frame and first frame
        if frame_loop == 0:
            phase_image_old = int16_image
        else:
            shift = utils.find_shift(phase_image_old, int16_image)  
            frame_shift_stored[frame_loop] = frame_shift_stored[frame_loop-1] + shift
            phase_image_old = int16_image
            
        #Adjust centroids for frame shift
        frame_properties_df['x'] = frame_properties_df['x'] - frame_shift_stored[frame_loop, 0] # adjust x coordinate
        frame_properties_df['y'] = frame_properties_df['y'] - frame_shift_stored[frame_loop, 1] # adjust y coordinate
        
        #Collect tracking data for each frame
        location_properties_df = pd.concat([location_properties_df, frame_properties_df], ignore_index=True) 
            
        #Store frame, time, phase image and label image
        time_stored[frame_loop] = round(frame_time, 3)
        image_stored[frame_loop, :, :] = int16_image
        label_stored[frame_loop, :, :] = uint16_label
    
    if maps_dict is not None:
        location_conds_df = get_loc_conditions(location, maps_dict, add_conds_df=add_conds_df)
        broadcasted_conds = location_conds_df.loc[location_conds_df.index.repeat(location_properties_df.shape[0])].reset_index(drop=True)
        location_properties_df = pd.concat([broadcasted_conds, location_properties_df], axis=1)
        
    #Save data and images
    image_folder_path = os.path.join(exp_folder_path,image_folder) # directory path
    os.makedirs(image_folder_path, exist_ok=True) # make the directory if not exists
    image_file_path = os.path.join(image_folder_path, f'processed_phase_loc_{location}.ome.tiff') # name the phase image file
    utils.save_ome_tiff(image_file_path, image_stored, metadata=ome_metadata) # save the phase images
    label_file_path = os.path.join(image_folder_path, f'processed_label_loc_{location}.ome.tiff') # name the label image file
    utils.save_ome_tiff(label_file_path, label_stored) # save the label images
    
    #Get pixels sums for overall mass calculations
    frames = frame_list_df['frame']
    pixel_sums = np.sum(image_stored.astype('float64'), axis=(1, 2)).reshape(-1,1) # sum pixels for each frame
    combined_pixel_data = np.column_stack((frames, time_stored, pixel_sums))
    
    #Save pixel sums
    data_folder_path = os.path.join(exp_folder_path,data_folder) # processed data folder
    os.makedirs(data_folder_path, exist_ok=True) # make the directory if not exists
    file_path_pixel = os.path.join(data_folder_path, f"pixel_sums_loc_{location}.csv") #assign pixel sums file name
    np.savetxt(file_path_pixel, combined_pixel_data, delimiter=',', header='Frame, Time(h), Pixel_Sum', comments='', fmt='%d,%.3f,%.3f') # save pixel sums
    
    
    
    
    return location_properties_df
    
    
    
    
    
    
    
    