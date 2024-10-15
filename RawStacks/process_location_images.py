import numpy  as np
import pandas as pd
import tracking_utils
from process_frame import process_frame
import os
import my_decorators

@my_decorators.timer
def process_location(location: int, exp_folder_path: str, common_back, exp_name='Np', exp_date='y24_d08_m08', image_size=(1200,1920),
                     bckg_params=None, watershed_params=None, imageprops_params=None,
                     reference_serial_time=None, frame_range=None,
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
    
    
    if bckg_params is None:
        bckg_params = {}
    if watershed_params is None:
        watershed_params = {}
    if imageprops_params is None:
        imageprops_params = {}    
    
    #Raw phase image folder
    raw_image_folder_path = os.path.join(exp_folder_path,raw_image_folder)
    
    #Get the list of raw phase images for current imaging location
    if frame_range is not None:
        frame_list = tracking_utils.find_files(raw_image_folder_path, pattern=file_pattern, location=location, frame=frame_range)
    else:
        frame_list = tracking_utils.find_files(raw_image_folder_path, pattern=file_pattern, location=location)
    
    if reference_serial_time is None:
        _ , reference_serial_time = tracking_utils.load_frame_and_time(os.path.join(raw_image_folder_path,frame_list[0][0]))

    #Number of frames in current imaging location
    num_of_frames=len(frame_list)
    
    #Initialize arrays
    image_stored = np.zeros([num_of_frames, image_size[0], image_size[1]], dtype=np.int16) # to store processed phase images
    label_stored = np.zeros([num_of_frames, image_size[0], image_size[1]], dtype=np.uint16) # to store cell label arrays
    frame_shift_stored = np.zeros([num_of_frames, 2]) # to store frame shift in pixels
    time_stored = np.zeros(num_of_frames) # to store frame times
    
    #Initialize dataframe to collect singe cell properties for current imageing location
    location_properties_df=pd.DataFrame()
    
    
    #Loop over the frames in current imaging location
    for frame_loop in range(num_of_frames):
        
        print(f'frame {frame_loop}')
        
        #Get file name and number
        frame_name, _, frame_number = frame_list[frame_loop]
        
        #Construct file path
        frame_path = os.path.join(raw_image_folder_path,frame_name)
        
        int16_image, uint16_label, frame_properties_df, frame_serial_time = process_frame(frame_path, 
                                                                                          common_back, 
                                                                                          bckg_params=bckg_params, 
                                                                                          watershed_params=watershed_params, 
                                                                                          imageprops_params=imageprops_params)
        
        #Add frame number and time
        frame_time = (frame_serial_time-reference_serial_time)*24  #hours
        frame_properties_df.insert(loc=0, column='frame', value=frame_number) # add frame number
        frame_properties_df.insert(loc=1, column='time', value=frame_time.item()) # add frame time
        
        
        #Find average shift between current frame and first frame
        if frame_loop == 0:
            phase_image_old = int16_image
        else:
            shift = tracking_utils.find_shift(phase_image_old, int16_image)  
            frame_shift_stored[frame_loop] = frame_shift_stored[frame_loop-1] + shift
            phase_image_old = int16_image
            
        #Adjust centroids for frame shift
        frame_properties_df['x'] = frame_properties_df['x'] - frame_shift_stored[frame_loop, 0] # adjust x coordinate
        frame_properties_df['y'] = frame_properties_df['y'] - frame_shift_stored[frame_loop, 1] # adjust y coordinate
        

        #Collect tracking data for each frame
        location_properties_df = pd.concat([location_properties_df, frame_properties_df], ignore_index=True) 
    
    
        #Store frame, time, phase image and label image
        time_stored[frame_loop] = round(frame_time.item(), 3)
        image_stored[frame_loop, :, :] = int16_image
        label_stored[frame_loop, :, :] = uint16_label
    
    
    #list of frame numbers
    frames = frame_list[:,2].astype(int)
    
    #Save pixels sums for overall mass calculations
    pixel_sums = np.sum(image_stored.astype('float64'), axis=(1, 2)).reshape(-1,1) # sum pixels for each frame
    combined_pixel_data = np.column_stack((frames, time_stored, pixel_sums))
    data_folder_path = os.path.join(exp_folder_path,data_folder) # processed data folder
    os.makedirs(data_folder_path, exist_ok=True) # make the directory if not exists
    file_path_pixel = os.path.join(data_folder_path, f"pixel_sums_loc_{location}.csv") #assign pixel sums file name
    np.savetxt(file_path_pixel, combined_pixel_data, delimiter=',', header='Frame, Time, Pixel_Sum', comments='', fmt='%d,%.3f,%.6f') # save pixel sums
    
    
    #Ome Tiff image metadata
    metadata = {'axes': 'TYX',
                'experiment_name': exp_name,
                'experiment date': exp_date,
                'experiment folder': exp_folder_path,
                'pixelsize (um)': imageprops_params.get('pixel_size', 'N/A'),
                'wavelength (um)': imageprops_params.get('wavelength', 'N/A'),
                'frame_range': frames.tolist(),
                'frame_times': time_stored.tolist(),
                'conversion_factor_to_rads': '(2 * np.pi) / 65536'
        }
    
    #Save data and images
    image_folder_path = os.path.join(exp_folder_path,image_folder) # directory path
    os.makedirs(image_folder_path, exist_ok=True) # make the directory if not exists
    image_file_path = os.path.join(image_folder_path, f'processed_phase_loc_{location}.ome.tiff') # name the phase image file
    tracking_utils.save_ome_tiff(image_file_path, image_stored, metadata=metadata) # save the phase images
    label_file_path = os.path.join(image_folder_path, f'processed_label_loc_{location}.ome.tiff') # name the label image file
    tracking_utils.save_ome_tiff(label_file_path, label_stored) # save the label images
    
    
    return location_properties_df
    
    
    
    
    
    
    
    