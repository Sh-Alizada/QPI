import numpy  as np
import math
import os
import glob
import trackpy as tp
import pandas as pd
import scipy.io
import tracking_utils
from imageprops_v3 import imageprops
from poly_back import poly_back
from canny_simple_label import canny_simple_label
from canny_watershed_label import canny_watershed_label
from scipy.signal import medfilt2d

#Main experiment folder
exp_folder = r"C:\Users\shukr\Desktop\NP_files_for_Python"

#Raw phase image folder and file names
froot = os.path.join(exp_folder, 'DPCImages')
fstart = "QPM10x_040124_pos"
fext = '.mat'

#Experiment condition file
exp_cond_file = "experiment_conditions.xlsx"
exp_cond_path = os.path.join(exp_folder, exp_cond_file)
drug_names_df, well_map_df, drug_map_df, concentration_map_df = tracking_utils.read_maps_from_excel(exp_cond_path) #Load experiment conditions

#Load common background
common_back = scipy.io.loadmat(os.path.join(exp_folder, 'Btotal.mat'))['Btotal']

#Folder to store tracking arrays and stored images
savefolder =os.makedirs(os.path.join(exp_folder, 'tracking_results'), exist_ok=True)



wavelength = 624 # nm
image_size = [1200, 1920] # image size [height, width]
pixel_size = 5e-4 # mm/pixel, for 10x

loc_per_well = 9 # fixed number of imaging locations per well

min_area_thresh = 400 # pixels
max_area_thresh = 10000 # pixels
min_MI_thresh = 5 # mean intensity in nanometers
max_MI_thresh = 800 # mean intensity in nanometers
mass_factor = 1.1 
search_radius = 30 # pixels

poly_order = 8 # polynomial order
poly_reduction = 10 # factor

tracking_memory = 1 # number of frames




#Get location parameters
loc_list, num_loc = tracking_utils.get_loc_list(froot, fstart, fext)

#First location frames
file_list_loc_1 = glob.glob(os.path.join(froot, f"{fstart}{loc_list[0]}_*{fext}"))

#Load time for the first location and frame
_ , initial_frame_time = tracking_utils.load_frame_and_time(file_list_loc_1[0])

#Loop over the imaging locations
for loc_loop in  range(num_loc):
    
    #File pattern for current location to get frames
    file_pattern = os.path.join(froot, f"{fstart}{loc_list[loc_loop]}_*{fext}")

    #Get the list of frames for current imaging location
    file_list_loc = glob.glob(file_pattern)
    
    #Number of frames in current imaging location
    num_frame_loc=len(file_list_loc)
    
    #Initialize array for stored phase and label images for current location
    image_stored = np.zeros([num_frame_loc, image_size[0], image_size[1]], dtype=np.int16)
    label_stored = np.zeros([num_frame_loc, image_size[0], image_size[1]], dtype=np.uint16)
    
    #Initialize tracking parameters for current location
    tracking_df=pd.DataFrame() # initialize a dataframe to store location tracking data
    frame_shift_stored = np.zeros([num_frame_loc, 2]) # initialize an array to store frame shift in pixels
    time_stored = np.zeros(num_frame_loc) # initialize an array to store frame times
    
    #Loop over the frames in current imaging location
    for frame_loop in range(1):#num_frame_loc):
        
        print('processing ' f"{file_list_loc[frame_loop]}")
        
        #Load current raw phase image and its time
        raw_phase_image, current_frame_time = tracking_utils.load_frame_and_time(file_list_loc[frame_loop])
        
        #Difference in serial date number of location 1 frame 1 and current frame converted to hours
        frame_time = (current_frame_time-initial_frame_time)*24  #hours
        
        if raw_phase_image.std() != 0:  # continue calculations if not blank image
        
        
            #Convert raw phase image to nanometers from saved data type (commonly int16) and apply background correction
            raw_phase_image = raw_phase_image.astype(np.float64) * (2 * np.pi) / 65536 * wavelength # ? randian*nanometer ?
            phase_image = raw_phase_image - common_back # subtract common background
            binary_label = canny_simple_label(phase_image) # get simple binary cell mask
            poly_back = poly_back(phase_image, binary_label, poly_order=poly_order, reduction_factor=poly_reduction) # get the polynomial background
            phase_image = phase_image - poly_back # subtract the polynomial background
            phase_image = phase_image - np.mean(phase_image[binary_label==0]) # subtract the mean value of background
            
            
            #Find average shift between current frame and first frame
            if frame_loop == 0:
                phase_image_old = phase_image
            else :
                shift = tracking_utils.find_shift(phase_image_old, raw_phase_image)  
                frame_shift_stored[frame_loop] = frame_shift_stored[frame_loop-1] + shift
                phase_image_old = phase_image
            
            #Label the cells
            label_image = canny_watershed_label(phase_image)
            
            #Calculate cell properties for current frame
            cell_properties_df = imageprops(label_image, phase_image , pixel_size, min_area_thresh, max_area_thresh, min_MI_thresh, max_MI_thresh)
            cell_properties_df['x'] = cell_properties_df['x'] - frame_shift_stored[frame_loop, 0] # adjust center coordinate
            cell_properties_df['y'] = cell_properties_df['y'] - frame_shift_stored[frame_loop, 1] # adjust center coordinate
            cell_properties_df.insert(loc=0, column='frame', value=frame_loop+1) # add frame number
            cell_properties_df.insert(loc=1, column='time', value=frame_time.item()) # add frame time
        
            #Collect tracking data for each frame
            tracking_df = pd.concat([tracking_df, cell_properties_df], ignore_index=True) 
        
            #Store frame time, phase image and label image
            time_stored[frame_loop] = frame_time.item()
            image_stored[frame_loop, :, :] = (phase_image / (2 * np.pi) * 65536 / wavelength).astype(np.int16)
            label_stored[frame_loop, :, :] = label_image[:, :].astype(np.uint16)
    
    #Get drug and concentration for current well
    well = math.ceil((loc_loop+1)/loc_per_well)
    _, drug, concentration = tracking_utils.get_drug_concentration(well, drug_names_df, well_map_df, drug_map_df, concentration_map_df)
    
    #Add location info to tracking data
    tracking_df.insert(loc=0, column='well', value=well) # add well number
    tracking_df.insert(loc=1, column='location', value=loc_loop+1) # add location number
    tracking_df.insert(loc=4, column='drug_num', value=drug) # add drug number
    tracking_df.insert(loc=5, column='concentration', value=concentration) # add concentration

    #Add z dimension to track the cells with mass in addition to x and y
    tracking_df['z'] = tracking_df['mass'] * mass_factor # z is adjusted mass with mass factor
    
    #tp.link tracks the cells using columns  ['z', 'y', 'x', 'frame'] and adds 'particle' column indicating the cell IDs
    tp.quiet() # stop printing number of trajectories
    tracking_df = tp.link(tracking_df, search_range=search_radius, memory=tracking_memory) # link the cells
    tracking_df.drop('z', axis=1, inplace=True) # remove z column
    tracking_df.insert(0, 'cell', tracking_df.pop('particle')) # change column name to cell and move to the first position
    tracking_df.sort_values(by=['cell', 'frame'], inplace=True) # sort tracking data by cell ID and frame number
    
    #Save location mass
    ##   ADD CODE HERE
    
    #Save data and images
    os.makedirs(savefolder, exist_ok=True) # ensure the directory exists
    file_path_data = os.path.join(savefolder, f"tracking_data_loc_{loc_loop + 1}.csv") # assign tracking data file name
    tracking_df.to_csv(file_path_data, index=False) # save tracking data
    file_path_images = os.path.join(savefolder, f"image_arrays_loc_{loc_loop + 1}.npz") # assign image arrays file name
    np.savez_compressed(file_path_images, time=time_stored, phase_images=image_stored, label_images=label_stored) # save image arrays
    