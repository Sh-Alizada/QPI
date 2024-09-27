import numpy  as np
import math
import os
import glob
import pandas as pd
import scipy.io
import tracking_utils
from imageprops_v3 import imageprops
from canny_watershed_label import canny_watershed_label
from bckg_correction import bckg_correction

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
tracking_folder = os.path.join(exp_folder, 'tracking_results')



wavelength = 624 # nm
pixel_size = 5e-4 # mm/pixel, for 10x

loc_per_well = 9 # fixed number of imaging locations per well

min_area_thresh = 400 # pixels
max_area_thresh = 10000 # pixels
min_MI_thresh = 5 # mean intensity in nanometers
max_MI_thresh = 800 # mean intensity in nanometers
mass_factor = 1.1 
search_radius = 30 # pixels

gauss_sigma = 1.4
canny_low_thr = 1 
canny_high_thr = 30
edge_dilate_kernel = (6,6) #tuple
remove_size = 400
mask_dilate_kernel = (4,4) #tuple

poly_order = 8 # polynomial order
poly_reduction = 10 # factor

tracking_memory = 1 # number of frames




#Get location files
loc_list, num_loc = tracking_utils.get_loc_list(froot, fstart, fext)

#First location frames
file_list_loc_1 = glob.glob(os.path.join(froot, f"{fstart}{loc_list[0]}_*{fext}"))

#Load time for the first location and frame
initial_frame , initial_time = tracking_utils.load_frame_and_time(file_list_loc_1[0])

#Get the image size
image_size = initial_frame.shape

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
    location_properties_df=pd.DataFrame() # initialize a dataframe to store location tracking data
    frame_shift_stored = np.zeros([num_frame_loc, 2]) # initialize an array to store frame shift in pixels
    time_stored = np.zeros(num_frame_loc) # initialize an array to store frame times
    
    
    #Loop over the frames in current imaging location
    for frame_loop in range(1):#num_frame_loc):
        
        print(f"processing {file_list_loc[frame_loop]}")
        
        #Load current raw phase image and its time
        raw_phase_image, current_time = tracking_utils.load_frame_and_time(file_list_loc[frame_loop])
        
        #Difference in serial date number of location 1 frame 1 and current frame converted to hours
        frame_time = (current_time-initial_time)*24  #hours
        
        if raw_phase_image.std() != 0:  # continue calculations if not blank image
        
        
            #Convert raw phase image to nanometers from saved data type (commonly int16)
            raw_phase_image = raw_phase_image.astype(np.float64) * (2 * np.pi) / 65536 * wavelength # ? randian*nanometer ?
            
            #Apply background correction
            phase_image = bckg_correction(raw_phase_image, common_back, poly_order, poly_reduction,
                                          gauss_sigma, canny_low_thr, canny_high_thr, 
                                          edge_dilate_kernel, remove_size, mask_dilate_kernel)
           
            
            #Find average shift between current frame and first frame
            if frame_loop == 0:
                phase_image_old = phase_image
            else :
                shift = tracking_utils.find_shift(phase_image_old, phase_image)  
                frame_shift_stored[frame_loop] = frame_shift_stored[frame_loop-1] + shift
                phase_image_old = phase_image
            
            #Label the cells
            label_image = canny_watershed_label(phase_image)
            
            #Calculate cell properties for current frame
            frame_properties_df = imageprops(label_image, phase_image , pixel_size, min_area_thresh, max_area_thresh, min_MI_thresh, max_MI_thresh)
            frame_properties_df['x'] = frame_properties_df['x'] - frame_shift_stored[frame_loop, 0] # adjust center coordinate
            frame_properties_df['y'] = frame_properties_df['y'] - frame_shift_stored[frame_loop, 1] # adjust center coordinate
            frame_properties_df.insert(loc=0, column='frame', value=frame_loop+1) # add frame number
            frame_properties_df.insert(loc=1, column='time', value=frame_time.item()) # add frame time
        
            #Collect tracking data for each frame
            location_properties_df = pd.concat([location_properties_df, frame_properties_df], ignore_index=True) 
        
            #Store frame time, phase image and label image
            time_stored[frame_loop] = frame_time.item()
            image_stored[frame_loop, :, :] = (phase_image / (2 * np.pi) * 65536 / wavelength).astype(np.int16)
            label_stored[frame_loop, :, :] = label_image[:, :].astype(np.uint16)
    
    #Get drug and concentration for current well
    well = math.ceil((loc_loop+1)/loc_per_well)
    _, drug, concentration = tracking_utils.get_drug_concentration(well, drug_names_df, well_map_df, drug_map_df, concentration_map_df)
    
    #Add location info to tracking data
    location_properties_df.insert(loc=0, column='well', value=well) # add well number
    location_properties_df.insert(loc=1, column='location', value=loc_loop+1) # add location number
    location_properties_df.insert(loc=4, column='drug_num', value=drug) # add drug number
    location_properties_df.insert(loc=5, column='concentration', value=concentration) # add concentration

    #Track cells
    tracking_df = tracking_utils.track_cells(location_properties_df , mass_factor, search_radius, tracking_memory)
    
    #Save pixels sums for overall mass calculations
    pixel_sums = np.sum(image_stored.astype('float64'), axis=(1, 2)).reshape(-1,1)
    
    #Save data and images
    os.makedirs(tracking_folder, exist_ok=True) # make the directory if not exists
    file_path_pixel = os.path.join(tracking_folder, f"pixel_sums_loc_{loc_loop + 1}.csv") #assign pixel sums file name
    np.savetxt(file_path_pixel, pixel_sums, delimiter=',', header='Pixel_Sum', comments='') # save pixel sums
    file_path_data = os.path.join(tracking_folder, f"tracking_data_loc_{loc_loop + 1}.csv") # assign tracking data file name
    location_properties_df.to_csv(file_path_data, index=False) # save tracking data
    file_path_images = os.path.join(tracking_folder, f"image_arrays_loc_{loc_loop + 1}.npz") # assign image arrays file name
    np.savez_compressed(file_path_images, time=time_stored, phase_images=image_stored, label_images=label_stored) # save image arrays
    
    
#Create data_allframes   
tracking_utils.concat_track_files(tracking_folder)
    

#Add Here: EC50 Calculation Function



#Add Here: Overall Mass Calculation Function
    
    
    
    
    
#if save true, data file exists, overwrite false skip processing and go to tracking


#numFrames=None or list
    
    
    
    
    
    
#add single cell props data
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    