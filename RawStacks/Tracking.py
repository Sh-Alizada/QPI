from imageprops_v2 import imageprops
from corr_shift import corr_shift
import numpy  as np
import os
import glob
import trackpy as tp
import pandas as pd
import tracking_utils
import scipy.io

froot = r"C:\Users\shukr\Desktop\NP_files_for_Python\DPCImages"
fstart = "QPM10x_040124_pos"
fext = '.mat'
date = '040124'
savefolder = froot

wavelength = 624 # nm
pixel_size = 5e-4 # mm/pixel, for 10x

minAreathresh = 400 # number of pixels
maxAreathresh = 100000 # number of pixels
minMIthresh = 5 # 
maxMIthresh = 800 #

image_size = [1200, 1900] # image size [height, width]

common_back = scipy.io.loadmat(os.path.join(froot, 'Btotal.mat'))['Btotal']


loc_list, num_loc, num_frame = tracking_utils.get_loc_list(froot, fstart, fext)


#First location frames
file_list_loc_1 = glob.glob(os.path.join(froot, f"{fstart}{loc_list[0]}_*{fext}"))

#Load time for the first location and frame
_ , initial_frame_time = tracking_utils.load_frame_and_time(file_list_loc_1[0])

#loop over the locations
for loc_loop in  range(num_loc):
    
    # file pattern for current location to get frames
    file_pattern = os.path.join(froot, f"{fstart}{loc_list[loc_loop]}_*{fext}")

    # Get the list of frames for current imaging location
    file_list_loc = glob.glob(file_pattern)
    
    #number of frames in current imaging location
    num_frame_loc=len(file_list_loc)
    
    # initiate arrayy for stored phase and label images for current location
    image_stored = np.zeros([num_frame_loc, image_size[0], image_size[1]], dtype=np.int16)
    label_stored = np.zeros([num_frame_loc, image_size[0], image_size[1]], dtype=np.uint16)
    
    # initialize tracking parameters for current location
    phase_image_old , _ =tracking_utils.load_frame_and_time(file_list_loc[0])
    yshift_store = np.zeros(num_frame_loc)
    xshift_store = np.zeros(num_frame_loc)
    time_stored = np.zeros(num_frame_loc)
    xshift_old = 0
    yshift_old = 0
    tracking_array=np.empty((0, 9))
    
    
    for frame_loop in range(num_frame_loc):
        
        print('processing ' f"{file_list_loc[frame_loop]}")
        
        #Load current raw phase image and its time
        raw_phase_image, current_frame_time = tracking_utils.load_frame_and_time(file_list_loc[frame_loop])
        
        #Convert raw phase image to nanometers from saved type (commonly int16) and apply background correction
        raw_phase_image = (raw_phase_image.astype(np.float64) * (2 * np.pi)) / 65536 * wavelength # nanometer
        phase_image = raw_phase_image - common_back # subtract common background
        
        
        # Difference in serial date number of location 1 frame 1 and current frame converted to hours
        time = (current_frame_time-initial_frame_time)*24  #hours
        
        # 
        
        volumes, masses, areas, mean_intensities, centroids, shape_factors, labels = imageprops(label_image, phase_image , pixel_size)
        
        if phase_image.std() != 0:  # skip if blank image
        
            # find average shift between current frame and first frame
            yshift, xshift = corr_shift(phase_image_old, phase_image)  
            yshift += yshift_old
            xshift += xshift_old
            yshift_store[frame_loop] = yshift
            xshift_store[frame_loop] = xshift
            xshift_old = xshift
            yshift_old = yshift
            
            phase_image_old = phase_image
            
            #store time
            time_stored[frame_loop] = time
            
            # Save phase image and label_image into image_stored and label_stored
            image_stored[:, :, frame_loop] = (((((phase_image / 2) / np.pi) * 65536) / wavelength)).astype(np.int16)
            label_stored[:, :, frame_loop] = label_image[:, :].astype(np.uint16)
            
            
            # Loop through all cells and find only the ones which meet area and mean intensity requirements
            for ii in range(len(volumes)):
                if (not np.isnan(centroids[ii][0]) and areas[ii] > minAreathresh and areas[ii] < maxAreathresh and
                    mean_intensities[ii] > minMIthresh and mean_intensities[ii] < maxMIthresh):
                    
                    new_row = np.zeros((1,9)) #initiate an empty array to be added to tracking_array
                    new_row[0, 0:2] = list(centroids[ii]) # store position in first two columns of tracking_array
                    new_row[0, 0:2] = new_row[0, 0:2] - [xshift, yshift] # remove shift due to movement of the entire frame
                    new_row[0, 2] = masses[ii] # store cell masses
                    new_row[0, 3] = time # store frame time 
                    new_row[0, 4] = areas[ii] #store areas
                    new_row[0, 5] = shape_factors[ii] #store shape factors
                    new_row[0, 6] = labels[ii] #store cell label IDs
                    new_row[0, 7] = loc_loop+1 #store location number
                    new_row[0, 8] = frame_loop+1 #store frame number
                    
                    #add current cell data to tracking array
                    tracking_array= np.vstack((tracking_array, new_row))
                   
    #convert tracking array to pandas DataFrame
    columns = ['x', 'y', 'mass', 'time', 'area', 'shape factor', 'label', 'locaation', 'frame']
    tracking_data = pd.DataFrame(tracking_array, columns=columns)
    
    #Add z dimension to track the cells with mass in addition to x and y
    tracking_data['z'] = tracking_data['mass'] * 1.1    # adjust mass with mass factor
    tracking_data.insert(0, 'z', tracking_data.pop('z')) # tp.link function below tracks ['z', 'y', 'x', 'frame'] unless indicated otherwise (look at function description)
    
    # tp.link tracks the cells and adds 'particle' column indicating the cell IDs
    tp.quiet() #stop printing number of trajectories
    tracking_data = tp.link(tracking_data, search_range=30, memory=1) #link the cells
    tracking_data.sort_values(by=['particle', 'frame'], inplace=True) #sort tracking data by cell ID and frame number
    tracking_data.drop('z', axis=1, inplace=True) #remove z column
    
    #save data and images
    os.makedirs(savefolder, exist_ok=True) # Ensure the directory exists
    file_path_data = os.path.join(savefolder, f"tracking_data_loc_{loc_loop + 1}.json") #assign tracking data file name
    tracking_data.to_json(file_path_data, orient='records', lines=True) #save tracking data to JSON file
    file_path_images = os.path.join(savefolder, f"image_arrays_loc_{loc_loop + 1}.npz") #assign image arrays file name
    np.savez_compressed(file_path_images, array1=image_stored, array2=label_stored) #save image arrays
    