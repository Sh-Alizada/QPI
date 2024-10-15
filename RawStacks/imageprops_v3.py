import numpy as np
from skimage.measure import regionprops
import pandas as pd
# import my_decorators

# @my_decorators.timer
def imageprops(I, L, wavelength=624, pixel_size=5e-4, min_area_thr=1, 
               max_area_thr=100000, min_MI_thr=0, max_MI_thr=1000, **kwargs):
    """
    Function to return volume, mass, area, mean intensity, standard deviation,
    centroid, and shape factor of regions in image I
    
    Parameters:
    L (ndarray): The label image. Should be nonzero regions where the image will be analyzed.
    I (ndarray): The phase image in nanometers.
    pixSize (float): The size of each pixel in the image (mm/pixel).
    minAreathresh (float): The minimum area threshold for filtering regions.
    maxAreathresh (float): The maximum area threshold for filtering regions.
    minMIthresh (float): The minimum mean intensity threshold for filtering regions.
    maxMIthresh (float): The maximum mean intensity threshold for filtering regions.
    
    Returns:
    pandas data frame that contains:
        volumes (list): The measured volume (um^3) of each region.
        masses (list): The measured mass (pg) of each region.
        areas (list): The area of each region (pixel).
        mean_intensities (list): The mean intensity of each region.
        std_devs (list): The standard deviation of pixel values in each region.
        centroids (list): The centroid of each region.
        shape_factors (list): The shape factor of each region, defined as the circularity 4*pi*A/(P^2).
        labels (list): The label number of each region.
    """
    #Convert image to nanometers
    Img = I*wavelength/(np.pi)
    
    # Conversion factor from optical volume to dry mass
    k = 5.55556
    
    # Get region properties using regionprops
    regions = regionprops(L, intensity_image=Img)
    
    # Initialize list to store the features for each region
    feature_list = []
    
    for region in regions:
        area = region.area
        mean_intensity = region.mean_intensity
        
    
        if (area > min_area_thr and area < max_area_thr and
            mean_intensity > min_MI_thr and mean_intensity < max_MI_thr):
                
                volume = mean_intensity * area * (pixel_size ** 2) * 1e3 # um^3
                mass = volume * k # pg
                label = region.label
                bbox = region.bbox
                shape_factor = (4 * np.pi * area) / (region.perimeter ** 2)
                std_dev = np.std(region.intensity_image[region.image])
                centroid = region.centroid
        
                feature_list.append({
                'label': label,
                'mass': mass,
                'y': centroid[0],
                'x': centroid[1],
                'volume': volume,
                'area': area,
                'mean_intensity': mean_intensity,
                'std_dev': std_dev,
                'shape_factor': shape_factor,
                'bounding_box': bbox
            })
    
    # Convert the feature list to a pandas DataFrame
    feature_list_df = pd.DataFrame(feature_list)
    
    return feature_list_df
