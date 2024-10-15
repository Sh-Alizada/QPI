import numpy as np
from skimage.measure import regionprops_table, regionprops

def imageprops(L, I, pixSize):
    """
    Function to return volume, mass, area, mean intensity, standard deviation,
    centroid, and shape factor of regions in image I
    
    Parameters:
    L (ndarray): The label image such as made by BW mask image. Should be nonzero regions where the image will be analyzed.
    I (ndarray): The image to be processed.
    pixSize (float): The size of each pixel in the image (mm/pixel).
    
    Returns:
    volumes (list): The measured volume (um^3) of each region.
    masses (list): The measured mass (pg) of each region.
    areas (list): The area of each region (pixel).
    mean_intensities (list): The mean intensity of each region.
    std_devs (list): The standard deviation of pixel values in each region.
    centroids (list): The centroid of each region.
    shape_factors (list): The shape factor of each region, defined as the circularity 4*pi*A/(P^2).
    labels (list): The label number of each region.
    """
    
    # Define the conversion factor
    k = (((1/(10000**3)/100)/0.0018)*1e12)/np.pi
    
    # Get region properties using regionprops_table for supported properties
    properties = regionprops_table(L, intensity_image=I, properties=['area', 'mean_intensity', 'perimeter', 'label', 'centroid'])
    
    areas = properties['area']
    mean_intensities = properties['mean_intensity']
    perimeters = properties['perimeter']
    centroids = list(zip(properties['centroid-0'], properties['centroid-1']))
    labels = properties['label']
    
    # Initialize lists to store shape factors and standard deviations
    shape_factors = []
    std_devs = []
    
    # Get pixel values using regionprops
    regions = regionprops(L, intensity_image=I)
    
    for area, perimeter, region in zip(areas, perimeters, regions):
        if perimeter > 0 and area > 9:
            shape_factor = (4 * np.pi * area) / (perimeter ** 2)
        else:
            shape_factor = np.nan
        shape_factors.append(shape_factor)
        
        # Calculate the standard deviation of pixel values
        std_devs.append(np.std(region.intensity_image[region.image]))
    
    # Calculate volume and mass
    volumes = [mean_intensity * area * (pixSize ** 2) * 1e3 for mean_intensity, area in zip(mean_intensities, areas)]
    masses = [volume * k for volume in volumes]
    
    return volumes, masses, areas, mean_intensities, centroids, shape_factors, labels
