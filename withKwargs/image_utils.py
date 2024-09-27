import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
from scipy.optimize import curve_fit
import pandas as pd
from skimage.segmentation import find_boundaries
from matplotlib.figure import figaspect
from palettable.lightbartlein.diverging import BlueDarkRed18_18
import os
from imageprops_v3 import imageprops

def overlay_outlines(I, L, cmap='temp', clim=None, fig_num=None, title=None):
    """
    Plots the boundaries of labeled regions on the original image.

    Parameters:
    - I: The phase image.
    - L: The label image.
    - cmap: colormap (Matlab 'temp' colormap if not specified otherwise)
    - clim: 1x2 array for the lower and higer limits of the colormap
    - fig_num: Optional; a specific figure number for the plot.
    - title: Optional; title for the plot.
    """

    set_matplotlib_formats('svg')

    w, h = figaspect(I)

    if fig_num is not None:
        plt.figure(fig_num, figsize=(w, h))
    else:
        plt.figure(figsize=(w, h))

    if cmap=='temp':
        # Get the BlueDarkRed18_18 (temp) colormap
        custom_cmap = BlueDarkRed18_18.mpl_colormap
    else:
        custom_cmap = cmap
    
    if clim is not None:
        vmin = clim[0]
        vmax = clim[1]
    else:
        vmin = np.min(I)
        vmax = np.max(I)
        
    # Create a copy of phase_image to avoid modifying the original
    phase_image_copy = I.copy()

    # Find boundaries of labeled regions
    boundaries = find_boundaries(L, mode='inner')

    # Overlay boundaries
    phase_image_copy[boundaries] = np.max(phase_image_copy)  # Set boundary pixels to maximum value

    plt.imshow(phase_image_copy, cmap=custom_cmap, vmin=vmin, vmax=vmax, interpolation='none')

    if title:
        plt.set_title(title)
    plt.axis('image')
    # plt.axis('off')
    plt.colorbar()
    # plt.show()
    

def annotate_props(I, L, prop='area', pixel_size=5e-4, wavelength=624,
                    min_area_thr=1, max_area_thr=100000, min_MI_thr=0, max_MI_thr=10000, prop_size=3):
    """
    Function to plot specified properties and centroid coordinates on phase images at the centroids of labeled regions.
    
    Parameters:
    phase_image (ndarray): The phase image to be processed.
    label_image (ndarray): The label image with labeled regions.
    pixel_size (float): The size of each pixel in the image (mm/pixel).
    prop (str): The property to be plotted. Available options are 'volume', 'mass', 'area',
                'mean_intensity', 'std_dev', 'shape_factor', 'label', and 'centroid'. Default is 'area'.
    
    Returns:
    None: Displays the annotated image.
    """
    # Calculate properties using the existing imageprops_L function
    im_props = imageprops(I, L, wavelength=wavelength, pixel_size=pixel_size, min_area_thr=min_area_thr, 
                   max_area_thr=max_area_thr, min_MI_thr=min_MI_thr, max_MI_thr=max_MI_thr)
    
    # Map the property name to the actual data
    properties = {
        'volume': im_props['volume'],
        'mass': im_props['mass'],
        'area': im_props['area'],
        'mean_intensity': im_props['mean_intensity'],
        'shape_factor': im_props['shape_factor'],
        'label': im_props['label'],
        'centroid': list(zip(im_props['x'], im_props['y']))
    }
    
    # Check if the requested property is valid
    if prop not in properties:
        raise ValueError(f"Property '{prop}' not recognized. Available properties are: {list(properties.keys())}")
    
    # Plot the outlines using the provided function
    overlay_outlines(I, L)
    
    # Capture the current figure and axis
    ax = plt.gca()
    
    # Get the selected property data
    selected_property = properties[prop]
    
    # Add annotations for the selected property at the centroids
    if prop == 'centroid':
        for (x, y) in selected_property:
            ax.text(x, y, f"({int(x)}, {int(y)})", color='red', fontsize=prop_size, ha='center', va='center')
    else:
        for value, (x, y) in zip(selected_property, properties['centroid']):
            ax.text(x, y, f"{value:.1f}" if prop != 'label' else f"{int(value)}", color='red', fontsize=prop_size, ha='center', va='center')
    
    # Show the updated plot with annotations
    plt.title(prop, fontsize=8)
    plt.colorbar()
    plt.show()
