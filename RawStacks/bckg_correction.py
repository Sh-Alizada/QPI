import numpy as np
from canny_simple_label import canny_simple_label
from poly_back_v3 import poly_back
# import my_decorators

# @my_decorators.timer
def bckg_correction(I, common_back, gauss_sigma=1.4, canny_low_thr=1, canny_high_thr=30, edge_dilate_kernel=(6,6), 
                    remove_size=400, mask_dilate_kernel=(4,4), poly_order=8, poly_reduction=10, **kwargs):
    """
    Apply background corrections to the image I.
    
    Parameters:
    - I (numpy array): The input phase image. Should be the same unit as common_back.
    - common_back (numpy array): Common background image. Should be the same unit as I.
    - gauss_sigma : Gaussian smoothing sigma.
    - canny_high_thr : Canny strong edge threshold (canny_high_thr > canny_low_thr).
    - canny_low_thr : Canny weak edge threshold. Edges above this threshold will be included 
                        if they are attached to the strong edges (canny_high_thr > canny_low_thr).
    - edge_dilate_kernel (int tuple): Kernel size to dilate the edges.
    - remove_size (int): Largest object size to be removed. Any mask smaller than remove_size will be removed.
    - mask_dilate_kernel (int tuple): Kernel size to dilate the final binary mask.
    - poly_order (int): The order of the polynomial to fit.
    - poly_reduction (int): The factor by which to reduce the fitting points.
    
    Returns:
    - bckg_corrected (numpy array): Background corrected phase image.
    """

    # subtract common background
    cb_corrected = I - common_back 
    
    # get binary cell mask
    binary_label = canny_simple_label(cb_corrected, gauss_sigma=gauss_sigma, canny_high_thr=canny_high_thr, 
                                    canny_low_thr=canny_low_thr, edge_dilate_kernel=edge_dilate_kernel, 
                                    remove_size=remove_size, mask_dilate_kernel=mask_dilate_kernel)
    
    # get polynomial background
    poly_background = poly_back(cb_corrected, binary_label, poly_order=poly_order, poly_reduction=poly_reduction)
    
    # subtract the polynomial background
    pb_corrected = cb_corrected - poly_background 
    
    # subtract the mean value of residual background
    bckg_corrected = pb_corrected - np.mean(pb_corrected[binary_label==0]) 

    return bckg_corrected, binary_label