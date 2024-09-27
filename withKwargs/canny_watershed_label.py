import numpy as np
import skimage
from scipy import ndimage as ndi
import cv2
from canny_simple_label import canny_simple_label
# import my_decorators

# @my_decorators.timer
def canny_watershed_label(I, gauss_sigma_1=1.4, canny_low_thr=1, canny_high_thr=30, 
                       edge_dilate_kernel=(6,6), remove_size=400, mask_dilate_kernel=(4,4),
                       gauss_sigma_2=1.4, maxima_thr=50, maxima_dilate_kernel=(4,4)):
    
    # Normalize the image to the range 0-255
    if I.dtype != np.uint8:
        normalized_image = cv2.normalize(I, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        normalized_image = I
    
    #Get the binary label
    binary_label=canny_simple_label(normalized_image, gauss_sigma=gauss_sigma_1, canny_high_thr=canny_high_thr, 
                                    canny_low_thr=canny_low_thr, edge_dilate_kernel=edge_dilate_kernel, 
                                    remove_size=remove_size, mask_dilate_kernel=mask_dilate_kernel)
        
    #Apply gaussian smoothing
    blur_image =  cv2.GaussianBlur(normalized_image, (0, 0), gauss_sigma_2)
    
    # Apply extended maxima transform to find the local maxima
    maxima= skimage.morphology.h_maxima(blur_image, maxima_thr)
    
    # Dilate the local maxima (Sometimes very small pixel islands are created causing oversegmentation. Dilation solves this issue.)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, maxima_dilate_kernel)
    maxima = cv2.dilate(maxima, kernel)
    
    #Create water markers by labeling connected components in maxima
    markers = ndi.label(maxima)[0]
    
    # Invert the image to have valley instead of hills
    inverted_image = skimage.util.invert(blur_image)
    
    # Apply watershed algorithm
    L = skimage.segmentation.watershed(inverted_image, markers, mask=binary_label)
    
    # Clear border
    L = skimage.segmentation.clear_border(L)
    
    return L