import cv2
import numpy as np
import skimage
from skimage.segmentation import clear_border

def canny_simple_label(I, gauss_sigma=1.4, canny_low_thr=1, canny_high_thr=30, 
                       edge_dilate_kernel=(6,6), remove_size=400, mask_dilate_kernel=(4,4), clean_border=False, **kwargs):

    # Normalize the image to the range 0-255
    if I.dtype != np.uint8:
        normalized_image = cv2.normalize(I, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        normalized_image = I
    
    # Apply GaussianBlur to reduce noise and improve edge detection (when 0s, kernel size is calculated from sigma)
    blurred_image = cv2.GaussianBlur(normalized_image, (0, 0), gauss_sigma)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, canny_low_thr, canny_high_thr)

    # Dilate the edges
    kernel = np.ones(edge_dilate_kernel, np.uint8)
    dilated_edges = cv2.dilate(edges, kernel)
    
    # Fill the edges
    mask = np.zeros_like(dilated_edges)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    # Remove objects smaller than min_siz pixels
    filtered_mask = skimage.morphology.remove_small_objects(mask.astype(bool), min_size=remove_size)
    
    if clean_border:
        filtered_mask=clear_border(filtered_mask).astype('uint8')*255
    else:
        filtered_mask = filtered_mask.astype('uint8')*255
    
    # Dilate the masks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, mask_dilate_kernel)
    dilated_mask = cv2.dilate(filtered_mask, kernel)
    
    return dilated_mask


