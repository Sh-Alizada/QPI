import numpy as np
from skimage.segmentation import clear_border
import cv2
from skimage import morphology

def canny_simple_label_cb(I):

    # Normalize the image to the range 0-255
    normalized_image = cv2.normalize(I, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 1.4)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, threshold1=1, threshold2=30)

    # Define a kernel for dilation
    kernel = np.ones((6, 6), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Fill the contours
    mask = np.zeros_like(dilated_edges)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    # Remove small objects smaller min_siz pixels
    min_size = 400
    filtered_mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_size).astype('uint8')*255
    
    #Clear Border
    cleared_mask=clear_border(filtered_mask)
    
    
    # Dilate the masks again
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    dilated_mask = cv2.dilate(cleared_mask, kernel, iterations=1)
    
    return dilated_mask