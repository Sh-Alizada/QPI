import numpy as np
import skimage
from scipy import ndimage as ndi
import cv2

def imposemin(img, minima):
    marker = np.full(img.shape, np.inf)
    marker[minima == 1] = 0
    mask = np.minimum(img, marker)
    return mask

def canny_simple_label_ws(I):

    # Normalize the image to the range 0-255
    if I.dtype != np.uint8:
        normalized_I = cv2.normalize(I, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        normalized_I = I
    
    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(normalized_I, (5, 5), 1.4)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, threshold1=1, threshold2=30)

    # Dilate the edges
    kernel = np.ones((6, 6), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel)
    
    # Fill the edges
    mask = np.zeros_like(dilated_edges)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    # Remove small objects smaller min_siz pixels
    min_size = 400
    filtered_mask = skimage.morphology.remove_small_objects(mask.astype(bool), min_size=min_size).astype('uint8')*255
    
    # Dilate the masks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv2.dilate(filtered_mask, kernel)
    
    return dilated_mask

def canny_watershed_label(I):
    
    # Normalize the image to the range 0-255
    if I.dtype != np.uint8:
        normalized_image = cv2.normalize(I, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        normalized_image = I
    
    #Get the binary label
    binary_label=canny_simple_label_ws(normalized_image)
        
    #Apply gaussian smoothing
    blur_image =  cv2.GaussianBlur(normalized_image, (5, 5), 1.4)
    
    # Apply extended maxima transform to find the local maxima
    thr = 50  # Adjust threshold
    mask_em= skimage.morphology.h_maxima(blur_image, thr)
    
    # Dilate the local maxima
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    mask_em = cv2.dilate(mask_em, kernel)
    
    # Invert the image to have valley instead of hills
    I_c = skimage.util.invert(blur_image)
    
    minima= ~binary_label.astype(bool) | mask_em.astype(bool)
    
    imposed_image=imposemin(I_c, minima)
    
    markers = ndi.label(mask_em)[0]
    
    # Apply watershed algorithm
    L = skimage.segmentation.watershed(imposed_image, markers, mask=binary_label)
    
    
    # Clear border
    L = skimage.segmentation.clear_border(L)
    
    return L