import numpy as np

def normalize_img(img):   
    image = (img - np.min(img))/(np.ptp(img)) # ptp range of values (min-max) along axis
    return image

    
    
    
 
 