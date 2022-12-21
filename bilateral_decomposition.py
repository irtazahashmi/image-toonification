import numpy as np
from bilateral_filter import bilateral_filter_me
import cv2

def bilateral_decomposition(img: np.ndarray):
    """
    Bilateral decomposition of the image.

    Args:
        img (np.ndarray): the image to be decomposed
    Returns:
        np.ndarray, np.ndarray: the base image and the detail image
    """
    # calculate height and width of the image
    height = img.shape[0]
    width = img.shape[1]

    # configurations
    # consistently produces good results
    # https://people.csail.mit.edu/soonmin/CV/bw_photo_ring_toss_standard.pdf
    d = 10
    sigma_r = 0.2
    sigma_s = min(width, height) / 16
    
    # decomposition
    # base layer B, large scale features
    B_img = cv2.bilateralFilter(img.astype(np.float32), d, sigma_r, sigma_s)
    #B_img = bilateral_filter_me(img.astype(np.float32), d, sigma_r, sigma_s)
    
    # D = I − B, detail layer D, small scale features
    D_img = img - B_img

    return B_img, D_img
