import numpy as np
from math import sqrt


def _gaussian(x : float, sigma : float, mu : float =0.0):
    """
    Returns pdf of a Gaussian distribution with given sigma (std dev) and mean (mu).

    Args:
        x (float): where to sample the pdf
        sigma (float): std dev of the Gaussian
        mu (float, optional): mean. Defaults to 0.0.

    Returns:
        float: pdf value
    """
    # calculate the gaussian value
    exponent = (x - mu) / sigma
    return np.exp(-0.5 * exponent ** 2)

def _calculate_dist(x0:float, y0:float, x1:float, y1:float):
    """
    Calculate the distance between two points.

    Args:
        x0 (float): x coordinate of point 0
        y0 (float): y coordinate of point 0
        x1 (float): x coordinate of point 1
        y1 (float): y coordinate of point 1

    Returns:
        float: the calculated distance between the two points
    """
    # calculate the distance between two points
    return sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

def _bilateral_pixel(src_img: np.ndarray, x:int, y:int, filter_diameter:int, sigma:float, guide_sigma:float):
    """
    Apply bilateral filter for a single pixel.

    Args:
        src_img (np.ndarray): the source image
        x (int): x coordinate of the pixel
        y (int): y coordiate of the pixel
        filter_diameter (int): filter diameter
        sigma (float): std dev of the Gaussian
        guide_sigma (float):  std dev of the Gaussian

    Returns:
        float: the output pixel value
    """
    # get height and width of image
    height = len(src_img[0])
    width =  len(src_img)

    # get filter radius
    radius = filter_diameter / 2

    # the output pixel value of the pixel at x,y
    output_pixel = 0
    # the sum of the weights
    total_weight = 0

    # loop over all pixels in filter diameter
    for neighbour_index_x in range(filter_diameter):
         for neighbour_index_y in range(filter_diameter):
            
            # get the neighbour pixel x and y
            neighbour_x = int(x - (radius - neighbour_index_x))
            neighbour_y = int(y - (radius - neighbour_index_y))

            # check if neighbour pixel is inside the image x 
            if neighbour_x < 0 or neighbour_x >= width:
                continue
       
            # check if neighbour pixel is inside the image y
            if neighbour_y < 0 or neighbour_y >= height:
                continue
            
            # calculate the distance between the pixel and the neighbour pixel
            px_distance = src_img[neighbour_x][neighbour_y] - src_img[x][y]
            
            # calculate distance between the pixel and the neighbour pixel
            dist = _calculate_dist(neighbour_x, neighbour_y, x, y)

            gaus_i = _gaussian(px_distance, sigma)
            gs = _gaussian(dist, guide_sigma)
            # calculate the weight of the pixel
            w_i = gaus_i * gs

            output_pixel += src_img[neighbour_x][neighbour_y] * w_i
            total_weight += w_i

    # compute weighted mean of all neighbouring pixels     
    if total_weight == 0:
        output_pixel = src_img[x][y]
    else:
        output_pixel = output_pixel / total_weight

    # return the result pixel
    return output_pixel



def bilateral_filter_me(src_img:np.ndarray, filter_diameter:int, sigma:float, guide_sigma:float):
    """
    Apply bilateral filter to an image.

    Args:
        src_img (np.ndarray): the source image
        filter_diameter (int): the diameter of the kernel
        sigma (float): std dev of the Gaussian
        guide_sigma (float): sigma value of the gaussian guide kernel.

    Returns:
        _type_: _description_
    """
    # empty output image with src image width and height
    height = len(src_img[0])
    width =  len(src_img)
    
    
    # result img empty
    result_img = np.zeros(src_img.shape)

    # loop over all pixels, x and y
    for x in range(width):
        for y in range(height):
            print(f"Progress: {x/width * 100} %")
            # apply bilateral filter for each pixel
            result_img[x][y] = _bilateral_pixel(src_img, x, y, filter_diameter, sigma, guide_sigma)
    
    # return result
    return result_img




