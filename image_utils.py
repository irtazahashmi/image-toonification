import cv2
import numpy as np
import skimage 
from matplotlib import pyplot as plt
from skimage.exposure import match_histograms


class ImageUtils:
    def load_image_rgb(path : str):
        """
        Load color image using OpenCV.

        Args:
            path (str): the path to the image

        Returns:
            np.ndarray: the image as a numpy array
        """
        img = cv2.imread(path)
        return img


    def save_image(path: str, img: np.ndarray):
        """
        Save image using OpenCV.

        Args:
            path (str): the path to save the image
            img (np.ndarray): the image as a numpy array to be saved
        """
        cv2.imwrite(path, img)


    def save_image_scaled(path: str, img: np.ndarray):
        """
        Save image using OpenCV by scaling the image.

        Args:
            path (str): the path to save the image
            img (np.ndarray): the image as a numpy array to be saved
        """
        # scale image to 0-255
        img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
        cv2.imwrite(path, img)

    
    def normalize_image(img:np.ndarray):
        """
        Normalize the image.

        Args:
            img (np.ndarray): the image to be normalized

        Returns:
            np.ndarray: normalized image
        """
        return img / np.amax(img)
        

    def plot_pixel_histogram(img:np.ndarray, title : str = ''):
        """
        Plots the pixel intensities of the pixels.

        Args:
            img (np.ndarray): the image to be plotted
            title (str): the title of the plot
        """

        histogram, bin_edges = np.histogram(img, bins=256, range=(img.min(), img.max()))

        plt.title(title)
        plt.xlabel("Pixel Value")
        plt.ylabel("Number of Pixels")
        plt.xlim(img.min(), img.max())
        plt.plot(bin_edges[0:-1], histogram)
        plt.show()
   


    def display_image(img:np.ndarray, image_title:str = '', normalize=False):
        """
        Display image using OpenCV.

        Args:
            img (np.ndarray): the image to be displayed
            img_title (str): the image title
        """
        # normalize image beweem 0 and 1 for open cv
        if normalize:
            img = (img - img.min()) / (img.max() - img.min()).astype('uint8')
            
        cv2.imshow(image_title, img)
        cv2.waitKey(0)

    def compute_image_luminance(img: np.ndarray):
        """
        Computing the luminance values on the basis of the RGB pixel input
        where: 
    
                L(x,y) = 0.212R(x, y) + 0.715G(x, y) + 0.072B(x, y)  [I.T.U. 1990]
        Args:
            img (np.ndarray): _description_

        Returns:
            np.ndarray: computed luminance values
        """

        # the luminosity coefficients according to the paper 
        # https://graphics.unizar.es/papers/LopezMoreno_NPAR2010.pdf
        luminosity_coeff_r = 0.213
        luminosity_coeff_g = 0.715
        luminosity_coeff_b = 0.072

        image_luminance_r = luminosity_coeff_r * img[:,:,0]
        image_luminance_g = luminosity_coeff_g * img[:,:,1]
        image_luminance_b = luminosity_coeff_b * img[:,:,2]
        
        image_luminance = image_luminance_r + image_luminance_g + image_luminance_b
                
        return image_luminance

    
    def apply_tonal_balance_on_image(img: np.ndarray, reference_img: np.ndarray):
        """
        Technique: histogram matching [Durand and Dorsey 2002].
        Tonal balance an image using a reference image.

        Args:
            img (np.ndarray): the image to apply tonal balance on.
            reference_img (np.ndarray): the reference image.

        Returns:
            np.ndarray: the tonally balanced image.
        """
        matched = match_histograms(img, reference_img, channel_axis=-1)
        return matched


    
    def calculate_final_approx_depth(B_layer: np.ndarray, D_layer: np.ndarray, fB : float, fD: float):
        """
        Calculate the final depth map using the depth map of the background and 
        the depth map of the foreground. The final depth map is calculated using:

                        Z(x, y) = FbB(x, y) + FdD(x, y)

        Args:
            B_layer (np.ndarray): the base layer of the image
            D_layer (np.ndarray): the depth laer of the image
            fB (float): user defined weighting factor to control the presence 
            of large features in the final image. ∈ [0, 1].
            fD (float): user defined weighting factor to control the presence 
            of small features in the image. ∈ [0, 1].

        Returns:
            np.ndarray: the final depth map
        """

        # calculate width and height of the image
        height = B_layer.shape[1]
        width = B_layer.shape[0]

        # initialize the final depth map with zeros
        final_approx_depth = np.zeros((width, height))

        # calculate the final depth map 
        for x in range(width):
            for y in range(height):
                final_approx_depth[x][y] = fB * B_layer[x][y] + fD * D_layer[x][y]
        
        return final_approx_depth


    def apply_thresholding(img:np.ndarray, threshold:float):
        """
        Display image using a threshold. The image < threshold is black and 
        the image > threshold is white.

        Args:
            img (np.ndarray): the image to be plotted 
            threshold (float): the threshold to be used to display the image
        """

        binary_mask = img > threshold

        plt.title('Thresholding')
        plt.imshow(binary_mask, cmap="gray")
        plt.axis('off')
        plt.show()

    def generate_image_combinations(B_layer: np.ndarray, D_layer: np.ndarray):
        """
        Calculate the approximate depth map using the base layer and the depth layer.
        Display the results.

        Args:
            B_layer (np.ndarray): the base layer of the image
            D_layer (np.ndarray): the depth laer of the image

        Returns:
            np.ndarray: the image combinations
        """
        fB = 1
        fD_combos = [0, 0.25, 0.5, 0.75, 1]

        for fD in fD_combos:
            final_approx_depth = ImageUtils.calculate_final_approx_depth(B_layer, D_layer, fB, fD)
            threshold = skimage.filters.threshold_otsu(final_approx_depth)
            ImageUtils.apply_thresholding(final_approx_depth, threshold)


    def apply_adaptive_thresholding(img: np.ndarray, threshold: float):
        """
        Apply adaptive thresholding on the image.

        "Adaptive Thresholding using the Integral Image". Derek Bradley &Gerhard Roth

        Args:
            img (np.ndarray): the image to apply adaptive thresholding on
            block_size (int): the size of the pixel neighborhood that is used to calculate the threshold value for the pixel
            C (int): a constant that is subtracted from the mean or weighted mean

        Returns:
            np.ndarray: the thresholded image
        """
        
        # resize image for faster processing
        src_img = cv2.resize(img, (256, 196))

        # calculate the height and width of the image
        width = src_img.shape[0]
        height = src_img.shape[1]

         # calculate the integral image according to the paper
        integral_image = np.zeros_like(src_img, dtype=np.float32)

        for y in range(height):
            for x in range(width):
                integral_image[x][y] = img[0:x, 0:y].sum()

        # sample window size s x s
        # Wellner used 1 / 8 of the image width but this produces better results for me
        sample_window_size = (height / 24)

        # initialize the thresholded image
        result = np.zeros_like(src_img)
        
        # loop over the image
        for y in range(height):
            for x in range(width):
                # calculate the x and y coordinates of the top left corner of the sample window
                x0 = round(max(0, x - sample_window_size ))
                x1 = round(min(width - 1, x + sample_window_size))

                y0 = round(max(0, y - sample_window_size ))
                y1 = round(min(height - 1, y + sample_window_size))

                # calculate the number of pixels in the sample window
                sample_window_pixels = (y1 - y0) * (x1 - x0) 

                # calculate the sum of the pixels in the sample window
                sum_of_pixels = integral_image[x0, y0] - integral_image[x0, y1] - integral_image[x1, y0]  + integral_image[x1, y1]         
                
                # calculate threshold in decimal
                thr = 1 - (threshold / 100)

                # calculate the mean of the pixels in the sample window
                if np.all(src_img[x, y] * sample_window_pixels < sum_of_pixels * thr):
                    # set the pixel to white
                    result[x, y] = 0
                else:
                    # set the pixel to black
                    result[x, y] = 255

        plt.title('Adaptive Thresholding')
        plt.imshow(result, cmap="gray")
        plt.axis('off')
        plt.show()

        return result






    

    
