import skimage 
import time
import cv2
from image_utils import ImageUtils
from bilateral_decomposition import bilateral_decomposition

# start timer
start_time = time.time()

# folder to load
image_to_load = 'm.png' # 'lady.png' # test_image.jpg # old_man.png
path = 'images/' + image_to_load

# load image
print("Loading image...")
img = ImageUtils.load_image_rgb(path)

# resize image for report comparison
#scale_factor = 4
#img_height, img_width, _ = img.shape
#img = cv2.resize(img, (img_width//scale_factor, img_height//scale_factor), interpolation=cv2.INTER_AREA)

# show image
#ImageUtils.display_image(img, 'input', normalize=True)

# compute luminance values
print("Computing image luminance...")
img_luminance = ImageUtils.compute_image_luminance(img)

# show luminance image
#ImageUtils.display_image(img_luminance, 'luminance', normalize=True)

# normalize luminance image
print("Normalizing image...")
img_luminance_normalized = ImageUtils.normalize_image(img_luminance)


# bilateral filter decomposition
print("Decomposing image into base and detail layer...")
B_img, D_img = bilateral_decomposition(img_luminance_normalized)

# show base image
#ImageUtils.display_image(B_img, 'base layer')

# show detail image
#ImageUtils.display_image(D_img, 'detail layer')

# apply tonal matching to the base layer
print("Applying tonal matching to the base layer...")
#ImageUtils.plot_pixel_histogram(B_img, 'base layer before tonal matching')

B_img = ImageUtils.apply_tonal_balance_on_image(B_img, D_img)

#ImageUtils.plot_pixel_histogram(B_img, 'base layer after tonal matching')
#ImageUtils.plot_pixel_histogram(D_img, 'detail layer')

# calc depth map from base layer and detail layer. Show results.
print("Calculating depth map and displaying results...")
ImageUtils.generate_image_combinations(B_img, D_img)


# halftoning techniques
print("Applying halftoning techniques...")
print("Applying thresholding...")
fB = 1
fD = 1
final_approx_depth = ImageUtils.calculate_final_approx_depth(B_img, D_img, fB, fD)
t = skimage.filters.threshold_otsu(final_approx_depth)
ImageUtils.apply_thresholding(final_approx_depth, t)


print("Applying adaptive thresholding...")
ImageUtils.apply_adaptive_thresholding(final_approx_depth, threshold=78)

end_time = time.time()

total = end_time - start_time
print("Total time: " + str(total))

