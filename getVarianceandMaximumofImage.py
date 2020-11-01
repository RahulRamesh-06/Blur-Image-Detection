import numpy as np 

from scipy.ndimage import variance
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.transform import resize


# Loading the image with imread

image = io.imread('/0_IPHONE-SE_F.JPG')

# Resizing the image. 
image = resize(image, (400, 600))

# Convert the image to greyscale
image = rgb2gray(image)

# Detecting the Edge of the image
edge = laplace(image, ksize=3)

# Print The variance and Maximum 
print(f"Variance of the egde detected: {variance(edge)}")
print(f"Maximum of the egde detected : {np.amax(edge)}")
