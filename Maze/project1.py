import cv2
import numpy as np
import matplotlib.pyplot as plt

############### Part (a) ###############
# Task 1: read an image (you may use your own photo!), resize image
# *** TODO: read a color image using OpenCV imread ***
# *** TODO: print image shape using numpy.ndarray.shape ***
# *** TODO: change the size of the image to 320x240 using OpenCV resize ***
# *** TODO: print the resized image shape using numpy.ndarray.shape ***

# Note: OpenCV imread() reads image as a NumPy array of size Height x Width x 3 with default parameters.
#       The order of color is BGR (blue, green, red).
#       To show image in commonly used color mode RGB, we need to convert image from BGR to RGB color mode.
#       We could also convert the image into other color modes. For example, RGB -> grayscale.
# *** TODO: convert the color space from BGR to RGB using OpenCV cvtColor ***
# *** TODO: convert the color space from BGR to grayscale using OpenCV cvtColor ***

# Task 2: display multiple images
plt.figure()
# *** TODO: display the RGB and grayscale image using matplotlib.pyplot.imshow ***
plt.show() 

# Task 3: save image
# *** TODO: save the grayscale image using OpenCV imwrite ***
########################################

############### Part (b) ###############
plt.figure()
# *** TODO: plot the histogram (16 bins) of the grayscale image using matplotlib.pyplot.hist ***
plt.show()

# *** TODO: adjust the brightness and contrast of the grayscale image using OpenCV convertScaleAbs ***
# *** You could choose any valid parameters for convertScaleAbs. *** 
# *** TODO: display the adjusted grayscale image and its histogram ***

# *** TODO: Generate the same adjusted image as convertScaleAbs without using any available functions ***
#increase the contrast and brightness

# *** TODO: Implement gamma correction without using any available functions ***
#increase the gamma

# *** TODO: Implement alpha-blending, which is a process to overlap two images through weighted sum. ***
# *** Let f and g be two grayscale images, the goal is to perform new_output = (1-alpha)*f+alpha*g with an alpha between 0 and 1. ***
########################################