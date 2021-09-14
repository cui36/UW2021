import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
img = cv2.imread("b.jpg")
print(img.shape)
plt.subplot(131)
plt.imshow(img)
plt.xticks([])
plt.yticks([])

imgResize = cv2.resize(img,(300,200))
plt.subplot(132)
plt.imshow(imgResize)
plt.xticks([])
plt.yticks([])