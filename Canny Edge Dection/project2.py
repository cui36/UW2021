'''
cited resources:
1. https://www.docs.opencv.org/master/da/d22/tutorial_py_canny.html#:~:text=Canny%20Edge%20Detection%20is%20a%20popular%20edge%20detection,in%20the%20image%20with%20a%205x5%20Gaussian%20filter.
2.  https://blog.csdn.net/weixin_34319111/article/details/92657381
3. https://blog.csdn.net/weixin_40877924/article/details/103751936
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math

# *** TODO: Read a image in grayscale (you may use your own photo!) ***
image = cv2.imread("b.jpg", 0)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# Step 1: Image denoising
# *** TODO: Apply Gaussian filter using OpenCV GaussianBlur and display the original and blurred image ***
image_blur = cv2.GaussianBlur(image, (5, 5), 0)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("original image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(image_blur, cmap="gray")
plt.title("blurred image")
plt.axis('off')
plt.show()

# Step 2: Find the intensity gradient
# *** TODO: Calculate the intensity gradient G and direction theta. ***
# ***       It's better to normalize G to [0,255]. ***

# sober算子
S_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
S_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
height_0,weight_0 = image.shape

# convolution
# border
img2 = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_REPLICATE)
tmp = img2.copy().astype(np.float)
I1 = np.zeros((height_0+2,weight_0+2), dtype=np.float)
I2 = np.zeros((height_0+2,weight_0+2), dtype=np.float)
for h in range(height_0):
    for w in range(weight_0):
        I1[h,w] = np.sum(S_x*tmp[h:h+3, w:w+3])
        I2[h,w] = np.sum(S_y*tmp[h:h+3, w:w+3])
I1 = np.clip(I1, 0, 255)
I2 = np.clip(I2, 0, 255)
G = np.sqrt(I1*I1+I2*I2)


# Step 3: Non-maximum suppression
def non_maximum_suppression(G):
    I_copy = np.zeros(G.shape)
    G_plus = cv2.copyMakeBorder(G, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    anchor = np.where(G_plus != 0)

    for i in range(len(anchor[0])):
        x = anchor[0][i]
        y = anchor[1][i]
        alter_point = G[x, y]
        g1 = G[x - 1, y - 1]
        g2 = G[x + 1, y - 1]
        g3 = G[x - 1, y + 1]
        g4 = G[x + 1, y + 1]
        if g1 < alter_point and g2 < alter_point and g3 < alter_point and g4 < alter_point:
            I_copy[x, y] = alter_point

    # img_uint8 = I_copy.astype(np.uint8)
    return I_copy
sup_G = non_maximum_suppression(G)
# *** TODO: Display the original gradient image and suppressed gradient image. ***
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(G, cmap="gray")
plt.title('gradient G')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(sup_G, cmap="gray")
plt.title('suprressed gradient G')
plt.axis('off')
plt.show()

# Step 4: Apply double threshold to determine potential edges
# *** TODO: Generate a strong and a week edge map. ***
threshold_low = 20
threshold_high = 80
strong_map = (sup_G >= threshold_high) * 1.
weak_map = ((sup_G < threshold_high) & (sup_G >= threshold_low)) * 1.
# *** TODO: Display the strong and week edge map. ***
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(strong_map, cmap="gray")
plt.title('strong edge map')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(weak_map, cmap="gray")
plt.title('week edge map')
plt.axis('off')
plt.show()


# Step 5: Track edge by hysteresis
def track_edge(strong, weak):
    height, width = strong.shape
    output = np.zeros_like(strong)  # initiate the final edge map

    # loop on each point (ignore the first&last row and first&last column)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # *** TODO: Decide if the target pixel is edge or not. ***
            if strong[i, j] == 1:
                output[i, j] = 255
            elif weak[i, j] == 1 and np.sum(strong[i - 1:i + 2, j - 1:j + 2]) > 0:
                output[i, j] = 255
    return output

final_edge_map = track_edge(strong_map, weak_map)

# *** TODO: Display the final edge map. ***
plt.figure()
plt.imshow(final_edge_map, cmap="gray")
plt.title('final edge map')
plt.axis('off')
plt.show()