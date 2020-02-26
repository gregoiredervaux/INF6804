import numpy as np
import cv2
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import os
import math

data_path = "Data/training/image_2/"
img1_name = data_path + '000000' + '_10.png'
img2_name = data_path + '000000' + '_11.png'
image1 = cv2.imread(img1_name)[0:99, 0:99]
image2 = cv2.imread(img2_name)[0:99, 0:99]

fig = plt.figure(figsize=(16,16))

premier = fig.add_subplot(1,2,1)
premier_plot = plt.imshow(image1)
premier.set_title('Première')

second = fig.add_subplot(1,2,2)
second_plot = plt.imshow(image2)
second.set_title('Seconde')
plt.show()

fd1, hog_image1 = hog(image1, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
# Rescale histogram for better display
hog_image_rescaled1 = np.array(exposure.rescale_intensity(hog_image1, in_range=(0, 10)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image2)
ax1.set_title('Input image')

ax2.axis('off')
ax2.imshow(hog_image_rescaled1, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

fd2, hog_image2 = hog(image2, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
# Rescale histogram for better display
hog_image_rescaled2 = np.array(exposure.rescale_intensity(hog_image2, in_range=(0, 10)))

# TODO: faire l'algo de cost_volume basé sur descripteur + adapter sgm pour prendre nos cost volumes

#  --left [LEFT IMAGE NAME] --right [RIGHT IMAGE NAME] --left_gt [LEFT GT IMAGE NAME] --right_gt [RIGHT GT IMAGE NAME] --output [OUTPUT IMAGE NAME] --disp [MAXIMUM DISPARITY] --images [TRUE OR FALSE] --eval [TRUE OR FALSE]
script = "semi_global_matching/sgm.py  --left %s --right %s --disp 64 --images False --eval False" % (img1_name, img2_name)
os.system("bash -c '%s'" % script)


