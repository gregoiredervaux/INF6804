import sys
import time as t

import numpy as np
from skimage.feature import hog, BRIEF


def dense_hog(img, dsc_size, num_orientations):

    rows, cols = img.shape
    img_hog = np.zeros((rows, cols, num_orientations))

    for i in range(rows):
        for j in range(cols):
            local_img = img[max(0, i - int(dsc_size / 2)):min(rows, i + int(dsc_size / 2)), max(0, j - int(dsc_size / 2)):min(cols, j + int(dsc_size / 2))]
            img_hog[i,j,:] = hog(local_img, orientations=num_orientations, pixels_per_cell=(local_img.shape[0], local_img.shape[1]), cells_per_block=(1,1))
    return img_hog

def apply_hog(left, right, descriptor_size, num_orientations):
    """
    computes HOG descriptor on both images.
    :param left: left image.
    :param right: right image.
    :param descriptor_size: number of pixels in a hog cell.
    :param num_orientations: number of HOG orientations.
    :return: (H x W x M) array, H = height, W = width and M = num_orientations, of type np.float32.
    """

    return (dense_hog(left, descriptor_size, num_orientations), dense_hog(right, descriptor_size, num_orientations))


def apply_brief(left, right, descriptor_size, num_elements):
    """
    computes BRIEF descriptor on both images.
    :param left: left image.
    :param right: right image.
    :param descriptor_size: size of window of the BRIEF descriptor.
    :param num_elements: length of the feature vector.
    :return: (H x W) array, H = height and W = width, of type np.int64
    """
    # TODO: apply BRIEF descriptor on both images. You will have to convert the BRIEF feature vector to a int64.


