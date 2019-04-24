import cv2
import imageio
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy as copy
import scipy.ndimage


def center_of_mass(gif, debug=0, folder='russ_dunk'):
    c = []
    for i in gif:
        # convert image to grayscale image
        img = copy(i)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # convert the grayscale image to binary image
        ret, thresh = cv2.threshold(gray_image, 1, 255, 0)

        # calculate moments of binary image
        M = cv2.moments(thresh)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        c.append((cX, cY))
    c = np.array(c)

    gauss_0 = c[:, 1]
    gauss20 = scipy.ndimage.filters.gaussian_filter1d(gauss_0, 20)

    if debug:
        new_gif = []
        gauss_5 = scipy.ndimage.filters.gaussian_filter1d(gauss_0, 5)
        gauss10 = scipy.ndimage.filters.gaussian_filter1d(gauss_0, 10)
        gauss15 = scipy.ndimage.filters.gaussian_filter1d(gauss_0, 15)
        from statistics import median as med
        a = gauss_0
        median = [int(med(a[0:2]))]
        for i in range(1, len(a) - 1):
            median.append(med(a[i - 1:i + 2]))
        median.append(int(med(a[len(a) - 2:len(a)])))
        for i in range(len(gif)):
            frame = np.copy(gif[i])
            for j in range(i + 1):
                cv2.circle(frame, (j + 1300 + (1300 // len(gif)) * j, gauss20[j] - 100), 5, (0, 0, 255), -1)
                cv2.circle(frame, (j + 1300 + (1300 // len(gif)) * j, gauss15[j]), 5, (0, 255, 255), -1)
                cv2.circle(frame, (j + 1300 + (1300 // len(gif)) * j, gauss10[j] + 100), 5, (0, 255, 0), -1)
                cv2.circle(frame, (j + 1300 + (1300 // len(gif)) * j, gauss_5[j] + 200), 5, (255, 255, 0), -1)
                cv2.circle(frame, (j + 1300 + (1300 // len(gif)) * j, gauss_0[j] + 400), 5, (255, 0, 0), -1)
                cv2.circle(frame, (j + 1300 + (1300 // len(gif)) * j, median[j] + 300), 5, (255, 0, 255), -1)
            new_gif.append(frame)

        imageio.mimsave(folder + '/centroid_lines.gif', new_gif)

    return gauss20


if __name__ == '__main__':
    stabilized_gif = imageio.mimread('russ_dunk/stabilized.gif', memtest=False)


    # def shaped_mask(m):
    #     r = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
    #     for i in range(3):
    #         r[:, :, i] = m.copy()
    #     return np.array(m, dtype=np.uint8)
    #
    #
    # def intersection_count(m1, m2):
    #     return np.sum(np.logical_and(m1, m2))


    # nice_masks0 = np.load('../r-cnn_masks/nice_mask0.npy')
    # previous_player_mask_idx = np.argmax([np.sum(i) for i in nice_masks0])
    #
    # images = []
    # image = skimage.io.imread('../russ_dunk_images/russ_dunk_frame0.png')
    # for i in tqdm(range(1, 260)):
    #     images.append(cv2.bitwise_and(image, image, mask=shaped_mask(nice_masks0[previous_player_mask_idx])))
    #     image = skimage.io.imread('../russ_dunk_images/russ_dunk_frame' + str(i) + '.png')
    #     nice_masks1 = np.load('../r-cnn_masks/nice_mask' + str(i) + '.npy')
    #     previous_player_mask_idx = np.argmax(
    #         [intersection_count(nice_masks0[previous_player_mask_idx], i) for i in nice_masks1])
    #     nice_masks0 = nice_masks1