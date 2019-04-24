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
import numpy.polynomial.polynomial as poly



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


def poly_reg(x,y):
  # returns a polynomial fit to given x and y points
  coefs = poly.polyfit(x, y, 2)
  ffit = poly.Polynomial(coefs)
  return ffit


def apply_poly(x,poly):
  # returns corresponding y points to the x points given, given the polynomial function
  x_new = np.linspace(x[0], x[-1], num=len(x))
  exp_x_new = np.expand_dims(x_new,axis=1)
  exp_y_new = np.expand_dims(poly(x_new),axis=1)
  return np.concatenate([exp_x_new,exp_y_new],axis=1)


def RANSAC_poly(x, y, max_iter, eps):
    # Input:
    #     x: set of x points
    #     y: set of y points
    #     max_iter: max iteration number of RANSAC
    #     eps: tolerance of RANSAC
    # Output:
    #     inliers_id: the indices of matched pairs when using the homography given by RANSAC
    #     poly: the polynomial regressor
    n = x.shape[0]
    inliers_id = []
    max_inliers = 0
    poly = None
    for i in range(max_iter):
        idxs = np.random.randint(n, size=4)
        src_pts = np.array([x[idxs[0]], x[idxs[1]], x[idxs[2]], x[idxs[3]]])
        dst_pts = np.array([y[idxs[0]], y[idxs[1]], y[idxs[2]], y[idxs[3]]])
        temp_poly = poly_reg(src_pts, dst_pts)
        fit_pts = apply_poly(x, temp_poly)
        temp_inliers = []
        temp_max = 0
        for j in range(n):
            dist = np.linalg.norm(fit_pts[j][1] - y[j])
            if dist <= eps:
                temp_inliers.append(j)
                temp_max += 1
        if temp_max > max_inliers:
            inliers_id = temp_inliers
            max_inliers = temp_max
            poly = temp_poly

    return inliers_id, poly

def exaggerated_poly(gauss20, exaggeration=100, ):
    #Outputs:
    #  x_poly - list of x points for exaggerated poly
    #  y_poly_final - list of y points for exaggerated poly
    #  idxs_to_adjust - list of indexs of x_poly where jump occurs

    # x and y inputs for RANSAC
    x = np.linspace(0, len(gauss20), num=len(gauss20))
    y = np.asarray(gauss20)

    # poly = polynomial fit by RANSAC
    _, poly = RANSAC_poly(x, y, 1000, 3)
    # y points for polynomial
    y_poly = poly(x)

    # find where dunk occurs by looking at where gauss20 intersects with polynomial
    idxs_to_adjust = []

    for idx, pt in enumerate(gauss20):
        if np.absolute(pt - y_poly[idx]) < 5:
            idxs_to_adjust.append(idx)

    # Take left most point, highest point + exaggeration (but minus cause images), right most point, then fit new polynomial to those

    xs_fit = [idxs_to_adjust[0], np.argmin(y_poly), idxs_to_adjust[-1]]
    ys_fit = [y_poly[idxs_to_adjust[0]], np.min(y_poly) - exaggeration, y_poly[idxs_to_adjust[-1]]]

    new_poly = poly_reg(xs_fit, ys_fit)

    # final polynomial
    y_poly_final = new_poly(x)
    return x, y_poly_final, idxs_to_adjust


if __name__ == '__main__':
    stabilized_gif = imageio.mimread('russ_dunk/stabilized.gif', memtest=False)

    # get smoothed trajectory
    gauss20 = center_of_mass(stabilized_gif, debug=0, folder='russ_dunk')

    # get exaggerated polynomial
    x,y,idxs_to_adjust = exaggerated_poly(gauss20)






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