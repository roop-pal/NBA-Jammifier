# Written by Max Ogryzko and Roop Pal
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
from IPython import embed as e


def compute_homography(src, dst):
    # This function computes the homography from src to dst.
    #
    # Input:
    #     src: source points, shape (n, 2)
    #     dst: destination points, shape (n, 2)
    # Output:
    #     H: homography from source points to destination points, shape (3, 3)

    # Code here...

    A = np.zeros([2 * src.shape[0], 9])
    for i in range(src.shape[0]):
        A[2 * i, :] = np.array([src[i, 0], src[i, 1], 1, 0, 0, 0,
                                -dst[i, 0] * src[i, 0], -dst[i, 0] * src[i, 1], -dst[i, 0]])
        A[2 * i + 1, :] = np.array([0, 0, 0, src[i, 0], src[i, 1], 1,
                                    -dst[i, 1] * src[i, 0], -dst[i, 1] * src[i, 1], -dst[i, 1]])

    w, v = np.linalg.eig(np.dot(A.T, A))
    index = np.argmin(w)
    H = v[:, index].reshape([3, 3])
    return H


def apply_homography(src, H):
    # Applies a homography H onto the source points, src.
    #
    # Input:
    #     src: source points, shape (n, 2)
    #     H: homography from source points to destination points, shape (3, 3)
    # Output:
    #     dst: destination points, shape (n, 2)

    n, _ = src.shape
    ones = np.ones((n, 1))
    appended = np.append(src, ones, axis=1)  # n x 3 (added 1 to the end of each x,y point)
    dst = np.matmul(H, appended.T).T
    # divide x and y by z
    zs = dst[:, 2].reshape(n, 1)
    dst = dst / zs
    dst = dst[:, :2]

    return dst


def RANSAC(Xs, Xd, max_iter, eps, num_points=4):
    # Input:
    #     pts1: the first set of points, shape [n, 2]
    #     pts2: the second set of points matched to the first set, shape [n, 2]
    #     max_iter: max iteration number of RANSAC
    #     eps: tolerance of RANSAC
    # Output:
    #     inliers_id: the indices of matched pairs when using the homography given by RANSAC
    #     H: the homography, shape [3, 3]
    n, _ = Xs.shape
    inliers_id = []
    max_inliers = 0
    H = np.array([])

    for i in range(max_iter):
        idxs = np.random.randint(n, size=num_points)
        src_pts = np.array(Xs[idxs[:num_points]])
        dst_pts = np.array(Xd[idxs[:num_points]])
        temp_H = compute_homography(src_pts, dst_pts)
        temp_inliers = []
        temp_max = 0
        proj_pts = apply_homography(Xs, temp_H)
        for j in range(n):
            src_pt = np.append(Xs[j], 1)
            proj_pt = proj_pts[j]
            real_pt = Xd[j]
            dist = np.linalg.norm(proj_pt - real_pt)
            if dist <= eps:
                temp_inliers.append(j)
                temp_max += 1

        if temp_max > max_inliers:
            inliers_id = temp_inliers
            max_inliers = temp_max
            H = temp_H

    return inliers_id, H


def genSIFTMatchPairs(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.zeros((250, 2))
    pts2 = np.zeros((250, 2))
    for i in range(250):
        pts1[i, :] = kp1[matches[i].queryIdx].pt
        pts2[i, :] = kp2[matches[i].trainIdx].pt

    return pts1, pts2, matches[:250], kp1, kp2


#Roop
def generate_homographies(gif, debug=2, folder='russ_dunk'):
    stabilized_gif = [gif[0]]
    Hs = []
    for i in tqdm(range(1, len(gif))):
        if debug == 2 and i % 20 == 0:
            imageio.mimsave(folder + '/stabilized.gif', stabilized_gif)
        base = stabilized_gif[-1]
        img = gif[i]
        pts1, pts2, matches, kp1, kp2 = genSIFTMatchPairs(img, base)
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        #inliers_idx, H = RANSAC(pts1, pts2, 500, 20, 4) # 2000, 10, 10
        Hs.append(H)
        stabilized_gif.append(cv2.warpPerspective(img, H, img.shape[:2][::-1]))

    if debug == 1:
        np.save(folder + '/Hs.npy', Hs)

    return Hs, stabilized_gif

def generate_stabilized_masks(masks, Hs, height, width):
    stabilized_masks = []
    for i, mask in enumerate(masks):
        stabilized_masks.append(
                cv2.warpPerspective(np.array(mask * 255, dtype=np.uint8), Hs[i], (width, height)))
    return stabilized_masks

#Max
def generate_hs(gif, save=True, folder='russ_dunk'):

    # Find Bounds

    stabilized_gif = [gif[0]]
    prev_frame = gif[0]
    Hs = [np.identity(3)]
    height = gif[0].shape[0]
    width = gif[0].shape[1]
    adjs = []
    for i in tqdm(range(1, len(gif))):
        base = prev_frame
        img = gif[i]
        pts1, pts2, matches, kp1, kp2 = genSIFTMatchPairs(img, base)
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        nonzero = np.nonzero(img[:, :, 0])
        nonzero_pts = np.zeros((nonzero[0].shape[0], 2))
        nonzero_pts[:, 1] = nonzero[0]
        nonzero_pts[:, 0] = nonzero[1]
        new_nonzeros = cv2.perspectiveTransform(np.float32(nonzero_pts.reshape(-1,1,2)), H).reshape(-1,2)
        bx, by, bwidth, bheight = cv2.boundingRect(new_nonzeros)
        new_corners = np.array([[bx,by],[bx+bwidth,by],[bx,by+bheight],[bx+bwidth,by+bheight]])
        xy_adj = [0,0]
        if bx < 0:
            Ht = np.array([[1, 0, -bx],[0, 1, 0],[0, 0, 1]])
            H = Ht.dot(H)
            new_corners[:,0] -= bx
            xy_adj[0] = -bx
        if by < 0:
            Ht = np.array([[1, 0, 0], [0, 1, -by], [0, 0, 1]])
            H = Ht.dot(H)
            new_corners[:,1] -= by
            xy_adj[1] = -by
        adjs.append(xy_adj)
        if bwidth + np.max(bx,0) > width:
            width = bwidth + np.max(bx,0)
        if bheight + np.max(by,0) > height:
            height = bheight + np.max(by,0)
        prev_frame = cv2.warpPerspective(img, H, (width,height))
        stabilized_gif.append(prev_frame)
        Hs.append(H)
    max_height = 0
    max_width = 0
    for i,frame in enumerate(stabilized_gif):
        x_adj = np.sum(np.array(adjs)[i:,0])
        y_adj = np.sum(np.array(adjs)[i:,1])
        if max_height < frame.shape[0] + y_adj: max_height = frame.shape[0] + y_adj
        if max_width < frame.shape[1] + x_adj: max_width = frame.shape[1] + x_adj
    for i, frame in enumerate(stabilized_gif):
        x_adj = np.sum(np.array(adjs)[i:, 0])
        y_adj = np.sum(np.array(adjs)[i:, 1])
        new = np.zeros((max_height, max_width, frame.shape[2]))
        new[y_adj:y_adj+frame.shape[0], x_adj:x_adj+frame.shape[1]] = frame
        stabilized_gif[i] = new
        Ht = np.array([[1, 0, x_adj], [0, 1, y_adj], [0, 0, 1]])
        Hs[i] = Ht.dot(Hs[i])
    if save == True:
        np.save(folder + '/Hs.npy', Hs)
        imageio.mimsave(folder+'/stabilized.gif', stabilized_gif)
    return Hs, stabilized_gif, max_height, max_width


if __name__ == '__main__':
    source_gif = np.array(imageio.mimread('russ_dunk/russ_dunk_88.gif'))[:,:300]

    Hs, stabilized_gif = generate_homographies(source_gif)
