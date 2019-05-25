import cv2
import imageio
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy as copy
import scipy.ndimage
import numpy.polynomial.polynomial as poly
from statistics import median as med


#Max
def get_centroid_of_mask_pts(points):
#Input: x,y points of mask
#Output: centroid of given points
    x = points[:,0]
    y = points[:,1]
    return (np.sum(x) / len(x), np.sum(y) / len(y))

#Max
def get_smoothed_trajectory(masks, Hs, gauss_num):
#Input: masks from original gif, Hs
#Output: x positions, adjusted y positions
    centroids = []
    for idx,mask in enumerate(masks):
        if(idx == 0): pass
        # convert mask to only xy points of player
        mask_pts = zip(*np.nonzero(mask))
        mask_pts = list(mask_pts)
        mask_pts = np.asarray(mask_pts, dtype=np.float32)
        mask_pts = mask_pts.reshape(-1, 1, 2) #reshape for perspectiveTransform
        # transform points based on homography
        transformed_mask_pts = cv2.perspectiveTransform(mask_pts,Hs[idx-1])
        transformed_mask_pts = transformed_mask_pts.reshape(-1,2) #reshape back to normal
        # get centroid
        centroids.append(get_centroid_of_mask_pts(transformed_mask_pts))
    centroids = np.asarray(centroids)


    import matplotlib

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_plot = np.linspace(0, len(masks), num=len(masks))
    ax.plot(x_plot,centroids[:,1])
    fig.savefig('centroids.png')



    gauss20 = scipy.ndimage.filters.gaussian_filter1d(centroids[:,1], gauss_num)
    return centroids[:,0], gauss20


def get_centroid(frame):
#Input: grayscale image (0-255 and 1 channel)
# Output: centroid of image

    ret, thresh = cv2.threshold(frame, 1, 255, 0)

    # calculate moments of binary image
    M = cv2.moments(thresh)

    try:
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # cX, cY = scipy.ndimage.measurements.center_of_mass(i)
        return (cX, cY)
    except:
        return False


#Roop
def center_of_mass(gif, gauss, debug=0, folder='./russ_dunk'):
    c = []
    for i in gif:
        # convert image to grayscale image
        if get_centroid(i):
            c.append(get_centroid(i))
        else:
            c.append(c[-1])

    c = np.array(c)

    gauss_0 = c[:, 1]
    gauss20 = scipy.ndimage.filters.gaussian_filter1d(gauss_0, gauss)

    if debug:
        new_gif = []
        gauss_5 = scipy.ndimage.filters.gaussian_filter1d(gauss_0, 5)
        gauss10 = scipy.ndimage.filters.gaussian_filter1d(gauss_0, 10)
        gauss15 = scipy.ndimage.filters.gaussian_filter1d(gauss_0, 15)
        a = gauss_0
        median = [int(med(a[0:2]))]
        for i in range(1, len(a) - 1):
            median.append(med(a[i - 1:i + 2]))
        median.append(int(med(a[len(a) - 2:len(a)])))
        for i in range(len(gif)):
            frame = np.copy(gif[i])
            for j in range(i):
                cv2.circle(frame, (j + (1000 // len(gif)) * j, gauss20[j] - 50), 5, (0, 0, 255), -1)
                cv2.circle(frame, (j + (1000 // len(gif)) * j, gauss15[j]), 5, (0, 255, 255), -1)
                cv2.circle(frame, (j + (1000 // len(gif)) * j, gauss10[j] + 50), 5, (0, 255, 0), -1)
                cv2.circle(frame, (j + (1000 // len(gif)) * j, gauss_5[j] + 100), 5, (255, 255, 0), -1)
                cv2.circle(frame, (j + (1000 // len(gif)) * j, gauss_0[j] + 150), 5, (255, 0, 0), -1)
                cv2.circle(frame, (j + (1000 // len(gif)) * j, median[j] + 200), 5, (255, 0, 255), -1)
            new_gif.append(frame)

        imageio.mimsave(folder + '/centroid_lines.gif', new_gif)

    return c[:, 0], gauss20, c[:,1]


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

def exaggerated_poly(gauss20, exaggeration=50):
    #Outputs:
    #  x_poly - list of x points for exaggerated poly
    #  y_poly_final - list of y points for exaggerated poly
    #  idxs_to_adjust - list of indexs of x_poly where jump occurs

    # x and y inputs for RANSAC
    x = np.linspace(0, len(gauss20), num=len(gauss20))
    y = np.asarray(gauss20)

    # poly = polynomial fit by RANSAC
    _, poly = RANSAC_poly(x, y, 1000, 5)
    # y points for polynomial
    y_poly = poly(x)

    # find where dunk occurs by looking at where gauss20 intersects with polynomial
    idxs_to_adjust = []

    for idx, pt in enumerate(gauss20):
        if 0 < pt - y_poly[idx] < 5:
            idxs_to_adjust.append(idx)

    # Take left most point, highest point + exaggeration (but minus cause images), right most point, then fit new polynomial to those

    xs_fit = [idxs_to_adjust[0], np.argmin(y_poly), idxs_to_adjust[-1]]
    ys_fit = [y_poly[idxs_to_adjust[0]], np.min(y_poly) - exaggeration, y_poly[idxs_to_adjust[-1]]]

    new_poly = poly_reg(xs_fit, ys_fit)

    # final polynomial
    y_poly_final = new_poly(x)
    return x, y_poly_final, idxs_to_adjust, y_poly


def overlay_pic(background, overlay):
    return_image = np.empty_like(background)
    for i in range(len(background)):
        for j in range(len(background[i])):
            return_image[i][j] = overlay[i][j][:3] if sum(overlay[i][j][:3]) > 0 else background[i][j]
    return return_image

def shaped_mask(m):
    r = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        r[:,:,i] = m.copy()
    return np.array(m, dtype=np.uint8)


def move_mask(mask_pts, background, player, y_adj):
  new_frame = np.array(background)
  for pt in mask_pts:
      if pt[0] - y_adj >= 0:
          new_frame[pt[0] - y_adj,pt[1]] = player[pt[0],pt[1]]
  return new_frame


def overlay_gif(original_gif, Hs, masks, xs, ys, jump_start_frame_num, jump_end_frame_num, stab_gif, stab_gif_masks):
    overlayed = []
    # reformat xs and ys
    adj_centroids = np.zeros((len(xs), 2))
    adj_centroids[:, 0], adj_centroids[:, 1] = xs, ys

    y_start = 0

    filled_points = {}

    stab_gif_3 = np.array(stab_gif)[:, :, :, :3]
    stab_gif_masks = np.array(stab_gif_masks)
    print('inpainting background...')
    for i in tqdm(range(len(original_gif))):
        if jump_start_frame_num <= i <= jump_end_frame_num:
            # remove background from image
            player = cv2.bitwise_and(original_gif[i], original_gif[i], mask=shaped_mask(masks[i]))
            # remove player from image
            background = cv2.bitwise_and(original_gif[i], original_gif[i], mask=1 - shaped_mask(masks[i]))

            mask_pts = zip(*np.nonzero(masks[i]))
            mask_pts = list(mask_pts)
            mask_pts = np.asarray(mask_pts)
            dilated = scipy.ndimage.morphology.binary_dilation(masks[i], iterations=55)
            new_mask_pts = zip(*np.nonzero(dilated))
            new_mask_pts = list(new_mask_pts)
            new_mask_pts = np.asarray(new_mask_pts)

            '''
            # get all xy points on mask
            frame_3_deep = player[:, :, :3]
            mask_pts = zip(*np.nonzero(frame_3_deep))
            mask_pts = list(mask_pts)
            mask_pts = np.asarray(mask_pts)
            
            by, bx, bheight, bwidth = cv2.boundingRect(mask_pts[:,:2].reshape(-1,1,2))
            # calculate box around mask pts
            new_mask_pts = []
            for x in range(bx-30,bx+bwidth+30):
                for y in range(by-30,by+bheight+30):
                    new_mask_pts.append((y,x))
            '''

            # transform points based on inverse homography
            adj_centroid = np.asarray(adj_centroids[i], dtype=np.float32).reshape(-1, 1, 2) # reshape for perspectiveTransform
            transformed_centroid = cv2.perspectiveTransform(adj_centroid, np.linalg.inv(Hs[i]))
            transformed_centroid = transformed_centroid.reshape(2)  # reshape back to normal
            transformed_centroid = np.array(transformed_centroid, dtype=np.uint8)
            # calculate y_adj based on transformed parabola
            if i == jump_start_frame_num:
                y_start = int(transformed_centroid[1])
            y_adj = abs(y_start - int(transformed_centroid[1]))


            # FILL BLACK PARTS
            for pt in new_mask_pts:
                pt = pt[:2]
                y = pt[0]
                x = pt[1]
                pt_xy = np.array([x,y])
                if not(0 < y < background.shape[0]) or not(0 < x < background.shape[1]):
                    continue
                transformed_pt = cv2.perspectiveTransform(np.array(pt_xy,dtype=np.float32).reshape(-1,1,2), Hs[i]).reshape(2)
                y_h = int(transformed_pt[1])
                x_h = int(transformed_pt[0])
                if (y_h,x_h) in filled_points:
                    background[y,x][:3] = filled_points[(y_h,x_h)]
                else:
                    rgb = stab_gif_3[background_pixel(stab_gif_3[:,y_h,x_h], stab_gif_masks[:,y_h,x_h])][y_h][x_h]
                    filled_points[(y_h,x_h)] = rgb
                    background[y,x][:3] = rgb

            # move mask up
            adj_player = move_mask(mask_pts, background, player, y_adj)

            overlayed.append(adj_player)
        else:
            overlayed.append(original_gif[i])

    return overlayed


'''
def overlay_gif(original_gif, Hs, masks, xs, ys, jump_start_frame_num, jump_end_frame_num, stab_gif):
    overlayed = []
    # reformat xs and ys
    adj_centroids = np.zeros((len(xs), 2))
    adj_centroids[:, 0], adj_centroids[:, 1] = xs, ys

    y_start = 0

    bp = {}

    a = np.array(stab_gif)[:, :, :, :3]
    print('generating background image...')
    background_npy = generate_background_image(stab_gif)
    print('inpainting background...')
    for i in tqdm(range(len(original_gif))):
        if jump_start_frame_num <= i <= jump_end_frame_num:
            # remove background from image
            player = cv2.bitwise_and(original_gif[i], original_gif[i], mask=shaped_mask(masks[i]))
            # remove player from image
            background = cv2.bitwise_and(original_gif[i], original_gif[i], mask=1 - shaped_mask(masks[i]))

            # move mask up based on y adj
            frame_3_deep = player[:, :, :3]
            mask_pts = zip(*np.nonzero(frame_3_deep))
            mask_pts = list(mask_pts)
            mask_pts = np.asarray(mask_pts)
            # transform points based on inverse homography
            adj_centroid = np.asarray(adj_centroids[i], dtype=np.float32).reshape(-1, 1, 2) # reshape for perspectiveTransform
            transformed_centroid = cv2.perspectiveTransform(adj_centroid, np.linalg.inv(Hs[i]))
            transformed_centroid = transformed_centroid.reshape(2)  # reshape back to normal
            transformed_centroid = np.array(transformed_centroid, dtype=np.uint8)
            # calculate y_adj based on transformed parabola
            if i == jump_start_frame_num:
                y_start = int(transformed_centroid[1])
            y_adj = abs(y_start - int(transformed_centroid[1]))
            # move mask up
            adj_player = move_mask(mask_pts, background, player, y_adj)



            # FILL BLACK PARTS
            filled_image = np.empty_like(adj_player)
            asdf = cv2.warpPerspective(background_npy, np.linalg.inv(Hs[i - 1]), background_npy.shape[:2][::-1])
            for y in range(len(adj_player)):
                for x in range(len(adj_player[y])):
                    if sum(adj_player[y][x][:3]):
                        filled_image[y][x] = adj_player[y][x]
                    else:
                        to_project = np.asarray([x, y], dtype=np.float32).reshape(-1, 1, 2)

                        if y < len(asdf) and x < len(asdf[0]):
                            filled_image[y][x] = asdf[y][x]
                        #
                        # wx, wy = np.array(cv2.perspectiveTransform(to_project, np.linalg.inv(Hs[i - 1]))[0][0], dtype=np.uint8)
                        # if (wy, wx) not in bp:
                        #     bp[(wy, wx)] = stab_gif[background_pixel(a[:,wy,wx], wy, wx)][wy][wx]
                        # filled_image[y][x] = bp[(wy, wx)]
            # overlay and append to gif
            overlayed.append(filled_image)

            #overlayed.append(adj_player)
        else:
            overlayed.append(original_gif[i])

    return overlayed
'''


def generate_background_image(stabilized_gif):
    a = np.array(stabilized_gif)[:, :, :, :3] # 80x300x640x3

    c = [[[] for i in range(a.shape[2])] for j in range(a.shape[1])]
    for i, frame in tqdm(enumerate(a), total=len(a)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for row in range(len(frame)):
            for col in range(len(frame[row])):
                rgb = gray[row][col]
                if rgb != 0:
                    c[row][col] = sorted(c[row][col] + [(rgb, i)], key=lambda x: x[0])

    background_image = np.empty_like(stabilized_gif[0])
    for i in range(background_image.shape[0]):
        for j in range(background_image.shape[1]):
            background_image[i][j] = stabilized_gif[c[i][j][len(c[i][j])//2][1]][i][j]

    # plt.imshow(background_image)
    # plt.show()

    # np.save('background.npy', background_image)
    return background_image

def generate_background_image_max(stabilized_gif):
    a = np.array(stabilized_gif)[:, :, :, :3] # 80x300x640x3

    c = [[[] for i in range(a.shape[2])] for j in range(a.shape[1])]
    for i, frame in tqdm(enumerate(a), total=len(a)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for row in range(len(gray)):
            for col in range(len(gray[row])):
                rgb = gray[row][col]
                if rgb != 0:
                    c[row][col] = sorted(c[row][col] + [(rgb, i)], key=lambda x: x[0])

    background_image = np.empty_like(stabilized_gif[0])
    for i in range(len(c)):
        for j in range(len(c[i])):
            background_image[i][j] = stabilized_gif[c[i][j][len(c[i][j])//2][1]][i][j]

    # np.save('background.npy', background_image)
    return background_image


def background_pixel(stabilized_gif_pt, stabilized_gif_masks_pt):
    median_values = []
    for i, v in enumerate(stabilized_gif_pt):
        rgb = np.sum(v)
        if stabilized_gif_masks_pt.shape[0] > i:
            mask_present = stabilized_gif_masks_pt[i]
        else:
            mask_present = False
        if rgb != 0 and not mask_present:
            median_values.append((rgb, i))
    s = sorted(median_values, key=lambda x: x[0])
    if len(s) > 0:
        idx = s[len(s)//2][1]
        return idx
    else:
        return 0

if __name__ == '__main__':
    stabilized_gif = imageio.mimread('blake_dunk/stabilized.gif', memtest=False)

    import time
    start = time.time()
    background_image = generate_background_image_max(stabilized_gif)
    end = time.time()

    print(end-start)

    plt.imshow(background_image)
    plt.show()

    # get smoothed trajectory
    #gauss20 = center_of_mass(stabilized_gif, debug=0, folder='russ_dunk')

    # get exaggerated polynomial
    #x,y,idxs_to_adjust = exaggerated_poly(gauss20)
