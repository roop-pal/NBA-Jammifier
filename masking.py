import numpy as np
from tqdm import tqdm

import imageio
import scipy.misc
import os


def nice_mask(r):
    #Input: results of mask rcnn
    #Output: list of numpy arrays representing human masks

    # Remove non-person masks
    idx = 0
    for i in r['class_ids']:
        if not (i == 1):
            r['rois'] = np.delete(r['rois'], idx, 0)
            r['class_ids'] = np.delete(r['class_ids'], idx)
            r['scores'] = np.delete(r['scores'], idx)
            r['masks'] = np.delete(r['masks'], idx, 2)
        else:
            idx += 1

    nice_masks = np.array(np.dsplit(r['masks'], r['masks'].shape[-1])).reshape(
        (r['masks'].shape[-1], r['masks'].shape[0], r['masks'].shape[1]))
    return list(nice_masks)


# return intersection percentage between masks
def intersection_percentage(m1, m2):
    #Input:
    # m1,m2 = masks
    #Output:
    # percentage of intersecting pixels
  return np.sum(np.logical_and(m1,m2))/np.sum(m2)

# return intersection sum between masks
def intersection_count(m1, m2):
    #Input:
    # m1,m2 = masks
    #Output:
    # sum of intersecting pixels
  return np.sum(np.logical_and(m1,m2))

def combine_masks(masks):
    combined = np.zeros_like(masks[0])
    for mask in masks:
        combined = np.logical_or(combined,mask)
    return combined


# Main function to use in this file
def get_masks(gif, xy, mask_rcnn_model, save=True, folder= './russ_dunk'):
    #Input:
    # gif - iterable of images
    # xy - tuple of point contained in player
    # mask_rcnn_model
    # save - whether to save gif
    # save_path - where to save gif
    #Output:
    # list of player mask for each frame in gif
    masks = []
    for idx,frame in tqdm(enumerate(gif)):
        # Run detection
        frame = frame[:, :, :3]
        results = mask_rcnn_model.detect([frame], verbose=1)
        r = results[0]
        all_masks = nice_mask(r)

        if idx == 0:
            for mask in all_masks:
                if mask[xy[1],xy[0]] == True:
                    masks.append(mask)
        else:
            '''
            #for debugging
            try:
                os.makedirs("./russ_dunk/frame"+str(idx))
            except FileExistsError:
                # directory already exists
                pass
            for i,mask in enumerate(all_masks):
                scipy.misc.imsave('./russ_dunk/frame' + str(idx) + '/all_mask' + str(i)+ '.jpg', mask * 255)
            scipy.misc.imsave('./russ_dunk/frame' + str(idx) + '/prev_mask.jpg', masks[-1] * 255)
            '''

            # combine masks with highest intersection percentage to account for Westbrook losing legs corner case
            player_mask_idx = np.argmax([intersection_count(masks[-1], i) for i in all_masks])
            mask_potentials = [intersection_percentage(masks[-1], i) for i in all_masks]
            mask_potentials = [all_masks[mask_potentials.index(i)] for idx, i in enumerate(mask_potentials) if i > .8 and idx != player_mask_idx]
            mask_potentials.append(all_masks[player_mask_idx])
            combined_mask = combine_masks(mask_potentials)
            masks.append(combined_mask)

            '''
            #for debugging
            scipy.misc.imsave('./russ_dunk/frame' + str(idx) + '/combined_mask.jpg', combined_mask*255)
            np.save('./russ_dunk/mask_vault/mask_vault'+str(idx)+'.npy', combined_mask)
            scipy.misc.imsave('./russ_dunk/mask_vault/mask_vault'+str(idx)+'.jpg', combined_mask*255)
            '''


    #save if asked
    if save:
        for idx,mask in enumerate(masks):
            np.save(folder + '/mask' + str(idx) + '.npy', mask)
        to_gif = list(np.asarray(masks)*255)
        imageio.mimsave(folder + '/masks.gif', to_gif)

    return masks
