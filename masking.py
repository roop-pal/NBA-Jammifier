# SHOULD BE IN MAIN

import os
import sys
import random
import math
import numpy as np
import skimage.io
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("./Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import mrcnn.model as modellib

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# SHOULD BE IN MAIN^

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


# return intersection count between masks
def intersection_count(m1, m2):
    #Input:
    # m1,m2 = masks
    #Output:
    # number of intersecting pixels
  return np.sum(np.logical_and(m1,m2))


# Main function to use in this file
def get_masks(gif, xy, mask_rcnn_model, save=True, save_path= '.'):
    #Input:
    # gif - iterable of images
    # xy - tuple of point contained in player
    # mask_rcnn_model
    # save - whether to save gif
    # save_path - where to save gif
    #Output:
    # list of player mask for each frame in gif
    masks = []
    for idx,frame in enumerate(gif):
        # Run detection
        results = mask_rcnn_model.detect([frame], verbose=1)
        r = results[0]
        all_masks = nice_mask(r)

        if idx == 0:
            for mask in all_masks:
                if mask[xy[1],xy[0]] == True:
                    masks.append(mask)
        else:
            # append mask with highest intersection count
            player_mask_idx = np.argmax([intersection_count(masks[-1], i) for i in all_masks])
            masks.append(all_masks[player_mask_idx])

    #save if asked
    if save:
        for idx,mask in enumerate(masks):
            np.save(save_path + '/mask' + idx + '.npy', mask)

    return masks
