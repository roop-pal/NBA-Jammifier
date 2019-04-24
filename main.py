import argparse
import os
import sys
import random
import math
import numpy as np
import skimage.io
import cv2
import masking
import imageio
import player_choose





if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--gif", required=True, help="Path to gif")
    ap.add_argument("-m", "--model", required=True, help="Path to Mask R-CNN folder")
    args = vars(ap.parse_args())

    path_to_model = args["model"]
    path_to_gif = args["gif"]

    #Load Mask R-CNN Model

    ROOT_DIR = os.path.abspath(path_to_model)
    sys.path.append(ROOT_DIR)  # To find local version of the library
    import mrcnn.model as modellib
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    import coco
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
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
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # Load gif

    gif = imageio.mimread(path_to_gif, memtest=False)

    # Open window and ask for xy coordinate contained in player

    xy = player_choose.click_window(gif[0])

    # Get masks

    masks = masking.get_masks(gif, xy, model, save=True, save_path='./russ_gif') #creates folder called 'russ_gif' and adds masks there

    



