import argparse
import os
import sys
import random
import math
import numpy as np
#import skimage.io
import cv2
import masking
import imageio
import player_choose
import stabilize
import exaggeration
import glob
import scipy.misc
from PIL import Image





if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gif", required=True, help="Path to gif")
    ap.add_argument("-f", "--savefolder", required=True, help="Path to save gif in")
    ap.add_argument("-m", "--model", required=True, help="Path to Mask R-CNN folder")
    ap.add_argument("-e", "--exaggeration", required=True, help="Exaggeration to apply")
    ap.add_argument("-l", "--load", required=False, help="Load masks and homographies")
    args = vars(ap.parse_args())

    save_folder = args['savefolder']
    path_to_model_folder = args["model"]
    path_to_gif = args["gif"]
    exag = int(args['exaggeration'])
    if args['load']:
        load = True
    else:
        load = False

    if not load:
        #Load Mask R-CNN Model

        print("loading Mask R-CNN model...")

        ROOT_DIR = os.path.abspath(path_to_model_folder)
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
    print("loading dunk gif...")
    gif = imageio.mimread(path_to_gif, memtest=False)

    if not load:
        print("click on player and press q to exit")
        # Open window and ask for xy coordinate contained in player
        xy = player_choose.click_window(gif[0])


    if load:
        print("loading masks...")
        masks = np.load(save_folder + '/masks.npy')
    else:
        # Get masks
        print("creating masks...")
        masks = masking.get_masks(gif, xy, model, save=True, folder=save_folder)

    print(masks[0].shape, gif[0][:,:,0].shape)
    assert(masks[0].shape == gif[0][:,:,0].shape)


    if load:
        print("loading homographies...")
        Hs = np.load(save_folder + '/Hs.npy')
    else:
        # Get homographies
        print("calculating and saving homographies...")
        Hs, stabilized_gif = stabilize.generate_hs(gif, folder='./russ_dunk')

    # Generate stabilized gif of masks
    stabilized_masks = stabilize.generate_stabilized_masks(masks, Hs)

    # Get poly function for exaggeration
    print("calculating exaggeration...")

    gauss_kernel = 20 if len(gif) > 200 else len(gif)//10
    xs, gauss = exaggeration.center_of_mass(stabilized_masks, gauss_kernel)


    _, ys, idxs_to_adjust = exaggeration.exaggerated_poly(gauss, exaggeration=exag)

    start_jump_frame_num = idxs_to_adjust[0]
    end_jump_frame_num = idxs_to_adjust[-1]


    # Exaggerate movement
    print("exaggerating...")
    adj_gif = exaggeration.overlay_gif(gif, Hs, masks, xs, ys, start_jump_frame_num, end_jump_frame_num)
    imageio.mimsave(save_folder+'/final.gif', adj_gif)








