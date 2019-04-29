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



def load_masks(filepath, save_pngs=True, save_gif=True):
    masks = []
    for idx,file in enumerate(list(glob.glob(filepath + '/*mask*.npy'))):
        mask = np.load(file)
        masks.append(mask)
        if save_pngs:
            scipy.misc.imsave('russ_dunk/' + str(idx) + '.jpg', np.load(file)*255)
    if save_gif:
        to_gif = np.asarray(masks) * 255
        imageio.mimsave('russ_dunk/masks.gif', to_gif)
    return masks



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gif", required=True, help="Path to gif")
    ap.add_argument("-m", "--model", required=True, help="Path to Mask R-CNN folder")
    ap.add_argument("-e", "--exaggeration", required=True, help="Exaggeration to apply")
    args = vars(ap.parse_args())

    path_to_model_folder = args["model"]
    path_to_gif = args["gif"]
    exag = int(args['exaggeration'])

    '''
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
    '''
    # Load gif
    print("loading dunk gif...")
    gif = imageio.mimread(path_to_gif, memtest=False)

    # Open window and ask for xy coordinate contained in player
    #xy = player_choose.click_window(gif[0])


    # Get masks
    print("creating masks...")
    #masks = masking.get_masks(gif, xy, model, save=True, folder='./blake_dunk')
    masks = load_masks('./blake_dunk', save_pngs=False,save_gif=False)


    # Get homographies
    print("calculating homographies...")
    #Hs = stabilize.generate_hs(gif, folder='./blake_dunk')
    Hs = np.load('blake_dunk/Hs.npy')

    # Get poly function for exaggeration
    print("calculating exaggeration...")

    gauss_kernel = 20 if len(gif) > 200 else len(gif)//10
    xs, gauss20 = exaggeration.get_smoothed_trajectory(masks, Hs, gauss_kernel)

    '''
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_plot = np.linspace(0, len(gif)-1, num=len(gif)-1)
    ax.plot(x_plot,gauss20)
    fig.savefig('gauss20.png')
    '''

    _, ys, idxs_to_adjust = exaggeration.exaggerated_poly(gauss20, exaggeration=exag)

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_plot = np.linspace(0, len(gif)-1, num=len(gif)-1)
    ax.plot(x_plot, ys)
    fig.savefig('adj_poly.png')
    '''

    start_jump_frame_num = idxs_to_adjust[0]
    end_jump_frame_num = idxs_to_adjust[-1]


    # Exaggerate movement
    print("exaggerating...")
    adj_gif = exaggeration.overlay_gif(gif, Hs, masks, xs, ys, start_jump_frame_num, end_jump_frame_num)
    imageio.mimsave('./blake_dunk/final.gif', adj_gif)








