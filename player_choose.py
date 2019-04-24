# import the necessary packages
import argparse
import cv2


def click(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the
    # (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = (x, y)

def click_window():
    # initialize the list of reference points
    refPt = ()

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(args["image"])
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    # close all open windows
    cv2.destroyAllWindows()
    return refPt

if __name__ == '__main__':


    xy = click_window()

