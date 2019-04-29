# import the necessary packages
import argparse
import cv2


def click(event, x, y, flags, param):
    # grab references to the global variables
    global refPt

    # if the left mouse button was clicked, record the
    # (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = (x, y)

def click_window(image):
    # initialize the list of reference points
    #refPt = ()

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    # close window
    cv2.destroyAllWindows()
    return refPt

if __name__ == '__main__':


    xy = click_window()

