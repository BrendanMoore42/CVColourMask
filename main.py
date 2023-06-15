"""
OpenCV Colour Mask
Create realtime fun colour changing backgrounds for any webcam or movie source, or create
a sketch drawn effect, or pixelate the cam, any other interesting possibilities.

author: brendanmoore42@github.com
date: June 2023
"""
import cv2
import time
import numpy as np

def change_hue(im, frame):
    # Convert image to HSV
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # Modifying the frame*modifier will change colours, speed of change, etc
    hsv[:, :, 0] = frame*5 % 255  # change hue value
    hsv[:, :, 1] = frame*2 % 255  # change saturation value
    # hsv[:, :, 2] = 200  # changes light value, but not useful

    return hsv


def main(hue=False, pixelate=False, reduce=False):
    """Set hue True for colour changing, pixelate for changing pixelated size,
    and reduce colours for smoothing out returned image"""
    count = 0
    webcam = cv2.VideoCapture(0)

    while True:
        count += 0.5  # this value will change how quickly the colours change

        # Retrieve frame from webcam
        (_, image) = webcam.read()

        # Flip image (optional)
        output = cv2.flip(image, 1)

        if pixelate:  # input as tuple of dimensions - ex. (780, 780)
            # Get input image size
            height, width = image.shape[:2]

            # Pixelate image to desired size
            # p_height, p_width = pixelate
            p_image = cv2.resize(output, pixelate, interpolation=cv2.INTER_LINEAR)

            # Resize to output
            output = cv2.resize(p_image, (width, height), interpolation=cv2.INTER_LINEAR)

        if hue:
            # Run image through recolour function
            output = change_hue(output, count)  # use frame count to change colour
            # output = change_hue(output, int(round(time.time())))  # use time to change colour

        if reduce:
            # Reduce colours
            output = output // 100 * 100 + 100 // 2

        # Press q to quit
        if cv2.waitKey(1) == ord("q"):
            break

        # Flip image horizontally
        output = cv2.flip(output, 1)

        # Display image
        cv2.imshow("Output", output)

    # On close, destroy the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting webcam")

    # Example
    main(hue=True, pixelate=(500, 500), reduce=True)

    print("Closing webcam")
