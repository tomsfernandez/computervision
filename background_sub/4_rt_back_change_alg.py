import numpy as np
import cv2
from SubstractorProvider import SubstractorProvider

# OpenCV extra modules must be installed to be able to run this sim.

cap = cv2.VideoCapture(0)
title_window = "Substraction"
slider_max = 255
learning_rate = 0.5
cv2.namedWindow(title_window, cv2.WINDOW_NORMAL)
ret, frame = cap.read()
rows, cols, channels = frame.shape

provider = SubstractorProvider()
algorithm = provider.get_algorithm("5")


def on_trackbar(val):
    global learning_rate
    learning_rate = float(val) / slider_max
    print(f"Using learning rate {learning_rate}")


cv2.createTrackbar('Learning Rate Trackbar', title_window, 0, slider_max, on_trackbar)

while 1:
    _, frame = cap.read()
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = algorithm.apply(grey_image, None, learning_rate)
    background = algorithm.getBackgroundImage()
    cv2.imshow('Original', grey_image)
    cv2.imshow(title_window, mask)
    cv2.imshow('Background', background)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
