import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
title_window = "Substraction"
slider_max = 255
learning_rate = 0.5
cv2.namedWindow(title_window, cv2.WINDOW_NORMAL)
ret, frame = cap.read()
rows,cols,channels = frame.shape

def on_trackbar(val):
    global learning_rate
    learning_rate = float(val)/slider_max
    print(f"Using learning rate {learning_rate}")
cv2.createTrackbar('Learning Rate Trackbar', title_window , 0, slider_max, on_trackbar)


while(1):
    ret, frame = cap.read()
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(grey_image, None, learning_rate)
    background = fgbg.getBackgroundImage()
    cv2.imshow('Original', grey_image)
    cv2.imshow(title_window, fgmask)
    cv2.imshow('Background', background)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()