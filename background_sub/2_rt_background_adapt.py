import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
title_window = "Substraction"
slider_max = 255
learning_rate = 0.5
cv2.namedWindow(title_window, cv2.WINDOW_NORMAL)

def on_trackbar(val):
    global learning_rate
    learning_rate = float(val)/slider_max
    print(f"Using learning rate {learning_rate}")
cv2.createTrackbar('Learning Rate Trackbar', title_window , 0, slider_max, on_trackbar)


while(1):
    ret, frame = cap.read()
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(grey_image, None, learning_rate)
    cv2.imshow('original', grey_image)
    cv2.imshow(title_window, fgmask)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()