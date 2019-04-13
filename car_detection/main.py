import time

import cv2
import numpy as np


def remove_noise(image):
    erosion_kernel = np.ones((3, 3), np.uint8)
    dilation_kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(image, erosion_kernel, iterations=1)
    return cv2.dilate(img, dilation_kernel, iterations=1)


def draw_contour_center(contour, image):
    M = cv2.moments(contour)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
    # put text and highlight the center
    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)


video = cv2.VideoCapture('../resources/cars.avi')
bgSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
kalman = cv2.KalmanFilter(2, 1)

fps = video.get(cv2.CAP_PROP_FPS)
print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")

while video.isOpened():
    ret, frame = video.read()

    cv2.KalmanFilter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgbg = bgSub.apply(gray)
    blur = cv2.GaussianBlur(fgbg, (25, 25), 0)
    noise_reduced = remove_noise(blur)
    _, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
#        draw_contour_center(contour, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("1. Grey", gray)
    cv2.imshow("2. Foreground", fgbg)
    cv2.imshow("3. Blurred foreground", blur)
    cv2.imshow("4. Eroded and dilated foreground", noise_reduced)
    cv2.imshow("5. Thresholded", thresh1)
    cv2.imshow("6. Result", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
