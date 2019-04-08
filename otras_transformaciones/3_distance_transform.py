import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)


while True:
    ret, frame = video_capture.read()
    # Esto deberia ser con el threshold
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dist_img = cv2.distanceTransform(grey_image, cv2.DIST_L2, 5).astype(np.uint8)

    cv2.imshow('Distance Transform', dist_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
