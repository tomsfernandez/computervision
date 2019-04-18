import cv2
import numpy as np

def remove_noise(image):
    erosion_kernel = np.ones((2, 2), np.uint8)
    dilation_kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(image, erosion_kernel, iterations=2)
    dilated = cv2.dilate(eroded, dilation_kernel, iterations=2)
    return dilated

video = cv2.VideoCapture('../resources/cars.avi')
bgSub = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=50, detectShadows=False)
# print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")

while video.isOpened():
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgbg = bgSub.apply(gray)
    #noise_reduced = remove_noise(fgbg)
    blur = cv2.GaussianBlur(fgbg, (25, 25), 25)
    _, thresh1 = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    cv2.imshow("1. Grey", gray)
    cv2.imshow("2. Foreground", fgbg)
    cv2.imshow("3. Blurred foreground", blur)
    #cv2.imshow("4. Eroded and dilated foreground", noise_reduced)
    cv2.imshow("5. Thresholded", thresh1)
    key_pressed = cv2.waitKey(0)
    if key_pressed == ord('q'):
        break

video.release()
cv2.destroyAllWindows()