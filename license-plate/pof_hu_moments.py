import cv2
import numpy as np

GREEN_RGB = (0, 255, 0)

image = cv2.imread("resources/patente_nueve.PNG")
if image is None:
    print("No se pudo leer la imagen!")
    exit(1)


def get_biggest_contour(contours):
    contours_with_area = list(map(lambda x: (x, cv2.contourArea(x)), contours))
    return max(contours_with_area, key=lambda x: x[1])[0]


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gris", gray)
blurred = cv2.GaussianBlur(gray, (11, 11), 10)
cv2.imshow("Blurred", blurred)
threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 15)
cv2.imshow("Thresholded", threshold)
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
outer_contour = get_biggest_contour(contours)
cv2.drawContours(image, outer_contour, -1, GREEN_RGB, 5)
cv2.imshow("Original", image)
moments = cv2.moments(outer_contour)
huMoments = cv2.HuMoments(moments)
print(huMoments)
cv2.waitKey(0)
