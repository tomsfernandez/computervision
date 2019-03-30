import cv2

source = cv2.imread('../resources/mountains.jpg')

horizontal_img = cv2.flip(source, 0)
vertical_img = cv2.flip(source, 1)
both_img = cv2.flip(source, -1)

# display the images on screen with imshow()
cv2.imshow("Original", source)
cv2.imshow("Horizontal flip", horizontal_img)
cv2.imshow("Vertical flip", vertical_img)
cv2.imshow("Both flip", both_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
