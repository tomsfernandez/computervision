import cv2
from collections import deque

base_coor_queue = deque([])
target_coor_queue = deque([])
image = cv2.imread('../resources/door.jpg')
image = cv2.resize(image, (500, 700))
target_image = cv2.imread('../resources/slant_door.jpg')
target_image = cv2.resize(target_image, (500, 700))
GREEN_SCALAR = (0, 0, 255)

base_param = (base_coor_queue, image, lambda: cv2.imshow("Base Image", image))
target_param = (target_coor_queue, target_image, lambda: cv2.imshow("Target Image", target_image))


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        set_point(x, y, param[0])
        cv2.circle(param[1], (x, y), 3, GREEN_SCALAR, -1, 8, 0)
        print('x = %d, y = %d' % (x, y))
        param[2]()


def set_point(x, y, queue):
    if len(queue) == 3:
        queue.popleft()
    queue.append((x, y))


cv2.imshow("Base Image", image)
cv2.imshow("Target Image", target_image)
cv2.setMouseCallback('Base Image', on_mouse, base_param)
cv2.setMouseCallback('Target Image', on_mouse, target_param)
cv2.waitKey(0)
