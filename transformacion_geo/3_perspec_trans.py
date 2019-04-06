import cv2
import numpy as np
from collections import deque

base_coor_queue = deque([])
target_coor_queue = deque([])
image = cv2.imread('../resources/door.jpg')
image = cv2.resize(image, (500, 700))
target_image = cv2.imread('../resources/slant_door.jpg')
target_image = cv2.resize(target_image, (500, 700))
RED_SCALAR = (0, 0, 255)
rows,cols,channels = image.shape
base_param = (base_coor_queue, image, lambda: cv2.imshow("Base Image", image))
target_param = (target_coor_queue, target_image, lambda: cv2.imshow("Target Image", target_image))


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        set_point(x, y, param[0])
        cv2.circle(param[1], (x, y), 3, RED_SCALAR, -1, 8, 0)
        print('x = %d, y = %d' % (x, y))
        param[2]()


def set_point(x, y, queue):
    if len(queue) == 4:
        queue.popleft()
    queue.append((x, y))

def get_affine_array(array):
    first = [array[0][0], array[0][1]]
    second = [array[1][0], array[1][1]]
    third = [array[2][0], array[2][1]]
    fourth = [array[3][0], array[3][1]]
    return np.float32([first,second,third, fourth])

cv2.imshow("Base Image", image)
cv2.imshow("Target Image", target_image)
cv2.setMouseCallback('Base Image', on_mouse, base_param)
cv2.setMouseCallback('Target Image', on_mouse, target_param)
while True:

    if cv2.waitKey(1) & 0xFF == ord('c'):
        if len(base_coor_queue) < 4 or len(target_coor_queue) < 4:
            print("Eliga 4 coordenadas en cada imagen")
        else: 
            pts1 = get_affine_array(base_coor_queue)
            pts2 = get_affine_array(target_coor_queue)
            transform = cv2.getPerspectiveTransform(pts1,pts2)
            print(transform)
            result = cv2.warpPerspective(image, transform,(cols, rows))
            cv2.imshow("Result Image", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break