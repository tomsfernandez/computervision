import time

import cv2
import numpy as np

from car import Car
from car_factory import CarFactory
from area import Area
from car_keeper import CarKeeper
from car_comparator import CarComparator
from area_filter import AreaFilter

car_sizes = {"CAR": Area(0, 3000), "VAN": Area(3000, 7000), "TRUCK": Area(7000, 10000)}
GREEN_RGB = (0, 255, 0)
YELLOW_RGB = (0, 255, 255)
RED_RGB = (0, 0, 255)
car_colors = {"CAR": GREEN_RGB, "VAN": YELLOW_RGB, "TRUCK": RED_RGB}


# def remove_noise(image):
#     erosion_kernel = np.ones((2, 2), np.uint8)
#     dilation_kernel = np.ones((2, 2), np.uint8)
#     eroded = cv2.erode(image, erosion_kernel, iterations=2)
#     dilated = cv2.dilate(eroded, dilation_kernel, iterations=2)
#     return eroded


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('x = %d, y = %d' % (x, y))


def draw_contour_center(contour, image):
    M = cv2.moments(contour)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
    # put text and highlight the center
    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)


def process_and_show_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fgbg = bgSub.apply(gray)
    blur = cv2.GaussianBlur(fgbg, (25, 25), 25)
    _, thresh1 = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    cv2.imshow("Grey", gray)
    cv2.imshow("Foreground", fgbg)
    cv2.imshow("Blurred foreground", blur)
    cv2.imshow("Thresholded", thresh1)
    return thresh1


def draw_car_boxes(image, cars):
    for car in cars:
        x, y, w, h = cv2.boundingRect(car.contour)
        color = car_colors[car.type]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


def draw_car_speeds(image, cars, speed_conversion_factor):
    for car in cars:
        speed = int(car.speed * speed_conversion_factor)
        x, y, w, h = cv2.boundingRect(car.contour)
        color = car_colors[car.type]
        cv2.putText(image, f"{speed} km/h", (x, y + 60), font, 0.5, YELLOW_RGB, 2, cv2.LINE_AA)


video = cv2.VideoCapture('../resources/cars.avi')
bgSub = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=50, detectShadows=False)
kalman = cv2.KalmanFilter(2, 1)
car_factory = CarFactory(car_sizes)
car_comparator = CarComparator(0.8)
frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
area_filter = AreaFilter(300, frame_height * frame_width - 1000)

fps = video.get(cv2.CAP_PROP_FPS)
sedan_approximated_length = 3.5
sedan_pixel_length = 60
metres_per_pixel_factor = sedan_approximated_length / sedan_approximated_length
speed_conversion_factor = fps * metres_per_pixel_factor * (3600.0 / 1000.0) / 10
font = cv2.FONT_HERSHEY_SIMPLEX
print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")
car_keeper = CarKeeper(car_comparator)

while video.isOpened():
    ret, frame = video.read()
    processed_image = process_and_show_image(frame)
    contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda x: area_filter.applies(x), contours))
    detected_cars = list(map(lambda contour: car_factory.make_car(contour), contours))
    car_keeper.add(detected_cars)
    cars = car_keeper.cars
    draw_car_boxes(frame, cars)
    draw_car_speeds(frame, cars, speed_conversion_factor)
    cv2.imshow("6. Result", frame)
    cv2.setMouseCallback('6. Result', on_mouse)
    key_pressed = cv2.waitKey(0)
    if key_pressed == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
