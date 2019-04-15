import time

import cv2
import numpy as np

from car import Car
from car_factory import CarFactory
from area import Area
from car_keeper import CarKeeper

car_sizes = {"CAR": Area(0, 300000)}
GREEN_RGB = (0, 255, 0)
YELLOW_RGB = (255, 255, 0)
RED_RGB = (255, 0, 0)
car_colors = {"CAR": GREEN_RGB, "VAN": YELLOW_RGB, "TRUCK": RED_RGB}


def remove_noise(image):
    erosion_kernel = np.ones((3, 3), np.uint8)
    dilation_kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(image, erosion_kernel, iterations=1)
    return cv2.dilate(img, dilation_kernel, iterations=1)


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
    blur = cv2.GaussianBlur(fgbg, (25, 25), 0)
    noise_reduced = remove_noise(blur)
    _, thresh1 = cv2.threshold(noise_reduced, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("1. Grey", gray)
    cv2.imshow("2. Foreground", fgbg)
    cv2.imshow("3. Blurred foreground", blur)
    cv2.imshow("4. Eroded and dilated foreground", noise_reduced)
    cv2.imshow("5. Thresholded", thresh1)
    return thresh1


def draw_car_boxes(image, cars):
    for car in cars:
        x, y, w, h = cv2.boundingRect(car.contour)
        color = car_colors[car.type]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


def draw_car_speeds(image, cars, speed_conversion_factor):
    for car in cars:
        speed = car.speed * speed_conversion_factor
        x, y, w, h = cv2.boundingRect(car.contour)
        color = car_colors[car.type]
        cv2.putText(image, f"{speed} km/h", (x, y + 60), font, 0.5, color, 2, cv2.LINE_AA)


video = cv2.VideoCapture('../resources/cars.avi')
bgSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
kalman = cv2.KalmanFilter(2, 1)
car_factory = CarFactory(car_sizes)

fps = video.get(cv2.CAP_PROP_FPS)
sedan_approximated_length = 3.5
sedan_pixel_length = 60
metres_per_pixel_factor = sedan_approximated_length / sedan_approximated_length
speed_conversion_factor = fps * metres_per_pixel_factor * (3600.0 / 1000.0)
font = cv2.FONT_HERSHEY_SIMPLEX
print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")
car_keeper = CarKeeper()

while video.isOpened():
    ret, frame = video.read()
    processed_image = process_and_show_image(frame)
    contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected_cars = list(map(lambda contour: car_factory.make_car(contour), contours))
    # Hay que arreglar aca
    car_keeper.add(detected_cars)
    cars = car_keeper.cars
    # de ahora en más solo habría que usar los cars ya que son los cars trackeados, todavía no funciona bien.
    draw_car_boxes(frame, detected_cars)
    draw_car_speeds(frame, detected_cars, speed_conversion_factor)
    cv2.imshow("6. Result", frame)
    cv2.setMouseCallback('6. Result', on_mouse)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
