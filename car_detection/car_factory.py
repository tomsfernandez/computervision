from car import Car
import cv2
from position import Position


def get_counter_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
    return Position(cX, cY)


class CarFactory:

    def __init__(self, sizes):
        self.sizes = sizes

    def make_car(self, contour):
        car_area = cv2.contourArea(contour)
        position = get_counter_center(contour)
        car_type = self.get_type(car_area)
        return Car(car_type, car_area, position, contour)

    def get_type(self, car_size):
        for size in self.sizes.keys():
            if self.sizes[size].inside(car_size):
                return size


