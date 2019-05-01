import cv2
import math
import numpy as np

class SummaryPainter:

    def __init__(self, car_colors, columns):
        self.car_colors = car_colors
        self.columns = columns

    def paint(self, source, target, cars):
        rows = np.size(target, 0)
        cols = np.size(target, 1)
        column_delta = math.floor(cols/(self.columns + 1))
        current_column_pixel = 10
        current_row_pixel = 10
        current_column = 0
        current_row = 0
        cars.sort(key=lambda x: x.id)
        for car in cars:
            _, _, w, h = cv2.boundingRect(car.contour)
            if current_column_pixel + column_delta >= cols:
                current_col = 0
                current_column_pixel = 10
                current_row_pixel = current_row_pixel + 110
            x, y = current_column_pixel, current_row_pixel
            color = self.car_colors[car.type]
            cv2.rectangle(target, (x, y), (x + column_delta, y + 100), color, 2)
            current_column += 1
            current_column_pixel = current_column_pixel + column_delta + 10
