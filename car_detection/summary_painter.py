import cv2
import numpy as np

class SummaryPainter:

    def __init__(self, car_colors):
        self.car_colors = car_colors

    def paint(self, source, target, cars):
        rows = np.size(target, 0)
        cols = np.size(target, 1)
        current_col = 10
        current_row = 10
        cars.sort(key=lambda x: x.id)
        for car in cars:
            _, _, w, h = cv2.boundingRect(car.contour)
            if current_col + w >= cols:
                current_col = 10
                current_row = current_row + 100
            x, y = current_col, current_row
            color = self.car_colors[car.type]
            cv2.rectangle(target, (x, y), (x + w, y + h), color, 2)
            current_col += w + 10
