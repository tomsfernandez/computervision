import cv2
from rectangle import Rectangle


class CarComparator:

    def __init__(self, common_area_ratio):
        self.common_area_ratio = common_area_ratio

    def compare(self, aCar, anotherCar):
        x, y, w, h = cv2.boundingRect(aCar.contour)
        x1, y1, w1, h1 = cv2.boundingRect(anotherCar.contour)
        aRectangle = Rectangle(x, y, w, h)
        anotherRectangle = Rectangle(x1, y1, w1, h1)
        overlapping_area = aRectangle.area_overlap(anotherRectangle)
        another_overlap_ratio = float(overlapping_area) / anotherRectangle.area()
        an_overlap_ratio = float(overlapping_area) / aRectangle.area()
        return another_overlap_ratio >= self.common_area_ratio or an_overlap_ratio >= self.common_area_ratio

    def overlaps(self, aCar, allCars):
        overlapping_cars = list(filter(lambda x: self.compare(aCar, x), allCars))
        return len(overlapping_cars)
