import cv2

class CarKeeper:

    def __init__(self, car_comparator):
        self.cars = []
        self.car_comparator = car_comparator

    def add(self, detected_cars):
        new_cars = list(filter(lambda x: self.car_comparator.overlaps(x, self.cars) <= 1 and self.get_similar(x) is None, detected_cars))
        existing_cars = list(filter(lambda x: self.get_similar(x) is not None, detected_cars))
        for detected_car in existing_cars:
            similar_car = self.get_similar(detected_car)
            if similar_car is not None:
                detected_car.update_speed(similar_car.last_location)
        self.cars = new_cars + existing_cars

    def get_similar(self, detected_car):
        for car in self.cars:
            if self.car_comparator.compare(car, detected_car):
                return car
        return None