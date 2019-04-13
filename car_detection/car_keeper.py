def are_similar(car, detected_car):
    similar_area = abs(car.area - detected_car.area) <= 500
    similar_position = car.last_location.distance(detected_car.last_location) <= 100
    return similar_area and similar_position


class CarKeeper:

    def __init__(self):
        self.cars = []

    def add(self, detected_cars):
        new_cars = list(filter(lambda x: self.get_similar(x) is None, detected_cars))
        for detected_car in detected_cars:
            similar_car = self.get_similar(detected_car)
            if similar_car is not None:
                similar_car.last_location = detected_car.last_location
        self.cars.extend(new_cars)

    def get_similar(self, detected_car):
        for car in self.cars:
            if are_similar(car, detected_car):
                return car
        return None