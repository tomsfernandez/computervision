import cv2


class CarKeeper:

    def __init__(self, car_comparator):
        self.id_seq = 1
        self.cars = []
        self.car_comparator = car_comparator

    def add(self, detected_cars):
        new_cars = self.get_new_cars(detected_cars)
        existing_cars = self.get_existing_cars(detected_cars)
        non_detected_cars = self.get_not_detected_cars(existing_cars)

        for detected_car in existing_cars:
            similar_car = self.get_similar(detected_car)
            if similar_car is not None:
                detected_car.update(similar_car)
        for car in new_cars:
            car.assign_id(self.id_seq)
            self.id_seq = self.id_seq + 1
            print(f"Found new car! {self.id_seq - 1}")
        self.cars = new_cars + existing_cars + non_detected_cars

    def get_new_cars(self, detected_cars):
        return list(filter(lambda x: self.car_comparator.overlaps(x, self.cars) <= 1 and self.get_similar(x) is None, detected_cars))

    def get_existing_cars(self, detected_cars):
        return list(filter(lambda x: self.get_similar(x) is not None, detected_cars))

    def get_not_detected_cars(self, existing_and_detected):
        existing_and_detected_ids = list(map(lambda x: x.id, existing_and_detected))
        non_detected = list(filter(lambda x: x.id in existing_and_detected_ids, self.cars))
        return non_detected

    def get_similar(self, detected_car):
        for car in self.cars:
            if self.car_comparator.compare(car, detected_car):
                return car
        return None