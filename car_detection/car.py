class Car:

    def __init__(self, type, last_location, contour):
        self.type = type
        self.last_location = last_location
        self.speed = 0
        self.contour = contour
        self.existance = 0

    def update_speed(self, previous_location):
        self.speed = abs(self.last_location.x - previous_location.x)