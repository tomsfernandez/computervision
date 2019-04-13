class Car:

    def __init__(self, type, area, last_location, contour):
        self.type = type
        self.area = area
        self.last_location = last_location
        self.speed = 0
        self.contour = contour

    def update_location(self, new_location):
        self.last_location = new_location
        # TODO: Update speed

    def update_contour(self, contour):
        self.contour = contour