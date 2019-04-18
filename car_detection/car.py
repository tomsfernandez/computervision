class Car:

    def __init__(self, type, last_location, contour):
        self.type = type
        self.last_location = last_location
        self.speed = 0
        self.contour = contour
        self.existance = 0
        self.id = 0

    def assign_id(self, id):
        self.id = id

    def update(self, previous):
        self.existance = previous.existance + 1
        pixels_toured = abs(self.last_location.x - previous.last_location.x)
        average_speed = ((self.existance - 1) * previous.speed + pixels_toured)/float(self.existance)
        self.speed = average_speed
        self.id = previous.id