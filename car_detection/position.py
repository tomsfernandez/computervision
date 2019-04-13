from math import sqrt


class Position:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, another_position):
        return sqrt(((self.x - another_position.x) ** 2) + ((self.y - another_position.y) ** 2))
