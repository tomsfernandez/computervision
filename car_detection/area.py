class Area:

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def inside(self, area_amount):
        return self.min <= area_amount < self.max