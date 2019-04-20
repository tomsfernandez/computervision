class Rectangle:

    def __init__(self, lowerLeftX, lowerLeftY, width, height):
        self.lowerLeftX = lowerLeftX
        self.lowerLeftY = lowerLeftY
        self.upperRightX = lowerLeftX + width
        self.upperRightY = lowerLeftY + height
        self.width = width
        self.height = height

    def area_overlap(self, rectangle):
        x_distance = min(self.upperRightX, rectangle.upperRightX) - max(self.lowerLeftX, rectangle.lowerLeftX)
        y_distance = max(self.upperRightY, rectangle.upperRightY) - min(self.lowerLeftY, rectangle.lowerLeftY)
        area = x_distance * y_distance
        if area < 0:
            return 0
        else:
            return area

    def area(self):
        return float(self.width) * self.height
