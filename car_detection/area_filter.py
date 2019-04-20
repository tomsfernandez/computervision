import cv2


class AreaFilter:

    def __init__(self, minSize, maxSize):
        self.maxSize = maxSize
        self.minSize = minSize

    def applies(self, contour):
        return self.minSize < cv2.contourArea(contour) < self.maxSize
