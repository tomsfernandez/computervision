import cv2


#
# bgsegm.createBackgroundSubtractorMOG
# bgsegm.createBackgroundSubtractorGMG
#
# Dont have getBackgroundImage implemented

def get_algorithm_dictionary(detect_shadows):
    return {
        "1": cv2.createBackgroundSubtractorMOG2(detectShadows=detect_shadows),
        "2": cv2.createBackgroundSubtractorKNN(detectShadows=detect_shadows),
        "3": cv2.bgsegm.createBackgroundSubtractorLSBP(),
        "4": cv2.bgsegm.createBackgroundSubtractorCNT(),
        "5": cv2.bgsegm.createBackgroundSubtractorGSOC()
    }


class SubstractorProvider:

    def __init__(self, detect_shadows=False):
        self.algorithms = get_algorithm_dictionary(detect_shadows)

    def get_algorithm(self, number):
        return self.algorithms[number]
