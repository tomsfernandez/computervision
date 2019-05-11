import cv2

video_capture = cv2.VideoCapture(0)

threshold = 50


def on_trackbar(val):
    global threshold
    threshold = val


reference = None
detector = cv2.ORB_create()
cv2.namedWindow("Keypoints")
cv2.createTrackbar("Threshold", "Keypoints", 0, 100, on_trackbar)
while True:
    ret, frame = video_capture.read()
    if reference is None:
        reference = frame
    currentKeypoints, currentDescriptors = detector.detectAndCompute(frame, None)
    referenceKeypoints, referenceDescriptors = detector.detectAndCompute(frame, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(currentDescriptors, referenceDescriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    frame_with_matches = cv2.drawMatches(frame, currentKeypoints, reference, referenceKeypoints, matches[:10], flags=2, outImg=None)
    cv2.imshow('Keypoints', frame_with_matches)
    key = cv2.waitKey(1)
    if key == ord('o'):
        detector = cv2.ORB_create()
        print("Using ORB detector!")
        pass
    elif key == ord('z'):
        detector = cv2.AKAZE_create(threshold=threshold)
        print("Using Akaze detector!")
        pass
    elif key == ord('n'):
        print("Using Flann Matcher!")
        pass
    elif key == ord('b'):
        print("Using Brute Force Matcher!")
        pass
    elif key == ord('r'):
        print("New reference!")
        reference = frame
        pass
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
