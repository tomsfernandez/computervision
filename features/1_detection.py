import cv2

video_capture = cv2.VideoCapture(0)

threshold = 50


def on_trackbar(val):
    global threshold
    threshold = val


detector = cv2.FastFeatureDetector_create()
cv2.namedWindow("Keypoints")
cv2.createTrackbar("Threshold", "Keypoints", 0, 100, on_trackbar)
while True:
    ret, frame = video_capture.read()
    keypoints = detector.detect(frame, None)
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))
    cv2.imshow('Keypoints', frame_with_keypoints)
    key = cv2.waitKey(1)
    if key == ord('f'):
        detector = cv2.FastFeatureDetector_create(threshold=threshold)
        print("Using FAST detector!")
        pass
    elif key == ord('o'):
        detector = cv2.ORB_create()
        print("Using ORB detector!")
        pass
    elif key == ord('z'):
        detector = cv2.AKAZE_create(threshold=threshold)
        print("Using Akaze detector!")
        pass
    elif key == ord('a'):
        detector = cv2.AgastFeatureDetector_create(threshold=threshold)
        print("Using Agast detector!")
        pass
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
