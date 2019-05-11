import cv2

video_capture = cv2.VideoCapture(0)

min_threshold = 10
max_threshold = 200


def on_trackbar_min(val):
    global min_threshold
    min_threshold = val
    global params
    global detector
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = min_threshold
    params.maxThreshold = max_threshold
    detector = cv2.SimpleBlobDetector_create(params)


def on_trackbar_max(val):
    global max_threshold
    max_threshold = val
    global params
    global detector
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = min_threshold
    params.maxThreshold = max_threshold
    detector = cv2.SimpleBlobDetector_create(params)


params = cv2.SimpleBlobDetector_Params()
params.minThreshold = min_threshold
params.maxThreshold = max_threshold

detector = cv2.SimpleBlobDetector_create(params)
cv2.namedWindow("Simple Blob")
cv2.createTrackbar("Min Threshold", "Simple Blob", min_threshold, 200, on_trackbar_min)
cv2.createTrackbar("Max Threshold", "Simple Blob", max_threshold, 200, on_trackbar_max)
while True:
    ret, frame = video_capture.read()
    keypoints = detector.detect(frame, None)
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, (0, 255, 0),
                                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Simple Blob', frame_with_keypoints)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
