import cv2

video_capture = cv2.VideoCapture(0)

threshold = 50


def on_trackbar(val):
    global threshold
    threshold = val


reference = None
flannFlag = False
detector = cv2.ORB_create()
# FLANN - me tira error
sift = cv2.xfeatures2d.SIFT_create()

cv2.namedWindow("Keypoints")
cv2.createTrackbar("Threshold", "Keypoints", 0, 100, on_trackbar)
while True:
    ret, frame = video_capture.read()
    if reference is None:
        reference = frame

    if not flannFlag:
        currentKeypoints, currentDescriptors = detector.detectAndCompute(frame, None)
        referenceKeypoints, referenceDescriptors = detector.detectAndCompute(reference, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(currentDescriptors, referenceDescriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        frame_with_matches = cv2.drawMatches(frame, currentKeypoints, reference, referenceKeypoints, matches[:10],
                                             flags=2,
                                             outImg=None)

    else:
        currentKeypoints, currentDescriptors = sift.detectAndCompute(frame, None)
        referenceKeypoints, referenceDescriptors = sift.detectAndCompute(reference, None)
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(currentDescriptors, referenceDescriptors, k=2)
        matchesMask = [[0, 0] for i in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
        frame_with_matches = cv2.drawMatchesKnn(frame, currentKeypoints, reference, referenceKeypoints, matches, None,
                                                matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                                                matchesMask=matchesMask,
                                                flags=0)

    cv2.imshow('Keypoints', frame_with_matches)
    key = cv2.waitKey(1)
    if key == ord('o'):
        detector = cv2.ORB_create()
        print("Using ORB detector!")
        pass
    elif key == ord('z'):
        detector = cv2.AKAZE_create(threshold=threshold)
        print("Using Akaze!")
        pass
    elif key == ord('g'):
        # No usa detectAndCompute, hay que cambiar cosas
        # detector = cv2.AgastFeatureDetector_create(threshold=threshold)
        print("Using Agast!")
        pass
    elif key == ord('n'):
        flannFlag = True
        print("Using Flann Matcher!")
        pass
    elif key == ord('b'):
        flannFlag = False
        detector = cv2.ORB_create()
        print("Using Brute Force Matcher!")
        pass
    elif key == ord('r'):
        print("New reference!")
        reference = frame
        pass
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
